"""
PyTorch autograd integration for the Triton renderer.

Implements custom autograd Function for differentiable rendering,
enabling gradient computation through the rendering pipeline.

Key gradient computation approaches:
1. Interior gradients: Standard backprop through color/alpha
2. Boundary gradients: Reynolds transport theorem for shape boundaries

Ported from diffvg gradient computation logic.
"""

import torch
from torch.autograd import Function
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass

from .scene import FlattenedScene, flatten_scene, FlattenedPaths, FlattenedShapeGroup
from .render import render_scene, RenderConfig


@dataclass
class GradientConfig:
    """Configuration for gradient computation."""
    num_boundary_samples: int = 16  # Samples per shape for boundary gradient
    boundary_sample_seed: int = 12345
    use_boundary_gradient: bool = True  # Enable Reynolds transport gradient


class TritonRenderFunction(Function):
    """
    Custom autograd Function for differentiable vector graphics rendering.

    Forward pass: Render scene to image
    Backward pass: Compute gradients for shape parameters and colors

    Usage:
        image = TritonRenderFunction.apply(
            canvas_width, canvas_height,
            shapes, shape_groups,
            num_samples_x, num_samples_y,
            seed, background_color
        )
    """

    @staticmethod
    def forward(
        ctx,
        canvas_width: int,
        canvas_height: int,
        *args,
    ):
        """
        Forward rendering pass.

        Args:
            ctx: Autograd context for saving tensors
            canvas_width: Output width
            canvas_height: Output height
            *args: Flattened list of shape parameters and config

        Returns:
            [H, W, 4] rendered image
        """
        # Parse arguments
        # Expected format: path_points, path_colors, ..., config_tensors
        # This is a simplified interface - full version needs proper arg handling

        # For now, expect args to be:
        # (shapes_list, shape_groups_list, num_samples_x, num_samples_y, seed, background, use_prefilter)

        shapes = args[0]
        shape_groups = args[1]
        num_samples_x = args[2] if len(args) > 2 else 2
        num_samples_y = args[3] if len(args) > 3 else 2
        seed = args[4] if len(args) > 4 else 42
        background = args[5] if len(args) > 5 else None
        use_prefiltering = args[6] if len(args) > 6 else False

        if background is None:
            background = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # Flatten scene
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scene = flatten_scene(canvas_width, canvas_height, shapes, shape_groups, device=device)

        # Configure
        config = RenderConfig(
            num_samples_x=num_samples_x,
            num_samples_y=num_samples_y,
            seed=seed,
            background_color=tuple(background.cpu().tolist()),
            use_prefiltering=use_prefiltering,
        )

        # Render
        image = render_scene(scene, config, use_python_reference=True)

        # Save for backward
        ctx.save_for_backward(image, background)
        ctx.scene = scene
        ctx.config = config
        ctx.shapes = shapes
        ctx.shape_groups = shape_groups
        ctx.canvas_width = canvas_width
        ctx.canvas_height = canvas_height

        return image

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for gradient computation.

        Computes gradients for:
        - Shape control points (via boundary gradient)
        - Fill/stroke colors
        - Other parameters

        Args:
            ctx: Autograd context
            grad_output: [H, W, 4] gradient w.r.t. output image

        Returns:
            Tuple of gradients matching forward args
        """
        image, background = ctx.saved_tensors
        scene = ctx.scene
        config = ctx.config
        shapes = ctx.shapes
        shape_groups = ctx.shape_groups

        # Initialize gradient tensors
        # For each shape, we need gradients for its parameters

        grad_shapes = []
        for shape in shapes:
            shape_grad = {}
            if hasattr(shape, 'points') and shape.points is not None:
                shape_grad['points'] = torch.zeros_like(shape.points)
            if hasattr(shape, 'stroke_width') and shape.stroke_width is not None:
                if isinstance(shape.stroke_width, torch.Tensor):
                    shape_grad['stroke_width'] = torch.zeros_like(shape.stroke_width)
            grad_shapes.append(shape_grad)

        grad_shape_groups = []
        for group in shape_groups:
            group_grad = {}
            if group.fill_color is not None and isinstance(group.fill_color, torch.Tensor):
                group_grad['fill_color'] = torch.zeros_like(group.fill_color)
            if group.stroke_color is not None and isinstance(group.stroke_color, torch.Tensor):
                group_grad['stroke_color'] = torch.zeros_like(group.stroke_color)
            grad_shape_groups.append(group_grad)

        # Compute gradients
        # This is a simplified version - full implementation needs:
        # 1. Interior gradient: d(image)/d(color) - straightforward backprop
        # 2. Boundary gradient: Reynolds transport theorem

        _compute_color_gradients(
            grad_output, scene, shapes, shape_groups,
            grad_shapes, grad_shape_groups, config
        )

        _compute_boundary_gradients(
            grad_output, scene, shapes, shape_groups,
            grad_shapes, config
        )

        # Pack gradients back into shapes
        for i, shape in enumerate(shapes):
            for key, grad in grad_shapes[i].items():
                if hasattr(shape, key):
                    param = getattr(shape, key)
                    if isinstance(param, torch.Tensor) and param.requires_grad:
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad += grad

        for i, group in enumerate(shape_groups):
            for key, grad in grad_shape_groups[i].items():
                if hasattr(group, key):
                    param = getattr(group, key)
                    if isinstance(param, torch.Tensor) and param.requires_grad:
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad += grad

        # Return None for non-tensor args
        return (None, None, None, None, None, None, None, None, None)


def _compute_color_gradients(
    grad_output: torch.Tensor,
    scene: FlattenedScene,
    shapes: list,
    shape_groups: list,
    grad_shapes: list,
    grad_shape_groups: list,
    config: RenderConfig,
):
    """
    Compute gradients for color parameters.

    For each pixel, trace back which shapes contributed and
    accumulate gradients to their color parameters.
    """
    height, width = grad_output.shape[:2]
    device = grad_output.device

    # For each pixel with non-zero gradient
    for py in range(height):
        for px in range(width):
            pixel_grad = grad_output[py, px]

            # Skip if gradient is negligible
            if pixel_grad.abs().sum() < 1e-8:
                continue

            # Sample position (center of pixel)
            pt = (px + 0.5, py + 0.5)

            # Find contributing shapes
            for group_idx, group in enumerate(shape_groups):
                if group.fill_color is not None and isinstance(group.fill_color, torch.Tensor):
                    # Check if this group's shapes contain the pixel
                    # Simplified: assume uniform contribution
                    # Full version needs proper visibility/occlusion handling

                    if group_idx < len(grad_shape_groups) and 'fill_color' in grad_shape_groups[group_idx]:
                        # Gradient flows from pixel to color
                        grad_shape_groups[group_idx]['fill_color'] += pixel_grad[:4] if pixel_grad.shape[0] >= 4 else pixel_grad


def _compute_boundary_gradients(
    grad_output: torch.Tensor,
    scene: FlattenedScene,
    shapes: list,
    shape_groups: list,
    grad_shapes: list,
    config: RenderConfig,
    num_boundary_samples: int = 16,
):
    """
    Compute boundary gradients using Reynolds transport theorem.

    For each shape boundary, sample points and compute:
    gradient += (color_inside - color_outside) * velocity · normal / pdf

    This captures how moving control points affects the rendered image
    through changes in shape boundaries.
    """
    from .kernels.boundary import sample_path_boundary
    from .kernels.rng import PCG32

    if scene.paths is None:
        return

    device = grad_output.device
    height, width = grad_output.shape[:2]

    rng = PCG32(seed=config.seed + 1000)

    # For each path
    for path_idx in range(scene.paths.num_paths):
        if path_idx >= len(shapes):
            continue

        shape = shapes[path_idx]
        if not hasattr(shape, 'points') or shape.points is None:
            continue

        # Get path data
        num_segments = scene.paths.num_segments[path_idx].item()
        point_offset = scene.paths.point_offsets[path_idx].item()
        num_points = scene.paths.num_points[path_idx].item()
        is_closed = scene.paths.is_closed[path_idx].item()

        segment_types = scene.paths.segment_types[path_idx, :num_segments].cpu().tolist()
        points_flat = scene.paths.points[point_offset:point_offset + num_points].cpu().tolist()
        points = [(p[0], p[1]) for p in points_flat]

        if len(points) < 2 or len(segment_types) == 0:
            continue

        # Sample boundary points
        for _ in range(num_boundary_samples):
            u = rng.uniform()

            try:
                pos, normal, seg_idx, t, pdf = sample_path_boundary(
                    u, segment_types, points, is_closed
                )
            except (IndexError, ZeroDivisionError):
                continue

            # Check if boundary point is within image bounds
            px, py = int(pos[0]), int(pos[1])
            if px < 0 or px >= width or py < 0 or py >= height:
                continue

            # Get gradient at this pixel
            pixel_grad = grad_output[py, px].cpu()

            # Color difference would be (inside - outside)
            # For simplicity, use gradient magnitude as proxy
            color_diff = pixel_grad[:3].sum().item() if pixel_grad.shape[0] >= 3 else 0.0

            if abs(color_diff) < 1e-8:
                continue

            # Compute velocity for each control point
            # Accumulate gradient contribution
            seg_type = segment_types[seg_idx]

            # Find which points belong to this segment
            pt_idx = 0
            for i in range(seg_idx):
                if segment_types[i] == 0:
                    pt_idx += 1
                elif segment_types[i] == 1:
                    pt_idx += 2
                else:
                    pt_idx += 3

            # Velocity coefficients based on segment type and parameter t
            one_minus_t = 1.0 - t

            if seg_type == 0:
                # Line: 2 points
                velocities = [one_minus_t, t]
                num_ctrl_pts = 2
            elif seg_type == 1:
                # Quadratic: 3 points
                velocities = [
                    one_minus_t * one_minus_t,
                    2.0 * one_minus_t * t,
                    t * t
                ]
                num_ctrl_pts = 3
            else:
                # Cubic: 4 points
                t_sq = t * t
                velocities = [
                    one_minus_t * one_minus_t * one_minus_t,
                    3.0 * one_minus_t * one_minus_t * t,
                    3.0 * one_minus_t * t_sq,
                    t_sq * t
                ]
                num_ctrl_pts = 4

            # Accumulate gradients
            contribution = color_diff / (pdf * num_boundary_samples)

            for i in range(min(num_ctrl_pts, len(velocities))):
                local_pt_idx = pt_idx + i
                if local_pt_idx >= num_points:
                    break

                vel = velocities[i]

                # Gradient = contribution * velocity * normal
                grad_x = contribution * vel * normal[0]
                grad_y = contribution * vel * normal[1]

                # Add to gradient tensor
                if path_idx < len(grad_shapes) and 'points' in grad_shapes[path_idx]:
                    grad_tensor = grad_shapes[path_idx]['points']
                    if local_pt_idx < grad_tensor.shape[0]:
                        grad_tensor[local_pt_idx, 0] += grad_x
                        grad_tensor[local_pt_idx, 1] += grad_y


def render_grad(
    canvas_width: int,
    canvas_height: int,
    shapes: list,
    shape_groups: list,
    num_samples_x: int = 2,
    num_samples_y: int = 2,
    seed: int = 42,
    background_color: torch.Tensor = None,
    use_prefiltering: bool = False,
) -> torch.Tensor:
    """
    Differentiable rendering function.

    This is the main entry point for differentiable rendering.
    Gradients will flow back to shape parameters.

    Args:
        canvas_width: Output width
        canvas_height: Output height
        shapes: List of shape objects with tensor parameters
        shape_groups: List of ShapeGroup objects
        num_samples_x: AA samples in x
        num_samples_y: AA samples in y
        seed: Random seed
        background_color: Background RGBA
        use_prefiltering: Use SDF-based anti-aliasing

    Returns:
        [H, W, 4] rendered image (differentiable)
    """
    if background_color is None:
        background_color = torch.tensor([1.0, 1.0, 1.0, 1.0])

    return TritonRenderFunction.apply(
        canvas_width, canvas_height,
        shapes, shape_groups,
        num_samples_x, num_samples_y,
        seed, background_color,
        use_prefiltering,
    )


# Convenience class for easier gradient computation
class DifferentiableRenderer:
    """
    High-level wrapper for differentiable rendering.

    Example:
        renderer = DifferentiableRenderer(256, 256)

        # Create shapes with requires_grad=True
        points = torch.tensor([[0, 0], [100, 50], [50, 100]], requires_grad=True)
        path = Path(points, ...)

        # Render
        image = renderer.render(shapes, shape_groups)

        # Compute loss and backprop
        loss = (image - target).pow(2).mean()
        loss.backward()

        # Gradients are now in points.grad
    """

    def __init__(
        self,
        canvas_width: int,
        canvas_height: int,
        num_samples_x: int = 2,
        num_samples_y: int = 2,
        seed: int = 42,
        background_color: torch.Tensor = None,
        use_prefiltering: bool = False,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.num_samples_x = num_samples_x
        self.num_samples_y = num_samples_y
        self.seed = seed
        self.background_color = background_color
        self.use_prefiltering = use_prefiltering

    def render(self, shapes: list, shape_groups: list) -> torch.Tensor:
        """Render shapes to image with gradient support."""
        return render_grad(
            self.canvas_width,
            self.canvas_height,
            shapes,
            shape_groups,
            self.num_samples_x,
            self.num_samples_y,
            self.seed,
            self.background_color,
            self.use_prefiltering,
        )
