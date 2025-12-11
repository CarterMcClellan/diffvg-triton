"""
Main rendering functions for the Triton backend.

Implements the core rendering pipeline:
1. Sample generation (multi-sample anti-aliasing)
2. Fragment collection (winding number + distance tests)
3. Color sampling (fill + stroke)
4. Fragment compositing (back-to-front blending)
5. Filter splatting (optional)

Ported from diffvg/diffvg.cpp render_kernel
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import IntEnum

from .scene import FlattenedScene, FlattenedPaths, FlattenedShapeGroup, ShapeType
from .kernels.composite import composite_over, smoothstep_coverage_py
from .kernels.filter import FilterType, splat_samples_to_image
from .kernels.rng import TorchPCG32, generate_sample_offsets


class RenderMode(IntEnum):
    """Rendering mode selection."""
    HARD = 0       # Hard edges, multi-sample AA
    PREFILTER = 1  # Smooth edges using SDF


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    num_samples_x: int = 2       # Samples per pixel in x
    num_samples_y: int = 2       # Samples per pixel in y
    seed: int = 42               # Random seed for sample jittering
    filter_type: int = FilterType.BOX
    filter_radius: float = 1.0
    use_prefiltering: bool = False
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


def _compute_winding_number_py(
    pt: Tuple[float, float],
    paths: FlattenedPaths,
    path_idx: int,
) -> int:
    """
    Compute winding number for a point against a path.

    Python reference implementation.
    """
    from .kernels.winding import compute_winding_number_path_py

    # Get path data
    point_offset = paths.point_offsets[path_idx].item()
    num_points = paths.num_points[path_idx].item()
    num_segments = paths.num_segments[path_idx].item()
    is_closed = paths.is_closed[path_idx].item()

    # Extract segment types and points for this path
    segment_types = paths.segment_types[path_idx, :num_segments].detach().cpu().tolist()
    points_flat = paths.points[point_offset:point_offset + num_points].detach().cpu().tolist()
    points = [(p[0], p[1]) for p in points_flat]

    # Compute winding number
    winding = compute_winding_number_path_py(pt, segment_types, points, is_closed)

    return winding


def _compute_closest_distance_py(
    pt: Tuple[float, float],
    paths: FlattenedPaths,
    path_idx: int,
) -> Tuple[float, Tuple[float, float], float]:
    """
    Compute closest distance from point to path.

    Returns: (distance, closest_point, t_parameter)
    """
    from .kernels.distance import (
        closest_point_line_py,
        closest_point_quadratic_bezier_py,
        closest_point_cubic_bezier_py,
    )

    point_offset = paths.point_offsets[path_idx].item()
    num_segments = paths.num_segments[path_idx].item()

    segment_types = paths.segment_types[path_idx, :num_segments].detach().cpu().tolist()
    points_flat = paths.points.detach().cpu().numpy()

    best_dist_sq = float('inf')
    best_closest = pt
    best_t = 0.0

    current_point = point_offset

    for seg_type in segment_types:
        if seg_type == 0:
            # Line
            p0 = tuple(points_flat[current_point])
            p1 = tuple(points_flat[current_point + 1])
            closest, t, dist_sq = closest_point_line_py(pt, p0, p1)
            current_point += 1
        elif seg_type == 1:
            # Quadratic
            p0 = tuple(points_flat[current_point])
            p1 = tuple(points_flat[current_point + 1])
            p2 = tuple(points_flat[current_point + 2])
            closest, t, dist_sq = closest_point_quadratic_bezier_py(pt, p0, p1, p2)
            current_point += 2
        else:
            # Cubic
            p0 = tuple(points_flat[current_point])
            p1 = tuple(points_flat[current_point + 1])
            p2 = tuple(points_flat[current_point + 2])
            p3 = tuple(points_flat[current_point + 3])
            closest, t, dist_sq = closest_point_cubic_bezier_py(pt, p0, p1, p2, p3)
            current_point += 3

        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_closest = closest
            best_t = t

    return best_dist_sq ** 0.5, best_closest, best_t


def _sample_color_at_point(
    pt: Tuple[float, float],
    scene: FlattenedScene,
    use_prefiltering: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Sample color at a point by collecting and compositing fragments.

    Python reference implementation.
    """
    from .kernels.composite import blend_over_py, smoothstep_coverage_py

    # Collect fragments from all shape groups
    fragments = []

    for group_idx in range(scene.groups.num_groups):
        if not scene.groups.shape_mask[group_idx, 0].item():
            continue

        # Get shapes in this group
        num_shapes = scene.groups.num_shapes[group_idx].item()
        shape_ids = scene.groups.shape_ids[group_idx, :num_shapes].cpu().tolist()

        has_fill = scene.groups.has_fill[group_idx].item()
        has_stroke = scene.groups.has_stroke[group_idx].item()
        use_even_odd = scene.groups.use_even_odd_rule[group_idx].item()

        fill_color = None
        if has_fill and scene.groups.fill_color is not None:
            fc = scene.groups.fill_color[group_idx].cpu().tolist()
            fill_color = tuple(fc)

        stroke_color = None
        if has_stroke and scene.groups.stroke_color is not None:
            sc = scene.groups.stroke_color[group_idx].cpu().tolist()
            stroke_color = tuple(sc)

        # Check each shape
        for shape_id in shape_ids:
            shape_type = scene.shape_types[shape_id].item()
            shape_idx = scene.shape_indices[shape_id].item()

            if shape_type == ShapeType.PATH and scene.paths is not None:
                # Compute winding number for fill
                if has_fill and fill_color is not None:
                    winding = _compute_winding_number_py(pt, scene.paths, shape_idx)

                    # Apply fill rule
                    if use_even_odd:
                        is_inside = (winding % 2) != 0
                    else:
                        is_inside = winding != 0

                    if is_inside:
                        if use_prefiltering:
                            # Get distance for smooth coverage
                            dist, _, _ = _compute_closest_distance_py(pt, scene.paths, shape_idx)
                            coverage = smoothstep_coverage_py(-dist)  # Negative for inside
                            color = (
                                fill_color[0],
                                fill_color[1],
                                fill_color[2],
                                fill_color[3] * coverage
                            )
                        else:
                            color = fill_color

                        fragments.append((color, group_idx, False))

                # Compute distance for stroke
                if has_stroke and stroke_color is not None:
                    stroke_width = scene.paths.stroke_width[shape_idx].item()
                    if stroke_width > 0:
                        dist, _, _ = _compute_closest_distance_py(pt, scene.paths, shape_idx)

                        if use_prefiltering:
                            # Smooth stroke coverage
                            half_width = stroke_width / 2.0
                            coverage = smoothstep_coverage_py(abs(dist) - half_width)
                            if coverage > 0:
                                color = (
                                    stroke_color[0],
                                    stroke_color[1],
                                    stroke_color[2],
                                    stroke_color[3] * coverage
                                )
                                fragments.append((color, group_idx, True))
                        else:
                            if dist <= stroke_width / 2.0:
                                fragments.append((stroke_color, group_idx, True))

    # Sort fragments by group_id (back to front)
    fragments.sort(key=lambda x: x[1])

    return fragments


def render_scene_py(
    scene: FlattenedScene,
    config: RenderConfig = None,
) -> torch.Tensor:
    """
    Render a scene using Python reference implementation.

    This is a slow but correct reference implementation for testing.

    Args:
        scene: Flattened scene data
        config: Render configuration

    Returns:
        [H, W, 4] RGBA image tensor
    """
    from .kernels.composite import composite_fragments_py

    if config is None:
        config = RenderConfig()

    width = scene.canvas_width
    height = scene.canvas_height
    device = scene.device

    # Initialize output
    output = torch.zeros((height, width, 4), dtype=torch.float32, device='cpu')

    num_samples = config.num_samples_x * config.num_samples_y
    background = config.background_color

    # Generate sample offsets
    sample_offsets = []
    for sy in range(config.num_samples_y):
        for sx in range(config.num_samples_x):
            # Stratified sampling
            ox = (sx + 0.5) / config.num_samples_x
            oy = (sy + 0.5) / config.num_samples_y
            sample_offsets.append((ox, oy))

    # Render each pixel
    for py in range(height):
        for px in range(width):
            # Accumulate samples
            acc_color = [0.0, 0.0, 0.0, 0.0]

            for ox, oy in sample_offsets:
                # Sample position in canvas coordinates
                sx = px + ox
                sy = py + oy
                pt = (sx, sy)

                # Collect fragments at this sample
                fragments = _sample_color_at_point(
                    pt, scene, config.use_prefiltering
                )

                # Composite fragments
                if fragments:
                    frag_colors = [f[0] for f in fragments]
                    sample_color = composite_fragments_py(frag_colors, background)
                else:
                    sample_color = background

                # Accumulate
                for c in range(4):
                    acc_color[c] += sample_color[c]

            # Average samples
            for c in range(4):
                output[py, px, c] = acc_color[c] / num_samples

    return output.to(device)


# Triton kernel for rendering (optimized version)
@triton.jit
def render_sample_kernel(
    # Output
    sample_colors_ptr,   # [H * W * num_samples, 4]
    # Sample positions
    sample_x_ptr,        # [H * W * num_samples]
    sample_y_ptr,        # [H * W * num_samples]
    # Path data
    segment_types_ptr,   # [num_paths, max_segments]
    segment_mask_ptr,    # [num_paths, max_segments]
    num_segments_ptr,    # [num_paths]
    points_ptr,          # [total_points, 2]
    point_offsets_ptr,   # [num_paths]
    num_points_ptr,      # [num_paths]
    is_closed_ptr,       # [num_paths]
    stroke_width_ptr,    # [num_paths]
    # Shape group data
    group_shape_ids_ptr,      # [num_groups, max_shapes_per_group]
    group_shape_mask_ptr,     # [num_groups, max_shapes_per_group]
    group_num_shapes_ptr,     # [num_groups]
    fill_color_ptr,           # [num_groups, 4]
    has_fill_ptr,             # [num_groups]
    stroke_color_ptr,         # [num_groups, 4]
    has_stroke_ptr,           # [num_groups]
    use_even_odd_ptr,         # [num_groups]
    # Shape mapping
    shape_types_ptr,          # [num_shapes]
    shape_indices_ptr,        # [num_shapes]
    # Background
    bg_r, bg_g, bg_b, bg_a,
    # Dimensions
    width: tl.constexpr,
    height: tl.constexpr,
    num_samples: tl.constexpr,
    num_paths: tl.constexpr,
    num_groups: tl.constexpr,
    max_segments: tl.constexpr,
    max_shapes_per_group: tl.constexpr,
    total_points: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Render samples kernel.

    Each thread processes one sample, computing its color by:
    1. Iterating over all shape groups
    2. For each shape, computing winding number (fill) and distance (stroke)
    3. Compositing fragment colors
    """
    pid = tl.program_id(0)
    sample_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_samples = width * height * num_samples
    mask = sample_idx < total_samples

    # Load sample position
    sx = tl.load(sample_x_ptr + sample_idx, mask=mask, other=0.0)
    sy = tl.load(sample_y_ptr + sample_idx, mask=mask, other=0.0)

    # Initialize accumulated color with background
    acc_r = bg_r
    acc_g = bg_g
    acc_b = bg_b
    acc_a = bg_a

    # Iterate over shape groups (back to front)
    # Note: This is a simplified version - full implementation needs:
    # - Proper fragment sorting
    # - BVH traversal for efficiency
    # - Full winding number computation

    # For now, store background color
    out_offset = sample_idx * 4
    tl.store(sample_colors_ptr + out_offset, acc_r, mask=mask)
    tl.store(sample_colors_ptr + out_offset + 1, acc_g, mask=mask)
    tl.store(sample_colors_ptr + out_offset + 2, acc_b, mask=mask)
    tl.store(sample_colors_ptr + out_offset + 3, acc_a, mask=mask)


def render_scene(
    scene: FlattenedScene,
    config: RenderConfig = None,
    use_python_reference: bool = True,  # Use Python for correctness, Triton for speed
) -> torch.Tensor:
    """
    Render a flattened scene to an image.

    Args:
        scene: Flattened scene containing shapes and groups
        config: Rendering configuration
        use_python_reference: If True, use slow but correct Python implementation

    Returns:
        [H, W, 4] RGBA image tensor
    """
    if config is None:
        config = RenderConfig()

    if use_python_reference:
        return render_scene_py(scene, config)

    # Triton implementation
    device = scene.device
    width = scene.canvas_width
    height = scene.canvas_height
    num_samples = config.num_samples_x * config.num_samples_y
    total_samples = width * height * num_samples

    # Generate sample positions
    sample_positions = torch.zeros((total_samples, 2), dtype=torch.float32, device=device)

    # Generate stratified sample positions
    idx = 0
    for py in range(height):
        for px in range(width):
            for sy in range(config.num_samples_y):
                for sx in range(config.num_samples_x):
                    ox = (sx + 0.5) / config.num_samples_x
                    oy = (sy + 0.5) / config.num_samples_y
                    sample_positions[idx, 0] = px + ox
                    sample_positions[idx, 1] = py + oy
                    idx += 1

    # Allocate output
    sample_colors = torch.zeros((total_samples, 4), dtype=torch.float32, device=device)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = ((total_samples + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    bg = config.background_color

    # For now, fall back to Python implementation
    # Full Triton kernel implementation is complex and needs careful optimization
    return render_scene_py(scene, config)


def render(
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
    High-level render function compatible with pydiffvg API.

    Args:
        canvas_width: Output image width
        canvas_height: Output image height
        shapes: List of shape objects (Path, Circle, etc.)
        shape_groups: List of ShapeGroup objects
        num_samples_x: Anti-aliasing samples in x
        num_samples_y: Anti-aliasing samples in y
        seed: Random seed
        background_color: [4] background RGBA, defaults to white
        use_prefiltering: Use SDF-based smooth edges

    Returns:
        [H, W, 4] RGBA image tensor
    """
    from .scene import flatten_scene

    if background_color is None:
        background_color = torch.tensor([1.0, 1.0, 1.0, 1.0])

    # Flatten scene
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scene = flatten_scene(canvas_width, canvas_height, shapes, shape_groups, device=device)

    # Configure renderer
    config = RenderConfig(
        num_samples_x=num_samples_x,
        num_samples_y=num_samples_y,
        seed=seed,
        background_color=tuple(background_color.cpu().tolist()),
        use_prefiltering=use_prefiltering,
    )

    # Render
    return render_scene(scene, config, use_python_reference=True)
