"""
Pure PyTorch differentiable renderer.

This module provides a differentiable renderer implemented entirely in PyTorch,
enabling gradient flow through the rendering pipeline for optimization tasks.
"""

import torch
from typing import List


def render_pytorch(
    canvas_width: int,
    canvas_height: int,
    shapes: list,
    shape_groups: list,
    num_samples: int = 2,
) -> torch.Tensor:
    """
    Differentiable rendering using pure PyTorch.

    Args:
        canvas_width: Output image width
        canvas_height: Output image height
        shapes: List of Path objects
        shape_groups: List of ShapeGroup objects
        num_samples: Anti-aliasing samples per dimension

    Returns:
        [H, W, 4] RGBA image tensor with gradients
    """
    device = shapes[0].points.device if shapes else torch.device('cpu')
    height, width = canvas_height, canvas_width
    total_samples = num_samples * num_samples

    # Generate sample positions
    py = torch.arange(height, device=device, dtype=torch.float32)
    px = torch.arange(width, device=device, dtype=torch.float32)
    oy = (torch.arange(num_samples, device=device, dtype=torch.float32) + 0.5) / num_samples
    ox = (torch.arange(num_samples, device=device, dtype=torch.float32) + 0.5) / num_samples

    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')
    sample_x = px_grid + ox_grid
    sample_y = py_grid + oy_grid

    sample_pos = torch.stack([sample_x, sample_y], dim=-1)
    N = height * width * total_samples
    sample_pos_flat = sample_pos.reshape(N, 2)

    # Initialize with white background
    sample_colors = torch.ones(N, 4, device=device)

    # Process each shape group
    for group in shape_groups:
        fill_color = group.fill_color
        if fill_color is None:
            continue

        for shape_idx in group.shape_ids.tolist():
            if shape_idx >= len(shapes):
                continue

            shape = shapes[shape_idx]
            points = shape.points
            num_ctrl = shape.num_control_points

            if len(points) < 2:
                continue

            # Sample curve as polyline
            curve_points = _sample_path(points, num_ctrl, device)
            if len(curve_points) < 2:
                continue

            # Compute fill coverage
            fill_coverage = _compute_fill_coverage(sample_pos_flat, curve_points, shape.is_closed)
            fill_alpha = fill_coverage * fill_color[3]
            sample_colors = _alpha_blend(sample_colors, fill_color[:3], fill_alpha)

    # Average samples per pixel
    sample_colors = sample_colors.reshape(height, width, total_samples, 4)
    return sample_colors.mean(dim=2)


def _sample_path(points: torch.Tensor, num_ctrl: torch.Tensor, device: torch.device,
                 samples_per_segment: int = 16) -> torch.Tensor:
    """Sample a path as a dense polyline."""
    result = [points[0:1]]
    pt_idx = 1

    for seg_type in num_ctrl.tolist():
        if seg_type == 0:  # Line
            if pt_idx < len(points):
                result.append(points[pt_idx:pt_idx+1])
                pt_idx += 1
        elif seg_type == 1:  # Quadratic
            if pt_idx + 1 < len(points):
                p0, p1, p2 = result[-1][-1], points[pt_idx], points[pt_idx + 1]
                t = torch.linspace(0, 1, samples_per_segment, device=device)[1:]
                w0 = ((1-t)**2).unsqueeze(-1)
                w1 = (2*(1-t)*t).unsqueeze(-1)
                w2 = (t**2).unsqueeze(-1)
                result.append(w0*p0 + w1*p1 + w2*p2)
                pt_idx += 2
        else:  # Cubic
            if pt_idx + 2 < len(points):
                p0, p1, p2, p3 = result[-1][-1], points[pt_idx], points[pt_idx+1], points[pt_idx+2]
                t = torch.linspace(0, 1, samples_per_segment, device=device)[1:]
                w0 = ((1-t)**3).unsqueeze(-1)
                w1 = (3*(1-t)**2*t).unsqueeze(-1)
                w2 = (3*(1-t)*t**2).unsqueeze(-1)
                w3 = (t**3).unsqueeze(-1)
                result.append(w0*p0 + w1*p1 + w2*p2 + w3*p3)
                pt_idx += 3

    return torch.cat(result, dim=0) if result else torch.zeros(0, 2, device=device)


def _compute_fill_coverage(sample_pos: torch.Tensor, curve_points: torch.Tensor,
                           is_closed: bool) -> torch.Tensor:
    """Compute fill coverage using soft winding number."""
    curve_closed = torch.cat([curve_points, curve_points[:1]], dim=0)
    p0, p1 = curve_closed[:-1], curve_closed[1:]

    sample_pos_exp = sample_pos.unsqueeze(1)
    p0_exp, p1_exp = p0.unsqueeze(0), p1.unsqueeze(0)

    dy = p1_exp[..., 1] - p0_exp[..., 1]
    pt_y, pt_x = sample_pos_exp[..., 1], sample_pos_exp[..., 0]

    dy_safe = torch.where(torch.abs(dy) > 1e-8, dy, torch.ones_like(dy) * 1e-8)
    t = (pt_y - p0_exp[..., 1]) / dy_safe
    x_int = p0_exp[..., 0] + t * (p1_exp[..., 0] - p0_exp[..., 0])

    softness = 0.1
    t_valid = torch.sigmoid((t + 0.01) / softness) * torch.sigmoid((1.01 - t) / softness)
    x_valid = torch.sigmoid((x_int - pt_x + 0.01) / softness)

    direction = torch.where(dy > 0, torch.ones_like(dy), -torch.ones_like(dy))
    contrib = torch.where(torch.abs(dy) > 1e-8, direction * t_valid * x_valid, torch.zeros_like(t_valid))
    winding = contrib.sum(dim=-1)

    return torch.sigmoid((torch.abs(winding) - 0.5) * 10.0)


def _alpha_blend(dst: torch.Tensor, src_rgb: torch.Tensor, src_alpha: torch.Tensor) -> torch.Tensor:
    """Alpha blend src over dst."""
    src_alpha = src_alpha.unsqueeze(-1)
    src_rgb = src_rgb.unsqueeze(0)

    out_alpha = src_alpha + dst[:, 3:4] * (1.0 - src_alpha)
    out_rgb = src_rgb * src_alpha + dst[:, :3] * dst[:, 3:4] * (1.0 - src_alpha)
    safe_alpha = torch.clamp(out_alpha, min=1e-8)
    out_rgb = out_rgb / safe_alpha

    return torch.cat([out_rgb, out_alpha], dim=-1)
