"""
Batched differentiable rendering for VAE training.

This module provides GPU-optimized, fully-vectorized rendering that:
1. Processes entire batches in parallel
2. Uses PyTorch operations for automatic gradient computation
3. Eliminates Python loops for GPU efficiency

Designed for training VAEs where gradients flow through the rendering.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


def _sample_cubic_bezier_batch(
    control_points: torch.Tensor,  # [B, P, S, 4, 2] - batch, paths, segments, 4 control pts, xy
    num_curve_samples: int = 17,
) -> torch.Tensor:
    """
    Sample points along cubic bezier curves.

    Args:
        control_points: [B, P, S, 4, 2] control points for cubic beziers
        num_curve_samples: number of samples per segment

    Returns:
        [B, P, S, num_curve_samples, 2] sampled curve points
    """
    device = control_points.device
    dtype = control_points.dtype

    # Parameter t from 0 to 1
    t = torch.linspace(0, 1, num_curve_samples, device=device, dtype=dtype)

    # Bezier weights: (1-t)^3, 3(1-t)^2*t, 3(1-t)*t^2, t^3
    one_minus_t = 1.0 - t
    w0 = one_minus_t ** 3                          # [T]
    w1 = 3.0 * (one_minus_t ** 2) * t
    w2 = 3.0 * one_minus_t * (t ** 2)
    w3 = t ** 3

    # Stack weights: [T, 4]
    weights = torch.stack([w0, w1, w2, w3], dim=-1)

    # Expand for batched matmul: [1, 1, 1, T, 4]
    weights = weights.view(1, 1, 1, num_curve_samples, 4)

    # control_points: [B, P, S, 4, 2]
    # We want: sum over 4 control points weighted by weights
    # Result: [B, P, S, T, 2]
    curve_points = torch.einsum('...tc,bpscd->bpstd', weights.squeeze(0).squeeze(0).squeeze(0), control_points)

    return curve_points


def _eval_cubic_bezier(t: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
    """
    Evaluate cubic Bezier at parameter t.

    Args:
        t: [...] parameter values in [0, 1]
        control_points: [..., 4, 2] control points

    Returns:
        [..., 2] points on curve
    """
    t = t.unsqueeze(-1)  # [..., 1]
    one_minus_t = 1.0 - t

    w0 = one_minus_t ** 3
    w1 = 3.0 * (one_minus_t ** 2) * t
    w2 = 3.0 * one_minus_t * (t ** 2)
    w3 = t ** 3

    # control_points: [..., 4, 2]
    # weights: [..., 1]
    p0 = control_points[..., 0, :]  # [..., 2]
    p1 = control_points[..., 1, :]
    p2 = control_points[..., 2, :]
    p3 = control_points[..., 3, :]

    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


def _eval_cubic_bezier_deriv(t: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
    """
    Evaluate derivative of cubic Bezier at parameter t.

    Args:
        t: [...] parameter values
        control_points: [..., 4, 2] control points

    Returns:
        [..., 2] derivative vectors
    """
    t = t.unsqueeze(-1)  # [..., 1]
    one_minus_t = 1.0 - t

    w0 = 3.0 * (one_minus_t ** 2)
    w1 = 6.0 * one_minus_t * t
    w2 = 3.0 * (t ** 2)

    p0 = control_points[..., 0, :]
    p1 = control_points[..., 1, :]
    p2 = control_points[..., 2, :]
    p3 = control_points[..., 3, :]

    return w0 * (p1 - p0) + w1 * (p2 - p1) + w2 * (p3 - p2)


def _closest_point_cubic_bezier_newton(
    pt: torch.Tensor,  # [..., 2]
    control_points: torch.Tensor,  # [..., 4, 2]
    num_init_samples: int = 8,
    num_newton_iters: int = 4,
) -> torch.Tensor:
    """
    Find closest point on cubic Bezier using sampling + Newton refinement.

    This is a differentiable implementation for PyTorch autograd.

    Args:
        pt: [..., 2] query points
        control_points: [..., 4, 2] Bezier control points
        num_init_samples: number of initial samples for coarse search
        num_newton_iters: Newton refinement iterations

    Returns:
        [...] squared distances to closest points
    """
    device = pt.device
    dtype = pt.dtype

    # Initial sampling to find good starting point
    t_samples = torch.linspace(0, 1, num_init_samples + 1, device=device, dtype=dtype)

    # Evaluate curve at all sample points
    # t_samples: [T], control_points: [..., 4, 2] -> need [..., T, 2]
    shape = control_points.shape[:-2]  # [...]
    T = len(t_samples)

    # Expand for broadcasting
    t_exp = t_samples.view(*([1] * len(shape)), T)  # [..., T] with leading 1s
    t_exp = t_exp.expand(*shape, T)

    # Evaluate at all t
    # We need to compute for each [...] and each T
    # Reshape control_points: [..., 4, 2] -> [..., 1, 4, 2]
    cp_exp = control_points.unsqueeze(-3)  # [..., 1, 4, 2]
    cp_exp = cp_exp.expand(*shape, T, 4, 2)  # [..., T, 4, 2]

    # Evaluate curve: [..., T, 2]
    curve_pts = _eval_cubic_bezier(t_exp, cp_exp)

    # Query points: [..., 2] -> [..., 1, 2]
    pt_exp = pt.unsqueeze(-2)  # [..., 1, 2]

    # Squared distances: [..., T]
    diff = curve_pts - pt_exp
    dist_sq = (diff ** 2).sum(dim=-1)

    # Find best initial t
    best_idx = dist_sq.argmin(dim=-1)  # [...]
    best_t = t_samples[best_idx]  # [...]

    # Newton refinement
    # We minimize f(t) = |B(t) - pt|^2
    # f'(t) = 2 * (B(t) - pt) · B'(t)
    # f''(t) = 2 * (B'(t) · B'(t) + (B(t) - pt) · B''(t))
    # Newton step: t_new = t - f'(t) / f''(t)

    t = best_t
    for _ in range(num_newton_iters):
        # Evaluate B(t) and B'(t)
        B = _eval_cubic_bezier(t, control_points)  # [..., 2]
        dB = _eval_cubic_bezier_deriv(t, control_points)  # [..., 2]

        # f' = 2 * (B - pt) · dB
        residual = B - pt  # [..., 2]
        f_prime = 2.0 * (residual * dB).sum(dim=-1)  # [...]

        # For f'', we need B''(t)
        # B''(t) = 6(1-t)(P2-2P1+P0) + 6t(P3-2P2+P1)
        t_unsq = t.unsqueeze(-1)
        one_minus_t = 1.0 - t_unsq
        p0 = control_points[..., 0, :]
        p1 = control_points[..., 1, :]
        p2 = control_points[..., 2, :]
        p3 = control_points[..., 3, :]
        d2B = 6.0 * one_minus_t * (p2 - 2*p1 + p0) + 6.0 * t_unsq * (p3 - 2*p2 + p1)

        # f'' = 2 * (dB · dB + residual · d2B)
        f_double_prime = 2.0 * ((dB * dB).sum(dim=-1) + (residual * d2B).sum(dim=-1))

        # Newton step with safety
        step = f_prime / (f_double_prime + 1e-6 * (f_double_prime.abs() < 1e-6).float())
        t = t - step

        # Clamp to valid range
        t = torch.clamp(t, 0.0, 1.0)

    # Final distance computation
    B_final = _eval_cubic_bezier(t, control_points)
    final_dist_sq = ((B_final - pt) ** 2).sum(dim=-1)

    return torch.sqrt(final_dist_sq + 1e-8)


def _compute_min_distance_bezier_batch(
    sample_pos: torch.Tensor,     # [N, 2] flat sample positions
    control_points: torch.Tensor,  # [B, P, S, 4, 2] cubic bezier control points
) -> torch.Tensor:
    """
    Compute minimum distance from samples to cubic Bezier curves.

    Uses proper closest-point-on-curve algorithm with Newton refinement.

    Args:
        sample_pos: [N, 2] sample positions (H*W*num_samples)
        control_points: [B, P, S, 4, 2] control points for cubic beziers

    Returns:
        [B, P, N] minimum distances
    """
    B, P, S, _, _ = control_points.shape
    N = sample_pos.shape[0]
    device = sample_pos.device
    dtype = sample_pos.dtype

    # For each sample point, find distance to each path
    # We need to compute distance to each segment and take minimum

    # sample_pos: [N, 2] -> [1, 1, N, 1, 2]
    # control_points: [B, P, S, 4, 2] -> [B, P, 1, S, 4, 2]

    sample_exp = sample_pos.view(1, 1, N, 1, 2).expand(B, P, N, S, 2)  # [B, P, N, S, 2]
    cp_exp = control_points.unsqueeze(2).expand(B, P, N, S, 4, 2)  # [B, P, N, S, 4, 2]

    # Compute distance for each segment
    # Flatten to [B*P*N*S, 2] and [B*P*N*S, 4, 2]
    sample_flat = sample_exp.reshape(-1, 2)
    cp_flat = cp_exp.reshape(-1, 4, 2)

    # Get distances for each segment
    dist_flat = _closest_point_cubic_bezier_newton(sample_flat, cp_flat)  # [B*P*N*S]

    # Reshape and take minimum over segments
    dist = dist_flat.view(B, P, N, S)
    min_dist = dist.min(dim=-1).values  # [B, P, N]

    return min_dist


def _compute_min_distance_batch(
    sample_pos: torch.Tensor,     # [N, 2] flat sample positions
    curve_points: torch.Tensor,   # [B, P, S*T, 2] all curve points flattened per path
) -> torch.Tensor:
    """
    Compute minimum distance from samples to curve points.

    NOTE: This is the legacy sampling-based approach. Use _compute_min_distance_bezier_batch
    for proper curve distance with Newton refinement.

    Args:
        sample_pos: [N, 2] sample positions (H*W*num_samples)
        curve_points: [B, P, C, 2] curve sample points

    Returns:
        [B, P, N] minimum distances
    """
    # sample_pos: [N, 2] -> [1, 1, N, 2]
    # curve_points: [B, P, C, 2] -> [B, P, 1, C, 2]

    B, P, C, _ = curve_points.shape
    N = sample_pos.shape[0]

    # Compute distances in chunks to manage memory
    # For each (batch, path), compute distance to all samples

    # Reshape for broadcasting:
    # sample_pos: [1, 1, N, 1, 2]
    # curve_points: [B, P, 1, C, 2]
    sample_pos_exp = sample_pos.view(1, 1, N, 1, 2)
    curve_points_exp = curve_points.view(B, P, 1, C, 2)

    # Compute squared distances: [B, P, N, C]
    diff = sample_pos_exp - curve_points_exp
    dist_sq = (diff ** 2).sum(dim=-1)

    # Minimum over curve points: [B, P, N]
    min_dist_sq = dist_sq.min(dim=-1).values
    min_dist = torch.sqrt(min_dist_sq + 1e-8)

    return min_dist


def _soft_winding_number_batch(
    sample_pos: torch.Tensor,    # [N, 2]
    curve_points: torch.Tensor,  # [B, P, C, 2] sampled curve (treated as polyline)
    softness: float = 0.1,
) -> torch.Tensor:
    """
    Compute soft winding number for fill detection.

    Uses a soft ray-crossing algorithm that allows gradients to flow.

    Args:
        sample_pos: [N, 2] sample positions
        curve_points: [B, P, C, 2] curve sample points (polyline approximation)
        softness: transition sharpness

    Returns:
        [B, P, N] soft winding numbers
    """
    B, P, C, _ = curve_points.shape
    N = sample_pos.shape[0]
    device = sample_pos.device

    # Get line segments: p0 -> p1
    # p0: [B, P, C-1, 2], p1: [B, P, C-1, 2]
    p0 = curve_points[:, :, :-1, :]  # [B, P, C-1, 2]
    p1 = curve_points[:, :, 1:, :]   # [B, P, C-1, 2]

    # Expand for broadcasting
    # sample_pos: [1, 1, N, 1, 2]
    # p0, p1: [B, P, 1, C-1, 2]
    sample_exp = sample_pos.view(1, 1, N, 1, 2)
    p0_exp = p0.view(B, P, 1, C-1, 2)
    p1_exp = p1.view(B, P, 1, C-1, 2)

    # dy for each segment: [B, P, 1, C-1]
    dy = p1_exp[..., 1] - p0_exp[..., 1]

    # Compute t parameter where ray at sample_y intersects line
    # t = (sample_y - p0_y) / dy
    pt_y = sample_exp[..., 1]  # [1, 1, N, 1]
    pt_x = sample_exp[..., 0]

    dy_safe = torch.where(torch.abs(dy) > 1e-8, dy, torch.ones_like(dy) * 1e-8)
    t = (pt_y - p0_exp[..., 1]) / dy_safe  # [B, P, N, C-1]

    # X coordinate of intersection
    x_int = p0_exp[..., 0] + t * (p1_exp[..., 0] - p0_exp[..., 0])

    # Soft validity checks using sigmoid
    t_valid = torch.sigmoid((t + 0.01) / softness) * torch.sigmoid((1.01 - t) / softness)
    x_valid = torch.sigmoid((x_int - pt_x + 0.01) / softness)

    # Direction: +1 for upward (dy > 0), -1 for downward
    direction = torch.where(dy > 0, torch.ones_like(dy), -torch.ones_like(dy))

    # Contribution: zero for horizontal segments
    contrib = torch.where(
        torch.abs(dy) > 1e-8,
        direction * t_valid * x_valid,
        torch.zeros_like(t_valid)
    )

    # Sum over all segments: [B, P, N]
    winding = contrib.sum(dim=-1)

    return winding


# Triton kernel for computing minimum distance - more efficient than PyTorch
@triton.jit
def _min_dist_kernel(
    # Sample positions [N]
    sample_x_ptr,
    sample_y_ptr,
    # Curve points [num_curve_pts, 2] flattened
    curve_x_ptr,
    curve_y_ptr,
    # Output [N]
    min_dist_ptr,
    # Params
    N: tl.constexpr,
    num_curve_pts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute minimum distance from samples to curve points."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load sample position
    sx = tl.load(sample_x_ptr + offs, mask=mask, other=0.0)
    sy = tl.load(sample_y_ptr + offs, mask=mask, other=0.0)

    # Initialize minimum distance squared
    min_dist_sq = tl.full([BLOCK_SIZE], 1e10, dtype=tl.float32)

    # Iterate over all curve points
    for i in range(num_curve_pts):
        cx = tl.load(curve_x_ptr + i)
        cy = tl.load(curve_y_ptr + i)

        dx = sx - cx
        dy_val = sy - cy
        dist_sq = dx * dx + dy_val * dy_val

        min_dist_sq = tl.minimum(min_dist_sq, dist_sq)

    # Store result
    min_dist = tl.sqrt(min_dist_sq + 1e-8)
    tl.store(min_dist_ptr + offs, min_dist, mask=mask)


@triton.jit
def _soft_winding_kernel(
    # Sample positions [N]
    sample_x_ptr,
    sample_y_ptr,
    # Polyline points [num_pts, 2] flattened
    poly_x_ptr,
    poly_y_ptr,
    # Output [N]
    winding_ptr,
    # Params
    N: tl.constexpr,
    num_pts: tl.constexpr,
    softness,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute soft winding number using ray-crossing."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load sample position
    pt_x = tl.load(sample_x_ptr + offs, mask=mask, other=0.0)
    pt_y = tl.load(sample_y_ptr + offs, mask=mask, other=0.0)

    # Initialize winding
    winding = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over line segments
    num_segs = num_pts - 1
    for i in range(num_segs):
        p0_x = tl.load(poly_x_ptr + i)
        p0_y = tl.load(poly_y_ptr + i)
        p1_x = tl.load(poly_x_ptr + i + 1)
        p1_y = tl.load(poly_y_ptr + i + 1)

        dy = p1_y - p0_y
        dy_abs = tl.abs(dy)

        # Compute t where ray intersects segment
        dy_safe = tl.where(dy_abs > 1e-8, dy, 1e-8)
        t = (pt_y - p0_y) / dy_safe

        # X coordinate of intersection
        x_int = p0_x + t * (p1_x - p0_x)

        # Soft validity using sigmoid approximation
        # sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)
        t_low = (t + 0.01) / softness
        t_high = (1.01 - t) / softness
        x_diff = (x_int - pt_x + 0.01) / softness

        # Use tl.sigmoid (available in recent Triton versions)
        # If not available, approximate with: 1 / (1 + exp(-x))
        t_valid = 1.0 / (1.0 + tl.exp(-t_low)) * 1.0 / (1.0 + tl.exp(-t_high))
        x_valid = 1.0 / (1.0 + tl.exp(-x_diff))

        # Direction
        direction = tl.where(dy > 0.0, 1.0, -1.0)

        # Contribution
        contrib = tl.where(dy_abs > 1e-8, direction * t_valid * x_valid, 0.0)
        winding = winding + contrib

    # Store result
    tl.store(winding_ptr + offs, winding, mask=mask)


def render_batch(
    canvas_width: int,
    canvas_height: int,
    control_points: torch.Tensor,  # [B, P, S, 4, 2] - batch, paths, segments, 4 ctrl pts, xy
    stroke_widths: torch.Tensor,   # [B, P] stroke widths
    alphas: torch.Tensor,          # [B, P] alpha values
    num_samples: int = 2,
    use_fill: bool = True,
    background: float = 1.0,       # Background value (1.0 = white)
) -> torch.Tensor:
    """
    Batched differentiable vector graphics rendering.

    This is the main entry point for training. All operations are batched
    and run on GPU with automatic gradient computation.

    Args:
        canvas_width: Output image width
        canvas_height: Output image height
        control_points: [B, P, S, 4, 2] cubic bezier control points
        stroke_widths: [B, P] stroke width for each path
        alphas: [B, P] alpha (opacity) for each path
        num_samples: Anti-aliasing samples per axis
        use_fill: Whether to fill the interior of paths
        background: Background color (grayscale)

    Returns:
        [B, 1, H, W] grayscale images with gradients
    """
    device = control_points.device
    dtype = control_points.dtype
    B, P, S, _, _ = control_points.shape
    H, W = canvas_height, canvas_width

    # Number of curve samples for polyline approximation
    num_curve_samples = 17

    # Sample bezier curves: [B, P, S, T, 2]
    curve_samples = _sample_cubic_bezier_batch(control_points, num_curve_samples)

    # Flatten segments: [B, P, S*T, 2]
    curve_points = curve_samples.view(B, P, S * num_curve_samples, 2)

    # Generate sample positions: [H, W, num_samples^2, 2]
    samples_per_axis = num_samples
    total_samples = samples_per_axis * samples_per_axis

    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    oy = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis
    ox = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis

    # Meshgrid for all combinations
    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')
    sample_x = px_grid + ox_grid
    sample_y = py_grid + oy_grid

    # Flatten to [N, 2] where N = H*W*total_samples
    sample_pos = torch.stack([
        sample_x.reshape(-1),
        sample_y.reshape(-1)
    ], dim=-1)

    N = sample_pos.shape[0]

    # Initialize sample colors with background
    sample_colors = torch.full((B, N), background, device=device, dtype=dtype)

    # Process all paths in parallel
    # Compute minimum distances: [B, P, N]
    min_dist = _compute_min_distance_batch(sample_pos, curve_points)

    # Compute winding numbers if using fill: [B, P, N]
    if use_fill:
        # Close the curve by adding first point at end
        curve_closed = torch.cat([
            curve_points,
            curve_points[:, :, :1, :]
        ], dim=2)
        winding = _soft_winding_number_batch(sample_pos, curve_closed, softness=0.1)

        # Inside = |winding| >= 0.5 (soft threshold)
        inside = torch.sigmoid((torch.abs(winding) - 0.5) * 10.0)  # [B, P, N]
    else:
        inside = torch.zeros(B, P, N, device=device, dtype=dtype)

    # Compute stroke coverage from distance
    # Note: pydiffvg interprets stroke_width as the half-width (radius), not diameter
    half_widths = stroke_widths.view(B, P, 1)  # [B, P, 1] - don't divide by 2 to match pydiffvg
    transition_width = 0.25
    stroke_edge = torch.sigmoid((half_widths - min_dist) / transition_width)  # [B, P, N]

    # Total coverage: inside OR on stroke edge
    coverage = torch.maximum(inside, stroke_edge)  # [B, P, N]

    # Apply alpha for each path
    alphas_exp = alphas.view(B, P, 1)  # [B, P, 1]
    stroke_contrib = coverage * alphas_exp  # [B, P, N]

    # Composite all paths (back to front, multiplicative blending)
    # sample_colors = background * prod((1 - stroke_contrib) for each path)
    for p in range(P):
        sample_colors = sample_colors * (1.0 - stroke_contrib[:, p, :])

    # Reshape and average samples per pixel
    sample_colors = sample_colors.view(B, H, W, total_samples)
    pixel_colors = sample_colors.mean(dim=-1)  # [B, H, W]

    # Add channel dimension: [B, 1, H, W]
    output = pixel_colors.unsqueeze(1)

    return output


def render_batch_triton(
    canvas_width: int,
    canvas_height: int,
    control_points: torch.Tensor,  # [B, P, S, 4, 2]
    stroke_widths: torch.Tensor,   # [B, P]
    alphas: torch.Tensor,          # [B, P]
    num_samples: int = 2,
    use_fill: bool = True,
    background: float = 1.0,
) -> torch.Tensor:
    """
    Batched rendering using Triton kernels for inner loops.

    Similar to render_batch but uses custom Triton kernels for
    distance and winding computation for better performance.
    """
    device = control_points.device
    dtype = control_points.dtype

    if device.type != 'cuda':
        return render_batch(canvas_width, canvas_height, control_points,
                          stroke_widths, alphas, num_samples, use_fill, background)

    B, P, S, _, _ = control_points.shape
    H, W = canvas_height, canvas_width

    num_curve_samples = 17

    # Sample bezier curves
    curve_samples = _sample_cubic_bezier_batch(control_points, num_curve_samples)
    curve_points = curve_samples.view(B, P, S * num_curve_samples, 2)

    # Generate sample positions
    samples_per_axis = num_samples
    total_samples = samples_per_axis * samples_per_axis

    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    oy = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis
    ox = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis

    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')
    sample_x = (px_grid + ox_grid).reshape(-1).contiguous()
    sample_y = (py_grid + oy_grid).reshape(-1).contiguous()

    N = sample_x.shape[0]
    num_curve_pts = S * num_curve_samples

    # Initialize output
    sample_colors = torch.full((B, N), background, device=device, dtype=dtype)

    BLOCK_SIZE = 256
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Process each batch and path using Triton kernels
    for b in range(B):
        for p in range(P):
            # Get curve points for this path
            path_curve = curve_points[b, p].contiguous()  # [C, 2]
            curve_x = path_curve[:, 0].contiguous()
            curve_y = path_curve[:, 1].contiguous()

            # Allocate outputs
            min_dist = torch.empty(N, device=device, dtype=dtype)
            winding = torch.empty(N, device=device, dtype=dtype)

            # Compute minimum distance
            _min_dist_kernel[grid](
                sample_x, sample_y,
                curve_x, curve_y,
                min_dist,
                N, num_curve_pts, BLOCK_SIZE
            )

            # Compute winding number (add closing segment)
            curve_closed_x = torch.cat([curve_x, curve_x[:1]])
            curve_closed_y = torch.cat([curve_y, curve_y[:1]])

            _soft_winding_kernel[grid](
                sample_x, sample_y,
                curve_closed_x, curve_closed_y,
                winding,
                N, num_curve_pts + 1, 0.1, BLOCK_SIZE
            )

            # Compute coverage
            # Note: pydiffvg interprets stroke_width as the half-width (radius)
            half_width = stroke_widths[b, p]  # Don't divide by 2 to match pydiffvg
            alpha = alphas[b, p]

            if use_fill:
                inside = torch.sigmoid((torch.abs(winding) - 0.5) * 10.0)
            else:
                inside = torch.zeros_like(min_dist)

            stroke_edge = torch.sigmoid((half_width - min_dist) / 0.25)
            coverage = torch.maximum(inside, stroke_edge)
            stroke_contrib = coverage * alpha

            # Composite
            sample_colors[b] = sample_colors[b] * (1.0 - stroke_contrib)

    # Reshape and average
    sample_colors = sample_colors.view(B, H, W, total_samples)
    pixel_colors = sample_colors.mean(dim=-1)
    output = pixel_colors.unsqueeze(1)

    return output


# Optimized version that processes all paths together
def render_batch_fast(
    canvas_width: int,
    canvas_height: int,
    control_points: torch.Tensor,  # [B, P, S, 4, 2]
    stroke_widths: torch.Tensor,   # [B, P]
    alphas: torch.Tensor,          # [B, P]
    num_samples: int = 2,
    use_fill: bool = True,
    background: float = 1.0,
) -> torch.Tensor:
    """
    Fastest batched rendering - fully vectorized PyTorch operations.

    Key optimizations:
    1. All bezier sampling done in parallel
    2. Distance computation uses efficient broadcasting
    3. Winding number computed with vectorized operations
    4. No Python loops over paths
    """
    device = control_points.device
    dtype = control_points.dtype
    B, P, S, _, _ = control_points.shape
    H, W = canvas_height, canvas_width

    num_curve_samples = 17

    # Sample all bezier curves at once: [B, P, S, T, 2]
    curve_samples = _sample_cubic_bezier_batch(control_points, num_curve_samples)

    # Flatten segments per path: [B, P, C, 2] where C = S*T
    C = S * num_curve_samples
    curve_points = curve_samples.reshape(B, P, C, 2).contiguous()

    # Generate sample positions
    samples_per_axis = num_samples
    total_samples = samples_per_axis * samples_per_axis

    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    oy = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis
    ox = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis

    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')
    sample_x = px_grid + ox_grid  # [H, W, Sy, Sx]
    sample_y = py_grid + oy_grid

    # Flatten samples: [N] where N = H*W*Sy*Sx
    N = H * W * total_samples
    sample_x_flat = sample_x.reshape(N)
    sample_y_flat = sample_y.reshape(N)

    # Stack: [N, 2]
    sample_pos = torch.stack([sample_x_flat, sample_y_flat], dim=-1)

    # Compute distances: [B, P, N]
    # Use proper closest-point-on-curve with Newton refinement for smooth strokes
    min_dist = _compute_min_distance_bezier_batch(sample_pos, control_points)

    # Winding number computation (vectorized)
    if use_fill:
        # Close the curve
        curve_closed = torch.cat([curve_points, curve_points[:, :, :1, :]], dim=2)  # [B, P, C+1, 2]

        # Line segments: p0 -> p1
        p0 = curve_closed[:, :, :-1, :]  # [B, P, C, 2]
        p1 = curve_closed[:, :, 1:, :]   # [B, P, C, 2]

        # Expand for broadcasting with samples
        # sample_pos: [N, 2] -> [1, 1, N, 1, 2]
        # p0, p1: [B, P, C, 2] -> [B, P, 1, C, 2]
        p0_exp = p0.view(B, P, 1, C, 2)
        p1_exp = p1.view(B, P, 1, C, 2)

        # dy for each segment
        dy = p1_exp[..., 1] - p0_exp[..., 1]  # [B, P, 1, C]

        # Sample positions
        pt_y = sample_pos[:, 1].view(1, 1, N, 1)  # [1, 1, N, 1]
        pt_x = sample_pos[:, 0].view(1, 1, N, 1)

        # t parameter
        dy_safe = torch.where(torch.abs(dy) > 1e-8, dy, torch.ones_like(dy) * 1e-8)
        t = (pt_y - p0_exp[..., 1]) / dy_safe  # [B, P, N, C]

        # X intersection
        x_int = p0_exp[..., 0] + t * (p1_exp[..., 0] - p0_exp[..., 0])  # [B, P, N, C]

        # Soft validity
        softness = 0.1
        t_valid = torch.sigmoid((t + 0.01) / softness) * torch.sigmoid((1.01 - t) / softness)
        x_valid = torch.sigmoid((x_int - pt_x + 0.01) / softness)

        # Direction
        direction = torch.where(dy > 0, torch.ones_like(dy), -torch.ones_like(dy))

        # Contribution
        contrib = torch.where(torch.abs(dy) > 1e-8, direction * t_valid * x_valid, torch.zeros_like(t_valid))

        # Sum over segments: [B, P, N]
        winding = contrib.sum(dim=-1)

        # Inside = |winding| >= 0.5
        inside = torch.sigmoid((torch.abs(winding) - 0.5) * 10.0)
    else:
        inside = torch.zeros(B, P, N, device=device, dtype=dtype)

    # Stroke coverage
    # Note: pydiffvg interprets stroke_width as the half-width (radius), not diameter
    # So stroke_width=10 means the stroke extends 10 pixels from the curve center
    half_widths = stroke_widths.view(B, P, 1)  # Don't divide by 2 to match pydiffvg
    stroke_edge = torch.sigmoid((half_widths - min_dist) / 0.25)

    # Total coverage
    coverage = torch.maximum(inside, stroke_edge)

    # Apply alpha
    alphas_exp = alphas.view(B, P, 1)
    stroke_contrib = coverage * alphas_exp  # [B, P, N]

    # Composite: product over paths
    # sample_colors = background * prod(1 - stroke_contrib[p]) for all p
    one_minus_contrib = 1.0 - stroke_contrib  # [B, P, N]
    combined = one_minus_contrib.prod(dim=1)  # [B, N]
    sample_colors = background * combined

    # Reshape and average
    sample_colors = sample_colors.view(B, H, W, total_samples)
    pixel_colors = sample_colors.mean(dim=-1)

    return pixel_colors.unsqueeze(1)
