"""
Winding number computation for point-in-polygon testing.

Implements ray-curve intersection algorithm for determining if a point
is inside a filled path. Shoots a horizontal ray from the query point
to the right and counts signed intersections with path segments.

Ported from diffvg/winding_number.h
"""

import torch
import triton
import triton.language as tl
import math

from .solve import solve_quadratic_kernel, solve_cubic_kernel


@triton.jit
def ray_line_intersection(
    pt_x, pt_y,      # Query point
    p0_x, p0_y,      # Line start
    p1_x, p1_y,      # Line end
):
    """
    Compute winding number contribution from ray-line intersection.

    Shoots horizontal ray from (pt_x, pt_y) to the right.
    Returns +1 if line crosses upward, -1 if downward, 0 if no intersection.
    """
    # Line: p(t) = p0 + t*(p1 - p0), t in [0, 1]
    # Ray: (pt_x + t', pt_y), t' >= 0
    #
    # Intersection: pt_y = p0_y + t*(p1_y - p0_y)
    # => t = (pt_y - p0_y) / (p1_y - p0_y)

    dy = p1_y - p0_y

    # Horizontal line - no intersection with horizontal ray
    no_intersect = tl.abs(dy) < 1e-10

    # Compute t parameter on line
    t = tl.where(no_intersect, -1.0, (pt_y - p0_y) / dy)

    # Check if t is in valid range [0, 1]
    valid_t = (t >= 0.0) & (t <= 1.0)

    # Compute x-coordinate of intersection point
    # t' = p0_x + t*(p1_x - p0_x) - pt_x
    tp = p0_x - pt_x + t * (p1_x - p0_x)

    # Ray goes to the right, so t' must be >= 0
    valid_intersection = valid_t & (tp >= 0.0) & (~no_intersect)

    # Direction determines sign
    # Upward crossing (dy > 0) -> +1, downward (dy < 0) -> -1
    contribution = tl.where(valid_intersection, tl.where(dy > 0, 1, -1), 0)

    return contribution


@triton.jit
def ray_quadratic_intersection(
    pt_x, pt_y,           # Query point
    p0_x, p0_y,           # Start point
    p1_x, p1_y,           # Control point
    p2_x, p2_y,           # End point
):
    """
    Compute winding number contribution from ray-quadratic Bezier intersection.

    Quadratic Bezier: p(t) = (1-t)^2 * p0 + 2(1-t)t * p1 + t^2 * p2
                          = (p0 - 2p1 + p2)t^2 + (-2p0 + 2p1)t + p0
    """
    # Coefficients for y(t) = a*t^2 + b*t + c
    a = p0_y - 2.0 * p1_y + p2_y
    b = -2.0 * p0_y + 2.0 * p1_y
    c = p0_y - pt_y

    # Coefficients for x(t) (needed for computing intersection x)
    ax = p0_x - 2.0 * p1_x + p2_x
    bx = -2.0 * p0_x + 2.0 * p1_x

    # Solve a*t^2 + b*t + c = 0 for y = pt_y
    has_roots, t0, t1 = solve_quadratic_kernel(a, b, c)

    winding = 0

    # Check first root
    valid_t0 = has_roots & (t0 >= 0.0) & (t0 <= 1.0)
    # x coordinate at t0
    x_at_t0 = ax * t0 * t0 + bx * t0 + p0_x
    tp0 = x_at_t0 - pt_x
    valid_int0 = valid_t0 & (tp0 >= 0.0)
    # Derivative dy/dt at t0 = 2*a*t + b
    dy_dt0 = 2.0 * a * t0 + b
    contrib0 = tl.where(valid_int0, tl.where(dy_dt0 > 0, 1, -1), 0)
    winding = winding + contrib0

    # Check second root
    valid_t1 = has_roots & (t1 >= 0.0) & (t1 <= 1.0) & (tl.abs(t1 - t0) > 1e-8)
    x_at_t1 = ax * t1 * t1 + bx * t1 + p0_x
    tp1 = x_at_t1 - pt_x
    valid_int1 = valid_t1 & (tp1 >= 0.0)
    dy_dt1 = 2.0 * a * t1 + b
    contrib1 = tl.where(valid_int1, tl.where(dy_dt1 > 0, 1, -1), 0)
    winding = winding + contrib1

    return winding


@triton.jit
def ray_cubic_intersection(
    pt_x, pt_y,           # Query point
    p0_x, p0_y,           # Start point
    p1_x, p1_y,           # Control point 1
    p2_x, p2_y,           # Control point 2
    p3_x, p3_y,           # End point
):
    """
    Compute winding number contribution from ray-cubic Bezier intersection.

    Cubic Bezier: p(t) = (1-t)^3 * p0 + 3(1-t)^2*t * p1 + 3(1-t)*t^2 * p2 + t^3 * p3
                      = (-p0 + 3p1 - 3p2 + p3)t^3 + (3p0 - 6p1 + 3p2)t^2 + (-3p0 + 3p1)t + p0
    """
    # Coefficients for y(t) = a*t^3 + b*t^2 + c*t + d
    a = -p0_y + 3.0 * p1_y - 3.0 * p2_y + p3_y
    b = 3.0 * p0_y - 6.0 * p1_y + 3.0 * p2_y
    c = -3.0 * p0_y + 3.0 * p1_y
    d = p0_y - pt_y

    # Coefficients for x(t)
    ax = -p0_x + 3.0 * p1_x - 3.0 * p2_x + p3_x
    bx = 3.0 * p0_x - 6.0 * p1_x + 3.0 * p2_x
    cx = -3.0 * p0_x + 3.0 * p1_x

    # Solve cubic for t
    num_roots, t0, t1, t2 = solve_cubic_kernel(a, b, c, d)

    winding = 0

    # Helper for computing contribution at a given t
    # x(t) = ax*t^3 + bx*t^2 + cx*t + p0_x
    # dy/dt = 3*a*t^2 + 2*b*t + c

    # Root 0
    valid_t0 = (num_roots >= 1) & (t0 >= 0.0) & (t0 <= 1.0)
    t0_sq = t0 * t0
    t0_cb = t0_sq * t0
    x_at_t0 = ax * t0_cb + bx * t0_sq + cx * t0 + p0_x
    tp0 = x_at_t0 - pt_x
    valid_int0 = valid_t0 & (tp0 > 0.0)  # Strict > to handle endpoints consistently
    dy_dt0 = 3.0 * a * t0_sq + 2.0 * b * t0 + c
    contrib0 = tl.where(valid_int0, tl.where(dy_dt0 > 0, 1, -1), 0)
    winding = winding + contrib0

    # Root 1
    valid_t1 = (num_roots >= 2) & (t1 >= 0.0) & (t1 <= 1.0)
    t1_sq = t1 * t1
    t1_cb = t1_sq * t1
    x_at_t1 = ax * t1_cb + bx * t1_sq + cx * t1 + p0_x
    tp1 = x_at_t1 - pt_x
    valid_int1 = valid_t1 & (tp1 > 0.0)
    dy_dt1 = 3.0 * a * t1_sq + 2.0 * b * t1 + c
    contrib1 = tl.where(valid_int1, tl.where(dy_dt1 > 0, 1, -1), 0)
    winding = winding + contrib1

    # Root 2
    valid_t2 = (num_roots >= 3) & (t2 >= 0.0) & (t2 <= 1.0)
    t2_sq = t2 * t2
    t2_cb = t2_sq * t2
    x_at_t2 = ax * t2_cb + bx * t2_sq + cx * t2 + p0_x
    tp2 = x_at_t2 - pt_x
    valid_int2 = valid_t2 & (tp2 > 0.0)
    dy_dt2 = 3.0 * a * t2_sq + 2.0 * b * t2 + c
    contrib2 = tl.where(valid_int2, tl.where(dy_dt2 > 0, 1, -1), 0)
    winding = winding + contrib2

    return winding


@triton.jit
def compute_winding_number_path_kernel(
    # Query points [N, 2]
    query_x_ptr,
    query_y_ptr,
    # Path data (single path)
    segment_types_ptr,   # [max_segments]
    points_ptr,          # [total_points * 2]
    num_segments: tl.constexpr,
    num_points: tl.constexpr,
    is_closed: tl.constexpr,
    # Output
    winding_ptr,
    # Grid info
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute winding number for multiple query points against a single path.

    This kernel processes BLOCK_SIZE query points in parallel, each computing
    its winding number by iterating over all path segments.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load query point
    pt_x = tl.load(query_x_ptr + offs, mask=mask, other=0.0)
    pt_y = tl.load(query_y_ptr + offs, mask=mask, other=0.0)

    winding = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    point_idx = 0

    # Iterate over segments
    for seg_idx in range(num_segments):
        seg_type = tl.load(segment_types_ptr + seg_idx)

        if seg_type == 0:
            # Line segment
            i0 = point_idx
            i1 = (point_idx + 1) % num_points if is_closed else point_idx + 1

            p0_x = tl.load(points_ptr + i0 * 2)
            p0_y = tl.load(points_ptr + i0 * 2 + 1)
            p1_x = tl.load(points_ptr + i1 * 2)
            p1_y = tl.load(points_ptr + i1 * 2 + 1)

            contrib = ray_line_intersection(pt_x, pt_y, p0_x, p0_y, p1_x, p1_y)
            winding = winding + contrib
            point_idx += 1

        elif seg_type == 1:
            # Quadratic Bezier (1 control point)
            i0 = point_idx
            i1 = point_idx + 1
            i2 = (point_idx + 2) % num_points if is_closed else point_idx + 2

            p0_x = tl.load(points_ptr + i0 * 2)
            p0_y = tl.load(points_ptr + i0 * 2 + 1)
            p1_x = tl.load(points_ptr + i1 * 2)
            p1_y = tl.load(points_ptr + i1 * 2 + 1)
            p2_x = tl.load(points_ptr + i2 * 2)
            p2_y = tl.load(points_ptr + i2 * 2 + 1)

            contrib = ray_quadratic_intersection(
                pt_x, pt_y, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y
            )
            winding = winding + contrib
            point_idx += 2

        elif seg_type == 2:
            # Cubic Bezier (2 control points)
            i0 = point_idx
            i1 = point_idx + 1
            i2 = point_idx + 2
            i3 = (point_idx + 3) % num_points if is_closed else point_idx + 3

            p0_x = tl.load(points_ptr + i0 * 2)
            p0_y = tl.load(points_ptr + i0 * 2 + 1)
            p1_x = tl.load(points_ptr + i1 * 2)
            p1_y = tl.load(points_ptr + i1 * 2 + 1)
            p2_x = tl.load(points_ptr + i2 * 2)
            p2_y = tl.load(points_ptr + i2 * 2 + 1)
            p3_x = tl.load(points_ptr + i3 * 2)
            p3_y = tl.load(points_ptr + i3 * 2 + 1)

            contrib = ray_cubic_intersection(
                pt_x, pt_y,
                p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y
            )
            winding = winding + contrib
            point_idx += 3

    # Store result
    tl.store(winding_ptr + offs, winding, mask=mask)


def compute_winding_number_path(
    query_points: torch.Tensor,  # [N, 2]
    segment_types: torch.Tensor,  # [num_segments]
    points: torch.Tensor,        # [num_points, 2]
    is_closed: bool = True,
) -> torch.Tensor:
    """
    Compute winding numbers for query points against a path.

    Args:
        query_points: [N, 2] tensor of query (x, y) coordinates
        segment_types: [num_segments] tensor of segment types (0=line, 1=quad, 2=cubic)
        points: [num_points, 2] tensor of control points
        is_closed: Whether the path is closed

    Returns:
        [N] tensor of winding numbers (integers)
    """
    N = query_points.shape[0]
    device = query_points.device

    # Flatten points to [num_points * 2]
    points_flat = points.view(-1).contiguous()

    # Prepare output
    winding = torch.zeros(N, dtype=torch.int32, device=device)

    # Split query points into x and y for kernel
    query_x = query_points[:, 0].contiguous()
    query_y = query_points[:, 1].contiguous()

    num_segments = segment_types.shape[0]
    num_points = points.shape[0]

    BLOCK_SIZE = 256
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch kernel
    compute_winding_number_path_kernel[grid](
        query_x,
        query_y,
        segment_types,
        points_flat,
        num_segments,
        num_points,
        is_closed,
        winding,
        N,
        BLOCK_SIZE,
    )

    return winding


# Python reference implementation for testing
def compute_winding_number_path_py(
    query_point: tuple,
    segment_types: list,
    points: list,
    is_closed: bool = True,
) -> int:
    """Python reference implementation of winding number computation."""
    pt_x, pt_y = query_point
    winding = 0
    point_idx = 0
    num_points = len(points)

    for seg_type in segment_types:
        if seg_type == 0:
            # Line
            i0 = point_idx
            i1 = (point_idx + 1) % num_points if is_closed else point_idx + 1
            p0 = points[i0]
            p1 = points[i1]

            dy = p1[1] - p0[1]
            if abs(dy) > 1e-10:
                t = (pt_y - p0[1]) / dy
                if 0 <= t <= 1:
                    tp = p0[0] - pt_x + t * (p1[0] - p0[0])
                    if tp >= 0:
                        if dy > 0:
                            winding += 1
                        else:
                            winding -= 1
            point_idx += 1

        elif seg_type == 1:
            # Quadratic
            i0 = point_idx
            i1 = point_idx + 1
            i2 = (point_idx + 2) % num_points if is_closed else point_idx + 2
            p0, p1, p2 = points[i0], points[i1], points[i2]

            a = p0[1] - 2 * p1[1] + p2[1]
            b = -2 * p0[1] + 2 * p1[1]
            c = p0[1] - pt_y

            ax = p0[0] - 2 * p1[0] + p2[0]
            bx = -2 * p0[0] + 2 * p1[0]

            discrim = b * b - 4 * a * c
            if discrim >= 0:
                sqrt_d = math.sqrt(discrim)
                for sign in [1, -1]:
                    if abs(a) > 1e-10:
                        t = (-b + sign * sqrt_d) / (2 * a)
                    else:
                        if abs(b) > 1e-10:
                            t = -c / b
                        else:
                            continue

                    if 0 <= t <= 1:
                        x_at_t = ax * t * t + bx * t + p0[0]
                        if x_at_t - pt_x >= 0:
                            dy_dt = 2 * a * t + b
                            if dy_dt > 0:
                                winding += 1
                            else:
                                winding -= 1
            point_idx += 2

        elif seg_type == 2:
            # Cubic - use reference cubic solver
            from .solve import solve_cubic_py

            i0 = point_idx
            i1 = point_idx + 1
            i2 = point_idx + 2
            i3 = (point_idx + 3) % num_points if is_closed else point_idx + 3
            p0, p1, p2, p3 = points[i0], points[i1], points[i2], points[i3]

            a = -p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]
            b = 3 * p0[1] - 6 * p1[1] + 3 * p2[1]
            c = -3 * p0[1] + 3 * p1[1]
            d = p0[1] - pt_y

            ax = -p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]
            bx = 3 * p0[0] - 6 * p1[0] + 3 * p2[0]
            cx = -3 * p0[0] + 3 * p1[0]

            num_roots, roots = solve_cubic_py(a, b, c, d)
            for j in range(num_roots):
                t = roots[j]
                if 0 <= t <= 1:
                    x_at_t = ax * t**3 + bx * t**2 + cx * t + p0[0]
                    if x_at_t - pt_x > 0:
                        dy_dt = 3 * a * t**2 + 2 * b * t + c
                        if dy_dt > 0:
                            winding += 1
                        else:
                            winding -= 1
            point_idx += 3

    return winding
