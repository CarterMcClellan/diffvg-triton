"""
Boundary sampling for gradient computation.

Implements Reynolds transport theorem based gradient computation
by sampling points on shape boundaries and accumulating contributions.

Ported from diffvg/sample_boundary.h and boundary gradient sections.
"""

import torch
import triton
import triton.language as tl
import math

from .distance import (
    eval_quadratic_bezier, eval_quadratic_bezier_deriv,
    eval_cubic_bezier, eval_cubic_bezier_deriv,
)


@triton.jit
def sample_boundary_line(
    t,           # Parameter in [0, 1]
    p0_x, p0_y,  # Start point
    p1_x, p1_y,  # End point
):
    """
    Sample point and normal on line segment boundary.

    Returns: (pos_x, pos_y, normal_x, normal_y, length)
    """
    # Position: linear interpolation
    pos_x = p0_x + t * (p1_x - p0_x)
    pos_y = p0_y + t * (p1_y - p0_y)

    # Tangent
    dx = p1_x - p0_x
    dy = p1_y - p0_y

    # Length
    length = tl.sqrt(dx * dx + dy * dy)
    safe_length = tl.where(length > 1e-10, length, 1.0)

    # Normal (perpendicular to tangent, normalized)
    # Rotate tangent 90 degrees: (dx, dy) -> (-dy, dx)
    normal_x = -dy / safe_length
    normal_y = dx / safe_length

    # Handle degenerate line
    normal_x = tl.where(length > 1e-10, normal_x, 0.0)
    normal_y = tl.where(length > 1e-10, normal_y, 1.0)

    return pos_x, pos_y, normal_x, normal_y, length


@triton.jit
def sample_boundary_quadratic(
    t,                    # Parameter in [0, 1]
    p0_x, p0_y,           # Start point
    p1_x, p1_y,           # Control point
    p2_x, p2_y,           # End point
):
    """
    Sample point and normal on quadratic Bezier boundary.

    Returns: (pos_x, pos_y, normal_x, normal_y, arc_length_approx)
    """
    # Position
    pos_x, pos_y = eval_quadratic_bezier(t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y)

    # Derivative (tangent direction)
    dx, dy = eval_quadratic_bezier_deriv(t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y)

    # Tangent magnitude
    tangent_len = tl.sqrt(dx * dx + dy * dy)
    safe_len = tl.where(tangent_len > 1e-10, tangent_len, 1.0)

    # Normal (perpendicular to tangent)
    normal_x = -dy / safe_len
    normal_y = dx / safe_len

    normal_x = tl.where(tangent_len > 1e-10, normal_x, 0.0)
    normal_y = tl.where(tangent_len > 1e-10, normal_y, 1.0)

    # Approximate arc length using chord length
    # More accurate would be Gaussian quadrature
    chord_x = p2_x - p0_x
    chord_y = p2_y - p0_y
    arc_length = tl.sqrt(chord_x * chord_x + chord_y * chord_y) * 1.2  # Rough approximation

    return pos_x, pos_y, normal_x, normal_y, arc_length


@triton.jit
def sample_boundary_cubic(
    t,                    # Parameter in [0, 1]
    p0_x, p0_y,           # Start point
    p1_x, p1_y,           # Control point 1
    p2_x, p2_y,           # Control point 2
    p3_x, p3_y,           # End point
):
    """
    Sample point and normal on cubic Bezier boundary.

    Returns: (pos_x, pos_y, normal_x, normal_y, arc_length_approx)
    """
    # Position
    pos_x, pos_y = eval_cubic_bezier(t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y)

    # Derivative (tangent direction)
    dx, dy = eval_cubic_bezier_deriv(t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y)

    # Tangent magnitude
    tangent_len = tl.sqrt(dx * dx + dy * dy)
    safe_len = tl.where(tangent_len > 1e-10, tangent_len, 1.0)

    # Normal (perpendicular to tangent)
    normal_x = -dy / safe_len
    normal_y = dx / safe_len

    normal_x = tl.where(tangent_len > 1e-10, normal_x, 0.0)
    normal_y = tl.where(tangent_len > 1e-10, normal_y, 1.0)

    # Approximate arc length
    chord_x = p3_x - p0_x
    chord_y = p3_y - p0_y
    arc_length = tl.sqrt(chord_x * chord_x + chord_y * chord_y) * 1.3

    return pos_x, pos_y, normal_x, normal_y, arc_length


@triton.jit
def sample_boundary_circle(
    t,           # Parameter in [0, 1] (fraction of circumference)
    center_x, center_y,
    radius,
):
    """
    Sample point and normal on circle boundary.

    Returns: (pos_x, pos_y, normal_x, normal_y, circumference)
    """
    # Angle
    two_pi = 6.28318530717958647692
    theta = t * two_pi

    # Position
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)

    pos_x = center_x + radius * cos_theta
    pos_y = center_y + radius * sin_theta

    # Normal (pointing outward)
    normal_x = cos_theta
    normal_y = sin_theta

    # Circumference
    circumference = two_pi * radius

    return pos_x, pos_y, normal_x, normal_y, circumference


@triton.jit
def compute_boundary_velocity_linear(
    t,           # Parameter on segment
    # Which point to compute velocity for (0 = p0, 1 = p1)
    point_idx: tl.constexpr,
):
    """
    Compute velocity field for linear segment control point.

    The velocity field represents how moving a control point
    affects the position on the boundary at parameter t.

    For linear: boundary_pos = (1-t)*p0 + t*p1
    d(boundary_pos)/d(p0) = (1-t) * I
    d(boundary_pos)/d(p1) = t * I

    Returns: velocity_x, velocity_y (per unit displacement of control point)
    """
    if point_idx == 0:
        # d/dp0
        vel = 1.0 - t
    else:
        # d/dp1
        vel = t

    return vel, vel  # Same for x and y (diagonal Jacobian)


@triton.jit
def compute_boundary_velocity_quadratic(
    t,           # Parameter on curve
    # Which point to compute velocity for (0, 1, or 2)
    point_idx: tl.constexpr,
):
    """
    Compute velocity field for quadratic Bezier control point.

    Quadratic: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2

    Returns: velocity magnitude (same for x and y components)
    """
    one_minus_t = 1.0 - t

    if point_idx == 0:
        vel = one_minus_t * one_minus_t
    elif point_idx == 1:
        vel = 2.0 * one_minus_t * t
    else:
        vel = t * t

    return vel, vel


@triton.jit
def compute_boundary_velocity_cubic(
    t,           # Parameter on curve
    # Which point to compute velocity for (0, 1, 2, or 3)
    point_idx: tl.constexpr,
):
    """
    Compute velocity field for cubic Bezier control point.

    Cubic: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3

    Returns: velocity magnitude (same for x and y components)
    """
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    t_sq = t * t

    if point_idx == 0:
        vel = one_minus_t_sq * one_minus_t
    elif point_idx == 1:
        vel = 3.0 * one_minus_t_sq * t
    elif point_idx == 2:
        vel = 3.0 * one_minus_t * t_sq
    else:
        vel = t_sq * t

    return vel, vel


@triton.jit
def accumulate_boundary_gradient_kernel(
    # Boundary samples
    boundary_pos_ptr,        # [num_samples, 2] boundary positions
    boundary_normal_ptr,     # [num_samples, 2] boundary normals
    boundary_t_ptr,          # [num_samples] parameter t
    boundary_segment_ptr,    # [num_samples] segment index
    boundary_path_ptr,       # [num_samples] path index
    # Color difference at boundary
    color_diff_ptr,          # [num_samples, 4] (inside - outside) color
    # PDF for importance sampling
    pdf_ptr,                 # [num_samples] sampling PDF
    # Path control points
    points_ptr,              # [total_points, 2]
    point_offsets_ptr,       # [num_paths] start index per path
    segment_types_ptr,       # [num_paths, max_segments]
    # Output gradients
    d_points_ptr,            # [total_points, 2] gradient output
    # Dimensions
    num_samples: tl.constexpr,
    num_paths: tl.constexpr,
    max_segments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Accumulate boundary gradients using Reynolds transport theorem.

    For each boundary sample:
      gradient += (color_inside - color_outside) * velocity · normal / pdf

    This kernel computes gradients for path control points.
    """
    pid = tl.program_id(0)
    sample_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = sample_idx < num_samples

    # Load boundary sample data
    pos_x = tl.load(boundary_pos_ptr + sample_idx * 2, mask=mask, other=0.0)
    pos_y = tl.load(boundary_pos_ptr + sample_idx * 2 + 1, mask=mask, other=0.0)
    normal_x = tl.load(boundary_normal_ptr + sample_idx * 2, mask=mask, other=0.0)
    normal_y = tl.load(boundary_normal_ptr + sample_idx * 2 + 1, mask=mask, other=0.0)
    t = tl.load(boundary_t_ptr + sample_idx, mask=mask, other=0.0)
    segment_idx = tl.load(boundary_segment_ptr + sample_idx, mask=mask, other=0)
    path_idx = tl.load(boundary_path_ptr + sample_idx, mask=mask, other=0)

    # Load color difference
    cdiff_r = tl.load(color_diff_ptr + sample_idx * 4, mask=mask, other=0.0)
    cdiff_g = tl.load(color_diff_ptr + sample_idx * 4 + 1, mask=mask, other=0.0)
    cdiff_b = tl.load(color_diff_ptr + sample_idx * 4 + 2, mask=mask, other=0.0)
    cdiff_a = tl.load(color_diff_ptr + sample_idx * 4 + 3, mask=mask, other=0.0)

    # Total color difference magnitude (for weighting)
    color_weight = cdiff_r + cdiff_g + cdiff_b

    # Load PDF
    pdf = tl.load(pdf_ptr + sample_idx, mask=mask, other=1.0)
    safe_pdf = tl.where(pdf > 1e-10, pdf, 1.0)

    # Contribution weight
    weight = color_weight / safe_pdf

    # Get point offset for this path
    point_offset = tl.load(point_offsets_ptr + path_idx, mask=mask, other=0)

    # Get segment type
    seg_type = tl.load(segment_types_ptr + path_idx * max_segments + segment_idx, mask=mask, other=0)

    # Compute velocity and accumulate gradient based on segment type
    # This is simplified - full implementation would iterate over all control points

    # For now, compute gradient contribution for the segment's control points
    # The velocity · normal gives the Jacobian for Reynolds transport

    # Velocity for t on this segment
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    t_sq = t * t

    # Linear: 2 points
    # Quadratic: 3 points
    # Cubic: 4 points
    is_line = seg_type == 0
    is_quad = seg_type == 1
    # is_cubic = seg_type == 2

    # Compute velocities for each control point type
    # For linear segment
    vel_p0_linear = one_minus_t
    vel_p1_linear = t

    # For quadratic
    vel_p0_quad = one_minus_t_sq
    vel_p1_quad = 2.0 * one_minus_t * t
    vel_p2_quad = t_sq

    # For cubic
    vel_p0_cubic = one_minus_t_sq * one_minus_t
    vel_p1_cubic = 3.0 * one_minus_t_sq * t
    vel_p2_cubic = 3.0 * one_minus_t * t_sq
    vel_p3_cubic = t_sq * t

    # Select based on segment type
    # Point 0 velocity
    vel_p0 = tl.where(is_line, vel_p0_linear, tl.where(is_quad, vel_p0_quad, vel_p0_cubic))

    # Gradient contribution: weight * velocity * normal
    # This gives d(color)/d(point_x) and d(color)/d(point_y)
    grad_p0_x = weight * vel_p0 * normal_x
    grad_p0_y = weight * vel_p0 * normal_y

    # Atomic add to gradient buffer
    # Point index depends on segment type and offset
    # This is simplified - real implementation needs proper point indexing
    tl.atomic_add(d_points_ptr + point_offset * 2, grad_p0_x, mask=mask)
    tl.atomic_add(d_points_ptr + point_offset * 2 + 1, grad_p0_y, mask=mask)


# Python reference implementations
def sample_boundary_line_py(t, p0, p1):
    """Python reference for line boundary sampling."""
    pos = (
        p0[0] + t * (p1[0] - p0[0]),
        p0[1] + t * (p1[1] - p0[1])
    )

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1e-10:
        return pos, (0.0, 1.0), 0.0

    normal = (-dy / length, dx / length)
    return pos, normal, length


def sample_boundary_quadratic_py(t, p0, p1, p2):
    """Python reference for quadratic Bezier boundary sampling."""
    from .distance import eval_quadratic_bezier_py

    pos = eval_quadratic_bezier_py(t, p0, p1, p2)

    # Derivative
    one_minus_t = 1.0 - t
    dx = 2.0 * (one_minus_t * (p1[0] - p0[0]) + t * (p2[0] - p1[0]))
    dy = 2.0 * (one_minus_t * (p1[1] - p0[1]) + t * (p2[1] - p1[1]))

    tangent_len = math.sqrt(dx * dx + dy * dy)
    if tangent_len < 1e-10:
        return pos, (0.0, 1.0), 0.0

    normal = (-dy / tangent_len, dx / tangent_len)

    # Approximate arc length
    chord = math.sqrt((p2[0] - p0[0])**2 + (p2[1] - p0[1])**2)
    arc_length = chord * 1.2

    return pos, normal, arc_length


def sample_boundary_cubic_py(t, p0, p1, p2, p3):
    """Python reference for cubic Bezier boundary sampling."""
    from .distance import eval_cubic_bezier_py

    pos = eval_cubic_bezier_py(t, p0, p1, p2, p3)

    # Derivative
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    t_sq = t * t

    dx = (3.0 * one_minus_t_sq * (p1[0] - p0[0]) +
          6.0 * one_minus_t * t * (p2[0] - p1[0]) +
          3.0 * t_sq * (p3[0] - p2[0]))
    dy = (3.0 * one_minus_t_sq * (p1[1] - p0[1]) +
          6.0 * one_minus_t * t * (p2[1] - p1[1]) +
          3.0 * t_sq * (p3[1] - p2[1]))

    tangent_len = math.sqrt(dx * dx + dy * dy)
    if tangent_len < 1e-10:
        return pos, (0.0, 1.0), 0.0

    normal = (-dy / tangent_len, dx / tangent_len)

    chord = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)
    arc_length = chord * 1.3

    return pos, normal, arc_length


def compute_path_length_cdf(segment_types, points):
    """
    Compute cumulative distribution function for sampling path boundary.

    Args:
        segment_types: List of segment types (0=line, 1=quad, 2=cubic)
        points: List of (x, y) control points

    Returns:
        (cdf, total_length) where cdf[i] is cumulative length up to segment i
    """
    lengths = []
    point_idx = 0

    for seg_type in segment_types:
        if seg_type == 0:
            # Line
            p0 = points[point_idx]
            p1 = points[point_idx + 1]
            length = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
            point_idx += 1
        elif seg_type == 1:
            # Quadratic - approximate
            p0 = points[point_idx]
            p2 = points[point_idx + 2]
            length = math.sqrt((p2[0] - p0[0])**2 + (p2[1] - p0[1])**2) * 1.2
            point_idx += 2
        else:
            # Cubic - approximate
            p0 = points[point_idx]
            p3 = points[point_idx + 3]
            length = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2) * 1.3
            point_idx += 3

        lengths.append(length)

    # Compute CDF
    total = sum(lengths)
    if total < 1e-10:
        return [1.0] * len(lengths), 1.0

    cdf = []
    cumsum = 0.0
    for length in lengths:
        cumsum += length / total
        cdf.append(cumsum)

    return cdf, total


def sample_path_boundary(
    u: float,  # Uniform random in [0, 1)
    segment_types: list,
    points: list,
    is_closed: bool = True,
):
    """
    Sample a point on path boundary proportional to arc length.

    Args:
        u: Uniform random value in [0, 1)
        segment_types: List of segment types
        points: List of control points
        is_closed: Whether path is closed

    Returns:
        (position, normal, segment_idx, t, pdf)
    """
    cdf, total_length = compute_path_length_cdf(segment_types, points)

    # Find segment
    segment_idx = 0
    for i, c in enumerate(cdf):
        if u < c:
            segment_idx = i
            break

    # Compute t within segment
    prev_cdf = cdf[segment_idx - 1] if segment_idx > 0 else 0.0
    segment_prob = cdf[segment_idx] - prev_cdf
    t = (u - prev_cdf) / segment_prob if segment_prob > 0 else 0.0

    # Get segment control points
    point_idx = 0
    for i in range(segment_idx):
        if segment_types[i] == 0:
            point_idx += 1
        elif segment_types[i] == 1:
            point_idx += 2
        else:
            point_idx += 3

    seg_type = segment_types[segment_idx]

    if seg_type == 0:
        p0, p1 = points[point_idx], points[point_idx + 1]
        pos, normal, length = sample_boundary_line_py(t, p0, p1)
    elif seg_type == 1:
        p0, p1, p2 = points[point_idx], points[point_idx + 1], points[point_idx + 2]
        pos, normal, length = sample_boundary_quadratic_py(t, p0, p1, p2)
    else:
        p0, p1, p2, p3 = points[point_idx:point_idx + 4]
        pos, normal, length = sample_boundary_cubic_py(t, p0, p1, p2, p3)

    # PDF = 1 / total_length (uniform sampling by arc length)
    pdf = 1.0 / total_length if total_length > 0 else 1.0

    return pos, normal, segment_idx, t, pdf
