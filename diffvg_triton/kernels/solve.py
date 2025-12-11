"""
Polynomial root solvers for Triton kernels.

Provides quadratic and cubic equation solvers needed for ray-curve intersection
in winding number computation.

Ported from diffvg/solve.h
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def solve_quadratic_kernel(
    a, b, c,
    # Returns: (has_solution, t0, t1) where t0 <= t1
):
    """
    Solve quadratic equation: a*t^2 + b*t + c = 0

    Returns (has_solution, t0, t1) where has_solution indicates if real roots exist.
    Based on PBRT's numerically stable quadratic solver.
    """
    discrim = b * b - 4.0 * a * c

    # No real roots
    has_solution = discrim >= 0.0

    # Compute roots using numerically stable formula
    root_discrim = tl.sqrt(tl.maximum(discrim, 0.0))

    # Choose sign of q to avoid catastrophic cancellation
    q = tl.where(b < 0.0, -0.5 * (b - root_discrim), -0.5 * (b + root_discrim))

    # Avoid division by zero
    t0 = tl.where(tl.abs(a) > 1e-10, q / a, 0.0)
    t1 = tl.where(tl.abs(q) > 1e-10, c / q, 0.0)

    # Ensure t0 <= t1
    t0_final = tl.minimum(t0, t1)
    t1_final = tl.maximum(t0, t1)

    return has_solution, t0_final, t1_final


@triton.jit
def solve_cubic_kernel(
    a, b, c, d,
    # Returns: (num_roots, t0, t1, t2)
):
    """
    Solve cubic equation: a*t^3 + b*t^2 + c*t + d = 0

    Returns (num_roots, t0, t1, t2) using Cardano's formula.
    Uses double precision internally for numerical stability.
    """
    # Handle degenerate case (quadratic)
    is_quadratic = tl.abs(a) < 1e-6

    # For quadratic fallback
    quad_has_sol, quad_t0, quad_t1 = solve_quadratic_kernel(b, c, d)
    quad_num_roots = tl.where(quad_has_sol, 2, 0)

    # Normalize cubic equation: t^3 + (b/a)t^2 + (c/a)t + (d/a) = 0
    # Using double precision for intermediate calculations
    a_d = a.to(tl.float64)
    b_d = b.to(tl.float64)
    c_d = c.to(tl.float64)
    d_d = d.to(tl.float64)

    b_norm = b_d / a_d
    c_norm = c_d / a_d
    d_norm = d_d / a_d

    # Cardano's formula
    # Q = (b^2 - 3c) / 9
    # R = (2b^3 - 9bc + 27d) / 54
    Q = (b_norm * b_norm - 3.0 * c_norm) / 9.0
    R = (2.0 * b_norm * b_norm * b_norm - 9.0 * b_norm * c_norm + 27.0 * d_norm) / 54.0

    Q3 = Q * Q * Q
    R2 = R * R

    # Check discriminant for number of real roots
    has_three_roots = R2 < Q3

    # Case 1: Three real roots (R^2 < Q^3)
    # theta = acos(R / sqrt(Q^3))
    # roots = -2*sqrt(Q)*cos((theta + 2*pi*k)/3) - b/3 for k=0,1,2
    sqrt_Q = tl.sqrt(tl.maximum(Q, 0.0))
    sqrt_Q3 = tl.sqrt(tl.maximum(Q3, 1e-20))
    theta = tl.libdevice.acos(tl.minimum(tl.maximum(R / sqrt_Q3, -1.0), 1.0))

    pi = 3.14159265358979323846
    t0_three = -2.0 * sqrt_Q * tl.cos(theta / 3.0) - b_norm / 3.0
    t1_three = -2.0 * sqrt_Q * tl.cos((theta + 2.0 * pi) / 3.0) - b_norm / 3.0
    t2_three = -2.0 * sqrt_Q * tl.cos((theta - 2.0 * pi) / 3.0) - b_norm / 3.0

    # Case 2: One real root (R^2 >= Q^3)
    # A = -sign(R) * (|R| + sqrt(R^2 - Q^3))^(1/3)
    # B = Q / A if |A| > epsilon else 0
    # root = A + B - b/3
    sqrt_discrim = tl.sqrt(tl.maximum(R2 - Q3, 0.0))
    abs_R = tl.abs(R)

    # Compute A = -sign(R) * (|R| + sqrt(R^2 - Q^3))^(1/3)
    inner = abs_R + sqrt_discrim
    cbrt_inner = tl.libdevice.cbrt(inner)
    A = tl.where(R > 0, -cbrt_inner, cbrt_inner)

    # B = Q / A if |A| > epsilon
    B = tl.where(tl.abs(A) > 1e-6, Q / A, 0.0)

    t0_one = (A + B) - b_norm / 3.0

    # Select based on number of roots
    # For cubic case
    cubic_num_roots = tl.where(has_three_roots, 3, 1)
    cubic_t0 = tl.where(has_three_roots, t0_three, t0_one).to(tl.float32)
    cubic_t1 = tl.where(has_three_roots, t1_three, 0.0).to(tl.float32)
    cubic_t2 = tl.where(has_three_roots, t2_three, 0.0).to(tl.float32)

    # Final selection: quadratic vs cubic
    num_roots = tl.where(is_quadratic, quad_num_roots, cubic_num_roots)
    t0 = tl.where(is_quadratic, quad_t0, cubic_t0)
    t1 = tl.where(is_quadratic, quad_t1, cubic_t1)
    t2 = tl.where(is_quadratic, 0.0, cubic_t2)

    return num_roots, t0, t1, t2


# Python reference implementations for testing
def solve_quadratic_py(a: float, b: float, c: float):
    """Python reference implementation of quadratic solver."""
    discrim = b * b - 4 * a * c
    if discrim < 0:
        return False, 0.0, 0.0

    root_discrim = math.sqrt(discrim)
    if b < 0:
        q = -0.5 * (b - root_discrim)
    else:
        q = -0.5 * (b + root_discrim)

    if abs(a) < 1e-10:
        t0 = 0.0
    else:
        t0 = q / a

    if abs(q) < 1e-10:
        t1 = 0.0
    else:
        t1 = c / q

    if t0 > t1:
        t0, t1 = t1, t0

    return True, t0, t1


def solve_cubic_py(a: float, b: float, c: float, d: float):
    """Python reference implementation of cubic solver."""
    if abs(a) < 1e-6:
        has_sol, t0, t1 = solve_quadratic_py(b, c, d)
        if has_sol:
            return 2, [t0, t1, 0.0]
        else:
            return 0, [0.0, 0.0, 0.0]

    # Normalize
    b = b / a
    c = c / a
    d = d / a

    Q = (b * b - 3 * c) / 9.0
    R = (2 * b * b * b - 9 * b * c + 27 * d) / 54.0

    if R * R < Q * Q * Q:
        # 3 real roots
        theta = math.acos(R / math.sqrt(Q * Q * Q))
        sqrt_Q = math.sqrt(Q)
        t0 = -2.0 * sqrt_Q * math.cos(theta / 3.0) - b / 3.0
        t1 = -2.0 * sqrt_Q * math.cos((theta + 2 * math.pi) / 3.0) - b / 3.0
        t2 = -2.0 * sqrt_Q * math.cos((theta - 2 * math.pi) / 3.0) - b / 3.0
        return 3, [t0, t1, t2]
    else:
        # 1 real root
        sqrt_discrim = math.sqrt(R * R - Q * Q * Q)
        if R > 0:
            A = -math.pow(R + sqrt_discrim, 1.0 / 3.0)
        else:
            A = math.pow(-R + sqrt_discrim, 1.0 / 3.0)

        if abs(A) > 1e-6:
            B = Q / A
        else:
            B = 0.0

        t0 = (A + B) - b / 3.0
        return 1, [t0, 0.0, 0.0]
