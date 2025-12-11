"""
Tests for Triton kernel implementations.

Tests the low-level computational kernels against Python reference implementations.
"""

import pytest
import torch
import math


class TestSolve:
    """Tests for polynomial root finding."""

    def test_quadratic_two_roots(self):
        """Test quadratic solver with two real roots."""
        from diffvg_triton.kernels.solve import solve_quadratic_py

        # x^2 - 5x + 6 = 0 => roots at 2 and 3
        has_sol, t0, t1 = solve_quadratic_py(1.0, -5.0, 6.0)
        assert has_sol
        assert abs(t0 - 2.0) < 0.01 or abs(t0 - 3.0) < 0.01
        assert abs(t1 - 2.0) < 0.01 or abs(t1 - 3.0) < 0.01

    def test_quadratic_one_root(self):
        """Test quadratic solver with one root (discriminant = 0)."""
        from diffvg_triton.kernels.solve import solve_quadratic_py

        # x^2 - 2x + 1 = 0 => root at 1 (double)
        has_sol, t0, t1 = solve_quadratic_py(1.0, -2.0, 1.0)
        assert has_sol
        assert abs(t0 - 1.0) < 0.01
        assert abs(t1 - 1.0) < 0.01

    def test_quadratic_no_roots(self):
        """Test quadratic solver with no real roots."""
        from diffvg_triton.kernels.solve import solve_quadratic_py

        # x^2 + 1 = 0 => no real roots
        has_sol, _, _ = solve_quadratic_py(1.0, 0.0, 1.0)
        assert not has_sol

    def test_cubic_three_roots(self):
        """Test cubic solver with three real roots."""
        from diffvg_triton.kernels.solve import solve_cubic_py

        # x^3 - 6x^2 + 11x - 6 = 0 => roots at 1, 2, 3
        num_roots, roots = solve_cubic_py(1.0, -6.0, 11.0, -6.0)
        assert num_roots == 3

        roots_sorted = sorted(roots[:num_roots])
        assert abs(roots_sorted[0] - 1.0) < 0.1
        assert abs(roots_sorted[1] - 2.0) < 0.1
        assert abs(roots_sorted[2] - 3.0) < 0.1

    def test_cubic_one_root(self):
        """Test cubic solver with one real root."""
        from diffvg_triton.kernels.solve import solve_cubic_py

        # x^3 + x + 1 = 0 => one real root near -0.68
        num_roots, roots = solve_cubic_py(1.0, 0.0, 1.0, 1.0)
        assert num_roots == 1
        # Verify root by substitution
        r = roots[0]
        residual = r**3 + r + 1
        assert abs(residual) < 0.1


class TestDistance:
    """Tests for distance/closest point computation."""

    def test_closest_point_line_midpoint(self):
        """Test closest point on line segment at midpoint."""
        from diffvg_triton.kernels.distance import closest_point_line_py

        # Line from (0, 0) to (10, 0), query point at (5, 3)
        # Closest should be at (5, 0)
        closest, t, dist_sq = closest_point_line_py((5, 3), (0, 0), (10, 0))

        assert abs(closest[0] - 5.0) < 0.01
        assert abs(closest[1] - 0.0) < 0.01
        assert abs(t - 0.5) < 0.01
        assert abs(dist_sq - 9.0) < 0.01

    def test_closest_point_line_endpoint(self):
        """Test closest point clamped to endpoint."""
        from diffvg_triton.kernels.distance import closest_point_line_py

        # Line from (0, 0) to (10, 0), query point at (15, 0)
        # Closest should be at (10, 0)
        closest, t, dist_sq = closest_point_line_py((15, 0), (0, 0), (10, 0))

        assert abs(closest[0] - 10.0) < 0.01
        assert abs(closest[1] - 0.0) < 0.01
        assert abs(t - 1.0) < 0.01
        assert abs(dist_sq - 25.0) < 0.01

    def test_bezier_evaluation(self):
        """Test Bezier curve evaluation."""
        from diffvg_triton.kernels.distance import eval_quadratic_bezier_py, eval_cubic_bezier_py

        # Quadratic at t=0 should return p0
        p0, p1, p2 = (0, 0), (5, 10), (10, 0)
        result = eval_quadratic_bezier_py(0.0, p0, p1, p2)
        assert abs(result[0] - 0.0) < 0.01
        assert abs(result[1] - 0.0) < 0.01

        # Quadratic at t=1 should return p2
        result = eval_quadratic_bezier_py(1.0, p0, p1, p2)
        assert abs(result[0] - 10.0) < 0.01
        assert abs(result[1] - 0.0) < 0.01

        # Cubic at t=0.5 should be at midpoint (for symmetric control)
        p0, p1, p2, p3 = (0, 0), (0, 10), (10, 10), (10, 0)
        result = eval_cubic_bezier_py(0.5, p0, p1, p2, p3)
        assert abs(result[0] - 5.0) < 0.1


class TestWinding:
    """Tests for winding number computation."""

    def test_winding_inside_square(self):
        """Test winding number for point inside a square path."""
        from diffvg_triton.kernels.winding import compute_winding_number_path_py

        # Square from (0,0) to (10,10)
        # CCW: right, up, left, down
        segment_types = [0, 0, 0, 0]  # All lines
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]

        # Point inside
        winding = compute_winding_number_path_py((5, 5), segment_types, points, is_closed=True)
        assert winding != 0  # Non-zero winding means inside

    def test_winding_outside_square(self):
        """Test winding number for point outside a square path."""
        from diffvg_triton.kernels.winding import compute_winding_number_path_py

        segment_types = [0, 0, 0, 0]
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]

        # Point outside
        winding = compute_winding_number_path_py((15, 5), segment_types, points, is_closed=True)
        assert winding == 0


class TestComposite:
    """Tests for alpha compositing."""

    def test_blend_over_opaque(self):
        """Test over blending with opaque source."""
        from diffvg_triton.kernels.composite import blend_over_py

        src = (1.0, 0.0, 0.0, 1.0)  # Opaque red
        dst = (0.0, 1.0, 0.0, 1.0)  # Opaque green

        result = blend_over_py(src, dst)

        # Opaque red should completely cover
        assert abs(result[0] - 1.0) < 0.01
        assert abs(result[1] - 0.0) < 0.01
        assert abs(result[2] - 0.0) < 0.01
        assert abs(result[3] - 1.0) < 0.01

    def test_blend_over_transparent(self):
        """Test over blending with semi-transparent source."""
        from diffvg_triton.kernels.composite import blend_over_py

        src = (1.0, 0.0, 0.0, 0.5)  # 50% red
        dst = (0.0, 1.0, 0.0, 1.0)  # Opaque green

        result = blend_over_py(src, dst)

        # Should be roughly orange-ish (red over green)
        assert result[0] > 0.3  # Some red
        assert result[1] > 0.3  # Some green
        assert abs(result[3] - 1.0) < 0.01  # Full alpha

    def test_smoothstep_coverage(self):
        """Test smoothstep coverage function."""
        from diffvg_triton.kernels.composite import smoothstep_coverage_py

        # Inside (negative distance) -> high coverage
        assert smoothstep_coverage_py(-2.0) > 0.99

        # On boundary -> 0.5 coverage
        assert abs(smoothstep_coverage_py(0.0) - 0.5) < 0.01

        # Outside (positive distance) -> low coverage
        assert smoothstep_coverage_py(2.0) < 0.01


class TestFilter:
    """Tests for filter weight computation."""

    def test_box_filter_inside(self):
        """Test box filter weight inside radius."""
        from diffvg_triton.kernels.filter import compute_filter_weights_py, FilterType

        weight = compute_filter_weights_py(0.3, 0.3, 1.0, FilterType.BOX)
        assert abs(weight - 1.0) < 0.01

    def test_box_filter_outside(self):
        """Test box filter weight outside radius."""
        from diffvg_triton.kernels.filter import compute_filter_weights_py, FilterType

        weight = compute_filter_weights_py(1.5, 0.0, 1.0, FilterType.BOX)
        assert abs(weight - 0.0) < 0.01

    def test_tent_filter_center(self):
        """Test tent filter weight at center."""
        from diffvg_triton.kernels.filter import compute_filter_weights_py, FilterType

        # At center (0, 0), weight should be maximum
        weight = compute_filter_weights_py(0.0, 0.0, 1.0, FilterType.TENT)
        assert weight > 0.9

    def test_tent_filter_edge(self):
        """Test tent filter weight at edge."""
        from diffvg_triton.kernels.filter import compute_filter_weights_py, FilterType

        # At edge, weight should be near zero
        weight = compute_filter_weights_py(0.99, 0.0, 1.0, FilterType.TENT)
        assert weight < 0.1


class TestRNG:
    """Tests for random number generation."""

    def test_pcg32_deterministic(self):
        """Test that PCG32 produces deterministic output."""
        from diffvg_triton.kernels.rng import PCG32

        rng1 = PCG32(seed=42)
        rng2 = PCG32(seed=42)

        for _ in range(100):
            assert rng1.next_uint32() == rng2.next_uint32()

    def test_pcg32_uniform_range(self):
        """Test that uniform output is in [0, 1)."""
        from diffvg_triton.kernels.rng import PCG32

        rng = PCG32(seed=123)

        for _ in range(1000):
            val = rng.uniform()
            assert 0.0 <= val < 1.0

    def test_pcg32_different_seeds(self):
        """Test that different seeds produce different sequences."""
        from diffvg_triton.kernels.rng import PCG32

        rng1 = PCG32(seed=1)
        rng2 = PCG32(seed=2)

        # Very unlikely to match by chance
        matches = sum(
            1 for _ in range(100)
            if rng1.next_uint32() == rng2.next_uint32()
        )
        assert matches < 5  # Allow some coincidental matches


class TestBoundary:
    """Tests for boundary sampling."""

    def test_line_boundary_sample(self):
        """Test sampling on line segment boundary."""
        from diffvg_triton.kernels.boundary import sample_boundary_line_py

        p0 = (0, 0)
        p1 = (10, 0)

        # Sample at t=0.5
        pos, normal, length = sample_boundary_line_py(0.5, p0, p1)

        assert abs(pos[0] - 5.0) < 0.01
        assert abs(pos[1] - 0.0) < 0.01
        assert abs(length - 10.0) < 0.01
        # Normal should be perpendicular (pointing up for horizontal line)
        assert abs(normal[0] - 0.0) < 0.01
        assert abs(abs(normal[1]) - 1.0) < 0.01

    def test_path_length_cdf(self):
        """Test path length CDF computation."""
        from diffvg_triton.kernels.boundary import compute_path_length_cdf

        # Two equal-length line segments
        segment_types = [0, 0]
        points = [(0, 0), (10, 0), (20, 0)]

        cdf, total = compute_path_length_cdf(segment_types, points)

        assert abs(total - 20.0) < 0.1
        assert abs(cdf[0] - 0.5) < 0.01  # First segment is half
        assert abs(cdf[1] - 1.0) < 0.01  # Second segment completes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
