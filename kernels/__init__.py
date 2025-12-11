"""
Triton kernels for differentiable vector graphics rendering.

This package contains the core computational kernels:
- solve: Polynomial root finding (quadratic, cubic)
- winding: Winding number computation for point-in-polygon tests
- distance: Closest point / signed distance field computation
- filter: Anti-aliasing filter weights
- composite: Alpha blending / Porter-Duff compositing
- boundary: Boundary sampling for gradient computation
- rng: PCG32 random number generation
"""

from .solve import (
    solve_quadratic_kernel,
    solve_cubic_kernel,
    solve_quadratic_py,
    solve_cubic_py,
)

from .winding import (
    ray_line_intersection,
    ray_quadratic_intersection,
    ray_cubic_intersection,
    compute_winding_number_path,
    compute_winding_number_path_py,
)

from .distance import (
    closest_point_line,
    closest_point_quadratic_bezier,
    closest_point_cubic_bezier,
    closest_point_circle,
    closest_point_rect,
    eval_quadratic_bezier,
    eval_cubic_bezier,
    eval_quadratic_bezier_deriv,
    eval_cubic_bezier_deriv,
    closest_point_line_py,
    closest_point_quadratic_bezier_py,
    closest_point_cubic_bezier_py,
    eval_quadratic_bezier_py,
    eval_cubic_bezier_py,
)

from .filter import (
    FilterType,
    compute_filter_weight,
    compute_filter_weight_box,
    compute_filter_weight_tent,
    compute_filter_weight_radial_parabolic,
    compute_filter_weight_hann,
    splat_samples_to_image,
    compute_filter_weights_py,
)

from .composite import (
    blend_over,
    blend_over_straight,
    premultiply_alpha,
    unpremultiply_alpha,
    smoothstep,
    smoothstep_coverage,
    composite_over,
    composite_fragment_list,
    blend_over_py,
    composite_fragments_py,
    smoothstep_py,
    smoothstep_coverage_py,
)

from .boundary import (
    sample_boundary_line,
    sample_boundary_quadratic,
    sample_boundary_cubic,
    sample_boundary_circle,
    compute_boundary_velocity_linear,
    compute_boundary_velocity_quadratic,
    compute_boundary_velocity_cubic,
    sample_boundary_line_py,
    sample_boundary_quadratic_py,
    sample_boundary_cubic_py,
    compute_path_length_cdf,
    sample_path_boundary,
)

from .rng import (
    pcg32_init,
    pcg32_next,
    pcg32_uniform,
    PCG32,
    TorchPCG32,
    generate_sample_offsets,
)


__all__ = [
    # Solve
    'solve_quadratic_kernel',
    'solve_cubic_kernel',
    'solve_quadratic_py',
    'solve_cubic_py',
    # Winding
    'ray_line_intersection',
    'ray_quadratic_intersection',
    'ray_cubic_intersection',
    'compute_winding_number_path',
    'compute_winding_number_path_py',
    # Distance
    'closest_point_line',
    'closest_point_quadratic_bezier',
    'closest_point_cubic_bezier',
    'closest_point_circle',
    'closest_point_rect',
    'eval_quadratic_bezier',
    'eval_cubic_bezier',
    'eval_quadratic_bezier_deriv',
    'eval_cubic_bezier_deriv',
    'closest_point_line_py',
    'closest_point_quadratic_bezier_py',
    'closest_point_cubic_bezier_py',
    'eval_quadratic_bezier_py',
    'eval_cubic_bezier_py',
    # Filter
    'FilterType',
    'compute_filter_weight',
    'compute_filter_weight_box',
    'compute_filter_weight_tent',
    'compute_filter_weight_radial_parabolic',
    'compute_filter_weight_hann',
    'splat_samples_to_image',
    'compute_filter_weights_py',
    # Composite
    'blend_over',
    'blend_over_straight',
    'premultiply_alpha',
    'unpremultiply_alpha',
    'smoothstep',
    'smoothstep_coverage',
    'composite_over',
    'composite_fragment_list',
    'blend_over_py',
    'composite_fragments_py',
    'smoothstep_py',
    'smoothstep_coverage_py',
    # Boundary
    'sample_boundary_line',
    'sample_boundary_quadratic',
    'sample_boundary_cubic',
    'sample_boundary_circle',
    'compute_boundary_velocity_linear',
    'compute_boundary_velocity_quadratic',
    'compute_boundary_velocity_cubic',
    'sample_boundary_line_py',
    'sample_boundary_quadratic_py',
    'sample_boundary_cubic_py',
    'compute_path_length_cdf',
    'sample_path_boundary',
    # RNG
    'pcg32_init',
    'pcg32_next',
    'pcg32_uniform',
    'PCG32',
    'TorchPCG32',
    'generate_sample_offsets',
]
