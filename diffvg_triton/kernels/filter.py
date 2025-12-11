"""
Anti-aliasing filter kernels for pixel splatting.

Implements filter weight computation for distributing sample colors
to neighboring pixels. Supports multiple filter types:
- Box filter
- Tent (bilinear) filter
- Radial parabolic filter
- Hann window filter

Ported from diffvg/filter.h
"""

import torch
import triton
import triton.language as tl
import math


# Filter type constants
class FilterType:
    BOX = 0
    TENT = 1
    RADIAL_PARABOLIC = 2
    HANN = 3


@triton.jit
def compute_filter_weight_box(
    dx, dy,      # Distance from sample to pixel center
    radius,      # Filter radius
):
    """
    Box filter: uniform weight within radius.

    Returns 1.0 if |dx| < radius and |dy| < radius, else 0.0
    """
    in_bounds = (tl.abs(dx) < radius) & (tl.abs(dy) < radius)
    return tl.where(in_bounds, 1.0, 0.0)


@triton.jit
def compute_filter_weight_tent(
    dx, dy,      # Distance from sample to pixel center
    radius,      # Filter radius
):
    """
    Tent (bilinear) filter: linear falloff from center.

    weight = (r - |dx|) * (r - |dy|) / r^4 if in bounds, else 0
    Normalized so integral over support = 1.
    """
    abs_dx = tl.abs(dx)
    abs_dy = tl.abs(dy)

    in_bounds = (abs_dx < radius) & (abs_dy < radius)

    # Linear falloff
    wx = radius - abs_dx
    wy = radius - abs_dy

    # Normalization factor: 1/r^4 makes integral = 1
    r4 = radius * radius * radius * radius
    weight = (wx * wy) / r4

    return tl.where(in_bounds, weight, 0.0)


@triton.jit
def compute_filter_weight_radial_parabolic(
    dx, dy,      # Distance from sample to pixel center
    radius,      # Filter radius
):
    """
    Radial parabolic filter: smooth falloff based on distance.

    weight = (4/3) * (1 - (d/r)^2)^2 if d < r, else 0
    where d = sqrt(dx^2 + dy^2)
    """
    d_sq = dx * dx + dy * dy
    r_sq = radius * radius

    in_bounds = d_sq < r_sq

    # Parabolic falloff
    ratio_sq = d_sq / r_sq
    falloff = 1.0 - ratio_sq
    weight = (4.0 / 3.0) * falloff * falloff / r_sq

    return tl.where(in_bounds, weight, 0.0)


@triton.jit
def compute_filter_weight_hann(
    dx, dy,      # Distance from sample to pixel center
    radius,      # Filter radius
):
    """
    Hann window filter: cosine-based smooth falloff.

    Separable: weight = hann(dx/r) * hann(dy/r) / r^2
    where hann(t) = 0.5 * (1 - cos(2*pi*t)) for t in [0, 1]

    Actually uses: 0.5 * (1 + cos(pi * t)) for t in [0, 1]
    which gives hann(0) = 1, hann(1) = 0
    """
    abs_dx = tl.abs(dx)
    abs_dy = tl.abs(dy)

    in_bounds = (abs_dx < radius) & (abs_dy < radius)

    pi = 3.14159265358979323846

    # Normalized distances
    tx = abs_dx / radius
    ty = abs_dy / radius

    # Hann window: 0.5 * (1 + cos(pi*t))
    hann_x = 0.5 * (1.0 + tl.cos(pi * tx))
    hann_y = 0.5 * (1.0 + tl.cos(pi * ty))

    # Separable weight
    weight = hann_x * hann_y / (radius * radius)

    return tl.where(in_bounds, weight, 0.0)


@triton.jit
def compute_filter_weight(
    dx, dy,      # Distance from sample to pixel center
    radius,      # Filter radius
    filter_type: tl.constexpr,  # Filter type constant
):
    """
    Compute filter weight for given distance and filter type.
    """
    if filter_type == 0:  # BOX
        return compute_filter_weight_box(dx, dy, radius)
    elif filter_type == 1:  # TENT
        return compute_filter_weight_tent(dx, dy, radius)
    elif filter_type == 2:  # RADIAL_PARABOLIC
        return compute_filter_weight_radial_parabolic(dx, dy, radius)
    else:  # HANN
        return compute_filter_weight_hann(dx, dy, radius)


@triton.jit
def splat_filter_kernel(
    # Sample data
    sample_x_ptr,        # [num_samples] sample x coordinates
    sample_y_ptr,        # [num_samples] sample y coordinates
    sample_color_ptr,    # [num_samples, 4] sample RGBA colors
    # Output image
    color_image_ptr,     # [H, W, 4] accumulated color
    weight_image_ptr,    # [H, W] accumulated weights
    # Image dimensions
    width: tl.constexpr,
    height: tl.constexpr,
    # Filter parameters
    filter_radius: tl.constexpr,
    filter_type: tl.constexpr,
    # Grid info
    num_samples: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Splat sample colors to nearby pixels using filter weights.

    Each sample contributes to a (2*radius+1)^2 neighborhood of pixels.
    """
    pid = tl.program_id(0)
    sample_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = sample_idx < num_samples

    # Load sample position
    sx = tl.load(sample_x_ptr + sample_idx, mask=mask, other=0.0)
    sy = tl.load(sample_y_ptr + sample_idx, mask=mask, other=0.0)

    # Load sample color (RGBA)
    color_offset = sample_idx * 4
    cr = tl.load(sample_color_ptr + color_offset, mask=mask, other=0.0)
    cg = tl.load(sample_color_ptr + color_offset + 1, mask=mask, other=0.0)
    cb = tl.load(sample_color_ptr + color_offset + 2, mask=mask, other=0.0)
    ca = tl.load(sample_color_ptr + color_offset + 3, mask=mask, other=0.0)

    # Determine pixel neighborhood
    # Sample coordinates are in [0, width] x [0, height] space
    # Pixel centers are at (px + 0.5, py + 0.5)

    # Find integer pixel range
    px_min = tl.maximum(tl.floor(sx - filter_radius).to(tl.int32), 0)
    px_max = tl.minimum(tl.floor(sx + filter_radius).to(tl.int32), width - 1)
    py_min = tl.maximum(tl.floor(sy - filter_radius).to(tl.int32), 0)
    py_max = tl.minimum(tl.floor(sy + filter_radius).to(tl.int32), height - 1)

    # Iterate over pixel neighborhood
    # Note: This is simplified - in practice we'd launch separate kernels
    # or use atomic operations for better parallelism
    radius_int = tl.ceil(filter_radius).to(tl.int32)

    for dy_int in range(-radius_int, radius_int + 1):
        for dx_int in range(-radius_int, radius_int + 1):
            # Pixel coordinates
            px = tl.floor(sx).to(tl.int32) + dx_int
            py = tl.floor(sy).to(tl.int32) + dy_int

            # Bounds check
            valid_pixel = (px >= 0) & (px < width) & (py >= 0) & (py < height) & mask

            # Distance from sample to pixel center
            dx = (px + 0.5) - sx
            dy = (py + 0.5) - sy

            # Compute filter weight
            weight = compute_filter_weight(dx, dy, filter_radius, filter_type)

            # Weighted color
            wr = weight * cr
            wg = weight * cg
            wb = weight * cb
            wa = weight * ca

            # Pixel index
            pixel_idx = py * width + px
            color_base = pixel_idx * 4

            # Atomic add to output (for thread safety)
            # Note: This is expensive but necessary for correctness
            tl.atomic_add(color_image_ptr + color_base, wr, mask=valid_pixel)
            tl.atomic_add(color_image_ptr + color_base + 1, wg, mask=valid_pixel)
            tl.atomic_add(color_image_ptr + color_base + 2, wb, mask=valid_pixel)
            tl.atomic_add(color_image_ptr + color_base + 3, wa, mask=valid_pixel)
            tl.atomic_add(weight_image_ptr + pixel_idx, weight, mask=valid_pixel)


@triton.jit
def normalize_by_weight_kernel(
    color_image_ptr,     # [H, W, 4] accumulated color (in/out)
    weight_image_ptr,    # [H, W] accumulated weights
    width: tl.constexpr,
    height: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalize accumulated colors by accumulated weights.

    final_color = accumulated_color / accumulated_weight
    """
    pid = tl.program_id(0)
    pixel_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_pixels = width * height
    mask = pixel_idx < num_pixels

    # Load weight
    weight = tl.load(weight_image_ptr + pixel_idx, mask=mask, other=1.0)

    # Avoid division by zero
    safe_weight = tl.where(weight > 1e-8, weight, 1.0)

    # Load and normalize color
    color_base = pixel_idx * 4

    r = tl.load(color_image_ptr + color_base, mask=mask, other=0.0)
    g = tl.load(color_image_ptr + color_base + 1, mask=mask, other=0.0)
    b = tl.load(color_image_ptr + color_base + 2, mask=mask, other=0.0)
    a = tl.load(color_image_ptr + color_base + 3, mask=mask, other=0.0)

    r_norm = r / safe_weight
    g_norm = g / safe_weight
    b_norm = b / safe_weight
    a_norm = a / safe_weight

    # Store normalized
    tl.store(color_image_ptr + color_base, r_norm, mask=mask)
    tl.store(color_image_ptr + color_base + 1, g_norm, mask=mask)
    tl.store(color_image_ptr + color_base + 2, b_norm, mask=mask)
    tl.store(color_image_ptr + color_base + 3, a_norm, mask=mask)


# Alternative: Per-pixel gather approach (more efficient for sparse samples)
@triton.jit
def gather_filter_kernel(
    # Sample data (sorted by pixel)
    sample_x_ptr,        # [num_samples]
    sample_y_ptr,        # [num_samples]
    sample_color_ptr,    # [num_samples, 4]
    # Output
    output_ptr,          # [H, W, 4]
    # Dimensions
    width: tl.constexpr,
    height: tl.constexpr,
    num_samples,
    # Filter params
    filter_radius,
    filter_type: tl.constexpr,
    # Block info
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """
    Per-pixel gather: each pixel gathers contributions from nearby samples.

    More cache-friendly for output image when samples are sparse.
    """
    # Pixel coordinates
    px = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    py = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)

    # Reshape to 2D grid
    px = px[:, None]  # [BLOCK_X, 1]
    py = py[None, :]  # [1, BLOCK_Y]

    # Pixel center in continuous coordinates
    pcx = px + 0.5
    pcy = py + 0.5

    # Accumulated color and weight
    acc_r = tl.zeros([BLOCK_X, BLOCK_Y], dtype=tl.float32)
    acc_g = tl.zeros([BLOCK_X, BLOCK_Y], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK_X, BLOCK_Y], dtype=tl.float32)
    acc_a = tl.zeros([BLOCK_X, BLOCK_Y], dtype=tl.float32)
    acc_w = tl.zeros([BLOCK_X, BLOCK_Y], dtype=tl.float32)

    # Iterate over all samples (could be optimized with spatial data structure)
    for s_idx in range(num_samples):
        # Load sample
        sx = tl.load(sample_x_ptr + s_idx)
        sy = tl.load(sample_y_ptr + s_idx)

        # Distance to pixel center
        dx = pcx - sx
        dy = pcy - sy

        # Filter weight
        weight = compute_filter_weight(dx, dy, filter_radius, filter_type)

        # Load sample color
        color_offset = s_idx * 4
        cr = tl.load(sample_color_ptr + color_offset)
        cg = tl.load(sample_color_ptr + color_offset + 1)
        cb = tl.load(sample_color_ptr + color_offset + 2)
        ca = tl.load(sample_color_ptr + color_offset + 3)

        # Accumulate
        acc_r += weight * cr
        acc_g += weight * cg
        acc_b += weight * cb
        acc_a += weight * ca
        acc_w += weight

    # Normalize
    safe_w = tl.where(acc_w > 1e-8, acc_w, 1.0)
    final_r = acc_r / safe_w
    final_g = acc_g / safe_w
    final_b = acc_b / safe_w
    final_a = acc_a / safe_w

    # Store output
    valid = (px < width) & (py < height)
    pixel_idx = py * width + px
    color_base = pixel_idx * 4

    tl.store(output_ptr + color_base, final_r, mask=valid)
    tl.store(output_ptr + color_base + 1, final_g, mask=valid)
    tl.store(output_ptr + color_base + 2, final_b, mask=valid)
    tl.store(output_ptr + color_base + 3, final_a, mask=valid)


# Python wrapper functions
def compute_filter_weights_py(dx: float, dy: float, radius: float, filter_type: int) -> float:
    """Python reference implementation for filter weight computation."""
    if filter_type == FilterType.BOX:
        if abs(dx) < radius and abs(dy) < radius:
            return 1.0
        return 0.0

    elif filter_type == FilterType.TENT:
        if abs(dx) < radius and abs(dy) < radius:
            wx = radius - abs(dx)
            wy = radius - abs(dy)
            return (wx * wy) / (radius ** 4)
        return 0.0

    elif filter_type == FilterType.RADIAL_PARABOLIC:
        d_sq = dx * dx + dy * dy
        r_sq = radius * radius
        if d_sq < r_sq:
            ratio_sq = d_sq / r_sq
            falloff = 1.0 - ratio_sq
            return (4.0 / 3.0) * falloff * falloff / r_sq
        return 0.0

    elif filter_type == FilterType.HANN:
        if abs(dx) < radius and abs(dy) < radius:
            tx = abs(dx) / radius
            ty = abs(dy) / radius
            hann_x = 0.5 * (1.0 + math.cos(math.pi * tx))
            hann_y = 0.5 * (1.0 + math.cos(math.pi * ty))
            return hann_x * hann_y / (radius * radius)
        return 0.0

    return 0.0


def splat_samples_to_image(
    sample_positions: torch.Tensor,  # [N, 2]
    sample_colors: torch.Tensor,     # [N, 4]
    width: int,
    height: int,
    filter_radius: float = 0.5,
    filter_type: int = FilterType.BOX,
) -> torch.Tensor:
    """
    Splat sample colors to image using filter kernel.

    Args:
        sample_positions: [N, 2] tensor of (x, y) sample positions
        sample_colors: [N, 4] tensor of RGBA colors
        width: Output image width
        height: Output image height
        filter_radius: Filter support radius
        filter_type: Type of filter (BOX, TENT, etc.)

    Returns:
        [H, W, 4] output image
    """
    device = sample_positions.device
    num_samples = sample_positions.shape[0]

    # Allocate output
    color_image = torch.zeros((height, width, 4), dtype=torch.float32, device=device)
    weight_image = torch.zeros((height, width), dtype=torch.float32, device=device)

    if num_samples == 0:
        return color_image

    # Flatten for kernel
    sample_x = sample_positions[:, 0].contiguous()
    sample_y = sample_positions[:, 1].contiguous()
    sample_colors = sample_colors.contiguous()

    # Simple CPU fallback for now (Triton atomic splatting is complex)
    # TODO: Implement efficient GPU splatting with tiling
    for i in range(num_samples):
        sx = sample_x[i].item()
        sy = sample_y[i].item()

        # Pixel range
        px_min = max(0, int(sx - filter_radius))
        px_max = min(width - 1, int(sx + filter_radius))
        py_min = max(0, int(sy - filter_radius))
        py_max = min(height - 1, int(sy + filter_radius))

        for py in range(py_min, py_max + 1):
            for px in range(px_min, px_max + 1):
                dx = (px + 0.5) - sx
                dy = (py + 0.5) - sy
                weight = compute_filter_weights_py(dx, dy, filter_radius, filter_type)

                if weight > 0:
                    color_image[py, px] += weight * sample_colors[i]
                    weight_image[py, px] += weight

    # Normalize
    weight_image = weight_image.unsqueeze(-1)
    mask = weight_image > 1e-8
    color_image = torch.where(mask, color_image / weight_image, color_image)

    return color_image
