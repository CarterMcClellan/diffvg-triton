"""
Alpha compositing and blending kernels.

Implements Porter-Duff compositing operations for combining
fragment colors in front-to-back or back-to-front order.

Ported from diffvg/diffvg.cpp (sample_color compositing section)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def blend_over(
    # Source color (premultiplied alpha)
    src_r, src_g, src_b, src_a,
    # Destination color (premultiplied alpha)
    dst_r, dst_g, dst_b, dst_a,
):
    """
    Standard Porter-Duff "over" operator.

    result = src + dst * (1 - src_alpha)

    Both src and dst are in premultiplied alpha format.
    Returns: (out_r, out_g, out_b, out_a)
    """
    one_minus_src_a = 1.0 - src_a

    out_r = src_r + dst_r * one_minus_src_a
    out_g = src_g + dst_g * one_minus_src_a
    out_b = src_b + dst_b * one_minus_src_a
    out_a = src_a + dst_a * one_minus_src_a

    return out_r, out_g, out_b, out_a


@triton.jit
def blend_over_straight(
    # Source color (straight alpha - not premultiplied)
    src_r, src_g, src_b, src_a,
    # Destination color (straight alpha)
    dst_r, dst_g, dst_b, dst_a,
):
    """
    "Over" operator for straight (non-premultiplied) alpha.

    out_alpha = src_a + dst_a * (1 - src_a)
    out_color = (src_color * src_a + dst_color * dst_a * (1 - src_a)) / out_alpha

    Returns: (out_r, out_g, out_b, out_a)
    """
    one_minus_src_a = 1.0 - src_a

    out_a = src_a + dst_a * one_minus_src_a

    # Avoid division by zero
    safe_out_a = tl.where(out_a > 1e-8, out_a, 1.0)

    # Numerator: src_color * src_a + dst_color * dst_a * (1 - src_a)
    num_r = src_r * src_a + dst_r * dst_a * one_minus_src_a
    num_g = src_g * src_a + dst_g * dst_a * one_minus_src_a
    num_b = src_b * src_a + dst_b * dst_a * one_minus_src_a

    out_r = num_r / safe_out_a
    out_g = num_g / safe_out_a
    out_b = num_b / safe_out_a

    # If out_a is near zero, color is undefined (use black)
    zero_alpha = out_a < 1e-8
    out_r = tl.where(zero_alpha, 0.0, out_r)
    out_g = tl.where(zero_alpha, 0.0, out_g)
    out_b = tl.where(zero_alpha, 0.0, out_b)

    return out_r, out_g, out_b, out_a


@triton.jit
def premultiply_alpha(r, g, b, a):
    """Convert straight alpha to premultiplied alpha."""
    return r * a, g * a, b * a, a


@triton.jit
def unpremultiply_alpha(r, g, b, a):
    """Convert premultiplied alpha to straight alpha."""
    safe_a = tl.where(a > 1e-8, a, 1.0)
    zero_alpha = a < 1e-8

    out_r = tl.where(zero_alpha, 0.0, r / safe_a)
    out_g = tl.where(zero_alpha, 0.0, g / safe_a)
    out_b = tl.where(zero_alpha, 0.0, b / safe_a)

    return out_r, out_g, out_b, a


@triton.jit
def smoothstep(edge0, edge1, x):
    """
    Hermite smoothstep interpolation.

    Returns smooth interpolation between 0 and 1 as x goes from edge0 to edge1.
    """
    # Clamp x to [edge0, edge1]
    t = (x - edge0) / (edge1 - edge0)
    t = tl.minimum(tl.maximum(t, 0.0), 1.0)

    # Hermite interpolation: 3t^2 - 2t^3
    return t * t * (3.0 - 2.0 * t)


@triton.jit
def smoothstep_coverage(signed_distance):
    """
    Convert signed distance to coverage using smoothstep.

    Coverage = 1 when inside (d < -1), 0 when outside (d > 1),
    smooth transition in between.

    This is used for anti-aliased rendering based on signed distance.
    """
    # Map signed distance to [0, 1] coverage
    # d = -1 -> coverage = 1
    # d = +1 -> coverage = 0
    t = (-signed_distance + 1.0) * 0.5
    t = tl.minimum(tl.maximum(t, 0.0), 1.0)

    # Smoothstep
    return t * t * (3.0 - 2.0 * t)


@triton.jit
def composite_fragments_kernel(
    # Fragment data (sorted back-to-front by group_id)
    frag_color_ptr,      # [max_frags, 4] RGBA colors
    frag_group_ptr,      # [max_frags] group IDs
    num_frags,           # Number of valid fragments
    # Background color
    bg_r, bg_g, bg_b, bg_a,
    # Output
    out_r_ptr, out_g_ptr, out_b_ptr, out_a_ptr,
    # Config
    max_frags: tl.constexpr,
):
    """
    Composite a list of fragments (back-to-front order).

    Uses straight alpha blending.
    """
    # Start with background
    acc_r = bg_r
    acc_g = bg_g
    acc_b = bg_b
    acc_a = bg_a

    # Composite each fragment over accumulated result
    for i in range(max_frags):
        valid = i < num_frags

        # Load fragment color
        frag_offset = i * 4
        fr = tl.load(frag_color_ptr + frag_offset, mask=valid, other=0.0)
        fg = tl.load(frag_color_ptr + frag_offset + 1, mask=valid, other=0.0)
        fb = tl.load(frag_color_ptr + frag_offset + 2, mask=valid, other=0.0)
        fa = tl.load(frag_color_ptr + frag_offset + 3, mask=valid, other=0.0)

        # Blend over accumulated color
        new_r, new_g, new_b, new_a = blend_over_straight(
            fr, fg, fb, fa,
            acc_r, acc_g, acc_b, acc_a
        )

        # Update accumulator (only if valid fragment)
        acc_r = tl.where(valid, new_r, acc_r)
        acc_g = tl.where(valid, new_g, acc_g)
        acc_b = tl.where(valid, new_b, acc_b)
        acc_a = tl.where(valid, new_a, acc_a)

    # Store result
    tl.store(out_r_ptr, acc_r)
    tl.store(out_g_ptr, acc_g)
    tl.store(out_b_ptr, acc_b)
    tl.store(out_a_ptr, acc_a)


@triton.jit
def composite_samples_kernel(
    # Per-pixel sample colors [H, W, num_samples, 4]
    sample_colors_ptr,
    # Output image [H, W, 4]
    output_ptr,
    # Background
    bg_r, bg_g, bg_b, bg_a,
    # Dimensions
    width: tl.constexpr,
    height: tl.constexpr,
    num_samples: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Average samples within each pixel.

    For multi-sample anti-aliasing, average all samples to get final pixel color.
    """
    pid = tl.program_id(0)
    pixel_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_pixels = width * height
    mask = pixel_idx < num_pixels

    # Accumulate all samples
    acc_r = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_g = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_a = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Base offset for this pixel's samples
    sample_base = pixel_idx * num_samples * 4

    for s in range(num_samples):
        sample_offset = sample_base + s * 4

        sr = tl.load(sample_colors_ptr + sample_offset, mask=mask, other=bg_r)
        sg = tl.load(sample_colors_ptr + sample_offset + 1, mask=mask, other=bg_g)
        sb = tl.load(sample_colors_ptr + sample_offset + 2, mask=mask, other=bg_b)
        sa = tl.load(sample_colors_ptr + sample_offset + 3, mask=mask, other=bg_a)

        acc_r += sr
        acc_g += sg
        acc_b += sb
        acc_a += sa

    # Average
    inv_samples = 1.0 / num_samples
    avg_r = acc_r * inv_samples
    avg_g = acc_g * inv_samples
    avg_b = acc_b * inv_samples
    avg_a = acc_a * inv_samples

    # Store to output
    out_base = pixel_idx * 4
    tl.store(output_ptr + out_base, avg_r, mask=mask)
    tl.store(output_ptr + out_base + 1, avg_g, mask=mask)
    tl.store(output_ptr + out_base + 2, avg_b, mask=mask)
    tl.store(output_ptr + out_base + 3, avg_a, mask=mask)


# Python reference implementations
def blend_over_py(src_rgba, dst_rgba):
    """Python reference for over blending (straight alpha)."""
    src_r, src_g, src_b, src_a = src_rgba
    dst_r, dst_g, dst_b, dst_a = dst_rgba

    one_minus_src_a = 1.0 - src_a
    out_a = src_a + dst_a * one_minus_src_a

    if out_a < 1e-8:
        return (0.0, 0.0, 0.0, 0.0)

    num_r = src_r * src_a + dst_r * dst_a * one_minus_src_a
    num_g = src_g * src_a + dst_g * dst_a * one_minus_src_a
    num_b = src_b * src_a + dst_b * dst_a * one_minus_src_a

    out_r = num_r / out_a
    out_g = num_g / out_a
    out_b = num_b / out_a

    return (out_r, out_g, out_b, out_a)


def composite_fragments_py(fragments, background=(1.0, 1.0, 1.0, 1.0)):
    """
    Python reference for fragment compositing.

    Args:
        fragments: List of (r, g, b, a) tuples, back-to-front order
        background: Background RGBA color

    Returns:
        Final RGBA color
    """
    result = background

    for frag in fragments:
        result = blend_over_py(frag, result)

    return result


def smoothstep_py(edge0, edge1, x):
    """Python reference for smoothstep."""
    t = (x - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def smoothstep_coverage_py(signed_distance):
    """Python reference for signed distance to coverage."""
    t = (-signed_distance + 1.0) * 0.5
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# Higher-level PyTorch functions
def composite_over(
    src: torch.Tensor,  # [*, 4] RGBA
    dst: torch.Tensor,  # [*, 4] RGBA
) -> torch.Tensor:
    """
    Composite src over dst using straight alpha blending.

    Args:
        src: Source RGBA tensor (straight alpha)
        dst: Destination RGBA tensor (straight alpha)

    Returns:
        Composited RGBA tensor
    """
    src_a = src[..., 3:4]
    dst_a = dst[..., 3:4]

    src_rgb = src[..., :3]
    dst_rgb = dst[..., :3]

    one_minus_src_a = 1.0 - src_a
    out_a = src_a + dst_a * one_minus_src_a

    # Numerator
    num_rgb = src_rgb * src_a + dst_rgb * dst_a * one_minus_src_a

    # Safe division
    safe_out_a = torch.where(out_a > 1e-8, out_a, torch.ones_like(out_a))
    out_rgb = torch.where(out_a > 1e-8, num_rgb / safe_out_a, torch.zeros_like(num_rgb))

    return torch.cat([out_rgb, out_a], dim=-1)


def composite_fragment_list(
    fragments: torch.Tensor,  # [N, 4] RGBA colors, back-to-front
    background: torch.Tensor = None,  # [4] background RGBA
) -> torch.Tensor:
    """
    Composite a list of fragments in back-to-front order.

    Args:
        fragments: [N, 4] tensor of RGBA colors
        background: [4] background color, defaults to white

    Returns:
        [4] final composited color
    """
    if background is None:
        background = torch.tensor([1.0, 1.0, 1.0, 1.0], device=fragments.device)

    result = background

    for i in range(fragments.shape[0]):
        result = composite_over(fragments[i:i+1], result.unsqueeze(0)).squeeze(0)

    return result
