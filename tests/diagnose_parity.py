#!/usr/bin/env python3
"""
Diagnostic script to find divergence between pydiffvg and diffvg-triton.

This creates identical scenes and compares rendering at multiple levels:
1. Scene flattening / serialization
2. Distance computations
3. Winding number computations
4. Final rendered output

Run in diffvg container with diffvg-triton mounted at /diffvg-triton:
    docker exec diffvg python /diffvg-triton/tests/diagnose_parity.py
"""

import sys
import os
import json
import numpy as np
import torch

# Add diffvg-triton to path
TRITON_PATH = '/workspace' if os.path.exists('/workspace/diffvg_triton') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TRITON_PATH not in sys.path:
    sys.path.insert(0, TRITON_PATH)


def log(msg, level="INFO"):
    print(f"[{level}] {msg}")


def create_simple_stroke_scene_pydiffvg():
    """Create a simple stroked cubic bezier using pydiffvg."""
    import pydiffvg

    pydiffvg.set_use_gpu(torch.cuda.is_available())

    # Simple cubic bezier stroke
    points = torch.tensor([
        [5.0, 14.0],   # Start
        [9.0, 5.0],    # Control 1
        [19.0, 5.0],   # Control 2
        [23.0, 14.0],  # End
    ], dtype=torch.float32)

    num_control_points = torch.tensor([2], dtype=torch.int32)  # 2 = cubic

    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=points,
        is_closed=False,
        stroke_width=torch.tensor(2.0)
    )

    group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=None,
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0])  # Black stroke
    )

    return [path], [group], points


def create_simple_stroke_scene_triton(points_tensor):
    """Create equivalent scene for diffvg-triton using same points."""

    class MockPath:
        def __init__(self, points, num_control_points, is_closed, stroke_width):
            self.points = points
            self.num_control_points = num_control_points
            self.is_closed = is_closed
            self.stroke_width = stroke_width
            self.thickness = None

    class MockShapeGroup:
        def __init__(self, shape_ids, fill_color, stroke_color):
            self.shape_ids = shape_ids
            self.fill_color = fill_color
            self.stroke_color = stroke_color
            self.use_even_odd_rule = True
            self.shape_to_canvas = None

    path = MockPath(
        points=points_tensor.clone(),
        num_control_points=torch.tensor([2], dtype=torch.int32),
        is_closed=False,
        stroke_width=torch.tensor([2.0])
    )

    group = MockShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32),
        fill_color=None,
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0])
    )

    return [path], [group]


def render_pydiffvg(shapes, groups, width, height, num_samples=2):
    """Render using pydiffvg."""
    import pydiffvg

    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, groups)

    render = pydiffvg.RenderFunction.apply
    img = render(
        width, height,
        num_samples, num_samples,
        0,  # seed
        None,  # background (white by default)
        *scene_args
    )

    return img.detach().cpu()


def render_triton(shapes, groups, width, height, num_samples=2):
    """Render using diffvg-triton."""
    from diffvg_triton.scene import flatten_scene
    from diffvg_triton.render import render_scene_py, RenderConfig

    device = torch.device('cpu')  # Use CPU for easier debugging
    scene = flatten_scene(width, height, shapes, groups, device=device)

    config = RenderConfig(
        num_samples_x=num_samples,
        num_samples_y=num_samples,
        background_color=(1.0, 1.0, 1.0, 1.0),
    )

    img = render_scene_py(scene, config)
    return img


def compare_pixel_values(img1, img2, name1="img1", name2="img2"):
    """Detailed pixel-by-pixel comparison."""
    log(f"\n{'='*60}")
    log(f"Comparing {name1} vs {name2}")
    log(f"{'='*60}")

    log(f"{name1} shape: {img1.shape}, dtype: {img1.dtype}")
    log(f"{name2} shape: {img2.shape}, dtype: {img2.dtype}")
    log(f"{name1} range: [{img1.min():.4f}, {img1.max():.4f}]")
    log(f"{name2} range: [{img2.min():.4f}, {img2.max():.4f}]")

    # Make sure shapes match
    if img1.shape != img2.shape:
        log(f"Shape mismatch! {img1.shape} vs {img2.shape}", "ERROR")
        return

    diff = torch.abs(img1 - img2)

    log(f"\nDifference statistics:")
    log(f"  Max diff: {diff.max().item():.6f}")
    log(f"  Mean diff: {diff.mean().item():.6f}")
    log(f"  Std diff: {diff.std().item():.6f}")

    # Find pixels with significant differences
    threshold = 0.01
    significant = diff > threshold
    num_significant = significant.sum().item()
    total = diff.numel()

    log(f"  Pixels with diff > {threshold}: {num_significant}/{total} ({100*num_significant/total:.2f}%)")

    if num_significant > 0:
        # Find locations of max differences
        max_idx = diff.argmax().item()
        h, w, c = img1.shape
        py = max_idx // (w * c)
        px = (max_idx % (w * c)) // c
        ch = max_idx % c

        log(f"\n  Max diff location: pixel ({px}, {py}), channel {ch}")
        log(f"    {name1}[{py},{px}]: {img1[py, px].tolist()}")
        log(f"    {name2}[{py},{px}]: {img2[py, px].tolist()}")

        # Sample a few more differing pixels
        log(f"\n  Sample of differing pixels (first 5):")
        diff_locs = torch.where(significant)
        for i in range(min(5, len(diff_locs[0]))):
            y, x, c = diff_locs[0][i].item(), diff_locs[1][i].item(), diff_locs[2][i].item()
            log(f"    [{y},{x},{c}]: {name1}={img1[y,x,c].item():.4f}, {name2}={img2[y,x,c].item():.4f}, diff={diff[y,x,c].item():.4f}")


def diagnose_distance_computation():
    """Test distance computation for a specific point."""
    log("\n" + "="*60)
    log("DIAGNOSING DISTANCE COMPUTATION")
    log("="*60)

    # Cubic bezier points
    p0 = (5.0, 14.0)
    p1 = (9.0, 5.0)
    p2 = (19.0, 5.0)
    p3 = (23.0, 14.0)

    # Test point (somewhere near the curve)
    test_pt = (14.0, 8.0)

    log(f"\nCubic bezier: P0={p0}, P1={p1}, P2={p2}, P3={p3}")
    log(f"Test point: {test_pt}")

    # Compute distance using diffvg-triton
    from diffvg_triton.kernels.distance import closest_point_cubic_bezier_py

    closest_triton, t_triton, dist_sq_triton = closest_point_cubic_bezier_py(test_pt, p0, p1, p2, p3)
    dist_triton = dist_sq_triton ** 0.5

    log(f"\ndiffvg-triton result:")
    log(f"  Closest point: {closest_triton}")
    log(f"  t parameter: {t_triton:.6f}")
    log(f"  Distance: {dist_triton:.6f}")

    # Try to compute reference distance by sampling
    log(f"\nReference (dense sampling):")
    min_dist = float('inf')
    best_t = 0
    best_pt = p0
    for i in range(1001):
        t = i / 1000.0
        # Cubic bezier formula
        mt = 1 - t
        x = mt**3 * p0[0] + 3*mt**2*t * p1[0] + 3*mt*t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3*mt**2*t * p1[1] + 3*mt*t**2 * p2[1] + t**3 * p3[1]

        d = ((x - test_pt[0])**2 + (y - test_pt[1])**2) ** 0.5
        if d < min_dist:
            min_dist = d
            best_t = t
            best_pt = (x, y)

    log(f"  Closest point: {best_pt}")
    log(f"  t parameter: {best_t:.6f}")
    log(f"  Distance: {min_dist:.6f}")

    log(f"\nDifference: {abs(dist_triton - min_dist):.6f}")


def diagnose_sample_positions():
    """Verify sample positions match between implementations."""
    log("\n" + "="*60)
    log("DIAGNOSING SAMPLE POSITIONS")
    log("="*60)

    width, height = 4, 4
    num_samples = 2

    log(f"Canvas: {width}x{height}, samples per axis: {num_samples}")

    # diffvg-triton sample positions (stratified)
    log("\ndiffvg-triton sample positions for pixel (0,0):")
    for sy in range(num_samples):
        for sx in range(num_samples):
            ox = (sx + 0.5) / num_samples
            oy = (sy + 0.5) / num_samples
            sample_x = 0 + ox
            sample_y = 0 + oy
            log(f"  Sample ({sx},{sy}): ({sample_x:.4f}, {sample_y:.4f})")

    log("\nExpected pydiffvg sample positions (typically same stratified pattern)")
    log("  (pydiffvg uses similar stratified sampling)")


def diagnose_stroke_coverage():
    """Test stroke coverage computation."""
    log("\n" + "="*60)
    log("DIAGNOSING STROKE COVERAGE")
    log("="*60)

    stroke_width = 2.0
    half_width = stroke_width / 2.0

    test_distances = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    log(f"Stroke width: {stroke_width}, half width: {half_width}")
    log("\ndiffvg-triton coverage (hard threshold):")
    for d in test_distances:
        inside = d <= half_width
        log(f"  dist={d:.1f}: inside={inside}")

    log("\npydiffvg coverage (likely uses SDF with smoothstep or similar)")
    log("  (Need to check diffvg source for exact formula)")


def save_debug_images(img_diffvg, img_triton, prefix="debug"):
    """Save images for visual comparison."""
    try:
        from PIL import Image

        # Convert to numpy uint8
        def to_uint8(img):
            arr = img.detach().cpu().numpy()
            arr = np.clip(arr, 0, 1) * 255
            return arr.astype(np.uint8)

        # Save diffvg output
        arr1 = to_uint8(img_diffvg)
        if arr1.shape[-1] == 4:
            Image.fromarray(arr1, 'RGBA').save(f'{prefix}_pydiffvg.png')
        else:
            Image.fromarray(arr1.squeeze(), 'L').save(f'{prefix}_pydiffvg.png')

        # Save triton output
        arr2 = to_uint8(img_triton)
        if arr2.shape[-1] == 4:
            Image.fromarray(arr2, 'RGBA').save(f'{prefix}_triton.png')
        else:
            Image.fromarray(arr2.squeeze(), 'L').save(f'{prefix}_triton.png')

        # Save difference (amplified)
        diff = torch.abs(img_diffvg - img_triton)
        diff_arr = to_uint8(diff * 10)  # Amplify differences
        if diff_arr.shape[-1] == 4:
            Image.fromarray(diff_arr, 'RGBA').save(f'{prefix}_diff.png')

        log(f"Saved debug images: {prefix}_*.png")

    except Exception as e:
        log(f"Could not save images: {e}", "WARN")


def run_full_diagnostics():
    """Run all diagnostic tests."""
    log("="*60)
    log("DIFFVG PARITY DIAGNOSTICS")
    log("="*60)

    # Check if pydiffvg is available
    try:
        import pydiffvg
        log("pydiffvg available: YES")
    except ImportError:
        log("pydiffvg available: NO - run inside diffvg container", "ERROR")
        return

    # Check if diffvg_triton is available
    try:
        from diffvg_triton import render, scene
        log("diffvg_triton available: YES")
    except ImportError as e:
        log(f"diffvg_triton available: NO - {e}", "ERROR")
        return

    # Test parameters
    width, height = 28, 28
    num_samples = 2

    log(f"\nTest parameters: {width}x{height}, {num_samples}x{num_samples} samples")

    # Create scenes
    log("\n--- Creating test scenes ---")
    shapes_pydiffvg, groups_pydiffvg, points = create_simple_stroke_scene_pydiffvg()
    shapes_triton, groups_triton = create_simple_stroke_scene_triton(points)

    log(f"Points: {points.tolist()}")
    log(f"Stroke width: 2.0")
    log(f"Stroke color: [0, 0, 0, 1] (black)")

    # Render with both
    log("\n--- Rendering with pydiffvg ---")
    img_diffvg = render_pydiffvg(shapes_pydiffvg, groups_pydiffvg, width, height, num_samples)
    log(f"Output shape: {img_diffvg.shape}")

    log("\n--- Rendering with diffvg-triton ---")
    img_triton = render_triton(shapes_triton, groups_triton, width, height, num_samples)
    log(f"Output shape: {img_triton.shape}")

    # Compare
    compare_pixel_values(img_diffvg, img_triton, "pydiffvg", "diffvg-triton")

    # Save debug images
    os.makedirs('results', exist_ok=True)
    save_debug_images(img_diffvg, img_triton, 'results/diagnose')

    # Additional diagnostics
    diagnose_distance_computation()
    diagnose_sample_positions()
    diagnose_stroke_coverage()

    # Print specific pixel values around the curve
    log("\n" + "="*60)
    log("PIXEL VALUES NEAR CURVE CENTER")
    log("="*60)

    # The curve is roughly centered around x=14, y ranges from 5 to 14
    for y in range(5, 16):
        for x in range(10, 20):
            v1 = img_diffvg[y, x, 0].item()
            v2 = img_triton[y, x, 0].item()
            diff = abs(v1 - v2)
            marker = " ***" if diff > 0.01 else ""
            if v1 < 0.99 or v2 < 0.99:  # Only show non-background
                log(f"  [{y:2d},{x:2d}] pydiffvg={v1:.4f}, triton={v2:.4f}, diff={diff:.4f}{marker}")


def diagnose_scene_serialization():
    """Compare how scenes are serialized/flattened."""
    log("\n" + "="*60)
    log("DIAGNOSING SCENE SERIALIZATION")
    log("="*60)

    try:
        import pydiffvg
    except ImportError:
        log("pydiffvg not available", "ERROR")
        return

    from diffvg_triton.scene import flatten_scene

    # Create simple scene
    shapes_pydiffvg, groups_pydiffvg, points = create_simple_stroke_scene_pydiffvg()
    shapes_triton, groups_triton = create_simple_stroke_scene_triton(points)

    width, height = 28, 28

    # pydiffvg serialization
    log("\npydiffvg scene serialization:")
    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes_pydiffvg, groups_pydiffvg)
    log(f"  Number of scene args: {len(scene_args)}")
    for i, arg in enumerate(scene_args[:10]):  # First 10 args
        if isinstance(arg, torch.Tensor):
            log(f"  arg[{i}]: Tensor shape={arg.shape}, dtype={arg.dtype}")
        else:
            log(f"  arg[{i}]: {type(arg).__name__} = {arg}")

    # diffvg-triton flattening
    log("\ndiffvg-triton scene flattening:")
    scene = flatten_scene(width, height, shapes_triton, groups_triton, device=torch.device('cpu'))

    log(f"  canvas: {scene.canvas_width}x{scene.canvas_height}")
    log(f"  num shapes: {len(scene.shape_types)}")
    log(f"  shape_types: {scene.shape_types.tolist()}")
    log(f"  shape_indices: {scene.shape_indices.tolist()}")

    if scene.paths:
        log(f"  paths.num_paths: {scene.paths.num_paths}")
        log(f"  paths.points shape: {scene.paths.points.shape}")
        log(f"  paths.points: {scene.paths.points.tolist()}")
        log(f"  paths.segment_types: {scene.paths.segment_types.tolist()}")
        log(f"  paths.num_segments: {scene.paths.num_segments.tolist()}")
        log(f"  paths.stroke_width: {scene.paths.stroke_width.tolist()}")

    log(f"  groups.num_groups: {scene.groups.num_groups}")
    log(f"  groups.has_stroke: {scene.groups.has_stroke.tolist()}")
    log(f"  groups.stroke_color: {scene.groups.stroke_color.tolist() if scene.groups.stroke_color is not None else None}")


if __name__ == '__main__':
    run_full_diagnostics()
    diagnose_scene_serialization()
