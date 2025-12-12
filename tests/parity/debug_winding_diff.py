#!/usr/bin/env python3
"""Debug soft winding number computation."""

import sys
sys.path.insert(0, '/workspace/examples')

import torch as th

def log(msg):
    print(f"[INFO] {msg}")


def _soft_ray_crossing(sample_pos, p0, p1, device, softness=0.1):
    """Soft ray-crossing contribution."""
    dy = p1[1] - p0[1]
    pt_y = sample_pos[:, 1]
    pt_x = sample_pos[:, 0]

    dy_safe = th.where(th.abs(dy) > 1e-8, dy, th.tensor(1e-8, device=device))
    t = (pt_y - p0[1]) / dy_safe
    x_int = p0[0] + t * (p1[0] - p0[0])

    t_valid = th.sigmoid((t + 0.01) / softness) * th.sigmoid((1.01 - t) / softness)
    x_valid = th.sigmoid((x_int - pt_x + 0.01) / softness)

    direction = th.where(dy > 0, th.tensor(1.0, device=device), th.tensor(-1.0, device=device))
    contrib = th.where(th.abs(dy) > 1e-8, direction * t_valid * x_valid, th.zeros_like(t_valid))

    return contrib


def main():
    log("="*60)
    log("DEBUG SOFT WINDING NUMBER")
    log("="*60)

    device = th.device('cpu')

    # Simple test: square path
    curve_points = th.tensor([
        [5.0, 5.0],
        [23.0, 5.0],
        [23.0, 23.0],
        [5.0, 23.0],
    ], dtype=th.float32, device=device)

    # Test points
    test_points = th.tensor([
        [14.0, 14.0],  # Inside square
        [14.0, 2.0],   # Above square
        [14.0, 26.0],  # Below square
        [2.0, 14.0],   # Left of square
        [26.0, 14.0],  # Right of square
        [14.0, 5.5],   # Just inside top edge
    ], dtype=th.float32, device=device)

    num_curve_pts = curve_points.shape[0]

    log(f"\nCurve points: {curve_points.tolist()}")

    for pt_idx, pt in enumerate(test_points):
        sample_pos = pt.unsqueeze(0)  # [1, 2]

        winding = th.zeros(1, device=device)

        log(f"\n--- Point {pt.tolist()} ---")

        for i in range(num_curve_pts):
            j = (i + 1) % num_curve_pts  # Close the loop
            contrib = _soft_ray_crossing(sample_pos, curve_points[i], curve_points[j], device)
            log(f"  Segment {i}->{j}: p0={curve_points[i].tolist()}, p1={curve_points[j].tolist()}, contrib={contrib.item():.4f}")
            winding = winding + contrib

        log(f"  Total winding: {winding.item():.4f}")

        inside = th.sigmoid((th.abs(winding) - 0.5) * 10.0)
        log(f"  Inside (sigmoid): {inside.item():.4f}")


if __name__ == "__main__":
    main()
