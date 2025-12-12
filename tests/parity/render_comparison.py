#!/usr/bin/env python3
"""Direct rendering comparison with identical parameters."""

import sys
sys.path.insert(0, '/workspace/examples')

import torch as th
import numpy as np

def log(msg):
    print(f"[INFO] {msg}")


def main():
    from mnist_vae import render_differentiable

    log("="*60)
    log("DIRECT RENDERING COMPARISON (triton)")
    log("="*60)

    device = th.device('cpu')

    # Use fixed parameters that should produce a clear shape
    # Single path with 3 cubic segments forming a recognizable shape
    all_points = th.tensor([[
        [
            [7.0, 14.0],   # Start
            [7.0, 7.0],    # Control 1
            [14.0, 7.0],   # Control 2
            [14.0, 14.0],  # End of seg 1

            [14.0, 21.0],  # Control 1
            [21.0, 21.0],  # Control 2
            [21.0, 14.0],  # End of seg 2

            [21.0, 7.0],   # Control 1
            [14.0, 7.0],   # Control 2
            [14.0, 14.0],  # End of seg 3
        ]
    ]], dtype=th.float32, device=device)

    all_widths = th.tensor([[2.0]], dtype=th.float32, device=device)
    all_alphas = th.tensor([[0.9]], dtype=th.float32, device=device)

    log(f"Points: {all_points[0, 0].tolist()}")
    log(f"Width: {all_widths[0, 0].item()}")
    log(f"Alpha: {all_alphas[0, 0].item()}")

    # Render
    output = render_differentiable(28, 28, all_points, all_widths, all_alphas, 3, samples=2)

    # Invert and map to [-1, 1]
    output = 1.0 - output
    output = output * 2.0 - 1.0

    log(f"\nOutput range: [{output.min():.4f}, {output.max():.4f}]")

    # Visualize
    out = output[0, 0].numpy()

    log("\n--- Rendered output ---")
    log("(## = >0.5, .. = >-0.5, space = <=-0.5)")
    for y in range(0, 28, 2):
        row = f"y={y:2d}: "
        for x in range(28):
            v = out[y, x]
            if v > 0.5:
                row += "##"
            elif v > -0.5:
                row += ".."
            else:
                row += "  "
        log(row)

    # Statistics
    stroke_pixels = (output > 0).sum().item()
    total = 28 * 28
    log(f"\nStroke pixels (>0): {stroke_pixels} ({100*stroke_pixels/total:.1f}%)")

    # Save params for pydiffvg comparison
    np.savez("/workspace/tests/results/render_params.npz",
             points=all_points.numpy(),
             widths=all_widths.numpy(),
             alphas=all_alphas.numpy())
    log("\nSaved params to /workspace/tests/results/render_params.npz")


if __name__ == "__main__":
    main()
