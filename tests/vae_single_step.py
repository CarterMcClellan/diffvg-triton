#!/usr/bin/env python3
"""Single step test of MNIST VAE to see actual rendering output."""

import torch as th
import sys
sys.path.insert(0, '/workspace/examples')

def log(msg):
    print(f"[INFO] {msg}")


def main():
    from mnist_vae import VectorMNISTVAE

    log("="*60)
    log("MNIST VAE SINGLE STEP TEST (triton)")
    log("="*60)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # Create model with same params as pydiffvg version
    model = VectorMNISTVAE(
        imsize=28,
        paths=4,
        segments=5,
        samples=2,
        zdim=128,
        conditional=False,
        variational=True,
        stroke_width=(1.0, 3.0)
    ).to(device)

    # Random latent code
    th.manual_seed(42)  # Same seed as pydiffvg
    z = th.randn(1, 128, device=device)

    # Decode
    log("Decoding random z...")
    output, auxdata = model.decode(z)

    log(f"Output shape: {output.shape}")
    log(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Visualize
    out = output[0, 0].detach().cpu().numpy()

    log("\n--- Decoded output ---")
    log("(## = >0.5 (stroke), .. = >-0.5, space = <=-0.5 (background))")
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
    bg_pixels = (output < 0).sum().item()
    total_pixels = 28 * 28

    log(f"\nStatistics:")
    log(f"  Stroke pixels (>0): {stroke_pixels} ({100*stroke_pixels/total_pixels:.1f}%)")
    log(f"  Background pixels (<0): {bg_pixels} ({100*bg_pixels/total_pixels:.1f}%)")


if __name__ == "__main__":
    main()
