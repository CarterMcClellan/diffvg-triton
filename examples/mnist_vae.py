#!/usr/bin/env python3
"""
Train a VAE MNIST generator using diffvg_triton.

This is a standalone example demonstrating differentiable vector graphics
rendering with the Triton backend.

Usage:
    python mnist_vae.py train
    python mnist_vae.py sample
"""

import argparse
import os
import sys

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(PACKAGE_DIR))

# Import from diffvg_triton package
from diffvg_triton.scene import flatten_scene, FlattenedScene
from diffvg_triton.render import render_scene_py, RenderConfig
from diffvg_triton.autograd import render_grad


# Simple logging
def log(msg, *args):
    print(f"[INFO] {msg}" % args if args else f"[INFO] {msg}")


# Output directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "mnist_vae_triton")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Flatten(th.nn.Module):
    """Flatten layer for the encoder."""
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Path:
    """
    Minimal Path class compatible with the Triton backend.

    This replaces pydiffvg.Path for standalone usage.
    """
    def __init__(self, num_control_points, points, stroke_width=1.0, is_closed=False):
        self.num_control_points = num_control_points
        self.points = points
        self.stroke_width = stroke_width if isinstance(stroke_width, th.Tensor) else th.tensor([stroke_width])
        self.is_closed = is_closed
        self.thickness = None


class ShapeGroup:
    """
    Minimal ShapeGroup class compatible with the Triton backend.

    This replaces pydiffvg.ShapeGroup for standalone usage.
    """
    def __init__(self, shape_ids, fill_color=None, stroke_color=None, use_even_odd_rule=True):
        self.shape_ids = shape_ids if isinstance(shape_ids, th.Tensor) else th.tensor(shape_ids, dtype=th.int32)
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.use_even_odd_rule = use_even_odd_rule
        self.shape_to_canvas = None


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
    """
    Render shapes using the Triton backend.

    Args:
        canvas_width: Width of output image
        canvas_height: Height of output image
        shapes: List of Path objects
        shape_groups: List of ShapeGroup objects
        samples: Number of anti-aliasing samples (per axis)

    Returns:
        [H, W, 4] RGBA image tensor
    """
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # Flatten scene
    scene = flatten_scene(
        canvas_width, canvas_height,
        shapes, shape_groups,
        device=th.device('cpu')  # Use CPU for now
    )

    # Render
    config = RenderConfig(
        num_samples_x=samples,
        num_samples_y=samples,
        background_color=(1.0, 1.0, 1.0, 1.0),
    )

    img = render_scene_py(scene, config)
    return img


def imwrite(image, path, gamma=1.0):
    """Save image to file."""
    try:
        from PIL import Image
    except ImportError:
        log("PIL not available, skipping image save")
        return

    if isinstance(image, th.Tensor):
        image = image.detach().cpu().numpy()

    # Apply gamma
    if gamma != 1.0:
        image = np.power(np.clip(image, 0, 1), 1.0 / gamma)

    # Convert to uint8
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # Handle different shapes
    if image.ndim == 2:
        mode = 'L'
    elif image.ndim == 3 and image.shape[2] == 4:
        mode = 'RGBA'
    elif image.ndim == 3 and image.shape[2] == 3:
        mode = 'RGB'
    else:
        mode = 'L'
        image = image.squeeze()

    Image.fromarray(image, mode=mode).save(path)
    log(f"Saved image to {path}")


def _onehot(label):
    """Convert label to one-hot encoding."""
    bs = label.shape[0]
    label_onehot = label.new_zeros(bs, 10)
    label_onehot.scatter_(1, label.unsqueeze(1), 1)
    return label_onehot.float()


class VectorMNISTVAE(th.nn.Module):
    """
    VAE that generates vector graphics for MNIST digits.

    Encoder: CNN that maps image -> latent space
    Decoder: MLP that maps latent -> Bezier control points
    Renderer: Triton backend renders paths to image
    """

    def __init__(self, imsize=28, paths=4, segments=5, samples=2, zdim=128,
                 conditional=False, variational=True, stroke_width=None):
        super(VectorMNISTVAE, self).__init__()

        self.samples = samples
        self.imsize = imsize
        self.paths = paths
        self.segments = segments
        self.zdim = zdim
        self.conditional = conditional
        self.variational = variational

        if stroke_width is None:
            self.stroke_width = (1.0, 3.0)
        else:
            self.stroke_width = stroke_width

        ncond = 10 if self.conditional else 0

        # Encoder (convolutional)
        self.encoder = th.nn.Sequential(
            th.nn.Conv2d(1 + ncond, 64, 4, padding=1, stride=2),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(64, 128, 4, padding=1, stride=2),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(128, 256, 4, padding=1, stride=2),
            th.nn.LeakyReLU(0.2, inplace=True),
            Flatten(),
        )

        # Compute encoder output size
        # 28 -> 14 -> 7 -> 3 (approximately)
        encoder_out_size = 256 * 3 * 3

        self.mu_predictor = th.nn.Linear(encoder_out_size, zdim)
        if self.variational:
            self.logvar_predictor = th.nn.Linear(encoder_out_size, zdim)

        # Decoder
        nc = 512
        self.decoder = th.nn.Sequential(
            th.nn.Linear(zdim + ncond, nc),
            th.nn.SELU(inplace=True),
            th.nn.Linear(nc, nc),
            th.nn.SELU(inplace=True),
        )

        # Output heads
        # 4 points bezier with n_segments -> 3*n_segments + 1 points
        num_points = self.segments * 3 + 1
        self.point_predictor = th.nn.Sequential(
            th.nn.Linear(nc, 2 * self.paths * num_points),
            th.nn.Tanh()
        )

        self.width_predictor = th.nn.Sequential(
            th.nn.Linear(nc, self.paths),
            th.nn.Sigmoid()
        )

        self.alpha_predictor = th.nn.Sequential(
            th.nn.Linear(nc, self.paths),
            th.nn.Sigmoid()
        )

    def encode(self, im, label):
        bs, _, h, w = im.shape

        if self.conditional:
            label_onehot = _onehot(label)
            label_onehot = label_onehot.view(bs, 10, 1, 1).repeat(1, 1, h, w)
            x = th.cat([im, label_onehot], 1)
        else:
            x = im

        out = self.encoder(x)
        mu = self.mu_predictor(out)

        if self.variational:
            logvar = self.logvar_predictor(out)
            return mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(logvar)
        return mu + std * eps

    def decode(self, z, label=None):
        bs = z.shape[0]

        if self.conditional and label is not None:
            label_onehot = _onehot(label)
            z = th.cat([z, label_onehot], 1)

        feats = self.decoder(z)

        # Predict path parameters
        all_points = self.point_predictor(feats)
        all_points = all_points.view(bs, self.paths, -1, 2)
        all_points = all_points * (self.imsize // 2 - 2) + self.imsize // 2

        all_widths = self.width_predictor(feats)
        min_width, max_width = self.stroke_width
        all_widths = (max_width - min_width) * all_widths + min_width

        all_alphas = self.alpha_predictor(feats)

        # Render each sample in batch
        outputs = []
        for k in range(bs):
            shapes = []
            shape_groups = []

            for p in range(self.paths):
                points = all_points[k, p].contiguous().cpu()
                width = all_widths[k, p].cpu()
                alpha = all_alphas[k, p].cpu()

                color = th.cat([th.ones(3), alpha.view(1,)])
                num_ctrl_pts = th.zeros(self.segments, dtype=th.int32) + 2

                path = Path(
                    num_control_points=num_ctrl_pts,
                    points=points,
                    stroke_width=width,
                    is_closed=False
                )
                shapes.append(path)

                path_group = ShapeGroup(
                    shape_ids=th.tensor([len(shapes) - 1]),
                    fill_color=None,
                    stroke_color=color
                )
                shape_groups.append(path_group)

            # Render
            out = render(self.imsize, self.imsize, shapes, shape_groups,
                        samples=self.samples)

            # Convert to grayscale
            out = out.permute(2, 0, 1)[:3].mean(0, keepdim=True)
            outputs.append(out)

        output = th.stack(outputs).to(z.device)

        auxdata = {
            "points": all_points,
        }

        # Map to [-1, 1]
        output = output * 2.0 - 1.0

        return output, auxdata

    def forward(self, im, label):
        if self.variational:
            mu, logvar = self.encode(im, label)
            z = self.reparameterize(mu, logvar)
        else:
            mu = self.encode(im, label)
            z = mu
            logvar = th.zeros_like(mu)

        if self.conditional:
            output, aux = self.decode(z, label=label)
        else:
            output, aux = self.decode(z)

        aux["logvar"] = logvar
        aux["mu"] = mu

        return output, aux


class MNISTDataset(th.utils.data.Dataset):
    """MNIST dataset wrapper."""

    def __init__(self, data_dir):
        super().__init__()
        self.mnist = dset.MNIST(
            root=data_dir,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        im, label = self.mnist[idx]
        # Normalize to [-1, 1]
        im = (im - 0.5) / 0.5
        return im, label


def train(args):
    """Train the VAE."""
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = th.device('cuda' if args.cuda and th.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    # Dataset
    dataset = MNISTDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )

    # Model
    model = VectorMNISTVAE(
        imsize=28,
        paths=args.paths,
        segments=args.segments,
        samples=args.samples,
        zdim=args.zdim,
        conditional=args.conditional,
        variational=True
    )
    model.to(device)

    # Optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.5))

    # Training loop
    log(f"Starting training for {args.num_epochs} epochs")
    log(f"Config: paths={args.paths}, segments={args.segments}, zdim={args.zdim}")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (im, label) in enumerate(dataloader):
            im = im.to(device)
            label = label.to(device)

            # Forward
            rendering, auxdata = model(im, label)

            # Loss
            data_loss = th.nn.functional.mse_loss(rendering, im)

            mu = auxdata["mu"]
            logvar = auxdata["logvar"]
            kld = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1).mean()

            loss = data_loss + args.kld_weight * kld

            # Backward
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                log(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Data: {data_loss.item():.4f}, KLD: {kld.item():.4f}")

            # Limit batches for testing
            if args.max_batches and batch_idx >= args.max_batches:
                break

        avg_loss = total_loss / num_batches
        log(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pt")
            th.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            log(f"Saved checkpoint to {ckpt_path}")

        # Generate sample images
        if (epoch + 1) % 5 == 0:
            generate_samples(model, epoch + 1, device)

    log("Training complete!")


def generate_samples(model, epoch, device):
    """Generate sample images from the model."""
    model.eval()

    with th.no_grad():
        # Random samples
        z = th.randn(16, model.zdim).to(device)

        if model.conditional:
            label = th.arange(10).repeat(2)[:16].to(device)
        else:
            label = None

        images, _ = model.decode(z, label=label)
        images = (images + 1) / 2  # Map to [0, 1]

    # Create grid
    n = 4
    grid = images.view(n, n, 28, 28).permute(0, 2, 1, 3)
    grid = grid.contiguous().view(n * 28, n * 28)
    grid = th.clamp(grid, 0, 1).cpu().numpy()

    path = os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch}.png")
    imwrite(grid, path, gamma=2.2)


def main():
    parser = argparse.ArgumentParser(description="MNIST VAE with Triton backend")
    parser.add_argument("command", choices=["train", "sample"], help="Command to run")
    parser.add_argument("--cuda", action="store_true", default=th.cuda.is_available(),
                       help="Use CUDA if available")
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training args
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--bs", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--kld_weight", type=float, default=1.0, help="KLD loss weight")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch (for testing)")

    # Model args
    parser.add_argument("--paths", type=int, default=4, help="Number of paths")
    parser.add_argument("--segments", type=int, default=3, help="Segments per path")
    parser.add_argument("--samples", type=int, default=2, help="AA samples")
    parser.add_argument("--zdim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--conditional", action="store_true", help="Conditional VAE")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "sample":
        log("Sample generation not yet implemented")


if __name__ == "__main__":
    main()
