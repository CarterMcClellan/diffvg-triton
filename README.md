# Triton Backend for Differentiable Vector Graphics

A pure Python/Triton implementation of differentiable vector graphics rendering, designed as a standalone replacement for the C++/CUDA diffvg backend.

## Overview

This module provides GPU-accelerated differentiable rendering of vector graphics (SVG-like paths) using OpenAI's Triton compiler. It enables gradient-based optimization of vector graphics parameters, useful for:

- Neural network-based SVG generation
- Vector graphics optimization
- Differentiable rendering research

## Features

- **Pure Python** - No C++/CUDA compilation required
- **Triton JIT kernels** - GPU-accelerated computational kernels
- **PyTorch integration** - Full autograd support for backpropagation
- **Standalone** - No dependency on the original diffvg C++ code

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+
- NumPy

```bash
pip install torch triton numpy
```

### Setup

Copy the `triton_backend` directory to your project:

```bash
cp -r triton_backend /path/to/your/project/
```

Or add to your Python path:

```python
import sys
sys.path.append('/path/to/triton_backend')
```

## Quick Start

### Basic Rendering

```python
import torch
from triton_backend import render, FlattenedScene
from triton_backend.scene import flatten_scene

# Define a simple path (triangle)
class Path:
    def __init__(self, points, num_control_points, is_closed=True):
        self.points = torch.tensor(points, dtype=torch.float32)
        self.num_control_points = torch.tensor(num_control_points, dtype=torch.int32)
        self.is_closed = is_closed
        self.stroke_width = torch.tensor([1.0])
        self.thickness = None

class ShapeGroup:
    def __init__(self, shape_ids, fill_color=None, stroke_color=None):
        self.shape_ids = torch.tensor(shape_ids, dtype=torch.int32)
        self.fill_color = torch.tensor(fill_color) if fill_color else None
        self.stroke_color = torch.tensor(stroke_color) if stroke_color else None
        self.use_even_odd_rule = True
        self.shape_to_canvas = None

# Create shapes
path = Path(
    points=[[50, 10], [90, 90], [10, 90]],
    num_control_points=[0, 0, 0],  # Line segments
    is_closed=True
)

group = ShapeGroup(
    shape_ids=[0],
    fill_color=[1.0, 0.0, 0.0, 1.0]  # Red
)

# Render
image = render(
    canvas_width=100,
    canvas_height=100,
    shapes=[path],
    shape_groups=[group],
    num_samples_x=2,
    num_samples_y=2
)

# image is [100, 100, 4] RGBA tensor
```

### Differentiable Rendering

```python
from triton_backend import render_grad, DifferentiableRenderer

# Create renderer
renderer = DifferentiableRenderer(
    canvas_width=100,
    canvas_height=100,
    num_samples_x=2,
    num_samples_y=2
)

# Create path with gradient-enabled points
points = torch.tensor([[50, 10], [90, 90], [10, 90]],
                      dtype=torch.float32, requires_grad=True)

path = Path(points=points, num_control_points=[0, 0, 0])
group = ShapeGroup(shape_ids=[0], fill_color=[1.0, 0.0, 0.0, 1.0])

# Render (differentiable)
image = renderer.render([path], [group])

# Compute loss and backprop
target = torch.zeros(100, 100, 4)
loss = (image - target).pow(2).mean()
loss.backward()

# Gradients are now in points.grad
print(points.grad)
```

## Module Structure

```
triton_backend/
├── __init__.py           # Main exports
├── scene.py              # Scene flattening (shapes → tensors)
├── render.py             # Main rendering pipeline
├── autograd.py           # PyTorch autograd integration
├── kernels/
│   ├── __init__.py       # Kernel exports
│   ├── solve.py          # Polynomial root solvers
│   ├── winding.py        # Winding number computation
│   ├── distance.py       # Closest point on curves
│   ├── filter.py         # Anti-aliasing filters
│   ├── composite.py      # Alpha blending
│   ├── boundary.py       # Boundary gradient sampling
│   └── rng.py            # PCG32 random number generator
├── examples/
│   └── mnist_vae.py      # MNIST VAE training example
└── tests/
    ├── test_kernels.py   # Kernel unit tests
    └── test_render.py    # Rendering tests
```

## Examples

### MNIST VAE

Train a VAE that generates MNIST digits as vector graphics:

```bash
cd examples
python mnist_vae.py train --paths 4 --segments 3 --num_epochs 50
```

Options:
- `--paths`: Number of Bezier paths per digit (default: 4)
- `--segments`: Segments per path (default: 3)
- `--zdim`: Latent dimension (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--bs`: Batch size (default: 8)

## Supported Primitives

### Shapes
- **Path**: Bezier curves (linear, quadratic, cubic segments)
- Circle, Ellipse, Rect (planned)

### Segment Types
- `0`: Line segment (2 points)
- `1`: Quadratic Bezier (3 points)
- `2`: Cubic Bezier (4 points)

### Fill Rules
- Even-odd rule
- Non-zero winding rule

### Colors
- Solid RGBA colors
- Gradients (planned)

## API Reference

### Main Functions

#### `render(canvas_width, canvas_height, shapes, shape_groups, ...)`
Render shapes to an image.

**Parameters:**
- `canvas_width`, `canvas_height`: Output image dimensions
- `shapes`: List of Path objects
- `shape_groups`: List of ShapeGroup objects
- `num_samples_x`, `num_samples_y`: Anti-aliasing samples (default: 2)
- `background_color`: Background RGBA (default: white)

**Returns:** `[H, W, 4]` RGBA tensor

#### `render_grad(...)`
Same as `render()` but with gradient support.

#### `flatten_scene(canvas_width, canvas_height, shapes, shape_groups, device)`
Convert shapes to GPU-friendly tensor representation.

### Classes

#### `RenderConfig`
Configuration for rendering.

```python
config = RenderConfig(
    num_samples_x=2,
    num_samples_y=2,
    seed=42,
    filter_type=FilterType.BOX,
    filter_radius=1.0,
    use_prefiltering=False,
    background_color=(1.0, 1.0, 1.0, 1.0)
)
```

#### `DifferentiableRenderer`
High-level wrapper for differentiable rendering.

```python
renderer = DifferentiableRenderer(width, height)
image = renderer.render(shapes, shape_groups)
```

## Performance Notes

The current implementation prioritizes correctness over speed:

1. **Python reference implementation** - Used for all rendering operations
2. **Triton kernels written but not fully enabled** - Ready for optimization
3. **Sequential pixel processing** - Can be parallelized

### Optimization Roadmap

1. Enable Triton JIT kernels for distance/winding computation
2. Parallelize per-pixel rendering loop
3. Add BVH acceleration for complex scenes
4. Implement tiled rendering for large images

## Testing

Run the test suite:

```bash
# From the triton_backend directory
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python -c "
from tests.test_kernels import *
TestSolve().test_quadratic_two_roots()
print('Tests passed!')
"
```

## License

MIT License - see the main diffvg repository for details.

## Acknowledgments

This is a Triton-based reimplementation of [diffvg](https://github.com/BachiLi/diffvg) by Tzu-Mao Li et al.

## Citation

If you use this code, please cite the original diffvg paper:

```bibtex
@article{li2020differentiable,
    title={Differentiable Vector Graphics Rasterization for Editing and Learning},
    author={Li, Tzu-Mao and Lukáč, Michal and Gharbi, Michaël and Ragan-Kelley, Jonathan},
    journal={ACM Trans. Graph. (Proc. SIGGRAPH Asia)},
    year={2020}
}
```
