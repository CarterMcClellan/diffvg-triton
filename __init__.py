"""
diffvg_triton - Differentiable Vector Graphics with Triton

A pure Python/Triton implementation of differentiable vector graphics rendering.
This is a standalone module that can be used independently of the original diffvg.

Key components:
- scene: Scene flattening (convert shapes to GPU-friendly tensors)
- render: Main rendering pipeline
- autograd: PyTorch autograd integration for differentiable rendering
- kernels: Low-level Triton kernels for rendering operations

Usage:
    from diffvg_triton import render, render_grad

    # Non-differentiable rendering
    image = render(width, height, shapes, shape_groups)

    # Differentiable rendering (gradients flow to shape parameters)
    image = render_grad(width, height, shapes, shape_groups)
"""

from .scene import (
    FlattenedPaths,
    FlattenedShapeGroup,
    FlattenedScene,
    ShapeType,
    flatten_paths,
    flatten_shape_groups,
    flatten_scene,
)

from .render import (
    RenderMode,
    RenderConfig,
    render_scene,
    render_scene_py,
    render,
)

from .autograd import (
    GradientConfig,
    TritonRenderFunction,
    render_grad,
    DifferentiableRenderer,
)


__all__ = [
    # Scene
    'FlattenedPaths',
    'FlattenedShapeGroup',
    'FlattenedScene',
    'ShapeType',
    'flatten_paths',
    'flatten_shape_groups',
    'flatten_scene',
    # Render
    'RenderMode',
    'RenderConfig',
    'render_scene',
    'render_scene_py',
    'render',
    # Autograd
    'GradientConfig',
    'TritonRenderFunction',
    'render_grad',
    'DifferentiableRenderer',
]


# Version info
__version__ = '0.1.0'
__backend__ = 'triton'
