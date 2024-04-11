# Differentiable Rendering Toolkit
## Rasterizer
Produces an index and depth image from projected image-space vertices and an index list. Includes two backends: CUDA and Vulkan.

## CUDA Renderer
Provides a PyTorch Autograd Function that uses an index image to interpolate:
* Barycentric Coordinates
* UV Coordinates
* Depth

in a differentiable fashion w.r.t. the input vertex positions.

## RenderLayer
Combines rasterization (CUDA or Vulkan) with CUDA rendering to render an image given
geometry, texture, and camera pose. Supports boundary-aware and
background-aware rendering to propagate gradients w.r.t. vertices from/to
adjacent pixels and background images.

For more details:
* [CUDA Renderer](src/render/README.md)
* [RenderLayer](drtk/renderlayer/README.md)

## Dependencies
* PyTorch >= 1.6
* [rpack](https://pypi.org/project/rectangle-packer/) for packing textures in the multi-mesh renderlayer.

If rpack isn't available, the multi-mesh renderlayer will not be available.

## Compiling

```
python setup.py build_ext --inplace
```
