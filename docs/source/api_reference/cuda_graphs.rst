Avoiding CPU synchronization
============================

Mesh topology and projected triangle coverage can change the amount of storage
needed by a call. By default, :func:`drtk.pixel_grid_mesh` reads the largest
vertex and face counts back to the CPU before allocating its outputs. The
transparent renderers and source-transmittance functions similarly read the
number of tile/triangle intersections before allocating their sorting buffers.
These reads make the CPU wait for the GPU.

Supply ``output_capacity`` to the mesh builder and ``tile_capacity`` to the
renderer or transmittance function to avoid those reads. Calls still allocate
ordinary PyTorch tensors; capacities specify their sizes. Actual counts remain
on the GPU. Geometry, topology, and appearance may change between calls while
the allocated shapes stay fixed.

Choosing capacities
-------------------

``pixel_grid_mesh(..., output_capacity=(vertices, faces))`` sets the maximum
vertex and face counts **per mesh**, rather than across the batch. Each mesh
occupies a compact prefix of its outputs. Spare entries use the documented
padding values and contribute no gradients. The renderer accepts the padded
face array directly.

For an H-by-W cell grid, conservative bounds valid for all topology decisions
are:

.. list-table::
   :header-rows: 1

   * - Mesh
     - Vertex capacity
     - Face capacity
   * - Without feathering
     - ``4 * H * W``
     - ``2 * H * W``
   * - With feathering
     - ``16 * H * W``
     - ``18 * H * W``

``tile_capacity`` bounds the total number of tile/triangle intersections
**across the batch**. Tiles contain 4-by-4 image pixels. Each triangle's
projected bounding box contributes an entry for every tile it overlaps.
Source-transmittance queries also include a one-pixel halo around that box.
Rendering and transmittance therefore may need different capacities for the
same mesh and image size.

A conservative bound for N meshes with F face slots and an H-by-W output is
``N * F * ((H + 3) // 4) * ((W + 3) // 4)``. This assumes every triangle
covers the entire image and can be much too large for a finely tessellated mesh.
Use a tighter bound when the geometry's projected extent is constrained.
Capacities must cover all inputs that will be replayed; counts observed in a
single frame are not a bound on later motion or topology changes.

Larger capacities consume more memory and increase padding and sorting work.
Removing a count readback alone does not guarantee a faster call.

Exceeding a capacity raises an asynchronous CUDA device assertion. As with
other CUDA device assertions, the process must be restarted afterward. The
operation does not silently drop geometry to fit the requested size.

CUDA graph capture
------------------

Fixed-capacity calls also support ``torch.cuda.CUDAGraph``. Capturing and
replaying a sequence of GPU operations reduces CPU launch overhead, which can
be useful when many small operations run together.

This example captures mesh construction, rendering, and gradients. Input
storage remains fixed; update its contents before replaying the graph.

.. code-block:: python

   import torch
   from drtk import pixel_grid_mesh, render_transparent

   n, h, w = 2, 8, 8
   depth = torch.ones(n, h, w, device="cuda", requires_grad=True)
   displacement = torch.zeros(n, h, w, 3, device="cuda", requires_grad=True)
   intrinsics = torch.eye(3, device="cuda")[None]
   alive = torch.ones(n, h, w, device="cuda", dtype=torch.bool)
   right = torch.ones(n, h, w - 1, device="cuda", dtype=torch.bool)
   down = torch.ones(n, h - 1, w, device="cuda", dtype=torch.bool)
   texture = torch.rand(1, 3, h, w, device="cuda")
   mesh_capacity = (4 * h * w, 2 * h * w)
   tile_capacity = n * mesh_capacity[1] * ((h + 3) // 4) * ((w + 3) // 4)

   def step():
       v, uv, _, faces, _, _, _ = pixel_grid_mesh(
           depth, displacement, intrinsics, alive, right, down,
           output_capacity=mesh_capacity,
       )
       # The identity intrinsics make projected XY equal to X/Z, Y/Z.
       screen = torch.cat((v[..., :2] / v[..., 2:], v[..., 2:]), dim=-1)
       rgb, alpha, _ = render_transparent(
           screen, faces, uv, texture, h, w, tile_capacity=tile_capacity,
       )
       gradients = torch.autograd.grad(
           rgb.square().mean() + alpha.mean(), (depth, displacement),
       )
       return rgb, gradients

   stream = torch.cuda.Stream()
   stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
       for _ in range(3):
           step()
   torch.cuda.current_stream().wait_stream(stream)

   graph = torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph, stream=stream):
       image, gradients = step()

   with torch.no_grad():
       displacement[..., 0].add_(0.05)
       alive[1, 3, 3] = False
   graph.replay()
   # image and gradients now contain the result for the updated inputs.

The texture and MLP renderer variants and all source-transmittance variants
accept the same ``tile_capacity`` option. Feature sampling with
:func:`drtk.sample_with_transmittance` already has fixed output shapes and
needs no capacity argument.

CUDA graph replay uses the same tensor addresses and operation structure on
every iteration. Copy new data into the captured inputs rather than replacing
them with new tensors. Changing batch dimensions, image dimensions, or
capacities requires a new capture. Read results on the CPU only when needed;
calling ``.item()`` or ``.cpu()`` each iteration would introduce another wait.
