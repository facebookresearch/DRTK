drtk.pixel_grid_mesh
====================

.. currentmodule:: drtk

.. autosummary::
    :nosignatures:

    pixel_grid_mesh
    bernoulli_topology_samples
    bernoulli_render_loss
    pixel_grid_mesh_ref

Build a mesh from an image-aligned grid, deform its cells, and learn which
cells and connections to keep. The mesh builder uses hard topology; the
Bernoulli helpers estimate gradients of the expected loss over sampled meshes.

See :doc:`pixel_grid_topology` for camera conventions, cuts, feathering, and
the gradient model, and :doc:`../tutorials/DRTK_Tutorial_pixel_grid_mesh` for
a rendered optimization example.

.. autofunction:: pixel_grid_mesh

.. autofunction:: bernoulli_topology_samples

.. autofunction:: bernoulli_render_loss

.. autofunction:: pixel_grid_mesh_ref
