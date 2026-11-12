
:github_url: https://github.com/facebookresearch/DRTK

DRTK API
===================================

DRTK is a PyTorch library that provides functionality for differentiable rasterization.

The package includes the following modules:

* :doc:`transform`
* :doc:`rasterize`
* :doc:`render`
* :doc:`transparent_render`
* :doc:`pixel_grid_mesh`
* :doc:`source_features`
* :doc:`interpolate`
* :doc:`filter2d`
* :doc:`edge_grad_estimator`


.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Main API

   transform
   rasterize
   render
   transparent_render
   pixel_grid_mesh
   source_features
   interpolate
   filter2d
   edge_grad_estimator
   mipmap_grid_sample
   grid_scatter
   msi


.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Utils

   projection
   geometry
   indexing


.. toctree::
   :maxdepth: 2
   :caption: Concepts

   pixel_grid_topology
   cuda_graphs
