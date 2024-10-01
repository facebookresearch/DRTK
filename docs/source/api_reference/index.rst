
:github_url: https://github.com/facebookresearch/DRTK

DRTK API
===================================

DRTK is a PyTorch library that provides functionality for differentiable rasterization.

The package consists of five main modules:

* :doc:`transform`
* :doc:`rasterize`
* :doc:`render`
* :doc:`interpolate`
* :doc:`edge_grad_estimator`


.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Main API

   transform
   rasterize
   render
   interpolate
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
