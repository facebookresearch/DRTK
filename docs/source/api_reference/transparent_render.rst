drtk.transparent_render
=======================

These functions render arbitrary triangle meshes with depth-sorted
transparency. Covered fragments are evaluated from front to back and
composited into a color and alpha image without a hard z-buffer selecting a
single surface.

Ordering uses each triangle's mean vertex depth, not a per-pixel depth sort.
This approximation suits small triangles with locally consistent ordering;
intersecting or depth-crossing triangles can require a different order at
different pixels and will not composite exactly. Subdivide or resolve such
intersections upstream.

``render_transparent`` samples an RGB or RGBA texture directly.
``render_transparent_mlp`` samples one feature texture before evaluating a
small per-fragment appearance network, while
``render_transparent_mlp_two_textures`` samples two textures at independent
``uv0`` and ``uv1`` coordinates and concatenates their features. The combined
MLP input may contain up to 32 channels.

Independent UV sets do not provide automatic visibility or provenance gating.
When combining original-source features with features sampled at a displaced
location, the caller must gate samples that belong to another surface and
provide the appearance model with the required validity information. For
example, a hidden chest surface must not consume features from a hand that
occludes its current projection. Topology and component ownership remain
upstream responsibilities.

All variants support nearest, bilinear, and bicubic filtering together with
zeros, border, and reflection padding. Texture coordinates consistently use
``align_corners=False`` semantics; this is deliberately not configurable.
They differentiate compositing, perspective-correct interpolation, texture
sampling, vertex opacity, and interior vertex motion. Triangle topology and
depth ordering remain discrete. All CUDA entry points add transparent
silhouette gradients to vertex X/Y motion by default. Set
``edge_gradients=False`` to disable the boundary estimator. The
forward image, compositing order, and appearance gradients are unchanged.

Transparent edge gradients
-------------------------

This estimator extends `Rasterized Edge Gradients
<https://arxiv.org/abs/2405.02508>`_ to transparent compositing. It averages the
two neighboring pixels' loss-weighted insertion/removal effects at each
geometric edge crossing. Each pixel supplies its own foreground attenuation
and suffix composite at the queried triangle's position in the existing sort
order. Final RGB/alpha alone cannot recover this information.

Common indexed edges between triangles on opposite projected sides are
ordinary tessellation and produce no boundary term. Incident triangles that
fold onto the same side move together at their shared edge; their alpha
interaction is evaluated jointly. Material opacity and sampled appearance
are evaluated at the geometric edge. A zero-alpha edge therefore contributes
zero even if nearby interior samples have nonzero alpha. Saturating an opacity
activation does not suppress a nonzero silhouette gradient.

Only vertex X/Y gradients receive this extra term. There are no derivatives
of per-triangle ordering, topology, triangle intersections, or clipping at the
image boundary. Separate coincident vertices remain separate geometry; callers
should share indices where they intend a continuous, jointly moving edge.
The pure-PyTorch references remain oracles for smooth/interior derivatives;
the boundary estimator is validated against compositing identities and
area-integrated references, not infinitesimal differences of hard coverage.

All edge discovery and queries run on CUDA during geometry backward. Shared
edges and zero material-opacity boundaries are removed before expensive
queries. Sparse queries reuse the saved tile lists and replay shading only at
boundary pixel pairs. No per-pixel sorting or fragment buffer is added, and
inference and appearance-only differentiation do not run this work.

Appearance activations
----------------------

Direct textures and neural appearance outputs use separate activation
contracts because their natural parameter ranges differ.

For direct RGB or RGBA textures, RGB defaults to ``max(x, 0)``
(``"relu"``). The alternatives are ``"identity"``, ``"shifted_sigmoid"``
(``sigmoid(x - 0.5)``), and ``"clamp"``. Texture alpha defaults to
``1 - exp(-max(x, 0))`` (``"relu_density"``). Its alternatives are
``"shifted_sigmoid"``, ``"exp_density"`` (``1 - exp(-exp(x))``), and
``"clamp"``. An RGB texture has implicit alpha one. Use
``rgb_activation="identity"`` and ``alpha_activation="clamp"`` for a
conventional RGBA texture whose channels already represent display color and
opacity.

For direct texture rendering, activation occurs *after* texture filtering.
Consequently, neighboring texels on opposite sides of the ReLU boundary can
produce an active interpolated value. Backpropagation through that filtered
sample distributes gradient to all contributing texels, including a texel
whose value would be in the dead region if sampled alone. The MLP variants
likewise activate only after feature filtering and the final affine layer.

The MLP output is naturally centered near zero. MLP RGB therefore defaults to
``max(x + 0.5, 0)`` (``"shifted_relu"``), and also supports
``"shifted_identity"`` (``x + 0.5``), ``"sigmoid"``, and
``"shifted_clamp"`` (``clamp(x + 0.5, 0, 1)``). MLP alpha defaults to
``1 - exp(-max(x + 0.5, 0))`` (``"shifted_relu_density"``), and also supports
``"sigmoid"``, ``"exp_density"``, and ``"shifted_clamp"``. Thus a raw MLP RGB
value of zero maps to 0.5 for every bounded or shifted RGB option. A raw MLP
alpha value of zero maps to ``1 - exp(-0.5)`` for the default density mapping
or 0.5 for sigmoid and shifted clamp.

The density mappings interpret a nonnegative value as optical density and
convert it to opacity. They are mathematically below one without the two-sided
saturation of a sigmoid. In particular, the ReLU zero set can describe complex
contours in a learned field, which is the level-set behavior highlighted by
`ReLU Fields: The Little Non-linearity That Could
<https://arxiv.org/abs/2205.10824>`_. The half-unit shifts and density-to-opacity
mapping are this renderer's parameterization choices. Here optical density
means per-fragment optical thickness; no world-space path length is integrated.
``"exp_density"``
corresponds to an exponential density followed by the density-to-opacity
conversion used by Instant-NGP-style volume models. Its raw input is capped at
15 before exponentiation to avoid overflow. At sufficiently large density,
float32 can still round the resulting opacity to exactly one; the fragment
alpha cap below handles that case before compositing.

Fragment alpha is clamped internally below one before compositing. This fixed
numeric safeguard prevents division by zero in reverse compositing and is
not a material parameter. It is applied after the appearance alpha and
interpolated vertex opacity are multiplied. Backward saves transmittance as a
scaled product separately from output alpha, so gradients remain valid when
output alpha rounds to one or an unscaled product underflows.

Appearance, material-opacity, and upper safety clamps use a unit subgradient
at an exact endpoint and zero outside the interval. This is an explicit DRTK
convention shared by the native operators and reference functions; it does not
depend on the installed PyTorch build's scalar-clamp convention. ReLU uses
zero gradient at zero, including the lower bound on final fragment alpha.

.. automodule:: drtk.transparent_render
    :no-index:

.. currentmodule:: drtk

.. autosummary::
    :nosignatures:

    render_transparent
    render_transparent_mlp
    render_transparent_mlp_two_textures
    render_transparent_ref
    render_transparent_mlp_ref
    render_transparent_mlp_two_textures_ref

.. autofunction:: render_transparent

.. autofunction:: render_transparent_mlp

.. autofunction:: render_transparent_mlp_two_textures

.. autofunction:: render_transparent_ref

.. autofunction:: render_transparent_mlp_ref

.. autofunction:: render_transparent_mlp_two_textures_ref
