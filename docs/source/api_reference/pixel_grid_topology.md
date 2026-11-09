# Pixel-grid topology and gradients

The mesh builder creates a triangle mesh from a regular grid of **cells centered on
source-image pixels**. Each live cell contributes a quad whose initial corners
are at pixel corners. Neighbor affinities determine which corner vertices are
shared. Geometry construction is independent of the model producing its inputs:
prediction heads, task losses, and camera-placement policies remain with the caller.

The differentiable training objective is the expected loss of sampled hard
meshes. It is not a derivative of triangle indices and is not the loss of an
averaged rendered image. At inference, construct one deterministic hard mesh.

For function signatures, see [the module reference](pixel_grid_mesh.rst).
The [optimization tutorial](../tutorials/DRTK_Tutorial_pixel_grid_mesh.rst)
shows how to use these pieces together.

## Inputs and outputs

```python
v, vt0, vt1, vi, vertex_opacity, face_cell_ids, face_ids = pixel_grid_mesh(
    depth_z, displacement_xyz, intrinsics,
    alive, join_right, join_down,
    diagonal=None,
    feather_width=None,
    return_metadata=False,
)
```

The return value is always a plain **seven-element tuple**. Optional entries
become `None`; they are never omitted and do not change position.

| Input | Shape | Meaning |
| --- | --- | --- |
| `depth_z` | H × W | Positive source-camera Z depth, not inverse depth or ray distance |
| `displacement_xyz` | H × W × 3 | Translation proposal in source-camera units |
| `intrinsics` | 3 × 3 | Pinhole K for this source grid, including any crop/resize |
| `alive` | H × W | Boolean cell participation |
| `join_right` | H × (W−1) | Boolean connection to the right neighbor |
| `join_down` | (H−1) × W | Boolean connection to the neighbor below |
| `diagonal` | H × W or `None` | False: TL–BR; True: TR–BL; None: geometric choice |
| `feather_width` | H × W or `None` | Nonnegative strip width in source-image pixels |

The first implementation operates on one grid per call. Call it per example
for a ragged batch; do not treat different examples as one connected grid.
Floating inputs share float32/float64 dtype and device; hard decisions are bool.
Depth, K, displacement, and widths must be finite and geometrically valid. The
displaced surface must remain in front of the source camera. Metadata/shape
validation is at the API boundary, not repeated in pixel loops.

| Output | Meaning |
| --- | --- |
| `v` | Final source-camera **3D positions, already fitted and displaced**, including strips when enabled |
| `vt0` | Original anchor UVs in [0,1]; strip vertices copy their boundary anchor |
| `vt1` | Source-camera projection of the final positions; may leave [0,1] |
| `vi` | Compact int32 triangle indices |
| `vertex_opacity` | Generated material-opacity multiplier: core 1, outer strip 0; `None` without strips |
| `face_cell_ids` | Source-cell ownership, or `None` unless metadata is requested |
| `face_ids` | Stable cell/local-face identifiers, or `None` unless metadata is requested |

Faces are emitted in increasing stable-ID order, including strips. Removing
unrelated cells preserves relative tie order for the renderer's equal-depth
triangles. Native and reference implementations use the same order.

`vertex_opacity` defines the boundary fade. Pass it as
`opacity=vertex_opacity` to the transparent renderer to multiply the appearance
alpha. None means an all-one multiplier.

Project `v` into the desired target camera before rendering: the transparent
renderer accepts target pixel XY and camera Z, not source-camera XYZ. Both UVs
use the existing 2D interpolation; no homogeneous/projective-UV extension is
introduced in this version. Exact projective resampling is deliberately deferred
because the triangles are small. Source visibility/provenance gating remains
the caller's responsibility; a reprojected UV alone does not establish that the
source feature belongs to this surface.

## Pixel and camera convention

Source pixel (x,y) has its center at integer coordinates. Its corners are
(x±0.5,y±0.5). Normalized texture UV is ((x+0.5)/W,(y+0.5)/H), without a vertical
flip. A K resized by sx,sy needs fx'=sx·fx and cx'=sx·(cx+0.5)−0.5, and likewise
for Y. The same convention must be used when resizing the source image and its camera.

Camera Z and camera-space displacement are sufficient. There is no inverse-depth
plane fitting in this implementation. World/placement transforms remain outside
the builder.

## Cuts and corner sharing

Connection decisions are undirected and exist once per neighboring pair. A
connection requires both cells alive. Alive/dead borders and the outer grid
boundary are exposed boundaries, regardless of a join bit.

An interior cut consisting of **one isolated grid edge is ignored**. It survives
when either endpoint touches another cut edge, or an already exposed boundary.
"Border" includes the outside grid and a live/dead boundary, including holes.
Two remote isolated cuts do not support each other. Two cuts meeting at a corner
do. Classification uses the original sampled cuts simultaneously, not an
iterative endpoint-pruning algorithm that would erase a two-edge chain.

Construct a connected fan among the at-most-four live cell corners incident on
each lattice corner. Give each fan one shared vertex. Diagonally touching but
unconnected cells remain separate. Connections elsewhere in the mesh never
weld a local cut. An open fan can contain all four incident cells: its two
boundary edges share one **crack-tip vertex**. That tip is intentional; do not
force an additional cut or duplicate the tip. Interior cuts with both tips on
one edge are already removed by the isolated-cut rule; no extra midpoint
vertices are introduced to represent them.

Cut realization depends only on sampled bits. The effective probability
that an edge is cut is therefore not simply its raw Bernoulli probability.
Training differentiates the distribution of proposals followed by this exact
deterministic realization.

## Closed-form displaced corners and diagonals

For a cell c and its source corner q, form the proposal

```{math}
Y_{cq}=d_c K^{-1}[q_x,q_y,1]^T+\Delta_c.
```

For a corner fan S, use its least-squares fit

```{math}
V_{q,S}=\frac{1}{|S|}\sum_{c\in S}Y_{cq}.
```

No averaging crosses a cut except the intentionally shared tip. This is local
and closed form, with ordinary autograd through depth, displacement and K.
The displacement is a **proposal**, not an exact constraint on every final
cell center after welding. Exact independent centers would require extra center
vertices or a coupled solve; neither is part of this contract.

Each live quad emits two triangles. Automatic diagonal choice compares the two
diagonal midpoints to the requested displaced cell center in 3D, choosing the
smaller squared error and TL–BR on a tie. This is a geometric heuristic, not a
claim of optimal rendered tessellation. The decision itself is discrete. Pass
explicit sampled diagonal bits to learn it from rendered loss. No soft overlay
of both triangulations is rendered. Degenerate/folded predictions are not
repaired into a different shape; geometry regularization belongs to the caller.

## Optional boundary feathering

Feathering extrudes source-screen-normal strips at the fitted boundary depth.
Widths at a boundary vertex average the two incident boundary-cell widths in
that fan. Separate sides of a cut do not share widths except at a shared tip.
The construction uses shared miter joins where bounded, bevels otherwise, and
caps for reversals. All outer vertices carry opacity multiplier zero; the core
vertices remain one. Linear interpolation gives the fade across each strip.

Original UVs extend constantly across the strip; reprojected UVs use the new
positions. The network supplies only width. Geometry and width receive ordinary
render gradients. The zero-alpha outer contour contributes no artificial
coverage jump to the existing edge estimator.

This is a **local extrusion**, not global polygon offset/Boolean surgery.
Large widths, acute crack tips, nearby opposing boundaries, and folded geometry
can cause strip overlap. The initial implementation does not claim to eliminate
those intersections or compensate their alpha. Keep strips narrow; test the
zero-feather baseline. Global overlap resolution and learned width policies are
separate work. Miter/bevel switching is also discrete.

## Stochastic topology objective

Let independent proposal bits have p=σ(η), and let M include the cut cleanup.
The intended objective is

```{math}
F(\eta,x)=\mathbb E_z[\mathcal L(R(M(z,x)))].
```

For one bit,

```{math}
\frac{\partial F}{\partial\eta_i}
=p_i(1-p_i)\mathbb E_{z_{-i}}[\mathcal L(z_i=1)-\mathcal L(z_i=0)].
```

`bernoulli_render_loss` uses ARM with the full scalar loss. For shared uniform u, sample
z⁺=1[u>1−p] and z⁻=1[u<p]. Its logit estimate is

```{math}
g_\eta=(\mathcal L(z^+)-\mathcal L(z^-))(u-\tfrac12).
```

```python
loss = bernoulli_render_loss(logits, evaluate, pairs=1, generator=rng)
loss.backward()
```

The callback unpacks hard bits, calls the mesh builder and renderer, and returns
a scalar loss. Each pair uses two renders. The returned value is the mean of
their losses. Its surrogate backward supplies ARM to the logits and preserves
ordinary continuous gradients from both evaluations. Detach the loss difference
inside the score term, not the ordinary rendered losses. If a callback has other
randomness, keep it fixed within a pair or make it independent of the topology
sampling. No visibility-dependent loss mask may silently discard the error of
deleted geometry. Target validity and loss normalization must not be controlled
by predicted alive bits.

Average **losses**, not images: E[L(I)] and L(E[I]) can have opposite gradients.
The latter can reward an unrealizable mixture of topologies. Likewise, a fixed
hard mesh cannot acquire these gradients merely by returning differentiable
positions alongside integer triangle indices.

The estimator remains valid after isolated-cut cleanup because cleanup is a
deterministic function of the sampled bits. Do not replace it with a detached
logit-dependent repair. A cut with no rendering consequence has zero expected
loss gradient; nearby views, geometry evidence and initialization matter.

## Native implementation and limitations

`pixel_grid_mesh` uses a native CUDA extension for topology realization, corner
fitting, output compaction, triangulation, feather construction, and backward.
`pixel_grid_mesh_ref` provides a tensor implementation for CPU/CUDA correctness
comparisons. The default path does not silently fall back to that implementation.

Native corner fans contain at most four cells. GPU prefix scans compute compact
vertex and triangle offsets. One small count transfer determines allocation
sizes; the native path is therefore not CUDA-graph static. Backward differentiates
positions, source UV reprojection, and strip geometry into depth, displacement,
intrinsics, and width. First-order gradients are supported; floating-point atomic
accumulation can differ in its last bits between runs.

The ARM estimator can have high variance on large grids. Increasing the pair
count trades additional evaluations for lower sampling variance. Geometry
regularization, loss design, optimization schedules, and convergence remain the
caller's responsibility. Absent cells must still contribute to the image loss;
masking evaluation by predicted visibility can hide missing geometry.

The renderer's per-triangle ordering and analytic continuous gradients are
unchanged. The builder does not add per-pixel depth sorting, global feather
intersection repair, or a geometry-validity constraint.

## Validation

The regression suite covers an independent local-topology oracle, cut tips and boundary cases,
native/reference geometry and gradients, finite differences, feathering, empty
meshes, stable face ordering, and exact expected-rendered-loss gradients for the
stochastic helper.

## References

- Deliot, Heitz, Belcour: [Transforming a Non-Differentiable Rasterizer into a
  Differentiable One with Stochastic Gradient Estimation](https://arxiv.org/abs/2404.09758).
- Yin, Zhou: [ARM: Augment-REINFORCE-Merge Gradient for Stochastic Binary
  Networks](https://arxiv.org/abs/1807.11143).
