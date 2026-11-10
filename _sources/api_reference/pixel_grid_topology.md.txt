# Pixel-grid topology and gradients

The mesh builder creates a triangle mesh from a regular grid of **cells centered on
source-image pixels**. Each live cell contributes a quad whose initial corners
are at pixel corners. Neighbor affinities determine which corner vertices are
shared.

Training minimizes the expected rendered loss of sampled meshes. At inference,
construct one deterministic mesh from the predicted decisions.

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

The function returns seven elements. Optional outputs are `None` when disabled.

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

The function operates on one grid per call. Call it per example
for a ragged batch.
Floating inputs share float32/float64 dtype and device; hard decisions are bool.
Depth, K, displacement, and widths must be finite and geometrically valid. The
displaced surface must remain in front of the source camera.

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
triangles. The reference function uses the same order.

`vertex_opacity` defines the boundary fade. Pass it as
`opacity=vertex_opacity` to the transparent renderer to multiply the appearance
alpha. None means an all-one multiplier.

Project `v` into the desired target camera before rendering: the transparent
renderer accepts target pixel XY and camera Z. Both UVs use 2D interpolation.
Use
[source-feature sampling](source_features.rst) to account for occlusion when
transferring features to the deformed mesh.

## Pixel and camera convention

Source pixel (x,y) has its center at integer coordinates. Its corners are
(x±0.5,y±0.5). Normalized texture UV is ((x+0.5)/W,(y+0.5)/H), without a vertical
flip. A K resized by sx,sy needs fx'=sx·fx and cx'=sx·(cx+0.5)−0.5, and likewise
for Y. The same convention must be used when resizing the source image and its camera.

Depth is camera Z; displacement is expressed in the same camera frame.
Apply world and placement transforms after building the mesh.

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
boundary edges share one **crack-tip vertex**. Interior cuts with both tips on one edge are removed by the isolated-cut rule.

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

The average is confined to each fan, including its shared crack tip.
Gradients propagate through depth, displacement, and K.
Averaging the shared corners can shift a cell center away from its individual
displacement proposal.

Each live quad emits two triangles. Automatic diagonal choice compares the two
diagonal midpoints to the requested displaced cell center in 3D, choosing the
smaller squared error and TL–BR on a tie. The decision is discrete. Pass sampled diagonal bits to learn it from
rendered loss. Regularize the predicted geometry to avoid degeneracies and folds.

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

Large widths, acute crack tips, nearby opposing boundaries, and folded
geometry can cause the extruded strips to overlap and accumulate opacity.
Keep strips narrow. Switching between miter and bevel joins is discrete.

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
ordinary continuous gradients from both evaluations. If the callback has other randomness, keep it fixed within a pair or
independent of topology sampling. Define target validity and loss normalization
independently of predicted alive bits, so missing geometry contributes to the loss.

Averaging losses, E[L(I)], and evaluating the loss of an averaged image,
L(E[I]), optimize different objectives. The latter can reward a mixture of
topologies that cannot be represented by a single mesh.

The estimator remains valid after isolated-cut cleanup because cleanup is a
deterministic function of the sampled bits. A cut with no rendering consequence
has zero expected
loss gradient; nearby views, geometry evidence and initialization matter.

## Supported inputs and limitations

`pixel_grid_mesh` accepts float32/float64 CUDA tensors. Its reference counterpart
also supports CPU tensors. CUDA graph capture is unsupported. First-order
gradients propagate from positions, reprojected UVs, and feather geometry to
depth, displacement, intrinsics, and feather width. Backward is not guaranteed
to be bitwise deterministic.

The ARM estimator can have high variance on large grids. Increasing the pair
count trades additional evaluations for lower sampling variance. Geometry
regularization, loss design, optimization schedules, and convergence remain the
caller's responsibility. Absent cells must still contribute to the image loss;
masking evaluation by predicted visibility can hide missing geometry.


## References

- Deliot, Heitz, Belcour: [Transforming a Non-Differentiable Rasterizer into a
  Differentiable One with Stochastic Gradient Estimation](https://arxiv.org/abs/2404.09758).
- Yin, Zhou: [ARM: Augment-REINFORCE-Merge Gradient for Stochastic Binary
  Networks](https://arxiv.org/abs/1807.11143).
