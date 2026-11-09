# Transparent triangle renderer: design and performance notes

## Purpose and scope

The transparent renderer is a fused, differentiable path for a specific kind
of scene: many small, potentially overlapping triangles whose opacity must be
composited in depth order. It was added because the normal DRTK pipeline
deliberately separates rasterization, barycentric interpolation, shading, and
edge-gradient estimation. That separation is useful for ordinary opaque mesh
rendering, but it cannot represent an arbitrary number of translucent
fragments per pixel without materializing and sorting a fragment list.

The new path keeps the entire per-fragment computation together:

1. bin triangles into small screen-space tiles;
2. sort each tile's triangle references by depth;
3. test the sorted candidates at every pixel in the tile;
4. apply DRTK's hard top-left coverage rule and compute perspective-correct
   attributes;
5. shade the fragment; and
6. immediately composite it into the final pixel.

No dense `[faces, height, width]` tensor and no per-pixel fragment buffer is
created. Memory is proportional to the triangle/tile intersection list rather
than to faces times pixels or to a fixed maximum number of layers per pixel.

This renderer consumes a triangle topology; it does **not** generate topology.
Quads can be supplied after a deterministic split into triangles. Topology
prediction, pixel-aligned resampling, vertex duplication at discontinuities,
and the decision to preserve or separate a foreground/background boundary are
upstream representation problems.

The implementation is not a volumetric shell or a ray marcher. For the close
novel-view use case, explicitly controlled edge opacity is cheaper and more
direct than pretending that a thin surface is a participating volume.

## The three appearance entry points

### Direct sampled-texture variant

`render_transparent` interpolates one UV coordinate and samples an RGB or RGBA
texture. Filtering supports nearest, bilinear (default), or bicubic, with
zeros, border (default), or reflection padding and fixed `align_corners=False`.
Activations follow interpolation: RGB defaults to `max(x, 0)` and alpha to
`1 - exp(-max(x, 0))`. An RGB texture has implicit alpha one. Appearance alpha
is multiplied by clamped interpolated material opacity before the fixed
fragment-alpha cap. For conventional preactivated RGBA, explicitly select
`rgb_activation="identity"` and `alpha_activation="clamp"`.

### Single- and dual-feature tiny-MLP variants

`render_transparent_mlp` samples one feature texture at one UV coordinate.
`render_transparent_mlp_two_textures` samples `texture0` at `uv0` and `texture1`
at independent `uv1`, then concatenates the vectors in that order before the
MLP. For a displaced-mesh workflow, one coordinate can follow the current mesh
and the other can retain the original pixel-aligned source location.

The two-texture API does not determine which source is visible or reliable.
Upstream code must gate samples using visibility and component provenance and
provide appropriate validity information to the appearance model. A hidden
chest vertex must not sample the foreground hand merely because both project
to the same current pixel. Independent UVs alone do not prevent this leakage.

The neural appearance model contract is intentionally narrow:

- exactly four affine layers by default, or five when the extra capacity is
  justified;
- every non-output layer has width 8;
- LeakyReLU follows every hidden affine layer;
- the final layer produces three RGB values and one alpha value, followed by
  the explicit appearance activations described below; and
- at most 32 total input feature channels are accepted.

There is no configurable wider variant. Width is fixed in the CUDA templates
so loops unroll and intermediate vectors remain small. A wider per-fragment
MLP quickly dominates a renderer because it is evaluated once for every
accepted translucent fragment, not once per output pixel.

If the two sampled feature vectors contain a total of `C` channels, the exact
affine multiply-accumulate count per accepted fragment is:

```text
4 affine layers: 8*C + 2*8*8 + 4*8 = 8*C + 160 MACs
5 affine layers: 8*C + 3*8*8 + 4*8 = 8*C + 224 MACs
```

For the common `C = 8` case, that is 224 MACs for four layers or 288 MACs for
five layers. The four-layer network has three width-8 hidden activations; the
five-layer network has four. Dual bilinear feature sampling performs `4*C`
scalar texel loads per accepted fragment before cache effects. By comparison,
the direct RGBA path performs 16 scalar texel loads.

Hard geometric coverage and material opacity deliberately remain outside the
MLP. The MLP alpha controls appearance-dependent translucency inside covered
geometry; interpolated vertex opacity remains an independent, interpretable
visibility control. An upstream model can therefore guarantee a zero-alpha
outer boundary even when the MLP predicts nonzero alpha.

MLP RGB defaults to `max(x + 0.5, 0)` (`shifted_relu`), and MLP alpha to
`1 - exp(-max(x + 0.5, 0))` (`shifted_relu_density`). Alternatives for RGB are
`shifted_identity` (`x + 0.5`), `sigmoid`, and `shifted_clamp`; alternatives for
alpha are `sigmoid`, `exp_density`, and `shifted_clamp`. Direct-texture RGB
alternatives are `identity`, `shifted_sigmoid` (`sigmoid(x - 0.5)`), and `clamp`;
direct-texture alpha alternatives are `shifted_sigmoid`, `exp_density`, and
`clamp`. `exp_density` computes `1 - exp(-exp(x))` with `x` capped at 15.

Appearance and material-opacity clamps, the density-input cap, and the upper
fragment-alpha cap use a unit subgradient at an exact endpoint and zero outside
the interval. ReLU still has zero gradient at zero, including the final
fragment-alpha lower bound. The reference uses explicit selections for these
clamps: fbcode and upstream PyTorch disagree on scalar-clamp endpoint gradients,
so delegating this convention to `Tensor.clamp` would make parity depend on the
PyTorch build. Endpoint tests use exactly representable barycentrics to avoid
crossing a clamp boundary through interpolation roundoff.

The post-interpolation ReLU is inspired by
[ReLU Fields: The Little Non-linearity That Could](https://arxiv.org/abs/2205.10824).
It allows an interpolated field to express a sharp zero set without requiring
each grid value to be nonnegative. The half-unit MLP shifts and opacity mapping
are renderer-specific choices. Density here is per-fragment optical thickness,
not a world-space density integrated along a ray. Material opacity remains
outside the MLP, so a zero-opacity boundary stays transparent.

### Minimal four-layer setup

The word "layer" counts affine transforms, including the final RGBA transform.
For two four-channel feature textures, a four-layer parameter set therefore
has these exact shapes:

```python
import torch
import drtk

weights = [
    torch.nn.Parameter(torch.randn(8, 8, device="cuda") * 0.02),
    torch.nn.Parameter(torch.randn(8, 8, device="cuda") * 0.02),
    torch.nn.Parameter(torch.randn(8, 8, device="cuda") * 0.02),
    torch.nn.Parameter(torch.randn(4, 8, device="cuda") * 0.02),
]
biases = [
    torch.nn.Parameter(torch.zeros(8, device="cuda")),
    torch.nn.Parameter(torch.zeros(8, device="cuda")),
    torch.nn.Parameter(torch.zeros(8, device="cuda")),
    torch.nn.Parameter(torch.zeros(4, device="cuda")),
]

color, alpha, depth = drtk.render_transparent_mlp_two_textures(
    projected_vertices,
    triangle_indices,
    displaced_uv,
    original_pixel_aligned_uv,
    displaced_features,
    original_features,
    weights,
    biases,
    height,
    width,
    opacity=vertex_opacity,
)
```

A five-layer setup inserts one additional `(8, 8)` weight and length-8 bias
immediately before the `(4, 8)` RGBA output transform. The implementation
rejects every other depth or hidden width instead of silently taking a slow
generic path.

## DRTK operator and autograd integration

The optimized API follows the same ownership boundary as the rest of DRTK.
Python performs lightweight validation, broadcasting, dtype normalization, and
default construction, then calls one of two native operators:

```text
transparent_render_ext::render_with_texture
transparent_render_ext::render_with_mlp
```

Both names denote complete transparent-rendering operations. The suffix only
selects the per-fragment appearance calculation performed inside the tiled
render kernel; neither operator exposes texture sampling or MLP evaluation as
a separate stage.

CUDA dispatch, autocast handling, saved forward state, and backward dispatch
all live in `transparent_render_module.cpp`. The CUDA forward/backward entry
points are not registered as Python-visible operators; they are implementation
details called by the native dispatcher and C++ `torch::autograd::Function`
classes. Consequently, Python never selects an inference path versus an
autograd path and never handles tile offsets or sorted face lists.

The MLP schema accepts `Tensor[]` weights and biases. PyTorch's C++ custom
function machinery expands each tensor-list element into its own autograd edge,
so the one native MLP operator supports both allowed depths while returning an
independent gradient for every weight and bias. Separate four- and five-layer
operator schemas are unnecessary.

The fused operators are CUDA-only. The explicit `render_transparent_ref`,
`render_transparent_mlp_ref`, and `render_transparent_mlp_two_textures_ref`
functions remain available as slow, pure-PyTorch
CPU correctness oracles. Calling an optimized entry point with CPU tensors
produces a direct error that points to the appropriate reference function.

## Coordinate, depth, and texture conventions

- Vertex tensors have shape `[N, V, 3]`.
- `x` and `y` are already projected DRTK pixel coordinates. Integer coordinates
  are pixel sample locations.
- `z` is positive camera depth; smaller values are nearer.
- Faces are int32 triangles with shape `[F, 3]`, `[1, F, 3]`, or `[N, F, 3]`.
- UV coordinates are normalized to `[0, 1]`.
- Sampling matches `torch.nn.functional.grid_sample`: nearest, bilinear, or
  bicubic filtering; zeros, border, or reflection padding; fixed
  `align_corners=False`. Defaults are bilinear filtering and border padding.
- Textures may have a batch of one, which broadcasts, or the same batch as the
  vertices.
- A triangle's sort depth is the arithmetic mean of its three vertex depths.
- Equal-depth triangles retain face-index order through a stable radix sort.

Triangles with invalid indices, duplicate vertex indices, non-finite vertices,
near-zero projected area, non-positive depth, or no overlap with the image are
discarded before tile-list emission.

## Forward pipeline

### 1. Screen-space bounds

For each triangle, the kernel computes its exact projected bounding box. The
bounds are converted to a rectangle of 4x4 tiles. A prefix sum of
per-triangle tile counts allocates a compact triangle/tile intersection list.

### 2. Stable tile/depth sort

Every intersection emits a 64-bit key:

```text
high 32 bits: flattened (batch, tile_y, tile_x) identifier
low  32 bits: positive float depth bit pattern
```

Positive IEEE-754 float bit patterns preserve numerical order. CUB performs a
stable radix sort of key/face pairs, so face order deterministically resolves
equal depths. A binary search for each tile boundary then creates an offset
array into the sorted face list.

This design follows the useful part of Gaussian-splatting renderers: duplicate
primitives into the tiles they touch, globally sort compact keys, and let one
pixel program consume a contiguous sorted range. The primitive test and
appearance evaluation are triangle-specific rather than Gaussian-specific.

### 3. One 4x4 CUDA block per tile

Each CUDA block owns one image tile and each thread owns one pixel. Candidate
triangles are loaded into shared memory in small batches. All pixels in the
tile traverse the same front-to-back candidate order, but reject candidates
that do not cover that pixel under the hard rasterization rule.

Four-pixel tiles were selected empirically for the current workload. Large
tiles amortize list and launch overhead but force every pixel to test many
triangles that only touch another part of the tile. Tiny tiles reduce candidate
tests but underfill the GPU and increase bin-list overhead.

### 4. Hard DRTK coverage, feathering, and transparent edge gradients

Coverage matches DRTK rasterization rather than defining a second soft
rasterizer. Edge functions use canonical vertex-index endpoint ordering, then
the top-left rule assigns a pixel exactly on a shared edge to one adjacent
triangle. Both windings are supported. This matters for transparency: if both
triangles owned the shared edge, that pixel would be composited twice and
would become spuriously opaque.

Coverage stays binary in forward. The renderer does not synthesize a global
`feather` distance: upstream geometry and vertex opacity still control the
width of soft transitions. A strip whose outside vertices have zero opacity
receives ordinary interior derivatives, with zero additional boundary term at
its outside edge.

All three CUDA APIs enable the boundary estimator by default; set
`edge_gradients=False` to disable it. The estimator adds a
transparent extension of [Rasterized Edge Gradients](https://arxiv.org/abs/2405.02508)
to geometry backward. Inference, forward values, and the smooth appearance /
interpolation derivatives are unchanged. It is a filtered boundary estimator,
not the almost-everywhere-zero derivative of point-sampled hard coverage.

For a virtual fragment with edge-trace color `c` and opacity `a`, let `F` be
the accumulated premultiplied foreground, `T` its remaining transmittance,
and `B` the suffix color including the background with unit incident light:

```text
C_without = F + T B
C_with    = F + T [a c + (1-a) B]
Delta C   = T a (c-B)
Delta A   = a T_total_without
```

`F` cancels. The neighbor needs its own prefix/suffix state at the fragment's
**mean-triangle-depth rank**, including the stable face-ID tie break. Final
RGB/alpha do not determine that state. Queries follow exactly the existing
forward ordering, even when actual depth at a pixel would suggest otherwise.
No per-pixel sort or triangle-intersection estimator is introduced.

The loss-weighted jump can be evaluated with scalar suffix adjoints. Initialize
`h = dot(g_rgb, background) - g_alpha` and `d = 0`, then traverse the ordered
list in reverse. Here `h` is the suffix on the negative side of the indexed
edge and `d` is the positive-minus-negative suffix difference:

```text
ordinary fragment on both sides:
  h = a dot(g_rgb, c) + (1-a) h
  d = (1-a) d
boundary fragment present only on positive side:
  d = a [dot(g_rgb, c)-h] + (1-a) d
boundary fragment present only on negative side:
  d = d + a [h-dot(g_rgb, c)]
  h = a dot(g_rgb, c) + (1-a) h
```

The boundary fragment uses perspective-correct attributes on the actual edge,
not an interior sample. These updates evaluate the whole incident-face bundle,
including nonlinear alpha interactions, without subtracting two large final
colors. Each of the two neighboring pixels uses its own incoming loss gradient
and its own non-bundle fragments; their jumps are averaged. Horizontal pixel
pairs scatter only X motion, vertical pairs only Y motion, with screen-linear
edge endpoint weights. Z already affects projection upstream and smooth
interpolation here; the boundary term adds no derivative of sorting.

Implementation:

1. Build GPU keys for undirected indexed edges, including batch identity, and
   radix-sort their incident faces. Invalid/rejected geometry is ignored.
2. Compact surviving groups on GPU. Two incident faces on opposite projected
   sides form a continuous tessellation edge and are suppressed. Shared folded
   edges are retained as one bundle. An edge whose two material endpoints are
   nonpositive is rejected before appearance evaluation.
3. A bounded persistent warp pool traverses the active groups. Each warp samples
   horizontal/vertical grid crossings along an edge; zero sampled appearance
   alpha also exits before a query.
4. For each crossing, merge the incident faces with each neighbor's saved tile
   list in **the same triangle order**, replay the scalar compositing jump, and
   atomically add the boundary force to the two endpoint X/Y gradients.

There is no host readback of the active count, no dense fragment tape, and no
additional inference work. Current queries replay shading at each boundary
pair; queries that touch the same pixel are not yet coalesced. This bounds
implementation complexity but leaves a performance opportunity in dense
triangle soup or deep overlap. Indexed topology avoids treating ordinary
triangulation as a large set of independent boundaries.

The estimator suppresses continuous shared tessellation edges by convention;
it does not differentiate artificial seams caused by changing mean-depth rank
across neighboring faces. Topology changes, rank swaps, per-pixel depth
intersections, and image-border clipping remain outside its scope. Duplicate
vertices are independent; representing a jointly moving shared edge requires
shared indices. Pixel-aligned crossings select the neighboring pair according to the same
canonical top-left ownership as forward; edge endpoints use half-open
grid-line intervals. The operator does not promise exact derivatives of
arbitrary spatial reconstruction filters or texture discontinuities.

### 5. Perspective-correct interpolation

Screen-space barycentrics `s_i` are converted using vertex depth:

```text
q_i = s_i / z_i
b_i = q_i / sum_j(q_j)
```

The `b_i` weights interpolate UVs, reference UVs, and material opacity. The
backward pass differentiates through this conversion, including depth `z`.

### 6. Front-to-back compositing

For fragment `k`, first compute clamped interpolated material opacity `m_k`.
The direct-texture path uses activated sampled texture alpha for `p_k` (or one
for RGB textures), while the MLP path uses its activated alpha output:

```text
m_k = clamp(sum_i(b_i * vertex_opacity_i), 0, 1)
a_k = min(m_k * p_k, 0.999)
```

Let `c_k` be fragment RGB and `T_k` be transmittance before the fragment:

```text
C_(k+1) = C_k + T_k * a_k * c_k
T_(k+1) = T_k * (1 - a_k)
```

After all fragments, the background contributes `T_final * background`, and
the returned foreground alpha is accumulated directly as `sum_k(T_k * a_k)`. This equals `1 - T_final` mathematically while avoiding subtraction cancellation at small opacity. With a zero/default background, returned
RGB is premultiplied by accumulated alpha. With an explicit background,
returned RGB is the fully composited color. This is the same output convention
used by gsplat's front-to-back renderer.

The fragment-alpha cap is fixed internally at 0.999 and is not a public setting. In inference, an optional
`transmittance_epsilon` can terminate traversal once the remaining
transmittance is negligible. Early termination is rejected when gradients are
enabled because it is a discrete approximation.

## Depth output

All native and reference APIs return `(rgb, alpha, depth)`. Fragment depth is
perspective-correct interpolated camera Z. Four modes share the RGB traversal
order, with no per-pixel depth sorting:

- `expected`: `sum(T * alpha * Z) + T_final * background_depth` (default).
- `accumulated`: foreground `sum(T * alpha * Z)` without a background term.
- `foreground_mean`: foreground sum divided by its accumulated opacity, only
  when opacity exceeds `depth_alpha_threshold`; otherwise background depth.
- `median`: the first fragment where cumulative opacity reaches 0.5, or
  background depth if no foreground fragment crosses it.

Background depth is a finite positive camera-space value, default 1000.
Expected depth treats the background as an opaque final layer. Foreground
alpha remains the alpha output; it is not replaced by one. The mean validity
threshold defaults to 1e-6. Invalid pixels never divide by small opacity and
have zero depth derivative. Background and validity settings are constants.

Depth accumulation is fused into the existing fragment loop. Median backward
uses the face selected in forward, avoiding changes at an exact threshold due
to recomputing floating-point transmittance in reverse. Smooth derivatives
include the depth attribute itself and its perspective interpolation. Expected,
accumulated, and valid mean depth also differentiate compositing weights.
Median selection and mean validity remain discrete.

Depth boundary queries replay the two hypothetical sides in forward order and
compare the requested final depth values, preserving the renderer's rank and
shared-edge bundling. This supports nonlinear mean and median outputs without
materializing fragment lists. These depth-specific queries run only where the
upstream depth gradient is nonzero. The existing color/alpha boundary term is
retained.

## Backward pipeline

For backward, forward saves a per-pixel transmittance mantissa and integer
power-of-two exponent, tile offsets, and sorted face identifiers; it does not
save a fragment buffer. The extra transmittance state costs eight bytes per
pixel, is used only when gradients are needed, and does not change the forward
color or alpha calculation. Inference does not allocate or update this state. The backward traverses each tile list in
reverse and recomputes hard coverage, barycentrics, texture samples, and the tiny
MLP.

Given transmittance after a fragment, the earlier value is recovered as:

```text
T_before = T_after / (1 - a)
```

The fixed 0.999 fragment cap keeps the denominator nonzero. Saving only final
alpha is insufficient: three opaque layers can round `1 - T` to exactly one,
erasing `T` and its reconstructed RGB gradients. Even saving an ordinary
float32 `T` is insufficient for deep overlap because the product can underflow.
The saved product is therefore rescaled by exact powers of two during forward
and reverse traversal; backward reconstructs the prefix transmittance from
that state, not from `1 - output_alpha`. No per-fragment logarithms or fragment
buffers are required.

The material-opacity clamp includes its endpoints in the differentiable range,
matching the explicit reference convention above. In particular, learnable
opacity initialized at exactly one can decrease. The exact fragment-alpha cap
also retains the incoming gradient; values above the cap have zero alpha
derivative. At zero fragment alpha, ReLU's zero derivative still applies.

The CUDA backward returns gradients for:

- projected vertex `x`, `y`, and `z`;
- current UV and reference UV;
- both feature textures or the RGB/RGBA texture;
- per-vertex material opacity;
- background RGB; and
- every tiny-MLP weight and bias.

Tiny-MLP parameter gradients first accumulate in shared memory per tile and
are flushed once per parameter to global memory. This substantially reduces
global atomic traffic. Parameter-gradient accumulation is still the dominant
cost of MLP backward; inference does not pay this cost.

### Deliberately discrete terms

The following choices do not receive gradients:

- face indices and topology;
- triangle-to-tile membership;
- triangle rejection and degeneracy tests;
- center-depth ordering and equal-depth tie order;
- the sign selected for triangle winding;
- exact hard coverage and top-left edge ownership (the boundary term
  differentiates a filtered surrogate instead); and
- inference-only early termination.

Smooth gradients differentiate shading and compositing while visibility order
and candidate membership are fixed. The boundary estimator adds
coverage-motion terms while preserving that same order.

## Performance model

Wall-clock time is governed more by candidate and accepted fragment counts
than by triangle count alone. Useful counters are:

```text
tile_intersections = sum(number of tiles touched by each triangle bound)
candidate_tests = sum_over_tiles(tile_face_count * pixels_in_tile)
accepted_fragments = candidates accepted by hard top-left coverage
```

The benchmark script reports `candidate_tests / output_pixels`. Accepted
fragments are the subset that incur barycentric interpolation, texture loads,
MLP evaluation, and compositing.

Use:

```bash
python test/transparent_render_benchmark.py \
  --resolution 128 \
  --feature-channels-per-texture 4 \
  --layers 4
```

The reference benchmark scene is a 128x128 pixel-aligned vertex grid with
16,384 vertices and 32,258 one-pixel triangles, two four-channel feature
textures, and width-8 RGBA shading. It is intentionally
triangle-dense and exposes tile-list overhead.

One RTX 3090 measurement with PyTorch 2.12/CUDA 12.6, 20 warm-up iterations,
and 100 measured iterations was:

| Phase | Time |
|---|---:|
| RGB/RGBA forward, including bin/sort | 0.17 to 0.18 ms |
| four-layer width-8 RGBA MLP forward, including bin/sort | about 0.25 ms |
| five-layer width-8 RGBA MLP forward, including bin/sort | about 0.24 ms |
| RGB/RGBA backward with saved bins | 0.16 to 0.17 ms |
| four-layer width-8 RGBA MLP backward with saved bins | 1.76 ms |
| five-layer width-8 RGBA MLP backward with saved bins | 2.00 ms |

These are implementation sanity numbers, **not headset projections**. The MLP
backward is expensive because every accepted fragment contributes gradients to
the same small parameter set and therefore requires atomic accumulation. A
headset renderer only needs the forward path.

The exact hard-bounds candidate counts for possible tile sizes on that same
scene are:

| Tile size | Candidate tests per pixel |
|---:|---:|
| 16 | 561.13 |
| 8 | 157.53 |
| 4 | 48.76 |
| 2 | 17.63 |

Tile size 2 minimizes primitive tests but also reduces occupancy and increases
bin-list overhead. The current kernel retains tile size 4 based on the earlier
timing sweep; the timing sweep should be repeated before changing it because
candidate count alone is not sufficient. Exact times vary with clocks, driver,
PyTorch allocator state, and feature-cache behavior; candidate counts and MAC
counts are more portable.

### Reasoning about lower-tier and headset GPUs

The CUDA kernel is a training/reference implementation, not a Quest binary.
For a Vulkan/mobile implementation, preserve the algorithmic structure but
remeasure tile size and data layout:

- reuse a GPU radix-sort and tile-range construction path, such as the one
  already used for sorted Gaussian primitives;
- pack width-8 activations and weights into two `vec4` groups;
- evaluate the 224-MAC four-layer network in mediump/FP16 when quality permits;
- store features in a texture format that provides bilinear filtering and
  cache locality;
- specialize the direct-texture and MLP fragment evaluators so unused work
  disappears; and
- expose candidate tests, accepted fragments, texture reads, and MLP MACs in
  device profiling rather than extrapolating from desktop TFLOPS.

With a pixel-aligned grid, preserving approximately the same number of tiny
triangles per pixel makes candidate and shading work grow roughly linearly
with output pixel count. A move from 128x128 to 1024x1024 has 64 times as many
pixels; the 3090 timing cannot simply be multiplied and transferred to a
mobile GPU, but it makes clear that full-resolution per-pixel topology needs
aggressive culling, coarser geometry, lower rendering resolution, or all
three. The intended close-view correction should exploit its narrow view
range rather than paying for general novel-view geometry.

## Limitations and failure modes

### Center sorting is an approximation

One scalar depth orders an entire triangle. Large triangles that cross in
depth, intersect, or have strong depth variation can require a different order
at different pixels. Center sorting cannot represent that. The approximation
is best for small triangles, modest displacement, and the very-close-view
regime. Subdivision is the simplest mitigation.

### There is no clipping stage

Vertices must already have positive depth. Faces crossing the near plane are
rejected rather than clipped. Upstream projection/meshing should clip or avoid
those faces.

### Explicit feather geometry can reveal too many layers

Every overlapping triangle in an authored feather strip contributes alpha. If
many nearly coincident strips overlap, they can become too opaque and
expensive. Topology and opacity regularization should avoid duplicate surfaces
except where they are semantically intended.

### Topology gradients are out of scope

The renderer differentiates vertex positions and attributes for fixed faces.
It cannot differentiate an edge split, merge, face insertion, face deletion,
or foreground/background connectivity decision. A transformer that predicts
topology needs a separate discrete or relaxed representation strategy before
calling this renderer.

### Determinism and floating-point atomics

Face ordering is stable, but backward texture, vertex, and parameter gradients
use floating-point atomics. Their last few bits can vary with GPU scheduling.
Tests use tight numerical tolerances against the pure-PyTorch reference rather
than bitwise equality.

## Alternatives considered

### Keep the normal DRTK stages separate

This would require either a fixed number of fragments per pixel or a large
materialized fragment list between stages. Both defeat the main memory and
compositing requirements.

### Expose a fully general fragment callback

A dynamic callback would prevent compile-time specialization, complicate
autograd, and make mobile deployment unpredictable. The direct texture and
dual-feature width-8 MLP paths cover the current use cases while keeping exact
operation counts.

### Keep the MLP RGB-only

An RGB-only head would make all appearance-dependent transparency flow through
the separate vertex-opacity field. That is simpler, but it prevents sampled
features from expressing local translucency at fragment resolution. The
implemented four-channel head uses the configurable RGB/alpha activations
described above, while clamped interpolated vertex opacity remains an independent multiplicative
control. A zero-opacity outer strip therefore remains guaranteed transparent.

### Model a thin volumetric shell

Ray marching a shell adds multiple samples and a thickness parameter without
solving the upstream topology decision. For tiny viewpoint corrections, a
transformer-controlled mesh plus explicitly authored opacity feathering is the
smaller and more controllable model.

## Verification strategy

The implementation has two independent correctness layers:

1. pure-PyTorch reference functions, suitable for CPU tests and `gradcheck` on
   small scenes; and
2. fused CUDA operators with native C++ autograd compared against those
   references.

Tests cover stable depth compositing, both windings, top-left ownership of
shared edges, premultiplied output, background compositing, opacity clamps,
interior geometry gradients, texture gradcheck, four- and five-layer RGBA
MLPs, rejection of non-width-8 parameter shapes, and CUDA parity for outputs
and all returned gradients. Regression cases include two/three/six/32 opaque
layers, float32 transmittance underflow, both MLP depths and texture variants,
material opacity at and outside clamp endpoints, and alpha exactly at the
fragment cap. Transparent boundary tests additionally check independent
area-integrated derivatives, neighbor-specific occlusion, mean-depth order and
tie order, tile crossings, shared-edge cancellation, folded and nonmanifold
bundles, zero-alpha traces with positive interior alpha, both MLP depths and
variants, batch separation, and empty/rejected geometry. The unchanged forward
outputs and non-geometry gradients are checked explicitly.

All transparent-render tests live alongside the existing DRTK GPU tests in
`fbcode/frl/codec_avatar/tre/projects/drtk_tests_gpu/src/transparent_render_tests.py`.
The single `transparent_render_tests` target uses the package's
`python_unittest_remote_gpu` configuration (`gpu-remote-execution`,
`xrcia_gpu_tests`, `par_style="xar"`). It runs both the CPU reference checks and
CUDA parity/regression checks. Missing CUDA fails the GPU tests; there is no
conditional skip or environment override.

```bash
buck test @fbcode/mode/opt \
  fbcode//frl/codec_avatar/tre/projects/drtk_tests_gpu:transparent_render_tests \
  -c test.threads=1
```

## Change inventory

The implementation is intentionally split by responsibility:

- `drtk/transparent_render.py`: public functional API, reference
  implementations, and lightweight input normalization;
- `drtk/transparent_render_ext.pyi`: declarations for the two public native
  operators;
- `src/transparent_render/transparent_render_module.cpp`: Torch operator
  schemas, CUDA/CPU/autocast dispatch, C++ autograd nodes, saved state, and
  backward routing;
- `src/transparent_render/transparent_render_kernel.h`: C++ declarations;
- `src/transparent_render/transparent_render_kernel.cu`: binning, stable sort,
  fragment evaluation, reverse compositing, and all CUDA gradients;
- `test/transparent_render_benchmark.py`: reproducible workload and timings;
- `fbcode/frl/codec_avatar/tre/projects/drtk_tests_gpu/src/transparent_render_tests.py`:
  CPU reference semantics plus CUDA
  fused/reference output and gradient parity;
- `test/transparent_render_demo.py`: runnable 1024 x 1024 three-layer compositing
  example that optimizes foreground vertex geometry and opacity while keeping
  UVs and textures fixed;
- `docs/source/tutorials/DRTK_Tutorial_transparent_render.rst`: demo walkthrough
  and asset-regeneration instructions;
- Buck/ARVR target definitions: internal CUDA builds; and
- `github_setup.py` and `dgx_setup.py`: standalone extension builds.

This inventory is also a review checklist: a future change to appearance inputs,
MLP shape, compositing semantics, or saved autograd state normally requires
coordinated updates in the Python API, native schema/type stub, C++ autograd,
CUDA validation, reference oracle, parity tests, benchmark, and this design
note.
