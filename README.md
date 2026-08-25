# PenroseDebruim

PenroseDebruim directly regresses the ordered output of a De Bruijn pentagrid
construction from five phase values `γ` and one independent scalar
`θoffset`. The algorithm maps `θoffset` to the global grid rotation `θ0`.
It predicts rhombus centers, angles, and colors in one forward pass. It is
intentionally not a diffusion model.

For one configured `kmax`, the generator returns

```text
N_kmax = 10 * (2*kmax + 1)^2
```

rhombi in fixed family-pair, line-i, line-j order. `kmax` may be any
nonnegative integer and defaults to `1`; at `kmax=0`, the model predicts the ten
family-pair rhombi. Every model checkpoint is therefore tied to one `kmax`.

## Method

- Input: `γ` contains five independent phase values in `[0,1)`, while
  `θoffset ~ Uniform[0,1]` is passed separately. The generator maps
  `θ0 = (θoffset - 0.5) * pi/5`, covering `[-pi/10,pi/10]`, and adds `θ0` to
  all five fifth-root direction angles. The neutral default is `θoffset=0.5`.
- Generator output: ordered raw `(x, y, angle)` for every rhombus, where
  `angle` is in radians, plus its binary color. The complete patch is not
  translated or mean-centered.
- Scaling: centers are divided by
  `sqrt((25/6)*kmax*(kmax+1) + 5.65)`. Angles are multiplied by `sqrt(3)/pi`
  to obtain the scaled angles used by the network. In this project, `angles`
  always means radians and `scaled_angles` means the unit-second-moment values
  with half-period `sqrt(3)`.
- Model: one `DebruijnTransformer`, conditioned on separate `γ` and `θoffset`,
  with a pre-norm TransformerEncoder, learned `N_kmax x 4` tile latents, and
  four global tokens. The tile latents represent
  `(x, y, scaled_angle, scaled_color)` and enter `d_model` through one shared
  bias-free linear embedding. Final normalized tile states are decoded directly
  through that same matrix's transpose; there is no separate output MLP.
  Colors are encoded as `-1` and `+1`.
- Objective: scaled Cartesian MSE plus circular scaled-angle MSE. The
  scaled-angle difference is wrapped to `[-sqrt(3),sqrt(3))` before squaring,
  keeping its scale comparable to `x` and `y`. Color uses MSE on its `-1/+1`
  encoding. The four output components receive equal per-component weight.
  There is no OT, matching, corruption, diffusion, or reverse process.

Writing `Lxy` for MSE averaged over scaled `x,y`, `Lsa` for wrapped
scaled-angle MSE, and `Lc` for scaled-color MSE, training minimizes:

```text
loss = (2*Lxy + Lsa + Lc) / 4
```

The training log and checkpoints record total loss, `xy_mse`,
`scaled_angle_mse`, `color_mse`, color accuracy, gradient norm, and learning
rate.

## Data flow

```text
γ[5], θoffset[1]
        │
        ├── De Bruijn generator ──> x, y, angles[radians], colors[0/1]
        │                              │
        │                              └──> x/xy_scale,
        │                                   scaled_angles=angles*sqrt(3)/pi,
        │                                   scaled_colors=2*colors-1
        │
        └── DebruijnTransformer ──> predicted
                                    (scaled_x, scaled_y,
                                     scaled_angle, scaled_color)
```

Generation and SVG vertex calculations use angles in radians. Only network
targets, predictions, and the wrapped angle loss use scaled angles. Sampling
unscales predicted angles back to radians before rendering.

## Tied geometry embedding

Let `W` have shape `d_model x 4`. There is one learned four-value latent per
ordered tile and one shared, bias-free geometry embedding:

```text
tile_latent (x, y, scaled_angle, scaled_color) --Wᵀ--> d_model
final normalized transformer state                  --W--> 4 predictions
```

The implementation creates `geometry_embedding = Linear(4, d_model,
bias=False)` and decodes with `geometry_embedding.weight.T`. There is no
separate decoder parameter or output MLP, so gradients from both directions
update the same matrix and the learned tile latents expose where the network
begins.

The readable NumPy reference lives in `debruijn_np.py`; the CUDA-capable
PyTorch training implementation lives in `debruijn.py`. The project has no
runtime dependency on PenroseSpur or PenroseDiffusion. SVG rendering is
isolated in `svg.py`.

## Local commands

Use the shared Diffusion workspace environment:

```bash
~/.aivenv/bin/python train.py
```

Useful strict overrides include:

```bash
~/.aivenv/bin/python train.py --kmax 3 --batch-size 32 --epochs 20
~/.aivenv/bin/python train.py -m d_model=64 -m num_layers=4 -t learning_rate=0.0005
~/.aivenv/bin/python train.py --no-rotation
```

Unknown sections and keys are rejected. The resolved configuration is printed
before training. Defaults include `kmax=1`, `d_model=128`, eight transformer
layers, eight heads, four global tokens, and dropout `0.0`.

With `--no-rotation`, the target generator always receives `θoffset=0.5`, so
its derived `θ0` is zero. The transformer still receives a random uniform
`θoffset`; it therefore learns that its sixth input is irrelevant in this
training mode.

## Resume

Checkpoint names contain a stable UTC run identifier and zero-padded epoch.
Only the newest and best-average-loss checkpoints are retained. Resume preserves
the identifier, output directory, RNG streams, optimizer, scheduler, and epoch
number:

```bash
~/.aivenv/bin/python train.py --resume outputs/debruim_0824_0700_128x8_kmax2_mse/checkpoints/debruim_0824_0700_128x8_kmax2_mse_e010.pt -t num_epochs=151
```

Architecture, `kmax`, rotation mode, tile order, and scaling are immutable on
resume. Checkpoints also declare the six-dimensional condition, four-dimensional
target, `-1/+1` color encoding, tied geometry embedding, angle/coordinate
scales, RNG states, and WandB run identity.

## Sampling and SVG comparison

Sampling is one model forward pass. The SVG overlays predicted and exact
rhombi, then draws one center-to-center correspondence line for every ordered
tile. Predicted rhombi use predicted colors; exact rhombi use generated colors:

```bash
~/.aivenv/bin/python sampler.py --resume PATH/TO/CHECKPOINT.pt --sample-n 4 --output-directory samples
~/.aivenv/bin/python sampler.py --resume PATH/TO/CHECKPOINT.pt --gamma 0.1 0.2 0.3 0.4 0.5 --theta-offset 0.75 --output-directory samples
```

Training also writes one such SVG for every completed epoch under the run's
`svg/` directory, plus a second `_latents.svg` rendering of the learned
four-value tile latents. For rendering, `scaled_color` is mapped back with
`color=(scaled_color+1)/2`. The rhombus top angle varies continuously as
`(3-2*color)*pi/5`, and the fill color is interpolated continuously between
the two endpoint colors. Colors outside `[0,1]` are intentionally left
unclipped so latent and prediction excursions remain visible. SVGs are never
uploaded to WandB.

## Scaling

The shared raw-center scale is:

```text
xy_scale(kmax) = sqrt((25/6) * kmax * (kmax + 1) + 5.65)
```

The line-index variance is `kmax*(kmax+1)/3`. Averaging the ten pentagrid
family pairs contributes a factor of two, while the De Bruijn dual projection
contributes `(5/2)^2`, giving the growing term
`(25/6)*kmax*(kmax+1)`.

This leading term is exact but is not the complete variance. `γ`-dependent
patch translation, ceiling quantization, rhombus-center offsets, and their
covariances contribute the residual. A seeded 100,000-condition raw-output sweep
for every `kmax=0..5` calibrated the simplest common residual to `5.65`.
Across those six values, the formula's largest relative scale error is about
`0.03%` (at `kmax=0`) and is below `0.01%` for `kmax=1..5`.

Reproduce the raw sweep without per-patch mean subtraction using:

```bash
~/.aivenv/bin/python tests/scaling/analyze_raw_scales.py \
  --samples 100000 --seed 20260823 --kmax 0 1 2 3 4 5
```

The CSV output is analytical evidence only; runtime scaling is computed from
the formula and has no CSV dependency. See `docs/xy_scale_derivation.tex` and
`docs/xy_scale_derivation.pdf` for the exact derivation and calibrated terms.
The derivation uses `K` for the maximum line index,
`k_i,k_j` for selected indices, and bold `k` for the random line index.

The CSV also records raw wrapped-angle statistics in radians before the
`sqrt(3)/pi` training multiplier. Under the equal-direction-count assumption,
uniform `θoffset` makes `θ0` fill the gaps between the ten original `pi/5` directions,
producing a uniform angle over `[-pi,pi)`. Consequently the arithmetic mean is
zero, the second moment is `pi^2/3`, and the existing `sqrt(3)/pi` multiplier
gives unit second moment.

`tests/scaling/angle_histogram.csv` records twenty equal-width `pi/10`
intervals. Each interval should contain approximately 5% of generated angles.

To verify the rotation directly and write two SVGs with identical phase
offsets but different `θoffset` values:

```bash
~/.aivenv/bin/python test_theta0.py
```

This writes `tests/theta0/thetaoffset_0.25.svg` and
`tests/theta0/thetaoffset_0.75.svg`.

## Test

```bash
~/.aivenv/bin/python test_smoke.py
```

The smoke test explicitly checks raw uncentered `kmax=0` NumPy/PyTorch parity,
its ten-tile shape, the calibrated formula through `kmax=5`, a finite
optimizer step, wrapped errors at radian/normalized/scaled-angle units,
`-1/+1` color targets, predicted/exact SVG colors, tied embedding transpose
decoding and shared gradients, ten SVG correspondence lines, checkpoint
retention, RNG restoration, sampling, and resume in a fresh process.
It retains representative black-background outputs at
`tests/smoke/comparison.svg`, `tests/smoke/training.svg`, and
`tests/smoke/sample.svg`, plus `tests/smoke/latents.svg`, instead of deleting
every SVG with its temporary run.

## Cloud launcher

The cloud launcher packages both generator implementations and runs local
smoke and θoffset checks before training:

```bash
cd ../at-cloud-scripts
DRY_RUN=1 RUN_MODE=smoke bash scripts/launch_penrose_debruim_kmax1.sh
RUN_MODE=smoke bash scripts/launch_penrose_debruim_kmax1.sh
RUN_MODE=full bash scripts/launch_penrose_debruim_kmax1.sh
```

Set `NO_ROTATION=true` to launch the rotation-invariant objective. Cloud
outputs contain logs, epoch SVGs, retained checkpoints, and a `COMPLETE`
sentinel after successful training.

## Deliberate omissions

There is no PenroseSpur import, dataset file/cache, diffuser, OT or permutation
matching, reverse solver, lattice metric, alternate model,
JAX/XLA/TPU path, or WandB artifact logging.
