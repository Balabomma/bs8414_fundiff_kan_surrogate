# bs8414_fundiff_kan_surrogate — KAN-FunDiff slice-field surrogate

A generative surrogate for the **BS 8414-1** large-scale facade fire test with
**KAN B-spline conditioning**. It produces the full 2D temperature field on the
standard slice planes — **181 timesteps × 128 (vertical Z) × 64 (horizontal)** per
plane — for any construction build-up, in seconds rather than the 12–27 hours the
FDS simulation takes.

Built on **FunDiff** (Wang, Dou, Shan, Liu & Lu, *"FunDiff: diffusion models over
function spaces for physics-informed generative modeling"*, Nature Communications
**17**:5749, 2026,
[doi:10.1038/s41467-026-72292-0](https://doi.org/10.1038/s41467-026-72292-0)).

One "function" is the **whole spatiotemporal field `T(t, y, z)` for a single
(simulation, slice)**, so the decoder gives arbitrary temporal *and* spatial
super-resolution and the diffusion stage works in a compact latent space rather
than on 1.48 M grid points.

---

## The corpus

The **Part1 geometry-variant corpus**, `D:\Bs8414_05052026\Part1\_completed`.

| Axis | Levels |
|---|---|
| Cladding | 12 systems (8 generic + 4 DCLG references) |
| Insulation | 5 products (MW, MWBC, PF, PIR, WC) |
| **Geometry** | **8** — the observed combinations of three construction modifiers |

HRR and mesh size are constants across the batch (HRRPUA = 2333.3 kW/m²,
dx = 0.10 m, T_END = 1800 s), so they are not inputs.

- 186 completed simulations → **184 usable**; two excluded by name in
  `config_part1.EXCLUDED_CHIDS`, each with its reason recorded.
- **Hash split: 141 train / 20 valid / 23 test** (`PART1_SPLIT=hash`, default).
  `PART1_SPLIT=system` keeps all 8 geometry variants of a build-up together and
  measures generalisation to an unseen system — a different, **non-comparable**
  protocol. Always state which one a number came from.
- Slice fields are extracted **once** and shared:
  `bs8414_slice_surrogate/data/part1_slices/*.npz`, float16, with **axis0 = Z**
  (128, vertical) after the 2026-07 geometry fix. Read via `PART1_SLICE_DIR`;
  never re-extract per project.
- The **92 `noair` cases have no ventilated cavity**, so `Wing_cavity` and
  `Main_cavity` do not exist for them — FDS writes 3 planes, not 5. They are not
  emitted as samples and are never zero-filled: a zero cavity plane would teach
  the model that removing the cavity makes the cavity cold.

The five planes are `Main_external`, `Wing_cavity`, `Wing_external`,
`Main_cavity`, `Mid_section`.

---

## Where the KAN goes, and why not in the DiT blocks

```
KAN-ised   Part1 conditioning -> AdaLN-Zero modulation vector
UNCHANGED  DiT transformer blocks, AdaLN-Zero mechanism, rectified-flow
           objective, and the entire Stage-1 Function Autoencoder
```

The boundary is a design decision with a concrete reason, not conservatism.
AdaLN-Zero works by having the conditioning vector emit per-block scale, shift
and gate parameters that start at **zero**, so each residual branch is initially
an identity. `KANLinear` ends in a `LayerNorm`, which cannot emit an exact zero
and would break that initialisation — the model would start with live,
randomly-scaled residual branches instead of a clean identity, and any accuracy
difference would be attributable to a broken warm start rather than to B-splines.
So the KAN sits **before** the modulation head: it shapes the conditioning
vector, and the zero-initialised projection still produces the modulation itself.

The Stage-1 FAE is untouched because it autoencodes the field and never sees the
parameter vector — there is no conditioning path in it to KAN-ise. That is also
why a `fae_best.pt` can be reused for a Stage-2-only experiment.

`kan_layers_part1.py` vendors `KANLinear` verbatim rather than re-implementing
it: a re-implementation differing in an initialisation constant or a basis width
would silently invalidate comparison against existing runs. Verify with:

```powershell
python -c "import kan_layers_part1 as k, hashlib, inspect; print(hashlib.sha256(inspect.getsource(k.KANLinear).encode()).hexdigest()[:16])"
```

---

## The models

| Stage | File | What it is | Params |
|---|---|---|---|
| **1 — FAE** | `model/fae.py` | 3D ViT patchify → Perceiver latent bottleneck → self-attention encoder; CViT continuous cross-attention decoder queried at arbitrary `(t, y, z)`. Hard ambient floor (`T = 18 °C + softplus`), soft growth-monotonicity and energy/HRR priors, SSIM term. Latent 128 tokens × 256. | **7,526,401** |
| **2 — DiT** | `model/dit.py` + `model_part1.py` | 8-block Diffusion Transformer, AdaLN-Zero, rectified-flow objective over the frozen FAE latents. Conditioning through `Part1KANParamConditioner`. | **8,507,177** |

`model_part1.py` exports `Part1KANDiT` (aliased `Part1Surrogate`),
`MODEL_NAME = "KAN-FunDiff DiT (Part1)"`, `LAMBDA_REG = 2e-3`, and
`regularization(model)` = spline L2 plus adjacent-knot smoothness over every
`KANLinear`.

Conditioning starts from `Part1Conditioner`, which expands
`[cladding_id, insulation_id, geom_id] + 13 material features` into a 49-d vector
(12 + 8 + 16 embeddings + 13 material), combined with a slice embedding and then
passed through B-spline edges.

Spline knots per edge default to **8** and are env-overridable with
`FUNDIFF_NUM_KNOTS`, which is how the capacity-match run was produced without
editing the tracked default.

**Geometry is an 8-way embedding** over observed flag combinations (bit 0
`noair`, bit 1 `nogap`, bit 2 `nocb`; id 0 = baseline), so it can learn arbitrary
interactions between the modifiers — but **a combination absent from training has
no embedding row and cannot be generated**.

### Inference is a deterministic conditional-mean readout

`P(field | params)` is near-deterministic for a fixed FDS deck, so evaluation and
the app both sample `N_COND_SAMPLES = 8` latents from fixed-seed noise
(`INFERENCE_SEED = 1234`), average them, and decode the full 181 × 128 × 64 grid
with `N_SAMPLING_STEPS = 50` midpoint-Euler ODE steps. Seed, sample count and
step count all change the readout, so all three are recorded with every result.

---

## Training

```powershell
cd D:\VS_projects\bs8414_fundiff_kan_surrogate
.\venv\Scripts\activate
nvidia-smi                              # confirm the 4090 is free
python verify_parity_part1.py           # shared data layer byte-identical
```

Both stages write **fixed filenames** (`fae_best.pt`, `dit_best.pt`), so the run
directory is set by an environment variable:

```powershell
$env:PART1_MODEL_DIR = "models_part1_fundiff_r3"
$env:FUNDIFF_SEED    = "43"                     # default 42
python -u train_fae.py > logs\train_fae_r3.log 2> logs\train_fae_r3.err.log   # ~2.6 h
python -u train_dit.py > logs\train_dit_r3.log 2> logs\train_dit_r3.err.log   # ~25 min
python evaluate.py
Remove-Item Env:PART1_MODEL_DIR, Env:FUNDIFF_SEED
```

**`PART1_MODEL_DIR` is not optional discipline.** Set it before *both* stages and
before evaluation — Stage 2 loads the FAE from the same directory, and leaving it
unset writes into the default `models/`.

**Stage-2-only experiment** (the FAE has no conditioning path, so reusing it is
free and keeps the comparison clean):

```bash
SRC=models_part1_fundiff_r2 ; DST=models_part1_fundiff_k2
mkdir -p "$DST" && cp "$SRC/fae_best.pt" "$DST/"
FUNDIFF_NUM_KNOTS=2 PART1_MODEL_DIR="$DST" ./venv/Scripts/python.exe train_dit.py
FUNDIFF_NUM_KNOTS=2 PART1_MODEL_DIR="$DST" ./venv/Scripts/python.exe evaluate.py
```

`FUNDIFF_NUM_KNOTS` must be set for **evaluation too** — it changes the model
shape, so an unset variable at load time builds an 8-knot model that cannot
accept a 2-knot checkpoint.

Other knobs: `PART1_SPLIT` (`hash` | `system`), `PART1_SIMS_DIR`,
`PART1_SLICE_DIR`.

### Runs on disk

| Directory | Knots | Seed | Test DiT(gen) R² | RMSE | FAE recon R² |
|---|---|---|---|---|---|
| **`models_part1_fundiff_r1`** ★ | 8 | 42 | **0.8740** | 70.8 °C | 0.8778 |
| `models_part1_fundiff_r2` | 8 | 42 | 0.8646 | — | — |
| `models_part1_fundiff_k2` | **2** (capacity match) | 42 | 0.8590 | 74.9 °C | 0.8719 |
| `models_part1_fundiff_s43` | 8 | 43 | trained, not evaluated | — | — |

★ = the deployed default, recorded in `app_assets/selected_model.json`. The
margin over the runner-up is **0.0093** — inside the ±0.02 band this work treats
as inconclusive, so it is the best available run, not a significantly best one.
The 2-knot capacity match sits 0.015 below the 8-knot run, also inside that band.

---

## Evaluation

```powershell
$env:PART1_MODEL_DIR = "models_part1_fundiff_r1"
python evaluate.py                          # -> metrics.json + outputs/fields/*.npz
python evaluate_peak.py --model-dir models_part1_fundiff_r1
python evaluate_uq.py                       # per-pixel mean/std, UQ_N_SAMPLES=24
python recalibrate.py                       # std scale fitted on VALID, applied to test
python smokeview_output.py [--mp4] [--uq]
```

`evaluate.py` reports **two readouts**, so autoencoder quality and generative
quality stay separable:

- **`FAE(recon)`** — encode ground truth → decode. The Stage-1 ceiling.
- **`DiT(gen)`** — sample latent → decode. Full FunDiff, and the number that
  ranks a run.

Metrics are R², RMSE, MAE, MBE, MAPE, p95 and SSIM in physical °C, global and per
slice plane, written to `metrics.json` in the model directory.

`evaluate_uq.py` exploits the fact that this is a generative model: uncertainty
is intrinsic, so drawing N independent latents and taking the per-pixel mean and
standard deviation gives an uncertainty map with no dropout or ensemble needed.
`recalibrate.py` fits the single Gaussian-NLL-optimal scale on the **validation**
split — never on test — and re-scores calibration.

---

## Streamlit app

```powershell
cd D:\VS_projects\bs8414_fundiff_kan_surrogate
.\run_app.ps1                 # http://localhost:8501
.\run_app.ps1 -Port 8502      # a second instance alongside the first
```

`run_app.ps1` activates the venv, exports the material table if missing, prints
GPU status, then starts Streamlit. Manual equivalent:
`.\venv\Scripts\activate ; streamlit run app_fundiff_part1.py`.

Pick a **cladding × insulation × geometry** build-up and it generates the slice
fields — about **4 s per plane** on a 4090 at the frozen 50 steps / 8 samples.
Tabs: a time-slider field snapshot on a common colour scale, peak and mean time
series, the vertical temperature envelope, and `.npz` export carrying the fields
plus the settings that produced them. ODE steps, sample count and seed are all
exposed, because changing them changes the readout.

**Prediction only** — the app reads neither `D:\Bs8414_05052026` nor the
extracted `part1_slices`. Its runtime inputs are `fae_best.pt` + `dit_best.pt`
(the normalisation stats ride inside them) and `app_assets/part1_materials.json`,
written once by an export step that refuses to write unless a cladding/insulation
id provably fixes its material block.

**Part1 is enforced from the checkpoint, not the filename**: a run is offered
only if its `dit_best.pt` carries `cond.conditioner.*` keys, i.e. the Part1
conditioner. Anything hidden is listed in the sidebar with the reason.

What the app refuses to claim:

- **`noair` has no cavity planes.** `Wing_cavity` and `Main_cavity` are not
  offered for a `noair` build-up — FDS writes 3 planes, not 5, and a generated
  cavity field there would be a fiction. Same rule as the training loader's mask.
- **Geometry cannot extrapolate** — an 8-way embedding over *observed* flag
  combinations; an absent build-up gets a warning banner, not a silent field.
- **The selected model is best-available, not significantly best** — with the
  margin shown.

---

## Layout

```
config.py                   FAE/DiT architecture constants, grid, sampling settings
config_part1.py             corpus contract: design space, split, planes, exclusions
data_loader_part1.py        CHID parsing, material extraction, split assignment
slice_loader_part1.py       reads the shared part1_slices/*.npz
dataset_part1.py            corpus adapter: build_datasets() / collate()
part1_conditioning.py       Part1Conditioner: ids + material -> 49-d
kan_layers_part1.py         KANLinear, vendored verbatim
data_processing/dataset.py  query-point sampling and full-grid coordinates
model/fae.py                Stage-1 Function Autoencoder
model/dit.py                Stage-2 DiT, rectified-flow loss, latent sampler
model/physics.py            growth-monotonicity + energy/HRR priors
model_part1.py              Part1KANDiT + Part1KANParamConditioner
train_fae.py                Stage 1 trainer
train_dit.py                Stage 2 trainer
evaluate.py                 frozen eval contract: FAE(recon) and DiT(gen)
evaluate_peak.py            adds the peak-error statistic
evaluate_uq.py              generative UQ: per-pixel predictive mean + std
recalibrate.py              post-hoc Gaussian-NLL std scaling
uq_metrics.py               calibration metrics
smokeview_output.py         Smokeview-style GIF/PNG renders
explain_part1.py            SHAP attribution
causal_part1.py             interventional / causal explainability
verify_parity_part1.py      SHA-256 + array-hash proof of the shared data layer
app_common_part1.py         shared Streamlit input layer
app_fundiff_part1.py        the Streamlit app
run_app.ps1                 app launcher
app_assets/                 part1_materials.json + selected_model.json
models_part1_fundiff_*/     checkpoints + metrics.json
logs/                       paired .log / .err.log — provenance of every number
```

Python sits at the project root deliberately: modules import flat, and
`config.py` / `config_part1.py` derive `PROJECT_DIR`, `MODEL_DIR` and
`SLICE_DIR` from `__file__`, so a `src/` package would silently repoint model and
slice paths.

`train_part1.py`, `evaluate_part1.py`, `train_slices_part1.py`,
`evaluate_slices_part1.py` and `slice_losses_part1.py` are the shared
sensor/regression contracts, carried for the parity manifest. They are not the
entry points here — this pipeline trains by rectified flow in two stages and
keeps its own trainers. `train_slices_part1.py` in particular does not run
against this model.

---

## Notes and caveats

- **Config is the source of truth** — `config.py` for the architecture and the
  FAE/DiT recipe, `config_part1.py` for the corpus.
- **Retrain variance.** Deltas inside **±0.02 R²** are reported inconclusive.
  Every margin currently on record in this project is inside it.
- **Small-data caveat.** The conditional distribution is near-deterministic and
  Part1 gives roughly 600 (sim × plane) functions, which is small for a diffusion
  model; the strong conditioning collapses the generator toward a deterministic
  map. The `FAE(recon)` − `DiT(gen)` gap measures how close Stage 2 gets to the
  Stage-1 ceiling.
- **The two splits are different experiments.** Always say `hash` or `system`.
- Weights are git-ignored by extension; the selected checkpoint is negated back
  in so a fresh clone can predict. Run logs and per-run JSON stay tracked.
