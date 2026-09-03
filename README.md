# bs8414_fundiff_kan_surrogate — KAN-FunDiff slice-field surrogate

Sibling of `bs8414_fundiff_surrogate`, and the **KAN arm** of the FunDiff
generative slice-field experiment on the BS 8414-1 facade fire test.

Same paper (Wang, Dou, Shan, Liu & Lu, *"FunDiff: diffusion models over function
spaces for physics-informed generative modeling"*, Nature Communications
**17**:5749, 2026, [doi:10.1038/s41467-026-72292-0](https://doi.org/10.1038/s41467-026-72292-0)),
same corpus, same split, same two-stage pipeline, same trainers. **One variable
changes:** the DiT's conditioning path is built from `KANLinear` B-spline edges
instead of a dense MLP.

Target: the 2D temperature field `T(t, z, y)` on the BS 8414 slice planes —
**181 timesteps × 128 (vertical Z) × 64 (horizontal)** per plane.

---

## Where the KAN goes, and why not in the DiT blocks

```
KAN-ised   Part1 conditioning -> AdaLN-Zero modulation vector
UNCHANGED  DiT transformer blocks, AdaLN-Zero mechanism, rectified-flow
           objective, and the entire Stage-1 FAE
```

Same boundary as `bs8414_physicsnemo_kan_surrogate`, and the same precedent as
`bs8414_KAN_surrogate` (KAN in the parameter encoder and decoders, backbone
untouched). The reason is concrete, not conservatism: AdaLN-Zero works by having
the conditioning vector emit per-block scale/shift/gate parameters that start at
**zero**, so the residual branch is initially an identity. `KANLinear` ends in a
`LayerNorm`, which cannot emit an exact zero and would break that
initialisation — any accuracy difference would then be attributable to a broken
warm start rather than to B-splines. So the KAN sits *before* the modulation
head: it shapes the conditioning vector, and the parent's zero-initialised
projection still produces the modulation itself.

The Stage-1 FAE is untouched because it autoencodes the field and never sees the
parameter vector — there is no conditioning path in it to KAN-ise. Both arms
therefore share an identical FAE recipe, which is why the `fae_best.pt` from one
run can be reused for a Stage-2-only experiment.

`kan_layers_part1.py` vendors `KANLinear` **byte-for-byte** from
`bs8414_KAN_surrogate/model.py` (lines 18–105), not re-implemented — the whole
point of a KAN-vs-baseline ablation is that the B-spline edge block is the same
block the existing KAN surrogate was measured with. Vendored rather than imported
because each project has its own venv. Verify:

```powershell
python -c "import kan_layers_part1 as k, hashlib, inspect; print(hashlib.sha256(inspect.getsource(k.KANLinear).encode()).hexdigest()[:16])"
```

---

## The models

| Stage | File | What it is | Params |
|---|---|---|---|
| **1 — FAE** | `model/fae.py` | 3D ViT patchify → Perceiver latent bottleneck → self-attention encoder; CViT continuous cross-attention decoder queried at arbitrary `(t, y, z)`. Hard ambient floor (`T = 18 °C + softplus`), soft growth-monotonicity and energy/HRR priors. **Identical to the non-KAN arm.** | **7,526,401** |
| **2 — DiT** | `model/dit.py` + `model_part1.py` | 8-block Diffusion Transformer, AdaLN-Zero, rectified-flow objective over the frozen FAE latents (128 tokens × 256). Conditioning through `Part1KANParamConditioner`. | **8,507,177** |

`model_part1.py` exports the uniform Part1 interface:
`Part1KANDiT` (aliased `Part1Surrogate`), `MODEL_NAME = "KAN-FunDiff DiT (Part1)"`,
`LAMBDA_REG = 2e-3`, `regularization(model)` = spline L2 + adjacent-knot
smoothness over every `KANLinear`.

Spline knots per edge default to **8** and are env-overridable with
`FUNDIFF_NUM_KNOTS`, which is how the capacity-match experiment
(`models_part1_fundiff_k2`) was run without editing the tracked default.

**Inference is a deterministic conditional-mean readout** — `P(field | params)`
is near-deterministic for a fixed FDS deck, so evaluation and the app sample
`N_COND_SAMPLES = 8` latents from fixed-seed noise (`INFERENCE_SEED = 1234`),
average, and decode the full 181 × 128 × 64 grid with `N_SAMPLING_STEPS = 50`
midpoint-Euler ODE steps. Seed, sample count and step count all change the
readout, so all three are recorded.

---

## Corpus

The **Part1 geometry corpus** (`D:\Bs8414_05052026\Part1\_completed`), read
through the shared data layer:

- 186 completed sims → **184 usable** (2 excluded by name in
  `config_part1.EXCLUDED_CHIDS`, each with its reason).
- Design axes: cladding(12) × insulation(5) × **geometry(8)**. HRR and mesh are
  constants in this batch and carry no signal, so they are not inputs.
- Hash split **141 train / 20 valid / 23 test** (`PART1_SPLIT=hash`, default).
  `PART1_SPLIT=system` keeps all 8 geometry variants of a build-up together and
  measures generalisation to an unseen system — a **different, non-comparable**
  protocol. Always state which one a number came from.
- Slice fields are extracted **once** and shared:
  `bs8414_slice_surrogate/data/part1_slices/*.npz`, float16, axis0 = Z (128,
  vertical) per the 2026-07 geometry fix. Read via `PART1_SLICE_DIR`; never
  re-extract per project.
- The 92 `noair` cases have **no ventilated cavity**, so `Wing_cavity` and
  `Main_cavity` do not exist for them — 3 planes, not 5. They are not emitted as
  samples and are never zero-filled: a zero cavity plane would teach the model
  that removing the cavity makes the cavity cold.

`config.py` still describes the parent 60-sim corpus and holds the FAE/DiT
architecture constants. This project has **no 60-sim checkpoints** — it was built
directly on Part1.

---

## Layout

```
config.py                   architecture + parent 60-sim corpus; MODEL_DIR honours $PART1_MODEL_DIR
config_part1.py             Part1 corpus contract                       (shared file)
dataset_part1.py            Part1 adapter — build_datasets()/collate()
slice_loader_part1.py       reads the shared part1_slices/*.npz          (shared file)
data_loader_part1.py        Part1 CHID/material/split logic              (shared file)
part1_conditioning.py       Part1Conditioner: ids + material -> 49-d     (shared file)
physics_part1.py            physics gates, optional closure/geom penalty (shared file)
kan_layers_part1.py         KANLinear, vendored verbatim from the KAN project
model/fae.py                Stage-1 Function Autoencoder (unchanged from the parent)
model/dit.py                Stage-2 DiT, rectified-flow loss, latent sampler
model/physics.py            growth-monotonicity + energy/HRR priors
model_part1.py              THE VARIABLE UNDER TEST — Part1KANDiT + KAN conditioner
train_fae.py                Stage 1 trainer
train_dit.py                Stage 2 trainer
evaluate.py                 frozen eval contract: FAE(recon) and DiT(gen), per slice
evaluate_peak.py            adds the peak-error statistic
evaluate_uq.py              generative UQ: per-pixel predictive mean + std
recalibrate.py              post-hoc Gaussian-NLL std scaling, fitted on valid only
uq_metrics.py               calibration metrics
smokeview_output.py         Smokeview-style GIF/PNG renders
explain_part1.py            SHAP attribution                            (shared file)
causal_part1.py             interventional/causal explainability        (shared file)
verify_parity_part1.py      SHA-256 + array-hash proof of shared-layer identity
app_common_part1.py         shared Streamlit input layer
app_fundiff_part1.py        the Streamlit slice app
run_app.ps1                 app launcher
app_assets/                 part1_materials.json + selected_model.json
```

**Inert copies.** `train_part1.py`, `evaluate_part1.py`, `train_slices_part1.py`,
`evaluate_slices_part1.py` and `slice_losses_part1.py` are the *sensor* and
*deterministic-slice* contracts and do not run here — the generative pipeline
keeps its own two trainers. `train_slices_part1.py` fails outright
(`Part1KANDiT.__init__() missing 1 required positional argument: 'config'`), and
the empty `models_part1_slice_r1` directory is the residue of that attempt.

---

## Training

```powershell
cd D:\VS_projects\bs8414_fundiff_kan_surrogate
.\venv\Scripts\activate
python verify_parity_part1.py          # shared layer identical across projects
```

Both stages write **fixed filenames** (`fae_best.pt`, `dit_best.pt`), so the run
directory is set by an environment variable:

```powershell
$env:PART1_MODEL_DIR = "models_part1_fundiff_r3"
$env:FUNDIFF_SEED    = "43"                     # default 42
python -u train_fae.py > train_fae_r3.log 2> train_fae_r3.err.log   # Stage 1, ~2.6 h
python -u train_dit.py > train_dit_r3.log 2> train_dit_r3.err.log   # Stage 2, ~25 min
python evaluate.py
Remove-Item Env:PART1_MODEL_DIR, Env:FUNDIFF_SEED
```

Bash equivalent — how the replicate scripts at the root drive it:

```bash
d=models_part1_fundiff_r3
FUNDIFF_SEED=43 PART1_MODEL_DIR="$d" ./venv/Scripts/python.exe train_fae.py
FUNDIFF_SEED=43 PART1_MODEL_DIR="$d" ./venv/Scripts/python.exe train_dit.py
PART1_MODEL_DIR="$d" ./venv/Scripts/python.exe evaluate.py
```

Set `PART1_MODEL_DIR` before *both* stages and before evaluation — Stage 2 loads
the FAE from the same directory, and an unset variable writes into `models/`.

**Stage-2-only experiment** (the capacity match is exactly this — the FAE has no
conditioning path, so reusing it is free):

```bash
SRC=models_part1_fundiff_r2 ; DST=models_part1_fundiff_k2
mkdir -p "$DST" && cp "$SRC/fae_best.pt" "$DST/"
FUNDIFF_NUM_KNOTS=2 PART1_MODEL_DIR="$DST" ./venv/Scripts/python.exe train_dit.py
FUNDIFF_NUM_KNOTS=2 PART1_MODEL_DIR="$DST" ./venv/Scripts/python.exe evaluate.py
```

`FUNDIFF_NUM_KNOTS` must be set for **evaluation too** — it changes the model
shape, so an unset variable at load time builds an 8-knot model that cannot
accept the 2-knot checkpoint.

Other knobs: `PART1_SPLIT` (`hash` | `system`), `PART1_SIMS_DIR`,
`PART1_SLICE_DIR`.

### Runs on disk

| Directory | Knots | Seed | Test DiT(gen) R² | RMSE | FAE recon R² |
|---|---|---|---|---|---|
| **`models_part1_fundiff_r1`** ★ | 8 | 42 | **0.8740** | 70.8 °C | 0.8778 |
| `models_part1_fundiff_r2` | 8 | 42 | 0.8646 | — | — |
| `models_part1_fundiff_k2` | **2** (capacity match) | 42 | 0.8590 | 74.9 °C | 0.8719 |
| `models_part1_fundiff_s43` | 8 | 43 | trained, not evaluated | — | — |
| `models_part1_slice_r1` | — | — | empty; failed slice-trainer attempt | — | — |

★ = the deployed default, recorded in `app_assets/selected_model.json`. The
margin over the runner-up is **0.0093** — inside the ±0.02 inconclusive band, so
this is the best available run, not a significantly best one.

**KAN vs MLP is not established from these numbers.** 0.8740 (KAN r1) against
0.8658 (MLP-conditioned r1) is a 0.0082 gap on n=2 versus n=2 replicates, well
inside the noise band. The k2 capacity match (0.8590) sits below both. Report
this arm as *no measurable difference so far*, and see
`..\model_comparison.csv` for the full cross-architecture table.

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

`evaluate.py` reports **two readouts** so autoencoder quality and generative
quality stay separable:

- **`FAE(recon)`** — encode ground truth → decode. The Stage-1 ceiling.
- **`DiT(gen)`** — sample latent → decode. Full FunDiff, and the number that
  ranks a run.

Metrics are R², RMSE, MAE, MBE, MAPE, p95 and SSIM in physical °C, global and per
slice plane, written to `metrics.json` in the model directory — which is what the
root-level `select_best_model.py` reads.

---

## Streamlit app

```powershell
cd D:\VS_projects\bs8414_fundiff_kan_surrogate
.\run_app.ps1                 # http://localhost:8501
.\run_app.ps1 -Port 8502      # alongside the non-KAN FunDiff app for a side-by-side
```

`run_app.ps1` activates this venv, picks `app_fundiff_part1.py`, exports the
material table if missing, prints GPU status, then starts Streamlit. Manual
equivalent: `.\venv\Scripts\activate ; streamlit run app_fundiff_part1.py`.

Pick a **cladding × insulation × geometry** build-up and it generates the slice
fields, ~4 s per plane on the 4090 at the frozen 50 steps / 8 samples. Tabs: a
time-slider field snapshot on a common colour scale, peak/mean time series, the
vertical temperature envelope, and `.npz` export carrying the fields plus the
settings that produced them.

Because `app_fundiff_part1.py` binds only to `Part1Surrogate` / `MODEL_NAME` from
`model_part1.py`, this is the **same app file** as the non-KAN project — run both
on different ports and any difference you see is attributable to the conditioning
architecture.

**Prediction only** — the app never reads `D:\Bs8414_05052026` or the extracted
`part1_slices`. Runtime inputs are `fae_best.pt` + `dit_best.pt` (normalisation
stats ride inside them) and `app_assets/part1_materials.json`, written once by
the root-level `export_app_assets.py`.

**Part1 enforced from the checkpoint, not the filename**: a run is offered only
if its `dit_best.pt` carries `cond.conditioner.*` keys. Hidden runs are listed in
the sidebar with the reason.

What the app refuses to claim:

- **`noair` has no cavity planes** — `Wing_cavity` and `Main_cavity` are not
  offered for a `noair` build-up, matching FDS and the training mask.
- **Geometry cannot extrapolate** — an 8-way embedding over *observed* flag
  combinations; an absent build-up gets a warning banner.
- **The selected model is best-available, not significantly best** — with the
  margin shown.

`app_common_part1.py`, `app_fundiff_part1.py` and `run_app.ps1` are
**byte-identical** across the projects that hold them. Never hand-edit one copy;
edit and re-copy. Full app contract: `..\APPS.md`.

---

### Repository layout

Run logs, analysis records and before-state snapshots are grouped so the project
root holds only what you run:

```
<project>/
  README.md            this file
  *.py                 all modules and entry points — flat, at the root
  models_*/            checkpoints + per-run provenance JSON
  app_assets/          part1_materials.json, selected_model.json
  docs/                results records and analyses (PART1_RESULTS.md, analysis_*.md, ...)
  logs/                paired .log / .err.log run logs — the provenance of every number
  archive/             before-state snapshots of deliberate edits (*.pre-*, *.bak)
```

**Python stays at the project root, deliberately.** Every module imports flat
(`from config_part1 import ...`) and `config.py` / `config_part1.py` derive
`PROJECT_DIR`, `MODEL_DIR`, `OUTPUT_DIR` and `SLICE_DIR` from `__file__` — moving
them into a `src/` package would silently repoint model and slice paths, and
those files must stay byte-identical across all eleven surrogate projects for
`verify_parity_part1.py` to pass. New run logs still land at the root; move them
into `logs/` when you tidy.

`CLAUDE.md` is git-ignored: it is the working brief for agent sessions, not part
of the published artefact.

---

## Notes and caveats

- **Config is the source of truth** — `config.py` for the architecture and the
  FAE/DiT recipe, `config_part1.py` for the corpus.
- **Retrain variance.** Deltas inside **±0.02 R²** are reported inconclusive.
  Every margin currently on record in this project is inside it.
- **The two splits are different experiments.** Always say `hash` or `system`.
- `verify_parity_part1.py` must pass before any cross-project comparison — it
  proves the shared modules are byte-identical (SHA-256) *and* that the arrays
  built inside each project's own venv hash the same.
- Weights are git-ignored by extension; the selected checkpoint is negated back
  in so a fresh clone can predict. Run logs and per-run JSON stay tracked.

## Related

`..\CLAUDE.md` (project map, Part1 contract) · `..\APPS.md` (app and deployment
contract) · `bs8414_fundiff_surrogate` (the MLP-conditioning arm) ·
`bs8414_KAN_surrogate` (source of the vendored `KANLinear`) ·
`bs8414_physicsnemo_kan_surrogate` (the KAN/non-KAN pair at operator granularity).
