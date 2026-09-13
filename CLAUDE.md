# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Code Ocean capsule holding one analysis: `code/01_sst_drvi`, a DRVI factorization of the
SST interneurons (231,107 nuclei, 18,892 genes, 902 libraries) in the SEA-AD `multiregion`
dataset, with a matched scVI fit as a baseline. There is no application and no test suite —
the deliverables are trained models, latent embeddings, and three notebooks that read them.

`code/01_sst_drvi/README.md` is the detailed reference and is kept current; read it before
changing the modelling. This file covers what spans several files.

## Environment and paths

```bash
PY=/scratch/.dotfiles/envs/conda/sst-drvi/bin/python     # the only interpreter to use
mamba env update -p /scratch/.dotfiles/envs/conda/sst-drvi -f environment/conda.yaml
```

`environment/conda.yaml` is hand-edited (direct dependencies only), not a `conda env
export`. `~/.condarc` points `envs_dirs`/`pkgs_dirs` at `/scratch`, because root has only
~5 GB. The `sst-drvi` Jupyter kernel is registered.

Capsule layout is load-bearing: `/data` is immutable input (never write there),
`/scratch/sst-drvi/` holds every artifact, `/results` is for explicitly requested final
outputs only. `/data`, `/results` and `/scratch` are gitignored. `scvi-tools >= 1.5` is
required — `scvi.external.DRVI` does not exist below it — and `drvi-py` supplies only
plotting/interpretability helpers, not the model.

## Commands

```bash
cd code/01_sst_drvi

$PY -u prepare_sst.py                     # /data -> /scratch/sst-drvi/prepared/sst_counts.h5ad
$PY -u train_drvi.py --smoke --devices 4  # end-to-end check: 20k cells, 3 epochs, ~3 min
$PY -u train_drvi.py --devices 4          # full DRVI fit
$PY -u train_scvi.py --devices 4          # matched scVI fit
$PY <script>.py --help                    # every hyperparameter is a flag

# notebooks (CPU-only; they read h5ad artifacts and never train)
$PY -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
   --output /scratch/sst-drvi/notebooks/01_qc.executed.ipynb 01_qc.ipynb
```

**`--smoke` is the test suite.** It exercises the whole path — setup, train, embedding,
interpretability, UMAP, write — on a 20k-cell subsample, so run it after touching either
runner. There is nothing else to run; validate notebook changes by executing them headless
and checking for zero `output_type == "error"` cells.

Multi-fit runs live in `/scratch/sst-drvi/*.sh` (sequential, because each fit takes all four
GPUs) with their result readers beside them: `run_probe.sh` + `probe_summary.py`,
`run_long_fits.sh` + `convergence_check.py`.

## Architecture

One shared module, thin CLIs, read-only notebooks:

- **`sst_drvi.py`** — what the *runners* need and the notebooks only reference: paths and
  the `/data` contract (including the `CPS_*` and scCODA constants), `DrviConfig`/
  `ScviConfig`, model builders, the covariate scaler, the DDP and KL-schedule helpers, and
  the two readers both sides share (`latent_stats`, `load_prepared`). Analysis statistics
  do **not** live here — see *Where analysis code lives* below.
- **`prepare_sst.py`, `train_drvi.py`, `train_scvi.py`** — argument parsing, one
  `replace(Config(), ...)` call, orchestration, logging. No analysis logic belongs here.
- **`01_qc.ipynb`** (is each fit trustworthy?), **`02_explore.ipynb`** (what does each
  factor mean?) and **`03_progression.ipynb`** (do the factors track AD severity, and is it
  state or composition?). CPU-only. All GPU work — training, latent representation,
  interpretability scores, UMAP — happens in the runners so the notebooks only read a small
  h5ad. `03` additionally reads the finished scCODA results under
  `/data/multiregion/pertpy/CPS_Local/`, which this capsule does not fit; see
  *Neuropathology inputs* in the analysis README before using them: the delivered
  per-region table is one region-agnostic effect per supertype repeated across regions, and
  significance lives in `Inclusion probability` rather than in a non-zero effect.

**Where analysis code lives.** Each notebook carries its own statistics. A helper is
defined only where it has more than one caller — `factor_association` in `01`/`02`,
`cps_trend` (seven calls) in `03` — and anything called once is written out at the point of
use, with the branches that call does not take dropped. Each notebook opens with one
"helpers used more than once" cell holding those few defs; everything else reads top to
bottom. This does duplicate `factor_association` between `01` and `02`, which is the
accepted cost of not having a shared statistics module.

**Everything expensive is precomputed into the embedding h5ad**: `X` = latent dimensions,
`var` = per-dimension stats, `varm` = interpretability scores (DRVI only), `obsm["X_umap"]`,
and `uns` carrying the JSON config, run record and timings. If a notebook needs something
new, compute it in the runner and store it there rather than in the notebook.

**Run naming** is `sst_k{n_latent}[_{suffix}]`, resolving to `S.MODELS_DIR/<run>/<model>/`
and `S.EMBED_DIR/<run>_<model>_embed.h5ad`; the notebooks build those two paths inline. Both models share a run
name and differ only by subdirectory. Use `--suffix` to keep a variant beside a baseline
instead of overwriting it; existing runs (`sst_k64`, `sst_k64_covar`, `sst_k64_1gpu`,
`sst_k64_probe_*`, `sst_k64*_long`) are comparison sets and should not be clobbered.

## Invariants worth knowing before changing training or reading the factors

These each cost a wasted run to learn. Verify against installed scvi-tools sources rather
than memory — the behaviours are version-specific.

- **The KL warmup must be identical across model families.** Left to their defaults they
  diverge: `DRVITrainingPlan` uses `n_epochs_kl_warmup="auto"` (= `max_epochs`, so it never
  settles) and scvi's `TrainingPlan` a fixed 400 (so a 200-epoch run peaks at
  `kl_weight = 0.4975` and minimises `recon + 0.5·KL`). Both then report an identically
  defined ELBO from different objectives, and the weaker penalty simply buys the better
  reconstruction — which was read as a model-quality difference before anyone checked.
  `S.resolve_kl_warmup` and `S.plan_kwargs` are the single source for both runners; do not
  reintroduce a per-runner schedule. `S.check_kl_schedule` verifies it afterwards.
- **A fit counts as converged only if the last-50-epoch validation ELBO slope is shallower
  than −0.1/epoch.** Epoch counts prove nothing: every 200-epoch fit here has its validation
  minimum on the final epoch at ~−1.4/epoch.
- **Under DDP, `--batch-size` is per device and steps are what matter.** Four GPUs at 256 is
  an effective batch of 1024 and only 203 Adam updates per epoch against 813 at batch 256.
  Scale `--lr` with the batch or lower `--batch-size`; per-epoch compute is roughly
  batch-size independent here, so the smaller batch is cheaper than it looks.
- **scvi-tools disables early stopping under DDP**, and `check_val_every_n_epoch` only
  defaults to 1 when early stopping/checkpointing/an LR monitor is active — so without
  requesting it a DDP run never validates and the history has no `*_validation` metrics at
  all. `--reduce-lr-on-plateau` is a scheduler, not a callback, so it does survive DDP.
- **Non-spawn DDP re-executes the whole script per rank.** Anything after `model.train` must
  sit behind `S.finish_distributed()`, which returns `True` only on rank 0. Ranks also share
  one log file and clobber each other's lines, so `setup_logging` suffixes ranks above zero
  and timings are persisted to `models/<run>/<model>/timings.json`.
- **`find_unused_parameters` is required, not an optimisation**, given
  `dispersion="gene-batch"` and `batch_representation="embedding"` over 902 libraries.
  `px_r` alone is `18892 × 902` = 17.0M of DRVI's 24.4M parameters.
- **The `CPS_*` scores are not per-cell.** `CPS_Global` is constant within a donor and
  `CPS_Local` within a donor x brain region, so a correlation over 231,107 nuclei reports an
  `n` two orders of magnitude larger than the 84 or 530 independent units behind it.
  `03_progression.ipynb` aggregates to a unit first, and measures the granularity in
  section 1 rather than assuming it. Which unit then follows is a real choice, not a detail:
  its per-region, pan-region and pooled-with-cluster-bootstrap views rank the directions
  differently, and pooling every donor x region unit still mixes a between-region contrast
  into the estimate however the *n* is corrected.
- **η² is a variance ratio, so it is scale-invariant** and cannot tell a live latent
  dimension from one collapsed to the prior. The `factor_association` helper in `01`/`02`
  prunes on `vanished` (DRVI) or `used` (scVI) and logs the count; a table shorter than
  `n_latent` is expected, and a copy of that rule lives in each of the two notebooks.
- **Continuous covariates must be standardized before registration.** scvi-tools stacks
  `continuous_covariate_keys` verbatim, and raw `Number of UMIs` spans 195–2.3e5.
  `S.scale_continuous_covariates` log1p's count-like columns, z-scores all of them,
  registers derived `<name> [scaled]` columns, and records the transform in
  `uns["covariate_scaling"]` so the covariate table in `01_qc.ipynb` can report it
  truthfully.
- **Covariate fits are decoder-only on purpose**: `--encode-covariates` is deliberately not
  passed, leaving `q(z|x)` covariate-blind.

## Conventions

Comments explain *why*, especially where the code works around a library behaviour — most
existing comments name the upstream file that forces the workaround. Match that: a comment
restating the code is noise, one naming the constraint is the point.

Prefer adding a flag with a documented default over changing a default, so existing runs
stay reproducible. Keep derived data out of `/data`, and do not create `run`/`run.sh` or
`conda-lock` — reproducibility here is opt-in and not yet requested.

## Other agent configs

An OpenAI Codex config exists at `~/.codex/config.toml`. If you want its user-level items
(MCP servers, slash commands, subagents, skills, instructions) available here, reply
`/import` to scan and list what's importable, then `/import --yes=<digest>` with the digest
that scan prints. If `/import` is unavailable on this surface, run `claude import` from a
terminal.
