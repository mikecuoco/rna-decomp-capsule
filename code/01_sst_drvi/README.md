# 01_sst_drvi — DRVI factorization of SEA-AD multiregion SST neurons

Learns a disentangled latent representation (`k=64`) of the SST interneurons in the
`multiregion` dataset with [DRVI](https://github.com/theislab/DRVI), so individual
latent factors can be related to supertype, brain region, donor and neuropathology.

## Cohort

`Subclass == "Sst"` (231,107 nuclei, 22 supertypes) taken from the per-supertype
QC-passed matrices under
`data/multiregion/Data/GPBoost_inputs/**/<Supertype>_goodcells_for_gpboost.h5ad`.
These files hold raw counts in `X`, and their `obs` is the richest in the dataset —
it is the only place carrying the `CPS_*` pseudo-progression scores.
Genes are the intersection across source files (their per-supertype gene sets differ).

`Sst Chodl` (4,383 nuclei) is excluded by default; pass `--include-chodl` to add it.

## Environment

Defined in `environment/conda.yaml` at the capsule root. The model comes from
`scvi.external.DRVI` (scvi-tools >= 1.5); `drvi-py` supplies only the plotting and
interpretability helpers.

```bash
mamba env update -p /scratch/.dotfiles/envs/conda/sst-drvi \
  -f /root/capsule/environment/conda.yaml
```

`~/.condarc` points `envs_dirs`/`pkgs_dirs` at `/scratch`, so nothing lands on the
~5 GB root filesystem. The `sst-drvi` Jupyter kernel is already registered.

## Running

```bash
PY=/scratch/.dotfiles/envs/conda/sst-drvi/bin/python

$PY -u prepare_sst.py                          # ~20 min -> prepared/sst_counts.h5ad
$PY -u train_drvi.py --smoke --devices 4       # end-to-end check, 20k cells, 3 epochs
$PY -u train_drvi.py --devices 4               # full DRVI run
$PY -u train_scvi.py --devices 4               # scVI, for comparison
```

The three fits behind the notebooks are driven by `/scratch/sst-drvi/run_ddp_fits.sh`,
which runs them sequentially because each takes all four GPUs:

```bash
COVAR=(--categorical-covariates "Sex" "Chemistry"
       --continuous-covariates "Fraction mitochondrial UMIs" "Number of UMIs" "PMI")

$PY -u train_drvi.py --devices 4                                # sst_k64        (baseline)
$PY -u train_drvi.py --devices 4 --suffix covar "${COVAR[@]}"   # sst_k64_covar
$PY -u train_scvi.py --devices 4 --suffix covar "${COVAR[@]}"   # sst_k64_covar, scVI
```

The covariate fits are decoder-only: `--encode-covariates` is deliberately *not* passed,
so the encoder stays covariate-blind and only the decoder is told about them.

All scripts are idempotent: they skip immediately if their output exists, unless
`--force` is passed. `--help` lists the hyperparameter overrides.

Then open the notebooks with the `sst-drvi` kernel — both are CPU-only and read the
artifacts below.

| Notebook | Answers |
|---|---|
| `01_qc.ipynb` | Is each fit trustworthy, and how do the fits differ? Model design, how each covariate is modelled, train/validation loss curves, dimension usage, latent UMAPs, and latent-vs-covariate association per fit. |
| `02_explore.ipynb` | What does each factor mean? Per-factor activity, factor x group heatmaps, the gene program behind each direction, and the sign-split analysis. DRVI only. |

`01_qc.ipynb` loops over whichever fits are on disk and prints a notice for the rest, so
it is readable while runs are still going. `02_explore.ipynb` targets `sst_k64_covar` and
falls back to `sst_k64`.

## Multi-GPU

`--devices N` trains under DDP. Four things about that path are not obvious, and
`ddp_trainer_kwargs` in `sst_drvi.py` exists to handle them:

- **`find_unused_parameters` is required, not an optimisation.** `dispersion="gene-batch"`
  fits one `px_r` row per library and `batch_representation="embedding"` one embedding row
  per library. With 902 libraries, nearly every row sees no cell on a given rank and so
  receives no gradient — plain `"ddp"` treats that as an error. The strategy used is
  `ddp_find_unused_parameters_true`.
- **`--batch-size` becomes per device.** Four GPUs at 256 is an effective batch of 1024
  with the learning rate unchanged, so a DDP fit is not step-for-step comparable with a
  single-GPU one. The effective size is logged, and `devices` is recorded in the config
  stored in `uns`.
- **scvi-tools disables early stopping under DDP** (`scvi/train/_trainer.py`), so a
  distributed run always uses all `max_epochs`. Worse, `check_val_every_n_epoch` is only
  defaulted to 1 when early stopping or checkpointing is active — without asking for it
  explicitly a DDP run never validates and the history comes back with no `*_validation`
  metrics at all. It is requested explicitly.
- **Non-spawn DDP re-executes the whole script per rank.** Everything after `model.train`
  writes files, so `finish_distributed()` barriers, tears down the process group, and
  returns `False` on every rank but zero, which then returns early. Without it four
  processes race to write the same model, embedding and h5ad. scvi-tools also redirects
  `SimpleLogger` to on-disk history under DDP with `save_dir` defaulting to the *current
  working directory*, so `log_save_dir` points it at `/scratch/sst-drvi/lightning_logs`
  rather than dropping a tree into this repository.

## Outputs (all under `/scratch`, nothing in `/results` or `/data`)

| Path | Contents |
|---|---|
| `/scratch/sst-drvi/prepared/sst_counts.h5ad` | cohort, raw counts in `X` (no duplicate `layers`) |
| `/scratch/sst-drvi/gpboost_file_index.csv` | cached index of the 207 GPBoost input files |
| `/scratch/sst-drvi/models/<run>/{drvi,scvi}/` | trained model + full per-epoch `history*.csv` |
| `/scratch/sst-drvi/embeddings/<run>_{drvi,scvi}_embed.h5ad` | latent AnnData: `X` = 64 dimensions, `var` = per-dimension stats, `varm` = interpretability scores (DRVI only), `obsm["X_umap"]` |
| `/scratch/sst-drvi/models/sst_k64_1gpu/`, `…/embeddings/sst_k64_1gpu_drvi_embed.h5ad` | the original single-GPU DRVI baseline, kept for comparison against the DDP refit |
| `/scratch/sst-drvi/lightning_logs/` | Lightning's on-disk history under DDP (redirected out of the repo) |
| `/scratch/sst-drvi/logs/` | run logs |

## Model configuration

Defaults live in `DrviConfig` in `sst_drvi.py`:

| | |
|---|---|
| `n_latent` | 64, every dimension its own split (`n_split_latent=None`) |
| decoder | `split_method="split_map"`, `split_aggregation="logsumexp"` |
| likelihood | `pnb` (log-space negative binomial), `dispersion="gene-batch"` |
| batch | `library_prep` (~600 levels) via `batch_representation="embedding"` |
| architecture | `n_hidden=128`, `n_layers=2` |
| training | 200 epochs max, KL warmup over the whole run, early stopping on `elbo_validation` |

`dispersion="gene-batch"` fits one dispersion per gene per library from ~390 cells
each, which can diverge. Both runners check `torch.isfinite(module.px_r)` and log
`px_r_finite`; if that is `False`, retrain with `--dispersion gene`.

`ScviConfig` mirrors this on every shared knob — same `n_latent`, architecture, batch
handling, dispersion and schedule — so a difference between the two embeddings is
attributable to DRVI's split decoder rather than to the setup. Two things cannot match:
`gene_likelihood` is `nb`, because `pnb` is DRVI's log-space variant and `SCVI` does not
accept it; and the long KL warmup is not forced on scVI, which does not need it to
disentangle (`--kl-warmup-epochs` makes the schedules identical if you want that).
scVI gets no interpretability scores — its dimensions are not claimed to be individually
meaningful — so its latent dimensions are named `Z_n` rather than `DR_n`, and comparing
how much of `n_latent` each model used goes through `latent_stats`, which applies one
shared rule to both.

## Modelling nuisance covariates

The baseline fit (`sst_k64`) gave the model only the discrete library batch. Inspection
showed it therefore spent latent dimensions on covariates it was never told about:

| Factor | Top genes | Covariate |
|---|---|---|
| `DR 23-` | `MT-ATP6, MT-CO2, MT-ND3, MT-CO3, MT-ND4` | mito fraction, eta<sup>2</sup>=0.95 |
| `DR 3+` | `SRRM2, PNISR, CCDC144A, LINC00342` | detection breadth, eta<sup>2</sup>=0.56 |
| `DR 41` | `USP9Y, UTY, NLGN4Y` / `XIST, TSIX, JPX` | sex (real biology, not an artifact) |

To fit with those modelled instead, keeping the baseline beside it:

```bash
$PY -u train_drvi.py --devices 4 --suffix covar \
  --categorical-covariates "Sex" "Chemistry" \
  --continuous-covariates "Fraction mitochondrial UMIs" "Number of UMIs" "PMI"
```

Three things this does that are worth knowing:

- **Continuous covariates are standardized first.** scvi-tools stacks
  `continuous_covariate_keys` verbatim without scaling, and raw `Number of UMIs` spans
  195 to 2.3e5 — it would dominate every other network input. `scale_continuous_covariates`
  log1p's count-like columns, z-scores all of them, and registers the derived
  `<name> [scaled]` columns. The originals are left untouched.
- **The covariates go to the decoder only.** `--encode-covariates` exists but is not used:
  the fits keep scvi's default, so `q(z|x)` never sees them. That is a real trade-off
  rather than an oversight. Decoder-only means the reconstruction can explain a
  sex-linked or mitochondrial gene from the covariate instead of from `z`, which removes
  the *pressure* on the latent space to encode it — but a covariate-blind encoder can
  still put it there, so leakage is reduced, not ruled out. Judge how far it got from the
  `covariate_design` table read next to the association heatmap, not from the flag.
- **`Sex` and `Chemistry` are collinear with `library_prep`** (library nests in donor,
  donor in sex; chemistry is a property of the library), so the batch embedding can
  already represent them. Naming them separately still helps the decoder: it gets a
  2-level effect shared across all 902 libraries instead of having to learn it inside a
  902-row embedding.

`scale_continuous_covariates` records the transform it chose per covariate in
`uns["covariate_scaling"]`, because the `log1p` decision is data-dependent and cannot be
recovered from the saved model. `covariate_design` reports the exact transform when handed
that record and the generic "standardized" otherwise.

`scvi-tools 1.5.0` hardcodes `cont_values=None` inside
`get_effect_of_splits_out_of_distribution`, so the OOD traversal crashes on a decoder built
with continuous covariates. `S.calculate_interpretability` wraps it and substitutes zeros,
which is the correct reference point precisely because the covariates are z-scored (zero is
the mean). Remove the shim when upstream fixes it.

## Performance notes

Measured on 4× Tesla T4 (15 GB each), 48 cores, 186 GB RAM, `--devices 4
--batch-size 256 --num-workers 0`: DRVI runs at ~39.5 s/epoch, i.e. ~2h10m for 200
epochs, against 54.7 s/epoch (~3h02m) for the same model on a single A10G. All four GPUs
sit at 89-94% utilisation and 6.5 GB, and the four ranks together hold ~38 GB RSS because
each loads the cohort itself.

`--num-workers N` adds dataloader workers; the runs are GPU-bound with `--num-workers 0`,
so it only helps on faster cards. Raising `--batch-size` changes the fit, and under DDP it
multiplies by the device count.

## Reading section 6 of the QC notebook

The association heatmap and the covariate table are meant to be read together, and neither
means much alone:

- a covariate marked `not modelled` that scores **high** means a latent dimension was
  spent on nuisance structure — this is how the baseline's mitochondrial and
  detection-breadth factors were found;
- a covariate that *is* modelled and still scores high means it was not absorbed. With
  decoder-only covariates that is the expected failure mode rather than a surprise, since
  the encoder never saw it;
- eta-squared rises with level count, so compare a dimension's `library_prep` score (902
  levels) against its own `Supertype` score (19), never against 1.0.

## Not included

No `run`/`run.sh` entrypoint and no `conda-lock` — this arm is still exploratory.
