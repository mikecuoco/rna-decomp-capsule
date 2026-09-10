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

$PY -u prepare_sst.py                 # ~20 min -> /scratch/sst-drvi/prepared/sst_counts.h5ad
$PY -u train_drvi.py --smoke          # end-to-end check on 20k cells, 3 epochs
$PY -u train_drvi.py                  # full run on one GPU
```

Both scripts are idempotent: they skip immediately if their output exists, unless
`--force` is passed. `--help` lists the hyperparameter overrides.

Then open `01_inspect_factors.ipynb` with the `sst-drvi` kernel — it is CPU-only and
reads the artifacts below.

## Outputs (all under `/scratch`, nothing in `/results` or `/data`)

| Path | Contents |
|---|---|
| `/scratch/sst-drvi/prepared/sst_counts.h5ad` | cohort, raw counts in `X` (no duplicate `layers`) |
| `/scratch/sst-drvi/gpboost_file_index.csv` | cached index of the 207 GPBoost input files |
| `/scratch/sst-drvi/models/sst_k64/drvi/` | trained model + `history.csv` |
| `/scratch/sst-drvi/embeddings/sst_k64_embed.h5ad` | latent AnnData: `X` = 64 factors, `var` = per-factor stats, `varm` = interpretability scores, `obsm["X_umap"]` |
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
each, which can diverge. `train_drvi.py` checks `torch.isfinite(module.px_r)` and logs
`px_r_finite`; if that is `False`, retrain with `--dispersion gene`.

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
$PY -u train_drvi.py --suffix covar \
  --categorical-covariates "Sex" \
  --continuous-covariates "Fraction mitochondrial UMIs" "Number of UMIs" "PMI" \
  --encode-covariates
```

Three things this does that are worth knowing:

- **Continuous covariates are standardized first.** scvi-tools stacks
  `continuous_covariate_keys` verbatim without scaling, and raw `Number of UMIs` spans
  195 to 2.3e5 — it would dominate every other network input. `scale_continuous_covariates`
  log1p's count-like columns, z-scores all of them, and registers the derived
  `<name> [scaled]` columns. The originals are left untouched.
- **`--encode-covariates` also feeds them to the encoder.** scvi's default is decoder-only,
  which leaves `q(z|x)` batch-blind and lets library structure leak into the latent
  regardless of the batch key.
- **`Sex` is collinear with `library_prep`** (library nests in donor, donor in sex), so the
  batch embedding can already represent it; passing it explicitly mainly matters together
  with `--encode-covariates`.

`scvi-tools 1.5.0` hardcodes `cont_values=None` inside
`get_effect_of_splits_out_of_distribution`, so the OOD traversal crashes on a decoder built
with continuous covariates. `S.calculate_interpretability` wraps it and substitutes zeros,
which is the correct reference point precisely because the covariates are z-scored (zero is
the mean). Remove the shim when upstream fixes it.

## Notes for a larger GPU box

Training pins `devices=1` — scvi's DDP path is not worth the risk here. On a bigger card,
raise `--batch-size` rather than reaching for multiple GPUs, and note that changing it
changes the fit. `--num-workers N` adds dataloader workers; the baseline ran GPU-bound at
95% utilisation with `--num-workers 0`, so it only helps if the GPU gets much faster.

## Not included

No `run`/`run.sh` entrypoint and no `conda-lock` — this arm is still exploratory.
