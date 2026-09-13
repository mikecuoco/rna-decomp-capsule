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

## Neuropathology inputs

Two things in `/data` carry disease severity, and they are read, never written.

**The `CPS_*` columns** in the prepared cohort are SEA-AD's continuous pseudo-progression
scores. They sit in `obs`, one value per nucleus, which makes it easy to treat them as
per-cell measurements — they are not. `CPS_Global` is constant within a donor (84 distinct
values over 84 donors) and `CPS_Local` within a donor x brain region (530 units). A
correlation over all 231,107 nuclei therefore reports an *n* two orders of magnitude larger
than the number of independent units, so `03_progression.ipynb` aggregates first, and
section 1 there measures the granularity rather than assuming it.

Aggregating leaves a choice of unit, and the three that notebook uses answer different
questions: `cps_trend(..., by="Brain Region")` keeps donors as the unit within one region
(section 2), the pan-region table collapses to 84 independent donors after removing the
region effect (2b), and the clustered test pools all 516 donor x region units with a
donor-level bootstrap (2c). The last recovers the *n* the pooled test gets wrong but not the
region confound, which is a property of the design; the three orderings differ, and the
notebook shows by how much rather than picking one silently.

**`data/multiregion/pertpy/CPS_Local/`** holds a finished scCODA compositional analysis of
all 174 supertypes against `CPS_Local`. Nothing in this capsule fits it. Three artifacts
of one analysis, answering different questions:

Paths and constants for them live in `sst_drvi.py`; the readers are written out in
section 4 of `03_progression.ipynb`.

| File | What it is |
|---|---|
| `pertpy_summary_CPS_Local.<date>.csv` | the delivered effect, one value per supertype x region |
| `<class group>_Supertype_results.csv` | the evidence: every effect re-estimated against each of the 174 possible reference cell types, over 6 covariates and 11 regions — the ten anatomical ones plus scCODA's own region-agnostic `Global` fit (~180 MB) |
| `objects/<class group>_Supertype_abundances.h5ad` | scCODA's input: 907 libraries x 174 supertype counts, with `CPS_Local` in `obs` |

Four properties of those files were checked, and each changes how they may be used:

- **The summary is the sweep's region-agnostic `Global` fit, thresholded and broadcast.**
  For every SST supertype the summary's repeated value matches the median `Global` effect
  over the 173 references to within 0.016, and the ones written as `0.0` are exactly those
  whose median posterior inclusion probability falls below ~0.83. Two departures: a
  supertype confined to one region takes that region's own fit (`Sst_27-SEAAD` appears only
  in V1C, where the sweep gives −1.132 against a `Global` estimate of +0.005, and the
  summary carries −1.106), and MTG's values match neither the sweep's MTG rows nor `Global`
  — they come from outside this file, presumably SEA-AD's separately-fit MTG dataset, so an
  MTG cell is a different study's answer rather than a regional contrast.
- **The summary's ten region columns are not ten measurements.** One estimate repeated
  across each region a supertype appears in, MTG excepted — independent per-region MCMC fits
  cannot agree to six decimal places. Section 4 surfaces this; the
  usable quantity is one effect per supertype, and the regional resolution in
  `03_progression.ipynb` comes from the latent space instead.
- **`NaN`, `0.0` and a non-zero value mean three different things**: not modelled (no nuclei
  of that supertype in that region, verified against the abundance counts), modelled but not
  credible, and a credible shift. A `fillna(0)` would collapse the first two.
- **Significance in the sweep is `Inclusion probability`, not `Final Parameter != 0`.** In
  the sweep a zero appears only on a reference's own self-row (264 of 45,936 SST rows, all
  with inclusion probability exactly 0), so counting non-zeros scores every effect at
  1 − 1/174 regardless of the evidence. The sweep reader drops the self-rows and thresholds
  on inclusion instead; `S.SCCODA_INCLUSION_THRESHOLD` is 0.83, recovered from
  where the summary's own calls fall — they separate `Sst_9` (0.815, written as `0.0`) from
  `Sst_22` (0.841, kept), and 0.83 reproduces all 18 of them.

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

Those three runs are the 200-epoch set, and **none of them converged** (see *Convergence
and the KL schedule*). The refits are driven in two stages:

```bash
/scratch/sst-drvi/run_probe.sh                 # 4 arms x 40 epochs, batch/LR/precision
$PY /scratch/sst-drvi/probe_summary.py         # s/epoch, ELBO drop per hour, finiteness

BATCH=… LR=… PRECISION=… EPOCHS=… \
  /scratch/sst-drvi/run_long_fits.sh           # the three *_long fits
$PY /scratch/sst-drvi/convergence_check.py     # kl_weight reached, trailing ELBO slope
```

The probe holds the KL schedule fixed across its arms (`--kl-warmup-epochs 13`) so they
differ only in batch, learning rate and precision, and uses `--train-only` to skip the
~11-minute embedding tail it does not need. The long fits deliberately do *not* pass
`--kl-warmup-epochs`, so both families take the shared default of a third of the budget.

All scripts are idempotent: they skip immediately if their output exists, unless
`--force` is passed. `--help` lists the hyperparameter overrides.

Then open the notebooks with the `sst-drvi` kernel — all three are CPU-only and read the
artifacts below.

| Notebook | Answers |
|---|---|
| `01_qc.ipynb` | Is each fit trustworthy, and how do the fits differ? Model design, how each covariate is modelled, train/validation loss curves, dimension usage, latent UMAPs, and latent-vs-covariate association per fit. |
| `02_explore.ipynb` | What does each factor mean? Per-factor activity, factor x group heatmaps, the gene program behind each direction, and the sign-split analysis. DRVI only. |
| `03_progression.ipynb` | Do the factors track AD severity, and is it state or composition? Sign-split directions against the `CPS_*` scores at three units of analysis — within region, per donor pan-region, and every donor x region pooled with a donor-cluster bootstrap — plus the ABeta-vs-pTau split, the same tests with supertype held fixed, and the delivered scCODA abundance results read next to all of it. DRVI only. |

`01_qc.ipynb` loops over whichever fits are on disk and prints a notice for the rest, so
it is readable while runs are still going. `02_explore.ipynb` and `03_progression.ipynb`
prefer `sst_k64_covar_long`, falling back to `sst_k64_covar` then `sst_k64`.

## Multi-GPU

`--devices N` trains under DDP. Four things about that path are not obvious, and
`ddp_trainer_kwargs` in `sst_drvi.py` exists to handle them:

- **`find_unused_parameters` is required, not an optimisation.** `dispersion="gene-batch"`
  fits one `px_r` row per library and `batch_representation="embedding"` one embedding row
  per library. With 902 libraries, nearly every row sees no cell on a given rank and so
  receives no gradient — plain `"ddp"` treats that as an error. The strategy used is
  `ddp_find_unused_parameters_true`.
- **`--batch-size` becomes per device, and that costs optimisation progress.** Adam steps
  once per minibatch, so four GPUs at 256 is an effective batch of 1024 and only
  203 updates per epoch against 813 at batch 256 — 40,600 steps in 200 epochs where a
  single-GPU run of the same length takes 162,600. At an unchanged `lr` each step travels
  about as far as before, so a DDP run lands roughly a quarter of the way along the same
  trajectory: the first 4×GPU fit reached a validation ELBO of 12246.9 where the
  single-GPU fit reached 12036.7. Either scale `--lr` with the batch (the linear scaling
  rule) or drop `--batch-size` to 64 to recover the step count. Effective batch and
  steps/epoch are both logged, and `devices` is recorded in the config stored in `uns`.
- **scvi-tools disables early stopping under DDP** (`scvi/train/_trainer.py`), so a
  distributed run always uses all `max_epochs`. Worse, `check_val_every_n_epoch` is only
  defaulted to 1 when early stopping or checkpointing is active — without asking for it
  explicitly a DDP run never validates and the history comes back with no `*_validation`
  metrics at all. It is requested explicitly.
- **Non-spawn DDP re-executes the whole script per rank.** Everything after `model.train`
  writes files, so `finish_distributed()` barriers, tears down the process group, and
  returns `False` on every rank but zero, which then returns early. The same
  re-execution had all four ranks opening one log file in `mode="w"` and writing at
  colliding offsets, which silently ate a line of rank 0's output in an early probe run;
  `setup_logging` now suffixes ranks above zero (`train_<run>.rank2.log`) so the named log
  is rank 0's clean record. Timings are also written to
  `models/<run>/<model>/timings.json`, since the log is not a reliable channel and a
  `--train-only` run has no embedding `uns` to carry them. Without it four
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
| `/scratch/sst-drvi/models/sst_k64_probe_{a,b,c,d}/drvi/` | Stage 1 batch/LR/precision probe: history only, no embedding (`--train-only`) |
| `/scratch/sst-drvi/models/sst_k64{,_covar}_long/` | the converged refits, plus `…/embeddings/sst_k64{,_covar}_long_*_embed.h5ad` |
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
| training | `--max-epochs` budget, KL warmup over a third of it (`S.resolve_kl_warmup`), early stopping on `elbo_validation` where available |

`dispersion="gene-batch"` fits one dispersion per gene per library from ~390 cells
each, which can diverge. Both runners check `torch.isfinite(module.px_r)` and log
`px_r_finite`; if that is `False`, retrain with `--dispersion gene`.

`ScviConfig` mirrors this on every shared knob — same `n_latent`, architecture, batch
handling, dispersion and schedule — so a difference between the two embeddings is
attributable to DRVI's split decoder rather than to the setup. Two things cannot match:
`gene_likelihood` is `nb`, because `pnb` is DRVI's log-space variant and `SCVI` does not
accept it; and scVI has no interpretability machinery. The KL schedule, by contrast, *is*
forced to match — both families resolve it through `S.resolve_kl_warmup`, because letting
each keep its own default is what made the first scVI fit's ELBO incomparable (see
*Convergence and the KL schedule*).
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
  covariate table read next to the association heatmap in `01_qc.ipynb`, not from the flag.
- **`Sex` and `Chemistry` are collinear with `library_prep`** (library nests in donor,
  donor in sex; chemistry is a property of the library), so the batch embedding can
  already represent them. Naming them separately still helps the decoder: it gets a
  2-level effect shared across all 902 libraries instead of having to learn it inside a
  902-row embedding.

`scale_continuous_covariates` records the transform it chose per covariate in
`uns["covariate_scaling"]`, because the `log1p` decision is data-dependent and cannot be
recovered from the saved model. The covariate table in `01_qc.ipynb` reports the exact
transform when handed that record and the generic "standardized" otherwise.

`scvi-tools 1.5.0` hardcodes `cont_values=None` inside
`get_effect_of_splits_out_of_distribution`, so the OOD traversal crashes on a decoder built
with continuous covariates. `S.calculate_interpretability` wraps it and substitutes zeros,
which is the correct reference point precisely because the covariates are z-scored (zero is
the mean). Remove the shim when upstream fixes it.

## Convergence and the KL schedule

Both models anneal the KL term from 0 up to 1.0 over `n_epochs_kl_warmup` epochs, and the
length of that ramp decides whether the run has a converged state at all.

- **The two families' defaults disagree.** `DRVITrainingPlan` defaults to
  `n_epochs_kl_warmup="auto"`, which resolves to `max_epochs` — the ramp always fills the
  whole run, so a longer budget just stretches it and the model never trains at a settled
  objective. scvi-tools' plain `TrainingPlan` defaults to a fixed 400 epochs, so a
  200-epoch scVI run **peaks at `kl_weight = 0.4975`** and minimises `recon + 0.5·KL` from
  start to finish.
- **That makes the ELBOs incomparable.** Both report the same quantity
  (`elbo = reconstruction + kl_local + kl_global/n`, `scvi/train/_trainingplans.py`), but a
  halved KL penalty simply buys a better reconstruction. The first scVI fit beat DRVI by 34
  nats on validation ELBO and by 34 on reconstruction with `kl_local` almost unchanged —
  the size of gain the schedule alone predicts. It was read as a model-quality difference
  before the schedules were checked.
- **So the warmup is set explicitly, once, for both.** `S.resolve_kl_warmup` returns
  `--kl-warmup-epochs` or a third of `--max-epochs`, and `S.plan_kwargs` is the single
  place both runners get their schedule from. Two thirds of the run then happens at
  `kl_weight = 1.0`.
- **And it is verified, not trusted.** `S.check_kl_schedule` reads `kl_weight` back out of
  the history after training and logs an error if it did not reach 0.99; the value is
  stored as `uns["timings"]["kl_weight_reached"]`. Section 3 of the QC notebook plots the
  schedules together.

A fit counts as converged only if the OLS slope of the last 50 validation ELBO epochs is
shallower than −0.1/epoch. None of the original 200-epoch fits clears that bar: all four
have their validation minimum on the final epoch, with slopes of −1.4/epoch (DDP) and
−0.6/epoch (single GPU) and no detectable flattening. `--reduce-lr-on-plateau` halves the
learning rate when the validation ELBO stalls, and unlike early stopping it survives DDP,
so it is the only mechanism by which a distributed run adapts.

### The converged fits

600 epochs at effective batch 1024, `lr=4e-3`, warmup 200, on 4x T4:

| run | val ELBO | slope/ep | `kl_max` | dims used | wall clock |
|---|---|---|---|---|---|
| `sst_k64_long` (drvi) | 11978.3 | −0.049 | 1.0000 | 61 / 64 | 390 min |
| `sst_k64_covar_long` (drvi) | 11969.1 | −0.048 | 1.0000 | 62 / 64 | 397 min |
| `sst_k64_covar_long` (scvi) | **11932.5** | −0.096 | 1.0000 | **20 / 64** | 163 min |

Three things worth carrying forward:

- **The DRVI-vs-scVI comparison is only meaningful here.** At 200 epochs scVI led by 34 nats
  on a KL weight half of DRVI's, which is uninterpretable. On the matched schedule it still
  leads, by 36.6 — the finding held, the earlier evidence for it did not.
- **scVI reaches that with a third of the dimensions.** Its `kl_local` drops from 38.97 at
  half KL weight to 27.55 at full, and it prunes to 20 live dimensions against DRVI's 62.
  The collapse is correct KL pruning, not an optimisation failure, and more KL weight prunes
  harder — so refitting scVI at a lower `k` would reproduce nearly the same ELBO while
  confounding model family with capacity.
- **Modelling the covariates is close to free**: 11978.3 to 11969.1 in validation ELBO.

## Performance notes

Measured on 4× Tesla T4 (15 GB each), 48 cores, 186 GB RAM, `--devices 4
--batch-size 256 --num-workers 0`: DRVI runs at ~39.5 s/epoch, i.e. ~2h10m for 200
epochs, against 54.7 s/epoch (~3h02m) for the same model on a single A10G. All four GPUs
sit at 89-94% utilisation and 6.5 GB, and the four ranks together hold ~38 GB RSS because
each loads the cohort itself.

`--num-workers N` adds dataloader workers; the runs are GPU-bound with `--num-workers 0`,
so it only helps on faster cards. Raising `--batch-size` changes the fit, and under DDP it
multiplies by the device count.

**Where the epoch goes.** DRVI's split decoder operates on `(n_obs, n_split, n_hidden)` and
its last layer emits `(n_obs, n_split, n_genes)` before the logsumexp aggregation
(`scvi/external/drvi/_base_components.py`). With all 64 latent dimensions split, that
intermediate is `256 × 64 × 18892 × 4 B` = **1.24 GB** at batch 256, and the layer alone is
~238 GFLOP per step — ~29 ms on a T4 at fp32 peak against 197 ms measured. The bottleneck
is that one tensor, not the dataloader and not communication, which has two consequences:

- Per-epoch compute is roughly **batch-size independent** (the same cells pass through the
  same layers either way); only the per-step overheads multiply. Halving `--batch-size`
  therefore costs far less than 2× per epoch, which is what makes recovering the step count
  affordable.
- **fp16 is the lever worth trying.** `--precision 16-mixed` reaches Lightning through
  scvi-tools' `Trainer(**kwargs)`; a T4's tensor cores are ~8× its fp32 peak and fp16 halves
  that 1.24 GB. The risk is overflow in the logsumexp over a log-space negative binomial, so
  check `px_r_finite` and the history for non-finite values — both runners do.

**Batch and learning rate were settled by measurement** (`run_probe.sh`, 4 arms x 40 epochs).
Per-device 64 at `lr=1e-3` and per-device 256 at `lr=4e-3` produced *identical* per-epoch
trajectories (12317.7 vs 12317.0 at epoch 39; trailing slopes −8.03 vs −8.09) — the linear
scaling rule holds exactly here — but the larger batch runs at 40.2 s/epoch against 68.0, so
it is 1.69x faster for the same progress. Decomposing the two points gives ~45.6 ms of
per-step overhead and ~30.9 s of batch-size-independent per-epoch compute, which is a floor:
doubling the batch again would buy only 12%, and no batch size beats 30.9 s/epoch.

**Mixed precision does not help this model.** fp16 measured 7% faster (37.3 vs 40.2 s/epoch)
with an ELBO 1.9 nats worse. Under `torch.autocast` the batched matmul runs in fp16, but
`logsumexp`, `exp` and `log` are all kept in fp32 — so DRVI's split aggregation upcasts the
decoder's fp16 output straight back to a 1.24 GB fp32 tensor, and the `pnb` likelihood is
fp32 throughout. Only the GEMM sped up, and the GEMM is not the bottleneck. bf16 and TF32
are unavailable on a T4 regardless: it is compute capability 7.5, and both need 8.0.
`torch.cuda.is_bf16_supported()` returns `True` there — pass `including_emulation=False` to
get the honest answer.

**`px_r` is 70% of the model.** `dispersion="gene-batch"` over 902 libraries makes it
`18892 × 902` = 17,040,584 of DRVI's 24,382,516 parameters (58% of scVI's 29,255,633), and
therefore 70% of the ~97 MB all-reduced every step. Switching to `dispersion="gene"` would
remove that and the need for `find_unused_parameters`, but it is a modelling change, not a
free speedup.

`--train-only` stops after the model and loss history are written, skipping the embedding,
interpretability scores and UMAP (~11 min on the full cohort) — for runs whose only product
is the loss curves, such as a batch/LR probe.

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
