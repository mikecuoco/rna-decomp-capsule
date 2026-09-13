#!/usr/bin/env python
"""Train scVI on the prepared SEA-AD SST cohort, as the baseline DRVI is compared against.

Model: ``scvi.model.SCVI``. Mirrors ``train_drvi.py`` on every shared setting -- latent
size, architecture, batch handling, dispersion, schedule -- so that a difference between
the two embeddings is attributable to DRVI's split decoder rather than to the setup.
What is deliberately absent is DRVI's interpretability machinery: scVI's dimensions are
not claimed to be individually meaningful, so there are no per-direction traversal scores
to compute, and the latent dimensions are named ``Z_n`` rather than ``DR_n``.

    python train_scvi.py [--devices 4] [--smoke] [--max-epochs N] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch

import sst_drvi as S

logger = logging.getLogger("sst_drvi.train_scvi")

SMOKE_CELLS = 20_000
SMOKE_EPOCHS = 3


def parse_args() -> argparse.Namespace:
    cfg = S.ScviConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-latent", type=int, default=cfg.n_latent)
    p.add_argument("--max-epochs", type=int, default=cfg.max_epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--batch-key", default=cfg.batch_key)
    p.add_argument(
        "--dispersion",
        default=cfg.dispersion,
        choices=["gene", "gene-batch", "gene-label", "gene-cell"],
    )
    p.add_argument(
        "--gene-likelihood",
        default=cfg.gene_likelihood,
        choices=["nb", "zinb", "poisson", "normal"],
        help='scVI has no "pnb"; the DRVI runs use that log-space variant',
    )
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument(
        "--devices",
        type=int,
        default=cfg.devices,
        help="GPUs to train on. >1 switches to DDP with find_unused_parameters (the "
        "gene-batch dispersion and the library embedding leave most rows without a "
        "gradient on any given rank). --batch-size is then PER DEVICE, so the effective "
        "batch is batch_size * devices, and scvi-tools disables early stopping",
    )
    p.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=cfg.kl_warmup_epochs,
        help="epochs over which to ramp the KL weight to 1.0. Unset uses a third of "
        "--max-epochs, the same rule as train_drvi.py, so the two families optimise the "
        "same objective and their ELBOs are comparable. Do not fall back to scvi's own "
        "400-epoch default: against a shorter run it caps the weight below 1.0",
    )
    p.add_argument("--lr", type=float, default=cfg.lr, help="Adam learning rate")
    p.add_argument(
        "--precision",
        default=cfg.precision,
        help="Lightning precision, e.g. 16-mixed",
    )
    p.add_argument(
        "--reduce-lr-on-plateau",
        action="store_true",
        help="halve the learning rate when elbo_validation stops improving; unlike early "
        "stopping this survives DDP",
    )
    p.add_argument(
        "--continuous-covariates",
        nargs="*",
        default=list(cfg.continuous_covariate_keys),
        metavar="OBS_COL",
        help="obs columns to model as continuous covariates so the latent space does not "
        "have to encode them; standardized first, as in train_drvi.py",
    )
    p.add_argument(
        "--categorical-covariates",
        nargs="*",
        default=list(cfg.categorical_covariate_keys),
        metavar="OBS_COL",
        help="additional obs columns to model as categorical covariates, beyond --batch-key",
    )
    p.add_argument(
        "--encode-covariates",
        action="store_true",
        help="also feed covariates to the encoder (scvi default is decoder-only, which "
        "leaves q(z|x) batch-blind and lets library structure leak into the latent)",
    )
    p.add_argument(
        "--suffix",
        default="",
        help="appended to the run name, to keep an alternative fit beside the baseline",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="dataloader workers; 0 keeps the in-memory AnnData path (usually fastest)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=S.PREPARED_SST,
        help=f"prepared counts h5ad (default: {S.PREPARED_SST})",
    )
    p.add_argument(
        "--include-chodl",
        action="store_true",
        help="label outputs as the Sst+Sst Chodl cohort (must match prepare_sst.py)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"subsample to {SMOKE_CELLS} cells and {SMOKE_EPOCHS} epochs to validate "
        "the full path quickly; writes to a *_smoke run name",
    )
    p.add_argument("--force", action="store_true", help="retrain even if the embedding exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        S.ScviConfig(),
        n_latent=args.n_latent,
        max_epochs=SMOKE_EPOCHS if args.smoke else args.max_epochs,
        batch_size=args.batch_size,
        batch_key=args.batch_key,
        dispersion=args.dispersion,
        gene_likelihood=args.gene_likelihood,
        seed=args.seed,
        devices=args.devices,
        lr=args.lr,
        # A smoke run keeps the default rule (a third of the budget) rather than being
        # pinned to its own epoch count, so it exercises the path where the ramp finishes
        # and the kl_weight guard passes.
        kl_warmup_epochs=args.kl_warmup_epochs,
        reduce_lr_on_plateau=args.reduce_lr_on_plateau,
        precision=args.precision,
        categorical_covariate_keys=tuple(args.categorical_covariates),
        continuous_covariate_keys=tuple(args.continuous_covariates),
        encode_covariates=args.encode_covariates,
    )

    run = (
        S.run_name(cfg.n_latent, args.include_chodl)
        + (f"_{args.suffix}" if args.suffix else "")
        + ("_smoke" if args.smoke else "")
    )
    model_path = S.MODELS_DIR / run / "scvi"
    out_embed = S.EMBED_DIR / f"{run}_scvi_embed.h5ad"

    S.setup_logging(S.LOGS_DIR / f"train_{run}_scvi.log")
    if out_embed.exists() and not args.force:
        logger.info("embedding exists, skipping: %s (use --force)", out_embed)
        return

    S.log_provenance()
    scvi.settings.seed = cfg.seed
    timings: dict[str, float | bool | int] = {}

    # ---------------------------------------------------------------- load cohort
    t = time.perf_counter()
    adata = S.load_prepared(args.input, backed=False)
    if args.smoke:
        rng = np.random.default_rng(cfg.seed)
        keep = rng.choice(adata.n_obs, size=min(SMOKE_CELLS, adata.n_obs), replace=False)
        adata = adata[np.sort(keep)].copy()
        # drop libraries that lost all their cells, or setup_anndata keeps empty levels
        adata.obs[cfg.batch_key] = adata.obs[cfg.batch_key].cat.remove_unused_categories()
    timings["load_min"] = (time.perf_counter() - t) / 60
    logger.info(
        "cohort=%s cells=%d genes=%d %s=%d",
        run,
        adata.n_obs,
        adata.n_vars,
        cfg.batch_key,
        adata.obs[cfg.batch_key].nunique(),
    )
    logger.info("config: %s", json.dumps(cfg.as_dict()))
    logger.info(
        "devices=%d batch_size=%d per device -> effective batch %d, %d optimizer steps/epoch",
        cfg.devices,
        cfg.batch_size,
        cfg.batch_size * cfg.devices,
        int(adata.n_obs * cfg.train_size) // (cfg.batch_size * cfg.devices),
    )

    # --------------------------------------------------------------------- train
    if cfg.continuous_covariate_keys:
        logger.info("standardizing %d continuous covariate(s)", len(cfg.continuous_covariate_keys))
        scaled = S.scale_continuous_covariates(adata, cfg.continuous_covariate_keys)
        # the model is registered on the scaled columns; the config keeps the source names
        cfg = replace(cfg, continuous_covariate_keys=tuple(scaled))

    model = S.build_scvi_model(adata, cfg)
    logger.info("%s", model)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    ddp = S.ddp_trainer_kwargs(cfg.devices)
    if ddp:
        logger.info("DDP: %s (early stopping unavailable, running all %d epochs)",
                    ddp["strategy"], cfg.max_epochs)
    plan = S.plan_kwargs(cfg)
    logger.info(
        "lr=%g KL warmup=%d of %d epochs (%d at kl_weight=1.0) precision=%s",
        cfg.lr,
        plan["n_epochs_kl_warmup"],
        cfg.max_epochs,
        max(cfg.max_epochs - plan["n_epochs_kl_warmup"], 0),
        cfg.precision or "32-true",
    )
    t = time.perf_counter()
    model.train(
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        train_size=cfg.train_size,
        early_stopping=not ddp,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_monitor="elbo_validation",
        plan_kwargs=plan,
        accelerator=accelerator,
        devices=cfg.devices,
        datasplitter_kwargs={
            "num_workers": args.num_workers,
            "persistent_workers": args.num_workers > 0,
        },
        **ddp,
        **S.precision_kwargs(cfg.precision),
    )
    timings["train_min"] = (time.perf_counter() - t) / 60

    # Non-spawn DDP re-runs this whole script once per GPU. Everything below writes
    # files, so only rank 0 may continue.
    if not S.finish_distributed():
        logger.info("rank %s done training; rank 0 handles the rest",
                    os.environ.get("LOCAL_RANK"))
        return

    n_epochs_run = len(model.history["elbo_train"])
    logger.info(
        "trained %d epochs in %.1f min (%.1f s/epoch)",
        n_epochs_run,
        timings["train_min"],
        timings["train_min"] * 60 / max(n_epochs_run, 1),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), overwrite=True)
    # every logged metric, not just the ELBO: the QC notebook plots reconstruction loss
    # too, and concatenating whatever is present cannot fail late in a long run.
    history = pd.concat(model.history.values(), axis=1)
    history.to_csv(model_path.parent / "history_scvi.csv")
    timings["kl_weight_reached"] = S.check_kl_schedule(
        history, plan["n_epochs_kl_warmup"], cfg.max_epochs
    )
    logger.info("history metrics: %s", list(history.columns))
    logger.info("saved model -> %s", model_path)

    # Same exposure as the DRVI runs: `dispersion="gene-batch"` fits one dispersion per
    # gene per library from ~390 cells, and all-zero (gene, library) cells can drive px_r
    # to +/-inf.
    px_r_finite = bool(torch.isfinite(model.module.px_r).all().item())
    timings["px_r_finite"] = px_r_finite
    logger.info("px_r_finite: %s (dispersion=%s)", px_r_finite, cfg.dispersion)
    if not px_r_finite:
        logger.error(
            "px_r contains non-finite values -- retrain with --dispersion gene before "
            "trusting this embedding"
        )

    # ------------------------------------------------------------------ embedding
    t = time.perf_counter()
    embed = S.latent_embedding(model, adata, cfg, prefix="Z")
    # scVI does not prune dimensions the way DRVI does, so there is no `vanished` flag to
    # write -- only the shared usage statistics, which is what lets the QC notebook
    # compare how much of n_latent each model actually spent.
    stats = S.latent_stats(embed, threshold=cfg.used_threshold)
    embed.var["title"] = [f"Z {i + 1}" for i in range(embed.n_vars)]
    for col in ("std", "mean_abs", "max_abs", "used"):
        embed.var[col] = stats[col].to_numpy()
    logger.info(
        "latent dims: %d of %d above |z| >= %.2f",
        int(stats["used"].sum()),
        embed.n_vars,
        cfg.used_threshold,
    )
    embed.uns["gene_names"] = adata.var_names.to_numpy(dtype=str)
    if "covariate_scaling" in adata.uns:
        embed.uns["covariate_scaling"] = adata.uns["covariate_scaling"]
    timings["embed_min"] = (time.perf_counter() - t) / 60

    # -------------------------------------------------------------- latent UMAP
    t = time.perf_counter()
    sc.pp.neighbors(embed, use_rep="X")
    sc.tl.umap(embed)
    timings["umap_min"] = (time.perf_counter() - t) / 60
    logger.info("latent UMAP done (%.1f min)", timings["umap_min"])

    # ------------------------------------------------------------------- persist
    # h5ad `uns` cannot hold tuples or None, which ScviConfig has both of, so the config
    # round-trips as JSON.
    embed.uns["scvi_config"] = json.dumps(cfg.as_dict())
    embed.uns["scvi_run"] = {
        "run": run,
        "model_path": str(model_path),
        "input": str(args.input),
        "n_epochs": n_epochs_run,
        "smoke": args.smoke,
    }
    embed.uns["timings"] = timings
    out_embed.parent.mkdir(parents=True, exist_ok=True)
    embed.write_h5ad(out_embed, compression="gzip")
    logger.info("wrote %s (%.2f GB)", out_embed, out_embed.stat().st_size / 1e9)
    # on disk as well as in the log: the four DDP ranks share one log file and can clobber
    # each other's lines
    (model_path.parent / "timings.json").write_text(json.dumps(timings, indent=2))
    logger.info("timings: %s", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in timings.items()})


if __name__ == "__main__":
    main()
