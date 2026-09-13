#!/usr/bin/env python
"""Train DRVI on the prepared SEA-AD SST cohort and write an interpretable embedding.

Model: ``scvi.external.DRVI`` (scvi-tools >= 1.5). Everything GPU-bound -- training,
the latent representation, the interpretability scores and the latent UMAP -- happens
here, so ``01_inspect_factors.ipynb`` only has to read a small h5ad.

    python train_drvi.py [--smoke] [--max-epochs N] [--batch-size N] [--force]
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

logger = logging.getLogger("sst_drvi.train")

SMOKE_CELLS = 20_000
SMOKE_EPOCHS = 3


def parse_args() -> argparse.Namespace:
    cfg = S.DrviConfig()
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
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--lr", type=float, default=cfg.lr, help="Adam learning rate")
    p.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=cfg.kl_warmup_epochs,
        help="epochs over which to ramp the KL weight to 1.0. Unset uses a third of "
        "--max-epochs, which leaves two thirds of the run at a settled objective. DRVI's "
        "own default ('auto') ramps over the whole run, so it never reaches a state whose "
        "convergence can be judged",
    )
    p.add_argument(
        "--precision",
        default=cfg.precision,
        help="Lightning precision, e.g. 16-mixed. On a T4 the tensor cores are ~8x the "
        "fp32 peak and fp16 halves the (batch, n_split, n_genes) decoder intermediate, "
        "but the logsumexp over a log-space NB can overflow -- check px_r_finite",
    )
    p.add_argument(
        "--reduce-lr-on-plateau",
        action="store_true",
        help="halve the learning rate when elbo_validation stops improving. Unlike early "
        "stopping this survives DDP, so it is the only way a distributed run adapts",
    )
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
        "--continuous-covariates",
        nargs="*",
        default=list(cfg.continuous_covariate_keys),
        metavar="OBS_COL",
        help="obs columns to model as continuous covariates so the latent space does not "
        'have to encode them, e.g. --continuous-covariates "Fraction mitochondrial UMIs" '
        '"Genes detected". Note these are only partly technical in neurons -- regressing '
        "out mitochondrial content can remove real metabolic signal",
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
    p.add_argument(
        "--train-only",
        action="store_true",
        help="stop after the model and loss history are written, skipping the embedding, "
        "interpretability scores and UMAP (~11 min on the full cohort). For runs whose "
        "only product is the loss curves, such as a batch/LR probe",
    )
    p.add_argument("--force", action="store_true", help="retrain even if the embedding exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        S.DrviConfig(),
        n_latent=args.n_latent,
        max_epochs=SMOKE_EPOCHS if args.smoke else args.max_epochs,
        batch_size=args.batch_size,
        batch_key=args.batch_key,
        dispersion=args.dispersion,
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
    model_path = S.MODELS_DIR / run / "drvi"
    out_embed = S.EMBED_DIR / f"{run}_drvi_embed.h5ad"

    S.setup_logging(S.LOGS_DIR / f"train_{run}.log")
    if out_embed.exists() and not args.force and not args.train_only:
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
    plan = S.plan_kwargs(cfg)
    logger.info(
        "lr=%g KL warmup=%d of %d epochs (%d at kl_weight=1.0) precision=%s",
        cfg.lr,
        plan["n_epochs_kl_warmup"],
        cfg.max_epochs,
        max(cfg.max_epochs - plan["n_epochs_kl_warmup"], 0),
        cfg.precision or "32-true",
    )

    # --------------------------------------------------------------------- train
    if cfg.continuous_covariate_keys:
        logger.info("standardizing %d continuous covariate(s)", len(cfg.continuous_covariate_keys))
        scaled = S.scale_continuous_covariates(adata, cfg.continuous_covariate_keys)
        # the model is registered on the scaled columns; the config keeps the source names
        cfg = replace(cfg, continuous_covariate_keys=tuple(scaled))

    model = S.build_drvi_model(adata, cfg)
    logger.info("%s", model)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # scvi-tools turns early stopping off under DDP anyway; say so rather than passing a
    # setting that is silently dropped.
    ddp = S.ddp_trainer_kwargs(cfg.devices)
    if ddp:
        logger.info("DDP: %s (early stopping unavailable, running all %d epochs)",
                    ddp["strategy"], cfg.max_epochs)
    t = time.perf_counter()
    model.train(
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        train_size=cfg.train_size,
        early_stopping=not ddp,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_monitor="elbo_validation",
        # Shared with train_scvi.py so the two families cannot drift onto different KL
        # schedules; see S.resolve_kl_warmup for why that matters.
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

    def save_timings() -> None:
        """Persist timings beside the model and log them.

        On disk as well as in the log because the four DDP ranks share one log file and
        can clobber each other's lines, and because a ``--train-only`` run writes no
        embedding whose ``uns`` would otherwise carry them.
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        (model_path.parent / "timings.json").write_text(json.dumps(timings, indent=2))
        logger.info("timings: %s", {k: (round(v, 2) if isinstance(v, float) else v)
                                    for k, v in timings.items()})

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
    history.to_csv(model_path.parent / "history.csv")
    logger.info("history metrics: %s", list(history.columns))
    timings["kl_weight_reached"] = S.check_kl_schedule(
        history, plan["n_epochs_kl_warmup"], cfg.max_epochs
    )
    logger.info("saved model -> %s", model_path)

    # `dispersion="gene-batch"` fits one dispersion per gene per library. With ~600
    # libraries and ~390 cells each, all-zero (gene, batch) cells can drive px_r to
    # +/-inf; a non-finite px_r makes every downstream score meaningless, so check it.
    px_r_finite = bool(torch.isfinite(model.module.px_r).all().item())
    timings["px_r_finite"] = px_r_finite
    logger.info("px_r_finite: %s (dispersion=%s)", px_r_finite, cfg.dispersion)
    if not px_r_finite:
        logger.error(
            "px_r contains non-finite values -- retrain with --dispersion gene before "
            "trusting the interpretability scores"
        )

    if args.train_only:
        logger.info(
            "--train-only: stopping after %d epochs of history; no embedding written",
            n_epochs_run,
        )
        save_timings()
        return

    # ------------------------------------------------------------------ embedding
    t = time.perf_counter()
    embed = S.latent_embedding(model, adata, cfg)
    model.set_latent_dimension_stats(embed, vanished_threshold=cfg.vanished_threshold)
    n_vanished = int(embed.var["vanished"].sum())
    logger.info(
        "latent dims: %d used, %d vanished (threshold %.2f)",
        embed.n_vars - n_vanished,
        n_vanished,
        cfg.vanished_threshold,
    )
    timings["embed_min"] = (time.perf_counter() - t) / 60

    # ------------------------------------------------------------ interpretability
    t = time.perf_counter()
    S.calculate_interpretability(embed=embed, model=model, methods=("IND", "OOD"), directional=True)
    embed.uns["gene_names"] = adata.var_names.to_numpy(dtype=str)
    if "covariate_scaling" in adata.uns:
        embed.uns["covariate_scaling"] = adata.uns["covariate_scaling"]
    timings["interpret_min"] = (time.perf_counter() - t) / 60
    logger.info(
        "interpretability score keys: %s (%.1f min)",
        sorted(embed.varm.keys()),
        timings["interpret_min"],
    )

    # -------------------------------------------------------------- latent UMAP
    t = time.perf_counter()
    sc.pp.neighbors(embed, use_rep="X")
    sc.tl.umap(embed)
    timings["umap_min"] = (time.perf_counter() - t) / 60
    logger.info("latent UMAP done (%.1f min)", timings["umap_min"])

    # ------------------------------------------------------------------- persist
    # h5ad `uns` cannot hold tuples or None, which DrviConfig has both of, so the
    # config round-trips as JSON.
    embed.uns["drvi_config"] = json.dumps(cfg.as_dict())
    embed.uns["drvi_run"] = {
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
    save_timings()


if __name__ == "__main__":
    main()
