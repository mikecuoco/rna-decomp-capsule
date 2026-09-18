#!/usr/bin/env python
"""Train DRVI on the full SEA-AD multiregion cohort.

Everything GPU-bound -- training, the latent representation, interpretability
scores, latent UMAP -- happens here, mirroring
``code/01_sst_drvi/train_drvi.py``, so a downstream notebook only ever reads a
small embedding h5ad.

The one structural difference from the SST runner: the prepared cohort here is
~270 GB (6,013,346 cells x 36,601 genes), far larger than this box's 186 GB RAM
even once, so it is never loaded fully into memory. ``S.load_prepared`` opens it
backed (``anndata.read_h5ad(..., backed="r")``), and scvi-tools' data loading is
backed-aware (``adata.isbacked``) -- it reads each minibatch's rows lazily from
the h5ad file rather than requiring the whole matrix resident, which is what
makes training against this file possible at all. Two consequences follow:

* No DDP by default. The SST runner replicates the *entire* in-memory cohort
  into every DDP rank's process (measured 38 GB RSS across 4 ranks for a 2.2 GB
  cohort there) -- at this dataset's scale that pattern does not fit even one
  rank's worth of RAM, let alone several. ``--devices`` defaults to 1;
  multi-device runs are only sound here if the loading model changes too, which
  it has not.
* ``--num-workers`` defaults to 0. h5py file handles are not generally safe to
  share across a DataLoader's forked/spawned worker processes; single-process
  data loading in the main process avoids that hazard entirely. Raise it only
  after verifying your own scvi-tools/h5py build tolerates it.

    python train_full.py [--n-latent 64] [--max-epochs 200] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import full_drvi as S

logger = logging.getLogger("full_drvi.train")

SMOKE_CELLS = 20_000
SMOKE_EPOCHS = 3


def parse_args() -> argparse.Namespace:
    default = S.DrviConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-latent", type=int, default=default.n_latent)
    p.add_argument("--n-split-latent", type=int, default=default.n_split_latent)
    p.add_argument("--split-method", default=default.split_method)
    p.add_argument("--split-aggregation", default=default.split_aggregation)
    p.add_argument("--gene-likelihood", default=default.gene_likelihood)
    p.add_argument(
        "--dispersion",
        default=default.dispersion,
        choices=["gene", "gene-batch", "gene-label", "gene-cell"],
    )
    p.add_argument("--batch-representation", default=default.batch_representation)
    p.add_argument("--n-hidden", type=int, default=default.n_hidden)
    p.add_argument("--n-layers", type=int, default=default.n_layers)
    p.add_argument("--batch-key", default=default.batch_key)
    p.add_argument("--categorical-covariates", nargs="*", default=[])
    p.add_argument("--continuous-covariates", nargs="*", default=[])
    p.add_argument("--encode-covariates", action="store_true")
    p.add_argument("--batch-size", type=int, default=default.batch_size)
    p.add_argument("--max-epochs", type=int, default=default.max_epochs)
    p.add_argument("--lr", type=float, default=default.lr)
    p.add_argument("--kl-warmup-epochs", type=int, default=None)
    p.add_argument("--reduce-lr-on-plateau", action="store_true")
    p.add_argument("--lr-patience", type=int, default=default.lr_patience)
    p.add_argument("--lr-factor", type=float, default=default.lr_factor)
    p.add_argument("--precision", default=None)
    p.add_argument("--early-stopping-patience", type=int, default=default.early_stopping_patience)
    p.add_argument("--train-size", type=float, default=default.train_size)
    p.add_argument("--seed", type=int, default=default.seed)
    p.add_argument(
        "--devices",
        type=int,
        default=default.devices,
        help="1 by default -- see the module docstring on why DDP is not the default here",
    )
    p.add_argument("--vanished-threshold", type=float, default=default.vanished_threshold)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 by default -- see the module docstring on backed h5py + multiprocessing",
    )
    p.add_argument("--input", type=Path, default=S.PREPARED_FULL)
    p.add_argument("--suffix", default=None, help="alternate run name suffix")
    p.add_argument("--smoke", action="store_true", help="20k-cell, 3-epoch end-to-end check")
    p.add_argument("--train-only", action="store_true", help="skip embedding/interpretability/UMAP")
    p.add_argument("--force", action="store_true", help="retrain even if the embedding exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = replace(
        S.DrviConfig(),
        n_latent=args.n_latent,
        n_split_latent=args.n_split_latent,
        split_method=args.split_method,
        split_aggregation=args.split_aggregation,
        gene_likelihood=args.gene_likelihood,
        dispersion=args.dispersion,
        batch_representation=args.batch_representation,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        batch_key=args.batch_key,
        categorical_covariate_keys=tuple(args.categorical_covariates),
        continuous_covariate_keys=tuple(args.continuous_covariates),
        encode_covariates=args.encode_covariates,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        kl_warmup_epochs=args.kl_warmup_epochs,
        reduce_lr_on_plateau=args.reduce_lr_on_plateau,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        precision=args.precision,
        early_stopping_patience=args.early_stopping_patience,
        train_size=args.train_size,
        seed=args.seed,
        devices=args.devices,
        vanished_threshold=args.vanished_threshold,
    )
    if args.smoke:
        # Exercises the path where the KL ramp finishes and the kl_weight guard
        # passes, same reasoning as the SST runner's smoke mode.
        cfg = replace(cfg, max_epochs=SMOKE_EPOCHS)

    run = S.run_name(cfg.n_latent) + (f"_{args.suffix}" if args.suffix else "") + ("_smoke" if args.smoke else "")
    model_path = S.MODELS_DIR / run / "drvi"
    out_embed = S.EMBED_DIR / f"{run}_drvi_embed.h5ad"

    S.setup_logging(S.LOGS_DIR / f"train_full_{run}.log")
    if out_embed.exists() and not args.force and not args.train_only:
        logger.info("output exists, skipping: %s (use --force to rebuild)", out_embed)
        return

    S.log_provenance()
    logger.info("config: %s", cfg.as_dict())
    logger.info("run=%s model_path=%s out_embed=%s", run, model_path, out_embed)
    t0 = time.perf_counter()

    if args.smoke:
        # Only the smoke path is ever materialized in RAM: a random subset is
        # cheap regardless of the source file's size. The real (non-smoke) path
        # never calls .to_memory() -- see the module docstring.
        backed = S.load_prepared(args.input, backed="r")
        rng = np.random.default_rng(cfg.seed)
        n = min(SMOKE_CELLS, backed.n_obs)
        idx = np.sort(rng.choice(backed.n_obs, size=n, replace=False))
        adata = backed[idx, :].to_memory()
        backed.file.close()
        adata.obs[cfg.batch_key] = adata.obs[cfg.batch_key].cat.remove_unused_categories()
        logger.info("smoke subsample: %d cells", adata.n_obs)
    else:
        adata = S.load_prepared(args.input, backed="r")
        logger.info("backed cohort: %d cells x %d genes", adata.n_obs, adata.n_vars)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    steps_per_epoch = int(adata.n_obs * cfg.train_size) // (cfg.batch_size * max(cfg.devices, 1))
    logger.info(
        "effective batch=%d (batch_size=%d x devices=%d), ~%d steps/epoch",
        cfg.batch_size * cfg.devices, cfg.batch_size, cfg.devices, steps_per_epoch,
    )

    plan = S.plan_kwargs(cfg)
    if cfg.continuous_covariate_keys:
        scaled_names = S.scale_continuous_covariates(adata, cfg.continuous_covariate_keys)
        cfg = replace(cfg, continuous_covariate_keys=tuple(scaled_names))

    model = S.build_drvi_model(adata, cfg)
    ddp = S.ddp_trainer_kwargs(cfg.devices)
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
    if not S.finish_distributed():
        return

    timings = {"train_seconds": time.perf_counter() - t0}

    def save_timings() -> None:
        (model_path.parent / "timings.json").write_text(json.dumps(timings, indent=2))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), overwrite=True)
    history = pd.concat(model.history.values(), axis=1)
    history.to_csv(model_path.parent / "history.csv")
    timings["kl_weight_reached"] = S.check_kl_schedule(history, plan["n_epochs_kl_warmup"], cfg.max_epochs)

    px_r_finite = bool(torch.isfinite(model.module.px_r).all().item())
    if not px_r_finite:
        logger.error("px_r has non-finite values; consider retrying with --dispersion gene")
    save_timings()

    if args.train_only:
        logger.info("--train-only: stopping before embedding/interpretability/UMAP")
        return

    embed = S.latent_embedding(model, adata, cfg)
    model.set_latent_dimension_stats(embed, vanished_threshold=cfg.vanished_threshold)
    n_used = int((~embed.var["vanished"]).sum()) if "vanished" in embed.var else embed.n_vars
    logger.info("latent dims: %d/%d used", n_used, embed.n_vars)

    S.calculate_interpretability(model, embed, methods=("IND", "OOD"), directional=True)
    embed.uns["gene_names"] = adata.var_names.to_numpy()
    if "covariate_scaling" in adata.uns:
        embed.uns["covariate_scaling"] = adata.uns["covariate_scaling"]

    sc.pp.neighbors(embed, use_rep="X")
    sc.tl.umap(embed)

    embed.uns["drvi_config"] = json.dumps(cfg.as_dict())
    embed.uns["drvi_run"] = {
        "run": run,
        "model_path": str(model_path),
        "input": str(args.input),
        "n_epochs": cfg.max_epochs,
        "smoke": args.smoke,
    }
    embed.uns["timings"] = timings

    out_embed.parent.mkdir(parents=True, exist_ok=True)
    embed.write_h5ad(out_embed, compression="gzip")
    save_timings()
    logger.info("done in %.1f min -> %s", (time.perf_counter() - t0) / 60, out_embed)


if __name__ == "__main__":
    main()
