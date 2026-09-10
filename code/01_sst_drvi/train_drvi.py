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
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
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
    out_embed = S.EMBED_DIR / f"{run}_embed.h5ad"

    S.setup_logging(S.LOGS_DIR / f"train_{run}.log")
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

    # --------------------------------------------------------------------- train
    if cfg.continuous_covariate_keys:
        logger.info("standardizing %d continuous covariate(s)", len(cfg.continuous_covariate_keys))
        scaled = S.scale_continuous_covariates(adata, cfg.continuous_covariate_keys)
        # the model is registered on the scaled columns; the config keeps the source names
        cfg = replace(cfg, continuous_covariate_keys=tuple(scaled))

    model = S.build_model(adata, cfg)
    logger.info("%s", model)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    t = time.perf_counter()
    model.train(
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        train_size=cfg.train_size,
        early_stopping=True,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_monitor="elbo_validation",
        # DRVI wants the KL warmup spread over the whole run; a short warmup collapses
        # the disentanglement the split decoder is meant to produce.
        plan_kwargs={"n_epochs_kl_warmup": cfg.max_epochs},
        accelerator=accelerator,
        devices=1,
        datasplitter_kwargs={
            "num_workers": args.num_workers,
            "persistent_workers": args.num_workers > 0,
        },
    )
    timings["train_min"] = (time.perf_counter() - t) / 60
    n_epochs_run = len(model.history["elbo_train"])
    logger.info(
        "trained %d epochs in %.1f min (%.1f s/epoch)",
        n_epochs_run,
        timings["train_min"],
        timings["train_min"] * 60 / max(n_epochs_run, 1),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), overwrite=True)
    history = model.history["elbo_train"].join(model.history["elbo_validation"], how="outer")
    history.to_csv(model_path.parent / "history.csv")
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
    logger.info("timings: %s", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in timings.items()})


if __name__ == "__main__":
    main()
