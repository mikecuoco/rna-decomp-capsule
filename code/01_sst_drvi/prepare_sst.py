#!/usr/bin/env python
"""Build the SST cohort for DRVI from the SEA-AD multiregion GPBoost inputs.

Concatenates the per-supertype "goodcells" count matrices belonging to
``Subclass == "Sst"`` into a single AnnData with raw counts in ``X``, restricted to
the genes shared by every source file.

Reads only from ``/data`` (immutable); writes only to ``/scratch``.

    python prepare_sst.py [--include-chodl] [--force]
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

import sst_drvi as S

logger = logging.getLogger("sst_drvi.prepare")

#: Expected cohort size for the default scope, from the master annotation table
#: (``Subclass == "Sst"`` and ``Used in analysis``).
EXPECTED_N_SST = 231_107
N_REGIONS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--include-chodl",
        action="store_true",
        help='also include the "Sst Chodl" subclass (+4,383 cells)',
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"output h5ad (default: {S.PREPARED_SST})",
    )
    p.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    p.add_argument(
        "--reindex", action="store_true", help="rebuild the cached GPBoost file index"
    )
    return p.parse_args()


def gene_intersection(paths: list[Path]) -> np.ndarray:
    """Genes present in every source file, read from ``var`` metadata only."""
    shared: set[str] | None = None
    for path in paths:
        with h5py.File(path, "r") as f:
            var = f["var"]
            index_key = var.attrs.get("_index", "index")
            genes = {g.decode() for g in var[index_key][:]}
        logger.info("  %-22s %6d genes", path.name.split("_goodcells")[0], len(genes))
        shared = genes if shared is None else shared & genes
    assert shared is not None
    return np.array(sorted(shared))


def load_one(path: Path, genes: np.ndarray, supertype: str) -> ad.AnnData:
    """Read one source file, subset to ``genes``, and strip everything we don't want.

    The SEA-AD per-supertype objects carry ``obsm['X_scVI']``/``obsm['X_umap']`` and a
    stale ``uns['log1p']`` from their own pipeline; none of that must leak into the
    cohort, so a fresh AnnData is built from ``X``/``obs`` alone.
    """
    src = ad.read_h5ad(path)
    src = src[:, genes]

    keep = [c for c in S.OBS_KEEP if c in src.obs.columns]
    missing = [c for c in S.OBS_KEEP if c not in src.obs.columns]
    if missing:
        logger.warning("  %s: missing obs columns %s", supertype, missing)

    x = src.X
    if not sparse.isspmatrix_csr(x):
        x = sparse.csr_matrix(x)
    x = x.astype(np.float32, copy=False)

    out = ad.AnnData(X=x, obs=src.obs[keep].copy(), var=pd.DataFrame(index=genes))
    out.obs["source_supertype"] = supertype
    del src
    return out


def main() -> None:
    args = parse_args()
    output = args.output or S.PREPARED_SST
    S.setup_logging(S.LOGS_DIR / "prepare_sst.log")

    if output.exists() and not args.force:
        logger.info("output exists, skipping: %s (use --force to rebuild)", output)
        return

    S.log_provenance()
    logger.info(
        "config: include_chodl=%s output=%s", args.include_chodl, output
    )
    t0 = time.perf_counter()

    index = S.build_file_index(force=args.reindex)
    selected = S.sst_files(index, include_chodl=args.include_chodl)
    logger.info(
        "%d source files, %d cells expected",
        len(selected),
        int(selected["n_obs"].sum()),
    )
    for row in selected.itertuples():
        logger.info(
            "  %-22s %-10s %8d cells %6d genes  [%s]",
            row.supertype,
            row.subclass,
            row.n_obs,
            row.n_vars,
            row.subdir,
        )

    paths = [Path(p) for p in selected["path"]]

    logger.info("pass 1/2: intersecting gene sets")
    genes = gene_intersection(paths)
    logger.info("shared genes: %d", len(genes))

    logger.info("pass 2/2: reading and concatenating")
    parts = []
    for path, supertype in zip(paths, selected["supertype"], strict=True):
        t = time.perf_counter()
        part = load_one(path, genes, supertype)
        parts.append(part)
        logger.info(
            "  %-22s %8d x %5d  (%.1fs, %.1f GB nnz)",
            supertype,
            part.n_obs,
            part.n_vars,
            time.perf_counter() - t,
            part.X.data.nbytes / 1e9,
        )

    adata = ad.concat(parts, join="outer", index_unique=None, merge="first")
    del parts

    # gene_ids (Ensembl) are stable across files; take them from the first source.
    with h5py.File(paths[0], "r") as f:
        var_index_key = f["var"].attrs.get("_index", "index")
        src_genes = np.array([g.decode() for g in f["var"][var_index_key][:]])
        src_ids = np.array([g.decode() for g in f["var"]["gene_ids"][:]])
    id_map = pd.Series(src_ids, index=src_genes)
    adata.var["gene_symbol"] = adata.var_names
    adata.var["gene_ids"] = id_map.reindex(adata.var_names).to_numpy()

    for col in S.OBS_CATEGORICAL:
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype("category")

    adata.uns["cohort"] = {
        "subclasses": sorted(selected["subclass"].unique().tolist()),
        "supertypes": selected["supertype"].tolist(),
        "source_files": [str(p) for p in paths],
        "n_shared_genes": int(len(genes)),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": Path(__file__).name,
    }

    # --- sanity checks before writing -------------------------------------------
    assert adata.obs_names.is_unique, "duplicate cell barcodes after concat"
    adata.var_names_make_unique()
    sample = adata.X[: min(5000, adata.n_obs)]
    assert sample.min() >= 0, "negative values in X"
    assert np.allclose(sample.data, np.round(sample.data)), "X is not raw counts"
    n_regions = adata.obs["Brain Region"].nunique()
    assert n_regions == N_REGIONS, f"{n_regions} regions, expected {N_REGIONS}"
    if not args.include_chodl:
        assert set(adata.obs["Subclass"].cat.categories) == {S.SST_SUBCLASS}
        if adata.n_obs != EXPECTED_N_SST:
            logger.warning(
                "n_obs=%d, expected %d from the annotation table",
                adata.n_obs,
                EXPECTED_N_SST,
            )

    logger.info("cohort: %d cells x %d genes", adata.n_obs, adata.n_vars)
    logger.info(
        "libraries=%d donors=%d regions=%d supertypes=%d",
        adata.obs["library_prep"].nunique(),
        adata.obs["Donor ID"].nunique(),
        n_regions,
        adata.obs["Supertype"].nunique(),
    )
    logger.info("cells per region:\n%s", adata.obs["Brain Region"].value_counts())

    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("writing %s", output)
    adata.write_h5ad(output, compression="gzip")
    logger.info(
        "done in %.1f min, %.1f GB on disk",
        (time.perf_counter() - t0) / 60,
        output.stat().st_size / 1e9,
    )


if __name__ == "__main__":
    main()
