#!/usr/bin/env python
"""Build the full multiregion cohort for DRVI: genes from the region files, cells
and CPS_*/QC-flag enrichment from the GPBoost inputs.

Genes and raw counts come from the 10 per-region "final-nuclei" h5ad files, which
share one identical 36,601-gene panel across every region -- unlike the 207
per-supertype GPBoost "goodcells" files, which are each pre-filtered to that
supertype's own genes and would shrink the panel considerably if intersected. Cell
membership (the QC-passed "goodcells" set) and the CPS_*/QC-flag obs columns the
region files lack instead come from the GPBoost files, joined on the
``exp_component_name`` cell barcode both file sets share.

Runs in three RAM-bounded phases so the ~270 GB merged cohort is never held in
memory at once:

  0. obs-only pass over all 207 GPBoost files -> the authoritative cell-ID set and
     the CPS_*/QC-flag enrichment table (a few GB, cached to parquet).
  1. backed, chunked pass over each region file -> per-chunk h5ad files under
     ``prepared/chunks/``, filtered to goodcells and coerced to raw-count CSR
     float32.
  2. ``anndata.experimental.concat_on_disk`` merges the chunk files into one h5ad
     without materializing the whole matrix in RAM.
  3. obs-only enrichment pass attaches CPS_*/QC-flag columns to the merged file in
     place (``/X`` is never re-read or re-written here).

Reads only from ``/data`` (immutable); writes only to ``/scratch``.

    python prepare_full.py [--regions LEC MTG] [--chunk-size 250000] [--force]
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.experimental import concat_on_disk
from anndata.io import sparse_dataset, write_elem
from scipy import sparse

import full_drvi as S

logger = logging.getLogger("full_drvi.prepare")

DEFAULT_CHUNK_SIZE = 250_000


def _release_memory() -> None:
    """Return freed chunk memory to the OS instead of letting glibc retain it.

    A resumed run's RSS climbed to ~109 GB (of 186 GB) after processing several
    large regions, despite each chunk's own footprint being an order of
    magnitude smaller -- consistent with malloc arena retention across many
    large sparse-array alloc/free cycles rather than a live-object leak.
    ``malloc_trim`` is glibc-specific; harmless no-op elsewhere.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--regions",
        nargs="*",
        choices=S.REGIONS,
        default=None,
        help="subset of regions to process (default: all 10 -- pass one for a quick check)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"cells per chunk written during Phase 1 (default: {DEFAULT_CHUNK_SIZE})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"output h5ad (default: {S.PREPARED_FULL})",
    )
    p.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    p.add_argument(
        "--reindex", action="store_true", help="rebuild the cached GPBoost file index"
    )
    p.add_argument(
        "--rebuild-goodcells",
        action="store_true",
        help="rebuild the cached Phase 0 goodcells/enrichment table",
    )
    return p.parse_args()


# ------------------------------------------------------------------------ Phase 0

def build_goodcells_table(index: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """obs-only pass over every GPBoost file -> cell-ID set + CPS_*/QC-flag table.

    Cached to parquet: this is metadata-only I/O (never touches ``X``), but 207
    file opens over the network-backed dataset still takes real time.
    """
    cache = S.PREPARED_DIR / "goodcells_obs.parquet"
    if cache.exists() and not force:
        logger.info("reusing goodcells table %s", cache)
        return pd.read_parquet(cache)

    logger.info("Phase 0: reading obs from %d GPBoost files", len(index))
    cols = list(S.GPBOOST_ONLY_OBS)
    frames = []
    t0 = time.perf_counter()
    for i, row in enumerate(index.itertuples()):
        obs = ad.read_h5ad(row.path, backed="r").obs
        present = [c for c in cols if c in obs.columns]
        missing = [c for c in cols if c not in obs.columns]
        if missing:
            logger.warning("  %s: missing obs columns %s", row.supertype, missing)
        frames.append(obs[present])
        if (i + 1) % 25 == 0:
            logger.info("  ...%d/%d files (%.1fs)", i + 1, len(index), time.perf_counter() - t0)

    table = pd.concat(frames)
    n_dupes = int(table.index.duplicated().sum())
    if n_dupes:
        logger.warning(
            "%d duplicate exp_component_name values across GPBoost files; keeping first",
            n_dupes,
        )
        table = table[~table.index.duplicated(keep="first")]

    logger.info(
        "goodcells table: %d cells (expected %d), %.1fs",
        len(table),
        S.EXPECTED_N_FULL,
        time.perf_counter() - t0,
    )
    if len(table) != S.EXPECTED_N_FULL:
        logger.warning(
            "goodcells count %d != expected %d from the annotation table",
            len(table),
            S.EXPECTED_N_FULL,
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache)
    return table


# ------------------------------------------------------------------------ Phase 1

def process_region(
    region: str,
    goodcells_index: pd.Index,
    chunk_size: int,
    reference_var_names: np.ndarray | None,
    force: bool,
) -> tuple[np.ndarray, int, int]:
    """Write ``region``'s goodcells rows to chunk h5ads under ``S.CHUNKS_DIR``.

    Returns ``(var_names, n_written, n_total)`` so the caller can cross-check gene
    panels across regions and tally the overall match rate.
    """
    path = S.resolve_region_file(region)
    logger.info("region %-4s: opening %s", region, path.name)
    t0 = time.perf_counter()
    # Deliberately not ``ad.read_h5ad(path, backed="r")``: anndata's backed mode
    # only keeps X/layers lazy and still eagerly loads *every* obs column (one
    # HDF5 read each) -- 43 minutes for a 213 GB/1.18M-cell region's 132
    # columns, measured. Reading h5py directly and keeping only the ~36 needed
    # columns cut that to 15s on a comparably sized file.
    f = h5py.File(path, "r")
    n, n_vars = (int(x) for x in f["layers"][S.RAW_COUNTS_LAYER].attrs["shape"])
    var_index_key = f["var"].attrs.get("_index", "index")
    var_names = np.array([g.decode() if isinstance(g, bytes) else g for g in f["var"][var_index_key][:]])
    if reference_var_names is not None and not np.array_equal(var_names, reference_var_names):
        raise RuntimeError(f"region {region!r} has a different gene panel than the reference region")

    keep_cols = [c for c in S.REGION_OBS_KEEP if c in f["obs"]]
    missing = [c for c in S.REGION_OBS_KEEP if c not in f["obs"]]
    if missing:
        logger.warning("region %s: missing obs columns %s", region, missing)
    obs_full = S.read_obs_columns(f, keep_cols)
    umis = sparse_dataset(f["layers"][S.RAW_COUNTS_LAYER])
    logger.info("region %-4s: %d cells, %d genes (opened in %.1fs)", region, n, n_vars, time.perf_counter() - t0)

    n_chunks = math.ceil(n / chunk_size)
    n_written = 0
    for i in range(n_chunks):
        # chunk_size is embedded in the filename so a resumed run with a
        # different --chunk-size can never mistake a stale, differently-sized
        # cached chunk for a complete one (caught the hard way: cached 50k/100k
        # chunks from earlier validation runs were silently accepted as
        # "chunk 0 done" under a 250k scheme, dropping most of that region).
        out = S.CHUNKS_DIR / f"{region}_{chunk_size}_{i:04d}.h5ad"
        start, end = i * chunk_size, min((i + 1) * chunk_size, n)
        if out.exists() and not force:
            n_written += out_n_obs_cached(out)
            continue

        t = time.perf_counter()
        x = umis[start:end]
        if not sparse.isspmatrix_csr(x):
            x = sparse.csr_matrix(x)
        x = x.astype(np.float32, copy=False)
        obs = obs_full.iloc[start:end].copy()

        out_adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=var_names))
        mask = out_adata.obs_names.isin(goodcells_index)
        out_adata = out_adata[mask].copy()

        out.parent.mkdir(parents=True, exist_ok=True)
        # lzf trades a larger file for ~7x faster writes (measured); chunks are
        # merged away in Phase 2, so their on-disk size doesn't matter and
        # /scratch has effectively unbounded capacity (see the plan's Context).
        out_adata.write_h5ad(out, compression="lzf")
        n_matched = out_adata.n_obs
        n_written += n_matched
        del x, obs, out_adata, mask
        _release_memory()
        logger.info(
            "  %-4s chunk %04d: %d/%d matched goodcells (%.1fs)",
            region, i, n_matched, end - start, time.perf_counter() - t,
        )

    f.close()
    logger.info("region %-4s: %d/%d cells matched goodcells", region, n_written, n)
    return var_names, n_written, n


def out_n_obs_cached(path: Path) -> int:
    """``n_obs`` of an already-written chunk, from ``X`` metadata only."""
    with h5py.File(path, "r") as f:
        return int(f["X"].attrs["shape"][0])


# ------------------------------------------------------------------------ Phase 3

def enrich_obs(output: Path, goodcells: pd.DataFrame) -> None:
    """Attach CPS_*/QC-flag columns to the merged file's obs, in place.

    obs-only: ``/X`` is never re-read or re-written.
    """
    merged = ad.read_h5ad(output, backed="r")
    obs = merged.obs.copy()
    merged.file.close()
    del merged

    n_unmatched_in_output = int((~obs.index.isin(goodcells.index)).sum())
    if n_unmatched_in_output:
        logger.warning(
            "%d cells in the merged output have no matching GPBoost enrichment row",
            n_unmatched_in_output,
        )
    n_orphan_goodcells = int((~goodcells.index.isin(obs.index)).sum())
    if n_orphan_goodcells:
        logger.warning(
            "%d GPBoost goodcells were not found in any region file's output",
            n_orphan_goodcells,
        )

    new_obs = obs.join(goodcells, how="left")
    for col in S.OBS_CATEGORICAL:
        if col in new_obs.columns:
            new_obs[col] = new_obs[col].astype("category")

    with h5py.File(output, "r+") as f:
        del f["obs"]
        write_elem(f, "obs", new_obs)


def main() -> None:
    args = parse_args()
    output = args.output or S.PREPARED_FULL
    S.setup_logging(S.LOGS_DIR / "prepare_full.log")

    if output.exists() and not args.force:
        logger.info("output exists, skipping: %s (use --force to rebuild)", output)
        return

    S.log_provenance()
    regions = args.regions or list(S.REGIONS)
    logger.info("config: regions=%s chunk_size=%d output=%s", regions, args.chunk_size, output)
    t0 = time.perf_counter()

    index = S.build_gpboost_file_index(force=args.reindex)
    goodcells = build_goodcells_table(index, force=args.rebuild_goodcells)
    goodcells_index = goodcells.index

    logger.info("Phase 1: chunked extraction from %d region file(s)", len(regions))
    reference_var_names = None
    total_written, total_seen = 0, 0
    for region in regions:
        var_names, n_written, n_total = process_region(
            region, goodcells_index, args.chunk_size, reference_var_names, args.force
        )
        reference_var_names = var_names
        total_written += n_written
        total_seen += n_total

    logger.info(
        "Phase 1 done: %d/%d region cells matched goodcells across %d region(s)",
        total_written, total_seen, len(regions),
    )

    logger.info("Phase 2: on-disk merge")
    # Scoped to this run's chunk_size (see the naming comment in process_region)
    # so a stale, differently-sized cache from an earlier run can never be
    # double-counted alongside this run's own chunks for the same region.
    chunk_files = sorted(
        f for r in regions for f in S.CHUNKS_DIR.glob(f"{r}_{args.chunk_size}_*.h5ad")
    )
    if not chunk_files:
        raise RuntimeError("no chunk files found to merge")
    building = output.with_suffix(".building.h5ad")
    building.unlink(missing_ok=True)
    # Passed as Path objects, not str: concat_on_disk's single-input-file
    # shortcut calls ``.is_dir()`` on each entry, which a plain str lacks.
    concat_on_disk(chunk_files, building, join="outer")

    logger.info("Phase 3: obs enrichment (CPS_*/QC flags)")
    enrich_obs(building, goodcells)

    # --- sanity checks before the final rename ------------------------------
    final = ad.read_h5ad(building, backed="r")
    assert final.obs_names.is_unique, "duplicate cell barcodes after concat"
    sample = final.X[: min(5000, final.n_obs)]
    if hasattr(sample, "to_memory"):
        sample = sample.to_memory()
    assert sample.min() >= 0, "negative values in X"
    assert np.allclose(sample.data, np.round(sample.data)), "X is not raw counts"
    n_obs, n_vars = final.n_obs, final.n_vars
    final.file.close()
    del final

    cohort = {
        "regions": regions,
        "n_gpboost_files": int(len(index)),
        "join_key": S.CELL_ID_COL,
        "raw_counts_layer": S.RAW_COUNTS_LAYER,
        "chunk_size": args.chunk_size,
        "n_genes": int(n_vars),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": Path(__file__).name,
    }
    with h5py.File(building, "r+") as f:
        if "uns" in f:
            del f["uns"]
        write_elem(f, "uns", {"cohort": cohort})

    logger.info("cohort: %d cells x %d genes", n_obs, n_vars)
    if len(regions) == len(S.REGIONS) and n_obs != S.EXPECTED_N_FULL:
        logger.warning("n_obs=%d, expected %d from the annotation table", n_obs, S.EXPECTED_N_FULL)

    building.rename(output)
    logger.info(
        "done in %.1f min, %.1f GB on disk",
        (time.perf_counter() - t0) / 60,
        output.stat().st_size / 1e9,
    )


if __name__ == "__main__":
    main()
