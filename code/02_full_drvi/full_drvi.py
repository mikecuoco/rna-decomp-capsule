"""Shared helpers for the SEA-AD multiregion full-dataset DRVI factorization.

Imported by ``prepare_full.py`` and ``train_full.py`` so that paths and constants
have exactly one definition. The DrviConfig/DDP/KL-warmup/model-builder pieces
below are adapted from ``code/01_sst_drvi/sst_drvi.py`` -- they are dataset-size-
and subclass-agnostic already, so the logic is unchanged; only the paths and the
absence of an SST-style scope toggle differ. Duplicated rather than imported
cross-pipeline, matching this capsule's one-shared-module-per-analysis
convention.

Filesystem contract (Code Ocean capsule):

* ``/data`` is immutable input and is only ever read.
* ``/scratch/full-drvi`` holds the prepared cohort, models, embeddings and logs.
* ``/results`` is untouched -- nothing here is a requested final deliverable yet.

Cohort definition, established against the master annotation table and the source
files themselves (see ``code/02_full_drvi/README`` once written):

* The full QC-passed cohort (``Used in analysis == True``) is 6,013,346 nuclei --
  reproduced both as the sum of the 10 per-region "final-nuclei" files below and as
  the sum of the 207 unique per-supertype GPBoost "goodcells" files (excluding 2
  duplicate files under ``GPBoost_inputs/gpboost_test/``, which would otherwise
  double-count 449 cells).
* Genes and raw counts come from the per-region final-nuclei files: all 10 share an
  identical 36,601-gene panel (verified byte-identical, in order, across all pairs
  checked), unlike the GPBoost files, which are each pre-filtered to that
  supertype's own expressed/variable genes (15,825-28,084 per file) and would
  therefore shrink the panel considerably if intersected across all 207.
* Cell selection and the CPS_*/QC-flag enrichment columns (absent from the region
  files) come from the GPBoost files instead.
* The join key between the two file sets is ``exp_component_name`` -- the AnnData
  ``obs_names`` index in both, verified byte-identical for a specific cell present
  in both a GPBoost file and its region's final-nuclei file (matching Donor ID /
  Supertype / Brain Region at that row ruled out a barcode collision).
* ``X`` in the region files is **not** raw counts -- ``uns["X_normalization"] ==
  "ln(UP10K+1)"``, confirmed empirically. Raw UMI counts live in
  ``layers["UMIs"]`` instead (also confirmed empirically: integer-valued,
  min 1.0); :data:`RAW_COUNTS_LAYER` names it and ``prepare_full.py`` reads counts
  from there, not from ``X``.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- paths

DATA = Path("/root/capsule/data/multiregion")
REGION_DATA_DIR = DATA / "Data"
GPBOOST_INPUTS = REGION_DATA_DIR / "GPBoost_inputs"

SCRATCH = Path("/scratch/full-drvi")
PREPARED_DIR = SCRATCH / "prepared"
CHUNKS_DIR = PREPARED_DIR / "chunks"
MODELS_DIR = SCRATCH / "models"
EMBED_DIR = SCRATCH / "embeddings"
LOGS_DIR = SCRATCH / "logs"
LIGHTNING_LOGS = SCRATCH / "lightning_logs"
FILE_INDEX = SCRATCH / "gpboost_file_index.csv"

PREPARED_FULL = PREPARED_DIR / "full_counts.h5ad"

#: Glob for the per-supertype, QC-passed ("goodcells") count matrices. Matched
#: relative to GPBOOST_INPUTS so the ``gpboost_test/`` exclusion in
#: :func:`build_gpboost_file_index` can check the top-level subdirectory name.
GPBOOST_GLOB = "**/*_goodcells_for_gpboost.h5ad"

#: ``gpboost_test/`` holds 2 files that are exact duplicates of files already
#: present under ``mmethods/`` (confirmed by content: same donor/supertype/region
#: and cell counts) -- including it would double-count 449 cells.
GPBOOST_EXCLUDE_SUBDIRS = ("gpboost_test",)

#: The 10 SEA-AD multiregion brain regions.
REGIONS = ("AnG", "DFC", "FI", "HIP", "ITG", "LEC", "MEC", "MTG", "STG", "V1C")

#: Glob (per region) for the raw per-region QC-passed count matrix.
REGION_FILE_GLOB = "SEAAD_{region}_RNAseq_final-nuclei.*.h5ad"

#: The AnnData obs index / join key shared by the region files and the GPBoost
#: files -- see the module docstring for how this was verified.
CELL_ID_COL = "exp_component_name"

#: Layer holding raw UMI counts in the region files -- ``X`` there is
#: log-normalized (``ln(UP10K+1)``), not raw counts. See the module docstring.
RAW_COUNTS_LAYER = "UMIs"

#: Expected total cohort size, from the master annotation table
#: (``Used in analysis == True``), reproduced by both source-file sets.
EXPECTED_N_FULL = 6_013_346
N_REGIONS = len(REGIONS)

logger = logging.getLogger("full_drvi")


def run_name(n_latent: int) -> str:
    """Stable name for one training run, used for model/embedding paths."""
    return f"full_k{n_latent}"


def model_dir(n_latent: int, model: str = "drvi") -> Path:
    """Directory one trained model was saved into. ``model`` is ``"drvi"`` or ``"scvi"``."""
    return MODELS_DIR / run_name(n_latent) / model


def embed_path(n_latent: int, model: str = "drvi") -> Path:
    """Latent AnnData for one run and model."""
    return EMBED_DIR / f"{run_name(n_latent)}_{model}_embed.h5ad"


def resolve_region_file(region: str) -> Path:
    """The one final-nuclei h5ad for ``region``.

    Raises rather than guessing if the glob matches zero or more than one file --
    the *all-nuclei* tier is known to have a stale duplicate-dated file for MEC, and
    this asserts the *final-nuclei* tier used here does not have the same problem
    for any region instead of assuming it.
    """
    matches = sorted(REGION_DATA_DIR.glob(REGION_FILE_GLOB.format(region=region)))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one final-nuclei file for region {region!r}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


# ------------------------------------------------------------------------ logging

def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging: stream to stdout, optionally tee to ``log_file``."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def log_provenance() -> None:
    """One-line record of the software and hardware a run executed on."""
    import anndata
    import scanpy
    import scvi
    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info(
        "torch %s | scvi-tools %s | anndata %s | scanpy %s | cuda available: %s | "
        "n_gpu: %d | device: %s",
        torch.__version__,
        scvi.__version__,
        anndata.__version__,
        scanpy.__version__,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        device,
    )
    try:
        import drvi

        logger.info("drvi-py %s", drvi.__version__)
    except ImportError:  # only needed for the plotting helpers
        logger.warning("drvi-py not importable; plotting helpers unavailable")


# ------------------------------------------------------------------- source files

def _read_categories(obs: h5py.Group, key: str) -> list[str]:
    """Return the categories of a categorical obs column without reading codes."""
    node = obs[key]
    if isinstance(node, h5py.Group) and "categories" in node:
        return [c.decode() for c in node["categories"][:]]
    raise TypeError(f"obs/{key} is not categorical in this file")


def _decode_obs_column(node: h5py.Group | h5py.Dataset):
    """Read one obs column (categorical group or plain dataset) as a numpy array."""
    if isinstance(node, h5py.Group) and "categories" in node:
        cats = [c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]]
        return pd.Categorical.from_codes(node["codes"][:], categories=cats)
    arr = node[:]
    if arr.dtype.kind == "S" or arr.dtype == object:
        arr = np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
    return arr


#: Columns whose schema is inconsistent across the 10 region source files -- a
#: plain HDF5 bool dataset in 9 of them, but an h5py categorical with the
#: single string category "False" in MEC (every MEC cell decodes to that one
#: value; whatever finer-grained truth MEC's source pipeline had for this
#: column did not survive into this file). Left as-is, ``concat_on_disk``
#: cannot merge a bool array with a string-categorical array for the same
#: column across chunks and fails writing it as a vlen string array. Coercing
#: to a plain bool here makes every chunk's encoding identical regardless of
#: which region it came from.
BOOL_LIKE_OBS_COLUMNS = ("Neurotypical reference",)


def _coerce_bool_like(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def read_obs_columns(f: h5py.File, columns: list[str]) -> pd.DataFrame:
    """Read a subset of obs columns straight from HDF5, skipping the rest.

    ``anndata.read_h5ad(..., backed="r")`` eagerly loads *every* obs column even
    in backed mode (only X/layers stay lazy) -- one HDF5 dataset read per column,
    each paying full request latency on this dataset's slow ``/data`` mount.
    Measured on a 213 GB, 1.18M-cell region file: reading all 132 columns this
    way (via the full anndata loader) is what made a comparable file's ``open()``
    take 43 minutes; reading only the ~36 columns actually needed took 15s.
    """
    obs = f["obs"]
    index_key = obs.attrs.get("_index", "index")
    index = _decode_obs_column(obs[index_key])
    data = {col: _decode_obs_column(obs[col]) for col in columns if col in obs}
    df = pd.DataFrame(data, index=pd.Index(index, name=index_key))
    for col in BOOL_LIKE_OBS_COLUMNS:
        if col in df.columns:
            df[col] = _coerce_bool_like(df[col])
    return df


def build_gpboost_file_index(force: bool = False) -> pd.DataFrame:
    """Index the per-supertype GPBoost input h5ads (metadata only, no matrix I/O).

    Excludes ``gpboost_test/`` (see :data:`GPBOOST_EXCLUDE_SUBDIRS`). Cached to
    :data:`FILE_INDEX` because the glob plus ~207 file opens takes a couple of
    minutes on the attached, network-backed dataset.
    """
    if FILE_INDEX.exists() and not force:
        logger.info("reusing file index %s", FILE_INDEX)
        return pd.read_csv(FILE_INDEX)

    logger.info("building file index from %s", GPBOOST_INPUTS)
    rows = []
    for path in sorted(GPBOOST_INPUTS.glob(GPBOOST_GLOB)):
        subdir = str(path.parent.relative_to(GPBOOST_INPUTS))
        if subdir in GPBOOST_EXCLUDE_SUBDIRS or subdir.split("/")[0] in GPBOOST_EXCLUDE_SUBDIRS:
            continue
        with h5py.File(path, "r") as f:
            n_obs, n_vars = (int(x) for x in f["X"].attrs["shape"])
            subclass = _read_categories(f["obs"], "Subclass")
            supertype = _read_categories(f["obs"], "Supertype")
        if len(subclass) != 1 or len(supertype) != 1:
            logger.warning(
                "%s holds %d subclasses / %d supertypes; expected 1 each",
                path.name,
                len(subclass),
                len(supertype),
            )
        rows.append(
            {
                "path": str(path),
                "subdir": subdir,
                "stem": path.name.removesuffix("_goodcells_for_gpboost.h5ad"),
                "supertype": supertype[0],
                "subclass": subclass[0],
                "n_obs": n_obs,
                "n_vars": n_vars,
            }
        )
    index = pd.DataFrame(rows)
    FILE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(FILE_INDEX, index=False)
    logger.info("indexed %d files -> %s", len(index), FILE_INDEX)
    return index


# --------------------------------------------------------------- obs column subset
#
# Same allowlist as sst_drvi.py's OBS_KEEP/OBS_CATEGORICAL: the source files carry
# 150+ obs columns, most of them per-library CellRanger metrics not useful here.
# Split into what the *region* files natively have vs. what only the *GPBoost*
# files add (CPS_* and the QC-pass flags), since prepare_full.py reads them from
# two different sources in two different phases.

REGION_OBS_KEEP = (
    # cell type hierarchy
    "Class",
    "Subclass",
    "Supertype",
    # anatomy / donor
    "Brain Region",
    "Donor ID",
    "Sex",
    "Age at Death",
    "PMI",
    "RIN",
    "Years of education",
    # technical / batch
    "library_prep",
    "sample_id",
    "method",
    "Chemistry",
    "alignment",
    "rna_amplification",
    "load_name",
    "batch_vendor_name",
    # per-cell QC
    "Number of UMIs",
    "Genes detected",
    "Doublet score",
    "Fraction mitochondrial UMIs",
    # neuropathology / clinical
    "Cognitive Status",
    "Overall AD neuropathological Change",
    "Thal",
    "Braak",
    "CERAD score",
    "Overall CAA Score",
    "Highest Lewy Body Disease",
    "LATE",
    "Atherosclerosis",
    "Arteriolosclerosis",
    "APOE Genotype",
    "APOE4_Status",
    "Last CASI Score",
    "Last MMSE Score",
    "Severely Affected Donor",
    "Neurotypical reference",
)

#: Columns only the GPBoost files carry -- attached in prepare_full.py's Phase 3.
GPBOOST_ONLY_OBS = (
    "CPS_Global",
    "CPS_Global_ABeta",
    "CPS_Global_pTau",
    "CPS_Local",
    "CPS_Local_ABeta",
    "CPS_Local_pTau",
    "PassedInitial_QC",
    "PassedFine_QC",
    "PassedManualFine_QC",
    "PassedUpdatedFine_QC",
    "PassedSupertypeRegion_QC",
)

#: obs columns to treat as categorical after concatenation.
OBS_CATEGORICAL = (
    "Class",
    "Subclass",
    "Supertype",
    "Brain Region",
    "Donor ID",
    "Sex",
    "library_prep",
    "sample_id",
    "method",
    "Chemistry",
    "alignment",
    "rna_amplification",
    "load_name",
    "batch_vendor_name",
    "Cognitive Status",
    "Overall AD neuropathological Change",
    "Thal",
    "Braak",
    "CERAD score",
    "Overall CAA Score",
    "Highest Lewy Body Disease",
    "LATE",
    "Atherosclerosis",
    "Arteriolosclerosis",
    "APOE Genotype",
    "APOE4_Status",
)


# --------------------------------------------------------------------- multi-GPU

#: Lightning strategy string. scvi-tools keys distributed sampling off ``"ddp" in
#: strategy`` (``scvi/model/_utils.py``), and the ``find_unused_parameters`` variant
#: is required rather than optional -- see :func:`ddp_trainer_kwargs`.
DDP_STRATEGY = "ddp_find_unused_parameters_true"


def ddp_trainer_kwargs(devices: int, log_save_dir: Path = LIGHTNING_LOGS) -> dict:
    """Extra ``model.train`` kwargs for multi-GPU training; empty for a single device.

    Two things make plain ``"ddp"`` the wrong choice for these models:

    * ``dispersion="gene-batch"`` fits one ``px_r`` row per library and
      ``batch_representation="embedding"`` one embedding row per library. With 907
      libraries, almost every row sees no cell on a given rank and so receives no
      gradient, which plain DDP treats as an error. The ``find_unused_parameters``
      variant tolerates it.
    * Under DDP scvi-tools switches ``SimpleLogger`` to writing history to disk with
      ``save_dir`` defaulting to :func:`os.getcwd`, which would drop a
      ``lightning_logs/`` tree into the repository. ``log_save_dir`` sends it to
      scratch instead.

    scvi-tools also disables early stopping under DDP, and ``check_val_every_n_epoch``
    is only defaulted to 1 when early stopping/checkpointing/an LR monitor is active,
    so it is requested explicitly here to keep the validation curves.
    """
    if devices <= 1:
        return {}
    log_save_dir.mkdir(parents=True, exist_ok=True)
    return {
        "strategy": DDP_STRATEGY,
        "log_save_dir": str(log_save_dir),
        "check_val_every_n_epoch": 1,
    }


def precision_kwargs(precision: str | None) -> dict:
    """``{"precision": ...}`` when a non-default precision is requested, else empty.

    Kept separate from :func:`ddp_trainer_kwargs` because precision applies to
    single-GPU runs too, and because the runner tests that function's return value
    for truthiness to decide whether early stopping is available -- folding an
    unrelated key into it would silently disable early stopping on one GPU.
    """
    return {"precision": precision} if precision else {}


def finish_distributed() -> bool:
    """Synchronize, tear down the process group, and report whether this is rank 0.

    Lightning's non-spawn DDP launcher re-executes the whole script once per GPU, so
    without a guard every rank would run the post-training steps. Call this
    immediately after ``model.train`` and return early when it is ``False``.
    """
    import torch.distributed as dist

    rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return rank == 0


# ------------------------------------------------------------------- KL schedule

def resolve_kl_warmup(cfg) -> int:
    """Epochs over which to ramp the KL weight, for either model family.

    Defaults to a third of the run, leaving two thirds at ``kl_weight = 1.0`` --
    see ``code/01_sst_drvi/sst_drvi.py``'s fuller explanation of why the warmup
    must be identical across DRVI and scVI and why it must finish before the end
    of training.
    """
    if cfg.kl_warmup_epochs is not None:
        return int(cfg.kl_warmup_epochs)
    return max(int(cfg.max_epochs) // 3, 1)


def plan_kwargs(cfg) -> dict:
    """``plan_kwargs`` for ``model.train``, identical in shape for DRVI and scVI."""
    kwargs = {"lr": cfg.lr, "n_epochs_kl_warmup": resolve_kl_warmup(cfg)}
    if cfg.reduce_lr_on_plateau:
        kwargs.update(
            reduce_lr_on_plateau=True,
            lr_patience=cfg.lr_patience,
            lr_factor=cfg.lr_factor,
            lr_scheduler_metric="elbo_validation",
        )
    return kwargs


def check_kl_schedule(history: pd.DataFrame, warmup: int, max_epochs: int) -> float:
    """Log whether the KL ramp finished, and return the weight actually reached."""
    if "kl_weight" not in history:
        logger.warning("history has no kl_weight column; cannot verify the KL schedule")
        return float("nan")
    reached = float(pd.to_numeric(history["kl_weight"], errors="coerce").max())
    if reached < 0.99:
        logger.error(
            "KL weight only reached %.4f: warmup=%d epochs vs max_epochs=%d, so the model "
            "minimised a down-weighted KL throughout and its ELBO is not comparable to a "
            "fully warmed-up run",
            reached, warmup, max_epochs,
        )
    else:
        logger.info("KL weight reached %.4f over a %d-epoch warmup", reached, warmup)
    return reached


# -------------------------------------------------------------------- model config

@dataclass(frozen=True)
class DrviConfig:
    """DRVI hyperparameters for one run.

    Defaults match ``code/01_sst_drvi/sst_drvi.py``'s ``DrviConfig`` -- this is a
    new module with no prior run depending on a different default, but starting
    from the same, already-validated configuration is the safer default than
    guessing new values. The full dataset's greater cell-type diversity may
    warrant a larger ``--n-latent`` than SST's 64; that is a modelling choice to
    make per-run via the flag, not a reason to change the default here.
    """

    n_latent: int = 64
    n_split_latent: int | None = None
    split_method: str = "split_map"
    split_aggregation: str = "logsumexp"
    gene_likelihood: str = "pnb"
    dispersion: str = "gene-batch"
    batch_representation: str = "embedding"
    n_hidden: int = 128
    n_layers: int = 2
    batch_key: str = "library_prep"
    categorical_covariate_keys: tuple[str, ...] = field(default_factory=tuple)
    continuous_covariate_keys: tuple[str, ...] = field(default_factory=tuple)
    encode_covariates: bool = False
    batch_size: int = 256
    max_epochs: int = 200
    lr: float = 1e-3
    kl_warmup_epochs: int | None = None
    reduce_lr_on_plateau: bool = False
    lr_patience: int = 15
    lr_factor: float = 0.5
    precision: str | None = None
    early_stopping_patience: int = 20
    train_size: float = 0.9
    seed: int = 0
    devices: int = 1
    vanished_threshold: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def model_kwargs(self) -> dict:
        """Kwargs for the :class:`scvi.external.DRVI` constructor."""
        return {
            "n_latent": self.n_latent,
            "n_split_latent": self.n_split_latent,
            "split_method": self.split_method,
            "split_aggregation": self.split_aggregation,
            "gene_likelihood": self.gene_likelihood,
            "dispersion": self.dispersion,
            "batch_representation": self.batch_representation,
            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
            "encode_covariates": self.encode_covariates,
        }


@dataclass(frozen=True)
class ScviConfig:
    """scVI hyperparameters, for the model the DRVI fit is compared against.

    Mirrors :class:`DrviConfig` on every knob the two models share; see
    ``code/01_sst_drvi/sst_drvi.py``'s fuller docstring for why
    ``gene_likelihood`` cannot match and why the KL warmup is resolved through
    the shared :func:`resolve_kl_warmup` rather than either model's own default.
    """

    n_latent: int = 64
    n_hidden: int = 128
    n_layers: int = 2
    dropout_rate: float = 0.1
    gene_likelihood: str = "nb"
    dispersion: str = "gene-batch"
    batch_representation: str = "embedding"
    batch_key: str = "library_prep"
    categorical_covariate_keys: tuple[str, ...] = ()
    continuous_covariate_keys: tuple[str, ...] = field(default_factory=tuple)
    encode_covariates: bool = False
    batch_size: int = 256
    max_epochs: int = 200
    lr: float = 1e-3
    kl_warmup_epochs: int | None = None
    reduce_lr_on_plateau: bool = False
    lr_patience: int = 15
    lr_factor: float = 0.5
    precision: str | None = None
    early_stopping_patience: int = 20
    train_size: float = 0.9
    seed: int = 0
    devices: int = 1
    used_threshold: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def model_kwargs(self) -> dict:
        """Kwargs for the :class:`scvi.model.SCVI` constructor."""
        return {
            "n_latent": self.n_latent,
            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
            "dropout_rate": self.dropout_rate,
            "gene_likelihood": self.gene_likelihood,
            "dispersion": self.dispersion,
            "batch_representation": self.batch_representation,
            "encode_covariates": self.encode_covariates,
        }


#: Suffix marking a covariate column this module derived, rather than one from the data.
SCALED_SUFFIX = " [scaled]"


def scale_continuous_covariates(adata, keys: Sequence[str]) -> list[str]:
    """Write standardized copies of continuous covariates and return their names.

    See ``code/01_sst_drvi/sst_drvi.py``'s fuller docstring: scvi-tools stacks
    ``continuous_covariate_keys`` verbatim without centering/scaling, so each
    column is log1p'd when it is count-like and then z-scored.
    """
    names = []
    scaling: dict[str, dict] = {}
    for key in keys:
        if key not in adata.obs:
            raise KeyError(f"obs column {key!r} not found")
        values = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=np.float64)
        logged = np.nanmax(values) > 1_000 and np.nanmin(values) >= 0
        if logged:
            values = np.log1p(values)
        mean, std = np.nanmean(values), np.nanstd(values)
        scaled = (values - mean) / (std if std > 0 else 1.0)
        n_missing = int(np.isnan(scaled).sum())
        scaled = np.nan_to_num(scaled, nan=0.0)

        name = f"{key}{SCALED_SUFFIX}"
        adata.obs[name] = scaled
        names.append(name)
        scaling[key] = {"log1p": bool(logged), "mean": float(mean), "std": float(std)}
        logger.info(
            "  covariate %-30s log1p=%-5s mean=%.4g std=%.4g -> z-scored%s",
            key,
            logged,
            mean,
            std,
            f", {n_missing} missing filled with 0" if n_missing else "",
        )
    adata.uns["covariate_scaling"] = scaling
    return names


def build_drvi_model(adata, cfg: DrviConfig):
    """Register ``adata`` and construct the DRVI model.

    Counts live in ``adata.X`` (``layer=None``); ``prepare_full.py`` writes raw
    UMI counts there directly, with no duplicate ``layers["counts"]``. Works the
    same whether ``adata`` is backed or fully in memory -- scvi-tools' data
    loading is backed-aware (``adata.isbacked``), which is what makes training
    against the ~270 GB prepared file possible without loading it into RAM.
    """
    from scvi.external import DRVI

    DRVI.setup_anndata(
        adata,
        layer=None,
        batch_key=cfg.batch_key,
        categorical_covariate_keys=list(cfg.categorical_covariate_keys) or None,
        continuous_covariate_keys=list(cfg.continuous_covariate_keys) or None,
    )
    return DRVI(adata, **cfg.model_kwargs)


def build_scvi_model(adata, cfg: ScviConfig):
    """Register ``adata`` and construct the scVI model, mirroring :func:`build_drvi_model`."""
    from scvi.model import SCVI

    SCVI.setup_anndata(
        adata,
        layer=None,
        batch_key=cfg.batch_key,
        categorical_covariate_keys=list(cfg.categorical_covariate_keys) or None,
        continuous_covariate_keys=list(cfg.continuous_covariate_keys) or None,
    )
    return SCVI(adata, **cfg.model_kwargs)


def latent_embedding(model, adata, cfg, prefix: str = "DR"):
    """AnnData of the latent space: cells x latent dimensions, obs carried over.

    The returned object is small (``n_cells x n_latent``) and fully in memory
    regardless of whether ``adata`` was backed.
    """
    import anndata as ad

    latent = model.get_latent_representation(adata, batch_size=cfg.batch_size)
    obs = adata.obs.drop(columns=[c for c in adata.obs.columns if c.startswith("_scvi")])
    embed = ad.AnnData(latent, obs=obs.copy())
    embed.var_names = [f"{prefix}_{i + 1}" for i in range(embed.n_vars)]
    return embed


# ------------------------------------------------------------------ interpretation

@contextmanager
def _ood_continuous_covariate_shim(model):
    """Make the OOD latent traversal work on a model that has continuous covariates.

    See ``code/01_sst_drvi/sst_drvi.py``'s fuller docstring: scvi-tools 1.5
    hardcodes ``cont_values=None`` in the OOD traversal, so zeros are substituted
    -- correct because :func:`scale_continuous_covariates` z-scores everything,
    making zero the mean.
    """
    n_cont = int(getattr(model.summary_stats, "n_extra_continuous_covs", 0) or 0)
    if n_cont == 0:
        yield
        return

    original = model.iterate_on_decoded_latent_samples

    def with_zero_cont_covs(*args, **kwargs):
        if kwargs.get("cont_values") is None:
            z = kwargs.get("z", args[0] if args else None)
            kwargs["cont_values"] = np.zeros((len(z), n_cont), dtype=np.float32)
        return original(*args, **kwargs)

    model.iterate_on_decoded_latent_samples = with_zero_cont_covs
    try:
        logger.info(
            "OOD traversal: substituting zeros for %d standardized continuous covariate(s) "
            "(scvi 1.5 passes none)",
            n_cont,
        )
        yield
    finally:
        model.iterate_on_decoded_latent_samples = original


def calculate_interpretability(model, embed, methods=("IND", "OOD"), directional: bool = True):
    """Compute DRVI interpretability scores into ``embed.varm``, covariates included."""
    with _ood_continuous_covariate_shim(model):
        model.calculate_interpretability_scores(
            embed, methods=list(methods), directional=directional
        )


def latent_stats(embed, threshold: float = 0.5) -> pd.DataFrame:
    """Per-dimension usage statistics that a DRVI *or* an scVI embedding supports.

    See ``code/01_sst_drvi/sst_drvi.py``'s fuller docstring on why this shared
    rule -- not either model's own self-report -- is what makes DRVI and scVI's
    used-dimension counts comparable.
    """
    x = np.asarray(embed.X, dtype=np.float32)
    stats = pd.DataFrame(
        {
            "std": x.std(axis=0),
            "mean_abs": np.abs(x).mean(axis=0),
            "max_abs": np.abs(x).max(axis=0),
        },
        index=embed.var_names,
    )
    stats["used"] = stats["max_abs"] >= threshold
    if "vanished" in embed.var:
        stats["vanished"] = embed.var["vanished"].to_numpy()
    if "title" in embed.var:
        stats.insert(0, "title", embed.var["title"].to_numpy())
    return stats


# ------------------------------------------------------------------------- loaders

def load_prepared(path: Path = PREPARED_FULL, backed: bool | str = "r"):
    """Load the prepared full-dataset counts. ``backed="r"`` keeps ~270 GB off the heap."""
    import anndata as ad

    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run prepare_full.py first")
    return ad.read_h5ad(path, backed=backed) if backed else ad.read_h5ad(path)
