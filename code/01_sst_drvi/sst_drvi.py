"""Shared helpers for the SEA-AD multiregion SST DRVI factorization.

Imported by ``prepare_sst.py``, ``train_drvi.py`` and ``01_inspect_factors.ipynb`` so
that paths, the cohort definition and the model hyperparameters have exactly one
definition.

Filesystem contract (Code Ocean capsule):

* ``/data`` is immutable input and is only ever read.
* ``/scratch/sst-drvi`` holds the prepared cohort, models, embeddings and logs.
* ``/results`` is untouched -- nothing here is a requested final deliverable yet.
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
GPBOOST_INPUTS = DATA / "Data" / "GPBoost_inputs"

SCRATCH = Path("/scratch/sst-drvi")
PREPARED_DIR = SCRATCH / "prepared"
MODELS_DIR = SCRATCH / "models"
EMBED_DIR = SCRATCH / "embeddings"
LOGS_DIR = SCRATCH / "logs"
#: scvi-tools persists training history here under DDP (see :func:`ddp_trainer_kwargs`).
LIGHTNING_LOGS = SCRATCH / "lightning_logs"
FILE_INDEX = SCRATCH / "gpboost_file_index.csv"

PREPARED_SST = PREPARED_DIR / "sst_counts.h5ad"

#: Glob for the per-supertype, QC-passed ("goodcells") count matrices.
GPBOOST_GLOB = "**/*_goodcells_for_gpboost.h5ad"

logger = logging.getLogger("sst_drvi")


def run_name(n_latent: int, include_chodl: bool = False) -> str:
    """Stable name for one training run, used for model/embedding paths."""
    scope = "sst_chodl" if include_chodl else "sst"
    return f"{scope}_k{n_latent}"


def run_paths(run: str, model: str = "drvi") -> tuple[Path, Path]:
    """``(model directory, embedding path)`` for a run name.

    Takes the run name directly, so it also addresses suffixed runs such as
    ``sst_k64_covar`` that :func:`run_name` does not construct.
    """
    return MODELS_DIR / run / model, EMBED_DIR / f"{run}_{model}_embed.h5ad"


def model_dir(n_latent: int, include_chodl: bool = False, model: str = "drvi") -> Path:
    """Directory one trained model was saved into. ``model`` is ``"drvi"`` or ``"scvi"``."""
    return MODELS_DIR / run_name(n_latent, include_chodl) / model


def embed_path(n_latent: int, include_chodl: bool = False, model: str = "drvi") -> Path:
    """Latent AnnData for one run and model."""
    return EMBED_DIR / f"{run_name(n_latent, include_chodl)}_{model}_embed.h5ad"


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


# --------------------------------------------------------------------- multi-GPU

#: Lightning strategy string. scvi-tools keys distributed sampling off ``"ddp" in
#: strategy`` (``scvi/model/_utils.py``), and the ``find_unused_parameters`` variant is
#: required here rather than optional -- see :func:`ddp_trainer_kwargs`.
DDP_STRATEGY = "ddp_find_unused_parameters_true"


def ddp_trainer_kwargs(devices: int, log_save_dir: Path = LIGHTNING_LOGS) -> dict:
    """Extra ``model.train`` kwargs for multi-GPU training; empty for a single device.

    Two things make plain ``"ddp"`` the wrong choice for these models:

    * ``dispersion="gene-batch"`` fits one ``px_r`` row per library and
      ``batch_representation="embedding"`` one embedding row per library. With 902
      libraries, almost every row sees no cell on a given rank and so receives no
      gradient, which plain DDP treats as an error ("expected to have finished reduction
      in the prior iteration"). The ``find_unused_parameters`` variant tolerates it.
    * Under DDP scvi-tools switches ``SimpleLogger`` to writing history to disk with
      ``save_dir`` defaulting to :func:`os.getcwd`, which would drop a ``lightning_logs/``
      tree into the repository. ``log_save_dir`` sends it to scratch instead.

    scvi-tools also disables early stopping under DDP (``scvi/train/_trainer.py``), so a
    distributed run always uses all ``max_epochs``. That has a non-obvious consequence:
    ``check_val_every_n_epoch`` is only defaulted to 1 when early stopping, checkpointing
    or an LR monitor is active, and otherwise stays at ``sys.maxsize`` -- so a DDP run
    would never validate and ``history`` would come back with no ``*_validation`` metrics
    at all. It is requested explicitly here to keep the validation curves.
    """
    if devices <= 1:
        return {}
    log_save_dir.mkdir(parents=True, exist_ok=True)
    return {
        "strategy": DDP_STRATEGY,
        "log_save_dir": str(log_save_dir),
        "check_val_every_n_epoch": 1,
    }


def finish_distributed() -> bool:
    """Synchronize, tear down the process group, and report whether this is rank 0.

    Lightning's non-spawn DDP launcher re-executes the whole script once per GPU, so
    without a guard every rank would run the post-training steps -- four processes racing
    to write the same model, embedding and h5ad. Call this immediately after
    ``model.train`` and return early when it is ``False``.
    """
    import torch.distributed as dist

    rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # no rank leaves until all of them are done training
        dist.destroy_process_group()
    return rank == 0


# ------------------------------------------------------------------- source files

def _read_categories(obs: h5py.Group, key: str) -> list[str]:
    """Return the categories of a categorical obs column without reading codes."""
    node = obs[key]
    if isinstance(node, h5py.Group) and "categories" in node:
        return [c.decode() for c in node["categories"][:]]
    raise TypeError(f"obs/{key} is not categorical in this file")


def build_file_index(force: bool = False) -> pd.DataFrame:
    """Index the per-supertype GPBoost input h5ads (metadata only, no matrix I/O).

    Cached to :data:`FILE_INDEX` because the glob plus 207 file opens takes a
    couple of minutes on the attached dataset.
    """
    if FILE_INDEX.exists() and not force:
        logger.info("reusing file index %s", FILE_INDEX)
        return pd.read_csv(FILE_INDEX)

    logger.info("building file index from %s", GPBOOST_INPUTS)
    rows = []
    for path in sorted(GPBOOST_INPUTS.glob(GPBOOST_GLOB)):
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
                "subdir": str(path.parent.relative_to(GPBOOST_INPUTS)),
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


#: Exact ``Subclass`` strings. Verified against the master annotation table; there is
#: no "SST" variant anywhere in the dataset.
SST_SUBCLASS = "Sst"
SST_CHODL_SUBCLASS = "Sst Chodl"


def sst_files(index: pd.DataFrame, include_chodl: bool = False) -> pd.DataFrame:
    """Rows of ``index`` belonging to the SST cohort, ordered by supertype."""
    wanted = [SST_SUBCLASS] + ([SST_CHODL_SUBCLASS] if include_chodl else [])
    selected = index[index["subclass"].isin(wanted)].copy()
    if selected.empty:
        raise RuntimeError(f"no files found for subclass(es) {wanted}")
    return selected.sort_values(["subclass", "supertype"]).reset_index(drop=True)


# --------------------------------------------------------------- obs column subset
#
# The GPBoost inputs carry 150+ obs columns, ~60 of which are per-library CellRanger
# metrics (GEX_*/ATAC_*) that would bloat the prepared file without being useful.
# Keep an explicit list instead; anything missing from a source file is skipped with
# a warning rather than failing the build.

OBS_KEEP = (
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
    # continuous pseudo-progression scores (only present in the GPBoost inputs)
    "CPS_Global",
    "CPS_Global_ABeta",
    "CPS_Global_pTau",
    "CPS_Local",
    "CPS_Local_ABeta",
    "CPS_Local_pTau",
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
    "source_supertype",
)


# -------------------------------------------------------------------- model config

@dataclass(frozen=True)
class DrviConfig:
    """DRVI hyperparameters for one run.

    Defaults are the configuration agreed for this analysis. ``n_split_latent=None``
    splits every latent dimension, which is what ``directional=True`` interpretability
    requires.
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

    Mirrors :class:`DrviConfig` on every knob the two models share -- latent size,
    architecture, batch handling, dispersion, schedule -- so that a difference between
    the two embeddings is attributable to DRVI's split decoder rather than to the setup.
    Two things cannot match:

    * ``gene_likelihood``: ``"pnb"`` is DRVI's log-space negative binomial and
      :class:`scvi.model.SCVI` does not accept it, so the plain ``"nb"`` is used.
    * ``kl_warmup_epochs`` defaults to ``None``, meaning scvi's own warmup. DRVI needs the
      KL ramped over the whole run for its split decoder to disentangle; forcing that on
      plain scVI would make it a non-standard baseline. Set it to match if you want the
      identical schedule.
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
    early_stopping_patience: int = 20
    train_size: float = 0.9
    seed: int = 0
    devices: int = 1
    kl_warmup_epochs: int | None = None
    used_threshold: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def model_kwargs(self) -> dict:
        """Kwargs for the :class:`scvi.model.SCVI` constructor.

        ``batch_representation`` and ``encode_covariates`` are not named parameters of
        ``SCVI.__init__``; they reach :class:`scvi.module.VAE` through its ``**kwargs``.
        """
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

    scvi-tools stacks ``continuous_covariate_keys`` verbatim -- it does **not** center or
    scale them. Feeding a raw count column such as ``Number of UMIs`` (195 to 2.3e5) into
    the encoder/decoder would dominate every other input, so each column is log1p'd when
    it is count-like and then z-scored. Derived columns are suffixed with
    :data:`SCALED_SUFFIX`; the originals are left untouched.

    What was done to each column is recorded in ``adata.uns["covariate_scaling"]`` -- the
    ``log1p`` decision is data-dependent, so it cannot be recovered from the saved model
    afterwards, and :func:`covariate_design` reports it only if handed this record.
    """
    names = []
    scaling: dict[str, dict] = {}
    for key in keys:
        if key not in adata.obs:
            raise KeyError(f"obs column {key!r} not found")
        values = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=np.float64)
        # A wide, strictly-positive range means a count; compress it before standardizing.
        logged = np.nanmax(values) > 1_000 and np.nanmin(values) >= 0
        if logged:
            values = np.log1p(values)
        mean, std = np.nanmean(values), np.nanstd(values)
        scaled = (values - mean) / (std if std > 0 else 1.0)
        # post-standardization the mean is 0, so that is the neutral fill for gaps
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

    Counts live in ``adata.X`` (``layer=None``); the prepared file deliberately has no
    duplicate ``layers["counts"]``.
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

    ``prefix`` names the dimensions -- ``DR_n`` for DRVI factors, ``Z_n`` for scVI's,
    which are not claimed to be independently interpretable.
    """
    import anndata as ad

    latent = model.get_latent_representation(adata, batch_size=cfg.batch_size)
    # setup_anndata writes _scvi_* bookkeeping columns into obs; they are noise here.
    obs = adata.obs.drop(columns=[c for c in adata.obs.columns if c.startswith("_scvi")])
    embed = ad.AnnData(latent, obs=obs.copy())
    embed.var_names = [f"{prefix}_{i + 1}" for i in range(embed.n_vars)]
    return embed


# ------------------------------------------------------------------------- loaders

def load_prepared(path: Path = PREPARED_SST, backed: bool | str = "r"):
    """Load the prepared SST counts. ``backed="r"`` keeps 231k x 19k off the heap."""
    import anndata as ad

    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run prepare_sst.py first")
    return ad.read_h5ad(path, backed=backed) if backed else ad.read_h5ad(path)


def load_embedding(n_latent: int = 64, include_chodl: bool = False, model: str = "drvi"):
    """Load the latent AnnData written by ``train_drvi.py`` / ``train_scvi.py``."""
    import anndata as ad

    path = embed_path(n_latent, include_chodl, model)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run train_{model}.py first")
    return ad.read_h5ad(path)


# ----------------------------------------------------------------- saved artifacts
#
# Everything the QC notebook needs to know about a fit -- its design, its training
# history, which covariates it was given -- is inside the saved ``model.pt``. Reading it
# directly means the notebook needs neither the 2.3 GB cohort nor a GPU, and it works on
# runs that finished long ago.

def _as_list(value) -> list:
    """Registry entries are sometimes numpy arrays, whose truthiness raises."""
    return [] if value is None else list(value)


def _model_file(model_path: Path | str) -> Path:
    path = Path(model_path)
    return path / "model.pt" if path.is_dir() else path


def _load_saved(model_path: Path | str) -> dict:
    """``torch.load`` a saved scvi-tools model without constructing the model."""
    import torch

    path = _model_file(model_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; train the model first")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_history(model_path: Path | str) -> pd.DataFrame:
    """Epochs x metrics training history, read out of a saved model.

    scvi-tools keeps every logged loss component in ``attr_dict["history_"]`` -- ELBO,
    reconstruction loss, the KL terms and the warmup weight, train and validation each --
    so the loss curves need no retraining and no reload of the module.
    """
    history = _load_saved(model_path)["attr_dict"].get("history_") or {}
    if not history:
        raise ValueError(f"{_model_file(model_path)} has no training history")
    out = pd.concat(history.values(), axis=1, join="outer")
    out.index.name = "epoch"
    return out.sort_index()


def model_design(model_path: Path | str) -> dict:
    """What a saved model is: constructor arguments, registry, size and how far it trained."""
    saved = _load_saved(model_path)
    attrs = saved["attr_dict"]
    registry = attrs["registry_"]

    init = dict(attrs["init_params_"].get("non_kwargs") or {})
    # both SCVI and DRVI funnel the module-level knobs through a nested "kwargs" entry
    init.update((attrs["init_params_"].get("kwargs") or {}).get("kwargs") or {})

    stats: dict = {}
    for field in registry["field_registries"].values():
        stats.update(field.get("summary_stats") or {})

    n_params: dict[str, int] = {}
    for key, tensor in saved["model_state_dict"].items():
        if hasattr(tensor, "numel"):
            head = key.split(".")[0]
            n_params[head] = n_params.get(head, 0) + tensor.numel()

    history = attrs.get("history_") or {}
    return {
        "model": registry.get("model_name"),
        "scvi_version": registry.get("scvi_version"),
        "setup_args": dict(registry.get("setup_args") or {}),
        "field_registries": registry["field_registries"],
        "init_params": init,
        "summary_stats": stats,
        "n_params": n_params,
        "n_params_total": sum(n_params.values()),
        "epochs_run": len(history.get("elbo_train", ())),
        "is_trained": bool(attrs.get("is_trained_", False)),
        # these are numpy arrays, so `or ()` would raise on the ambiguous truth value
        "n_train": 0 if attrs.get("train_indices_") is None else len(attrs["train_indices_"]),
        "n_validation": (
            0
            if attrs.get("validation_indices_") is None
            else len(attrs["validation_indices_"])
        ),
    }


def design_table(designs: dict[str, Path | str]) -> pd.DataFrame:
    """Side-by-side model design, one column per named model.

    Rows that do not apply to a model come back as ``None`` -- ``split_method`` is DRVI's
    alone, for instance -- so the table doubles as a record of where the two differ.
    """
    columns = {}
    for name, path in designs.items():
        d = model_design(path)
        init, stats, setup = d["init_params"], d["summary_stats"], d["setup_args"]
        columns[name] = {
            "model": d["model"],
            "scvi-tools": d["scvi_version"],
            "n_latent": init.get("n_latent"),
            "n_hidden": init.get("n_hidden"),
            "n_layers": init.get("n_layers"),
            "gene_likelihood": init.get("gene_likelihood"),
            "dispersion": init.get("dispersion"),
            "batch_representation": init.get("batch_representation"),
            "encode_covariates": init.get("encode_covariates"),
            "split_method": init.get("split_method"),
            "split_aggregation": init.get("split_aggregation"),
            "batch_key": setup.get("batch_key"),
            "n_batch": stats.get("n_batch"),
            "n_extra_categorical_covs": stats.get("n_extra_categorical_covs"),
            "n_extra_continuous_covs": stats.get("n_extra_continuous_covs"),
            "n_cells": d["n_train"] + d["n_validation"],
            "n_genes": stats.get("n_vars"),
            "epochs_run": d["epochs_run"],
            "parameters": f"{d['n_params_total']:,}",
        }
    return pd.DataFrame(columns)


def covariate_design(
    model_path: Path | str,
    tested: Sequence[str],
    scaling: dict | None = None,
) -> pd.DataFrame:
    """How each covariate in ``tested`` enters the model -- including the ones that do not.

    The ``not modelled`` rows are the point of this table. A covariate the model was never
    told about, which then explains much of a latent dimension, means a dimension was
    spent on nuisance structure; a covariate marked ``batch key`` that still scores high
    means structure leaked past the correction. Read it beside :func:`factor_association`.

    ``tested`` should be the same covariate list the association test uses, so nothing
    that gets scored is missing from the table. Pass ``scaling`` --
    ``embed.uns["covariate_scaling"]``, written by :func:`scale_continuous_covariates` --
    to report whether a continuous covariate was log1p'd as well as z-scored; without it
    the transform is reported as the generic "standardized".
    """
    scaling = scaling or {}
    d = model_design(model_path)
    setup, stats, fields = d["setup_args"], d["summary_stats"], d["field_registries"]
    init = d["init_params"]

    batch_key = setup.get("batch_key")
    cat_state = (fields.get("extra_categorical_covs") or {}).get("state_registry") or {}
    cat_levels = dict(
        zip(
            _as_list(cat_state.get("field_keys")),
            _as_list(cat_state.get("n_cats_per_key")),
            strict=False,
        )
    )
    cont_state = (fields.get("extra_continuous_covs") or {}).get("state_registry") or {}
    # continuous covariates are registered on the derived, standardized columns
    cont_keys = {
        str(col).removesuffix(SCALED_SUFFIX): str(col).endswith(SCALED_SUFFIX)
        for col in _as_list(cont_state.get("columns"))
    }
    encoded = bool(init.get("encode_covariates"))

    rows = []
    for name in tested:
        if name == batch_key:
            role, representation = "batch key", init.get("batch_representation", "one-hot")
            levels = stats.get("n_batch")
        elif name in cat_levels:
            role, representation = "categorical covariate", "one-hot"
            levels = cat_levels[name]
        elif name in cont_keys:
            role = "continuous covariate"
            if not cont_keys[name]:
                representation = "as given"
            elif name in scaling:
                representation = (
                    "log1p + z-scored" if scaling[name].get("log1p") else "z-scored"
                )
            else:
                representation = "standardized"
            levels = None
        else:
            role, representation, levels = "not modelled", "-", None
        modelled = role != "not modelled"
        rows.append(
            {
                "role": role,
                "representation": representation,
                "n_levels": levels,
                "reaches_encoder": encoded if modelled else False,
                "reaches_decoder": modelled,
                "affects_dispersion": role == "batch key"
                and init.get("dispersion") == "gene-batch",
            }
        )
    return pd.DataFrame(rows, index=pd.Index(list(tested), name="covariate"))


def latent_stats(embed, threshold: float = 0.5) -> pd.DataFrame:
    """Per-dimension usage statistics that a DRVI *or* an scVI embedding supports.

    DRVI prunes dimensions it does not need and flags them itself, in
    ``var["vanished"]`` written by ``set_latent_dimension_stats``. scVI has no such
    notion, so the only way to compare how much of ``n_latent`` each model actually used
    is a shared rule: a dimension counts as used when some cell drives it past
    ``threshold`` in absolute value. Where DRVI's own flag is present it is carried
    through, so the two can be checked against each other rather than trusted blindly.
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


# ------------------------------------------------------------------ interpretation

@contextmanager
def _ood_continuous_covariate_shim(model):
    """Make the OOD latent traversal work on a model that has continuous covariates.

    scvi-tools 1.5.0 hardcodes ``cont_values=None`` inside
    ``get_effect_of_splits_out_of_distribution`` (``_interpretability_mixin.py``), so the
    traversal feeds the decoder no continuous covariates even when it was built expecting
    them, and the matmul fails. Substituting zeros is the principled fix *because*
    :func:`scale_continuous_covariates` z-scores every continuous covariate: zero is the
    mean, so the traversal is evaluated at an average cell. Categorical covariates are
    already handled upstream and are left alone.

    Drop this shim once the upstream fix lands.
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


def interpretability_scores(
    embed,
    gene_names=None,
    key: str = "OOD_combined",
    directional: bool = True,
    hide_vanished: bool = True,
):
    """Genes x factors score table from ``embed.varm``, without reloading the model.

    Reproduces :meth:`scvi.external.DRVI.get_interpretability_scores` from the
    artifacts ``train_drvi.py`` already wrote, so inspection needs no GPU. Columns are
    ordered by reconstruction effect and titled ``DR n`` (``DR n+`` / ``DR n-`` when
    ``directional``); vanished factors/directions are dropped by default.
    """
    if gene_names is None:
        if "gene_names" not in embed.uns:
            raise KeyError("pass gene_names, or use an embedding with uns['gene_names']")
        gene_names = embed.uns["gene_names"]
    gene_names = pd.Index(np.asarray(gene_names).astype(str))

    if directional:
        effect = np.concatenate([embed.varm[f"{key}_positive"], embed.varm[f"{key}_negative"]])
        info = (
            pd.concat([embed.var.assign(direction="+"), embed.var.assign(direction="-")])
            .assign(title=lambda df: df["title"] + df["direction"])
            .reset_index(drop=True)
        )
        vanished = np.where(
            info["direction"] == "+",
            info["vanished_positive_direction"],
            info["vanished_negative_direction"],
        )
    else:
        effect = embed.varm[key]
        info = embed.var.assign(direction="").reset_index(drop=True)
        vanished = info["vanished"].to_numpy()

    info["keep"] = ~vanished if hide_vanished else True
    ordered = info[info["keep"]].sort_values(["order", "direction"])["title"]
    return pd.DataFrame(effect, columns=gene_names, index=info["title"]).loc[ordered].T


def top_genes_per_factor(scores: pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
    """Tidy ``factor, rank, gene, score`` table from an interpretability score matrix."""
    rows = []
    for factor in scores.columns:
        top = scores[factor].nlargest(n_top)
        rows.extend(
            {"factor": factor, "rank": rank, "gene": gene, "score": score}
            for rank, (gene, score) in enumerate(top.items(), start=1)
        )
    return pd.DataFrame(rows)


def split_by_sign(embed, hide_vanished: bool = True) -> pd.DataFrame:
    """Per-cell activation magnitude of each factor *direction*.

    DRVI factors are directional: ``DR n+`` and ``DR n-`` can encode unrelated programs,
    which is why the interpretability scores are computed per direction. A signed factor
    value conflates the two, so an association carried by only one direction is diluted
    by the cells sitting on the other side of zero. This returns ``relu(x)`` and
    ``relu(-x)`` as separate non-negative columns.

    Columns are labelled and ordered exactly like :func:`interpretability_scores`
    (``DR 1+``, ``DR 1-``, ``DR 2+``, ...), so the two tables join on them. Directions
    flagged vanished by ``set_latent_dimension_stats`` are dropped by default.
    """
    x = np.asarray(embed.X, dtype=np.float32)
    info = embed.var
    columns = {}
    for pos, (_, row) in enumerate(info.iterrows()):
        values = x[:, pos]
        for direction, magnitude, dead in (
            ("+", np.maximum(values, 0.0), row["vanished_positive_direction"]),
            ("-", np.maximum(-values, 0.0), row["vanished_negative_direction"]),
        ):
            if hide_vanished and dead:
                continue
            columns[f"{row['title']}{direction}"] = magnitude

    out = pd.DataFrame(columns, index=embed.obs_names)
    order = (
        pd.concat([info.assign(direction="+"), info.assign(direction="-")])
        .assign(title=lambda df: df["title"] + df["direction"])
        .sort_values(["order", "direction"])["title"]
    )
    return out[[c for c in order if c in out.columns]]


def direction_activity(embed, hide_vanished: bool = True) -> pd.DataFrame:
    """How often each factor direction is active, and how strongly.

    Context for :func:`factor_association` in directional mode: a direction active in 2%
    of cells with a high eta-squared means something very different from one active in 60%.
    """
    split = split_by_sign(embed, hide_vanished=hide_vanished)
    return pd.DataFrame(
        {
            "frac_active": (split > 0).mean(),
            "mean_when_active": split.where(split > 0).mean(),
            "max": split.max(),
        }
    )


def direction_asymmetry(assoc: pd.DataFrame) -> pd.DataFrame:
    """Per-factor gap between its ``+`` and ``-`` association profiles.

    Takes the directional output of :func:`factor_association`. A large gap means the two
    directions of one factor track different covariates, which is exactly the structure a
    signed association test hides.
    """
    rows = {}
    for name in assoc.index:
        base, direction = name[:-1].strip(), name[-1]
        rows.setdefault(base, {})[direction] = assoc.loc[name]
    records = {}
    for base, directions in rows.items():
        if set(directions) != {"+", "-"}:
            continue
        records[base] = (directions["+"] - directions["-"]).abs()
    out = pd.DataFrame(records).T
    out.index.name = "factor"
    return out


def _eta_squared_many(x: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Eta-squared of every column of ``x`` against one integer-coded grouping.

    Vectorized with ``bincount``: a per-group Python loop is O(n x n_groups), which is
    unusable at 902 ``library_prep`` levels.
    """
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    live = counts > 0
    grand = x.mean(axis=0)
    total = ((x - grand) ** 2).sum(axis=0)

    out = np.full(x.shape[1], np.nan)
    if live.sum() < 2:
        return out
    for i in range(x.shape[1]):
        if total[i] == 0:
            continue
        sums = np.bincount(codes, weights=x[:, i], minlength=n_groups)
        means = sums[live] / counts[live]
        out[i] = (counts[live] * (means - grand[i]) ** 2).sum() / total[i]
    return out


def _associate(
    x: np.ndarray,
    names,
    obs: pd.DataFrame,
    categorical: tuple[str, ...],
    continuous: tuple[str, ...],
) -> pd.DataFrame:
    """Association of every column of ``x`` with each named covariate.

    Categorical -> eta-squared (variance explained); continuous -> ``|Spearman rho|``.
    """
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    out = pd.DataFrame(index=pd.Index(names), dtype=float)

    for col in categorical:
        if col not in obs:
            logger.warning("skipping missing obs column %s", col)
            continue
        cat = pd.Categorical(obs[col])
        keep = cat.codes >= 0
        out[col] = _eta_squared_many(
            x[keep], cat.codes[keep].astype(np.intp), len(cat.categories)
        )

    for col in continuous:
        if col not in obs:
            logger.warning("skipping missing obs column %s", col)
            continue
        values = pd.to_numeric(obs[col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(values)
        if keep.sum() < 3:
            out[col] = np.nan
            continue
        # ranks every column against the covariate in one call
        rho = stats.spearmanr(x[keep], values[keep]).statistic
        out[col] = np.abs(np.atleast_2d(rho)[:-1, -1]) if x.shape[1] > 1 else abs(rho)
    return out


def factor_association(
    embed,
    categorical: tuple[str, ...] = (),
    continuous: tuple[str, ...] = (),
    directional: bool = False,
) -> pd.DataFrame:
    """Factors x covariates association matrix.

    Categorical covariates get eta-squared (variance explained, 0-1); continuous ones
    get ``|Spearman rho|``, so a single heatmap separates factors driven by biology
    from those tracking technical covariates.

    Parameters
    ----------
    directional
        ``False`` (default) associates on the signed factor value, one row per factor.
        ``True`` splits each factor into its two directions via :func:`split_by_sign`
        first, giving one row per ``DR n+`` / ``DR n-``. Prefer ``True`` whenever a
        factor might be bidirectional -- read it alongside
        :func:`direction_activity`.

    Only non-vanished factors (or directions, when ``directional``) are returned.
    """
    if directional:
        split = split_by_sign(embed, hide_vanished=True)
        return _associate(
            split.to_numpy(), split.columns, embed.obs, categorical, continuous
        )

    if {"order", "vanished", "title"} <= set(embed.var.columns):
        used = embed.var.sort_values("order")
        used = used[~used["vanished"]]
        x = np.asarray(embed[:, used.index].X)
        names = used["title"].to_numpy()
    else:
        # An scVI embedding has no vanished/order bookkeeping to reorder by, but it does
        # carry `used` from latent_stats -- and collapsed dimensions MUST be dropped here.
        # Eta-squared is a variance ratio, so it is scale-invariant and cannot tell a live
        # dimension from one sitting at the prior: a dimension with std 0.008 that wiggles
        # slightly with library scores just as high as a real one, which fills the heatmap
        # with noise that reads as signal.
        keep = embed.var.index
        if "used" in embed.var:
            keep = embed.var.index[embed.var["used"].to_numpy().astype(bool)]
            dropped = embed.n_vars - len(keep)
            if dropped:
                logger.info(
                    "dropping %d of %d collapsed latent dimensions (var['used'] is False); "
                    "eta-squared is scale-invariant and would score them like live ones",
                    dropped,
                    embed.n_vars,
                )
        subset = embed[:, keep]
        x = np.asarray(subset.X)
        names = (
            subset.var["title"].to_numpy()
            if "title" in subset.var
            else subset.var_names.to_numpy()
        )
    return _associate(x, names, embed.obs, categorical, continuous)
