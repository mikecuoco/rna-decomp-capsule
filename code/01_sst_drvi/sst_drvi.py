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

from collections import Counter

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


def model_dir(n_latent: int, include_chodl: bool = False, model: str = "drvi") -> Path:
    """Directory one trained model was saved into. ``model`` is ``"drvi"`` or ``"scvi"``."""
    return MODELS_DIR / run_name(n_latent, include_chodl) / model


def embed_path(n_latent: int, include_chodl: bool = False, model: str = "drvi") -> Path:
    """Latent AnnData for one run and model."""
    return EMBED_DIR / f"{run_name(n_latent, include_chodl)}_{model}_embed.h5ad"


# ------------------------------------------------------------------------ logging

def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging: stream to stdout, optionally tee to ``log_file``.

    Under non-spawn DDP every rank re-executes the script, so all of them would open the
    same file in ``mode="w"`` and write at colliding offsets -- which silently ate a line
    of rank 0's output in an early probe run. Ranks above zero get their own suffixed
    file instead, leaving the named log as rank 0's clean record.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        rank = int(os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or 0)
        if rank:
            log_file = log_file.with_suffix(f".rank{rank}{log_file.suffix}")
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


def precision_kwargs(precision: str | None) -> dict:
    """``{"precision": ...}`` when a non-default precision is requested, else empty.

    Kept separate from :func:`ddp_trainer_kwargs` because precision applies to
    single-GPU runs too, and because the runners test that function's return value for
    truthiness to decide whether early stopping is available -- folding an unrelated key
    into it would silently disable early stopping on one GPU.

    ``precision`` is not a named parameter of scvi-tools' ``Trainer``; it reaches
    Lightning through its ``**kwargs`` (``scvi/train/_trainer.py``). ``"16-mixed"`` is
    the interesting value on a T4, whose tensor cores are roughly 8x its fp32 peak and
    which has no bf16 support. Worth measuring rather than assuming: DRVI aggregates its
    split decoder with ``logsumexp`` over a log-space negative binomial, and fp16 can
    lose that to overflow.
    """
    return {"precision": precision} if precision else {}


def resolve_kl_warmup(cfg) -> int:
    """Epochs over which to ramp the KL weight, for either model family.

    Defaults to a third of the run, leaving two thirds at ``kl_weight = 1.0``. Both the
    ramp length and the fact that it is the *same* for DRVI and scVI matter:

    * A model still annealing its objective on the final epoch has no converged state to
      measure, so "still falling" cannot be separated from "the loss it is minimising is
      still changing". :class:`scvi.external.drvi.DRVITrainingPlan` defaults to
      ``n_epochs_kl_warmup="auto"``, which is ``max_epochs`` -- a longer budget would
      stretch the ramp with it and never settle.
    * Left to their own defaults the two families diverge. scvi-tools' plain
      ``TrainingPlan`` uses ``n_epochs_kl_warmup=400``, so a 200-epoch scVI run peaks at
      a KL weight of 0.5 and minimises ``recon + 0.5 * KL`` throughout, while DRVI
      reaches ~1.0. Both then report a comparably defined ELBO
      (``elbo = rec_loss + kl_local + kl_global / n``) obtained from different objectives,
      and the weaker penalty simply buys the better reconstruction.
    """
    if cfg.kl_warmup_epochs is not None:
        return int(cfg.kl_warmup_epochs)
    return max(int(cfg.max_epochs) // 3, 1)


def plan_kwargs(cfg) -> dict:
    """``plan_kwargs`` for ``model.train``, identical in shape for DRVI and scVI.

    Both runners go through here so the KL schedule and optimiser settings cannot drift
    apart between the two model families -- see :func:`resolve_kl_warmup`.
    """
    kwargs = {"lr": cfg.lr, "n_epochs_kl_warmup": resolve_kl_warmup(cfg)}
    if cfg.reduce_lr_on_plateau:
        # A plateau scheduler is a scheduler, not a callback, so unlike early stopping it
        # survives DDP. It monitors the validation ELBO, which requires
        # ``check_val_every_n_epoch`` to be set -- see :func:`ddp_trainer_kwargs`.
        kwargs.update(
            reduce_lr_on_plateau=True,
            lr_patience=cfg.lr_patience,
            lr_factor=cfg.lr_factor,
            lr_scheduler_metric="elbo_validation",
        )
    return kwargs


def check_kl_schedule(history: pd.DataFrame, warmup: int, max_epochs: int) -> float:
    """Log whether the KL ramp finished, and return the weight actually reached.

    A run whose ``kl_weight`` never reaches 1.0 optimised a down-weighted KL for its whole
    length, which inflates its reconstruction and makes its ELBO incomparable to a run
    that did reach 1.0 -- even though both report the same quantity. That happened here
    once already (see :class:`ScviConfig`) and was only noticed while interpreting the
    results, so it is checked automatically now rather than trusted.
    """
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

    Mirrors :class:`DrviConfig` on every knob the two models share -- latent size,
    architecture, batch handling, dispersion, schedule -- so that a difference between
    the two embeddings is attributable to DRVI's split decoder rather than to the setup.
    Two things cannot match:

    * ``gene_likelihood``: ``"pnb"`` is DRVI's log-space negative binomial and
      :class:`scvi.model.SCVI` does not accept it, so the plain ``"nb"`` is used.
    ``kl_warmup_epochs`` used to default to scvi's own warmup here, on the reasoning that
    DRVI's whole-run ramp is specific to its split decoder and imposing it would make this
    a non-standard baseline. That was a mistake: scvi's default is 400 epochs, so a
    200-epoch run peaked at a KL weight of 0.4975 and this model minimised
    ``recon + 0.5 * KL`` from start to finish while the DRVI fits reached ~1.0. Both still
    report an identically defined ELBO, so the two numbers invite a comparison that the
    schedules do not support -- the weaker penalty buys the better reconstruction. The
    warmup now resolves through :func:`resolve_kl_warmup`, shared with :class:`DrviConfig`,
    and the runners refuse to stay quiet if the weight does not reach 1.0.
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
    afterwards, and the covariate table in ``01_qc.ipynb`` reports it only if handed this
    record.
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


# ------------------------------------------ neuropathology: CPS and cell abundance
#
# The CPS_* columns are SEA-AD's continuous pseudo-progression scores, and they are the
# only truly continuous measure of disease severity in this cohort. They are *not*
# per-cell measurements: `CPS_Global` is one value per donor and `CPS_Local` one value
# per donor x brain region -- checked in section 1 of `03_progression.ipynb` rather than
# assumed. Correlating a per-cell latent value against them over 231,107 rows therefore
# tests 84 or 530 independent units with an n of 231,107 -- the p-value is meaningless and
# even the effect size is dominated by whichever donors contributed the most nuclei. That
# notebook aggregates to a unit first, and these names are what it reads.

PERTPY_DIR = DATA / "pertpy"
CPS_LOCAL_DIR = PERTPY_DIR / "CPS_Local"
SCCODA_OBJECTS_DIR = CPS_LOCAL_DIR / "objects"

SCCODA_SUMMARY_GLOB = "pertpy_summary_CPS_Local.*.csv"
SCCODA_RESULTS_GLOB = "*_Supertype_results.csv"
SCCODA_ABUNDANCE_GLOB = "*_Supertype_abundances.h5ad"

#: the pseudo-progression scores: the composite first, then its ABeta and pTau parts.
CPS_COLUMNS = (
    "CPS_Local",
    "CPS_Local_ABeta",
    "CPS_Local_pTau",
    "CPS_Global",
    "CPS_Global_ABeta",
    "CPS_Global_pTau",
)


# ------------------------------------------------------- the existing scCODA results
#
# `/data/multiregion/pertpy/CPS_Local/` holds a finished scCODA run over all 174
# supertypes, not something this capsule computed. Only the paths live here; the readers
# are in section 4 of `03_progression.ipynb`. Three artifacts:
#
#   pertpy_summary_CPS_Local.<date>.csv   the delivered per-region CPS_Local effect
#                                         ("Local Model"), one value per supertype
#   <class group>_Supertype_results.csv   the full sweep: every effect re-estimated
#                                         against each of the 174 possible reference
#                                         cell types, for 6 covariates and 11 regions
#   objects/<class group>_Supertype_abundances.h5ad
#                                         scCODA's input -- 907 libraries x 174
#                                         supertype counts, with CPS_Local in obs
#
# The summary is the sweep's region-agnostic `Global` fit, thresholded and broadcast. For
# every SST supertype the summary's repeated value matches the median `Global` effect over
# the 173 references to within 0.016, and the ones it writes as 0.0 are exactly those whose
# median posterior inclusion probability falls below ~0.83 (`SCCODA_INCLUSION_THRESHOLD`).
# Two departures from that rule, both verified here:
#
#   * a supertype confined to one region takes that region's own fit instead of Global --
#     `Sst_27-SEAAD` appears only in V1C, where the sweep gives -1.132 against a Global
#     estimate of +0.005, and the summary carries -1.106;
#   * MTG's values match neither the sweep's MTG rows nor Global (they differ by up to
#     1.1), so they come from outside this file -- presumably SEA-AD's separately-fit MTG
#     dataset. Treat an MTG cell as a different study's answer, not a regional contrast.
#
# So the two files answer different questions rather than duplicating one: the summary
# gives the delivered, thresholded effect, the sweep gives the evidence behind it -- the
# inclusion probability and how far the estimate moves as the reference cell type changes.


#: posterior inclusion probability above which a swept effect is called credible.
#: scCODA's own cut is FDR-derived per model (``credible_effects``) and is not stored in
#: the delivered files, so it is recovered from where the summary's own calls fall: on the
#: Global fit they separate ``Sst_9`` (median inclusion 0.815, written as 0.0) from
#: ``Sst_22`` (0.841, kept), and 0.83 reproduces every one of the 18 calls.
SCCODA_INCLUSION_THRESHOLD = 0.83


