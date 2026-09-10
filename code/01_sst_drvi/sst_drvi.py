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
FILE_INDEX = SCRATCH / "gpboost_file_index.csv"

PREPARED_SST = PREPARED_DIR / "sst_counts.h5ad"

#: Glob for the per-supertype, QC-passed ("goodcells") count matrices.
GPBOOST_GLOB = "**/*_goodcells_for_gpboost.h5ad"

logger = logging.getLogger("sst_drvi")


def run_name(n_latent: int, include_chodl: bool = False) -> str:
    """Stable name for one training run, used for model/embedding paths."""
    scope = "sst_chodl" if include_chodl else "sst"
    return f"{scope}_k{n_latent}"


def model_dir(n_latent: int, include_chodl: bool = False) -> Path:
    return MODELS_DIR / run_name(n_latent, include_chodl) / "drvi"


def embed_path(n_latent: int, include_chodl: bool = False) -> Path:
    return EMBED_DIR / f"{run_name(n_latent, include_chodl)}_embed.h5ad"


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


#: Suffix marking a covariate column this module derived, rather than one from the data.
SCALED_SUFFIX = " [scaled]"


def scale_continuous_covariates(adata, keys: Sequence[str]) -> list[str]:
    """Write standardized copies of continuous covariates and return their names.

    scvi-tools stacks ``continuous_covariate_keys`` verbatim -- it does **not** center or
    scale them. Feeding a raw count column such as ``Number of UMIs`` (195 to 2.3e5) into
    the encoder/decoder would dominate every other input, so each column is log1p'd when
    it is count-like and then z-scored. Derived columns are suffixed with
    :data:`SCALED_SUFFIX`; the originals are left untouched.
    """
    names = []
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
        logger.info(
            "  covariate %-30s log1p=%-5s mean=%.4g std=%.4g -> z-scored%s",
            key,
            logged,
            mean,
            std,
            f", {n_missing} missing filled with 0" if n_missing else "",
        )
    return names


def build_model(adata, cfg: DrviConfig):
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


def latent_embedding(model, adata, cfg: DrviConfig):
    """AnnData of the latent space: cells x latent dimensions, obs carried over."""
    import anndata as ad

    latent = model.get_latent_representation(adata, batch_size=cfg.batch_size)
    # setup_anndata writes _scvi_* bookkeeping columns into obs; they are noise here.
    obs = adata.obs.drop(columns=[c for c in adata.obs.columns if c.startswith("_scvi")])
    embed = ad.AnnData(latent, obs=obs.copy())
    embed.var_names = [f"DR_{i + 1}" for i in range(embed.n_vars)]
    return embed


# ------------------------------------------------------------------------- loaders

def load_prepared(path: Path = PREPARED_SST, backed: bool | str = "r"):
    """Load the prepared SST counts. ``backed="r"`` keeps 231k x 19k off the heap."""
    import anndata as ad

    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run prepare_sst.py first")
    return ad.read_h5ad(path, backed=backed) if backed else ad.read_h5ad(path)


def load_embedding(n_latent: int = 64, include_chodl: bool = False):
    """Load the latent AnnData written by ``train_drvi.py``."""
    import anndata as ad

    path = embed_path(n_latent, include_chodl)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run train_drvi.py first")
    return ad.read_h5ad(path)


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

    used = embed.var.sort_values("order")
    used = used[~used["vanished"]]
    x = np.asarray(embed[:, used.index].X)
    return _associate(x, used["title"].to_numpy(), embed.obs, categorical, continuous)
