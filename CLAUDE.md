# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Code Ocean capsule for deconvolving fine cell-type composition from bulk RNA-seq using the SEA-AD
cohort and paired public datasets from the AD Knowledge Portal (ADKP). The project is staged and
hard-gated: each stage produces an audit packet and stops for explicit approval before the next begins.

`code/bulk_composition/CHECKPOINT.md` is the authoritative project status; read it before proceeding
with any stage. This file covers what spans several files.

## Environment and paths

```bash
PY=/scratch/.dotfiles/envs/conda/bulk-comp/bin/python
mamba env update -p /scratch/.dotfiles/envs/conda/bulk-comp -f environment/conda.yaml
```

`environment/conda.yaml` is hand-edited (direct dependencies only). `~/.condarc` points `envs_dirs`
and `pkgs_dirs` at `/scratch`. The `bulk-comp` Jupyter kernel is not yet registered.

Capsule layout: `/data` is immutable input (never write here), `/scratch/bulk-comp/` holds every
artifact, `/results` is for explicitly requested final outputs only.

## Code layout

All analysis code lives under `code/bulk_composition/`:

- **`CHECKPOINT.md`** — stage-gate tracker; current status, gate decisions, blockers, next steps.
- **`provision/01_pull_adkp_bulk.py`** — Synapse → S3 transfer script. Dry-run by default;
  pass `--execute` to transfer. Uses `~/.synapseConfig` for Synapse auth and the `sensitive` AWS
  SSO profile for S3.
- **`provision/transfers.yaml`** — transfer manifest (source Synapse IDs, S3 keys, expected MD5s).
- **`registry/datasets.yaml`** — dataset provenance registry written after each verified transfer.

## Current status (as of 2026-10-07)

Stage 2 transfers are complete: 65/65 files on
`s3://sea-ad-prod-highly-sensitive-711387118892-us-west-2/adkp/genomics/`, verified by MD5 and size.
Stage 2 QC, re-quantification, and Code Ocean data-asset creation are **not yet authorized**.
Stages 3–6 are not started. See `CHECKPOINT.md` for gate decisions awaiting review.

## Conventions

- Staged work: do not create `run`/`run.sh` or `conda-lock` until explicitly requested.
- Keep derived data out of `/data`. Write scratch intermediates to `/scratch/bulk-comp/`.
- Comments explain *why* (a constraint, a workaround, a subtle invariant), not what.
- Prefer adding a flag with a documented default over changing a default.
- Reference code as `file:line`.
