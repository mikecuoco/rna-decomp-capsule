# Creating the ADKP Bulk RNA-seq Code Ocean Data Asset

This document describes how to create a Code Ocean data asset from the ADKP bulk RNA-seq
files transferred to the SEA-AD highly-sensitive S3 bucket on 2026-10-07. The asset is
created once and then attached to any capsule that needs it.

## What is being exposed

All 65 files live under a single S3 prefix:

```
s3://sea-ad-prod-highly-sensitive-711387118892-us-west-2/adkp/genomics/
```

Three sub-prefixes, one per dataset:

| Sub-prefix | Tier | Contents | Size |
|---|---|---|---|
| `SEAAD_MTG_bulk_RNAseq/` | T1 (pilot) | 10 BAM + 10 FPKM + 20 FASTQ + 4 metadata | 23.5 GB |
| `ROSMAP_bulk_RNAseq/` | T2 (training) | 12 harmonized count matrices (batches 1–4) + legacy FPKM + Green n437 annotation + 4 metadata | 790 MB |
| `AMP-AD_DiverseCohorts_bulk_RNAseq/` | T3 (metadata only) | 3 metadata CSVs | 1.4 MB |

Access: AD Knowledge Portal controlled access (Synapse Data Use Certificate required).
All files must remain in highly-sensitive storage — do not copy to standard-tier S3 or
expose via a public data asset.

## Prerequisites

- Code Ocean account with access to the `sea-ad` organization
- AWS SSO access to the `sensitive` profile
  (`sea-ad-prod-highly-sensitive-711387118892-us-west-2`)
- Highly-sensitive data asset creation enabled for your account (contact Code Ocean support
  if the option is not visible)

## Steps

### 1. Open the Data Assets page

Go to **codeocean.allenneuraldynamics.org → Data Assets → New Data Asset**.

### 2. Fill in the asset details

| Field | Value |
|---|---|
| **Name** | `adkp-bulk-rnaseq` |
| **Description** | ADKP bulk RNA-seq for SEA-AD fine-type composition: SEA-AD MTG (T1 pilot), ROSMAP DLPFC (T2 training), Diverse Cohorts (T3 metadata). Transferred from Synapse 2026-10-07. AD Knowledge Portal controlled access. |
| **Tags** | `bulk-rnaseq`, `adkp`, `sea-ad`, `controlled-access` |

### 3. Choose the source

Select **Amazon S3** as the source type, then set:

| Field | Value |
|---|---|
| **Bucket** | `sea-ad-prod-highly-sensitive-711387118892-us-west-2` |
| **Prefix** | `adkp/genomics/` |
| **AWS region** | `us-west-2` |
| **Credential / IAM role** | the highly-sensitive role configured for Code Ocean (ask your admin if unsure) |

Leave **Recursive** checked so all sub-prefixes are included.

### 4. Set visibility

Set to **Private** (organization-only). Do not make this asset public — the data is
controlled access under a Synapse DUC.

### 5. Create the asset

Click **Create**. Code Ocean indexes the prefix; this takes a few minutes for 24+ GB.

### 6. Verify the file listing

Once indexed, confirm the asset shows three top-level directories:

```
adkp-bulk-rnaseq/
  AMP-AD_DiverseCohorts_bulk_RNAseq/
    metadata/
      AMP-AD_DiverseCohorts_assay_RNAseq_metadata.csv
      AMP-AD_DiverseCohorts_biospecimen_metadata.csv
      AMP-AD_DiverseCohorts_individual_metadata.csv
  ROSMAP_bulk_RNAseq/
    metadata/
      ROSMAP_Covariates_ages_censored.tsv
      ROSMAP_assay_RNAseq_metadata.csv
      ROSMAP_biospecimen_metadata.csv
      ROSMAP_clinical.csv
    processed_data/
      ROSMAP_batch{1..4}_gene_all_counts_matrix_clean.txt   (4 files)
      ROSMAP_batch{1..4}_Study_all_metrics_matrix_clean.txt (4 files)
      ROSMAP_batch{1..4}_provenance.csv                     (4 files)
      ROSMAP_RNAseq_FPKM_gene.tsv
      cell-annotation.n437.csv
  SEAAD_MTG_bulk_RNAseq/
    metadata/
      SEA-AD_MTG_assay_RNAseq-BULK.csv
      SEA-AD_biospecimen_metadata.csv
      SEA-AD_individual_metadata.csv
      SEA-AD_individual_metadata_harmonized.csv
    processed_data/
      SQ_BTR3002-02-{1..10}_*.bam              (10 BAM files)
      SQ_BTR3002-02-{1..10}_genes.fpkm_tracking (10 FPKM files)
    raw_data/
      SQ_BTR3002-02-{1..10}_*_R{1,2}_001.fastq.gz (20 FASTQ files)
```

Total should be **65 files**. Cross-check against `registry/datasets.yaml` if any are missing.

### 7. Attach to this capsule

In this capsule (**rna-decomp-capsule**, capsule-4857703):

1. Go to **Environment → Data**.
2. Click **Add Data Asset** and search for `adkp-bulk-rnaseq`.
3. Attach it. The files will mount at `/data/adkp-bulk-rnaseq/` inside every run.

## Path layout inside the capsule

After attaching, analysis code should reference files like:

```python
from pathlib import Path

DATA = Path("/data/adkp-bulk-rnaseq")

# T1 pilot
MTG_META   = DATA / "SEAAD_MTG_bulk_RNAseq/metadata"
MTG_PROC   = DATA / "SEAAD_MTG_bulk_RNAseq/processed_data"
MTG_RAW    = DATA / "SEAAD_MTG_bulk_RNAseq/raw_data"

# T2 training
ROSMAP_META = DATA / "ROSMAP_bulk_RNAseq/metadata"
ROSMAP_PROC = DATA / "ROSMAP_bulk_RNAseq/processed_data"

# T3 metadata
DC_META    = DATA / "AMP-AD_DiverseCohorts_bulk_RNAseq/metadata"
```

## Provenance

Source Synapse IDs and verified MD5s for every file are in
`code/bulk_composition/registry/datasets.yaml`. If a file is ever suspect, compare its
S3 ETag against the `md5` field in that registry before re-using it.
