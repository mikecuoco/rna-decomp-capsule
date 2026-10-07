# Checkpoint — SEA-AD fine-type composition from bulk RNA-seq

Hard-gated, staged project. Each stage ends with an audit packet and stops for explicit approval.

## Status

| Stage | State | Approval |
|---|---|---|
| 1 Discovery and feasibility | Complete (2026-10-07), revision 1 (ADKP sweep) | Packet delivered; **gate decisions pending** |
| 2 Pilot data assets and QC | **Transfers only** (approved 2026-10-07): T3 ✅, T2 ✅, T1 ✅ (65/65 files verified) | QC, re-quantification and Code Ocean assets **not authorized** |
| 3–6 | Not started | Not authorized |

The full Stage 1 audit packet is in
`/root/.claude/plans/pasted-content-id-6a7d-make-each-shiny-island.md` (session plan file).
Approving that plan authorized only this checkpoint file.

## Stage 2: Synapse → S3 transfers
Approved plan: copy ADKP bulk RNA-seq to `prod-hs` = `s3://sea-ad-prod-highly-sensitive-711387118892-us-west-2/adkp/genomics/`.

| Tier | Dataset prefix | Files | Size | Status |
|---|---|---|---|---|
| T3 | `AMP-AD_DiverseCohorts_bulk_RNAseq/` | 3 (metadata) | 1.4 MB | ✅ verified (MD5 vs Synapse, S3 size) |
| T2 | `ROSMAP_bulk_RNAseq/` | 18 (12 harmonized-count files from batches 1–4, legacy FPKM, Green n437 annotation, 4 metadata) | 0.79 GB | ✅ verified |
| T1 | `SEAAD_MTG_bulk_RNAseq/` | 44 (10 BAM, 10 FPKM, 20 FASTQ, 4 metadata) | 23.45 GB | ✅ verified |

Before any transfer, the existing S3 contents were checked:
- Green 2024, Mathys 2023, the other IAC snRNA sets, Diverse Cohorts multiome and MIT PFC multiome were already present.
- No bulk RNA-seq was present.
- Details are in the plan file.

How the transfers were run:
- Code: `provision/transfers.yaml` and `provision/01_pull_adkp_bulk.py` (dry run by default; `--execute` to transfer).
- Registry: `registry/datasets.yaml`.
- Environment: `/scratch/.dotfiles/envs/conda/bulk-comp` (`environment/conda.yaml`).
- Credentials: Synapse uses the user's PAT in `~/.synapseConfig`; AWS uses the `sensitive` SSO profile.
- All three tiers were run with `--execute` by the user. A final dry run (2026-10-07) reports 65/65 files on S3 with matching size and MD5. The staging directory is empty.
- Next proposed action (needs approval): the rest of Stage 2, i.e. QC (pairing, count integrity, donor overlap) and SEA-AD BAM/FASTQ re-quantification, then Code Ocean data assets.
- Deferred, not authorized: Mayo, Huuki-Myers, and the Diverse Cohorts bulk BAMs (about 17 TB; Synapse has no processed matrix for them).

## Stage 1 key findings
- **No bulk RNA-seq is available locally or in SEA-AD.** The only `/data` asset is `multiregion` (snRNA, annotation with 207 supertypes, QNP, scCODA abundances).
- **Public SEA-AD S3 (unsigned):**
  - `Multiregion_2026/pseudobulk_objects/` holds 29 per-subclass h5ads, about 3.5 GB in total.
  - MERFISH is in `s3://sea-ad-spatial-transcriptomics/`.
  - The licence is the Allen Institute Terms of Use.
- **Paired candidates:**
  - ROSMAP/Green 2024: 419 paired donors; controlled (DUC).
  - Huuki-Myers DLPFC: 10 donors and 19 blocks with bulk × 6 preps, snRNA and RNAScope; public apart from the PsychENCODE snRNA copy.
  - CommonMind: overlap unverified.
- **Berson/Cellformer as described (MTG tissue-bulk + nuclei-bulk + snRNA) was not found.** Cellformer is ATAC, and its RNA follow-up uses mouse hippocampus.

## Stage 1 revision 1 (2026-10-07): AD Knowledge Portal sweep
Method: anonymous Synapse REST, using `entity/children`, annotations, and SQL on the ADKP file view
`syn11346063`. The Synapse MCP tools were not loaded in this session. Donor counts are intersections of
`individualID` annotations. **They are lower bounds**: annotation coverage is incomplete, and Green Exp2
files carry no `individualID`.

**This corrects Stage 1: SEA-AD MTG bulk RNA-seq exists.**
- Location: `Bulk RNAseq - MTG` (syn51792520) and assay metadata `SEA-AD_MTG_assay_RNAseq-BULK.csv` (syn52118824).
- Samples: 10 specimens, `SQ_BTR3002-02-1..10`, total RNA, paired-end, GRCh38.
- Files per specimen: FASTQ, STAR BAM, and Cufflinks `genes.fpkm_tracking`. There is no count matrix; counts must be produced from the BAMs.
- Donors: H20.33.011–020. All 10 have local MTG snRNA with native SEA-AD supertypes (14.7k–21.1k nuclei each). Most also have 8–9 other regions.
- Pathology spread: ADNC is Low ×1, Intermediate ×5, High ×4. None is a severely affected donor.

ADKP bulk × single-nucleus pairs (same donor, same region):
| Cohort | snRNA source | Bulk source | Paired donors |
|---|---|---|---|
| SEA-AD MTG | SEA-AD snRNA (local) | SEA-AD bulk syn51792520 | 10 |
| ROSMAP DLPFC | MIT_ROSMAP_Multiomics PFC (Mathys 2023, 427 donors) | ROSMAP bulk syn3388564 | ≥261 |
| ROSMAP DLPFC | Green 2024 Exp2, syn31512863 | ROSMAP bulk | 419 per the paper; unverifiable from annotations |
| Diverse Cohorts DLPFC | DC 10x Multiome (301 donors) | DC bulk ∪ ROSMAP bulk | ≥179 |
| Diverse Cohorts STG | DC Multiome (310) | DC bulk | ≥59 |
| Diverse Cohorts caudate | DC Multiome (191) | DC bulk ∪ ROSMAP caudate bulk | ≥106 |
| Mayo temporal cortex | MC-BrAD + MC_snRNA | MayoRNAseq | ≥23 |

Notes:
- The Green Synapse ID "conflict" is resolved: syn53366818 is the "Cell type objects" subfolder of syn31512863 (DLPFC Exp2). It also holds `cell-annotation.n437.csv` (syn53694215).
- About 80% of Diverse Cohorts bulk IDs are non-Rush numeric IDs, so DC overlap is likely undercounted. A Rush projid mapping exists at syn55128291.
- The DC multiome DLPFC donors and the MIT PFC donors share 71 donors. Splits must therefore group donors across studies.

## Decisions awaiting review (recommendation first)
1. Pilot paired dataset: ROSMAP DLPFC (Green 2024).
2. Independent validation: Huuki-Myers DLPFC, with SEA-AD MERFISH as a secondary check.
3. Access-free pre-pilot: SEA-AD multiregion pseudobulk plus DFC donor objects.
4. Target taxonomy: the 2026-06-22 multiregion taxonomy (207 supertypes), with subclass as the fallback.
5. Baselines: NNLS, Bisque, hspe/dtangle, BayesPrism, CelMod and Scaden. TAPE, BLEND and HIDE are deferred.
6. CommonMind: defer.
7. Berson/Cellformer: citation needed from the user.
8. Housekeeping: confirm the `code/` deletions and the empty `conda.yaml`; confirm this directory and the env path `/scratch/.dotfiles/envs/conda/bulk-comp`.

Revised recommendation after revision 1, still awaiting approval:
- Pilot: SEA-AD MTG, 10 paired donors. It is small, but it is the only set with truth already in the SEA-AD taxonomy and no mapping step.
- Training scale: ROSMAP DLPFC (Mathys and Green snRNA × ROSMAP bulk).
- Independent validation: Diverse Cohorts STG/caudate, Mayo temporal cortex, and Huuki-Myers DLPFC (assay shift and imaging).

## Blockers
- AD Knowledge Portal DUC / PsychENCODE access for ROSMAP and CMC: unconfirmed.
- Synapse MCP tools are not loaded in this session. Authenticated reads are needed for the metadata CSVs: Green n437 annotation, DC projid mapping, SEA-AD bulk assay CSV.
- No Code Ocean CLI or SDK, no AWS credentials, no Synapse token.
- No analysis environment; `environment/conda.yaml` is empty.
- No licence is declared for CelMod or BLEND, and the HIDE code licence is unknown.

## Next proposed action
Stage 2, once approved:
- Fetch the SEA-AD pseudobulk, the DFC metadata, and the Huuki-Myers processed bulk and RNAScope data into `/scratch/bulk-comp/raw/` with checksums.
- Build the environment.
- Write import scripts and an asset registry.
- Document the manual Code Ocean asset-creation steps.
- ROSMAP only after access is confirmed.
