# TRU-ROH — ancestry-aware high-confidence runs of homozygosity (locked GRID_ID 1433)

TRU-ROH calls runs of homozygosity (ROH) in an admixed cohort, keeps only **recurrent, technically
reliable, ancestry-adjusted high-confidence ROH regions**, and tests them for association with
Alzheimer's disease (AD) using global and local (RFMix) ancestry. It is the real-data arm of the
TRU-ROH framework: every threshold was locked from simulation in
[TRUROH_Simulations](https://github.com/vivekruhela/TRUROH_Simulations) and is **not re-tuned on
real data**. Region-level AD sensitivity models (ROH × APOE, local ancestry, GLMM/kinship) are run
downstream by [TRUROH_AD_Sensitivity](https://github.com/vivekruhela/TRUROH_AD_Sensitivity).

The pipeline runs on any SLURM cluster (`sbatch` array + dependent finalizer) or sequentially on one
node, and produces result tables, figures and a self-contained HTML/PDF report.

```
PLINK BED/BIM/FAM ─┐
RFMix v2 .msp.tsv ─┤   per chromosome (SLURM array)                finalizer (afterany)
phenotypes + PCs ──┘   ├─ BCFtools/RoH calls (no LD pruning)       ├─ locked gates → high-confidence ROH
                       ├─ candidate regions (≥1 Mb, recurrent)     ├─ Firth logistic AD models (M0, M1, APOE, LA)
                       ├─ HVR / HRS technical metrics              ├─ per-person ROH burden
                       └─ ancestry-adjusted RDI                    └─ figures + HTML/PDF report
```

## Locked parameters (GRID_ID 1433)

| Parameter | Value |
|---|---|
| ROH caller | BCFtools `roh` (no caller-input LD pruning) |
| Minimum region length | ≥ 1 Mb |
| Minimum carriers | max(3, ⌈0.005 × N⌉) (e.g. 12 for N = 2,278) |
| Heterozygous-variant rate (HVR, median raw) | ≤ 0.005 |
| Homozygous-run support (HRS, mean raw) | ≥ 0.5 |
| Recurrence-depth index (RDI, ancestry-adjusted) | ≥ 0.7 |
| Technical assessability | required |

They are stored in `config/locked_grid1433.json`, are part of the run signature, and cannot be changed
from the cohort configuration. Equations are in `docs/METHODS.md`.

## Requirements

* Linux, Bash ≥ 4, SLURM (`sbatch`) for cluster runs
* Python ≥ 3.10 with numpy, pandas, scipy, scikit-learn, matplotlib
* **PLINK 1.9** (not PLINK 2) and **BCFtools ≥ 1.18** with the `fill-tags` plugin

Easiest is the bundled conda environment:

```bash
git clone https://github.com/vivekruhela/TRUROH_Ancestry.git
cd TRUROH_Ancestry
conda env create -f environment.yml       # creates "truroh-grid1433"
conda activate truroh-grid1433
```

Clusters that provide PLINK/BCFtools as modules can use those instead; the wizard records them.

## Your inputs

| Input | Format |
|---|---|
| Genotypes | QC'd, **not LD-pruned** PLINK 1 binary `prefix.bed/.bim/.fam`, GRCh38, autosomes. The FAM defines the cohort and the carrier-floor denominator N |
| Local ancestry | RFMix v2 `chr{CHR}.msp.tsv` per chromosome (one path with a `{CHR}` placeholder), haplotype columns `IID.0`/`IID.1`; optional IID→RFMix-ID map |
| Phenotypes | Tab-separated file with a header: FID, IID, AD status, age, sex, optionally APOE genotype and extra numeric covariates |
| PCs | Eigenvector file (`FID IID PC1 … PC5 …`, with or without header), e.g. from `plink --pca` |

You tell the pipeline how outcome and sex are coded (e.g. AD `2 = case, 1 = control`; sex `2 = female`).
Nothing is guessed: unmapped codes stop preflight with a clear message.

## Quick start

### 1 · Configure (interactive wizard)

```bash
bash configure.sh
```

The wizard asks for every input, output and scheduler setting, and inspects your files while you
answer it. It reads the FAM size and BIM autosomes, lists the phenotype header and the observed outcome
and sex codes, and reads the ancestry labels from the MSP header. It then writes `config/<PROJECT>.json`,
validates it with the same loader the pipeline uses, and (if you name cluster modules or a conda
environment) writes `config/site.env`, which every SLURM job sources. It asks about:

* project ID and report display name
* PLINK prefix and chromosomes (`1-22`, or e.g. `21,22` for a test)
* RFMix MSP pattern and optional sample map
* phenotype file, column names, outcome/sex codes, extra numeric covariates
* PCA file, association covariates, ancestries (primary, e.g. `NAM`; reference, e.g. `EUR`)
* output directory
* PLINK/BCFtools paths, cluster modules / conda environment
* SLURM partition, account, constraint, CPUs, memory, time, array concurrency (blank = omitted)

Start from an existing file with `bash configure.sh --from config/cohort.example.json`, or run it
non-interactively with `--answers answers.json` (see `tests/e2e/run_e2e.sh` for a complete example).
You can also copy `config/cohort.example.json` and edit it by hand; `config/ch.json` and
`config/gapp.json` show the Caribbean Hispanic and Peruvian (GAPP) settings used in the paper, with
placeholder paths.

### 2 · Check

```bash
bash run.sh --config config/MY_COHORT.json config      # print normalized settings, touch nothing
bash run.sh --config config/MY_COHORT.json preflight   # check files, IDs, coding, tool versions
bash submit.sh --config config/MY_COHORT.json --dry-run  # write the plan, print the sbatch commands
```

### 3 · Run on SLURM

```bash
bash submit.sh --config config/MY_COHORT.json
bash run.sh   --config config/MY_COHORT.json status    # progress per chromosome
```

`submit.sh` submits one array job (one task per chromosome, `%max_concurrent`) and a finalizer with
`--dependency=afterany`. If a chromosome fails, the finalizer still writes a diagnostic report, but it
withholds scoring until every configured chromosome is complete. Re-running `submit.sh` reuses completed
chromosomes after verifying their checksums.

### Or run on one node

```bash
bash run.sh --config config/MY_COHORT.json local
```

Individual stages can be (re)run as `chromosome <N>`, `finalize`, `associate` or `report`.

## Outputs

```
<output_root>/
├── run_manifest.json              cohort, N, scientific signature, software and input hashes
├── final_status.json              completed | incomplete, with reasons
├── inputs/                        cohort sample list, prepared phenotypes, phenotype audit
├── chr*/                          per-chromosome calls, metrics, status.json
├── submissions/<id>/              frozen read-only config + SHA-256 used by every job
├── results/
│   ├── high_confidence_ROH.tsv        locked high-confidence ROH regions  ← main result
│   ├── regions_scored.tsv             every candidate with metrics, gate results and rejection reasons
│   ├── members.tsv.gz                 carriers of each region
│   ├── associations.tsv               all AD models (M0 crude, M1 adjusted, APOE, ROH×APOE, local ancestry)
│   ├── primary_<ANC>_associations.tsv models for the primary ancestry (e.g. NAM)
│   ├── primary_ancestry_selection.json how the primary ancestry was chosen (explicit or auto) + mean global ancestry
│   ├── AD_associated_nominal.tsv      nominal P < 0.05 (the paper reports nominal associations)
│   ├── individual_ROH_burden.tsv      per-person raw and high-confidence ROH burden
│   ├── global_ancestry.tsv, local_ancestry_for_association.tsv.gz
│   └── locked_thresholds.tsv          thresholds actually applied (incl. the carrier floor for this N)
└── report/
    ├── TRUROH_<PROJECT>_report.html   self-contained report (figures embedded)
    ├── TRUROH_<PROJECT>_report.pdf
    └── figures/  01_chromosome_status · 02_locked_selection · 03_global_ancestry · 04_pca ·
                  05_roh_burden · 06_ancestry_associations · 07_roh_karyogram
```

Zero high-confidence regions is a valid, completed result. Missing chromosomes or a failed preflight are
never reported as zeros.

### Next step: AD sensitivity models

```bash
git clone https://github.com/vivekruhela/TRUROH_AD_Sensitivity.git
bash TRUROH_AD_Sensitivity/run.sh --config config/MY_COHORT.json --apoe-map TRUROH_AD_Sensitivity/apoe_dosage_map.json
```

## Reproducibility safeguards

* **No default cohort.** Every command needs `--config` (or `TRUROH_CONFIG`); the selected project,
  config path and output directory are printed on each call.
* **Frozen submissions.** Each submission writes a read-only `config.resolved.json` plus its SHA-256.
  Every job verifies the hash before touching data, so editing your JSON after submission cannot change
  queued jobs.
* **Output ownership.** `run_identity.json` reserves an output directory for one project. If you change
  scientific inputs, coding, covariates, chromosomes or code, use a new output directory. Scheduling
  resources can change freely.
* Relative paths inside a JSON resolve from that JSON's directory. Absolute paths are recommended on HPC.

## Configuration reference

| Field | Meaning |
|---|---|
| `project` | Short filename-safe ID (letters, digits, `.`, `_`, `-`) |
| `cohort_label` | Display name in reports |
| `output_root` | Dedicated output directory |
| `plink_prefix`, `rfmix_msp_pattern`, `rfmix_sample_map`, `phenotype_file`, `pca_file` | Inputs (above) |
| `gene_annotation_file` | Optional CHR/BP1/BP2/gene table; blank uses the bundled GRCh38 annotation |
| `chromosomes` | Autosomes to analyse |
| `columns` | Source headers for `fid`, `iid`, `outcome`, `age`, `sex`, `apoe_genotype`, plus `covariates: {NAME: "Header"}` for extra numeric covariates |
| `outcome_map`, `sex_map` | Source code → 0/1 (outcome 1 = case; sex 1 = female) |
| `association` | `covariates`, `ancestries`, `primary_ancestry` (an ancestry from `ancestries`, or `"auto"`), `global_reference`, fit settings, APOE models |
| `resources` | `cpus`, `memory`, `time`, `max_concurrent`, `partition`, `account`, `constraint`, `final_*` |
| `tools` | `plink`, `bcftools` executables |

**`primary_ancestry`.** This setting only picks which ancestry's ROH × local-ancestry models form the pre-specified primary table
(`primary_<ANC>_associations.tsv`). ROH calling and the locked gates never use it, and every configured ancestry is still fitted.
Set it explicitly when you have a hypothesis. For example, the CH and GAPP configs use `NAM` for Native American founder ROH, although CH is
majority European. `"auto"` picks the configured ancestry with the highest mean global ancestry across the cohort's samples,
computed from the RFMix output after merging source aliases (AFR + Africa, EUR + Europe, AMR + America). It never uses the
outcome. Either way the choice, the mean global ancestries and any note are written to `results/primary_ancestry_selection.json`. Notes flag an
`auto` close call (top two within 0.05) or an explicit choice that is not the cohort majority.

**Multiple testing.** P values are nominal. No BH, Bonferroni or Holm correction is applied: with the number of individuals and
high-confidence regions in these cohorts, corrected tests have essentially no power. Associations are exploratory and are judged by
replication across cohorts.

`Age`/`AGE` and `Sex`/`SEX` normalize to `AGE` and `SEX`; PCs to `PC1`–`PC5`. Empty
partition/account/constraint values are omitted from `sbatch`. `final_constraint` defaults to
`constraint`. The modules and conda environment in `config/site.env` (or the `TRUROH_MODULES` variable)
are loaded by every job.

## Testing

```bash
bash tests/run_tests.sh                 # 38 unit/integration tests (synthetic fixtures, ~1 min)
bash tests/e2e/run_e2e.sh /tmp/e2e      # end-to-end with real PLINK + BCFtools (~1 min)
TRUROH_AD_DIR=../TRUROH_AD_Sensitivity bash tests/e2e/run_e2e.sh /tmp/e2e   # + AD pipeline
```

The end-to-end test builds an artificial admixed cohort: N = 400, chromosomes 21–22, three-way
ancestry, one planted recurrent NAM-founder ROH in 30 carriers, plus private ROHs and related pairs.
It then:

1. drives `configure.sh` non-interactively;
2. runs preflight and a local run;
3. submits through a synchronous `sbatch` emulator (`tests/e2e/fake_slurm`), which rejects empty
   `--constraint`/`--partition`/`--account`;
4. checks the results:
   * the planted region is the single high-confidence region,
   * carrier recall is ≥ 0.9 and every locked gate is satisfied,
   * the crude (M0) association converges with OR > 1,
   * the SLURM and local runs give identical results.

It does not exercise a live scheduler. See `docs/VALIDATION.md`.

## Repository layout

```
configure.sh, src/configure.py   interactive configuration wizard
run.sh, submit.sh                launchers (local actions / SLURM submission)
slurm/job.sbatch                 batch script used for array tasks and the finalizer
src/                             pipeline (candidates, metrics, scoring, Firth, ancestry, report)
config/                          locked thresholds, example cohort configs, environment.sh
resources/                       bundled GRCh38 gene annotation
reference/v3.8.0/                simulation-stage provenance (unchanged; contains the original lab paths)
docs/                            METHODS, VALIDATION, MIGRATION_v2
tests/                           unit tests and tests/e2e
```

## Citation

If you use TRU-ROH, please cite the accompanying manuscript (citation to be added on publication) and
this repository; see `CITATION.cff`. Related repositories:

* [TRUROH_Simulations](https://github.com/vivekruhela/TRUROH_Simulations) — simulation framework and parameter tuning
* [TRUROH_AD_Sensitivity](https://github.com/vivekruhela/TRUROH_AD_Sensitivity) — ROH–AD, ROH × APOE and local-ancestry sensitivity models
