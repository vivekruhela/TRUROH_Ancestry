# Release validation — 2026-10-01

**Result: 38 automated checks passed.** Python source compilation and Bash syntax checks also passed. Runtime versions are recorded in `build_environment.json`; the complete test log is included as `test_results.txt`.

The suite uses synthetic PLINK BED/BIM/FAM, RFMix MSP, phenotype and eigenvector fixtures. It executes the actual candidate construction, exact BED-counting metrics, ancestry/RDI calculations, locked selection, Firth associations and HTML/PDF reporting. BCFtools RG records are fixtures, not outputs from an external BCFtools execution. Scheduler calls are mocked; shell entry points and the batch argument forwarding are executed locally.

## Multiple-cohort regression checks

| Verified behavior | Evidence |
|---|---|
| Explicit selection | Both launchers reject a missing configuration; neither silently selects GAPP |
| One config interface | `--config` before/after the action, the explicit positional submit form and `TRUROH_CONFIG` work; an explicit flag overrides the environment |
| CH phenotype headers | `Age` and `Sex` normalize to `AGE` and `SEX`; explicit AD 1/2 coding yields the expected synthetic controls/cases |
| Additional covariates | An explicitly mapped numeric education variable is available for adjustment; missing values reduce the association complete-case set |
| Actionable errors | An unmapped covariate is identified by name with mapping guidance |
| Paths | Relative JSON paths resolve from the JSON directory; resolved copies remain valid in another directory; spaces in paths are exercised |
| Cohort separation | CH and GAPP preflights produce separate identities and signatures; a foreign project cannot overwrite another project's output files on failure |
| Run compatibility | A scientific config change blocks reuse; changing scheduling resources alone does not |
| Frozen submission | Both array and finalizer receive the same absolute resolved JSON path and SHA-256; editing the original JSON does not alter their config |
| Scheduler options | Resource constraints propagate separately to chromosome and finalizer commands; finalization uses `afterany` |
| Worker integrity | Altering a checksummed config fails before writing outputs; the actual batch script forwards the frozen path despite a conflicting inherited config variable |
| Submission preview | `--dry-run` does not call `sbatch` |
| Dynamic CH reports | Synthetic CH scoring, associations and HTML/PDF generation complete with CH filenames and Caribbean Hispanic labels; a one-ancestry EUR analysis produces the corresponding primary table |
| Diagnostic reports | Missing CH input yields a CH incomplete report and preserves the original error |

## Retained scientific regression checks

| Verified behavior | Evidence |
|---|---|
| Locked selection | Carrier rounding, exact threshold boundaries, failed gate rejection and missing-RDI rejection; N=2,278 requires 12 carriers |
| BED interpretation | Exact heterozygote/nonmissing counts, missing genotypes and chunk boundaries |
| Chromosome extraction | Genotypes, sample order, IDs and marker mapping preserved; internal phenotype masked |
| Candidate clustering | Transitive overlap chains cannot collapse without pairwise reciprocal support |
| HRS/backgrounds | Hand-calculated seed recall/span, no-hit handling, deterministic background lengths and exclusions |
| Phenotypes/PCA | Exact IDs and FID/IID pairs; supplied PCA precedence; technical denominator independent of missing age |
| Coding | Unsupported outcome codes fail; recognized APOE genotypes decode and opaque coding is not guessed |
| Ancestry | Cohort MSP subsetting, local proportions and missing-sample failure |
| Firth estimation | Analytic separation odds ratio, intercept-only estimate and adjusted-score numerical gradient |
| Ancestry effects | Centered-model effect at ancestry proportion one matches the uncentered parameter combination |
| Multiplicity | Expected Benjamini–Hochberg adjustment while preserving missing tests |
| Integration | Synthetic two-chromosome metrics, selected regions, association and reports |
| Empty/failed analyses | Zero candidates and all-QC-fail runs complete with valid schemas; missing chromosomes remain incomplete rather than biological zeros |

The core candidate, raw-metric, regional-scoring, scoring-math, Firth and RFMix helper source files are byte-identical to the prior adapter. The locked config is unchanged, and the seed-panel and locked-gate functions retain identical Python syntax trees. Bundled v3.8.0 reference files are unchanged. These checks are recorded in `release_validation.json`; original reference provenance remains in `reference/source_provenance.json`.

## Scope and limits

The original cohort inputs and live PLINK, BCFtools and SLURM installations were unavailable in the build environment. No real CH/GAPP counts, actual caller execution, cluster scheduling, or real-data association results have been verified here. Run the supplied `preflight` on the cluster to check the selected real inputs, coding, executable versions and BCFtools plugin. Preflight success establishes readiness checks; it does not establish that later analysis jobs will succeed.

The integration fixtures cover chromosomes 1–2, not genome-wide empirical performance. Source-compatible LOCO fallback indicators remain when training support is insufficient. Reported simulation-selection performance is reference information, not an estimate of real CH or GAPP accuracy.

## End-to-end validation with real PLINK and BCFtools — 2026-10-09

`tests/e2e/run_e2e.sh` was added after the 2.0.0 release checks. It exercises the complete user path with real executables: PLINK v1.9.1 (built from the plink-ng source) and BCFtools/HTSlib 1.21, on Python 3.13 with numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9 and matplotlib 3.11.

| Stage | Result |
|---|---|
| Synthetic cohort | N = 400, 39,944 markers on chr21–22, three-way AFR/EUR/NAM local ancestry (RFMix v2 MSP format), one planted recurrent NAM-founder ROH (chr21:15.0–17.5 Mb, 30 carriers), private 1–3 Mb ROHs in ~30% of individuals, 10 related pairs, one incomplete phenotype record |
| `configure.sh --answers` | Wrote a cohort JSON that passed the pipeline loader |
| `preflight` | Passed (N = 400, carrier floor 3) |
| `local` run | Completed; HTML + PDF report with 7 figures |
| `submit.sh --dry-run`, then `submit.sh` via the `sbatch` emulator | Two submissions: chromosome array `21,22` and a finalizer with `afterany`; all tasks exited 0 |
| Recovery | The planted region was the single high-confidence region on chr21; carrier recall 1.00; all locked gates satisfied; M0 converged with OR > 1 |
| Determinism | Local and SLURM-emulated runs gave identical `regions_scored.tsv` and `high_confidence_ROH.tsv` |
| Downstream | TRUROH_AD_Sensitivity ran on this output (APOE map, kinship file, 200 prediction draws) and wrote its report |

The emulator runs batch scripts synchronously and rejects empty `--partition`, `--account` and `--constraint` values. It verifies the commands and their dependency structure, but it is not a live SLURM scheduler.

## Reproduce

```bash
bash tests/run_tests.sh
bash tests/e2e/run_e2e.sh /tmp/truroh_e2e      # needs plink + bcftools on PATH
python3 -m compileall -q src tests
bash -n run.sh submit.sh config/environment.sh slurm/job.sbatch tests/run_tests.sh
```

Tests use temporary synthetic directories and do not require access to either real cohort. See `docs/MIGRATION_v2.md` for real-data launch commands.
