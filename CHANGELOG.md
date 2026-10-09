# Changes

## 2.0.0 public packaging — 2026-10-09 UTC

No change to the analysis code, locked thresholds or run signature equations.

- `configure.sh` interactive/non-interactive configuration wizard that inspects the user's files and writes a validated cohort JSON plus `config/site.env` (modules/conda for SLURM jobs).
- `config/environment.sh` sources `config/site.env` when present.
- Cohort example configs use placeholder paths instead of site-specific cluster paths.
- End-to-end test with real PLINK 1.9/BCFtools on a synthetic admixed cohort, with an `sbatch` emulator (`tests/e2e`).
- Rewritten README; `.gitignore`, `CITATION.cff`, GitHub Actions workflow.

## 2.0.0 — 2026-10-01 UTC

- Multi-cohort configuration loader; remove GAPP project restriction and default configuration.
- Shared `--config` interface for run, submit and batch workers; explicit config overrides TRUROH_CONFIG.
- Read-only per-submission resolved JSON and worker checksum validation.
- Cohort-specific job/report identity, generic intermediate paths, configurable ancestry panels and primary table.
- Normalize mapped source covariate headers; support explicitly mapped additional numeric covariates.
- Validate configs before touching outputs and protect directories against cross-cohort writes.
- Preserve code/input signatures; allow scheduling/display-only changes without biological invalidation.
- Add scheduler constraint/final_constraint propagation and submission dry-run.
- Correct report Path assembly, cohort labels, link consistency and indentation; preserve original analysis errors if reporting also fails.
- Include uploaded CH paths, a GAPP config and a new-cohort template with separate output roots.
- Preserve locked GRID_ID 1433 and core scientific implementations.
- Extend regression coverage from 22 to 38 checks.

## 1.0.0

Original GAPP-only real-data adapter of TRUROH-SIM v3.8.0.
