# Migration from the GAPP-only 1.0.0 adapter

1. Extract the complete `TRUROH_GRID1433_v2.0.0` directory. Do not copy the edited v1 `run.sh`, `submit.sh`, or `report.py` over these files.
2. Use the included `config/ch.json` for the uploaded CH input paths, or copy `config/cohort.example.json` for another cohort. Select it explicitly with `--config` for both preflight and submission.
3. The CH and GAPP example output roots are new v2 directories. Existing scientific outputs are not deleted or automatically migrated; code/input signatures intentionally prevent reuse across the release change.
4. Outcome and sex coding remain explicit. The uploaded CH mapping of AD 1=control, 2=case is retained. No coding inference was added.
5. `Age`/`Sex` entries in the association list now normalize to `AGE`/`SEX`; the source column mappings still use the exact phenotype headers.
6. CH/GAPP scheduler examples retain 4 CPUs and 64G for chromosome jobs, with 4 CPUs and 32G for finalization. Partition, account and node constraints are left empty for user configuration. All genotype, phenotype, PCA and RFMix input paths from the uploaded CH JSON are retained.
7. If those modules exist on your cluster, set `TRUROH_MODULES="Plink/1.9.10 BCFTOOLS"` before preflight/submission. Use an activated environment instead if preferred.
8. Intermediate filenames now use `cohort` rather than `gapp`; sample-audit field names are cohort-neutral. Reports use `TRUROH_<project>_report.*`; the primary ancestry table uses `primary_<configured ancestry>_associations.tsv`.

## Why the original preflight and submit disagreed

The original `submit.sh` inserted `config/gapp.json` independently of the selection in the manually edited `run.sh`. Hard-coding CH in one launcher did not establish one shared configuration interface. The old package also checked `project == GAPP` and contained GAPP names in manifests, reports, sample audits, intermediate files and job names.

Version 2 removes the implicit GAPP default. `submit.sh` forwards its arguments to the same Python parser as `run.sh`. Each invocation prints its selected config; every submitted worker receives a read-only, checksummed per-submission copy. CH runs cannot overwrite GAPP output metadata, even on failure.

The `report.py` Path/string concatenation and indentation problems are resolved in the delivered module. PDF writing and the HTML download link share one project-based report stem. Copying this complete release avoids manual indentation edits.

## CH commands

```bash
cd TRUROH_GRID1433_v2.0.0
export TRUROH_MODULES="Plink/1.9.10 BCFTOOLS"
bash run.sh --config config/ch.json preflight
bash submit.sh --config config/ch.json
```
