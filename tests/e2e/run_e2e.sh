#!/usr/bin/env bash
# End-to-end test with REAL PLINK 1.9 and BCFtools on a synthetic admixed cohort.
#
#   bash tests/e2e/run_e2e.sh [WORKDIR]
#
# Stages: synthetic data -> configure.sh (non-interactive) -> preflight -> local run ->
# SLURM submission through a synchronous sbatch emulator -> result checks.
# If TRUROH_AD_DIR points to a TRUROH_AD_Sensitivity checkout, its association
# pipeline is then run on the TRU-ROH output as well.
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${1:-$(mktemp -d "${TMPDIR:-/tmp}/truroh_e2e.XXXXXX")}"
WORK="$(mkdir -p "${WORK}" && cd "${WORK}" && pwd)"
PY="${PYTHON_BIN:-python3}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for tool in plink bcftools; do
  command -v "${tool}" >/dev/null || { echo "ERROR: ${tool} not on PATH (activate environment.yml)"; exit 2; }
done
echo "[e2e] work directory: ${WORK}"

echo "[e2e] 1/6 synthetic cohort"
"${PY}" "${PACKAGE_ROOT}/tests/e2e/make_synthetic_cohort.py" --out "${WORK}/data"

echo "[e2e] 2/6 configuration wizard (non-interactive answers)"
cat > "${WORK}/answers.json" <<EOF
{"project": "SYNTH", "cohort_label": "Synthetic admixed test cohort",
 "plink_prefix": "${WORK}/data/cohort", "chromosomes": "21-22",
 "rfmix_msp_pattern": "${WORK}/data/chr{CHR}.msp.tsv", "rfmix_sample_map": "",
 "phenotype_file": "${WORK}/data/phenotypes.tsv",
 "columns.fid": "FID", "columns.iid": "IID", "columns.outcome": "AD", "columns.age": "Age",
 "columns.sex": "Sex", "columns.apoe_genotype": "APOE_reported",
 "outcome_case": "2", "outcome_control": "1", "sex_female": "2", "sex_male": "1",
 "extra_covariates": "", "pca_file": "${WORK}/data/cohort.eigenvec",
 "association.covariates": "AGE,SEX,PC1,PC2,PC3,PC4,PC5",
 "association.ancestries": "NAM,EUR,AFR", "association.primary_ancestry": "NAM",
 "association.global_reference": "EUR", "output_root": "${WORK}/out_local",
 "tools.plink": "plink", "tools.bcftools": "bcftools", "site.modules": "", "site.conda": "",
 "resources.partition": "", "resources.account": "", "resources.constraint": "",
 "resources.cpus": 2, "resources.memory": "4G", "resources.time": "01:00:00",
 "resources.max_concurrent": 2, "resources.final_cpus": 1, "resources.final_memory": "4G",
 "resources.final_time": "01:00:00", "resources.final_constraint": ""}
EOF
bash "${PACKAGE_ROOT}/configure.sh" --answers "${WORK}/answers.json" --output "${WORK}/local.json" --site-env "${WORK}/site.env" >/dev/null

echo "[e2e] 3/6 preflight + local run"
bash "${PACKAGE_ROOT}/run.sh" --config "${WORK}/local.json" preflight
bash "${PACKAGE_ROOT}/run.sh" --config "${WORK}/local.json" local > "${WORK}/local.log" 2>&1

echo "[e2e] 4/6 SLURM submission through the sbatch emulator"
"${PY}" - "${WORK}/local.json" "${WORK}/slurm.json" "${WORK}/out_slurm" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); c["output_root"] = sys.argv[3]; c["project"] = "SYNTH_SLURM"
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
export FAKE_SLURM_STATE="${WORK}/fake_slurm"
PATH="${PACKAGE_ROOT}/tests/e2e/fake_slurm:${PATH}" bash "${PACKAGE_ROOT}/submit.sh" --config "${WORK}/slurm.json" --dry-run > "${WORK}/dryrun.log"
PATH="${PACKAGE_ROOT}/tests/e2e/fake_slurm:${PATH}" bash "${PACKAGE_ROOT}/submit.sh" --config "${WORK}/slurm.json" > "${WORK}/submit.log"

echo "[e2e] 5/6 result checks"
"${PY}" "${PACKAGE_ROOT}/tests/e2e/check_outputs.py" --truth "${WORK}/data/truth.json" \
  --local "${WORK}/out_local" --slurm "${WORK}/out_slurm" --sbatch-log "${FAKE_SLURM_STATE}/sbatch_calls.jsonl"

if [[ -n "${TRUROH_AD_DIR:-}" ]]; then
  echo "[e2e] 6/6 AD association pipeline (${TRUROH_AD_DIR}) on the TRU-ROH output"
  bash "${TRUROH_AD_DIR}/run.sh" --config "${WORK}/local.json" --apoe-map "${TRUROH_AD_DIR}/apoe_dosage_map.json" \
    --kinship "${WORK}/data/cohort.kin0" --out-dir "${WORK}/out_assoc" --prediction-draws 200 > "${WORK}/assoc.log" 2>&1
  test -s "${WORK}/out_assoc/report.html" && test -s "${WORK}/out_assoc/tables/all_association_tests.tsv"
  echo "[e2e] association report: ${WORK}/out_assoc/report.html"
else
  echo "[e2e] 6/6 skipped (set TRUROH_AD_DIR to also test the AD association pipeline)"
fi
echo "[e2e] PASSED  (outputs in ${WORK})"
