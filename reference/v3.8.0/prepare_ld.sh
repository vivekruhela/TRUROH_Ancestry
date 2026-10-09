#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_modules

CHR="${CHR:?set CHR}"
LD="${LD:?set LD}"
threads="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS}}"
original_vcf="$(render_pattern "${ORIGINAL_VCF_PATTERN}" "${CHR}")"
original_pfx="$(render_pattern "${ORIGINAL_PLINK_PATTERN}" "${CHR}")"
out="$(ld_prefix "${LD}" "${CHR}")"
status_file="${out}.status.tsv"
mkdir -p "$(dirname "${out}")" "${RUN_ROOT}/inputs/seeds/chr${CHR}"

ld_completion_valid() {
    [[ -f "${out}.complete" ]] || return 1
    all_files_nonempty "${status_file}" "${out}.fam" || return 1
    local state
    state="$(status_value "${status_file}" STATE)" || return 1
    if [[ "${state}" == "no_variants_after_pruning" ]]; then
        return 0
    fi
    [[ "${state}" == "ready" ]] || return 1
    all_files_nonempty "${out}.bed" "${out}.bim" "${out}.vcf.gz"
}

if [[ -f "${out}.complete" && "${FORCE:-0}" != "1" ]]; then
    if ld_completion_valid; then
        echo "SKIP complete: chr${CHR} LD=${LD}"
        exit 0
    fi
    echo "WARN: stale LD completion detected for chr${CHR} LD=${LD}; rebuilding" >&2
fi
rm -f -- "${out}.complete"

if [[ "${LD}" == "no" ]]; then
    for ext in bed bim fam; do
        ln -sfn "${original_pfx}.${ext}" "${out}.${ext}"
    done
    ln -sfn "${original_vcf}" "${out}.vcf.gz"
    if [[ -s "${original_vcf}.tbi" ]]; then
        ln -sfn "${original_vcf}.tbi" "${out}.vcf.gz.tbi"
    elif [[ -s "${original_vcf}.csi" ]]; then
        ln -sfn "${original_vcf}.csi" "${out}.vcf.gz.csi"
    else
        "${BCFTOOLS_BIN}" index -f -t "${original_vcf}"
        ln -sfn "${original_vcf}.tbi" "${out}.vcf.gz.tbi"
    fi

    seed_dir="${RUN_ROOT}/inputs/seeds/chr${CHR}"
    "${PLINK_BIN}" --bfile "${original_pfx}" \
        --maf "${SEED_MAF}" --geno "${SEED_GENO}" --hwe "${SEED_HWE}" \
        --write-snplist --allow-no-sex --threads "${threads}" \
        --out "${seed_dir}/eligible"
    if [[ -s "${seed_dir}/eligible.snplist" ]]; then
        "${PLINK_BIN}" --bfile "${original_pfx}" \
            --extract "${seed_dir}/eligible.snplist" \
            --indep-pairwise "${SEED_LD_WINDOW}" "${SEED_LD_STEP}" "${SEED_LD_R2}" \
            --allow-no-sex --threads "${threads}" --out "${seed_dir}/seed"
    else
        : > "${seed_dir}/seed.prune.in"
        : > "${seed_dir}/seed.prune.out"
        echo "WARN: chr${CHR} has no eligible HRS seed variants; HRS will be recorded as unassessable" >&2
    fi
else
    r2="$(ld_r2 "${LD}")"
    "${PLINK_BIN}" --bfile "${original_pfx}" \
        --indep-pairwise "${LD_WINDOW_SNPS}" "${LD_STEP_SNPS}" "${r2}" \
        --allow-no-sex --threads "${threads}" --out "${out}.prune"
    retained="$(awk 'END {print NR + 0}' "${out}.prune.prune.in")"
    if [[ "${retained}" -eq 0 ]]; then
        rm -f -- "${out}.bed" "${out}.bim" "${out}.fam" "${out}.vcf.gz" \
            "${out}.vcf.gz.tbi" "${out}.vcf.gz.csi" "${out}.positions.tsv"
        ln -sfn "${original_pfx}.fam" "${out}.fam"
        write_ld_status "${status_file}" no_variants_after_pruning \
            "PLINK pruning retained zero variants; downstream callers will emit valid empty outputs" 0
        touch "${out}.complete"
        echo "Prepared chr${CHR} LD=${LD}: zero variants retained; recorded and continuing"
        exit 0
    fi
    "${PLINK_BIN}" --bfile "${original_pfx}" --extract "${out}.prune.prune.in" \
        --make-bed --allow-no-sex --keep-allele-order --threads "${threads}" --out "${out}"

    vcf_chrom=$("${BCFTOOLS_BIN}" index -s "${original_vcf}" | awk 'NR==1 {print $1}')
    [[ -n "${vcf_chrom}" ]] || { echo "ERROR: could not determine VCF chromosome: ${original_vcf}" >&2; exit 1; }
    awk -v chrom="${vcf_chrom}" 'BEGIN{OFS="\t"} {print chrom,$4}' "${out}.bim" > "${out}.positions.tsv"
    "${BCFTOOLS_BIN}" view -R "${out}.positions.tsv" --threads "${threads}" \
        -Oz -o "${out}.vcf.gz" "${original_vcf}"
    "${BCFTOOLS_BIN}" index -f -t --threads "${threads}" "${out}.vcf.gz"
fi

for f in "${out}.bed" "${out}.bim" "${out}.fam" "${out}.vcf.gz"; do
    require_file "${f}"
done
marker_count="$(awk 'END {print NR + 0}' "${out}.bim")"
write_ld_status "${status_file}" ready "LD dataset prepared successfully" "${marker_count}"
touch "${out}.complete"
echo "Prepared chr${CHR} LD=${LD}: ${out}"
