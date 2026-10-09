#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
load_modules

CHR="${CHR:?set CHR}"; LD="${LD:?set LD}"
in="$(ld_prefix "${LD}" "${CHR}").vcf.gz"
outdir="$(caller_dir bcftools "${LD}" "${CHR}")"
mkdir -p "${outdir}"
if [[ "${FORCE:-0}" != "1" ]] && caller_completion_valid "${outdir}"; then echo "SKIP complete: ${outdir}"; exit 0; fi
rm -f -- "${outdir}/complete"
raw="${outdir}/bcftools.roh.txt"
marker_count="$(awk 'END {print NR + 0}' "$(ld_prefix "${LD}" "${CHR}").bim" 2>/dev/null || printf '0\n')"

if [[ "${marker_count}" -lt "${CALLER_MIN_SNPS}" ]]; then
    write_empty_normalized_calls "${outdir}/individual_roh.tsv"
    write_caller_status "${outdir}/status.tsv" bcftools "${LD}" "${CHR}" \
        skipped_insufficient_markers "${marker_count} markers retained; normalized ROH cannot meet the ${CALLER_MIN_SNPS}-marker floor" \
        "${marker_count}" 0
    touch "${outdir}/complete"
    echo "BCFtools/RoH chr${CHR} LD=${LD}: insufficient markers; recorded zero ROH and continued"
    exit 0
fi

if ! "${BCFTOOLS_BIN}" roh -G "${BCFTOOLS_GT_ERROR_PHRED}" --AF-tag AF \
    --rec-rate "${BCFTOOLS_REC_RATE}" -O r -o "${raw}" "${in}"
then
    write_caller_status "${outdir}/status.tsv" bcftools "${LD}" "${CHR}" fatal_tool_error \
        "BCFtools/RoH exited nonzero; this was not classified as a biological zero-result" "${marker_count}" 0
    exit 1
fi
if [[ ! -s "${raw}" ]]; then
    write_caller_status "${outdir}/status.tsv" bcftools "${LD}" "${CHR}" fatal_missing_output \
        "BCFtools/RoH exited successfully but created no readable raw output" "${marker_count}" 0
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/normalize_calls.py" --tool bcftools --ld "${LD}" --chr "${CHR}" \
    --input "${raw}" --output "${outdir}/individual_roh.tsv" \
    --min-length "${CALLER_MIN_LENGTH_BP}" --min-snps "${CALLER_MIN_SNPS}"
roh_count="$(table_data_rows "${outdir}/individual_roh.tsv")"
state=success_with_roh; detail="BCFtools/RoH completed and retained ${roh_count} normalized ROH"
if [[ "${roh_count}" -eq 0 ]]; then state=success_zero_roh; detail="BCFtools/RoH completed normally and retained no ROH under this parameter set"; fi
write_caller_status "${outdir}/status.tsv" bcftools "${LD}" "${CHR}" "${state}" "${detail}" "${marker_count}" "${roh_count}"
touch "${outdir}/complete"
