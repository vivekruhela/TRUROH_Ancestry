#!/usr/bin/env python3
"""Assertions for the synthetic end-to-end test (see run_e2e.sh)."""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--truth", type=Path, required=True)
ap.add_argument("--local", type=Path, required=True)
ap.add_argument("--slurm", type=Path, required=True)
ap.add_argument("--sbatch-log", type=Path, required=True)
a = ap.parse_args()
truth = json.loads(a.truth.read_text())
reg = truth["planted_region"]
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


for label, root in [("local", a.local), ("slurm", a.slurm)]:
    print(f"[{label}] {root}")
    status = json.loads((root / "final_status.json").read_text())
    check(status.get("state") == "completed", "final_status.state == completed")
    for f in ["results/regions_scored.tsv", "results/high_confidence_ROH.tsv", "results/associations.tsv",
              "results/individual_ROH_burden.tsv", "results/locked_thresholds.tsv", "run_manifest.json"]:
        check((root / f).is_file(), f"{f} exists")
    reports = list((root / "report").glob("TRUROH_*_report.html"))
    check(bool(reports) and reports[0].stat().st_size > 10_000, "self-contained HTML report written")
    check(bool(list((root / "report").glob("*.pdf"))), "PDF report written")
    check(bool(list((root / "report/figures").glob("*.png"))), "report figures written")
    hc = pd.read_csv(root / "results/high_confidence_ROH.tsv", sep="\t")
    hit = hc[(hc.CHR == reg["chrom"]) & (hc.CONS_START < reg["end"]) & (hc.CONS_END > reg["start"])]
    check(len(hit) == 1, "planted shared ROH recovered as exactly one high-confidence region")
    if len(hit):
        members = pd.read_csv(root / "results/members.tsv.gz", sep="\t", dtype=str)
        rid = hit.REGION_ID.iloc[0]
        found = set(members.loc[members.REGION_ID == rid, "IID"])
        recall = len(found & set(truth["carriers"])) / len(truth["carriers"])
        check(recall >= 0.9, f"planted carriers recovered (recall {recall:.2f} >= 0.90)")
        lt = pd.read_csv(root / "results/locked_thresholds.tsv", sep="\t").iloc[0]
        row = hit.iloc[0]
        check(row.LENGTH_BP >= lt.LENGTH_BP_MIN and row.MEDIAN_HVR_RAW <= lt.HVR_MAX and row.MEAN_HRS_RAW >= lt.HRS_MIN and row.RDI_A >= lt.RDI_MIN,
              "selected region satisfies every locked GRID_ID 1433 threshold")
        assoc = pd.read_csv(root / "results/associations.tsv", sep="\t")
        m0 = assoc[(assoc.REGION_ID == rid) & (assoc.MODEL == "M0")]
        check(len(m0) == 1 and m0.STATUS.iloc[0] == "converged" and m0.OR.iloc[0] > 1, "M0 association converged with OR > 1 for the planted (risk) region")

calls = [json.loads(line) for line in a.sbatch_log.read_text().splitlines()]
check(len(calls) == 2, "two sbatch submissions (chromosome array + report)")
if len(calls) == 2:
    check(calls[0]["options"].get("array", "").startswith("21,22"), "chromosome array covers the configured autosomes")
    check(calls[1]["options"].get("dependency", "").startswith("afterany:"), "report job depends afterany on the array")
    check(all(t["rc"] == 0 for c in calls for t in c["tasks"]), "every emulated SLURM task exited 0")
for f in ["results/regions_scored.tsv", "results/high_confidence_ROH.tsv"]:
    x = pd.read_csv(a.local / f, sep="\t").sort_values("REGION_ID").reset_index(drop=True)
    y = pd.read_csv(a.slurm / f, sep="\t").sort_values("REGION_ID").reset_index(drop=True)
    cols = [c for c in x.columns if c in y.columns and c not in {"N_SAMPLES"}]
    check(x[cols].equals(y[cols]), f"local and SLURM runs give identical {f}")
if failures:
    raise SystemExit(f"{len(failures)} end-to-end check(s) failed")
print("All end-to-end checks passed")
