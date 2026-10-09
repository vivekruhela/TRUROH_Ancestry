#!/usr/bin/env python3
"""Ancestry-compatible fragment merging and complete-linkage reciprocal-overlap clustering."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd

from rfmix_utils import MSPData, safe_ancestry_label

REGION_BASE = ["REGION_ID", "TOOL", "LD", "CHR", "CONS_START", "CONS_END", "UNION_START", "UNION_END", "LENGTH_BP", "N_CARRIERS"]
MEMBER_BASE = ["REGION_ID", "TOOL", "LD", "CHR", "IID", "SEG_START", "SEG_END", "SEG_LENGTH_BP", "NSNP"]


def reciprocal(a: dict, b: dict) -> float:
    ov = max(0, min(a["END"], b["END"]) - max(a["START"], b["START"]) + 1)
    if ov <= 0:
        return 0.0
    return min(ov / (a["END"] - a["START"] + 1), ov / (b["END"] - b["START"] + 1))


def tv(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return math.inf
    return 0.5 * float(np.abs(a - b).sum())


def merge_fragments(df: pd.DataFrame, msp: MSPData, gap: int, max_tv: float) -> list[dict]:
    merged: list[dict] = []
    for iid, sub in df.sort_values(["IID", "START", "END"]).groupby("IID", sort=False):
        current = None
        for row in sub.to_dict("records"):
            seg = {"IID": str(iid), "START": int(row["START"]), "END": int(row["END"]), "NSNP": int(row["NSNP"])}
            seg["LA"] = msp.local_proportions(seg["IID"], seg["START"], seg["END"])
            if current is None:
                current = seg
                continue
            physical_gap = seg["START"] - current["END"] - 1
            if physical_gap <= gap and tv(current["LA"], seg["LA"]) <= max_tv:
                current["END"] = max(current["END"], seg["END"])
                current["NSNP"] += seg["NSNP"]
                current["LA"] = msp.local_proportions(current["IID"], current["START"], current["END"])
            else:
                merged.append(current)
                current = seg
        if current is not None:
            merged.append(current)
    return merged


def complete_linkage(segments: list[dict], threshold: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    cluster_max_end: list[int] = []
    active: list[int] = []
    for seg in sorted(segments, key=lambda x: (x["START"], x["END"], x["IID"])):
        # A cluster whose rightmost endpoint is left of this segment can never
        # overlap any later segment, so retire it from candidate comparisons.
        active = [idx for idx in active if cluster_max_end[idx] >= seg["START"]]
        eligible = []
        for idx in active:
            cluster = clusters[idx]
            if any(m["IID"] == seg["IID"] for m in cluster):
                continue
            scores = [reciprocal(seg, member) for member in cluster]
            if scores and min(scores) >= threshold:
                eligible.append((min(scores), idx))
        if eligible:
            _, best = max(eligible)
            clusters[best].append(seg)
            cluster_max_end[best] = max(cluster_max_end[best], seg["END"])
        else:
            clusters.append([seg])
            cluster_max_end.append(seg["END"])
            active.append(len(clusters) - 1)
    return clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--ld", required=True)
    ap.add_argument("--chr", type=int, required=True)
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--msp", type=Path, required=True)
    ap.add_argument("--fam", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--length-floor", type=int, default=250_000)
    ap.add_argument("--min-carrier-fraction", type=float, default=0.0025)
    ap.add_argument("--min-carriers", type=int, default=3)
    ap.add_argument("--merge-gap", type=int, default=250_000)
    ap.add_argument("--merge-tv", type=float, default=0.25)
    ap.add_argument("--reciprocal-overlap", type=float, default=0.5)
    args = ap.parse_args()

    calls = pd.read_csv(args.calls, sep="\t", dtype={"IID": str}, keep_default_na=False)
    msp = MSPData.read(args.msp)
    la_cols = [f"LA_{safe_ancestry_label(x)}" for x in msp.ancestry_names]
    n_samples = sum(1 for line in args.fam.open() if line.strip())
    n_min = max(args.min_carriers, math.ceil(args.min_carrier_fraction * n_samples))
    calls = calls[(calls["CHR"].astype(int) == args.chr) & (calls["LENGTH_BP"].astype(int) >= args.length_floor)].copy()

    region_rows, member_rows = [], []
    if not calls.empty:
        segments = merge_fragments(calls, msp, args.merge_gap, args.merge_tv)
        clusters = [c for c in complete_linkage(segments, args.reciprocal_overlap) if len(c) >= n_min]
        kept_idx = 0
        for cluster in clusters:
            starts = np.array([x["START"] for x in cluster], dtype=int)
            ends = np.array([x["END"] for x in cluster], dtype=int)
            cons_start, cons_end = int(np.median(starts)), int(np.median(ends))
            if cons_end < cons_start:
                continue
            kept_idx += 1
            rid = f"{args.tool}.{args.ld}.chr{args.chr}.R{kept_idx:06d}"
            region_rows.append({
                "REGION_ID": rid, "TOOL": args.tool, "LD": args.ld, "CHR": args.chr,
                "CONS_START": cons_start, "CONS_END": cons_end,
                "UNION_START": int(starts.min()), "UNION_END": int(ends.max()),
                "LENGTH_BP": cons_end - cons_start + 1, "N_CARRIERS": len(cluster),
            })
            for seg in cluster:
                region_la = msp.local_proportions(seg["IID"], cons_start, cons_end)
                row = {
                    "REGION_ID": rid, "TOOL": args.tool, "LD": args.ld, "CHR": args.chr,
                    "IID": seg["IID"], "SEG_START": seg["START"], "SEG_END": seg["END"],
                    "SEG_LENGTH_BP": seg["END"] - seg["START"] + 1, "NSNP": seg["NSNP"],
                }
                row.update(dict(zip(la_cols, region_la)))
                member_rows.append(row)

    regions = pd.DataFrame(region_rows, columns=REGION_BASE)
    members = pd.DataFrame(member_rows, columns=MEMBER_BASE + la_cols)
    args.regions.parent.mkdir(parents=True, exist_ok=True)
    args.members.parent.mkdir(parents=True, exist_ok=True)
    regions.to_csv(args.regions, sep="\t", index=False)
    members.to_csv(args.members, sep="\t", index=False)
    print(f"n_samples={n_samples} n_min={n_min} regions={len(regions)} members={len(members)}")


if __name__ == "__main__":
    main()
