#!/usr/bin/env python3
"""Compute exact matched-background HVR and seed-span HRS for candidate ROH.

The PLINK BED is streamed by variant chunks. Cumulative heterozygous and
nonmissing counts are captured only at requested interval boundaries, so every
candidate/background count is exact without materializing a genome-wide
sample-by-variant matrix.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


INDIV_BASE = [
    "REGION_ID", "TOOL", "LD", "CHR", "IID", "N_R", "HET_R", "HVR",
    "EMP_P", "Z_H", "ESS", "MATCH_CONF", "HVR_ASSESSABLE", "HVR_PASS",
    "HRS", "HRS_PASS", "N_SEEDS",
]
REG_BASE = [
    "REGION_ID", "TOOL", "LD", "CHR", "N_CARRIERS", "N_SEEDS",
    "HVR_VALID_FRAC", "HVR_PASS_FRAC", "HRS_PASS_FRAC", "HRS_ZERO_FRAC",
]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return math.nan
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum())])


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def sample_backgrounds(start: int, end: int, chrom_end: int, buffer_bp: int, n: int, seed: int) -> list[tuple[int, int]]:
    length = end - start + 1
    max_start = chrom_end - length + 1
    left_hi = min(max_start, start - buffer_bp - length)
    right_lo = max(1, end + buffer_bp + 1)
    ranges = []
    if left_hi >= 1:
        ranges.append((1, left_hi))
    if max_start >= right_lo:
        ranges.append((right_lo, max_start))
    if not ranges or n <= 0:
        return []
    widths = np.array([hi - lo + 1 for lo, hi in ranges], dtype=float)
    probs = widths / widths.sum()
    rng = np.random.default_rng(seed)
    which = rng.choice(len(ranges), size=n, replace=True, p=probs)
    result = []
    for idx in which:
        lo, hi = ranges[int(idx)]
        s = int(rng.integers(lo, hi + 1))
        result.append((s, s + length - 1))
    return result


def read_fam(prefix: Path) -> list[str]:
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None, dtype=str, keep_default_na=False)
    iids = fam.iloc[:, 1].astype(str).tolist()
    if len(iids) != len(set(iids)):
        raise ValueError("BED/FAM contains duplicate IID values; unique IIDs are required")
    return iids


def read_bim(prefix: Path) -> pd.DataFrame:
    bim = pd.read_csv(str(prefix) + ".bim", sep=r"\s+", header=None, dtype=str, keep_default_na=False,
                      names=["CHR", "ID", "CM", "POS", "A1", "A2"])
    bim["POS"] = bim["POS"].astype(int)
    return bim


def seed_positions(original_bim: Path, pruned_list: Path, eligible_list: Path) -> tuple[np.ndarray, np.ndarray]:
    bim = pd.read_csv(original_bim, sep=r"\s+", header=None, dtype=str, keep_default_na=False,
                      names=["CHR", "ID", "CM", "POS", "A1", "A2"])
    pos = dict(zip(bim["ID"].astype(str), bim["POS"].astype(int)))
    def load(path: Path) -> np.ndarray:
        if not path.exists() or path.stat().st_size == 0:
            return np.array([], dtype=int)
        ids = [x.strip().split()[0] for x in path.read_text().splitlines() if x.strip()]
        return np.array(sorted({pos[x] for x in ids if x in pos}), dtype=int)
    return load(pruned_list), load(eligible_list)


def build_interval_counts(
    prefix: Path,
    fam_iids: list[str],
    positions: np.ndarray,
    members_by_region: dict[str, list[str]],
    intervals_by_region: dict[str, list[tuple[int, int]]],
    chunk_variants: int,
    threads: int,
) -> dict[str, dict[str, np.ndarray]]:
    fam_index = {iid: idx for idx, iid in enumerate(fam_iids)}
    carrier_union = sorted({iid for ids in members_by_region.values() for iid in ids})
    missing = sorted(set(carrier_union) - set(fam_index))
    if missing:
        raise ValueError(f"Carrier IIDs absent from FAM (first 10): {missing[:10]}")
    bed_sample_idx = np.array([fam_index[x] for x in carrier_union], dtype=int)
    union_idx = {iid: idx for idx, iid in enumerate(carrier_union)}
    local_idx = {rid: np.array([union_idx[x] for x in ids], dtype=int) for rid, ids in members_by_region.items()}

    counts: dict[str, dict[str, np.ndarray]] = {}
    events: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for rid, intervals in intervals_by_region.items():
        c = len(members_by_region[rid])
        counts[rid] = {
            "nm": np.zeros((len(intervals), c), dtype=np.int32),
            "het": np.zeros((len(intervals), c), dtype=np.int32),
        }
        for j, (start, end) in enumerate(intervals):
            left = int(np.searchsorted(positions, start, side="left"))
            right = int(np.searchsorted(positions, end, side="right"))
            events[left].append((rid, j, -1))
            events[right].append((rid, j, +1))

    boundaries = sorted(events)
    boundary_ptr = 0
    cumulative_nm = np.zeros(len(carrier_union), dtype=np.int64)
    cumulative_het = np.zeros(len(carrier_union), dtype=np.int64)
    n_variants = len(positions)

    def apply_boundary(boundary: int, pref_nm: np.ndarray, pref_het: np.ndarray) -> None:
        for rid, interval_idx, sign in events[boundary]:
            idx = local_idx[rid]
            counts[rid]["nm"][interval_idx, :] += sign * pref_nm[idx].astype(np.int32)
            counts[rid]["het"][interval_idx, :] += sign * pref_het[idx].astype(np.int32)

    bed_path = Path(str(prefix) + ".bed")
    with bed_path.open("rb") as handle:
        magic = handle.read(3)
    if magic != bytes([0x6C, 0x1B, 0x01]):
        raise ValueError(f"Expected a PLINK variant-major BED file: {bed_path}")
    bytes_per_variant = (len(fam_iids) + 3) // 4
    expected_size = 3 + n_variants * bytes_per_variant
    if bed_path.stat().st_size != expected_size:
        raise ValueError(f"BED size mismatch: expected {expected_size}, observed {bed_path.stat().st_size}")
    bed_bytes = np.memmap(bed_path, dtype=np.uint8, mode="r", offset=3,
                          shape=(n_variants, bytes_per_variant), order="C")
    byte_index = bed_sample_idx // 4
    bit_shift = (2 * (bed_sample_idx % 4)).astype(np.uint8)
    for vstart in range(0, n_variants, chunk_variants):
        vend = min(n_variants, vstart + chunk_variants)
        packed = np.asarray(bed_bytes[vstart:vend, :][:, byte_index], dtype=np.uint8).T
        code = (packed >> bit_shift[:, None]) & np.uint8(3)
        # PLINK BED two-bit codes: 01=missing, 10=heterozygous.
        nm_cs = np.cumsum(code != 1, axis=1, dtype=np.int64)
        het_cs = np.cumsum(code == 2, axis=1, dtype=np.int64)
        while boundary_ptr < len(boundaries) and boundaries[boundary_ptr] <= vend:
            b = boundaries[boundary_ptr]
            if b < vstart:
                raise RuntimeError("Internal boundary ordering failure")
            if b == vstart:
                pref_nm, pref_het = cumulative_nm, cumulative_het
            else:
                offset = b - vstart - 1
                pref_nm = cumulative_nm + nm_cs[:, offset]
                pref_het = cumulative_het + het_cs[:, offset]
            apply_boundary(b, pref_nm, pref_het)
            boundary_ptr += 1
        cumulative_nm += nm_cs[:, -1]
        cumulative_het += het_cs[:, -1]

    while boundary_ptr < len(boundaries):
        b = boundaries[boundary_ptr]
        if b != n_variants:
            raise RuntimeError(f"Unresolved BED boundary {b}/{n_variants}")
        apply_boundary(b, cumulative_nm, cumulative_het)
        boundary_ptr += 1
    return counts


def hrs_for_region(member_sub: pd.DataFrame, seeds: np.ndarray) -> dict[str, float]:
    result = {}
    if len(seeds) == 0:
        return {str(iid): math.nan for iid in member_sub["IID"]}
    full_span = int(seeds[-1] - seeds[0])
    for row in member_sub.itertuples(index=False):
        hits = seeds[(seeds >= int(row.SEG_START)) & (seeds <= int(row.SEG_END))]
        recall = len(hits) / len(seeds)
        if len(hits) == 0:
            span = 0.0
        elif full_span <= 0:
            span = recall
        else:
            span = min(1.0, max(0.0, (int(hits[-1]) - int(hits[0])) / full_span))
        result[str(row.IID)] = min(1.0, max(0.0, 0.75 * recall + 0.25 * span))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ld", required=True)
    ap.add_argument("--chr", type=int, required=True)
    ap.add_argument("--regions", type=Path, required=True); ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--bed-prefix", type=Path, required=True)
    ap.add_argument("--original-bim", type=Path, required=True)
    ap.add_argument("--seed-pruned", type=Path, required=True); ap.add_argument("--seed-eligible", type=Path, required=True)
    ap.add_argument("--individual-output", type=Path, required=True); ap.add_argument("--regional-output", type=Path, required=True)
    ap.add_argument("--backgrounds", type=int, default=500); ap.add_argument("--proposal-multiplier", type=int, default=4)
    ap.add_argument("--buffer-bp", type=int, default=10_000_000)
    ap.add_argument("--site-ratio-min", type=float, default=0.25); ap.add_argument("--site-ratio-max", type=float, default=4.0)
    ap.add_argument("--min-sites", type=int, default=10); ap.add_argument("--min-ess", type=float, default=40)
    ap.add_argument("--individual-min-z", type=float, default=-3.0); ap.add_argument("--individual-max-p", type=float, default=0.99)
    ap.add_argument("--hrs-min", type=float, default=0.01); ap.add_argument("--chunk-variants", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=108); ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    regions = pd.read_csv(args.regions, sep="\t")
    members = pd.read_csv(args.members, sep="\t", dtype={"IID": str}, keep_default_na=False)
    la_cols = [c for c in members.columns if c.startswith("LA_")]
    if regions.empty:
        args.individual_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=INDIV_BASE + la_cols).to_csv(args.individual_output, sep="\t", index=False)
        pd.DataFrame(columns=REG_BASE).to_csv(args.regional_output, sep="\t", index=False)
        print("regions=0; wrote schema-valid empty metric files")
        return

    bim = read_bim(args.bed_prefix)
    positions = bim["POS"].to_numpy(dtype=int)
    chrom_end = int(positions.max())
    fam_iids = read_fam(args.bed_prefix)
    pruned_seeds, eligible_seeds = seed_positions(args.original_bim, args.seed_pruned, args.seed_eligible)

    members_by_region: dict[str, list[str]] = {}
    intervals_by_region: dict[str, list[tuple[int, int]]] = {}
    for row in regions.itertuples(index=False):
        rid = str(row.REGION_ID)
        sub = members[members["REGION_ID"].astype(str) == rid]
        members_by_region[rid] = sub["IID"].astype(str).tolist()
        proposals = sample_backgrounds(int(row.CONS_START), int(row.CONS_END), chrom_end, args.buffer_bp,
                                       args.backgrounds * args.proposal_multiplier,
                                       stable_seed(rid, args.seed))
        intervals_by_region[rid] = [(int(row.CONS_START), int(row.CONS_END))] + proposals

    counts = build_interval_counts(args.bed_prefix, fam_iids, positions, members_by_region,
                                   intervals_by_region, args.chunk_variants, args.threads)

    individual_rows, regional_rows = [], []
    for row in regions.itertuples(index=False):
        rid = str(row.REGION_ID)
        sub = members[members["REGION_ID"].astype(str) == rid].reset_index(drop=True)
        seed_mask = (pruned_seeds >= int(row.UNION_START)) & (pruned_seeds <= int(row.UNION_END))
        seeds = pruned_seeds[seed_mask]
        if len(seeds) == 0:
            mask = (eligible_seeds >= int(row.UNION_START)) & (eligible_seeds <= int(row.UNION_END))
            seeds = eligible_seeds[mask]
        hrs = hrs_for_region(sub, seeds)
        nm, het = counts[rid]["nm"], counts[rid]["het"]
        valid_n, pass_h, pass_r, zero_r = 0, 0, 0, 0
        for j, member in sub.iterrows():
            iid = str(member["IID"])
            n_r, h_r = int(nm[0, j]), int(het[0, j])
            hvr = h_r / n_r if n_r > 0 else math.nan
            bg_n, bg_h = nm[1:, j].astype(float), het[1:, j].astype(float)
            ratio = bg_n / n_r if n_r > 0 else np.full(len(bg_n), np.nan)
            valid = (bg_n > 0) & (ratio >= args.site_ratio_min) & (ratio <= args.site_ratio_max)
            bg_n, bg_h, ratio = bg_n[valid], bg_h[valid], ratio[valid]
            bg_hvr = bg_h / bg_n if len(bg_n) else np.array([], dtype=float)
            bg_intervals = np.array(intervals_by_region[rid][1:], dtype=int)[valid] if len(valid) else np.empty((0, 2), int)
            if len(bg_hvr) > args.backgrounds:
                bg_hvr, ratio, bg_intervals = bg_hvr[:args.backgrounds], ratio[:args.backgrounds], bg_intervals[:args.backgrounds]
            if len(bg_hvr):
                dist = np.where(bg_intervals[:, 1] < int(row.CONS_START),
                                int(row.CONS_START) - bg_intervals[:, 1],
                                bg_intervals[:, 0] - int(row.CONS_END)).astype(float)
                positive = dist[dist > 0]
                kappa = float(np.median(positive)) if len(positive) else 1.0
                weights = np.exp(-(dist**2) / (2 * max(kappa, 1.0)**2)) * np.exp(-np.abs(np.log(ratio)))
                ess = float(weights.sum() ** 2 / np.square(weights).sum()) if np.square(weights).sum() > 0 else 0.0
                p = float((1.0 + weights[bg_hvr <= hvr].sum()) / (1.0 + weights.sum())) if np.isfinite(hvr) else math.nan
                center = weighted_median(bg_hvr, weights)
                mad = 1.4826 * weighted_median(np.abs(bg_hvr - center), weights)
                scale = max(mad if np.isfinite(mad) else 0.0, 1e-6)
                z_h = (center - hvr) / scale if np.isfinite(hvr) else math.nan
                dist_norm = dist / max(chrom_end, 1)
                match_conf = min(1.0, ess / args.min_ess) * math.exp(-weighted_median(dist_norm, weights))
            else:
                ess = 0.0; p = z_h = match_conf = math.nan
            assessable = bool(n_r >= args.min_sites and np.isfinite(ess) and ess >= args.min_ess)
            hvr_pass = bool(assessable and z_h >= args.individual_min_z and p <= args.individual_max_p)
            hrs_value = float(hrs.get(iid, math.nan))
            hrs_pass = bool(np.isfinite(hrs_value) and hrs_value >= args.hrs_min)
            valid_n += int(assessable); pass_h += int(hvr_pass); pass_r += int(hrs_pass)
            zero_r += int(np.isfinite(hrs_value) and hrs_value <= 0)
            out = {
                "REGION_ID": rid, "TOOL": args.tool, "LD": args.ld, "CHR": args.chr, "IID": iid,
                "N_R": n_r, "HET_R": h_r, "HVR": hvr, "EMP_P": p, "Z_H": z_h,
                "ESS": ess, "MATCH_CONF": match_conf, "HVR_ASSESSABLE": int(assessable),
                "HVR_PASS": int(hvr_pass), "HRS": hrs_value, "HRS_PASS": int(hrs_pass), "N_SEEDS": len(seeds),
            }
            out.update({c: member[c] for c in la_cols})
            individual_rows.append(out)
        denom = max(len(sub), 1)
        regional_rows.append({
            "REGION_ID": rid, "TOOL": args.tool, "LD": args.ld, "CHR": args.chr,
            "N_CARRIERS": len(sub), "N_SEEDS": len(seeds),
            "HVR_VALID_FRAC": valid_n / denom, "HVR_PASS_FRAC": pass_h / denom,
            "HRS_PASS_FRAC": pass_r / denom, "HRS_ZERO_FRAC": zero_r / denom,
        })

    args.individual_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(individual_rows, columns=INDIV_BASE + la_cols).to_csv(args.individual_output, sep="\t", index=False)
    pd.DataFrame(regional_rows, columns=REG_BASE).to_csv(args.regional_output, sep="\t", index=False)
    print(f"regions={len(regions)} individual_rows={len(individual_rows)}")


if __name__ == "__main__":
    main()
