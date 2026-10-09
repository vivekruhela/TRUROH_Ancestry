#!/usr/bin/env python3
"""LOCO ancestry correction, three-component RDI, knee calibration, and truth benchmark."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge


def read_many(pattern: str, chromosomes=range(1, 23)) -> pd.DataFrame:
    frames = []
    for chrom in chromosomes:
        path = Path(pattern.replace("{CHR}", str(chrom)))
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, sep="\t"))
    nonempty = [x for x in frames if not x.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else (frames[0].iloc[0:0].copy() if frames else pd.DataFrame())


def geo_mean(values: list[float]) -> float:
    values = np.asarray(values, float)
    if np.any(~np.isfinite(values)):
        return math.nan
    values = np.clip(values, 0.0, 1.0)
    if np.any(values <= 0):
        return 0.0
    return float(np.exp(np.log(values).mean()))


def harmonic(values: list[float]) -> float:
    values = np.asarray(values, float)
    if np.any(~np.isfinite(values)):
        return math.nan
    if np.any(values <= 0):
        return 0.0
    return float(len(values) / np.sum(1.0 / values))


def feature_matrix(df: pd.DataFrame, q_cols: list[str], la_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    q_by_suffix = {x.removeprefix("Q_"): x for x in q_cols}
    la_by_suffix = {x.removeprefix("LA_"): x for x in la_cols}
    common = [x for x in q_by_suffix if x in la_by_suffix]
    if len(common) < 2:
        raise ValueError(f"Need at least two matched global/local ancestry components; Q={q_cols}, LA={la_cols}")
    # Omit the final component to remove compositional collinearity.
    used = common[:-1]
    arrays, names = [], []
    for suffix in used:
        q = pd.to_numeric(df[q_by_suffix[suffix]], errors="coerce").to_numpy(float)
        la = pd.to_numeric(df[la_by_suffix[suffix]], errors="coerce").to_numpy(float)
        arrays.extend([q, la - q]); names.extend([f"Q_{suffix}", f"EXCESS_{suffix}"])
    q_all = np.column_stack([pd.to_numeric(df[q_by_suffix[x]], errors="coerce") for x in common])
    la_all = np.column_stack([pd.to_numeric(df[la_by_suffix[x]], errors="coerce") for x in common])
    divergence = 0.5 * np.abs(la_all - q_all).sum(axis=1)
    safe = np.clip(la_all, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1) / math.log(len(common))
    arrays.extend([divergence, entropy]); names.extend(["LA_GLOBAL_TV", "LA_ENTROPY"])
    return np.column_stack(arrays), names


def loco_adjust(df: pd.DataFrame, y_col: str, X: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    chrom = pd.to_numeric(df["CHR"], errors="coerce").to_numpy(int)
    adjusted = y.copy()
    fallback = np.ones(len(df), dtype=int)
    center = float(np.nanmedian(y)) if np.any(np.isfinite(y)) else math.nan
    for c in np.unique(chrom):
        test = chrom == c
        train = ~test & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        predmask = test & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if train.sum() < max(20, X.shape[1] + 5) or predmask.sum() == 0:
            continue
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0)
        sd[sd < 1e-12] = 1.0
        model = Ridge(alpha=ridge_lambda, fit_intercept=True)
        model.fit((X[train] - mu) / sd, y[train])
        pred = model.predict((X[predmask] - mu) / sd)
        adjusted[predmask] = center + y[predmask] - pred
        fallback[predmask] = 0
    return adjusted, fallback


def knee_base(values: np.ndarray) -> float:
    values = np.sort(values[np.isfinite(values)])
    if len(values) == 0:
        return math.nan
    if len(values) < 3 or values[-1] <= values[0]:
        return float(values.min())
    x = np.arange(len(values), dtype=float) / (len(values) - 1)
    y = (values - values[0]) / (values[-1] - values[0])
    return float(values[int(np.argmax(np.abs(y - x) / math.sqrt(2.0)))])


def full_threshold(df: pd.DataFrame, bad_quantile: float, min_bad: int, min_bad_chr: int) -> tuple[float, float, float, str]:
    eligible = df.get("ADAPTIVE_FILTER_PASS", pd.Series(1, index=df.index)).astype(int) == 1
    assess = df[(df["TECHNICALLY_ASSESSABLE"] == 1) & eligible]
    base = knee_base(assess["RDI_A"].to_numpy(float))
    bad = assess[assess["SOFT_PASS"] == 0]
    status = "base_only_insufficient_bad_reference"
    relaxed = base
    alpha = 1.0
    if np.isfinite(base) and len(bad) >= min_bad and bad["CHR"].nunique() >= min_bad_chr:
        bq = float(np.nanquantile(bad["RDI_A"], bad_quantile))
        alpha = float(np.clip(bq / (base + 1e-12), 0.0, 1.0))
        relaxed = alpha * base
        status = "empirically_relaxed"
    return base, relaxed, alpha, status


def bootstrap_thresholds(df: pd.DataFrame, n: int, seed: int, bad_quantile: float, min_bad: int, min_bad_chr: int) -> np.ndarray:
    chroms = np.array(sorted(df["CHR"].dropna().astype(int).unique()))
    if len(chroms) == 0:
        return np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    out = np.full(n, np.nan)
    by_chr = {c: df[df["CHR"].astype(int) == c] for c in chroms}
    for b in range(n):
        chosen = rng.choice(chroms, size=len(chroms), replace=True)
        blocks = []
        for replicate_idx, c in enumerate(chosen):
            block = by_chr[int(c)].copy()
            block["CHR"] = replicate_idx + 1
            blocks.append(block)
        boot = pd.concat(blocks, ignore_index=True)
        _, relaxed, _, _ = full_threshold(boot, bad_quantile, min_bad, min_bad_chr)
        out[b] = relaxed
    return out


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else math.nan


def benchmark(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = scored[(scored["TECHNICALLY_ASSESSABLE"] == 1) & scored["TRUE_ROH"].notna()].copy()
    pred = eval_df["HIGH_CONFIDENCE"].astype(int)
    truth = eval_df["TRUE_ROH"].astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum()); tn = int(((pred == 0) & (truth == 0)).sum())
    sensitivity = safe_div(tp, tp + fn); specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    balanced = float(np.nanmean([sensitivity, specificity])) if np.any(np.isfinite([sensitivity, specificity])) else math.nan
    counts = pd.DataFrame([{"TP": tp, "FP": fp, "FN": fn, "TN": tn, "N_EVALUATED": len(eval_df)}])
    metrics = pd.DataFrame([{
        "SENSITIVITY": sensitivity, "SPECIFICITY": specificity, "PRECISION": precision,
        "F1": f1, "BALANCED_ACCURACY": balanced,
    }])
    return counts, metrics


def write_empty_outputs(args, regions: pd.DataFrame, indiv: pd.DataFrame, status: str,
                        lmin: float, n_min: int) -> None:
    """Write schema-valid, aggregatable outputs when no regions survive."""
    scored = regions.iloc[0:0].copy()
    for col in [
        "QH_A", "QR_A", "MEDIAN_HVR_RAW", "MEAN_HRS_RAW", "CARRIER_SUPPORT", "RDI_A", "HVR_VALID_FRAC_A",
        "HVR_PASS_FRAC_A", "HRS_PASS_FRAC_A", "HRS_ZERO_FRAC_A",
        "MEDIAN_Z_H_A", "MEDIAN_EMP_P", "HVR_META_Z_A", "HVR_META_P_A",
        "N_SEEDS", "TECHNICALLY_ASSESSABLE", "SOFT_PASS", "N_TMRCA_MEMBERS",
        "MEDIAN_TMRCA", "MEAN_TMRCA", "FRAC_BP_TMRCA_LT_TRUE", "TMRCA_CLASS",
        "TRUE_ROH", "KNEE_BASE", "KNEE_RELAXED", "HIGH_CONFIDENCE", "STATUS",
        "ADAPTIVE_FILTER_PASS",
    ]:
        if col not in scored:
            scored[col] = pd.Series(dtype="object")
    for col in ["Z_S", "Z_H_A", "Z_S_A", "HRS_A", "HVR_PASS_A", "HRS_PASS_A"]:
        if col not in indiv:
            indiv[col] = pd.Series(dtype="float64")

    counts = pd.DataFrame([{
        "TOOL": args.tool, "LD": args.ld, "TP": 0, "FP": 0, "FN": 0,
        "TN": 0, "N_EVALUATED": 0,
    }])
    metrics = pd.DataFrame([{
        "TOOL": args.tool, "LD": args.ld, "SENSITIVITY": math.nan,
        "SPECIFICITY": math.nan, "PRECISION": math.nan, "F1": math.nan,
        "BALANCED_ACCURACY": math.nan, "N_CANDIDATE": 0,
        "N_ASSESSABLE": 0, "N_ADAPTIVE_ELIGIBLE": 0, "N_HIGH_CONFIDENCE": 0,
        "MEAN_HC_LENGTH_BP": math.nan, "MEAN_HC_RDI": math.nan,
    }])
    threshold = pd.DataFrame([{
        "TOOL": args.tool, "LD": args.ld, "ADAPTIVE_MIN_LENGTH_BP": lmin,
        "MIN_CARRIERS": n_min, "KNEE_BASE": math.nan,
        "RELAXATION_ALPHA": math.nan, "KNEE_RELAXED": math.nan,
        "CALIBRATION_STATUS": status, "BOOTSTRAP_VALID_FRACTION": 0.0,
        "BOOTSTRAP_RELIABLE": 0, "BOOTSTRAP_MEDIAN": math.nan,
        "BOOTSTRAP_LOW_2_5": math.nan, "BOOTSTRAP_HIGH_97_5": math.nan,
        "LOCO_FEATURES": "",
    }])
    scored.to_csv(args.outdir / "regions_scored.tsv", sep="\t", index=False)
    indiv.iloc[0:0].to_csv(args.outdir / "individual_scored.tsv", sep="\t", index=False)
    counts.to_csv(args.outdir / "benchmark_counts.tsv", sep="\t", index=False)
    metrics.to_csv(args.outdir / "benchmark_metrics.tsv", sep="\t", index=False)
    threshold.to_csv(args.outdir / "thresholds.tsv", sep="\t", index=False)
    pd.DataFrame({
        "BOOTSTRAP": np.arange(1, args.bootstraps + 1),
        "KNEE_RELAXED": np.full(args.bootstraps, np.nan),
    }).to_csv(args.outdir / "knee_bootstrap.tsv", sep="\t", index=False)
    print(f"{status}; wrote schema-valid empty benchmark outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ld", required=True)
    ap.add_argument("--region-pattern", required=True); ap.add_argument("--individual-pattern", required=True)
    ap.add_argument("--regional-raw-pattern", required=True); ap.add_argument("--truth-pattern", required=True)
    ap.add_argument("--global-ancestry", type=Path, required=True); ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--length-floor", type=int, default=250000); ap.add_argument("--length-ceiling", type=int, default=500000)
    ap.add_argument("--length-quantile", type=float, default=0.10); ap.add_argument("--min-carrier-fraction", type=float, default=0.0025)
    ap.add_argument("--min-carriers", type=int, default=3); ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--hvr-valid-min", type=float, default=0.05); ap.add_argument("--hvr-pass-min", type=float, default=0.05)
    ap.add_argument("--hvr-indiv-z", type=float, default=-3.0); ap.add_argument("--hvr-indiv-p", type=float, default=0.99)
    ap.add_argument("--hvr-region-z", type=float, default=-2.5); ap.add_argument("--hvr-region-p", type=float, default=0.95)
    ap.add_argument("--hvr-meta-p", type=float, default=0.05); ap.add_argument("--hrs-min", type=float, default=0.01)
    ap.add_argument("--hrs-min-seeds", type=int, default=3); ap.add_argument("--hrs-zero-max", type=float, default=0.95)
    ap.add_argument("--bad-quantile", type=float, default=0.95); ap.add_argument("--min-bad", type=int, default=30)
    ap.add_argument("--min-bad-chr", type=int, default=5); ap.add_argument("--bootstraps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=108); ap.add_argument("--min-valid-bootstrap", type=float, default=0.80)
    args = ap.parse_args()

    regions = read_many(args.region_pattern); indiv = read_many(args.individual_pattern)
    regional_raw = read_many(args.regional_raw_pattern); truth = read_many(args.truth_pattern)
    ancestry = pd.read_csv(args.global_ancestry, sep="\t", dtype={"IID": str})
    args.outdir.mkdir(parents=True, exist_ok=True)
    n_min = max(args.min_carriers, math.ceil(args.min_carrier_fraction * len(ancestry)))

    if regions.empty:
        write_empty_outputs(args, regions, indiv, "no_candidate_regions",
                            float(args.length_floor), n_min)
        return

    lq = float(np.nanquantile(regions["LENGTH_BP"], args.length_quantile))
    lmin = max(args.length_floor, min(args.length_ceiling, lq))
    # Score every permissive-floor candidate so length and carrier thresholds
    # can be tuned later. The adaptive filter is recorded, not destructively
    # applied before the held-out grid search.
    regions = regions[regions["N_CARRIERS"] >= n_min].copy()
    keep = set(regions["REGION_ID"].astype(str))
    indiv = indiv[indiv["REGION_ID"].astype(str).isin(keep)].copy()
    regional_raw = regional_raw[regional_raw["REGION_ID"].astype(str).isin(keep)].copy()
    truth = truth[truth["REGION_ID"].astype(str).isin(keep)].copy()
    if regions.empty:
        write_empty_outputs(args, regions, indiv, "no_regions_after_carrier_filter",
                            float(lmin), n_min)
        return

    ancestry["IID"] = ancestry["IID"].astype(str); indiv["IID"] = indiv["IID"].astype(str)
    indiv = indiv.merge(ancestry, on="IID", how="left", validate="many_to_one")
    q_cols = [c for c in ancestry.columns if c.startswith("Q_")]
    la_cols = [c for c in indiv.columns if c.startswith("LA_")]
    X, feature_names = feature_matrix(indiv, q_cols, la_cols)
    indiv["Z_S"] = norm.ppf(1.0 - np.clip(pd.to_numeric(indiv["EMP_P"], errors="coerce"), 1e-30, 1 - 1e-16))
    for source, target in [("Z_H", "Z_H_A"), ("Z_S", "Z_S_A"), ("HRS", "HRS_A")]:
        adjusted, fallback = loco_adjust(indiv, source, X, args.ridge_lambda)
        if source == "HRS":
            adjusted = np.clip(adjusted, 0.0, 1.0)
        indiv[target] = adjusted
        indiv[f"{target}_LOCO_FALLBACK"] = fallback
    indiv["HVR_PASS_A"] = ((indiv["HVR_ASSESSABLE"] == 1) & (indiv["Z_H_A"] >= args.hvr_indiv_z) & (indiv["EMP_P"] <= args.hvr_indiv_p)).astype(int)
    indiv["HRS_PASS_A"] = (indiv["HRS_A"] >= args.hrs_min).astype(int)

    rows = []
    for region in regions.itertuples(index=False):
        rid = str(region.REGION_ID)
        sub = indiv[indiv["REGION_ID"].astype(str) == rid]
        valid = sub[sub["HVR_ASSESSABLE"] == 1]
        ncar = max(len(sub), 1)
        valid_frac = len(valid) / ncar; hpass = sub["HVR_PASS_A"].sum() / ncar; rpass = sub["HRS_PASS_A"].sum() / ncar
        zero_frac = float((sub["HRS_A"] <= 0).mean()) if len(sub) else math.nan
        if len(valid):
            med_zh = float(np.nanmedian(valid["Z_H_A"])); med_p = float(np.nanmedian(valid["EMP_P"]))
            mean_one_p = float(np.nanmean(1.0 - valid["EMP_P"])); mean_match = float(np.nanmean(valid["MATCH_CONF"]))
            qh = geo_mean([float(norm.cdf(med_zh)), mean_one_p, mean_match])
            weights = pd.to_numeric(valid["N_R"], errors="coerce").to_numpy(float)
            zs = pd.to_numeric(valid["Z_S_A"], errors="coerce").to_numpy(float)
            mask = np.isfinite(weights) & np.isfinite(zs) & (weights > 0)
            meta_z = float(np.sum(weights[mask] * zs[mask]) / math.sqrt(np.sum(weights[mask] ** 2))) if np.any(mask) else math.nan
            meta_p = float(norm.sf(meta_z)) if np.isfinite(meta_z) else math.nan
        else:
            med_zh = med_p = qh = meta_z = meta_p = math.nan
        qr = float(np.nanmean(sub["HRS_A"])) if len(sub) and np.any(np.isfinite(sub["HRS_A"])) else math.nan
        median_hvr_raw = float(np.nanmedian(sub["HVR"])) if len(sub) and np.any(np.isfinite(sub["HVR"])) else math.nan
        mean_hrs_raw = float(np.nanmean(sub["HRS"])) if len(sub) and np.any(np.isfinite(sub["HRS"])) else math.nan
        support = math.sqrt(max(0.0, hpass) * max(0.0, rpass))
        rdi = harmonic([qh, qr, support])
        n_seeds = int(np.nanmax(sub["N_SEEDS"])) if len(sub) else 0
        assessable = int(len(sub) >= n_min and valid_frac >= args.hvr_valid_min and n_seeds >= args.hrs_min_seeds and np.all(np.isfinite([qh, qr, support])))
        soft = int(
            valid_frac >= args.hvr_valid_min and hpass >= args.hvr_pass_min and
            np.isfinite(med_zh) and med_zh >= args.hvr_region_z and
            np.isfinite(med_p) and med_p <= args.hvr_region_p and
            np.isfinite(meta_p) and meta_p <= args.hvr_meta_p and
            np.isfinite(zero_frac) and zero_frac <= args.hrs_zero_max
        )
        rows.append({
            "REGION_ID": rid, "QH_A": qh, "QR_A": qr, "MEDIAN_HVR_RAW": median_hvr_raw,
            "MEAN_HRS_RAW": mean_hrs_raw, "CARRIER_SUPPORT": support, "RDI_A": rdi,
            "HVR_VALID_FRAC_A": valid_frac, "HVR_PASS_FRAC_A": hpass, "HRS_PASS_FRAC_A": rpass,
            "HRS_ZERO_FRAC_A": zero_frac, "MEDIAN_Z_H_A": med_zh, "MEDIAN_EMP_P": med_p,
            "HVR_META_Z_A": meta_z, "HVR_META_P_A": meta_p, "N_SEEDS": n_seeds,
            "TECHNICALLY_ASSESSABLE": assessable, "SOFT_PASS": soft,
        })
    scores = pd.DataFrame(rows)
    scored = regions.merge(regional_raw.drop(columns=[c for c in ["N_CARRIERS", "N_SEEDS"] if c in regional_raw], errors="ignore"), on=["REGION_ID", "TOOL", "LD", "CHR"], how="left")
    scored = scored.merge(scores, on="REGION_ID", how="left").merge(truth, on=["REGION_ID", "TOOL", "LD", "CHR"], how="left")
    scored["ADAPTIVE_FILTER_PASS"] = ((scored["LENGTH_BP"] >= lmin) & (scored["N_CARRIERS"] >= n_min)).astype(int)

    base, relaxed, alpha, calibration_status = full_threshold(scored, args.bad_quantile, args.min_bad, args.min_bad_chr)
    boots = bootstrap_thresholds(scored, args.bootstraps, args.bootstrap_seed, args.bad_quantile, args.min_bad, args.min_bad_chr)
    valid_boot = boots[np.isfinite(boots)]
    valid_fraction = len(valid_boot) / max(args.bootstraps, 1)
    boot_reliable = int(valid_fraction >= args.min_valid_bootstrap)
    scored["KNEE_BASE"] = base; scored["KNEE_RELAXED"] = relaxed
    scored["HIGH_CONFIDENCE"] = ((scored["TECHNICALLY_ASSESSABLE"] == 1) &
                                  (scored["ADAPTIVE_FILTER_PASS"] == 1) & np.isfinite(relaxed) &
                                  (scored["RDI_A"] >= relaxed)).astype(int)
    scored["STATUS"] = np.select(
        [scored["TECHNICALLY_ASSESSABLE"] == 0,
         scored["ADAPTIVE_FILTER_PASS"] == 0,
         (scored["HIGH_CONFIDENCE"] == 1) & (scored["SOFT_PASS"] == 1),
         (scored["HIGH_CONFIDENCE"] == 1) & (scored["SOFT_PASS"] == 0)],
        ["technical_unassessable", "below_adaptive_length_or_carriers",
         "high_confidence_core", "high_confidence_soft_gate_rescue"],
        default="below_empirical_knee",
    )
    counts, metrics = benchmark(scored)
    for frame in [counts, metrics]:
        frame.insert(0, "LD", args.ld); frame.insert(0, "TOOL", args.tool)
    metrics["N_CANDIDATE"] = len(scored); metrics["N_ASSESSABLE"] = int(scored["TECHNICALLY_ASSESSABLE"].sum())
    metrics["N_ADAPTIVE_ELIGIBLE"] = int(((scored["TECHNICALLY_ASSESSABLE"] == 1) & (scored["ADAPTIVE_FILTER_PASS"] == 1)).sum())
    metrics["N_HIGH_CONFIDENCE"] = int(scored["HIGH_CONFIDENCE"].sum())
    metrics["MEAN_HC_LENGTH_BP"] = float(scored.loc[scored["HIGH_CONFIDENCE"] == 1, "LENGTH_BP"].mean())
    metrics["MEAN_HC_RDI"] = float(scored.loc[scored["HIGH_CONFIDENCE"] == 1, "RDI_A"].mean())
    threshold = pd.DataFrame([{
        "TOOL": args.tool, "LD": args.ld, "ADAPTIVE_MIN_LENGTH_BP": lmin, "MIN_CARRIERS": n_min,
        "KNEE_BASE": base, "RELAXATION_ALPHA": alpha, "KNEE_RELAXED": relaxed,
        "CALIBRATION_STATUS": calibration_status, "BOOTSTRAP_VALID_FRACTION": valid_fraction,
        "BOOTSTRAP_RELIABLE": boot_reliable,
        "BOOTSTRAP_MEDIAN": float(np.nanmedian(valid_boot)) if len(valid_boot) else math.nan,
        "BOOTSTRAP_LOW_2_5": float(np.nanquantile(valid_boot, 0.025)) if len(valid_boot) else math.nan,
        "BOOTSTRAP_HIGH_97_5": float(np.nanquantile(valid_boot, 0.975)) if len(valid_boot) else math.nan,
        "LOCO_FEATURES": ";".join(feature_names),
    }])
    scored.to_csv(args.outdir / "regions_scored.tsv", sep="\t", index=False)
    indiv.to_csv(args.outdir / "individual_scored.tsv", sep="\t", index=False)
    counts.to_csv(args.outdir / "benchmark_counts.tsv", sep="\t", index=False)
    metrics.to_csv(args.outdir / "benchmark_metrics.tsv", sep="\t", index=False)
    threshold.to_csv(args.outdir / "thresholds.tsv", sep="\t", index=False)
    pd.DataFrame({"BOOTSTRAP": np.arange(1, args.bootstraps + 1), "KNEE_RELAXED": boots}).to_csv(args.outdir / "knee_bootstrap.tsv", sep="\t", index=False)
    print(f"tool={args.tool} ld={args.ld} candidates={len(scored)} HC={int(scored['HIGH_CONFIDENCE'].sum())} knee={relaxed}")


if __name__ == "__main__":
    main()
