"""Regional v3.8.0 scoring calculations copied verbatim; outcome independent."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scoring_math import feature_matrix,loco_adjust,geo_mean,harmonic

def score_regions(regions, indiv, ancestry, args, n_min):
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
    return pd.DataFrame(rows), indiv, feature_names
