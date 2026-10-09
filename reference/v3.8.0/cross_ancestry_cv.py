#!/usr/bin/env python3
"""Leakage-safe primary pipeline selection and secondary LD sensitivity.

Primary inference jointly selects one caller and one TRUROH threshold set from
the configured primary LD input (no pruning by default), without using the
held chromosome fold or target ancestry. Pruned inputs are evaluated only in
locked-pipeline and independently retuned secondary sensitivity analyses.
"""

from __future__ import annotations

import argparse
import base64
import html
import itertools
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GRCH38_LENGTHS = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555,
    5: 181538259, 6: 170805979, 7: 159345973, 8: 145138636,
    9: 138394717, 10: 133797422, 11: 135086622, 12: 133275309,
    13: 114364328, 14: 107043718, 15: 101991189, 16: 90338345,
    17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468,
}
PARAM_COLS = ["LENGTH_BP_MIN", "CARRIER_FRACTION_MIN", "HVR_MAX", "HRS_MIN", "RDI_MIN"]
REQUIRED_REGION_COLS = {
    "CHR", "LENGTH_BP", "N_CARRIERS", "MEDIAN_HVR_RAW", "MEAN_HRS_RAW",
    "RDI_A", "TRUE_ROH", "TECHNICALLY_ASSESSABLE",
}
COUNT_NAMES = ["TP", "FP", "FN", "TN"]


def numbers(text: str, cast=float) -> list:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def split_domains(text: str) -> list[str]:
    return [x.strip().upper() for x in str(text).split(",") if x.strip()]


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(num), np.shape(den)), np.nan, dtype=float)
    return np.divide(num, den, out=out, where=np.asarray(den) != 0)


def metric_arrays(counts: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    tp, fp, fn, tn = (counts[..., i] for i in range(4))
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    precision = safe_ratio(tp, tp + fp)
    f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
    balanced = np.where(
        np.isfinite(sensitivity) & np.isfinite(specificity),
        (sensitivity + specificity) / 2,
        np.where(np.isfinite(sensitivity), sensitivity, specificity),
    )
    return {
        "SENSITIVITY": sensitivity, "SPECIFICITY": specificity,
        "PRECISION": precision, "F1": f1, "BALANCED_ACCURACY": balanced,
    }


def scalar_metrics(counts: np.ndarray) -> dict[str, float]:
    values = metric_arrays(np.asarray(counts, dtype=float))
    return {k: float(np.asarray(v)) for k, v in values.items()}


def nanmean_stack(values: list[np.ndarray], n_grid: int) -> np.ndarray:
    if not values:
        return np.full(n_grid, np.nan)
    matrix = np.vstack(values)
    finite = np.isfinite(matrix)
    denom = finite.sum(axis=0)
    total = np.nansum(matrix, axis=0)
    return np.divide(total, denom, out=np.full(n_grid, np.nan), where=denom > 0)


def nanmin_stack(values: list[np.ndarray], n_grid: int) -> np.ndarray:
    if not values:
        return np.full(n_grid, np.nan)
    matrix = np.vstack(values)
    masked = np.where(np.isfinite(matrix), matrix, np.inf)
    result = masked.min(axis=0)
    result[~np.isfinite(result) | (result == np.inf)] = np.nan
    return result


def make_fold_assignment(n_folds: int) -> pd.DataFrame:
    if n_folds < 2 or n_folds > 22:
        raise ValueError("n_folds must be between 2 and 22")
    totals = [0] * n_folds
    members: list[list[int]] = [[] for _ in range(n_folds)]
    for chrom, length in sorted(GRCH38_LENGTHS.items(), key=lambda x: (-x[1], x[0])):
        fold = min(range(n_folds), key=lambda x: (totals[x], x))
        members[fold].append(chrom); totals[fold] += length
    rows = []
    for fold, chroms in enumerate(members, start=1):
        for chrom in sorted(chroms):
            rows.append({"FOLD": fold, "CHR": chrom, "CHR_LENGTH_BP": GRCH38_LENGTHS[chrom],
                         "FOLD_TOTAL_BP": totals[fold - 1]})
    return pd.DataFrame(rows).sort_values(["FOLD", "CHR"]).reset_index(drop=True)


def build_grid(args: argparse.Namespace) -> pd.DataFrame:
    values = itertools.product(
        numbers(args.length_grid, int), numbers(args.carrier_fraction_grid),
        numbers(args.hvr_max_grid), numbers(args.hrs_min_grid), numbers(args.rdi_grid),
    )
    grid = pd.DataFrame(values, columns=PARAM_COLS)
    if grid.empty:
        raise ValueError("Threshold grid is empty")
    grid.insert(0, "GRID_ID", np.arange(len(grid), dtype=int))
    return grid


def load_inputs(args: argparse.Namespace, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment_catalog = pd.read_csv(args.experiment_table, sep="\t", comment="#", dtype={"experiment_id": str})
    design = pd.read_csv(args.cv_experiments, sep="\t", comment="#", dtype={"experiment_id": str})
    required_design = {"enabled", "experiment_id", "domain"}
    if not required_design.issubset(design.columns):
        raise ValueError(f"CV experiment design lacks: {sorted(required_design - set(design.columns))}")
    design = design[design["enabled"].astype(int) == 1].copy()
    if design.empty:
        raise ValueError("No enabled cross-ancestry CV experiments")
    design["domain"] = design["domain"].astype(str).str.upper()
    if design["experiment_id"].duplicated().any():
        raise ValueError("Each experiment_id may occur only once in the CV experiment design")
    if design["domain"].duplicated().any():
        raise ValueError("The default CV engine requires exactly one experiment per ancestry domain")
    missing_catalog = sorted(set(design["experiment_id"]) - set(experiment_catalog["experiment_id"].astype(str)))
    if missing_catalog:
        raise ValueError(f"CV experiments are absent from experiments.tsv: {missing_catalog}")

    scenarios = pd.read_csv(args.cv_scenarios, sep="\t", comment="#", dtype=str)
    required_scenarios = {"enabled", "scenario_id", "analysis_class", "source_domains", "target_domains"}
    if not required_scenarios.issubset(scenarios.columns):
        raise ValueError(f"CV scenario table lacks: {sorted(required_scenarios - set(scenarios.columns))}")
    scenarios = scenarios[scenarios["enabled"].astype(int) == 1].copy()
    if scenarios.empty:
        raise ValueError("No enabled cross-ancestry validation scenarios")
    if scenarios["scenario_id"].duplicated().any():
        raise ValueError("Enabled cross-ancestry scenario_id values must be unique")
    available_domains = set(design["domain"])
    for row in scenarios.itertuples(index=False):
        sources = split_domains(row.source_domains)
        targets = split_domains(row.target_domains)
        if not sources or not targets:
            raise ValueError(f"Scenario {row.scenario_id} must define both source and target domains")
        overlap = sorted(set(sources) & set(targets))
        if overlap:
            raise ValueError(
                f"Scenario {row.scenario_id} leaks target domains into threshold selection: {overlap}"
            )
        requested = set(sources + targets)
        missing = requested - available_domains
        if missing:
            raise ValueError(f"Scenario {row.scenario_id} requests unavailable domains: {sorted(missing)}")

    fold_map = dict(zip(folds["CHR"].astype(int), folds["FOLD"].astype(int)))
    frames, manifest_rows = [], []
    catalog_by_id = experiment_catalog.set_index("experiment_id", drop=False)
    for spec in design.itertuples(index=False):
        catalog = catalog_by_id.loc[spec.experiment_id]
        run = args.analysis_base_root / spec.experiment_id / args.run_subdir
        ancestry_path = run / "ancestry" / "global_ancestry.tsv"
        if not ancestry_path.exists():
            raise FileNotFoundError(f"Missing global ancestry for {spec.experiment_id}: {ancestry_path}")
        n_samples = len(pd.read_csv(ancestry_path, sep="\t", usecols=["IID"], dtype={"IID": str}))
        if n_samples <= 0:
            raise ValueError(f"No samples in {ancestry_path}")
        expected_n = int(catalog["n_query"]) if "n_query" in catalog.index else n_samples
        if n_samples != expected_n:
            raise ValueError(f"Sample-count mismatch for {spec.experiment_id}: ancestry={n_samples}, design={expected_n}")
        for tool in args.tools:
            for ld in args.ld_levels:
                path = run / "results" / tool / ld / "regions_scored.tsv"
                if not path.exists():
                    raise FileNotFoundError(f"Missing scored regions: {path}")
                frame = pd.read_csv(path, sep="\t")
                missing_cols = REQUIRED_REGION_COLS - set(frame.columns)
                if missing_cols:
                    raise ValueError(f"{path} lacks columns: {sorted(missing_cols)}")
                total = len(frame)
                if not frame.empty:
                    frame = frame[(frame["TECHNICALLY_ASSESSABLE"] == 1) & frame["TRUE_ROH"].notna()].copy()
                    frame["CHR"] = frame["CHR"].astype(int)
                    unknown_chr = sorted(set(frame["CHR"]) - set(fold_map))
                    if unknown_chr:
                        raise ValueError(f"Unsupported chromosomes in {path}: {unknown_chr}")
                    frame["FOLD"] = frame["CHR"].map(fold_map).astype(int)
                    frame["EXPERIMENT_ID"] = spec.experiment_id
                    frame["DOMAIN"] = spec.domain
                    frame["N_SAMPLES"] = n_samples
                    frame["TOOL"] = tool
                    frame["LD"] = ld
                    frames.append(frame)
                manifest_rows.append({
                    "EXPERIMENT_ID": spec.experiment_id, "DOMAIN": spec.domain,
                    "QUERY_POPULATION": str(catalog.get("query_population", spec.domain)),
                    "N_SAMPLES": n_samples, "TOOL": tool, "LD": ld,
                    "N_CANDIDATE_ROWS": total, "N_ASSESSABLE_TRUTH_ROWS": len(frame),
                    "PATH": str(path),
                })
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data, pd.DataFrame(manifest_rows), scenarios


def precompute_counts(
    data: pd.DataFrame, grid: pd.DataFrame, n_folds: int, minimum_carriers: int,
    chunk_size: int,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Return GRID x FOLD x (TP,FP,FN,TN) arrays per domain/configuration."""
    cube: dict[tuple[str, str, str], np.ndarray] = {}
    if data.empty:
        return cube
    grid_length = grid["LENGTH_BP_MIN"].to_numpy(float)
    grid_fraction = grid["CARRIER_FRACTION_MIN"].to_numpy(float)
    grid_hvr = grid["HVR_MAX"].to_numpy(float)
    grid_hrs = grid["HRS_MIN"].to_numpy(float)
    grid_rdi = grid["RDI_MIN"].to_numpy(float)
    n_grid = len(grid)

    for (domain, tool, ld, experiment), frame in data.groupby(
        ["DOMAIN", "TOOL", "LD", "EXPERIMENT_ID"], observed=True, sort=False
    ):
        key = (str(domain), str(tool), str(ld))
        target = cube.setdefault(key, np.zeros((n_grid, n_folds, 4), dtype=np.int64))
        length = frame["LENGTH_BP"].to_numpy(float)
        carriers = frame["N_CARRIERS"].to_numpy(float)
        hvr = frame["MEDIAN_HVR_RAW"].to_numpy(float)
        hrs = frame["MEAN_HRS_RAW"].to_numpy(float)
        rdi = frame["RDI_A"].to_numpy(float)
        truth = frame["TRUE_ROH"].astype(int).to_numpy() == 1
        fold = frame["FOLD"].astype(int).to_numpy() - 1
        n_samples = int(frame["N_SAMPLES"].iloc[0])

        for start in range(0, n_grid, chunk_size):
            end = min(start + chunk_size, n_grid)
            carrier_min = np.maximum(minimum_carriers, np.ceil(grid_fraction[start:end] * n_samples))
            pred = (
                (length[:, None] >= grid_length[None, start:end]) &
                (carriers[:, None] >= carrier_min[None, :]) &
                (hvr[:, None] <= grid_hvr[None, start:end]) &
                (hrs[:, None] >= grid_hrs[None, start:end]) &
                (rdi[:, None] >= grid_rdi[None, start:end])
            )
            for fold_idx in range(n_folds):
                take = fold == fold_idx
                if not np.any(take):
                    continue
                p = pred[take]
                t = truth[take, None]
                target[start:end, fold_idx, 0] += np.sum(p & t, axis=0)
                target[start:end, fold_idx, 1] += np.sum(p & ~t, axis=0)
                target[start:end, fold_idx, 2] += np.sum(~p & t, axis=0)
                target[start:end, fold_idx, 3] += np.sum(~p & ~t, axis=0)
    return cube


def evaluation_rows(
    cube: dict[tuple[str, str, str], np.ndarray], grid_id: int,
    domains: list[str], folds: list[int], tools: list[str], lds: list[str],
) -> tuple[list[dict], list[dict]]:
    config_rows, domain_rows = [], []
    for domain in domains:
        available = []
        for tool in tools:
            for ld in lds:
                block = cube.get((domain, tool, ld))
                if block is None:
                    continue
                counts = block[grid_id, folds, :].sum(axis=0)
                if int(counts.sum()) == 0:
                    continue
                metrics = scalar_metrics(counts)
                row = {"DOMAIN": domain, "TOOL": tool, "LD": ld}
                row.update({name: int(counts[i]) for i, name in enumerate(COUNT_NAMES)})
                row["N"] = int(counts.sum()); row["N_PREDICTED"] = int(counts[0] + counts[1])
                row.update(metrics); config_rows.append(row); available.append(row)
        if not available:
            domain_rows.append({"DOMAIN": domain, "STATUS": "no_assessable_rows"})
            continue
        pooled = np.array([[x[name] for name in COUNT_NAMES] for x in available], dtype=np.int64).sum(axis=0)
        pooled_metrics = scalar_metrics(pooled)
        summary = {
            "DOMAIN": domain, "STATUS": "evaluated", "N_CONFIGURATIONS": len(available),
            **{f"POOLED_{name}": int(pooled[i]) for i, name in enumerate(COUNT_NAMES)},
            "POOLED_N": int(pooled.sum()),
        }
        summary.update({f"POOLED_{k}": v for k, v in pooled_metrics.items()})
        for metric in ["F1", "BALANCED_ACCURACY", "PRECISION", "SENSITIVITY", "SPECIFICITY"]:
            vals = np.array([x[metric] for x in available], dtype=float)
            summary[f"MACRO_CONFIG_{metric}"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else math.nan
        domain_rows.append(summary)
    return config_rows, domain_rows


def aggregate_oof(config: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = ["SCENARIO_ID", "ANALYSIS_CLASS", "EVALUATION_SET", "DOMAIN", "TOOL", "LD"]
    count_cols = COUNT_NAMES
    grouped = config.groupby(group_cols, dropna=False, observed=True)
    summed = grouped[count_cols].sum().reset_index()
    if "FOLD" in config:
        coverage = grouped["FOLD"].nunique().rename("N_FOLDS_EVALUATED").reset_index()
        summed = summed.merge(coverage, on=group_cols, how="left")
    rows = []
    for row in summed.itertuples(index=False):
        counts = np.array([getattr(row, x) for x in COUNT_NAMES])
        out = {x: getattr(row, x) for x in group_cols}
        out.update({x: int(getattr(row, x)) for x in COUNT_NAMES})
        if hasattr(row, "N_FOLDS_EVALUATED"):
            out["N_FOLDS_EVALUATED"] = int(row.N_FOLDS_EVALUATED)
        out["N"] = int(counts.sum()); out["N_PREDICTED"] = int(counts[0] + counts[1])
        out.update(scalar_metrics(counts)); rows.append(out)
    config_oof = pd.DataFrame(rows)
    domain_rows = []
    domain_group = ["SCENARIO_ID", "ANALYSIS_CLASS", "EVALUATION_SET", "DOMAIN"]
    for keys, frame in config_oof.groupby(domain_group, dropna=False, observed=True):
        pooled = frame[COUNT_NAMES].to_numpy(np.int64).sum(axis=0)
        out = dict(zip(domain_group, keys)); out["N_CONFIGURATIONS"] = len(frame)
        out.update({f"POOLED_{x}": int(pooled[i]) for i, x in enumerate(COUNT_NAMES)})
        out["POOLED_N"] = int(pooled.sum())
        out.update({f"POOLED_{k}": v for k, v in scalar_metrics(pooled).items()})
        for metric in ["F1", "BALANCED_ACCURACY", "PRECISION", "SENSITIVITY", "SPECIFICITY"]:
            vals = frame[metric].to_numpy(float)
            out[f"MACRO_CONFIG_{metric}"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else math.nan
        domain_rows.append(out)
    return config_oof, pd.DataFrame(domain_rows)


def threshold_stability(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty or not {"STATUS", "SCENARIO_ID", "ANALYSIS_CLASS"}.issubset(selected.columns):
        return pd.DataFrame()
    rows = []
    good = selected[selected["STATUS"] == "selected"].copy()
    for (scenario, analysis_class), frame in good.groupby(["SCENARIO_ID", "ANALYSIS_CLASS"], observed=True):
        row = {"SCENARIO_ID": scenario, "ANALYSIS_CLASS": analysis_class, "N_SELECTED_FOLDS": len(frame)}
        for col in PARAM_COLS:
            vals = frame[col].to_numpy(float)
            modes = pd.Series(vals).mode()
            row[f"{col}_MODE"] = float(modes.iloc[0]) if not modes.empty else math.nan
            row[f"{col}_MEDIAN"] = float(np.nanmedian(vals))
            row[f"{col}_MIN"] = float(np.nanmin(vals)); row[f"{col}_MAX"] = float(np.nanmax(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def genericity_summary(oof_domain: pd.DataFrame) -> pd.DataFrame:
    """Summarize primary held-ancestry performance without imposing a post-hoc pass rule."""
    required = {"ANALYSIS_CLASS", "EVALUATION_SET", "DOMAIN"}
    if oof_domain.empty or not required.issubset(oof_domain.columns):
        return pd.DataFrame([{
            "STATUS": "not_estimable", "DETAIL": "No primary cross-ancestry out-of-fold metrics",
        }])
    primary = oof_domain[
        (oof_domain["ANALYSIS_CLASS"] == "primary") &
        (oof_domain["EVALUATION_SET"] == "cross_ancestry_validation")
    ].copy()
    if primary.empty:
        return pd.DataFrame([{
            "STATUS": "not_estimable", "DETAIL": "No primary leave-one-domain-out rows",
        }])
    out: dict[str, object] = {
        "STATUS": ("descriptive_complete" if "COMPLETE_FOLD_COVERAGE" not in primary or
                   bool(primary["COMPLETE_FOLD_COVERAGE"].astype(int).all()) else
                   "descriptive_incomplete_fold_coverage"),
        "DETAIL": "Preregister a scientific acceptance margin before assigning pass/fail",
        "N_HELD_DOMAINS": int(primary["DOMAIN"].nunique()),
        "HELD_DOMAINS": ",".join(sorted(primary["DOMAIN"].astype(str).unique())),
    }
    metrics = (["F1", "BALANCED_ACCURACY", "PRECISION", "SENSITIVITY"]
               if "F1" in primary else
               ["MACRO_CONFIG_F1", "MACRO_CONFIG_BALANCED_ACCURACY",
                "MACRO_CONFIG_PRECISION", "MACRO_CONFIG_SENSITIVITY"])
    for metric in metrics:
        if metric not in primary:
            continue
        values = primary[metric].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values):
            out[f"MIN_{metric}"] = float(values.min())
            out[f"MEAN_{metric}"] = float(values.mean())
            out[f"MAX_{metric}"] = float(values.max())
            out[f"RANGE_{metric}"] = float(values.max() - values.min())
    return pd.DataFrame([out])


def predict_generic(data: pd.DataFrame, threshold: pd.Series, minimum_carriers: int) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    carrier_min = np.maximum(
        minimum_carriers,
        np.ceil(float(threshold["CARRIER_FRACTION_MIN"]) * data["N_SAMPLES"].to_numpy(float)),
    )
    pred = (
        (data["LENGTH_BP"].to_numpy(float) >= float(threshold["LENGTH_BP_MIN"])) &
        (data["N_CARRIERS"].to_numpy(float) >= carrier_min) &
        (data["MEDIAN_HVR_RAW"].to_numpy(float) <= float(threshold["HVR_MAX"])) &
        (data["MEAN_HRS_RAW"].to_numpy(float) >= float(threshold["HRS_MIN"])) &
        (data["RDI_A"].to_numpy(float) >= float(threshold["RDI_MIN"]))
    )
    out = data.loc[pred].copy()
    out.insert(0, "PRED_HIGH_CONFIDENCE_GENERIC", 1)
    for col in reversed(PARAM_COLS):
        out.insert(1, f"GENERIC_{col}", threshold[col])
    return out


def savefig(path: Path) -> None:
    plt.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close()


def html_table(frame: pd.DataFrame, columns: Iterable[str] | None = None, digits: int = 4) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    use = frame[list(columns)].copy() if columns is not None else frame.copy()
    return use.round(digits).to_html(index=False, border=0, na_rep="NA", classes="dataframe")


# v3.7.2 primary/no-LD and secondary/LD-sensitivity engine -----------------

def configuration_scores(
    cube: dict[tuple[str, str, str], np.ndarray], grid: pd.DataFrame,
    source_domains: list[str], folds: list[int], tools: list[str], lds: list[str],
    minimum_predicted: int,
) -> pd.DataFrame:
    """Score every TOOL x LD x threshold without pooling configurations."""
    n_grid = len(grid)
    blocks = []
    for tool in tools:
        for ld in lds:
            domain_f1, domain_bal, domain_precision = [], [], []
            eligible = np.ones(n_grid, dtype=bool)
            complete = True
            for domain in source_domains:
                counts = cube.get((domain, tool, ld))
                if counts is None:
                    complete = False
                    break
                summed = counts[:, folds, :].sum(axis=1)
                if not np.any(summed.sum(axis=1) > 0):
                    complete = False
                    break
                metrics = metric_arrays(summed)
                domain_f1.append(metrics["F1"])
                domain_bal.append(metrics["BALANCED_ACCURACY"])
                domain_precision.append(metrics["PRECISION"])
                eligible &= (summed[:, 0] + summed[:, 1]) >= minimum_predicted
            if not complete:
                continue
            block = grid.copy()
            block.insert(1, "TOOL", tool)
            block.insert(2, "LD", ld)
            block["MIN_SOURCE_DOMAIN_F1"] = nanmin_stack(domain_f1, n_grid)
            block["MEAN_SOURCE_DOMAIN_F1"] = nanmean_stack(domain_f1, n_grid)
            block["MIN_SOURCE_DOMAIN_BALANCED_ACCURACY"] = nanmin_stack(domain_bal, n_grid)
            block["MEAN_SOURCE_DOMAIN_BALANCED_ACCURACY"] = nanmean_stack(domain_bal, n_grid)
            block["MEAN_SOURCE_DOMAIN_PRECISION"] = nanmean_stack(domain_precision, n_grid)
            eligible &= np.isfinite(block["MIN_SOURCE_DOMAIN_F1"].to_numpy(float))
            block["ELIGIBLE"] = eligible.astype(int)
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def best_configuration(scored: pd.DataFrame) -> pd.Series | None:
    """Choose one configuration by ancestry-robust performance and conservative ties."""
    if scored.empty or "ELIGIBLE" not in scored:
        return None
    eligible = scored[scored["ELIGIBLE"] == 1].copy()
    if eligible.empty:
        return None
    ranked = eligible.sort_values(
        ["MIN_SOURCE_DOMAIN_F1", "MEAN_SOURCE_DOMAIN_F1",
         "MIN_SOURCE_DOMAIN_BALANCED_ACCURACY", "MEAN_SOURCE_DOMAIN_PRECISION",
         "RDI_MIN", "HRS_MIN", "HVR_MAX", "LENGTH_BP_MIN", "CARRIER_FRACTION_MIN",
         "TOOL", "LD"],
        ascending=[False, False, False, False, False, False, True, False, False, True, True],
        na_position="last",
    )
    return ranked.iloc[0]


def add_prefix(rows: list[dict], prefix: dict) -> list[dict]:
    return [{**prefix, **row} for row in rows]


def evaluate_fixed(
    cube: dict[tuple[str, str, str], np.ndarray], grid_id: int,
    domains: list[str], folds: list[int], tool: str, ld: str,
) -> tuple[list[dict], list[dict]]:
    config, domain = evaluation_rows(cube, grid_id, domains, folds, [tool], [ld])
    for row in domain:
        row["SELECTED_TOOL"] = tool
        row["SELECTED_LD"] = ld
    return config, domain


def aggregate_selected_pipeline(
    config: pd.DataFrame, ld_column: str = "LD", required_folds: int | None = None,
) -> pd.DataFrame:
    """OOF performance of a selection algorithm even when its tool varies by fold."""
    if config.empty:
        return pd.DataFrame()
    group_cols = ["SCENARIO_ID", "ANALYSIS_CLASS", "EVALUATION_SET", "DOMAIN"]
    if ld_column != "LD" and ld_column in config:
        group_cols.append(ld_column)
    rows = []
    for keys, frame in config.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts = frame[COUNT_NAMES].to_numpy(np.int64).sum(axis=0)
        out = dict(zip(group_cols, keys))
        out.update({name: int(counts[i]) for i, name in enumerate(COUNT_NAMES)})
        out["N"] = int(counts.sum())
        out["N_PREDICTED"] = int(counts[0] + counts[1])
        out.update(scalar_metrics(counts))
        out["TOOLS_SELECTED"] = ",".join(sorted(frame["TOOL"].astype(str).unique()))
        out["N_DISTINCT_TOOLS"] = int(frame["TOOL"].nunique())
        out["N_FOLDS_EVALUATED"] = int(frame["FOLD"].nunique()) if "FOLD" in frame else math.nan
        if required_folds is not None:
            out["REQUIRED_FOLDS"] = required_folds
            out["COMPLETE_FOLD_COVERAGE"] = int(out["N_FOLDS_EVALUATED"] == required_folds)
        rows.append(out)
    return pd.DataFrame(rows)


def robust_rankings(
    oof_config: pd.DataFrame, required_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return generic and ancestry-specific TOOL x LD rankings from primary LODO OOF rows."""
    required = {"ANALYSIS_CLASS", "EVALUATION_SET", "DOMAIN", "TOOL", "LD", "F1"}
    if oof_config.empty or not required.issubset(oof_config.columns):
        return pd.DataFrame(), pd.DataFrame()
    primary = oof_config[
        (oof_config["ANALYSIS_CLASS"] == "primary") &
        (oof_config["EVALUATION_SET"] == "cross_ancestry_validation")
    ].copy()
    if primary.empty:
        return pd.DataFrame(), pd.DataFrame()

    ancestry = primary.copy()
    if required_folds is not None and "N_FOLDS_EVALUATED" in ancestry:
        ancestry["REQUIRED_FOLDS"] = required_folds
        ancestry["COMPLETE_FOLD_COVERAGE"] = (
            ancestry["N_FOLDS_EVALUATED"].astype(int) == required_folds
        ).astype(int)
    else:
        ancestry["COMPLETE_FOLD_COVERAGE"] = 1
    ancestry = ancestry.sort_values(
        ["DOMAIN", "COMPLETE_FOLD_COVERAGE", "F1", "BALANCED_ACCURACY", "PRECISION", "TOOL", "LD"],
        ascending=[True, False, False, False, False, True, True], na_position="last",
    )
    ancestry["RANK_WITHIN_ANCESTRY"] = ancestry.groupby("DOMAIN").cumcount() + 1

    rows = []
    for (tool, ld), frame in primary.groupby(["TOOL", "LD"], observed=True):
        row: dict[str, object] = {
            "TOOL": tool, "LD": ld, "N_HELD_DOMAINS": int(frame["DOMAIN"].nunique()),
            "HELD_DOMAINS": ",".join(sorted(frame["DOMAIN"].astype(str).unique())),
        }
        if "COMPLETE_FOLD_COVERAGE" in frame:
            row["N_DOMAINS_COMPLETE_FOLD_COVERAGE"] = int(frame["COMPLETE_FOLD_COVERAGE"].sum())
            row["COMPLETE_ALL_DOMAIN_FOLDS"] = int(frame["COMPLETE_FOLD_COVERAGE"].all())
        else:
            row["COMPLETE_ALL_DOMAIN_FOLDS"] = 1
        for metric in ["F1", "BALANCED_ACCURACY", "PRECISION", "SENSITIVITY", "SPECIFICITY"]:
            values = frame[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            row[f"WORST_DOMAIN_{metric}"] = float(values.min()) if len(values) else math.nan
            row[f"MEAN_DOMAIN_{metric}"] = float(values.mean()) if len(values) else math.nan
            row[f"BEST_DOMAIN_{metric}"] = float(values.max()) if len(values) else math.nan
            row[f"RANGE_DOMAIN_{metric}"] = float(values.max() - values.min()) if len(values) else math.nan
        pooled = frame[COUNT_NAMES].to_numpy(np.int64).sum(axis=0)
        row.update({f"POOLED_{name}": int(pooled[i]) for i, name in enumerate(COUNT_NAMES)})
        row.update({f"POOLED_{name}": value for name, value in scalar_metrics(pooled).items()})
        rows.append(row)
    generic = pd.DataFrame(rows).sort_values(
        ["COMPLETE_ALL_DOMAIN_FOLDS", "WORST_DOMAIN_F1", "MEAN_DOMAIN_F1", "WORST_DOMAIN_BALANCED_ACCURACY",
         "MEAN_DOMAIN_PRECISION", "TOOL", "LD"],
        ascending=[False, False, False, False, False, True, True], na_position="last",
    ).reset_index(drop=True)
    generic.insert(0, "GENERIC_RANK", np.arange(1, len(generic) + 1))
    return generic, ancestry


def ld_delta_table(oof_config: pd.DataFrame, primary_ld: str) -> pd.DataFrame:
    if oof_config.empty:
        return pd.DataFrame()
    use = oof_config[
        (oof_config["ANALYSIS_CLASS"] == "primary") &
        (oof_config["EVALUATION_SET"] == "cross_ancestry_validation")
    ].copy()
    if use.empty:
        return pd.DataFrame()
    metrics = ["F1", "BALANCED_ACCURACY", "PRECISION", "SENSITIVITY", "SPECIFICITY"]
    base = use[use["LD"] == primary_ld][["DOMAIN", "TOOL", *metrics]].copy()
    base = base.rename(columns={x: f"{x}_PRIMARY_LD" for x in metrics})
    merged = use.merge(base, on=["DOMAIN", "TOOL"], how="left")
    for metric in metrics:
        merged[f"DELTA_{metric}_VS_{primary_ld.upper()}"] = (
            merged[metric] - merged[f"{metric}_PRIMARY_LD"]
        )
    return merged


def locked_ld_delta_table(locked_oof: pd.DataFrame, primary_ld: str) -> pd.DataFrame:
    if locked_oof.empty:
        return pd.DataFrame()
    keys = ["SCENARIO_ID", "ANALYSIS_CLASS", "EVALUATION_SET", "DOMAIN"]
    metrics = ["F1", "BALANCED_ACCURACY", "PRECISION", "SENSITIVITY", "SPECIFICITY"]
    base = locked_oof[locked_oof["PERTURBATION_LD"] == primary_ld][keys + metrics].rename(
        columns={x: f"{x}_PRIMARY_LD" for x in metrics}
    )
    out = locked_oof.merge(base, on=keys, how="left")
    for metric in metrics:
        out[f"DELTA_{metric}_VS_{primary_ld.upper()}"] = out[metric] - out[f"{metric}_PRIMARY_LD"]
    return out


def candidate_retention(manifest: pd.DataFrame, primary_ld: str) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    keys = ["EXPERIMENT_ID", "DOMAIN", "TOOL"]
    base = manifest[manifest["LD"] == primary_ld][keys + ["N_CANDIDATE_ROWS"]].rename(
        columns={"N_CANDIDATE_ROWS": "N_CANDIDATE_ROWS_PRIMARY_LD"}
    )
    out = manifest.merge(base, on=keys, how="left")
    out["CANDIDATE_RATIO_VS_PRIMARY_LD"] = safe_ratio(
        out["N_CANDIDATE_ROWS"].to_numpy(float),
        out["N_CANDIDATE_ROWS_PRIMARY_LD"].to_numpy(float),
    )
    return out


def make_primary_figures(
    outdir: Path, selected: pd.DataFrame, selected_oof: pd.DataFrame,
    tool_generic: pd.DataFrame, tool_ancestry: pd.DataFrame,
    final_domain: pd.DataFrame, grid: pd.DataFrame,
) -> list[str]:
    figdir = outdir / "figures" / "primary"
    figdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    files = []

    primary = selected_oof[
        (selected_oof.get("ANALYSIS_CLASS") == "primary") &
        (selected_oof.get("EVALUATION_SET") == "cross_ancestry_validation")
    ].copy() if not selected_oof.empty else pd.DataFrame()
    plt.figure(figsize=(8, 5))
    if not primary.empty:
        sns.barplot(data=primary, x="DOMAIN", y="F1", color="#0F766E")
        plt.ylim(0, 1); plt.xlabel("Completely held-out ancestry"); plt.ylabel("Out-of-fold F1")
    else:
        plt.axis("off"); plt.text(.5, .5, "No primary LODO results", ha="center", va="center")
    plt.title("Primary selected pipeline across ancestries", weight="bold")
    name = "01_selected_pipeline_lodo_f1.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(10, 6))
    if not tool_ancestry.empty:
        pivot = tool_ancestry.pivot_table(index="TOOL", columns="DOMAIN", values="F1", aggfunc="first")
        sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="viridis", vmin=0, vmax=100,
                    cbar_kws={"label": "Out-of-fold F1 (%)"})
        plt.xlabel("Held-out ancestry"); plt.ylabel("")
    else:
        plt.axis("off"); plt.text(.5, .5, "No no-LD tool comparison", ha="center", va="center")
    plt.title("No-pruning caller performance", weight="bold")
    name = "02_no_ld_tool_by_ancestry.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(8, 5))
    good = selected[(selected.get("STATUS") == "selected") &
                    (selected.get("ANALYSIS_CLASS") == "primary")].copy() if not selected.empty else pd.DataFrame()
    if not good.empty:
        counts = good["TOOL"].value_counts().rename_axis("TOOL").reset_index(name="N_FOLDS")
        sns.barplot(data=counts, x="TOOL", y="N_FOLDS", color="#2563EB")
        plt.xlabel(""); plt.ylabel("Outer folds selected")
    else:
        plt.axis("off"); plt.text(.5, .5, "No selected tools", ha="center", va="center")
    plt.title("Primary tool-selection stability", weight="bold")
    name = "03_tool_selection_frequency.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(10, max(5, 0.34 * len(good))))
    if not good.empty:
        scaled = good[PARAM_COLS].astype(float).copy()
        for col in PARAM_COLS:
            lo, hi = float(grid[col].min()), float(grid[col].max())
            scaled[col] = (scaled[col] - lo) / (hi - lo) if hi > lo else 0.5
        scaled.index = good["SCENARIO_ID"].astype(str) + " · fold " + good["FOLD"].astype(str) + " · " + good["TOOL"].astype(str)
        sns.heatmap(scaled, cmap="mako", vmin=0, vmax=1,
                    cbar_kws={"label": "Normalized grid position"})
        plt.xlabel(""); plt.ylabel("")
    else:
        plt.axis("off"); plt.text(.5, .5, "No threshold selections", ha="center", va="center")
    plt.title("Primary threshold stability", weight="bold")
    name = "04_primary_threshold_stability.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(10, 5))
    if not selected_oof.empty:
        summary = selected_oof.groupby(["ANALYSIS_CLASS", "EVALUATION_SET"], observed=True)["F1"].mean().reset_index()
        sns.barplot(data=summary, x="ANALYSIS_CLASS", y="F1", hue="EVALUATION_SET",
                    palette=["#0F766E", "#F59E0B"])
        plt.ylim(0, 1); plt.xlabel(""); plt.ylabel("Mean out-of-fold F1"); plt.legend(title="")
    else:
        plt.axis("off"); plt.text(.5, .5, "No source/target metrics", ha="center", va="center")
    plt.title("Source testing and ancestry transfer", weight="bold")
    name = "05_source_test_vs_transfer.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(8, 5))
    if not final_domain.empty:
        sns.barplot(data=final_domain, x="DOMAIN", y="MACRO_CONFIG_F1", color="#7C3AED")
        plt.ylim(0, 1); plt.xlabel(""); plt.ylabel("Full-data refit F1")
    else:
        plt.axis("off"); plt.text(.5, .5, "No final pipeline metrics", ha="center", va="center")
    plt.title("Final generic no-pruning pipeline", weight="bold")
    name = "06_final_pipeline_by_ancestry.png"; savefig(figdir / name); files.append(name)
    return files


def make_sensitivity_figures(
    outdir: Path, generic_rank: pd.DataFrame, ancestry_rank: pd.DataFrame,
    delta: pd.DataFrame, locked_oof: pd.DataFrame, retention: pd.DataFrame,
    primary_ld: str,
) -> list[str]:
    figdir = outdir / "figures" / "ld_sensitivity"
    figdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    files = []

    plt.figure(figsize=(12, max(6, 0.45 * len(generic_rank))))
    if not ancestry_rank.empty:
        plot = ancestry_rank.copy(); plot["CONFIGURATION"] = plot["TOOL"].astype(str) + " · " + plot["LD"].astype(str)
        pivot = plot.pivot_table(index="CONFIGURATION", columns="DOMAIN", values="F1", aggfunc="first")
        sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="crest", vmin=0, vmax=100,
                    cbar_kws={"label": "Out-of-fold F1 (%)"})
        plt.xlabel("Held-out ancestry"); plt.ylabel("")
    else:
        plt.axis("off"); plt.text(.5, .5, "No LD-sensitivity metrics", ha="center", va="center")
    plt.title("Retuned tool × LD performance by ancestry", weight="bold")
    name = "01_retuned_tool_ld_by_ancestry.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(12, max(6, 0.42 * len(generic_rank))))
    if not delta.empty:
        plot = delta[delta["LD"] != primary_ld].copy()
        plot["CONFIGURATION"] = plot["TOOL"].astype(str) + " · " + plot["LD"].astype(str)
        col = f"DELTA_F1_VS_{primary_ld.upper()}"
        pivot = plot.pivot_table(index="CONFIGURATION", columns="DOMAIN", values=col, aggfunc="first")
        limit = float(np.nanmax(np.abs(pivot.to_numpy(float)))) if pivot.size else 1.0
        limit = max(limit, 0.05)
        sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="vlag", center=0, vmin=-limit, vmax=limit,
                    cbar_kws={"label": f"F1 difference versus {primary_ld}"})
        plt.xlabel("Held-out ancestry"); plt.ylabel("")
    else:
        plt.axis("off"); plt.text(.5, .5, "No LD deltas", ha="center", va="center")
    plt.title("Retuned LD sensitivity relative to primary input", weight="bold")
    name = "02_retuned_delta_vs_primary_ld.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(10, 7))
    if not generic_rank.empty:
        sns.scatterplot(data=generic_rank, x="MEAN_DOMAIN_F1", y="WORST_DOMAIN_F1",
                        hue="LD", style="TOOL", s=150)
        for row in generic_rank.itertuples(index=False):
            plt.text(row.MEAN_DOMAIN_F1, row.WORST_DOMAIN_F1, f"{row.TOOL}:{row.LD}", fontsize=8)
        plt.xlim(0, 1); plt.ylim(0, 1); plt.xlabel("Mean held-ancestry F1"); plt.ylabel("Worst held-ancestry F1")
    else:
        plt.axis("off"); plt.text(.5, .5, "No generic rankings", ha="center", va="center")
    plt.title("Generic robustness of all tool × LD configurations", weight="bold")
    name = "03_generic_worst_vs_mean_f1.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(11, 6))
    if not locked_oof.empty:
        use = locked_oof[(locked_oof["ANALYSIS_CLASS"] == "primary") &
                         (locked_oof["EVALUATION_SET"] == "cross_ancestry_validation")]
        sns.barplot(data=use, x="DOMAIN", y="F1", hue="PERTURBATION_LD")
        plt.ylim(0, 1); plt.xlabel("Held-out ancestry"); plt.ylabel("F1"); plt.legend(title="LD input")
    else:
        plt.axis("off"); plt.text(.5, .5, "No locked-pipeline sensitivity", ha="center", va="center")
    plt.title("Locked primary pipeline under LD perturbation", weight="bold")
    name = "04_locked_pipeline_ld_perturbation.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(12, 7))
    if not retention.empty:
        plot = retention.copy(); plot["CONFIGURATION"] = plot["TOOL"].astype(str) + " · " + plot["LD"].astype(str)
        pivot = plot.pivot_table(index="CONFIGURATION", columns="DOMAIN",
                                 values="CANDIDATE_RATIO_VS_PRIMARY_LD", aggfunc="mean")
        if not pivot.empty and np.any(np.isfinite(pivot.to_numpy(float))):
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="rocket_r", cbar_kws={"label": "Candidate ratio"})
            plt.xlabel(""); plt.ylabel("")
        else:
            plt.axis("off"); plt.text(.5, .5, "All candidate counts were zero", ha="center", va="center")
    else:
        plt.axis("off"); plt.text(.5, .5, "No candidate-retention audit", ha="center", va="center")
    plt.title("Candidate retention relative to no pruning", weight="bold")
    name = "05_candidate_retention_by_ld.png"; savefig(figdir / name); files.append(name)

    plt.figure(figsize=(10, 6))
    if not ancestry_rank.empty:
        best = ancestry_rank.sort_values("RANK_WITHIN_ANCESTRY").groupby(["DOMAIN", "TOOL"], observed=True).first().reset_index()
        label = best.pivot(index="TOOL", columns="DOMAIN", values="LD")
        values = np.zeros(label.shape)
        sns.heatmap(values, annot=label.to_numpy(), fmt="", cmap=["#E0F2FE"], cbar=False,
                    xticklabels=label.columns, yticklabels=label.index, linewidths=1, linecolor="white")
        plt.xlabel("Held-out ancestry"); plt.ylabel("")
    else:
        plt.axis("off"); plt.text(.5, .5, "No ancestry-specific LD rankings", ha="center", va="center")
    plt.title("Best retuned LD within each tool and ancestry", weight="bold")
    name = "06_best_ld_by_tool_and_ancestry.png"; savefig(figdir / name); files.append(name)
    return files


def embedded_image_gallery(outdir: Path, prefix: str, files: list[str]) -> tuple[str, int]:
    """Embed generated PNGs so each CV report remains portable as one HTML file."""
    figure_root = (outdir / "figures" / prefix).resolve()
    blocks: list[str] = []
    for name in files:
        path = (figure_root / name).resolve()
        if path.parent != figure_root:
            raise ValueError(f"Unsafe CV figure path: {name!r}")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"CV figure missing or empty: {path}")
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        caption = html.escape(name.replace(".png", "").replace("_", " "))
        alt = html.escape(caption, quote=True)
        blocks.append(
            f"<figure><img src='data:image/png;base64,{payload}' alt='{alt}' "
            f"decoding='async'><figcaption>{caption}</figcaption></figure>"
        )
    return "\n".join(blocks), len(blocks)


def write_v371_reports(
    outdir: Path, primary_ld: str, folds: pd.DataFrame, manifest: pd.DataFrame,
    selected: pd.DataFrame, selected_oof: pd.DataFrame,
    tool_generic: pd.DataFrame, tool_ancestry: pd.DataFrame,
    genericity: pd.DataFrame, final_pipeline: pd.DataFrame, final_domain: pd.DataFrame,
    stability: pd.DataFrame, tool_selection_stability: pd.DataFrame, primary_figures: list[str],
    sensitivity_generic: pd.DataFrame, sensitivity_ancestry: pd.DataFrame,
    sensitivity_delta: pd.DataFrame, locked_oof: pd.DataFrame, locked_delta: pd.DataFrame,
    retention: pd.DataFrame, sensitivity_figures: list[str], n_hc: int,
) -> None:
    primary_gallery, n_primary = embedded_image_gallery(outdir, "primary", primary_figures)
    sensitivity_gallery, n_sensitivity = embedded_image_gallery(
        outdir, "ld_sensitivity", sensitivity_figures,
    )
    css = """<style>body{font-family:Arial,sans-serif;max-width:1450px;margin:32px auto;color:#172033;line-height:1.45}h1,h2{color:#12355b}.lead{background:#edf6f5;padding:16px;border-left:5px solid #0f766e}.note{background:#fff7ed;padding:12px;border-left:5px solid #f59e0b}.table-wrap{width:100%;overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.78rem}th,td{padding:6px;border-bottom:1px solid #dbe3ec;text-align:right}th:first-child,td:first-child{text-align:left}figure{margin:30px 0}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{text-align:center;color:#64748b}</style>"""
    primary_doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>TRUROH primary generic pipeline</title>{css}</head><body>
<!-- TRUROH_SELF_CONTAINED_CV_REPORT_V1 figures={n_primary} -->
<h1>TRUROH primary generic cross-ancestry pipeline</h1>
<p class='lead'>Only the <b>{html.escape(primary_ld)}</b> LD input participates in primary selection. Within each outer fold, every caller receives its own training-only threshold fit; one caller and threshold set are selected using worst-source-ancestry performance, locked, and transferred to the held source fold and ancestry excluded from selection.</p>
<div class='note'>The final full-data caller/threshold refit is for deployment and is not used to calculate out-of-fold performance.</div>
<h2>Genericity summary</h2>{html_table(genericity)}
<h2>Primary selected-pipeline out-of-fold results</h2>{html_table(selected_oof)}
<h2>Fold-specific caller and threshold selections</h2>{html_table(selected)}
<h2>No-pruning generic caller ranking</h2>{html_table(tool_generic)}
<h2>No-pruning ancestry-wise caller ranking</h2>{html_table(tool_ancestry)}
<h2>Final generic pipeline</h2><p>High-confidence candidate rows exported: {n_hc:,}.</p>{html_table(final_pipeline)}
<h2>Final pipeline ancestry metrics</h2>{html_table(final_domain)}
<h2>Caller-selection stability</h2>{html_table(tool_selection_stability)}
<h2>Fold selection and threshold stability</h2>{html_table(stability)}
<h2>Figures</h2>{primary_gallery}
</body></html>"""
    (outdir / "TRUROH_primary_generic_pipeline_report.html").write_text(primary_doc)

    sensitivity_doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>TRUROH LD sensitivity</title>{css}</head><body>
<!-- TRUROH_SELF_CONTAINED_CV_REPORT_V1 figures={n_sensitivity} -->
<h1>TRUROH secondary cross-ancestry LD-sensitivity analysis</h1>
<p class='lead'>Pruned LD inputs never influence the primary caller or threshold. Two secondary analyses are reported: (1) the locked primary caller/threshold applied unchanged to every LD input, and (2) the best achievable training-only threshold refit within every tool × LD configuration.</p>
<h2>Locked primary pipeline under LD perturbation</h2>{html_table(locked_oof)}
<h2>Locked-pipeline difference from {html.escape(primary_ld)}</h2>{html_table(locked_delta)}
<h2>Retuned generic tool × LD ranking</h2>{html_table(sensitivity_generic)}
<h2>Retuned ancestry-wise ranking</h2>{html_table(sensitivity_ancestry)}
<h2>Performance difference from {html.escape(primary_ld)}</h2>{html_table(sensitivity_delta)}
<h2>Candidate-retention audit</h2>{html_table(retention)}
<h2>Figures</h2>{sensitivity_gallery}
</body></html>"""
    (outdir / "TRUROH_secondary_LD_sensitivity_report.html").write_text(sensitivity_doc)

    overview = f"""<!doctype html><html><head><meta charset='utf-8'><title>TRUROH cross-ancestry CV</title>{css}</head><body>
<!-- TRUROH_SELF_CONTAINED_CV_REPORT_V1 figures={n_primary + n_sensitivity} -->
<h1>TRUROH cross-ancestry validation</h1>
<p class='lead'>Primary inference uses one common no-pruning input policy and performs caller plus threshold selection without target-ancestry leakage. All pruned LD regimes are retained as secondary sensitivity analyses and cannot alter the primary recommendation.</p>
<h2>Primary conclusion</h2>{html_table(final_pipeline)}{html_table(genericity)}
<h2>Primary-analysis figures</h2>{primary_gallery}
<h2>Secondary LD-sensitivity figures</h2>{sensitivity_gallery}
<h2>Chromosome folds</h2>{html_table(folds.groupby('FOLD').agg(CHROMOSOMES=('CHR', lambda x: ','.join(map(str, sorted(x)))), TOTAL_BP=('CHR_LENGTH_BP','sum')).reset_index(), digits=0)}
<h2>Input manifest</h2>{html_table(manifest)}
</body></html>"""
    (outdir / "TRUROH_cross_ancestry_CV_report.html").write_text(overview)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-table", type=Path, required=True)
    ap.add_argument("--cv-experiments", type=Path, required=True)
    ap.add_argument("--cv-scenarios", type=Path, required=True)
    ap.add_argument("--analysis-base-root", type=Path, required=True)
    ap.add_argument("--run-subdir", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tools", nargs="+", required=True)
    ap.add_argument("--ld-levels", nargs="+", required=True)
    ap.add_argument("--length-grid", required=True)
    ap.add_argument("--carrier-fraction-grid", required=True)
    ap.add_argument("--hvr-max-grid", required=True)
    ap.add_argument("--hrs-min-grid", required=True)
    ap.add_argument("--rdi-grid", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--minimum-carriers", type=int, default=3)
    ap.add_argument("--minimum-predicted-per-domain", type=int, default=1)
    ap.add_argument("--grid-chunk-size", type=int, default=64)
    ap.add_argument("--primary-ld", default="no")
    ap.add_argument("--selection-rule", default="maximin_source_domain_f1_then_mean")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.grid_chunk_size < 1:
        raise ValueError("grid-chunk-size must be positive")
    if args.primary_ld not in args.ld_levels:
        raise ValueError(f"primary-ld {args.primary_ld!r} is absent from --ld-levels")
    if args.selection_rule != "maximin_source_domain_f1_then_mean":
        raise ValueError(f"Unsupported selection rule: {args.selection_rule}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = args.output_dir / "complete"
    required_outputs = [
        args.output_dir / "TRUROH_cross_ancestry_CV_report.html",
        args.output_dir / "TRUROH_primary_generic_pipeline_report.html",
        args.output_dir / "TRUROH_secondary_LD_sensitivity_report.html",
        args.output_dir / "cv_oof_domain_metrics.tsv",
        args.output_dir / "generic_threshold_full_refit.tsv",
        args.output_dir / "generic_high_confidence_regions.tsv.gz",
    ]
    if complete.exists() and not args.force and all(x.exists() and x.stat().st_size > 0 for x in required_outputs):
        print(f"SKIP complete cross-ancestry CV: {args.output_dir}")
        return
    if complete.exists():
        complete.unlink()

    folds = make_fold_assignment(args.n_folds)
    grid = build_grid(args)
    data, manifest, scenarios = load_inputs(args, folds)
    folds.to_csv(args.output_dir / "fold_assignment.tsv", sep="\t", index=False)
    grid.to_csv(args.output_dir / "threshold_grid_definition.tsv.gz", sep="\t", index=False, compression="gzip")
    manifest.to_csv(args.output_dir / "input_manifest.tsv", sep="\t", index=False)
    pd.read_csv(args.cv_experiments, sep="\t", comment="#").to_csv(
        args.output_dir / "cv_experiment_design.tsv", sep="\t", index=False
    )
    scenarios.to_csv(args.output_dir / "cv_scenario_design.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"PARAMETER": "pipeline_version", "VALUE": "3.7.2"},
        {"PARAMETER": "primary_ld", "VALUE": args.primary_ld},
        {"PARAMETER": "selection_rule", "VALUE": args.selection_rule},
        {"PARAMETER": "n_folds", "VALUE": args.n_folds},
        {"PARAMETER": "tools", "VALUE": ",".join(args.tools)},
        {"PARAMETER": "ld_sensitivity_levels", "VALUE": ",".join(args.ld_levels)},
        {"PARAMETER": "minimum_predicted_per_domain", "VALUE": args.minimum_predicted_per_domain},
    ]).to_csv(args.output_dir / "cv_analysis_contract.tsv", sep="\t", index=False)

    cube = precompute_counts(data, grid, args.n_folds, args.minimum_carriers, args.grid_chunk_size)
    all_folds = list(range(args.n_folds))
    primary_selected, primary_fold_config, primary_fold_domain = [], [], []
    primary_tool_selected, primary_tool_fold_config, primary_tool_fold_domain = [], [], []
    locked_fold_config, locked_fold_domain = [], []
    sensitivity_selected, sensitivity_fold_config, sensitivity_fold_domain = [], [], []

    for scenario in scenarios.itertuples(index=False):
        source_domains = split_domains(scenario.source_domains)
        target_domains = split_domains(scenario.target_domains)
        for fold in range(1, args.n_folds + 1):
            held = fold - 1
            train = [x for x in all_folds if x != held]
            scored = configuration_scores(
                cube, grid, source_domains, train, args.tools, args.ld_levels,
                args.minimum_predicted_per_domain,
            )
            common = {
                "SCENARIO_ID": scenario.scenario_id, "ANALYSIS_CLASS": scenario.analysis_class,
                "FOLD": fold, "SOURCE_DOMAINS": ",".join(source_domains),
                "TARGET_DOMAINS": ",".join(target_domains),
            }

            # Primary: jointly select one caller and one threshold using no-pruning training data.
            primary_scored = scored[scored["LD"] == args.primary_ld].copy() if not scored.empty else pd.DataFrame()
            best = best_configuration(primary_scored)
            if best is None:
                primary_selected.append({
                    **common, "STATUS": "no_selection",
                    "DETAIL": "No eligible no-pruning caller/threshold predicted the minimum in every source domain",
                    "LD": args.primary_ld,
                })
            else:
                choice = {**common, "STATUS": "selected", "DETAIL": ""}
                choice.update(best.to_dict()); primary_selected.append(choice)
                for evaluation_set, domains in [
                    ("source_test", source_domains),
                    ("cross_ancestry_validation", target_domains),
                ]:
                    config_rows, domain_rows = evaluate_fixed(
                        cube, int(best["GRID_ID"]), domains, [held], str(best["TOOL"]), args.primary_ld,
                    )
                    prefix = {
                        "SCENARIO_ID": scenario.scenario_id, "ANALYSIS_CLASS": scenario.analysis_class,
                        "FOLD": fold, "EVALUATION_SET": evaluation_set,
                        "GRID_ID": int(best["GRID_ID"]), "SELECTION_POLICY": "primary_no_ld_joint_tool_threshold",
                    }
                    primary_fold_config.extend(add_prefix(config_rows, prefix))
                    primary_fold_domain.extend(add_prefix(domain_rows, prefix))

                    # Locked sensitivity: change only the LD input; never retune tool or threshold.
                    for perturbation_ld in args.ld_levels:
                        ld_config, ld_domain = evaluate_fixed(
                            cube, int(best["GRID_ID"]), domains, [held], str(best["TOOL"]), perturbation_ld,
                        )
                        ld_prefix = {
                            **prefix, "PERTURBATION_LD": perturbation_ld,
                            "PRIMARY_SELECTED_TOOL": str(best["TOOL"]),
                            "PRIMARY_SELECTED_GRID_ID": int(best["GRID_ID"]),
                            "SELECTION_POLICY": "locked_primary_pipeline_ld_perturbation",
                        }
                        locked_fold_config.extend(add_prefix(ld_config, ld_prefix))
                        locked_fold_domain.extend(add_prefix(ld_domain, ld_prefix))

            # Primary detailed comparison: every caller is tuned independently on no-LD source folds.
            for tool in args.tools:
                block = primary_scored[primary_scored["TOOL"] == tool] if not primary_scored.empty else pd.DataFrame()
                tool_best = best_configuration(block)
                if tool_best is None:
                    primary_tool_selected.append({
                        **common, "TOOL": tool, "LD": args.primary_ld, "STATUS": "no_selection",
                        "DETAIL": "No eligible threshold for this no-pruning caller",
                    })
                    continue
                chosen = {**common, "STATUS": "selected", "DETAIL": ""}
                chosen.update(tool_best.to_dict()); primary_tool_selected.append(chosen)
                for evaluation_set, domains in [
                    ("source_test", source_domains),
                    ("cross_ancestry_validation", target_domains),
                ]:
                    config_rows, domain_rows = evaluate_fixed(
                        cube, int(tool_best["GRID_ID"]), domains, [held], tool, args.primary_ld,
                    )
                    prefix = {
                        "SCENARIO_ID": scenario.scenario_id, "ANALYSIS_CLASS": scenario.analysis_class,
                        "FOLD": fold, "EVALUATION_SET": evaluation_set,
                        "GRID_ID": int(tool_best["GRID_ID"]),
                        "SELECTION_POLICY": "tool_specific_training_fit_primary_ld",
                    }
                    primary_tool_fold_config.extend(add_prefix(config_rows, prefix))
                    primary_tool_fold_domain.extend(add_prefix(domain_rows, prefix))

            # Secondary best-achievable sensitivity: retune within every tool x LD on training folds.
            for tool in args.tools:
                for ld in args.ld_levels:
                    block = scored[(scored["TOOL"] == tool) & (scored["LD"] == ld)] if not scored.empty else pd.DataFrame()
                    config_best = best_configuration(block)
                    if config_best is None:
                        sensitivity_selected.append({
                            **common, "TOOL": tool, "LD": ld, "STATUS": "no_selection",
                            "DETAIL": "No eligible threshold for this tool/LD configuration",
                        })
                        continue
                    chosen = {**common, "STATUS": "selected", "DETAIL": ""}
                    chosen.update(config_best.to_dict()); sensitivity_selected.append(chosen)
                    for evaluation_set, domains in [
                        ("source_test", source_domains),
                        ("cross_ancestry_validation", target_domains),
                    ]:
                        config_rows, domain_rows = evaluate_fixed(
                            cube, int(config_best["GRID_ID"]), domains, [held], tool, ld,
                        )
                        prefix = {
                            "SCENARIO_ID": scenario.scenario_id, "ANALYSIS_CLASS": scenario.analysis_class,
                            "FOLD": fold, "EVALUATION_SET": evaluation_set,
                            "GRID_ID": int(config_best["GRID_ID"]),
                            "SELECTION_POLICY": "secondary_retuned_tool_ld",
                        }
                        sensitivity_fold_config.extend(add_prefix(config_rows, prefix))
                        sensitivity_fold_domain.extend(add_prefix(domain_rows, prefix))

    primary_selected = pd.DataFrame(primary_selected)
    primary_fold_config = pd.DataFrame(primary_fold_config)
    primary_fold_domain = pd.DataFrame(primary_fold_domain)
    primary_tool_selected = pd.DataFrame(primary_tool_selected)
    primary_tool_fold_config = pd.DataFrame(primary_tool_fold_config)
    primary_tool_fold_domain = pd.DataFrame(primary_tool_fold_domain)
    locked_fold_config = pd.DataFrame(locked_fold_config)
    locked_fold_domain = pd.DataFrame(locked_fold_domain)
    sensitivity_selected = pd.DataFrame(sensitivity_selected)
    sensitivity_fold_config = pd.DataFrame(sensitivity_fold_config)
    sensitivity_fold_domain = pd.DataFrame(sensitivity_fold_domain)

    selected_oof = aggregate_selected_pipeline(primary_fold_config, required_folds=args.n_folds)
    primary_tool_oof_config, primary_tool_oof_domain = aggregate_oof(primary_tool_fold_config)
    locked_oof = aggregate_selected_pipeline(
        locked_fold_config, "PERTURBATION_LD", required_folds=args.n_folds,
    )
    sensitivity_oof_config, sensitivity_oof_domain = aggregate_oof(sensitivity_fold_config)

    primary_tool_generic, primary_tool_ancestry = robust_rankings(
        primary_tool_oof_config, required_folds=args.n_folds,
    )
    sensitivity_generic, sensitivity_ancestry = robust_rankings(
        sensitivity_oof_config, required_folds=args.n_folds,
    )
    sensitivity_delta = ld_delta_table(sensitivity_oof_config, args.primary_ld)
    locked_delta = locked_ld_delta_table(locked_oof, args.primary_ld)
    retention = candidate_retention(manifest, args.primary_ld)
    stability = threshold_stability(primary_selected)
    genericity = genericity_summary(selected_oof)

    if (not primary_selected.empty and
            {"STATUS", "SCENARIO_ID", "ANALYSIS_CLASS", "TOOL", "LD"}.issubset(primary_selected.columns)):
        good = primary_selected[primary_selected["STATUS"] == "selected"].copy()
        selection_stability = (good.groupby(["SCENARIO_ID", "ANALYSIS_CLASS", "TOOL", "LD"], observed=True)
                               .size().rename("N_FOLDS_SELECTED").reset_index())
        selection_stability["SELECTION_FRACTION"] = selection_stability["N_FOLDS_SELECTED"] / args.n_folds
    else:
        selection_stability = pd.DataFrame()

    # Backward-compatible primary filenames now contain only the no-LD selection policy.
    primary_selected.to_csv(args.output_dir / "cv_selected_thresholds.tsv", sep="\t", index=False)
    primary_fold_config.to_csv(args.output_dir / "cv_fold_configuration_metrics.tsv", sep="\t", index=False)
    primary_fold_domain.to_csv(args.output_dir / "cv_fold_domain_metrics.tsv", sep="\t", index=False)
    primary_tool_oof_config.to_csv(args.output_dir / "cv_oof_configuration_metrics.tsv", sep="\t", index=False)
    selected_oof.to_csv(args.output_dir / "cv_oof_domain_metrics.tsv", sep="\t", index=False)
    stability.to_csv(args.output_dir / "threshold_stability.tsv", sep="\t", index=False)
    genericity.to_csv(args.output_dir / "genericity_summary.tsv", sep="\t", index=False)

    # Detailed primary outputs.
    primary_tool_selected.to_csv(args.output_dir / "primary_tool_selected_thresholds.tsv", sep="\t", index=False)
    primary_tool_fold_config.to_csv(args.output_dir / "primary_tool_fold_metrics.tsv", sep="\t", index=False)
    primary_tool_fold_domain.to_csv(args.output_dir / "primary_tool_fold_domain_metrics.tsv", sep="\t", index=False)
    primary_tool_oof_config.to_csv(args.output_dir / "primary_tool_oof_metrics.tsv", sep="\t", index=False)
    primary_tool_oof_domain.to_csv(args.output_dir / "primary_tool_oof_domain_metrics.tsv", sep="\t", index=False)
    primary_tool_generic.to_csv(args.output_dir / "primary_generic_tool_ranking.tsv", sep="\t", index=False)
    primary_tool_ancestry.to_csv(args.output_dir / "primary_ancestry_tool_ranking.tsv", sep="\t", index=False)
    selection_stability.to_csv(args.output_dir / "primary_tool_selection_stability.tsv", sep="\t", index=False)

    # Secondary LD-sensitivity outputs: locked perturbation and retuned best-achievable analyses.
    locked_fold_config.to_csv(args.output_dir / "ld_sensitivity_locked_fold_metrics.tsv", sep="\t", index=False)
    locked_fold_domain.to_csv(args.output_dir / "ld_sensitivity_locked_fold_domain_metrics.tsv", sep="\t", index=False)
    locked_oof.to_csv(args.output_dir / "ld_sensitivity_locked_oof_metrics.tsv", sep="\t", index=False)
    locked_delta.to_csv(args.output_dir / "ld_sensitivity_locked_delta_vs_primary.tsv", sep="\t", index=False)
    sensitivity_selected.to_csv(args.output_dir / "ld_sensitivity_retuned_selected_thresholds.tsv", sep="\t", index=False)
    sensitivity_fold_config.to_csv(args.output_dir / "ld_sensitivity_retuned_fold_metrics.tsv", sep="\t", index=False)
    sensitivity_fold_domain.to_csv(args.output_dir / "ld_sensitivity_retuned_fold_domain_metrics.tsv", sep="\t", index=False)
    sensitivity_oof_config.to_csv(args.output_dir / "ld_sensitivity_retuned_oof_metrics.tsv", sep="\t", index=False)
    sensitivity_oof_domain.to_csv(args.output_dir / "ld_sensitivity_retuned_oof_domain_metrics.tsv", sep="\t", index=False)
    sensitivity_generic.to_csv(args.output_dir / "ld_sensitivity_generic_tool_ld_ranking.tsv", sep="\t", index=False)
    sensitivity_ancestry.to_csv(args.output_dir / "ld_sensitivity_ancestry_tool_ld_ranking.tsv", sep="\t", index=False)
    sensitivity_delta.to_csv(args.output_dir / "ld_sensitivity_retuned_delta_vs_primary.tsv", sep="\t", index=False)
    retention.to_csv(args.output_dir / "ld_sensitivity_candidate_retention.tsv", sep="\t", index=False)

    if not sensitivity_generic.empty:
        best_ld_tool = (sensitivity_generic.sort_values("GENERIC_RANK")
                        .groupby("TOOL", observed=True).first().reset_index())
    else:
        best_ld_tool = pd.DataFrame()
    if not sensitivity_ancestry.empty:
        best_ld_ancestry = (sensitivity_ancestry.sort_values("RANK_WITHIN_ANCESTRY")
                            .groupby(["DOMAIN", "TOOL"], observed=True).first().reset_index())
    else:
        best_ld_ancestry = pd.DataFrame()
    best_ld_tool.to_csv(args.output_dir / "ld_sensitivity_best_ld_within_tool.tsv", sep="\t", index=False)
    best_ld_ancestry.to_csv(args.output_dir / "ld_sensitivity_best_ld_by_ancestry_tool.tsv", sep="\t", index=False)

    # Full-data deployment refit: one caller, primary LD only, and one generic threshold.
    all_domains = sorted(manifest["DOMAIN"].astype(str).str.upper().unique())
    full_scores = configuration_scores(
        cube, grid, all_domains, all_folds, args.tools, [args.primary_ld],
        args.minimum_predicted_per_domain,
    )
    final_best = best_configuration(full_scores)
    if final_best is not None:
        row = final_best.copy()
        row["STATUS"] = "selected"
        row["DETAIL"] = "full-data deployment refit after unbiased CV; primary LD only"
        generic = pd.DataFrame([row])
        generic_config_rows, generic_domain_rows = evaluate_fixed(
            cube, int(final_best["GRID_ID"]), all_domains, all_folds,
            str(final_best["TOOL"]), args.primary_ld,
        )
        generic_config = pd.DataFrame(generic_config_rows)
        generic_domain = pd.DataFrame(generic_domain_rows)
        chosen_data = data[(data["TOOL"] == str(final_best["TOOL"])) & (data["LD"] == args.primary_ld)].copy()
        high_confidence = predict_generic(chosen_data, final_best, args.minimum_carriers)
    else:
        generic = pd.DataFrame([{
            "STATUS": "no_selection",
            "DETAIL": "No eligible full-data no-pruning caller/threshold configuration",
            "LD": args.primary_ld,
        }])
        generic_config = pd.DataFrame(); generic_domain = pd.DataFrame()
        high_confidence = data.iloc[0:0].copy()
        high_confidence.insert(0, "PRED_HIGH_CONFIDENCE_GENERIC", pd.Series(dtype=int))
    generic.to_csv(args.output_dir / "generic_threshold_full_refit.tsv", sep="\t", index=False)
    generic.to_csv(args.output_dir / "recommended_generic_pipeline.tsv", sep="\t", index=False)
    generic_config.to_csv(args.output_dir / "generic_threshold_configuration_metrics.tsv", sep="\t", index=False)
    generic_domain.to_csv(args.output_dir / "generic_threshold_domain_metrics.tsv", sep="\t", index=False)
    high_confidence.to_csv(args.output_dir / "generic_high_confidence_regions.tsv.gz", sep="\t", index=False, compression="gzip")

    primary_figures = make_primary_figures(
        args.output_dir, primary_selected, selected_oof, primary_tool_generic,
        primary_tool_ancestry, generic_domain, grid,
    )
    sensitivity_figures = make_sensitivity_figures(
        args.output_dir, sensitivity_generic, sensitivity_ancestry,
        sensitivity_delta, locked_oof, retention, args.primary_ld,
    )
    write_v371_reports(
        args.output_dir, args.primary_ld, folds, manifest, primary_selected, selected_oof,
        primary_tool_generic, primary_tool_ancestry, genericity, generic, generic_domain,
        stability, selection_stability, primary_figures, sensitivity_generic, sensitivity_ancestry,
        sensitivity_delta, locked_oof, locked_delta, retention, sensitivity_figures, len(high_confidence),
    )
    summary = (
        "# TRUROH cross-ancestry validation summary\n\n"
        f"Primary LD input: {args.primary_ld}\n\n"
        "Pruned LD inputs were excluded from primary caller/threshold selection and retained only for sensitivity analyses.\n\n"
        f"Final generic pipeline: {generic.iloc[0].to_dict()}\n\n"
        f"High-confidence candidate rows: {len(high_confidence):,}\n"
    )
    (args.output_dir / "SUMMARY.md").write_text(summary)
    complete.write_text("TRUROH cross-ancestry CV v3.7.2 complete\n")
    selected_folds = int((primary_selected.get("STATUS", pd.Series(dtype=str)) == "selected").sum())
    print(
        f"domains={len(all_domains)} scenarios={len(scenarios)} primary_selected_folds={selected_folds} "
        f"grid={len(grid)} primary_ld={args.primary_ld} generic_high_confidence={len(high_confidence)}"
    )
if __name__ == "__main__":
    main()
