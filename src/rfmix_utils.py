#!/usr/bin/env python3
"""Small, strict RFMix v2 MSP parser used by the simulation benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


def _iid_from_hap(col: str) -> str:
    col = str(col).strip()
    for pat in [r":::[^:]+$", r"[._-](?:hap)?[012]$", r"[._-][AB]$"]:
        new = re.sub(pat, "", col, flags=re.IGNORECASE)
        if new != col:
            return new
    return col


@dataclass
class MSPData:
    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    ancestry_names: list[str]
    ancestry_codes: list[str]
    sample_haps: dict[str, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def read(cls, path: str | Path) -> "MSPData":
        path = Path(path)
        mapping: dict[str, str] = {}
        header: list[str] | None = None
        data_lines: list[str] = []
        with path.open() as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if stripped.startswith("#Subpopulation order/codes:"):
                    payload = stripped.split(":", 1)[1]
                    for name, code in re.findall(r"([^\s=,]+)\s*=\s*([^\s,]+)", payload):
                        mapping[str(code)] = str(name)
                elif stripped.startswith("#chm") or stripped.startswith("chm\t"):
                    header = stripped.lstrip("#").split("\t")
                elif stripped and not stripped.startswith("#"):
                    data_lines.append(stripped)
        if header is None:
            raise ValueError(f"RFMix MSP header beginning '#chm' not found: {path}")
        if not data_lines:
            raise ValueError(f"RFMix MSP contains no data rows: {path}")

        rows = [line.split("\t") for line in data_lines]
        df = pd.DataFrame(rows, columns=header)
        normalized = {re.sub(r"\s+", " ", str(col).strip().lower()): col for col in df.columns}
        required = ("chm", "spos", "epos", "n snps")
        missing = [name for name in required if name not in normalized]
        if missing:
            raise ValueError(f"RFMix MSP is missing required columns {missing}: {path}")
        chrom_col = normalized["chm"]
        start_col = normalized["spos"]
        end_col = normalized["epos"]
        last_meta_idx = header.index(normalized["n snps"])
        hap_cols = list(df.columns[last_meta_idx + 1:])
        if len(hap_cols) < 2:
            raise ValueError(f"No RFMix haplotype columns found: {path}")

        chrom = df[chrom_col].astype(str).str.replace(r"^chr", "", regex=True, case=False).to_numpy()
        start = pd.to_numeric(df[start_col], errors="raise").astype(int).to_numpy()
        end = pd.to_numeric(df[end_col], errors="raise").astype(int).to_numpy()
        if np.any(end < start):
            raise ValueError(f"RFMix MSP contains epos < spos: {path}")

        observed = sorted({str(x) for c in hap_cols for x in df[c].dropna().unique()})
        if not mapping:
            mapping = {code: code for code in observed}
        codes = sorted(mapping, key=lambda x: (not str(x).isdigit(), str(x)))
        names = [mapping[c] for c in codes]

        grouped: dict[str, list[str]] = {}
        for col in hap_cols:
            grouped.setdefault(_iid_from_hap(col), []).append(col)
        sample_haps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for iid, cols in grouped.items():
            if len(cols) != 2:
                raise ValueError(f"Expected two haplotype columns for {iid}; found {cols}")
            sample_haps[str(iid)] = (df[cols[0]].astype(str).to_numpy(), df[cols[1]].astype(str).to_numpy())
        return cls(chrom, start, end, names, codes, sample_haps)

    def local_proportions(self, iid: str, start: int, end: int) -> np.ndarray:
        if iid not in self.sample_haps:
            return np.full(len(self.ancestry_codes), np.nan)
        overlap = np.maximum(0, np.minimum(self.end, end) - np.maximum(self.start, start) + 1)
        mask = overlap > 0
        if not np.any(mask):
            return np.full(len(self.ancestry_codes), np.nan)
        h0, h1 = self.sample_haps[iid]
        totals = np.zeros(len(self.ancestry_codes), dtype=float)
        denom = 2.0 * overlap[mask].sum()
        for idx, code in enumerate(self.ancestry_codes):
            totals[idx] = overlap[mask][h0[mask] == code].sum() + overlap[mask][h1[mask] == code].sum()
        return totals / denom if denom > 0 else np.full(len(totals), np.nan)

    def global_sums(self) -> dict[str, np.ndarray]:
        length = self.end - self.start + 1
        result = {}
        for iid, (h0, h1) in self.sample_haps.items():
            sums = np.zeros(len(self.ancestry_codes), dtype=float)
            for idx, code in enumerate(self.ancestry_codes):
                sums[idx] = length[h0 == code].sum() + length[h1 == code].sum()
            result[iid] = sums
        return result


def safe_ancestry_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_") or "UNK"
