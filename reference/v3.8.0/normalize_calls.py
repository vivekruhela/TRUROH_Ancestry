#!/usr/bin/env python3
"""Normalize PLINK, BCFtools/RoH, or detectRUNS calls to one strict schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pandas as pd

OUTCOLS = ["TOOL", "LD", "CHR", "IID", "START", "END", "LENGTH_BP", "NSNP", "SOURCE"]


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTCOLS)


def read_plink(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return empty()
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    if df.empty:
        return empty()
    rename = {"POS1": "START", "POS2": "END"}
    df = df.rename(columns=rename)
    return df[["CHR", "IID", "START", "END", "NSNP"]]


def read_detectruns(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return empty()
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return empty()
    lower = {c.lower(): c for c in df.columns}
    needed = {"id", "chrom", "from", "to", "nsnp"}
    if not needed.issubset(lower):
        raise ValueError(f"detectRUNS columns not recognized: {list(df.columns)}")
    return pd.DataFrame({
        "CHR": df[lower["chrom"]], "IID": df[lower["id"]],
        "START": df[lower["from"]], "END": df[lower["to"]],
        "NSNP": df[lower["nsnp"]],
    })


def read_bcftools(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return empty()
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "RG":
                continue
            if len(fields) < 8:
                raise ValueError(f"Malformed BCFtools RG line: {line[:120]}")
            rows.append({
                "IID": fields[1], "CHR": fields[2], "START": fields[3],
                "END": fields[4], "NSNP": fields[6],
            })
    return pd.DataFrame(rows) if rows else empty()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["plink", "bcftools", "detectruns"], required=True)
    ap.add_argument("--ld", required=True)
    ap.add_argument("--chr", type=int, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-length", type=int, default=250_000)
    ap.add_argument("--min-snps", type=int, default=15)
    args = ap.parse_args()

    readers = {"plink": read_plink, "bcftools": read_bcftools, "detectruns": read_detectruns}
    df = readers[args.tool](args.input)
    if df.empty:
        out = empty()
    else:
        for col in ["CHR", "START", "END", "NSNP"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("chr", "", regex=False), errors="coerce")
        df["IID"] = df["IID"].astype(str)
        df = df.dropna(subset=["CHR", "IID", "START", "END"]).copy()
        df[["CHR", "START", "END"]] = df[["CHR", "START", "END"]].astype(int)
        df["NSNP"] = df["NSNP"].fillna(0).astype(int)
        df["LENGTH_BP"] = df["END"] - df["START"] + 1
        df = df[(df["CHR"] == args.chr) & (df["END"] >= df["START"])]
        df = df[(df["LENGTH_BP"] >= args.min_length) & (df["NSNP"] >= args.min_snps)]
        df["TOOL"] = args.tool
        df["LD"] = args.ld
        df["SOURCE"] = str(args.input)
        out = df[OUTCOLS].sort_values(["CHR", "IID", "START", "END"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"normalized_rows={len(out)} output={args.output}")


if __name__ == "__main__":
    main()

