#!/usr/bin/env python3
"""Generate a small synthetic admixed cohort for end-to-end testing of TRU-ROH.

The cohort is entirely artificial. It contains:
  * PLINK 1 binary genotypes (BED/BIM/FAM) on two short GRCh38-labelled chromosomes;
  * three-way local ancestry (AFR, EUR, NAM) written as RFMix v2 ``.msp.tsv`` files;
  * one recurrent autozygous segment planted in a fixed set of carriers (a "true" shared ROH),
    plus private autozygous segments in other individuals (which should not reach the carrier floor);
  * phenotypes (AD status, age, sex, APOE e4 dosage) with a modest ROH effect on AD;
  * a PCA eigenvector file and a KING-style kinship table with a few related pairs.

Nothing here is used for scientific inference; it only exercises every pipeline stage with
real PLINK 1.9 and BCFtools executables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ANCESTRIES = ["AFR", "EUR", "NAM"]


def write_bed(prefix: Path, codes: np.ndarray) -> None:
    """Write SNP-major PLINK BED from codes (0=hom A1, 2=het, 3=hom A2, 1=missing)."""
    m, n = codes.shape
    pad = np.ones((m, ((n + 3) // 4) * 4), dtype=np.uint8)
    pad[:, :n] = codes
    packed = pad[:, 0::4] | (pad[:, 1::4] << 2) | (pad[:, 2::4] << 4) | (pad[:, 3::4] << 6)
    Path(str(prefix) + ".bed").write_bytes(b"\x6c\x1b\x01" + packed.astype(np.uint8).tobytes())


def ancestry_tracts(rng, length, props, rate_per_bp):
    """Return list of (start, end, ancestry index) tiling [1, length]."""
    tracts, pos = [], 1
    while pos <= length:
        step = int(rng.exponential(1.0 / rate_per_bp)) + 1
        end = min(length, pos + step - 1)
        tracts.append((pos, end, int(rng.choice(3, p=props))))
        pos = end + 1
    return tracts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--chromosomes", default="21,22")
    ap.add_argument("--chrom-length", type=int, default=40_000_000)
    ap.add_argument("--spacing", type=int, default=2_000, help="Mean SNP spacing (bp)")
    ap.add_argument("--carriers", type=int, default=30, help="Carriers of the planted shared ROH")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    chroms = [int(c) for c in args.chromosomes.split(",")]
    n = args.n
    iids = [f"S{i:04d}" for i in range(1, n + 1)]
    fids = [f"F{i:04d}" for i in range(1, n + 1)]
    # Ten related pairs (second member copies one haplotype of the first): kinship ~0.25.
    related = [(2 * k, 2 * k + 1) for k in range(10)]
    for a, b in related:
        fids[b] = fids[a]

    global_q = rng.dirichlet([1.0, 2.0, 4.0], size=n)  # AFR, EUR, NAM
    carriers = rng.choice(np.arange(40, n), size=args.carriers, replace=False)
    planted = {"chrom": chroms[0], "start": 15_000_000, "end": 17_500_000}

    bims, code_blocks = [], []
    msp_window = 200_000
    for c in chroms:
        pos = np.cumsum(rng.integers(max(1, args.spacing // 2), args.spacing * 3 // 2 + 1, size=args.chrom_length // args.spacing))
        pos = pos[(pos >= 10_000) & (pos <= args.chrom_length)]
        m = len(pos)
        base = rng.uniform(0.05, 0.5, size=m)
        fst = 0.08
        freqs = np.stack([np.clip(rng.beta(base * (1 - fst) / fst, (1 - base) * (1 - fst) / fst), 0.01, 0.99) for _ in ANCESTRIES])
        haps = np.zeros((2, n, m), dtype=np.uint8)
        anc = np.zeros((2, n, m), dtype=np.uint8)
        founders = {k: (rng.random(m) < freqs[k]).astype(np.uint8) for k in range(3)}
        for i in range(n):
            for h in range(2):
                for s, e, k in ancestry_tracts(rng, args.chrom_length, global_q[i], 1.2e-7):
                    sel = (pos >= s) & (pos <= e)
                    anc[h, i, sel] = k
                    haps[h, i, sel] = (rng.random(sel.sum()) < freqs[k, sel]).astype(np.uint8)
        for a, b in related:
            haps[0, b], anc[0, b] = haps[0, a], anc[0, a]
        # private autozygous segments (1-3 Mb) in ~30 % of individuals
        for i in range(n):
            if rng.random() < 0.3:
                L = int(rng.integers(1_000_000, 3_000_000))
                s = int(rng.integers(1_000_000, args.chrom_length - L))
                sel = (pos >= s) & (pos <= s + L)
                haps[1, i, sel], anc[1, i, sel] = haps[0, i, sel], anc[0, i, sel]
        if c == planted["chrom"]:
            sel = (pos >= planted["start"]) & (pos <= planted["end"])
            for i in carriers:
                jitter = int(rng.integers(-30_000, 30_001))
                s2 = (pos >= planted["start"] + jitter) & (pos <= planted["end"] + jitter)
                haps[0, i, s2] = founders[2][s2]  # shared NAM founder haplotype
                haps[1, i, s2] = founders[2][s2]
                anc[0, i, s2] = 2
                anc[1, i, s2] = 2
        g = haps[0] + haps[1]  # n x m count of allele "1" (A2)
        codes = np.where(g == 0, 0, np.where(g == 1, 2, 3)).astype(np.uint8).T  # m x n
        miss = rng.random(codes.shape) < 0.002
        codes[miss] = 1
        code_blocks.append(codes)
        bims.append(pd.DataFrame({"CHR": c, "ID": [f"{c}:{p}" for p in pos], "CM": 0, "POS": pos, "A1": "A", "A2": "G"}))
        # MSP: majority ancestry per window and haplotype
        edges = np.arange(1, args.chrom_length + 1, msp_window)
        with (out / f"chr{c}.msp.tsv").open("w") as f:
            f.write("#Subpopulation order/codes: AFR=0\tEUR=1\tNAM=2\n")
            f.write("#chm\tspos\tepos\tsgpos\tegpos\tn snps\t" + "\t".join(f"{x}.{h}" for x in iids for h in (0, 1)) + "\n")
            for w, s in enumerate(edges):
                e = min(args.chrom_length, s + msp_window - 1)
                sel = (pos >= s) & (pos <= e)
                if not sel.any():
                    continue
                vals = []
                for i in range(n):
                    for h in range(2):
                        vals.append(int(np.bincount(anc[h, i, sel], minlength=3).argmax()))
                f.write("\t".join(map(str, [c, int(pos[sel][0]), int(pos[sel][-1]), w * 0.2, (w + 1) * 0.2, int(sel.sum())] + vals)) + "\n")

    bim = pd.concat(bims, ignore_index=True)
    codes = np.vstack(code_blocks)
    prefix = out / "cohort"
    write_bed(prefix, codes)
    bim.to_csv(str(prefix) + ".bim", sep="\t", header=False, index=False)
    sex = rng.choice([1, 2], size=n, p=[0.35, 0.65])
    pd.DataFrame({"FID": fids, "IID": iids, "PAT": 0, "MAT": 0, "SEX": sex, "PHENO": -9}).to_csv(str(prefix) + ".fam", sep=" ", header=False, index=False)

    # Phenotypes: AD ~ age + sex + APOE e4 + planted ROH (OR ~3) + NAM ancestry
    age = np.round(rng.normal(72, 8, size=n), 1)
    e4 = rng.choice([0, 1, 2], size=n, p=[0.75, 0.22, 0.03])
    roh = np.zeros(n)
    roh[carriers] = 1
    lin = -1.2 + 0.05 * (age - 72) + 0.2 * (sex == 2) + 1.0 * e4 + 1.1 * roh + 0.5 * global_q[:, 2]
    ad = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    pheno = pd.DataFrame({"FID": fids, "IID": iids, "Age": age, "Sex": sex, "AD": ad + 1, "APOE_reported": e4})
    pheno.loc[5, "Age"] = np.nan  # one incomplete record to exercise complete-case handling
    pheno.to_csv(out / "phenotypes.tsv", sep="\t", index=False, na_rep="NA")

    # PCA on standardized genotypes (LD-thinned)
    gsub = np.where(codes[::25] == 1, np.nan, np.where(codes[::25] == 0, 0, np.where(codes[::25] == 2, 1, 2))).astype(float).T
    gsub = np.nan_to_num(gsub - np.nanmean(gsub, 0)) / (np.nanstd(gsub, 0) + 1e-9)
    u, s, _ = np.linalg.svd(gsub, full_matrices=False)
    pcs = u[:, :10] * s[:10]
    pd.concat([pd.DataFrame({"FID": fids, "IID": iids}), pd.DataFrame(pcs, columns=[f"PC{j}" for j in range(1, 11)])], axis=1) \
        .to_csv(out / "cohort.eigenvec", sep=" ", header=False, index=False, float_format="%.6f")

    kin = pd.DataFrame([{"#FID1": fids[a], "IID1": iids[a], "FID2": fids[b], "IID2": iids[b], "NSNP": len(bim), "HETHET": 0.1, "IBS0": 0.0, "KINSHIP": 0.25} for a, b in related])
    kin.to_csv(out / "cohort.kin0", sep="\t", index=False)

    truth = {"planted_region": planted, "carriers": [iids[i] for i in sorted(carriers)], "n": n,
             "chromosomes": chroms, "markers": int(len(bim)), "related_pairs": [[iids[a], iids[b]] for a, b in related]}
    (out / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(f"Synthetic cohort: N={n}, markers={len(bim)}, chromosomes={chroms}, planted carriers={len(carriers)} -> {out}")


if __name__ == "__main__":
    main()
