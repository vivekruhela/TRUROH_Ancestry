#!/usr/bin/env python3
"""Interactive configuration wizard for TRU-ROH (locked GRID_ID 1433).

Asks for every input, output and coding choice, inspects the files you name (PLINK
FAM/BIM, phenotype header and codes, PCA file, RFMix MSP headers) and writes a cohort JSON
that is validated with the same loader the pipeline uses. It can also write
``config/site.env`` with the module/conda commands your SLURM jobs need.

Usage:
  bash configure.sh                                  # interactive
  bash configure.sh --from config/cohort.example.json --output config/my_cohort.json
  bash configure.sh --answers answers.json --output config/my.json   # non-interactive

When standard input is not a terminal, answers are read line by line from stdin (an empty
line accepts the default), which is how the automated tests drive the wizard.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from configuration import read_config  # noqa: E402

ANSWERS: dict = {}


def ask(key, question, default="", validate=None, help_text=""):
    """Prompt (or take a pre-supplied answer), apply the default and validate."""
    while True:
        if key in ANSWERS:
            raw = str(ANSWERS[key])
        else:
            if help_text and sys.stdin.isatty():
                print(f"  ({help_text})")
            prompt = f"{question}" + (f" [{default}]" if default != "" else "") + ": "
            if sys.stdin.isatty():
                raw = input(prompt)
            else:
                print(prompt, end="", flush=True)
                line = sys.stdin.readline()
                raw = line.rstrip("\n") if line else ""
                print(raw)
        value = raw.strip() or str(default)
        if validate is None:
            return value
        try:
            return validate(value)
        except ValueError as exc:
            print(f"  ! {exc}")
            if key in ANSWERS or not sys.stdin.isatty():
                raise SystemExit(f"Invalid answer for {key!r}: {exc}")


def yes(value):
    v = value.strip().lower()
    if v in {"y", "yes", "true", "1"}:
        return True
    if v in {"n", "no", "false", "0"}:
        return False
    raise ValueError("answer yes or no")


def path_exists(kind):
    def check(value):
        p = Path(value).expanduser()
        if not p.exists():
            print(f"  ! warning: {kind} not found now: {p} (kept; preflight will check again)")
        return str(p)
    return check


def plink_prefix(value):
    p = Path(value).expanduser()
    for ext in (".bed", ".bim", ".fam"):
        if not Path(str(p) + ext).exists():
            print(f"  ! warning: {p}{ext} not found now (preflight will check again)")
    return str(p)


def int_value(lo=0):
    def check(value):
        if not re.fullmatch(r"\d+", value) or int(value) < lo:
            raise ValueError(f"enter an integer >= {lo}")
        return int(value)
    return check


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, help="Where to write the cohort JSON (default config/<project>.json)")
    ap.add_argument("--from", dest="template", type=Path, default=ROOT / "config/cohort.example.json",
                    help="Existing JSON whose values become the defaults")
    ap.add_argument("--answers", type=Path, help="JSON of pre-supplied answers (non-interactive)")
    ap.add_argument("--site-env", type=Path, default=ROOT / "config/site.env", help="Where to write module/conda settings")
    args = ap.parse_args()
    global ANSWERS
    if args.answers:
        ANSWERS = json.loads(args.answers.read_text())
    t = json.loads(args.template.read_text()) if args.template and args.template.exists() else {}
    tc = t.get("columns", {})
    ta = t.get("association", {})
    tr = t.get("resources", {})

    print("\nTRU-ROH configuration wizard (locked GRID_ID 1433). Press Enter to accept [defaults].\n")
    print("1/6  Cohort identity")
    project = ask("project", "Short project ID (letters, digits, . _ -)", t.get("project", "MY_COHORT"),
                  lambda v: v if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", v) else (_ for _ in ()).throw(ValueError("invalid project ID")))
    label = ask("cohort_label", "Display name for reports", t.get("cohort_label", project))

    print("\n2/6  Genotypes and ancestry (GRCh38)")
    plink = ask("plink_prefix", "PLINK 1 binary prefix (QC'd, not LD-pruned; without .bed)", t.get("plink_prefix", ""), plink_prefix)
    chrom_default = t.get("chromosomes", "1-22")
    bim = Path(plink + ".bim")
    if bim.exists():
        with bim.open() as fh:
            present = sorted({int(r.split()[0]) for r in fh if r.split() and r.split()[0].isdigit() and 1 <= int(r.split()[0]) <= 22})
        if present:
            print(f"  BIM contains autosomes: {','.join(map(str, present))}")
            chrom_default = f"{present[0]}-{present[-1]}" if present == list(range(present[0], present[-1] + 1)) else ",".join(map(str, present))
    fam = Path(plink + ".fam")
    if fam.exists():
        print(f"  FAM: {sum(1 for _ in fam.open())} individuals (technical cohort)")
    chroms = ask("chromosomes", "Autosomes to analyse (e.g. 1-22 or 21,22)", chrom_default)
    msp = ask("rfmix_msp_pattern", "RFMix v2 MSP path with {CHR} placeholder", t.get("rfmix_msp_pattern", "/path/chr{CHR}.msp.tsv"),
              lambda v: v if ("{CHR}" in v or "{chr}" in v) else (_ for _ in ()).throw(ValueError("must contain {CHR}")))
    codes = []
    first = Path(msp.replace("{CHR}", re.split(r"[-,]", chroms)[0]).replace("{chr}", re.split(r"[-,]", chroms)[0]))
    if first.exists():
        with first.open() as fh:
            head = fh.readline()
        codes = [n for n, _ in re.findall(r"([^\s=,:]+)\s*=\s*(\d+)", head.split(":", 1)[-1])]
        print(f"  RFMix ancestry labels: {', '.join(codes)}")
    else:
        print(f"  ! warning: {first} not found now")
    sample_map = ask("rfmix_sample_map", "Optional IID→RFMix ID map (blank = identical IDs)", t.get("rfmix_sample_map", ""))

    print("\n3/6  Phenotypes and covariates")
    pheno = ask("phenotype_file", "Phenotype file (TAB-separated, with header)", t.get("phenotype_file", ""), path_exists("phenotype file"))
    header, rows = [], []
    if Path(pheno).exists():
        with open(pheno) as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader)
            rows = [r for _, r in zip(range(20000), reader)]
        print(f"  Columns: {', '.join(header)}")

    def column(v):
        if header and v not in header:
            raise ValueError(f"'{v}' is not a column of the phenotype file")
        return v

    def guess(options, fallback):
        options = [o for o in options if o]
        for o in options:
            for h in header:
                if h.lower() == o.lower():
                    return h
        return fallback

    cols = {
        "fid": ask("columns.fid", "Family ID column", guess([tc.get("fid"), "FID", "fam"], "FID"), column),
        "iid": ask("columns.iid", "Individual ID column", guess([tc.get("iid"), "IID", "id"], "IID"), column),
        "outcome": ask("columns.outcome", "Outcome (AD status) column", guess([tc.get("outcome"), "AD", "status", "Dx_binary"], "AD"), column),
        "age": ask("columns.age", "Age column", guess([tc.get("age"), "Age", "AGE"], "Age"), column),
        "sex": ask("columns.sex", "Sex column", guess([tc.get("sex"), "Sex", "SEX"], "Sex"), column),
    }
    apoe = ask("columns.apoe_genotype", "APOE genotype column (blank if none)", guess([tc.get("apoe_genotype"), "APOE", "APOE_reported", "apoe_wgs", "APOE_genotype"], ""),
               lambda v: column(v) if v else "")
    cols["apoe_genotype"] = apoe

    def values_of(col):
        if not header or col not in header:
            return []
        i = header.index(col)
        return sorted({r[i] for r in rows if len(r) > i and r[i] not in ("", "NA", "nan", "-9", ".")})

    ov = values_of(cols["outcome"])
    if ov:
        print(f"  Observed {cols['outcome']} codes: {', '.join(ov[:10])}")
    tom = t.get("outcome_map", {})
    case_default = next((k for k, v in tom.items() if v == 1), ov[-1] if ov else "1")
    ctrl_default = next((k for k, v in tom.items() if v == 0), ov[0] if ov else "0")
    case = ask("outcome_case", "Code meaning AD case", case_default)
    ctrl = ask("outcome_control", "Code meaning control", ctrl_default)
    sv = values_of(cols["sex"])
    if sv:
        print(f"  Observed {cols['sex']} codes: {', '.join(sv[:10])}")
    tsm = t.get("sex_map", {})
    female = ask("sex_female", "Code meaning female (analysed as 1)", next((k for k, v in tsm.items() if v == 1), "2"))
    male = ask("sex_male", "Code meaning male (analysed as 0)", next((k for k, v in tsm.items() if v == 0), "1"))
    extra = {}
    raw_extra = ask("extra_covariates", "Extra numeric covariates as NAME=Header, comma-separated (blank for none)",
                    ",".join(f"{k}={v}" for k, v in tc.get("covariates", {}).items()))
    for item in filter(None, [x.strip() for x in raw_extra.split(",")]):
        name, _, src = item.partition("=")
        extra[name.strip()] = column(src.strip())
    cols["covariates"] = extra
    pca = ask("pca_file", "PCA eigenvector file (FID IID PC1..PC5 ...)", t.get("pca_file", ""), path_exists("PCA file"))
    base_cov = ta.get("covariates", ["AGE", "SEX", "PC1", "PC2", "PC3", "PC4", "PC5"])
    covs = ask("association.covariates", "Baseline covariates (comma-separated)", ",".join(base_cov + [k for k in extra if k not in base_cov]))
    anc_default = ta.get("ancestries") or [c for c in ["NAM", "EUR", "AFR"]]
    ancestries = ask("association.ancestries", "Ancestries for local-ancestry models", ",".join(anc_default))
    primary = ask("association.primary_ancestry", "Primary ancestry (one of the ancestries above, or 'auto' = highest mean global ancestry in the cohort)", ta.get("primary_ancestry", ancestries.split(",")[0]))
    reference = ask("association.global_reference", "Reference ancestry for global-ancestry terms", ta.get("global_reference", "EUR"))

    print("\n4/6  Output")
    out = ask("output_root", "Dedicated output directory (new or this project's previous output)", t.get("output_root", f"/path/to/{project}_TRUROH"))

    print("\n5/6  Software")
    plink_bin = ask("tools.plink", "PLINK 1.9 executable", t.get("tools", {}).get("plink") or shutil.which("plink") or "plink")
    bcf_bin = ask("tools.bcftools", "BCFtools executable (needs the fill-tags plugin)", t.get("tools", {}).get("bcftools") or shutil.which("bcftools") or "bcftools")
    modules = ask("site.modules", "Cluster modules to load in jobs (blank for none, e.g. 'Plink/1.9.10 BCFTOOLS')", "")
    conda = ask("site.conda", "Conda environment to activate in jobs (blank for none, e.g. truroh-grid1433)", "")
    conda_sh = ask("site.conda_sh", "Path to conda.sh (only if a conda env was given)", "") if conda else ""

    print("\n6/6  SLURM resources")
    res = {
        "partition": ask("resources.partition", "Partition (blank = cluster default)", tr.get("partition", "")),
        "account": ask("resources.account", "Account (blank = none)", tr.get("account", "")),
        "constraint": ask("resources.constraint", "Node constraint (blank = none)", tr.get("constraint", "")),
        "cpus": ask("resources.cpus", "CPUs per chromosome job", tr.get("cpus", 4), int_value(1)),
        "memory": ask("resources.memory", "Memory per chromosome job", tr.get("memory", "64G")),
        "time": ask("resources.time", "Time limit per chromosome job", tr.get("time", "2-00:00:00")),
        "max_concurrent": ask("resources.max_concurrent", "Maximum concurrent chromosome jobs (0 = no limit)", tr.get("max_concurrent", 4), int_value(0)),
        "final_cpus": ask("resources.final_cpus", "CPUs for finalization", tr.get("final_cpus", 4), int_value(1)),
        "final_memory": ask("resources.final_memory", "Memory for finalization", tr.get("final_memory", "32G")),
        "final_time": ask("resources.final_time", "Time limit for finalization", tr.get("final_time", "1-00:00:00")),
        "final_constraint": ask("resources.final_constraint", "Node constraint for finalization (blank = same)", tr.get("final_constraint", "")),
    }

    cfg = {
        "project": project, "cohort_label": label, "genome_build": "GRCh38", "chromosomes": chroms,
        "plink_prefix": plink, "phenotype_file": pheno, "pca_file": pca, "output_root": out,
        "rfmix_msp_pattern": msp, "rfmix_sample_map": sample_map,
        "gene_annotation_file": t.get("gene_annotation_file", ""), "columns": cols,
        "outcome_map": {ctrl: 0, case: 1}, "sex_map": {male: 0, female: 1},
        "missing_values": t.get("missing_values", ["", "NA", "N/A", "NaN", "nan", "-9", ".", "None"]),
        "association": {**{k: v for k, v in ta.items() if k not in {"covariates", "ancestries", "primary_ancestry", "global_reference"}},
                        "covariates": [c.strip() for c in covs.split(",") if c.strip()],
                        "ancestries": [a.strip() for a in ancestries.split(",") if a.strip()],
                        "primary_ancestry": primary, "global_reference": reference},
        "tools": {"plink": plink_bin, "bcftools": bcf_bin}, "resources": res,
        "keep_intermediate_vcf": bool(t.get("keep_intermediate_vcf", False)),
    }
    target = args.output or ROOT / "config" / f"{project}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2) + "\n")
    try:
        read_config(target)
    except Exception as exc:  # report and leave the file for editing
        print(f"\nConfiguration written to {target} but it does not validate yet:\n  {exc}")
        sys.exit(1)
    if modules or conda:
        lines = ["# Written by configure.sh; sourced by config/environment.sh in every job."]
        if modules:
            lines.append(f'export TRUROH_MODULES="{modules}"')
        if conda:
            if conda_sh:
                lines.append(f'source "{conda_sh}"')
            lines.append(f'conda activate "{conda}"')
        args.site_env.write_text("\n".join(lines) + "\n")
        print(f"Site environment written to {args.site_env}")
    print(f"\nValid configuration written to {target}\n\nNext steps:\n"
          f"  bash run.sh --config {target} preflight          # checks data, coding and tools\n"
          f"  bash submit.sh --config {target} --dry-run       # shows the sbatch commands\n"
          f"  bash submit.sh --config {target}                 # submits chromosome array + report job\n")


if __name__ == "__main__":
    main()
