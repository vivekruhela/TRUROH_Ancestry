from __future__ import annotations
import csv, hashlib, json, math, os, re, subprocess, sys, tempfile, time
from pathlib import Path
from decimal import Decimal, ROUND_CEILING
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LOCK={'GRID_ID':1433,'TOOL':'bcftools','LD':'no','LENGTH_BP_MIN':1000000,'CARRIER_FRACTION_MIN':0.005,'HVR_MAX':0.005,'HRS_MIN':0.5,'RDI_MIN':0.7}
# Ancillary parameters are also preserved from the simulation release.
TECH=dict(caller_length=250000,caller_snps=15,gt_error=30,rec_rate=1e-8,
    discovery_fraction=0.0025,min_carriers=3,merge_gap=250000,merge_tv=0.25,overlap=0.5,
    backgrounds=500,proposal_multiplier=4,buffer_bp=10000000,site_ratio_min=0.25,site_ratio_max=4.0,
    min_sites=10,min_ess=40,individual_min_z=-3.0,individual_max_p=0.99,
    hvr_valid_min=0.05,hvr_pass_min=0.05,hvr_indiv_z=-3.0,hvr_indiv_p=0.99,
    hvr_region_z=-2.5,hvr_region_p=0.95,hvr_meta_p=0.05,hrs_min=0.01,hrs_min_seeds=3,
    hrs_zero_max=0.95,ridge_lambda=1.0,seed_maf=0.05,seed_geno=0.10,seed_hwe=1e-6,
    seed_window=200,seed_step=50,seed_r2=0.20,seed=108,chunk_variants=10000)
CHROM_LENGTHS=dict(enumerate([248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468],1))
ALIASES={'AMR':'NAM','America':'NAM','American':'NAM','Native_American':'NAM','NAM':'NAM',
    'EUR':'EUR','Europe':'EUR','European':'EUR','AFR':'AFR','Africa':'AFR','African':'AFR',
    'EAS':'EAS','East_Asia':'EAS','SAS':'SAS','Central_South_Asia':'SAS','Middle_East':'MDE','Oceania':'OCE'}

def canonical(x): return ALIASES.get(x,x)
def chromosomes(s):
    out=[]
    for part in str(s).split(','):
        if '-' in part:
            a,b=map(int,part.split('-')); out.extend(range(a,b+1))
        else: out.append(int(part))
    if not out or len(set(out))!=len(out) or any(c not in CHROM_LENGTHS for c in out):
        raise ValueError('chromosomes must contain unique autosomes 1-22')
    return sorted(out)

def carrier_min(n, fraction=0.005):
    return max(3,int((Decimal(str(fraction))*Decimal(int(n))).to_integral_value(rounding=ROUND_CEILING)))
def fingerprint(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()
def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def stat_signature(path):
    p=Path(path); s=p.stat()
    return {'path':str(p.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}
def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # JSON null for unavailable statistics, never fake zero.
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
        if isinstance(x,Path):return str(x)
        return x
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        json.dump(clean(data),f,indent=2,allow_nan=False); f.write('\n'); name=f.name
    os.replace(name,path)
def table(path, frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Compression must be selected explicitly because the temporary name differs.
    comp='gzip' if str(path).endswith('.gz') else None
    tmp=path.with_name(path.name+'.tmp')
    frame.to_csv(tmp,sep='\t',index=False,na_rep='NA',compression=comp)
    os.replace(tmp,path)
def read_table(path, **kwargs):
    return pd.read_csv(path,sep='\t',converters={'IID':str,'FID':str,'REGION_ID':str},**kwargs)
def load_config(path):
    from configuration import read_config
    return read_config(path)

def render(pattern,c):return pattern.replace('{CHR}',str(c)).replace('{chr}',str(c))
def run(command, logfile, *, cwd=None):
    command=[str(x) for x in command]
    Path(logfile).parent.mkdir(parents=True,exist_ok=True)
    import shlex
    with Path(logfile).open('a') as f:
        f.write('\n$ '+shlex.join(command)+'\n');f.flush()
        proc=subprocess.run(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    if proc.returncode:
        tail=Path(logfile).read_text(errors='replace')[-2500:]
        raise RuntimeError(f'Command exited {proc.returncode}: {command[0]}; log={logfile}\n{tail}')

def read_fam(prefix):
    f=pd.read_csv(str(prefix)+'.fam',sep=r'\s+',header=None,dtype=str,keep_default_na=False)
    if f.shape[1]!=6: raise ValueError('PLINK .fam must have exactly six columns')
    f.columns=['FID','IID','PAT','MAT','SEX_FAM','PHENO_FAM']
    if f.empty or f.IID.duplicated().any() or f.IID.eq('').any():
        raise ValueError('Nonempty FAM with unique nonempty IIDs is required')
    if f[['FID','IID']].apply(lambda s:s.str.contains(r'\s').any()).any():raise ValueError('Whitespace in FAM IDs')
    return f

def read_bim(prefix):
    b=pd.read_csv(str(prefix)+'.bim',sep=r'\s+',header=None,dtype=str,keep_default_na=False)
    if b.shape[1]!=6: raise ValueError('PLINK .bim must have six columns')
    b.columns=['CHR','ID','CM','POS','A1','A2']
    b.CHR=b.CHR.str.replace(r'^chr','',regex=True,case=False)
    b['POS']=pd.to_numeric(b.POS,errors='raise').astype(np.int64)
    return b

def verify_bed(prefix,n,m):
    p=Path(str(prefix)+'.bed')
    with p.open('rb') as f:magic=f.read(3)
    if magic!=b'\x6c\x1b\x01':raise ValueError('Expected variant-major PLINK BED')
    expected=3+m*((n+3)//4)
    if p.stat().st_size!=expected:raise ValueError(f'BED size mismatch: {p.stat().st_size} vs {expected}')

def split_bed(cfg,c,outprefix):
    fam=read_fam(cfg['plink_prefix']);bim=read_bim(cfg['plink_prefix'])
    verify_bed(cfg['plink_prefix'],len(fam),len(bim))
    idx=np.flatnonzero(bim.CHR.eq(str(c)).to_numpy());sub=bim.iloc[idx].copy()
    if sub.empty:raise ValueError(f'No genotype variants for chromosome {c}')
    if np.any(np.diff(sub.POS)<0):raise ValueError(f'Unsorted positions on chromosome {c}')
    if sub.POS.duplicated().any():raise ValueError(f'Duplicate chromosome-position records on chr{c}; normalize the input QC dataset first')
    if sub.POS.le(0).any() or sub.POS.gt(CHROM_LENGTHS[c]).any():raise ValueError(f'Invalid GRCh38 positions on chr{c}')
    if not (sub.A1.isin(list('ACGT')) & sub.A2.isin(list('ACGT')) & sub.A1.ne(sub.A2)).all():
        raise ValueError('Input must be the QC biallelic SNP dataset; non-ACGT/equal alleles found')
    sub['SOURCE_ID']=sub.ID
    sub.ID=sub.CHR+':'+sub.POS.astype(str)+':'+sub.A1+':'+sub.A2
    outprefix=Path(outprefix);outprefix.parent.mkdir(parents=True,exist_ok=True)
    table(outprefix.parent/'marker_id_map.tsv.gz',sub[['CHR','POS','SOURCE_ID','ID','A1','A2']])
    sub[['CHR','ID','CM','POS','A1','A2']].to_csv(str(outprefix)+'.bim',sep='\t',index=False,header=False)
    # Prevent outcome-dependent PLINK defaults in HWE/frequency preparation.
    fam.PHENO_FAM='-9';fam.to_csv(str(outprefix)+'.fam',sep='\t',index=False,header=False)
    width=(len(fam)+3)//4
    bed=np.memmap(cfg['plink_prefix']+'.bed',dtype=np.uint8,mode='r',offset=3,shape=(len(bim),width))
    with open(str(outprefix)+'.bed','wb') as f:
        f.write(b'\x6c\x1b\x01')
        for k in range(0,len(idx),50000): f.write(np.asarray(bed[idx[k:k+50000]]).tobytes())
    return fam,sub

def merged_length(intervals):
    total=0; start=end=None
    for s,e in sorted(intervals):
        s=int(s);e=int(e)
        if start is None:start,end=s,e
        elif s<=end+1:end=max(end,e)
        else:total+=end-start+1;start,end=s,e
    if start is not None:total+=end-start+1
    return total
