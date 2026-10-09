#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, shutil, subprocess, sys, time, traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from common import *
from configuration import VERSION, analysis_config, analysis_fingerprint, guard_output
from phenotypes import prepare_phenotypes
from ancestry import subset_msp,combine_ancestry
from regional_scoring import score_regions
from association import run_associations,ASSOC_COLS

REGCOL=['REGION_ID','TOOL','LD','CHR','CONS_START','CONS_END','UNION_START','UNION_END','LENGTH_BP','N_CARRIERS']
CALLCOL=['TOOL','LD','CHR','IID','START','END','LENGTH_BP','NSNP','SOURCE']
MEMCOL=['REGION_ID','TOOL','LD','CHR','IID','SEG_START','SEG_END','SEG_LENGTH_BP','NSNP']
SCORECOL=['QH_A','QR_A','MEDIAN_HVR_RAW','MEAN_HRS_RAW','CARRIER_SUPPORT','RDI_A','HVR_VALID_FRAC_A','HVR_PASS_FRAC_A','HRS_PASS_FRAC_A','HRS_ZERO_FRAC_A','MEDIAN_Z_H_A','MEDIAN_EMP_P','HVR_META_Z_A','HVR_META_P_A','N_SEEDS','TECHNICALLY_ASSESSABLE','SOFT_PASS']

def source_signature():
    paths=list((ROOT/'src').glob('*.py'))+[ROOT/'run.sh',ROOT/'submit.sh',ROOT/'slurm/job.sbatch',ROOT/'config/environment.sh',ROOT/'VERSION']
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}

def config_signature(cfg):return fingerprint({'config':analysis_config(cfg),'lock':LOCK,'technical':TECH,'code':source_signature()})
def check_manifest(cfg):
    guard_output(cfg)
    root=Path(cfg['output_root']);p=root/'run_manifest.json'
    if not p.is_file():raise ValueError('Run preflight or submit first')
    m=json.loads(p.read_text())
    if m['run_signature']!=config_signature(cfg):raise ValueError('Configuration or source changed during the run; use a new output_root')
    for key,old in m['inputs'].items():
        if stat_signature(old['path'])!=old:raise ValueError(f'Input changed during run: {key}; use a new output_root')
    return m

def check_tools(cfg):
    versions={}
    for name in ['plink','bcftools']:
        binary=cfg['tools'][name]
        if not shutil.which(binary):raise FileNotFoundError(f'{name} executable not found: {binary}. Activate environment.yml or edit config/environment.sh')
        r=subprocess.run([binary,'--version'],capture_output=True,text=True)
        if r.returncode:raise RuntimeError(f'{name} --version failed: {r.stderr[:300]}')
        versions[name]=r.stdout.splitlines()[0]
        if name=='plink' and ('PLINK v2' in r.stdout or 'PLINK 2' in r.stdout):raise ValueError('This deployment uses PLINK 1.9, not PLINK 2')
    r=subprocess.run([cfg['tools']['bcftools'],'plugin','-l'],capture_output=True,text=True)
    if r.returncode or 'fill-tags' not in r.stdout.split():raise RuntimeError('BCFtools fill-tags plugin is required; check BCFTOOLS_PLUGINS')
    return versions

def preflight(cfg,check_external=True,config_path=None):
    guard_output(cfg)
    root=Path(cfg['output_root'])
    # A failed attempt with another configuration must not overwrite an existing run.
    if (root/'run_manifest.json').exists():check_manifest(cfg)
    guard_output(cfg,claim=True)
    try:
        return _preflight(cfg,check_external,config_path)
    except Exception as exc:
        atomic_json(root/'preflight_status.json',{'state':'failed','project':cfg['project'],'error':str(exc),'run_signature':config_signature(cfg)})
        raise

def _preflight(cfg,check_external=True,config_path=None):
    root=Path(cfg['output_root'])
    required={ext:cfg['plink_prefix']+'.'+ext for ext in ['bed','bim','fam']}
    required.update(phenotype=cfg['phenotype_file'],pca=cfg['pca_file'])
    for c in chromosomes(cfg['chromosomes']):required[f'msp_chr{c}']=render(cfg['rfmix_msp_pattern'],c)
    if cfg.get('rfmix_sample_map'):required['rfmix_sample_map']=cfg['rfmix_sample_map']
    required['gene_annotation']=cfg.get('gene_annotation_file') or str(ROOT/'resources/hg38_ucsc_refgene_genes.tsv.gz')
    missing=[str(p) for p in required.values() if not Path(p).is_file()]
    if missing:raise FileNotFoundError('Missing input files:\n'+'\n'.join(missing))
    # The selected cohort's unpruned dataset is the sole genotype source.
    name=Path(cfg['plink_prefix']).name.lower()
    if 'pruned' in name and 'unpruned' not in name:raise ValueError('LD-pruned genotype filename is incompatible with GRID_ID 1433')
    sig=config_signature(cfg)
    if (root/'run_manifest.json').exists():
        old=json.loads((root/'run_manifest.json').read_text())
        if old['run_signature']!=sig:raise ValueError('Existing run has another configuration/source; choose a new output_root')
        check_manifest(cfg)
    fam,pheno,pa=prepare_phenotypes(cfg)
    tools_info=check_tools(cfg) if check_external else {'status':'not checked (Python fixture test only)'}
    inputs={k:stat_signature(v) for k,v in required.items()}
    manifest={'package_version':VERSION,'project':cfg['project'],'cohort':cfg['cohort_label'],'config_path':str(config_path) if config_path else None,'analysis_config_sha256':analysis_fingerprint(cfg),'run_signature':sig,'locked':LOCK,'technical':TECH,'technical_N':len(fam),'locked_min_carriers':carrier_min(len(fam)),
    'discovery_min_carriers':carrier_min(len(fam),TECH['discovery_fraction']),'inputs':inputs,'software':tools_info,'python':sys.version,'source':source_signature(),
    'chromosomes':chromosomes(cfg['chromosomes']),'genome_scope':'all_autosomes' if chromosomes(cfg['chromosomes'])==list(range(1,23)) else 'chromosome_subset_run',
    'upstream_LD_provenance':'User-configured PLINK dataset. This package performs no caller-input LD pruning; upstream provenance is not inferred from the filename.',
    'allele_frequency':'Computed from GT for all configured FAM samples after VCF export; not MAF copied into AF',
    'global_ancestry':'Cohort-only bp-weighted MSP hard calls, matching v3.8.0. RFMix Q/FB files are not used by source scoring.',
    'phenotype_audit':pa,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    atomic_json(root/'run_manifest.json',manifest);atomic_json(root/'config.resolved.json',cfg)
    atomic_json(root/'preflight_status.json',{'state':'passed','project':cfg['project'],'run_signature':sig,'technical_N':len(fam),'locked_min_carriers':carrier_min(len(fam))})
    (root/'logs').mkdir(exist_ok=True)
    print(f'{cfg["cohort_label"]} N={len(fam)}; fixed minimum carriers={carrier_min(len(fam))}; preflight passed',flush=True)
    return manifest

def normalize_bcftools(path,chrom,iids,out):
    rows=[];raw_count=0
    if not Path(path).exists() or Path(path).stat().st_size==0:raise ValueError('BCFtools produced no readable raw output')
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            a=line.rstrip('\r\n').split('\t')
            if a[0]!='RG':raise ValueError(f'Unexpected BCFtools region-output row {a[0]}')
            if len(a)<8:raise ValueError('Malformed BCFtools RG row')
            iid=a[1];c=int(a[2].removeprefix('chr'));s,e,n=int(a[3]),int(a[4]),int(a[6]);raw_count+=1
            if iid not in iids:raise ValueError(f'BCFtools sample outside the configured cohort: {iid}')
            if c!=chrom or s<1 or e<s:raise ValueError('BCFtools region chromosome/coordinates invalid')
            if e-s+1<TECH['caller_length'] or n<TECH['caller_snps']:continue
            rows.append(dict(TOOL='bcftools',LD='no',CHR=c,IID=iid,START=s,END=e,LENGTH_BP=e-s+1,NSNP=n,SOURCE=str(path)))
    df=pd.DataFrame(rows,columns=CALLCOL)
    if df.duplicated(['CHR','IID','START','END']).any():raise ValueError('Duplicate BCFtools RG segments')
    df=df.sort_values(['CHR','IID','START','END']);table(out,df)
    return raw_count,len(df)

def seed_panel(cfg,prefix,log,threads):
    plink=cfg['tools']['plink'];directory=Path(prefix).parent/'seeds';directory.mkdir(exist_ok=True)
    eligible=directory/'eligible';seed=directory/'seed'
    command=[plink,'--bfile',str(prefix),'--maf',str(TECH['seed_maf']),'--geno',str(TECH['seed_geno']),'--hwe',str(TECH['seed_hwe']),'include-nonctrl',
    '--nonfounders','--write-snplist','--allow-no-sex','--threads',str(threads),'--out',str(eligible)]
    try:run(command,log)
    except RuntimeError:
        text=(Path(str(eligible)+'.log').read_text(errors='replace') if Path(str(eligible)+'.log').exists() else '')
        if not any(s in text.lower() for s in ['no variants remaining','all variants excluded']):raise
        Path(str(eligible)+'.snplist').write_text('')
    if not Path(str(eligible)+'.snplist').exists():raise RuntimeError('PLINK seed eligibility list missing')
    if Path(str(eligible)+'.snplist').stat().st_size:
        run([plink,'--bfile',prefix,'--extract',str(eligible)+'.snplist','--indep-pairwise',TECH['seed_window'],TECH['seed_step'],TECH['seed_r2'],'--nonfounders','--allow-no-sex','--threads',threads,'--out',seed],log)
    else:Path(str(seed)+'.prune.in').write_text('')
    return str(seed)+'.prune.in',str(eligible)+'.snplist'

def chromosome(cfg,chrom,force=False):
    m=check_manifest(cfg);root=Path(cfg['output_root']);d=root/f'chr{chrom}';d.mkdir(parents=True,exist_ok=True)
    statefile=d/'status.json';sig=m['run_signature']
    if statefile.exists() and not force:
        old=json.loads(statefile.read_text())
        if old.get('state')=='completed' and old.get('run_signature')==sig:
            checks=old.get('output_hashes',{})
            if checks and all((d/p).is_file() and file_hash(d/p)==h for p,h in checks.items()):
                print(f'chr{chrom}: verified completed outputs; skipped');return
    start=time.monotonic();state={'chromosome':chrom,'state':'running','stage':'input','run_signature':sig}
    atomic_json(statefile,state)
    def stage(name):state['stage']=name;atomic_json(statefile,state)
    try:
        if chrom not in chromosomes(cfg['chromosomes']):raise ValueError('Chromosome is not in configured run')
        threads=int(os.environ.get('SLURM_CPUS_PER_TASK',cfg['resources']['cpus']))
        prefix=d/'input/cohort';log=root/f'logs/chr{chrom}.commands.log'
        fam,bim=split_bed(cfg,chrom,prefix);state['variants']=len(bim);state['samples']=len(fam)
        stage('ancestry');state['ancestry']=subset_msp(cfg,chrom,fam,d/'ancestry/cohort.msp.tsv')
        stage('HRS_seed_panel');seed,eligible=seed_panel(cfg,prefix,log,threads)
        stage('bcftools_calling');raw=d/'bcftools.roh.txt';calls=d/'calls.tsv'
        if len(bim)<TECH['caller_snps']:
            raw.write_text('# No callable ROH: fewer than 15 genotype markers.\n');table(calls,pd.DataFrame(columns=CALLCOL));raw_n=call_n=0
        else:
            export=d/'input/export';vcf=d/'input/cohort.AF.vcf.gz'
            run([cfg['tools']['plink'],'--bfile',prefix,'--recode','vcf-iid','bgz','--keep-allele-order','--allow-no-sex','--threads',threads,'--out',export],log)
            run([cfg['tools']['bcftools'],'+fill-tags',str(export)+'.vcf.gz','-Oz','-o',vcf,'--','-t','AF,AC,AN'],log)
            run([cfg['tools']['bcftools'],'index','-f','-t',vcf],log)
            ids=subprocess.run([cfg['tools']['bcftools'],'query','-l',str(vcf)],text=True,capture_output=True,check=True).stdout.splitlines()
            if len(ids)!=len(fam) or set(ids)!=set(fam.IID):raise ValueError('VCF sample IDs differ from configured cohort FAM')
            run([cfg['tools']['bcftools'],'roh','-G',TECH['gt_error'],'--AF-tag','AF','--rec-rate',TECH['rec_rate'],'-O','r','-o',raw,vcf],log)
            raw_n,call_n=normalize_bcftools(raw,chrom,set(fam.IID),calls)
        state['raw_RG_rows']=raw_n;state['normalized_ROH_calls']=call_n
        stage('candidate_construction')
        run([sys.executable,ROOT/'src/build_candidates.py','--tool','bcftools','--ld','no','--chr',chrom,'--calls',calls,'--msp',d/'ancestry/cohort.msp.tsv','--fam',str(prefix)+'.fam',
                            '--regions',d/'regions.tsv','--members',d/'members.tsv','--length-floor',TECH['caller_length'],'--min-carrier-fraction',TECH['discovery_fraction'],'--min-carriers',TECH['min_carriers'],
                            '--merge-gap',TECH['merge_gap'],'--merge-tv',TECH['merge_tv'],'--reciprocal-overlap',TECH['overlap']],log)
        stage('HVR_HRS')
        run([sys.executable,ROOT/'src/compute_raw_metrics.py','--tool','bcftools','--ld','no','--chr',chrom,'--regions',d/'regions.tsv','--members',d/'members.tsv','--bed-prefix',prefix,
                            '--original-bim',str(prefix)+'.bim','--seed-pruned',seed,'--seed-eligible',eligible,'--individual-output',d/'individual_raw.tsv','--regional-output',d/'regional_raw.tsv',
                            '--backgrounds',TECH['backgrounds'],'--proposal-multiplier',TECH['proposal_multiplier'],'--buffer-bp',TECH['buffer_bp'],'--site-ratio-min',TECH['site_ratio_min'],'--site-ratio-max',TECH['site_ratio_max'],
                            '--min-sites',TECH['min_sites'],'--min-ess',TECH['min_ess'],'--individual-min-z',TECH['individual_min_z'],'--individual-max-p',TECH['individual_max_p'],
                            '--hrs-min',TECH['hrs_min'],'--chunk-variants',TECH['chunk_variants'],'--seed',TECH['seed'],'--threads',threads],log)
        state['candidate_regions']=len(read_table(d/'regions.tsv'))
        outputs=['calls.tsv','regions.tsv','members.tsv','individual_raw.tsv','regional_raw.tsv','ancestry/cohort.msp.tsv','ancestry/global_ancestry_bp.tsv','ancestry/ancestry_components.tsv']
        state['output_hashes']={p:file_hash(d/p) for p in outputs}
        state.update(state='completed',stage='completed',seconds=round(time.monotonic()-start,2),detail='success_zero_candidates' if not state['candidate_regions'] else 'success_with_candidates')
        if not cfg['keep_intermediate_vcf']:
            for p in (d/'input').glob('*.vcf.gz*'):p.unlink()
        atomic_json(statefile,state);print(f'chr{chrom}: {state["candidate_regions"]} candidates; completed',flush=True)
    except Exception as e:
        state.update(state='failed',error=str(e),seconds=round(time.monotonic()-start,2));atomic_json(statefile,state);raise

def concat_chr(cfg,name):
    frames=[read_table(Path(cfg['output_root'])/f'chr{c}'/name) for c in chromosomes(cfg['chromosomes'])]
    nonempty=[x for x in frames if not x.empty]
    return pd.concat(nonempty,ignore_index=True) if nonempty else frames[0].iloc[:0].copy()

def apply_locked(scored,n):
    out=scored.copy();out['N_SAMPLES']=n;out['MIN_CARRIERS_LOCKED']=carrier_min(n);out['GRID_ID']=1433
    gates={'PASS_TECHNICAL':out.TECHNICALLY_ASSESSABLE.eq(1),'PASS_LENGTH':out.LENGTH_BP.ge(LOCK['LENGTH_BP_MIN']),
                                'PASS_CARRIERS':out.N_CARRIERS.ge(carrier_min(n)),'PASS_HVR':out.MEDIAN_HVR_RAW.le(LOCK['HVR_MAX']),
                                'PASS_HRS':out.MEAN_HRS_RAW.ge(LOCK['HRS_MIN']),'PASS_RDI':out.RDI_A.ge(LOCK['RDI_MIN'])}
    for k,v in gates.items():out[k]=v.fillna(False).astype(int)
    out['HIGH_CONFIDENCE']=out[list(gates)].all(axis=1).astype(int)
    out['STATUS']=['high_confidence' if flag else ';'.join(k.removeprefix('PASS_').lower() for k in gates if not out.loc[i,k]) for i,flag in out.HIGH_CONFIDENCE.items()]
    return out

def annotate(cfg,scored):
    p=Path(cfg['gene_annotation_file']) if cfg.get('gene_annotation_file') else ROOT/'resources/hg38_ucsc_refgene_genes.tsv.gz'
    genes=pd.read_csv(p,sep='\t',dtype={'CHR':str})
    if not {'CHR','BP1','BP2','gene'}.issubset(genes):raise ValueError('Gene annotation must have CHR, BP1, BP2, gene; 1-based closed coordinates')
    genes['CHR']=genes.CHR.str.replace('chr','',regex=False);out=[]
    for r in scored.itertuples(index=False):
        g=genes[(genes.CHR.eq(str(r.CHR)))&(genes.BP1.le(r.CONS_END))&(genes.BP2.ge(r.CONS_START))]
        out.append(';'.join(sorted(g.gene.dropna().astype(str).unique())) or 'NA')
    scored=scored.copy();scored['GENES']=out;return scored

def score(cfg):
    manifest=check_manifest(cfg);root=Path(cfg['output_root']);chroms=chromosomes(cfg['chromosomes'])
    for c in chroms:
        d=root/f'chr{c}';p=d/'status.json'
        if not p.exists():raise RuntimeError(f'chr{c} has not run; final scoring requires all configured chromosomes')
        state=json.loads(p.read_text())
        if state.get('state')!='completed' or state.get('run_signature')!=manifest['run_signature']:raise RuntimeError(f'chr{c} is not complete for this configuration')
        for rel,h in state.get('output_hashes',{}).items():
            if not (d/rel).exists() or file_hash(d/rel)!=h:raise RuntimeError(f'chr{c} output changed or missing: {rel}')
    fam=read_fam(cfg['plink_prefix']);regions=concat_chr(cfg,'regions.tsv');members=concat_chr(cfg,'members.tsv');indiv=concat_chr(cfg,'individual_raw.tsv')
    ancestry=combine_ancestry(cfg,fam,chroms);table(root/'results/global_ancestry.tsv',ancestry)
    if regions.REGION_ID.duplicated().any():raise ValueError('Duplicate region IDs')
    if regions.empty:
        scored=regions.reindex(columns=REGCOL+SCORECOL);scored['TOOL']='bcftools';scored['LD']='no';features=[]
    else:
        counts=members.groupby('REGION_ID').IID.nunique()
        if members.duplicated(['REGION_ID','IID']).any() or not np.array_equal(regions.N_CARRIERS.to_numpy(int),counts.reindex(regions.REGION_ID).to_numpy(int)):
            raise ValueError('Region carrier count disagrees with unique cohort membership')
        if len(indiv)!=len(members) or indiv.duplicated(['REGION_ID','IID']).any():raise ValueError('Individual metric/membership inconsistency')
        values,indiv,features=score_regions(regions,indiv,ancestry,SimpleNamespace(**TECH),carrier_min(len(fam),TECH['discovery_fraction']))
        raw=concat_chr(cfg,'regional_raw.tsv').drop(columns=['N_CARRIERS','N_SEEDS'],errors='ignore')
        scored=regions.merge(raw,on=['REGION_ID','TOOL','LD','CHR'],validate='one_to_one').merge(values,on='REGION_ID',validate='one_to_one')
    scored=annotate(cfg,apply_locked(scored,len(fam)))
    table(root/'results/regions_scored.tsv',scored);table(root/'results/high_confidence_ROH.tsv',scored[scored.HIGH_CONFIDENCE.eq(1)])
    table(root/'results/individual_scored.tsv.gz',indiv);table(root/'results/members.tsv.gz',members)
    table(root/'results/locked_thresholds.tsv',pd.DataFrame([dict(LOCK,N_SAMPLES=len(fam),N_CARRIERS_MIN=carrier_min(len(fam)),THRESHOLD_SOURCE='locked simulation deployment refit; no real-data tuning')]))
    fallback={c:int(indiv[c].sum()) for c in indiv if c.endswith('_LOCO_FALLBACK')}
    atomic_json(root/'results/scoring_audit.json',{'candidate_regions':len(scored),'high_confidence_regions':int(scored.HIGH_CONFIDENCE.sum()),'technical_N':len(fam),'LOCO_features':features,'LOCO_fallback_rows':fallback,'all_gates_required':True,'no_new_knee':True})
    burden(cfg,fam,scored,members)
    return scored,members,ancestry

def burden(cfg,fam,scored,members):
    calls=concat_chr(cfg,'calls.tsv');raw=calls[calls.LENGTH_BP.ge(LOCK['LENGTH_BP_MIN'])];hcids=set(scored.loc[scored.HIGH_CONFIDENCE.eq(1),'REGION_ID']);hc=members[members.REGION_ID.isin(hcids)]
    dnom=sum(CHROM_LENGTHS[c] for c in chromosomes(cfg['chromosomes']));rows=[]
    raw_group={k:g for k,g in raw.groupby('IID')};hc_group={k:g for k,g in hc.groupby('IID')}
    for f in fam.itertuples(index=False):
        a=raw_group.get(f.IID);b=hc_group.get(f.IID)
        def bp(df,s,e):
            if df is None:return 0
            return sum(merged_length(zip(g[s],g[e])) for _,g in df.groupby('CHR'))
        total=bp(a,'START','END');conf=bp(b,'SEG_START','SEG_END')
        rows.append({'FID':f.FID,'IID':f.IID,'N_RAW_ROH_GE_1MB':0 if a is None else len(a),'RAW_ROH_BP':total,'N_HC_REGIONS':0 if b is None else b.REGION_ID.nunique(),'HC_MEMBER_SEGMENT_BP':conf,'DENOMINATOR_REFERENCE_BP':dnom,'FROH_RAW_GE_1MB':total/dnom,'FROH_HC_MEMBER_SEGMENTS':conf/dnom})
    table(Path(cfg['output_root'])/'results/individual_ROH_burden.tsv',pd.DataFrame(rows))

def finalize(cfg):
    check_manifest(cfg)
    root=Path(cfg['output_root']);error=None
    try:
        regions,members,ancestry=score(cfg)
        if cfg['association']['enabled']:
            run_associations(cfg,regions,members,ancestry)
        else:
            table(root/'results/associations.tsv',pd.DataFrame(columns=ASSOC_COLS))
        atomic_json(root/'final_status.json',{'state':'completed','association_enabled':cfg['association']['enabled'],'run_signature':config_signature(cfg)})
    except Exception as e:
        error=e;atomic_json(root/'final_status.json',{'state':'incomplete','error':str(e),'run_signature':config_signature(cfg)})
    finally:
        try:
            from report import make_report
            make_report(cfg)
        except Exception as report_error:
            if error is None:raise
            print('Diagnostic report error: '+str(report_error),file=sys.stderr)
    if error:raise error

def submit(cfg,dry_run=False,config_path=None):
    m=preflight(cfg,config_path=config_path);root=Path(cfg['output_root']);res=cfg['resources'];chroms=chromosomes(cfg['chromosomes'])
    if not dry_run and not shutil.which('sbatch'):raise FileNotFoundError('sbatch not found; use run.sh --config <your-config.json> local on an allocated compute node')
    submission_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    directory=root/'submissions'/submission_id;directory.mkdir(parents=True,exist_ok=False)
    snap=directory/'config.resolved.json';atomic_json(snap,cfg);digest=file_hash(snap);snap.chmod(0o444)
    opts=['--parsable','--cpus-per-task',str(res['cpus']),'--mem',res['memory'],'--time',res['time']]
    for k in ['account','partition','constraint']:
        if res.get(k):opts+=['--'+k,res[k]]
    array=','.join(map(str,chroms))+('%'+str(res['max_concurrent']) if res['max_concurrent'] else '')
    job_name='TRUROH_'+cfg['project']+'_1433'
    command=['sbatch']+opts+['--job-name',job_name,'--array',array,'--output',str(root/'logs/chr_%A_%a.out'),'--error',str(root/'logs/chr_%A_%a.err'),str(ROOT/'slurm/job.sbatch'),str(ROOT),str(snap),'chromosome',digest]
    job='ARRAY_JOB_ID'
    if not dry_run:
        r=subprocess.run(command,text=True,capture_output=True,check=True);job=r.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Unexpected sbatch job ID: '+r.stdout)
    final=['sbatch','--parsable','--dependency','afterany:'+job,'--job-name',job_name+'_report','--cpus-per-task',str(res['final_cpus']),'--mem',res['final_memory'],'--time',res['final_time'],'--output',str(root/'logs/final_%j.out'),'--error',str(root/'logs/final_%j.err')]
    for k in ['account','partition']:
        if res.get(k):final+=['--'+k,res[k]]
    constraint=res.get('final_constraint') or res.get('constraint')
    if constraint:final+=['--constraint',constraint]
    final += [str(ROOT/'slurm/job.sbatch'),str(ROOT),str(snap),'finalize',digest]
    record={'project':cfg['project'],'source_config':str(config_path) if config_path else None,'submission_id':submission_id,'frozen_config':str(snap),'config_sha256':digest,'run_signature':m['run_signature'],'dry_run':dry_run,'array_job_id':None if dry_run else job,'array_command':command,'final_command':final,'final_job_id':None}
    def save_record():
        atomic_json(directory/'submission.json',record)
        atomic_json(root/('submission.plan.json' if dry_run else 'submission.json'),record)
    save_record()
    if dry_run:
        import shlex
        print('Dry run: preflight passed; no SLURM jobs submitted. Frozen config: '+str(snap))
        print(shlex.join(command));print(shlex.join(final));return record
    try:
        r=subprocess.run(final,text=True,capture_output=True,check=True);fin=r.stdout.strip().split(';')[0]
        if not fin.isdigit():raise RuntimeError('Unexpected finalizer job ID')
    except Exception as exc:
        record['error']=str(exc);save_record()
        import shlex
        print('Array was submitted. Finalizer submission failed; after it completes run '+shlex.join(['bash',str(ROOT/'run.sh'),'--config',str(snap),'finalize']),file=sys.stderr);raise
    record['final_job_id']=fin;save_record()
    print(f'Array job {job}; report job {fin} (afterany); output {root}')
    return record

def status(cfg):
    root=Path(cfg['output_root']);rows=[]
    for c in chromosomes(cfg['chromosomes']):
        p=root/f'chr{c}/status.json';d=json.loads(p.read_text()) if p.exists() else {'state':'not_run'}
        rows.append({'CHR':c,'STATE':d.get('state'),'STAGE':d.get('stage'),'CANDIDATES':d.get('candidate_regions'),'ERROR':d.get('error','')})
    print(pd.DataFrame(rows).to_string(index=False))

def main():
    ap=argparse.ArgumentParser(description='User-configured cohorts; locked BCFtools/no-LD GRID_ID 1433. No default cohort.')
    ap.add_argument('--config',type=Path,help='JSON for this cohort (or set TRUROH_CONFIG); required for every entry point')
    ap.add_argument('--force',action='store_true');ap.add_argument('--dry-run',action='store_true',help='submit only: run preflight and print SLURM commands without submitting')
    ap.add_argument('--expected-config-sha256',help=argparse.SUPPRESS)
    ap.add_argument('action',choices=['config','preflight','submit','chromosome','score','associate','finalize','report','status','local'])
    ap.add_argument('chrom',type=int,nargs='?');args=ap.parse_intermixed_args()
    selected=args.config or os.environ.get('TRUROH_CONFIG')
    if not selected:ap.error('Select a cohort with --config config/ch.json (or TRUROH_CONFIG). No config is selected by default.')
    if args.dry_run and args.action!='submit':ap.error('--dry-run is only valid with submit')
    if args.chrom is not None and args.action!='chromosome':ap.error('A chromosome number is only valid with chromosome')
    config_path=Path(selected).expanduser().resolve();cfg=None;owned=False
    try:
        if args.expected_config_sha256 and file_hash(config_path)!=args.expected_config_sha256:
            raise ValueError('Frozen configuration checksum mismatch; no inputs or outputs were touched')
        cfg=load_config(config_path)
        print(f'[TRUROH {VERSION}] project={cfg["project"]}; config={config_path}; output={cfg["output_root"]}',file=sys.stderr,flush=True)
        if args.action=='config':
            print(json.dumps(cfg,indent=2));return 0
        guard_output(cfg)
        if (Path(cfg['output_root'])/'run_manifest.json').exists():check_manifest(cfg)
        guard_output(cfg,claim=args.action!='status');owned=True
        if args.action=='preflight':preflight(cfg,config_path=config_path)
        elif args.action=='submit':submit(cfg,args.dry_run,config_path)
        elif args.action=='chromosome':chromosome(cfg,args.chrom or int(os.environ['SLURM_ARRAY_TASK_ID']),args.force)
        elif args.action=='score':score(cfg)
        elif args.action=='associate':
            root=Path(cfg['output_root']);check_manifest(cfg)
            run_associations(cfg,read_table(root/'results/regions_scored.tsv'),read_table(root/'results/members.tsv.gz'),read_table(root/'results/global_ancestry.tsv'))
        elif args.action=='finalize':finalize(cfg)
        elif args.action=='report':
            from report import make_report
            make_report(cfg)
        elif args.action=='status':status(cfg)
        elif args.action=='local':
            preflight(cfg,config_path=config_path)
            for c in chromosomes(cfg['chromosomes']):
                try:chromosome(cfg,c,args.force)
                except Exception as e:print(f'chr{c} failed: {e}',file=sys.stderr,flush=True)
            finalize(cfg)
    except Exception as e:
        print(f'ERROR: {e}',file=sys.stderr)
        if cfg is not None and owned and args.action in {'preflight','submit','local'}:
            root=Path(cfg['output_root'])
            atomic_json(root/'last_error.json',{'action':args.action,'project':cfg['project'],'error':str(e)})
            try:
                from report import make_report
                make_report(cfg)
            except Exception as report_error:print('Diagnostic report error: '+str(report_error),file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':sys.exit(main())
