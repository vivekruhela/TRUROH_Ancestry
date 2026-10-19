from __future__ import annotations
import csv, re
from pathlib import Path
import numpy as np
import pandas as pd
from rfmix_utils import MSPData, _iid_from_hap, safe_ancestry_label
from common import table, render, canonical, atomic_json, read_table

PRIMARY_CLOSE_MARGIN = 0.05   # top-two mean global ancestry closer than this -> "auto" choice is flagged as a close call


def resolve_primary_ancestry(cfg, write=True):
    """Primary ancestry for the association tables.

    Explicit config value: used as given (a pre-specified hypothesis); a note is recorded when it is not the
    cohort-majority ancestry. "auto": the configured ancestry with the highest mean global ancestry over the
    cohort FAM samples, from results/global_ancestry.tsv (RFMix MSP-derived), after collapsing source aliases
    (e.g. AFR + Africa). Outcome-free. Returns (ancestry, info); info is written to
    results/primary_ancestry_selection.json.
    """
    a = cfg['association']; configured = a['primary_ancestry']; root = Path(cfg['output_root'])
    path = root / 'results/global_ancestry.tsv'
    info = {'configured': configured, 'method': 'auto' if configured == 'auto' else 'explicit',
            'candidates': list(a['ancestries']), 'source': str(path), 'statistic': 'mean global ancestry over cohort FAM samples',
            'close_margin': PRIMARY_CLOSE_MARGIN, 'notes': []}
    if not path.exists():
        if configured == 'auto':
            raise ValueError('primary_ancestry "auto" needs results/global_ancestry.tsv (run scoring first)')
        info.update(selected=configured, mean_global_ancestry=None, majority=None)
        return configured, info
    q = read_table(path); collapsed = collapse_ancestry(q, 'Q_')
    means = {c[2:]: float(collapsed[c].mean()) for c in collapsed}
    ranked = sorted(((means[x], x) for x in a['ancestries'] if x in means), reverse=True)
    if not ranked:
        raise ValueError(f'None of association.ancestries {a["ancestries"]} is present in the RFMix ancestry labels {sorted(means)}')
    majority = ranked[0][1]; margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else None
    selected = majority if configured == 'auto' else configured
    if configured == 'auto' and margin is not None and margin < PRIMARY_CLOSE_MARGIN:
        info['notes'].append(f'close call: {ranked[0][1]} {ranked[0][0]:.3f} vs {ranked[1][1]} {ranked[1][0]:.3f}; '
                             'consider setting primary_ancestry explicitly')
    if configured != 'auto' and configured != majority:
        info['notes'].append(f'explicit primary_ancestry {configured} ({means.get(configured, float("nan")):.3f}) is not the '
                             f'cohort-majority ancestry {majority} ({ranked[0][0]:.3f}); kept as the pre-specified hypothesis')
    info.update(selected=selected, majority=majority, top_two_margin=None if margin is None else round(margin, 4),
                mean_global_ancestry={k: round(v, 4) for k, v in sorted(means.items(), key=lambda kv: -kv[1])})
    for note in info['notes']:
        print('NOTE primary_ancestry: ' + note, flush=True)
    if write:
        atomic_json(root / 'results/primary_ancestry_selection.json', info)
    return selected, info

def subset_msp(cfg,chrom,fam,dest):
    source=Path(render(cfg['rfmix_msp_pattern'],chrom))
    if not source.is_file():raise FileNotFoundError(f'RFMix MSP missing: {source}')
    desired=fam.IID.tolist(); names={iid:iid for iid in desired}
    if cfg.get('rfmix_sample_map'):
        m=pd.read_csv(cfg['rfmix_sample_map'],sep='\t',dtype=str,keep_default_na=False)
        if not {'IID','RFMIX_IID'}.issubset(m):raise ValueError('Ancestry sample map must have IID and RFMIX_IID columns')
        if m.IID.duplicated().any() or m.RFMIX_IID.duplicated().any():raise ValueError('Ancestry ID map must be one-to-one')
        names=dict(zip(m.IID,m.RFMIX_IID))
        if set(desired)-set(names):raise ValueError('Ancestry map is missing cohort IIDs')
    required={names[x]:x for x in desired}
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    mapping_line=None; keep=None; rows=0; excluded=set(); last_start=last_end=None
    with source.open() as src,dest.open('w') as out:
        for line in src:
            if line.startswith('#Subpopulation order/codes:'):
                mapping_line=line;out.write(line);continue
            if line.startswith('#chm') or line.startswith('chm\t'):
                h=line.lstrip('#').rstrip('\r\n').split('\t')
                norm={re.sub(r'\s+',' ',x.strip().lower()):i for i,x in enumerate(h)}
                if not {'chm','spos','epos','n snps'}.issubset(norm):raise ValueError('Malformed MSP metadata header')
                nmeta=norm['n snps']+1
                groups={}
                for i,col in enumerate(h[nmeta:],nmeta): groups.setdefault(_iid_from_hap(col),[]).append(i)
                missing=set(required)-set(groups)
                if missing:raise ValueError(f'chr{chrom}: {len(missing)} cohort IDs absent in MSP; first 10: {sorted(missing)[:10]}. No ancestry imputation is performed.')
                excluded=set(groups)-set(required);keep=[];new=h[:nmeta]
                for iid in desired:
                    indices=groups[names[iid]]
                    if len(indices)!=2:raise ValueError(f'MSP must have exactly two haplotypes for {iid}')
                    keep.extend(indices);new.extend([iid+'.0',iid+'.1'])
                out.write('#'+'\t'.join(new)+'\n');continue
            if line.startswith('#') or not line.strip():continue
            if keep is None or mapping_line is None:raise ValueError('MSP ancestry code mapping/header is missing')
            a=line.rstrip('\r\n').split('\t')
            if len(a)!=len(h):raise ValueError('MSP row has incorrect field count')
            actual=re.sub(r'^chr','',a[norm['chm']],flags=re.I)
            if actual!=str(chrom):raise ValueError(f'Expected chr{chrom} MSP, found {actual}')
            s,e=int(a[norm['spos']]),int(a[norm['epos']])
            if s<1 or e<s or (last_start is not None and (s<last_start or s<last_end)):
                raise ValueError('MSP intervals must be ordered and nonoverlapping (shared endpoints accepted as in source)')
            last_start,last_end=s,e
            out.write('\t'.join(a[:nmeta]+[a[i] for i in keep])+'\n');rows+=1
    if not rows:raise ValueError('No MSP tracts')
    msp=MSPData.read(dest)
    labels=[safe_ancestry_label(a) for a in msp.ancestry_names]
    if len(set(labels))!=len(labels) or len(labels)<2:raise ValueError('Need at least two unique ancestry component labels')
    codes=set(msp.ancestry_codes)
    for iid,(a,b) in msp.sample_haps.items():
        if (set(a)|set(b))-codes:raise ValueError(f'Unknown/missing MSP ancestry codes for {iid}')
    gs=msp.global_sums(); data=[]
    for iid in desired:
        row={'IID':iid};row.update({f'BP_{a}':v for a,v in zip(labels,gs[iid])});data.append(row)
    table(dest.parent/'global_ancestry_bp.tsv',pd.DataFrame(data))
    table(dest.parent/'ancestry_components.tsv',pd.DataFrame({'CODE':msp.ancestry_codes,'SOURCE_LABEL':msp.ancestry_names,'LABEL':labels,'PLOT_LABEL':[canonical(a) for a in labels]}))
    return {'source':str(source),'retained_cohort_samples':len(desired),'excluded_other_ids':len(excluded),'tracts':rows,'components':labels}

def combine_ancestry(cfg,fam,chroms):
    total=None;labels=None
    for c in chroms:
        f=pd.read_csv(Path(cfg['output_root'])/f'chr{c}/ancestry/global_ancestry_bp.tsv',sep='\t',converters={'IID':str})
        cols=[x for x in f if x.startswith('BP_')]
        if labels is None:labels=cols
        if set(cols)!=set(labels):raise ValueError('RFMix ancestry labels differ across chromosomes')
        if set(f.IID)!=set(fam.IID) or f.IID.duplicated().any():raise ValueError('Cohort ancestry sample mismatch across chromosomes')
        a=f.set_index('IID').reindex(fam.IID)[labels].to_numpy(float)
        total=a if total is None else total+a
    if np.any(total.sum(axis=1)<=0) or not np.isfinite(total).all():raise ValueError('Invalid global ancestry coverage')
    q=total/total.sum(axis=1)[:,None]
    out=pd.DataFrame(q,columns=[x.replace('BP_','Q_',1) for x in labels]);out.insert(0,'IID',fam.IID.to_numpy())
    return out

def collapse_ancestry(df,prefix):
    cols=[x for x in df if x.startswith(prefix)]
    out=pd.DataFrame(index=df.index)
    for col in cols:
        key=canonical(col[len(prefix):]);new=prefix+key
        values=pd.to_numeric(df[col],errors='coerce')
        out[new]=out[new]+values if new in out else values
    return out
