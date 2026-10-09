from __future__ import annotations
import math,time
from pathlib import Path
import numpy as np
import pandas as pd
from ancestry import collapse_ancestry
from common import table,atomic_json,read_table,canonical
from rfmix_utils import MSPData,safe_ancestry_label
from firth import fit,FitFailure,bh,safe_exp

ASSOC_COLS=['REGION_ID','CHR','CONS_START','CONS_END','MODEL','ANCESTRY','TESTED_TERM','FAMILY','STATUS','DETAIL','N','CASES','CONTROLS','CARRIERS','CARRIER_CASES','CARRIER_CONTROLS','NONCARRIERS','WEIGHTED_CARRIERS','CARRIER_LA_MIN','CARRIER_LA_MAX','LA1_EXTRAPOLATION','BETA','SE','OR','CI_LOW','CI_HIGH','P_VALUE','Q_VALUE','BONFERRONI_P','CONVERGED','ITERATIONS','MAX_SCORE','NULL_MAX_SCORE','CI_METHOD','P_METHOD','COVARIATES_USED','DROPPED_CONSTANTS','NOMINAL_ASSOCIATED','FDR_ASSOCIATED']

def one_model(base,cfg,model,ancestry='',global_cols=(),target='ROH',extra=()):
    a=cfg['association'];t=time.monotonic()
    cols=['AD',target]+list(extra)
    covars=[] if model=='M0' else list(a['covariates'])
    covars+=list(global_cols)
    # Stable unique order, preserving target as column 1.
    columns=list(dict.fromkeys([target]+list(extra)+covars))
    d=base.dropna(subset=list(dict.fromkeys(cols+covars))).copy()
    row={'MODEL':model,'ANCESTRY':ancestry,'TESTED_TERM':target,'STATUS':'pending','DETAIL':'','N':len(d),
    'CASES':int(d.AD.eq(1).sum()),'CONTROLS':int(d.AD.eq(0).sum()),'CARRIERS':int(d.ROH.sum()),
    'CARRIER_CASES':int(((d.ROH==1)&(d.AD==1)).sum()),'CARRIER_CONTROLS':int(((d.ROH==1)&(d.AD==0)).sum()),
    'NONCARRIERS':int((d.ROH==0).sum()),'CONVERGED':0,'CI_METHOD':'Firth-Wald approximation','P_METHOD':'1-df profile penalized likelihood-ratio',
    'FAMILY':'ancestry_primary' if ancestry else ('pooled_M1' if model=='M1' else 'secondary_models')}
    if ancestry:
        carriers=d.loc[d.ROH.eq(1),'LA_TARGET']
        row.update({'WEIGHTED_CARRIERS':float(carriers.sum()),'CARRIER_LA_MIN':float(carriers.min()) if len(carriers) else np.nan,'CARRIER_LA_MAX':float(carriers.max()) if len(carriers) else np.nan,
                                                        'LA1_EXTRAPOLATION':int(len(carriers)>0 and carriers.max()<1-1e-8)})
    if not len(d):row.update(STATUS='no_complete_samples',DETAIL='No samples with all model variables; APOE models require decoded genotypes from columns.apoe_genotype');return row
    if row['CASES']==0 or row['CONTROLS']==0:row.update(STATUS='single_outcome_class');return row
    if row['CARRIERS']<a['min_carriers'] or row['NONCARRIERS']<a['min_noncarriers']:
        row.update(STATUS='insufficient_carrier_support');return row
    if ancestry and row['WEIGHTED_CARRIERS']<a['min_weighted_carriers']:
        row.update(STATUS='insufficient_ancestry_weighted_carriers');return row
    kept=[];dropped=[];arrays=[np.ones(len(d))]
    for c in columns:
        v=d[c].to_numpy(float)
        if not np.isfinite(v).all():row.update(STATUS='nonfinite_design');return row
        if np.ptp(v)<1e-12:
            if c==target:row.update(STATUS='no_tested_term_variation');return row
            dropped.append(c);continue
        # Covariate scaling leaves tested ROH / interaction coefficient on its natural scale.
        if c in covars and c not in {'SEX','APOE4'}:v=(v-v.mean())/v.std()
        arrays.append(v);kept.append(c)
    X=np.column_stack(arrays)
    row['COVARIATES_USED']=';'.join(kept);row['DROPPED_CONSTANTS']=';'.join(dropped)
    if len(d)<X.shape[1]+5:row.update(STATUS='insufficient_complete_samples');return row
    if np.linalg.matrix_rank(X)!=X.shape[1]:row.update(STATUS='rank_deficient',DETAIL='Tested contrast not identifiable with this design');return row
    try:
        j=1+kept.index(target)
        f=fit(X,d.AD.to_numpy(float),max_iter=a['max_iterations'],tol=a['score_tolerance'],max_seconds=a['max_fit_seconds'],test_index=j)
        b=float(f['beta'][j]);se=math.sqrt(float(f['vcov'][j,j]))
        row.update(STATUS='converged',CONVERGED=1,BETA=b,SE=se,OR=safe_exp(b),CI_LOW=safe_exp(b-1.959963984540054*se),CI_HIGH=safe_exp(b+1.959963984540054*se),P_VALUE=f['p_value'],ITERATIONS=f['iterations'],MAX_SCORE=f['max_score'],NULL_MAX_SCORE=f['null_max_score'])
    except (FitFailure,np.linalg.LinAlgError,ValueError,OverflowError) as e:
        row.update(STATUS='fit_failed',DETAIL=str(e))
    return row

def run_associations(cfg,regions,members,global_q):
    out=Path(cfg['output_root']);pheno=read_table(out/'inputs/phenotypes_prepared.tsv')
    q=collapse_ancestry(global_q,'Q_');q.insert(0,'IID',global_q.IID.to_numpy())
    reference=cfg['association']['global_reference'];all_q=[x for x in q if x.startswith('Q_')]
    # Explicit reference when present; deterministic last component otherwise, with names recorded.
    omit='Q_'+reference if 'Q_'+reference in all_q else all_q[-1]
    qcols=[x for x in all_q if x!=omit]
    base=pheno.merge(q,on='IID',validate='one_to_one');hc=regions[regions.HIGH_CONFIDENCE.eq(1)]
    rows=[];local_rows=[];msp=None;chrom=None
    adjustment=''.join(' + '+x for x in cfg['association']['covariates'])
    models_manifest={'project':cfg['project'],'cohort':cfg['cohort_label'],'M0':'AD ~ ROH','M1':'AD ~ ROH'+adjustment,
    'M2_APOE':'M1 + APOE4','ROHxE':'AD ~ ROH + APOE4 + ROH:APOE4'+adjustment+'; tests interaction',
    'LA':'AD ~ ROH + (LA - 1) + ROH:(LA - 1)'+adjustment+' + global ancestry K-1; tests ROH at LA=1',
    'primary_ancestry':cfg['association']['primary_ancestry'],'global_reference_used':omit,
    'multiplicity':'BH and Bonferroni separately for all ancestry tests jointly, pooled M1 tests, and all secondary tests jointly; only converged fits enter correction',
    'CI_method':'Firth-Wald approximation','P_method':'Profile penalized likelihood-ratio; constrained coefficient fit retains the full Jeffreys determinant',
    'relatedness':'No kinship/random effect. Closely related participants require a separately designed sensitivity/relatedness analysis.'}
    for r in hc.sort_values(['CHR','CONS_START']).itertuples(index=False):
        if int(r.CHR)!=chrom:
            chrom=int(r.CHR);msp=MSPData.read(out/f'chr{chrom}/ancestry/cohort.msp.tsv')
        carriers=set(members.loc[members.REGION_ID.eq(r.REGION_ID),'IID']);d=base.copy();d['ROH']=d.IID.isin(carriers).astype(int)
        common={'REGION_ID':r.REGION_ID,'CHR':chrom,'CONS_START':r.CONS_START,'CONS_END':r.CONS_END}
        for model in ['M0','M1']:
            rows.append(dict(common,**one_model(d,cfg,model)))
        if cfg['association']['run_apoe_sensitivity']:
            rows.append(dict(common,**one_model(d,cfg,'M2_APOE',extra=['APOE4'])))
            de=d.copy();de['ROHxAPOE4']=de.ROH*de.APOE4
            rows.append(dict(common,**one_model(de,cfg,'ROHxE',target='ROHxAPOE4',extra=['ROH','APOE4'])))
        la=np.array([msp.local_proportions(iid,int(r.CONS_START),int(r.CONS_END)) for iid in d.IID])
        raw=pd.DataFrame(la,columns=['LA_'+safe_ancestry_label(x) for x in msp.ancestry_names]);collapsed=collapse_ancestry(raw,'LA_')
        for anc in cfg['association']['ancestries']:
            col='LA_'+anc
            if col not in collapsed:
                rows.append(dict(common,MODEL='LA_'+anc,ANCESTRY=anc,TESTED_TERM='ROH_at_LA1',FAMILY='ancestry_primary',STATUS='ancestry_component_unavailable',DETAIL='No such ancestry in RFMix mapping',CONVERGED=0));continue
            da=d.copy();da['LA_TARGET']=collapsed[col].to_numpy();da['LA_CENTERED']=da.LA_TARGET-1;da['ROHxLA_CENTERED']=da.ROH*da.LA_CENTERED
            row=one_model(da,cfg,'LA_'+anc,anc,global_cols=qcols,extra=['LA_CENTERED','ROHxLA_CENTERED']);row['TESTED_TERM']='ROH_at_LA1'
            rows.append(dict(common,**row))
        local=collapsed.copy();local.insert(0,'IID',d.IID.to_numpy());local.insert(0,'REGION_ID',r.REGION_ID);local['ROH']=d.ROH.to_numpy();local_rows.append(local)
        if len(rows)%35==0:print(f'Association: {len(rows)} model rows completed',flush=True)
    result=pd.DataFrame(rows).reindex(columns=ASSOC_COLS)
    for family in result.FAMILY.dropna().unique():
        ix=result.index[(result.FAMILY==family)&result.CONVERGED.eq(1)]
        pv=pd.to_numeric(result.loc[ix,'P_VALUE'],errors='coerce').to_numpy(float)
        result.loc[ix,'Q_VALUE']=bh(pv);result.loc[ix,'BONFERRONI_P']=np.minimum(1.,pv*len(pv))
    result['NOMINAL_ASSOCIATED']=(result.CONVERGED.eq(1)&pd.to_numeric(result.P_VALUE,errors='coerce').lt(cfg['association']['alpha'])).astype(int)
    result['FDR_ASSOCIATED']=(result.CONVERGED.eq(1)&pd.to_numeric(result.Q_VALUE,errors='coerce').lt(cfg['association']['alpha'])).astype(int)
    table(out/'results/associations.tsv',result)
    table(out/'results/AD_associated_nominal.tsv',result[result.NOMINAL_ASSOCIATED.eq(1)])
    table(out/'results/AD_associated_FDR.tsv',result[result.FDR_ASSOCIATED.eq(1)])
    table(out/'results'/f'primary_{cfg["association"]["primary_ancestry"]}_associations.tsv',result[result.MODEL.eq('LA_'+cfg['association']['primary_ancestry'])])
    l=pd.concat(local_rows,ignore_index=True) if local_rows else pd.DataFrame(columns=['REGION_ID','IID','ROH'])
    table(out/'results/local_ancestry_for_association.tsv.gz',l)
    atomic_json(out/'results/association_models.json',models_manifest)
    return result
