from __future__ import annotations
import csv,re
from pathlib import Path
import numpy as np
import pandas as pd
from common import table,atomic_json,read_fam
from configuration import normalize_covariates

def eigenvectors(path):
    with open(path) as f:
        first=next((line.strip() for line in f if line.strip()),'').split()
    if len(first)<7:raise ValueError('PCA file needs FID, IID and at least PC1-PC5')
    header=first[0].lstrip('#').upper() in {'FID','FAM'} and first[1].lstrip('#').upper() in {'IID','ID'}
    if header:
        p=pd.read_csv(path,sep=r'\s+',dtype=str,keep_default_na=False)
        p=p.rename(columns={p.columns[0]:'FID',p.columns[1]:'IID'})
    else:
        p=pd.read_csv(path,sep=r'\s+',header=None,dtype=str,keep_default_na=False)
        p.columns=['FID','IID']+[f'PC{i}' for i in range(1,p.shape[1]-1)]
    if not {'PC1','PC2','PC3','PC4','PC5'}.issubset(p):raise ValueError('PCA header lacks PC1-PC5')
    if p.IID.duplicated().any():raise ValueError('Duplicate PCA IID')
    for col in [f'PC{i}' for i in range(1,6)]:p[col]=pd.to_numeric(p[col],errors='coerce')
    return p[['FID','IID','PC1','PC2','PC3','PC4','PC5']]

def token(s):
    s=str(s).strip()
    if re.fullmatch(r'-?\d+\.0+',s):s=s.split('.')[0]
    return s

def apoe4(s):
    s=token(s).upper().replace('Ε','E').replace('Ɛ','E').replace('ε','E')
    s=s.replace('APOE','').replace('E','').replace('*','').replace(' ','')
    # Genotype strings only. A single digit (dosage or opaque code) is ambiguous.
    m=re.fullmatch(r'([234])(?:[/|,;:_-]?)([234])',s)
    return sum(x=='4' for x in m.groups()) if m else np.nan

def prepare_phenotypes(cfg):
    out=Path(cfg['output_root'])/'inputs';out.mkdir(parents=True,exist_ok=True)
    fam=read_fam(cfg['plink_prefix']);col=cfg['columns']
    with open(cfg['phenotype_file'],newline='') as f:
        header=next(csv.reader(f,delimiter='\t'))
    if len(header)==1:raise ValueError('The configured phenotype_file must be TAB-separated; spaces in column names are preserved')
    if len(set(header))!=len(header):raise ValueError('Duplicate phenotype column names')
    p=pd.read_csv(cfg['phenotype_file'],sep='\t',dtype=str,keep_default_na=False)
    required=[col[x] for x in ['fid','iid','outcome','age','sex']]+list(col.get('covariates',{}).values())
    if not set(required).issubset(p):raise ValueError(f'Missing phenotype columns: {sorted(set(required)-set(p))}')
    p=p.rename(columns={col['fid']:'FID',col['iid']:'IID'})
    if p.IID.duplicated().any():raise ValueError('Duplicate phenotype IDs; no row is chosen arbitrarily')
    for k in ['FID','IID']:p[k]=p[k].str.strip()
    pc=eigenvectors(cfg['pca_file'])
    # Compare pairs for shared IIDs; no FID=IID assumption or fuzzy matching.
    for label,d in [('phenotype',p),('PCA',pc)]:
        pairs=fam[['FID','IID']].merge(d[['FID','IID']],on='IID',suffixes=('_fam','_other'))
        bad=pairs.FID_fam.ne(pairs.FID_other)
        if bad.any():raise ValueError(f'{label} FID/IID pair mismatch for {int(bad.sum())} cohort samples; first IIDs: {pairs.loc[bad,"IID"].head(5).tolist()}')
    extra_pheno=int((~p.IID.isin(fam.IID)).sum());extra_pca=int((~pc.IID.isin(fam.IID)).sum())
    p=p[p.IID.isin(fam.IID)].copy()
    raw= fam[['FID','IID','SEX_FAM']].merge(p,on=['FID','IID'],how='left',validate='one_to_one')
    missing={token(x) for x in cfg['missing_values']}
    def norm_values(series):return series.fillna('').map(token)
    outcome=norm_values(raw[col['outcome']]);mapping={token(k):int(v) for k,v in cfg['outcome_map'].items()}
    if set(mapping.values())!={0,1}:raise ValueError('outcome_map must explicitly encode controls=0 and cases=1')
    unknown=set(outcome)-missing-set(mapping)
    atomic_json(out/'outcome_coding_audit.json',{'column':col['outcome'],'mapping':mapping,'observed_counts':outcome.value_counts().to_dict(),'unmapped_values':sorted(unknown)})
    if unknown:raise ValueError(f'Unmapped {col["outcome"]} values: {sorted(unknown)}. Set outcome_map explicitly; coding is never guessed.')
    prepared=raw[['FID','IID']].copy();prepared['AD']=outcome.map(mapping)
    prepared['AGE']=pd.to_numeric(raw[col['age']].where(~raw[col['age']].fillna('').isin(missing)),errors='coerce')
    prepared.loc[prepared.AGE.le(0),'AGE']=np.nan
    sex=norm_values(raw[col['sex']]);sexmapping={token(k):int(v) for k,v in cfg['sex_map'].items()}
    sexunknown=set(sex)-missing-set(sexmapping)-{'0'}
    if sexunknown:raise ValueError(f'Unmapped sex values: {sorted(sexunknown)}; configure sex_map')
    prepared['SEX']=sex.map(sexmapping)
    # FAM sex is always PLINK 1=male/2=female; flag disagreement with chosen column.
    famsex=raw.SEX_FAM.map({'1':0,'2':1});bad=prepared.SEX.notna() & famsex.notna() & prepared.SEX.ne(famsex)
    if bad.any():raise ValueError(f'{int(bad.sum())} selected sex values disagree with PLINK FAM; inspect sex_map/SEX column')
    prepared=prepared.merge(pc,on=['FID','IID'],how='left',validate='one_to_one')
    apcol=col.get('apoe_genotype','');prepared['APOE4']=raw[apcol].map(apoe4) if apcol in raw else np.nan
    for analysis_name,input_name in col.get('covariates',{}).items():
        source=norm_values(raw[input_name]);prepared[analysis_name]=pd.to_numeric(source.where(~source.isin(missing)),errors='coerce').replace([np.inf,-np.inf],np.nan)
    reason=[]
    covars=normalize_covariates(cfg)
    if not set(covars).issubset(prepared):raise ValueError(f'Prepared phenotype table lacks requested covariates: {sorted(set(covars)-set(prepared))}')
    for _,row in prepared.iterrows():
        why=[]
        if pd.isna(row.AD):why.append('missing_outcome')
        why.extend('missing_'+x for x in covars if not np.isfinite(row[x]))
        reason.append(';'.join(why))
    prepared['M1_EXCLUSION_REASON']=reason;prepared['M1_ELIGIBLE']=[int(not x) for x in reason]
    table(out/'cohort_samples.tsv',fam[['FID','IID']]);table(out/'phenotypes_prepared.tsv',prepared)
    table(out/'samples_excluded_from_M1.tsv',prepared[prepared.M1_ELIGIBLE.eq(0)])
    comp=[]
    for c in [f'PC{i}' for i in range(1,6)]:
        if c in raw:
            x=pd.to_numeric(raw[c],errors='coerce').to_numpy(float);y=prepared[c].to_numpy(float);ok=np.isfinite(x)&np.isfinite(y)
            corr=float(np.corrcoef(x[ok],y[ok])[0,1]) if ok.sum()>2 and np.std(x[ok])>0 and np.std(y[ok])>0 else np.nan
            comp.append({'PC':c,'N_COMPARED':int(ok.sum()),'CORRELATION_WITH_PHENOTYPE_PC':corr,'SOURCE_USED':cfg['pca_file']})
    table(out/'pca_source_comparison.tsv',pd.DataFrame(comp,columns=['PC','N_COMPARED','CORRELATION_WITH_PHENOTYPE_PC','SOURCE_USED']))
    summary={'project':cfg['project'],'cohort':cfg['cohort_label'],'technical_N':len(fam),'phenotype_rows_for_cohort':len(p),'excluded_noncohort_phenotype_rows':extra_pheno,'excluded_noncohort_PCA_rows':extra_pca,
    'AD_controls':int(prepared.AD.eq(0).sum()),'AD_cases':int(prepared.AD.eq(1).sum()),'AD_unknown':int(prepared.AD.isna().sum()),'M1_complete_N':int(prepared.M1_ELIGIBLE.sum()),
    'M1_controls':int(((prepared.AD==0)&prepared.M1_ELIGIBLE.eq(1)).sum()),'M1_cases':int(((prepared.AD==1)&prepared.M1_ELIGIBLE.eq(1)).sum()),
    'APOE_genotype_decoded_N':int(prepared.APOE4.notna().sum()),'apoe_genotype_column':apcol,'sex_column':col['sex'],'sex_map':sexmapping,'age_column':col['age'],'outcome_column':col['outcome'],'outcome_map':mapping,'PCA_source':cfg['pca_file'],'association_covariates':covars,'custom_covariate_columns':col.get('covariates',{})}
    atomic_json(out/'phenotype_audit.json',summary)
    return fam,prepared,summary
