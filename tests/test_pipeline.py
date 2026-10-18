from __future__ import annotations
import copy,importlib.util,json,math,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import *
from ancestry import subset_msp,combine_ancestry
from phenotypes import prepare_phenotypes,apoe4
from firth import fit,objective
from pipeline import preflight,score,finalize,normalize_bcftools,apply_locked,submit
from compute_raw_metrics import build_interval_counts,hrs_for_region,sample_backgrounds,stable_seed
from build_candidates import complete_linkage,reciprocal
from association import one_model


def write_bed(prefix,codes):
 m,n=codes.shape;pad=np.ones((m,((n+3)//4)*4),dtype=np.uint8);pad[:,:n]=codes
 packed=pad[:,0::4]|(pad[:,1::4]<<2)|(pad[:,2::4]<<4)|(pad[:,3::4]<<6)
 Path(str(prefix)+'.bed').write_bytes(b'\x6c\x1b\x01'+packed.tobytes())


def fixture(base,zero=False):
 base=Path(base);base.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'config/gapp.json').read_text())
 cfg.update(chromosomes='1-2',plink_prefix=str(base/'gapp'),phenotype_file=str(base/'covs.txt'),pca_file=str(base/'pca.eigenvec'),rfmix_msp_pattern=str(base/'ancestry_chr{CHR}.msp.tsv'),output_root=str(base/'out'))
 cfg['association']['max_fit_seconds']=10
 rng=np.random.default_rng(108);n=40;iids=[f'{i:04d}' for i in range(1,n)]+['NA'];fids=[f'F{i//2:03d}' for i in range(n)]
 fam=pd.DataFrame({'FID':fids,'IID':iids,'PAT':['0']*n,'MAT':['0']*n,'SEX':[str(i%2+1) for i in range(n)],'PHENO':[str(i%2+1) for i in range(n)]})
 fam.to_csv(str(base/'gapp')+'.fam',sep=' ',header=False,index=False)
 positions=np.arange(100000,100000001,100000);m=len(positions)
 b=pd.DataFrame({'CHR':np.repeat([1,2],m),'ID':['old'+str(i) for i in range(2*m)],'CM':0,'POS':np.tile(positions,2),'A1':'A','A2':'G'})
 b.to_csv(str(base/'gapp')+'.bim',sep='\t',header=False,index=False)
 codes=rng.choice([0,2,3,1],size=(2*m,n),p=[.45,.3,.24,.01]).astype(np.uint8)
 for c in [1,2]:
  target=(b.CHR.eq(c)&b.POS.between(20000000,22000000)).to_numpy();codes[target,:6]=0
 write_bed(base/'gapp',codes)
 pc=rng.normal(size=(n,5));p=pd.DataFrame({'fam':fids,'id':iids,'sex':fam.SEX,'Dx_binary':[str(i%2) for i in range(n)],'Age Data Analysis':rng.integers(55,85,n).astype(str),'apoe_wgs':['3/4' if i%3==0 else '3/3' for i in range(n)]})
 for j in range(5):p[f'PC{j+1}']=-pc[:,j]
 p.loc[7,'Age Data Analysis']='NA'
 p.to_csv(base/'covs.txt',sep='\t',index=False)
 pd.concat([fam[['FID','IID']],pd.DataFrame(pc)],axis=1).to_csv(base/'pca.eigenvec',sep=' ',header=False,index=False)
 names=iids+['EXCLUDED_PEADI'];ranges=[(1,30000000),(30000001,60000000),(60000001,100000000)]
 for c in [1,2]:
  with (base/f'ancestry_chr{c}.msp.tsv').open('w') as f:
   f.write('#Subpopulation order/codes: NAM=0 EUR=1 AFR=2\n')
   f.write('#chm\tspos\tepos\tsgpos\tegpos\tn snps\t'+'\t'.join(x+'.'+str(h) for x in names for h in [0,1])+'\n')
   for k,(s,e) in enumerate(ranges):
    vals=rng.integers(0,3,size=2*len(names));vals[:12]=[0,0,0,0,0,0,1,0,2,0,0,0] if k==0 else vals[:12]
    f.write('\t'.join(map(str,[f'chr{c}',s,e,k,k+1,100]+vals.tolist()))+'\n')
 return cfg,codes,fam


def finish_fixture(cfg,zero=False):
 manifest=preflight(cfg,check_external=False);root=Path(cfg['output_root'])
 for c in [1,2]:
  d=root/f'chr{c}';prefix=d/'input/cohort';fam,bim=split_bed(cfg,c,prefix);subset_msp(cfg,c,fam,d/'ancestry/cohort.msp.tsv')
  raw=d/'bcftools.roh.txt';lines=['# Synthetic BCFtools RG fixture; no real caller was run.']
  if not zero:
   for iid in fam.IID.iloc[:6]:lines.append(f'RG\t{iid}\t{c}\t20000000\t22000000\t2000001\t21\t99')
  raw.write_text('\n'.join(lines)+'\n');normalize_bcftools(raw,c,set(fam.IID),d/'calls.tsv')
  seed=d/'seeds';seed.mkdir(exist_ok=True);(seed/'eligible.snplist').write_text('\n'.join(bim.ID)+'\n');(seed/'seed.prune.in').write_text('\n'.join(bim.ID.iloc[::2])+'\n')
  command=[sys.executable,str(ROOT/'src/build_candidates.py'),'--tool','bcftools','--ld','no','--chr',str(c),'--calls',str(d/'calls.tsv'),'--msp',str(d/'ancestry/cohort.msp.tsv'),'--fam',str(prefix)+'.fam','--regions',str(d/'regions.tsv'),'--members',str(d/'members.tsv')]
  subprocess.run(command,check=True,capture_output=True,text=True)
  command=[sys.executable,str(ROOT/'src/compute_raw_metrics.py'),'--tool','bcftools','--ld','no','--chr',str(c),'--regions',str(d/'regions.tsv'),'--members',str(d/'members.tsv'),'--bed-prefix',str(prefix),'--original-bim',str(prefix)+'.bim','--seed-pruned',str(seed/'seed.prune.in'),'--seed-eligible',str(seed/'eligible.snplist'),'--individual-output',str(d/'individual_raw.tsv'),'--regional-output',str(d/'regional_raw.tsv')]
  r=subprocess.run(command,capture_output=True,text=True)
  if r.returncode:raise RuntimeError(r.stdout+r.stderr)
  outputs=['calls.tsv','regions.tsv','members.tsv','individual_raw.tsv','regional_raw.tsv','ancestry/cohort.msp.tsv','ancestry/global_ancestry_bp.tsv','ancestry/ancestry_components.tsv']
  atomic_json(d/'status.json',{'state':'completed','stage':'completed','run_signature':manifest['run_signature'],'samples':len(fam),'variants':len(bim),'candidate_regions':len(read_table(d/'regions.tsv')),'output_hashes':{x:file_hash(d/x) for x in outputs}})
 return root


class CoreTests(unittest.TestCase):
 def test_carrier_rounding(self):
  self.assertEqual(carrier_min(625),4);self.assertEqual(carrier_min(600),3);self.assertEqual(carrier_min(2000),10);self.assertEqual(carrier_min(601),4)
 def test_locked_gates_and_nan(self):
  r=pd.DataFrame([dict(LENGTH_BP=1000000,N_CARRIERS=4,TECHNICALLY_ASSESSABLE=1,MEDIAN_HVR_RAW=.005,MEAN_HRS_RAW=.5,RDI_A=.7,SOFT_PASS=0)]*7)
  for i,(k,v) in enumerate([('LENGTH_BP',999999),('N_CARRIERS',3),('TECHNICALLY_ASSESSABLE',0),('MEDIAN_HVR_RAW',.0050001),('MEAN_HRS_RAW',.49999),('RDI_A',np.nan)],1):r.loc[i,k]=v
  s=apply_locked(r,625);self.assertEqual(s.HIGH_CONFIDENCE.tolist(),[1,0,0,0,0,0,0])
 def test_exact_bed_counts_missing_and_chunk_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   prefix=Path(d)/'test';codes=np.array([[0,2,1],[2,0,3],[1,2,0],[3,1,2],[2,3,1]],np.uint8);write_bed(prefix,codes)
   iids=['0001','NA','b'];pos=np.array([10,20,30,40,50]);intervals={'r':[(10,30),(20,20),(31,49),(1,5),(10,50)]};members={'r':iids}
   for chunk in [1,2,5,6]:
    counts=build_interval_counts(prefix,iids,pos,members,intervals,chunk,1)['r']
    for j,(a,b) in enumerate(intervals['r']):
     part=codes[(pos>=a)&(pos<=b)]
     np.testing.assert_array_equal(counts['nm'][j],(part!=1).sum(axis=0));np.testing.assert_array_equal(counts['het'][j],(part==2).sum(axis=0))
 def test_complete_linkage_no_transitive_chain(self):
  seg=[dict(IID=str(i),START=a,END=b) for i,(a,b) in enumerate([(1,100),(51,150),(101,200)])]
  clusters=complete_linkage(seg,.5);self.assertEqual(sorted(map(len,clusters)),[1,2])
  for c in clusters:
   for a in c:
    for b in c:self.assertGreaterEqual(reciprocal(a,b),.5)
 def test_seed_span_hrs(self):
  m=pd.DataFrame([dict(IID='one',SEG_START=20,SEG_END=40),dict(IID='none',SEG_START=60,SEG_END=70)])
  h=hrs_for_region(m,np.array([10,20,30,40,50]));self.assertAlmostEqual(h['one'],.75*3/5+.25*20/40);self.assertEqual(h['none'],0)
 def test_backgrounds_deterministic_and_excluded_buffer(self):
  x=sample_backgrounds(20000000,22000000,100000000,10000000,500,stable_seed('test',108))
  self.assertEqual(x,sample_backgrounds(20000000,22000000,100000000,10000000,500,stable_seed('test',108)))
  self.assertTrue(all(e-s+1==2000001 and (e<10000000 or s>32000000) for s,e in x))
 def test_firth_separation_analytic_odds_ratio(self):
  x=np.column_stack([np.ones(20),[0]*10+[1]*10]);y=np.array([0]*10+[1]*10)
  f=fit(x,y,test_index=1);self.assertAlmostEqual(np.exp(f['beta'][1]),441.,places=3);self.assertTrue(0<f['p_value']<.001)
 def test_firth_intercept_and_score_gradient(self):
  X=np.ones((8,1));y=np.array([1,0,0,0,0,0,0,0]);f=fit(X,y);self.assertAlmostEqual(f['beta'][0],math.log(1.5/7.5),places=5)
  X=np.column_stack([np.ones(8),np.arange(8)/8]);b=np.array([.2,-.3]);ll,g,_=objective(b,X,y);eps=1e-5
  numeric=np.array([(objective(b+np.eye(2)[i]*eps,X,y)[0]-objective(b-np.eye(2)[i]*eps,X,y)[0])/(2*eps) for i in range(2)])
  np.testing.assert_allclose(g,numeric,atol=1e-7)
 def test_loco_ignores_extreme_scores_in_training(self):
  from scoring_math import loco_adjust
  rng=np.random.default_rng(7);n=2000;chrom=np.repeat(np.arange(1,5),n//4);X=rng.random((n,3))
  y=3+X@np.array([.5,-.3,.2])+rng.normal(0,.5,n);y[rng.choice(n,8,replace=False)]=3e5   # MAD-floor blow-ups
  df=pd.DataFrame({'CHR':chrom,'Y':y});ok=np.abs(y)<100
  naive,_=loco_adjust(df,'Y',X,1.0);robust,fb=loco_adjust(df,'Y',X,1.0,TECH['loco_train_abs_max'])
  self.assertLess(np.median(naive[ok]),-10)                                  # the v2.0.0 failure mode
  self.assertAlmostEqual(np.median(robust[ok]),np.median(y),delta=.2);self.assertEqual(fb.sum(),0)
  clean=df[ok];base,_=loco_adjust(clean,'Y',X[ok],1.0)
  np.testing.assert_allclose(robust[ok]-np.median(y),base-np.median(y[ok]),atol=1e-9)  # identical fit to dropping them
 def test_apoe_no_opaque_coding_guess(self):
  self.assertEqual(apoe4('3/4'),1);self.assertEqual(apoe4('E4/E4'),2);self.assertTrue(np.isnan(apoe4('1')))
 def test_ancestry_effect_at_one(self):
  rng=np.random.default_rng(12);n=150;d=pd.DataFrame({'AD':rng.integers(0,2,n),'ROH':rng.integers(0,2,n),'LA_TARGET':rng.choice([0,.5,1],n)})
  d['LA_CENTERED']=d.LA_TARGET-1;d['ROHxLA_CENTERED']=d.ROH*d.LA_CENTERED
  cfg=json.loads((ROOT/'config/gapp.json').read_text());cfg['association']['covariates']=[]
  r=one_model(d,cfg,'LA_NAM','NAM',extra=['LA_CENTERED','ROHxLA_CENTERED']);self.assertEqual(r['STATUS'],'converged')
  X=np.column_stack([np.ones(n),d.ROH,d.LA_TARGET,d.ROH*d.LA_TARGET]);f=fit(X,d.AD)
  self.assertAlmostEqual(r['BETA'],f['beta'][1]+f['beta'][3],places=4)


class IntegrationTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.cfg,self.codes,self.fam=fixture(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_phenotype_ids_pca_and_technical_denominator(self):
  fam,p,a=prepare_phenotypes(self.cfg);self.assertEqual(a['technical_N'],40);self.assertEqual(a['M1_complete_N'],39);self.assertEqual(fam.IID.iloc[0],'0001');self.assertEqual(p.IID.iloc[-1],'NA')
  src=pd.read_csv(self.cfg['pca_file'],sep=r'\s+',header=None,dtype={0:str,1:str});np.testing.assert_allclose(p.PC1,src.iloc[:,2])
  self.assertEqual(int(p.APOE4.sum()),14)
 def test_unmapped_outcome_fails(self):
  p=pd.read_csv(self.cfg['phenotype_file'],sep='\t',dtype=str,keep_default_na=False);p.loc[0,'Dx_binary']='2';p.to_csv(self.cfg['phenotype_file'],sep='\t',index=False)
  with self.assertRaisesRegex(ValueError,'Unmapped'):prepare_phenotypes(self.cfg)
 def test_subset_ancestry_excludes_other_cohort(self):
  fam=read_fam(self.cfg['plink_prefix']);dest=Path(self.tmp.name)/'subset.msp.tsv';a=subset_msp(self.cfg,1,fam,dest)
  self.assertEqual(a['retained_cohort_samples'],40);self.assertEqual(a['excluded_other_ids'],1);self.assertNotIn('EXCLUDED_PEADI',dest.read_text())
  from rfmix_utils import MSPData
  m=MSPData.read(dest);self.assertEqual(set(m.sample_haps),set(fam.IID));np.testing.assert_allclose(m.local_proportions('0001',20000000,22000000),[1,0,0])
 def test_missing_ancestry_sample_is_not_imputed(self):
  fam=read_fam(self.cfg['plink_prefix']);fam.loc[0,'IID']='not_in_msp'
  with self.assertRaisesRegex(ValueError,'absent in MSP'):subset_msp(self.cfg,1,fam,Path(self.tmp.name)/'missing.msp.tsv')
 def test_split_preserves_samples_genotypes_and_masks_outcomes(self):
  fam,b=split_bed(self.cfg,2,Path(self.tmp.name)/'split/gapp');out=np.frombuffer(Path(self.tmp.name+'/split/gapp.bed').read_bytes()[3:],np.uint8).reshape(len(b),10)
  dec=np.empty((len(b),40),np.uint8)
  for i in range(40):dec[:,i]=(out[:,i//4]>>(2*(i%4)))&3
  np.testing.assert_array_equal(dec,self.codes[1000:]);self.assertTrue(fam.PHENO_FAM.eq('-9').all());self.assertTrue(b.ID.str.startswith('2:').all())
 def test_full_scoring_association_reporting(self):
  root=finish_fixture(self.cfg);finalize(self.cfg)
  s=read_table(root/'results/regions_scored.tsv');self.assertEqual(len(s),2);self.assertEqual(s.HIGH_CONFIDENCE.tolist(),[1,1]);self.assertTrue(s.MEDIAN_HVR_RAW.eq(0).all());self.assertTrue(s.MEAN_HRS_RAW.eq(1).all())
  q=read_table(root/'results/global_ancestry.tsv');self.assertEqual(len(q),40);self.assertIn('NA',q.IID.tolist());self.assertNotIn('EXCLUDED_PEADI',q.IID.tolist())
  a=read_table(root/'results/associations.tsv');self.assertEqual(len(a),14);self.assertGreater(a.CONVERGED.sum(),0)
  self.assertEqual(json.loads((root/'report/summary.json').read_text())['state'],'COMPLETE');self.assertGreater((root/'report/TRUROH_GAPP_report.pdf').stat().st_size,10000)
  b=read_table(root/'results/individual_ROH_burden.tsv');self.assertEqual(len(b),40);self.assertEqual(b.loc[b.IID.eq('0007'),'HC_MEMBER_SEGMENT_BP'].iloc[0],0)
 def test_zero_candidates_complete_report(self):
  root=finish_fixture(self.cfg,zero=True);finalize(self.cfg)
  self.assertTrue(read_table(root/'results/high_confidence_ROH.tsv').empty);summary=json.loads((root/'report/summary.json').read_text());self.assertEqual(summary['state'],'COMPLETE');self.assertEqual(summary['high_confidence_ROH'],0)
  self.assertEqual(len(read_table(root/'results/individual_ROH_burden.tsv')),40)
 def test_all_candidates_fail_locked_qc_still_complete(self):
  root=finish_fixture(self.cfg)
  for c in [1,2]:
   d=root/f'chr{c}';f=d/'individual_raw.tsv';v=read_table(f);v['HVR']=0.1;table(f,v)
   status=json.loads((d/'status.json').read_text());status['output_hashes']['individual_raw.tsv']=file_hash(f);atomic_json(d/'status.json',status)
  finalize(self.cfg);summary=json.loads((root/'report/summary.json').read_text())
  self.assertEqual(summary['state'],'COMPLETE');self.assertEqual(summary['candidate_regions'],2);self.assertEqual(summary['high_confidence_ROH'],0)
  self.assertTrue(read_table(root/'results/associations.tsv').empty)
 def test_failed_chromosome_reports_incomplete_not_zero(self):
  root=finish_fixture(self.cfg,zero=True);(root/'chr2/status.json').unlink()
  with self.assertRaisesRegex(RuntimeError,'has not run'):finalize(self.cfg)
  s=json.loads((root/'report/summary.json').read_text());self.assertEqual(s['state'],'INCOMPLETE');self.assertIsNone(s['high_confidence_ROH']);self.assertTrue((root/'report/TRUROH_GAPP_report.html').exists())
 def test_slurm_afterany_and_absolute_root(self):
  calls=[]
  def mock_run(cmd,**kw):
   calls.append(cmd);return subprocess.CompletedProcess(cmd,0,stdout='9001\n' if len(calls)==1 else '9002\n',stderr='')
  with patch('pipeline.check_tools',return_value={'status':'stub'}),patch('pipeline.shutil.which',return_value='/stub/sbatch'),patch('pipeline.subprocess.run',side_effect=mock_run):submit(self.cfg)
  self.assertEqual(len(calls),2);self.assertIn('afterany:9001',calls[1]);self.assertIn(str(ROOT),calls[0]);self.assertIn('1,2%4',calls[0])
 def test_input_change_blocks_mixed_run(self):
  preflight(self.cfg,check_external=False)
  with open(self.cfg['phenotype_file'],'a') as f:f.write('\n')
  from pipeline import check_manifest
  with self.assertRaisesRegex(ValueError,'Input changed'):check_manifest(self.cfg)

if __name__=='__main__':unittest.main(verbosity=2)
