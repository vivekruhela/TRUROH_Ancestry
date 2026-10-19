"""Self-contained HTML and multipage scientific PDF, including zero/failure states."""
from __future__ import annotations
import base64,html,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import LOCK,CHROM_LENGTHS,read_table,atomic_json,table,chromosomes
from ancestry import collapse_ancestry,resolve_primary_ancestry
from configuration import analysis_fingerprint, guard_output

COLORS={'NAM':'#82ac14','EUR':'#29a4df','AFR':'#9c27b0'}
def load_json(p,default=None):return json.loads(p.read_text()) if p.exists() else (default or {})
def optional_table(p):return read_table(p) if p.exists() else pd.DataFrame()
def empty(ax,message):ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,wrap=True,color='#52616b');ax.set_axis_off()
def make_report(cfg):
    guard_output(cfg,claim=True)
    project=cfg['project'];label=cfg['cohort_label']
    try:primary,_=resolve_primary_ancestry(cfg,write=False)
    except (ValueError,OSError,KeyError):primary=cfg['association']['primary_ancestry']   # diagnostic report before scoring
    stem=f'TRUROH_{project}_report'
    root=Path(cfg['output_root']);out=root/'report';out.mkdir(parents=True,exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    manifest=load_json(root/'run_manifest.json');final=load_json(root/'final_status.json');pre=load_json(root/'preflight_status.json');audit=load_json(root/'inputs/phenotype_audit.json')
    compatible=manifest.get('analysis_config_sha256')==analysis_fingerprint(cfg)
    complete=compatible and final.get('state')=='completed' and pre.get('state')=='passed' and final.get('run_signature')==manifest.get('run_signature') and pre.get('run_signature')==manifest.get('run_signature')
    title=label+' | TRUROH GRID 1433'
    stages=[]
    for c in chromosomes(cfg['chromosomes']):
        s=load_json(root/f'chr{c}/status.json',{'state':'not_run'})
        stages.append({'CHR':c,'STATE':s.get('state'),'STAGE':s.get('stage',''),'GENOTYPE_SAMPLES':s.get('samples'),'VARIANTS':s.get('variants'),'RAW_RG':s.get('raw_RG_rows'),'NORMALIZED_ROH':s.get('normalized_ROH_calls'),'CANDIDATE_REGIONS':s.get('candidate_regions'),'ERROR':s.get('error','')})
    stage=pd.DataFrame(stages);table(out/'chromosome_status.tsv',stage)
    scored=optional_table(root/'results/regions_scored.tsv') if complete else pd.DataFrame()
    assoc=optional_table(root/'results/associations.tsv') if complete else pd.DataFrame()
    q=optional_table(root/'results/global_ancestry.tsv') if compatible else pd.DataFrame()
    pheno=optional_table(root/'inputs/phenotypes_prepared.tsv') if compatible else pd.DataFrame()
    burden=optional_table(root/'results/individual_ROH_burden.tsv') if complete else pd.DataFrame()
    n=int(manifest.get('technical_N',audit.get('technical_N',0)));hc=int(scored.HIGH_CONFIDENCE.sum()) if not scored.empty else (0 if complete else None)
    summary={'project':project,'cohort':label,'config_path':manifest.get('config_path'),'output_root':str(root),'state':'COMPLETE' if complete else 'INCOMPLETE','scope':manifest.get('genome_scope','unknown'),'technical_N':n or None,
    'chromosomes_expected':len(stage),'chromosomes_completed':int(stage.STATE.eq('completed').sum()),'chromosomes_failed':int(stage.STATE.eq('failed').sum()),
    'candidate_regions':len(scored) if complete else None,'high_confidence_ROH':hc,'phenotype':audit,'locked':LOCK,
    'minimum_carriers':manifest.get('locked_min_carriers'),'association_enabled':cfg['association']['enabled'],
    'converged_association_models':int(assoc.CONVERGED.eq(1).sum()) if not assoc.empty else 0,
    'nominal_associated_unique_regions':int(assoc.loc[assoc.NOMINAL_ASSOCIATED.eq(1),'REGION_ID'].nunique()) if not assoc.empty else 0,
    'primary_ancestry':primary,
    'error':final.get('error',pre.get('error',''))}
    if not complete:
        summary['converged_association_models']=summary['nominal_associated_unique_regions']=None
    atomic_json(out/'summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','axes.titleweight':'bold'})
    figures=[]
    with PdfPages(out/(stem+'.pdf')) as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,title,fontsize=22,weight='bold',color='#153448')
        state=summary['state'];status_color='#327d4a' if complete else '#b44b37'
        fig.text(.06,.87,state+' | '+summary['scope'].replace('_',' '),fontsize=14,color=status_color,weight='bold')
        desc=[f"Technical cohort: {n if n else 'unavailable'} {label} samples; minimum carriers: {summary['minimum_carriers']}",
                                'Locked: BCFtools / no LD; length >= 1 Mb; carriers >= 0.5%; HVR <= 0.005; HRS >= 0.5; RDI >= 0.7',
                                f"Chromosomes completed: {summary['chromosomes_completed']}/{len(stage)}; failed: {summary['chromosomes_failed']}",
                                f"Candidate regions: {summary['candidate_regions']}; high-confidence regions: {hc}",
                                f"AD controls: {audit.get('AD_controls','NA')}; cases: {audit.get('AD_cases','NA')}; missing outcome: {audit.get('AD_unknown','NA')}",
                                f"M1 complete-case samples: {audit.get('M1_complete_N','NA')}; technical scoring uses all {label} genotyped samples.",
                                f"Converged model rows: {summary['converged_association_models']}; nominally associated distinct regions (P < {cfg['association']['alpha']}): {summary['nominal_associated_unique_regions']}; primary ancestry: {primary}",
                                'HVR and HRS gates use raw regional summaries; RDI uses source LOCO ancestry correction.',
                                'No new knee, threshold tuning, disease-based ROH selection, or real-data truth labels.',
                                'Firth p-values: penalized likelihood-ratio; intervals: Wald approximation.',
                                'Ancestry-specific effects evaluate ROH at interval-average local ancestry = 1.',
                                'P values are nominal (no multiple-testing correction: too few individuals/regions for power after correction); association is not evidence of ROH ground truth.']
        import textwrap
        if summary['error']:desc.append('Error: '+summary['error'][:500]+' (full details in HTML/summary.json)')
        text='\n\n'.join('\n'.join(textwrap.wrap(x,112)) for x in desc)
        fig.text(.06,.8,text,va='top',fontsize=10,linespacing=1.2);pdf.savefig(fig);plt.close(fig)
        def save(fig,name):
            fig.tight_layout();p=figdir/(name+'.png');fig.savefig(p,dpi=160,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');figures.append((name,p));plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4.5));ax.set_title('Chromosome processing and candidate counts')
        completed=stage.STATE.eq('completed');vals=pd.to_numeric(stage.CANDIDATE_REGIONS,errors='coerce')
        ax.bar(stage.CHR[completed],vals[completed],color='#397b9b')
        for _,r in stage[~completed].iterrows():ax.text(r.CHR,0,'×',color='#b44b37',ha='center',va='bottom',fontsize=14)
        ax.set_xticks(stage.CHR);ax.set_xlabel('Chromosome');ax.set_ylabel('Candidate regions');ax.text(.99,.98,'× = failed / not run, never counted as biological zero',ha='right',va='top',transform=ax.transAxes,fontsize=8)
        save(fig,'01_chromosome_status')
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        if complete:
            counts=[len(scored)]
            gates=['PASS_TECHNICAL','PASS_LENGTH','PASS_CARRIERS','PASS_HVR','PASS_HRS','PASS_RDI'];mask=pd.Series(True,index=scored.index)
            for g in gates:mask &= scored[g].eq(1);counts.append(int(mask.sum()))
            axes[0].barh(['All candidates','Assessable','Length','Carrier count','HVR','HRS','RDI'],counts,color=['#8b9eac']*6+['#33895b']);axes[0].invert_yaxis();axes[0].set_xlabel('Regions remaining (cumulative gates)');axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            if not scored.empty:
                good=scored.HIGH_CONFIDENCE.eq(1);axes[1].scatter(scored.MEDIAN_HVR_RAW[~good],scored.MEAN_HRS_RAW[~good],s=12,color='#aab8c2',alpha=.5,label='Does not pass all gates')
                axes[1].scatter(scored.MEDIAN_HVR_RAW[good],scored.MEAN_HRS_RAW[good],s=20,color='#33895b',label='High confidence')
                axes[1].axvline(.005,color='#b45c48',ls='--');axes[1].axhline(.5,color='#b45c48',ls='--');axes[1].set_xscale('symlog',linthresh=.001);axes[1].set_xlim(left=0);axes[1].set_ylim(0,1.05);axes[1].legend(fontsize=8)
                axes[1].set_xlabel('Median raw HVR');axes[1].set_ylabel('Mean raw HRS')
            else:empty(axes[1],'No candidate ROHs. All outputs remain valid.')
        else:
            for ax in axes:empty(ax,'Final QC is unavailable until every configured chromosome completes.')
        axes[0].set_title('Locked selection');axes[1].set_title('Raw metric gates');save(fig,'02_locked_selection')
        fig,ax=plt.subplots(figsize=(11,4))
        if not q.empty:
            collapsed=collapse_ancestry(q,'Q_');sortcol='Q_'+primary if 'Q_'+primary in collapsed else collapsed.columns[0];ordered=collapsed.sort_values(sortcol,ascending=False);bottom=np.zeros(len(ordered))
            for col in ordered:
                name=col.removeprefix('Q_');v=ordered[col].to_numpy(float);ax.bar(np.arange(len(v)),v,bottom=bottom,width=1,color=COLORS.get(name),label=name);bottom+=v
            ax.set_xlim(-.5,len(ordered)-.5);ax.set_ylim(0,1);ax.set_xlabel(label+' samples sorted by decreasing '+sortcol.removeprefix('Q_')+' ancestry');ax.set_ylabel('Global ancestry fraction');ax.legend(ncol=min(6,len(ordered.columns)),fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17))
        else:empty(ax,'Global ancestry unavailable: complete all chromosome tasks.')
        ax.set_title(label+' global ancestry from MSP tract lengths');save(fig,'03_global_ancestry')
        fig,ax=plt.subplots(figsize=(7,5))
        if not pheno.empty and not q.empty:
            collapsed=collapse_ancestry(q,'Q_');p=q[['IID']].join(collapsed).merge(pheno,on='IID',validate='one_to_one')
            color=p['Q_'+primary] if 'Q_'+primary in p else np.zeros(len(p));scatter=ax.scatter(p.PC1,p.PC2,c=color,cmap='viridis',vmin=0,vmax=1,s=16,alpha=.8);fig.colorbar(scatter,ax=ax,label=primary+' ancestry' if 'Q_'+primary in p else 'Ancestry unavailable');ax.set_xlabel('PC1');ax.set_ylabel('PC2')
        else:empty(ax,'PCA/ancestry unavailable')
        ax.set_title(label+' genotype PCA');save(fig,'04_pca')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        if not burden.empty and not pheno.empty:
            b=burden.merge(pheno[['IID','AD']],on='IID',validate='one_to_one')
            for ax,col,title2 in zip(axes,['FROH_RAW_GE_1MB','FROH_HC_MEMBER_SEGMENTS'],['Raw calls >= 1 Mb','High-confidence member segments']):
                groups=[b.loc[b.AD.eq(k),col].to_numpy(float) for k in [0,1]]
                if all(len(x) for x in groups):ax.boxplot(groups,showfliers=False);ax.set_xticks([1,2],['Control','AD'])
                else:empty(ax,'Both AD groups are needed for comparison')
                ax.set_ylabel('FROH (GRCh38 reference denominator)');ax.set_title(title2)
        else:
            for ax in axes:empty(ax,'ROH burden unavailable')
        save(fig,'05_roh_burden')
        ancestries=cfg['association']['ancestries']
        fig,axes=plt.subplots(len(ancestries),1,figsize=(11,max(3,2.7*len(ancestries))),sharex=True,squeeze=False);axes=axes[:,0]
        chroms=chromosomes(cfg['chromosomes']);offset={};running=0
        for c in chroms:offset[c]=running;running+=CHROM_LENGTHS[c]
        active_axes=[]
        for ax,anc in zip(axes,ancestries):
            a=assoc[assoc.MODEL.eq('LA_'+anc)&assoc.CONVERGED.eq(1)].copy() if not assoc.empty else pd.DataFrame()
            if a.empty:empty(ax,anc+': no converged ancestry-specific tests')
            else:
                active_axes.append(ax)
                xpos=a.apply(lambda r:offset[int(r.CHR)]+(r.CONS_START+r.CONS_END)/2,axis=1)
                y=-np.log10(np.maximum(pd.to_numeric(a.P_VALUE).to_numpy(float),np.finfo(float).tiny));ax.scatter(xpos,y,c=COLORS.get(anc,'#397b9b'),s=17,alpha=.8)
                ax.axhline(-math.log10(.05),color='#a9b1b8',ls='--',lw=.7);ax.set_ylabel(anc+'\n-log10(p)')
            ax.set_title(anc+' local ancestry = 1',fontsize=10)
        if active_axes:
            active_axes[-1].set_xticks([offset[c]+CHROM_LENGTHS[c]/2 for c in chroms],[str(c) for c in chroms]);active_axes[-1].tick_params(labelbottom=True);active_axes[-1].set_xlabel('Chromosome')
        fig.suptitle('High-confidence ROH: ancestry-specific Firth associations',y=1.01);save(fig,'06_ancestry_associations')
        fig,ax=plt.subplots(figsize=(11,8));high=scored[scored.HIGH_CONFIDENCE.eq(1)] if not scored.empty else pd.DataFrame()
        if not high.empty:
            for c in chromosomes(cfg['chromosomes']):
                ax.plot([0,CHROM_LENGTHS[c]/1e6],[c,c],lw=6,color='#e1e6ea',solid_capstyle='round')
            for r in high.itertuples(index=False):ax.plot([r.CONS_START/1e6,r.CONS_END/1e6],[r.CHR,r.CHR],lw=6,color='#33895b')
            ax.set_yticks(chromosomes(cfg['chromosomes']));ax.invert_yaxis();ax.set_ylabel('Chromosome');ax.set_xlabel('Position (Mb)')
        else:empty(ax,'No high-confidence ROH to display' if complete else 'Final high-confidence ROH unavailable')
        ax.set_title(label+' high-confidence ROH intervals');save(fig,'07_roh_karyogram')
    # Tables use escaped text; output includes full TSVs beside this report.
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:15px/1.5 system-ui,sans-serif;margin:40px auto;max-width:1200px;color:#243844;padding:0 24px}h1,h2{color:#153448}.badge{padding:10px;background:#edf3f5;border-left:5px solid '+status_color+'}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;border-bottom:1px solid #dfe6ea;text-align:left}th{background:#eff4f6}img{max-width:100%;height:auto}.scroll{overflow:auto}pre{white-space:pre-wrap}a{color:#246e90}</style></head><body>']
    parts+=['<h1>'+html.escape(title)+'</h1>','<p class="badge">'+html.escape(summary['state']+' | '+summary['scope'].replace('_',' '))+'</p>',
                                    '<p>Fixed simulation-selected thresholds. Technical selection is independent of AD outcomes. Only '+html.escape(label)+' genotype IDs enter this analysis.</p>',
                                    '<h2>Run summary</h2><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>',
                                    '<h2>Chromosome status</h2><div class="scroll">'+stage.to_html(index=False,escape=True,na_rep='unavailable')+'</div>']
    for name,p in figures:
        data=base64.b64encode(p.read_bytes()).decode();parts+=['<h2>'+html.escape(name[3:].replace('_',' ').capitalize())+'</h2>','<img alt="'+name+'" src="data:image/png;base64,'+data+'">']
    if not scored.empty:
        fields=['REGION_ID','CHR','CONS_START','CONS_END','LENGTH_BP','N_CARRIERS','MEDIAN_HVR_RAW','MEAN_HRS_RAW','RDI_A','HIGH_CONFIDENCE','GENES']
        parts+=['<h2>High-confidence intervals (first 100; complete results in TSV)</h2><div class="scroll">'+scored[scored.HIGH_CONFIDENCE.eq(1)][fields].head(100).to_html(index=False,escape=True)+'</div>']
    if not assoc.empty:
        fields=['REGION_ID','MODEL','ANCESTRY','N','CARRIERS','OR','CI_LOW','CI_HIGH','P_VALUE','STATUS']
        parts+=['<h2>Association results (100 smallest converged p-values)</h2><div class="scroll">'+assoc[assoc.CONVERGED.eq(1)].sort_values('P_VALUE')[fields].head(100).to_html(index=False,escape=True)+'</div>',
    '<p>P-values use the Firth profile penalized likelihood-ratio test. Confidence intervals are Wald approximations. P values are nominal: no multiple-testing correction is applied, because cohort size and the number of high-confidence regions give no power after correction; associations are exploratory and should be judged by replication across cohorts. An effect at local ancestry 1 can require extrapolation; this is flagged in associations.tsv. The model does not include a kinship random effect.</p>']
    links=[('PDF report',stem+'.pdf'),('Run summary','summary.json'),('Chromosome status','chromosome_status.tsv')]
    for name in ['regions_scored.tsv','high_confidence_ROH.tsv','individual_scored.tsv.gz','members.tsv.gz','associations.tsv',f'primary_{primary}_associations.tsv','AD_associated_nominal.tsv','primary_ancestry_selection.json','global_ancestry.tsv','individual_ROH_burden.tsv','locked_thresholds.tsv','association_models.json','scoring_audit.json']:
        if (root/'results'/name).exists():links.append((name,'../results/'+name))
    parts+=['<h2>Files</h2><ul>']+['<li><a href="'+html.escape(url)+'">'+html.escape(label)+'</a></li>' for label,url in links]+['</ul>',
    '<p>Source protocol: TRUROH-SIM v3.8.0. A separate seed panel is pruned for HRS; BCFtools input is not LD-pruned. See the package docs for the equations, ancestry-source handling, assumptions, and software references.</p></body></html>']
    (out/(stem+'.html')).write_text('\n'.join(parts))
    print('Report written: '+str(out/(stem+'.html')),flush=True)
    return summary
