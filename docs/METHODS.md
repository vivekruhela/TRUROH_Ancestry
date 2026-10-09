# Methods and protocol preservation

## Analysis unit and cohort

Only individuals in the user-configured cohort binary genotype FAM are eligible. All IDs remain strings; FID and IID are independent fields. Technical scoring and its denominator are independent of the phenotype complete-case subsets. Phenotypes are read only for subsequent association and its audit; no AD outcome is used in calling, candidate construction, background matching, ancestry correction, RDI or locked selection.

The caller sees the supplied QC common-variant marker panel. This is not a claim that all WGS variants are retained. Marker ascertainment and cohort ancestry may differ from the simulation domains; the simulation performance measures do not estimate real-data accuracy.

## Calling and candidate construction

PLINK genotypes are split by chromosome with their sample order unchanged. The internal FAM phenotype field is set to missing. Alleles are retained in their supplied A1/A2 order. The exported VCF uses IID sample names, with AF calculated from its actual cohort genotypes. PLINK's A2-based VCF reference allele is not asserted to be the GRCh38 reference base. This VCF is used only for symmetric biallelic GT/AF ROH inference; coordinate-based gene annotation does not depend on its allele orientation. [PLINK data-management documentation](https://www.cog-genomics.org/plink/1.9/data)

The BCFtools settings inherited from the simulation release are GT error Phred 30, constant recombination rate 1e-8 per bp, AF from INFO/AF, and region output. No new map or caller parameter is introduced. Native RG fields are parsed explicitly. A missing/malformed output is an error; a successful header-only region output is a valid zero-call result. [BCFtools manual](https://samtools.github.io/bcftools/bcftools.html#roh)

Calls >=250 kb and >=15 markers enter discovery. Within an individual, fragments can merge across <=250 kb if their ancestry-proportion total-variation distance is <=0.25. Across individuals, complete-linkage greedy clustering requires every pair's reciprocal overlap to be >=0.5; the algorithm disallows duplicate IIDs within a cluster. The exact source tie-breaking is retained.

For cluster r, the representative endpoints are the integer medians of carrier segment starts and ends. They are not the strict common intersection. The union endpoints encompass all member segments. Regional length is representative end−start+1. HVR counts use the representative interval; HRS seeds come from the union. ROH presence in association means membership in this source-defined cluster. All genomic interval operations use the source's 1-based inclusive convention.

Discovery support is max(3, ceil(0.0025 N)); final selection uses max(3, ceil(0.005 N)). Both use the full cohort FAM N. Source fragment length/support settings remain permissive during scoring; only the final pass applies 1 Mb and the 0.5% carrier fraction.

## HVR

For carrier i and representative region r:

\[
HVR_{ir}=HET_{ir}/N_{ir},
\]

where N counts nonmissing genotypes and HET counts heterozygotes. BED decoding uses the PLINK two-bit codes, with missing excluded from N. Cumulative counts are captured at interval boundaries in variant chunks; no genome-wide sample-by-variant matrix is constructed.

The source algorithm proposes 2,000 same-length backgrounds outside a 10 Mb buffer, using deterministic SHA-256-derived seeds (base 108). It retains up to 500 backgrounds per carrier with a background/target site-count ratio in [0.25,4]. Proposals are sampled with replacement as in v3.8.0. Their distance/site-count matching weights, effective sample size, weighted empirical lower-tail p-value, weighted median and scaled MAD are copied from the source.

For distance d, its source median scale kappa, and site-count ratio rho:

\[
w_b=\exp[-d_b^2/(2\kappa^2)]\exp[-|\log\rho_b|],\quad
ESS=\frac{(\sum_b w_b)^2}{\sum_b w_b^2},
\]

\[
p_{ir}=\frac{1+\sum_b w_b\mathbf1(HVR_{ib}\le HVR_{ir})}{1+\sum_b w_b},\quad
Z^H_{ir}=\frac{\operatorname{wmedian}(HVR_{ib})-HVR_{ir}}{\max(1.4826\operatorname{wMAD},10^{-6})}.
\]

Individual assessability requires >=10 observed sites and ESS>=40. Matching confidence combines min(1,ESS/40) with exponential decay of the weighted median chromosome-normalized distance. Background sampling, matching, epsilon floors and the source empirical p calculation are unchanged.

The final HVR threshold **is not a threshold on Z or an empirical p-value**: it is applied to the median of finite raw carrier HVR values in the region, <=0.005.

## HRS

The HRS panel uses MAF>=0.05, missingness<=0.10 and HWE p>=1e-6, with seed-only LD pruning at 200 variants / 50 step / r²=0.20. All cohort genotypes are considered, with outcomes masked and `include-nonctrl`/`--nonfounders` explicit. This makes seed selection independent of case status or FAM phenotype defaults. [PLINK filtering documentation](https://www.cog-genomics.org/plink/1.9/filter)

For a region's union seed set S and seeds recovered within the member segment:

\[
HRS_{ir}=\operatorname{clip}_{[0,1]}\left(0.75\frac{|S_{ir}|}{|S_r|}+0.25\frac{\operatorname{span}(S_{ir})}{\operatorname{span}(S_r)}\right).
\]

Zero-hit and zero-span cases use the original source rules. If no pruned seed occurs in a union interval, all eligible seed positions in that interval are used, as in the source. No eligible seeds makes HRS unavailable. The locked threshold is **mean raw HRS >=0.5**, before ancestry correction.

## Ancestry and LOCO correction

MSP files are restricted to the configured cohort sample IDs before use. Hard-call ancestry proportions are weighted by covered interval lengths, using the original inclusive MSP convention. Windows that share a boundary retain the source's inclusive weighting. Interval proportions normalize over covered MSP windows; completely uncovered intervals return unavailable ancestry. Missing/unknown ancestry codes and chromosome/sample mismatches fail explicitly.

Global ancestry is the genome-wide fraction of covered haplotype bases from the configured chromosomes. Q files are not substituted because the v3.8.0 source derives global proportions from MSP. Original source ancestry components, their order and omission of the final component are retained in technical features; aliases are collapsed only downstream for interpretation.

The feature vector includes K−1 global proportions, K−1 local-minus-global proportions, local/global total-variation divergence, and normalized local ancestry entropy. Ridge regression (lambda=1) is fitted on other chromosomes, using training-only standardization, for Z_H, the empirical-p-derived normal score Z_S, and HRS. Adjustment is source median(y)+y−prediction, with adjusted HRS clipped to [0,1].

The source fallback to raw scores is retained when fewer than max(20, number_of_features+5) valid training rows are available. Every fallback is exported and counted. The centering median follows the original implementation (all finite rows); this is an outcome-free centering operation, not new threshold training. A chromosome subset run may have many such fallbacks and is labeled accordingly.

## RDI and locked selection

Let adjusted individual HVR pass require individual assessability, adjusted Z_H>=−3 and empirical p<=0.99; adjusted HRS pass requires HRS_A>=0.01. Their carrier fractions are f_H and f_R.

\[
Q^H_r=\left[\Phi(\operatorname{median}Z^H_{A,ir})\,
\operatorname{mean}(1-p_{ir})\,\operatorname{mean}(MATCH\_CONF_{ir})\right]^{1/3},
\]

where these HVR quantities are aggregated over assessable carriers. Let Q_R be mean adjusted HRS and S=sqrt(f_H f_R). Then:

\[
RDI_{A,r}=\frac{3}{1/Q^H_r+1/Q^R_r+1/S_r}.
\]

A nonfinite component yields unavailable RDI; a zero component yields zero RDI. Technical assessability requires source discovery support, >=5% assessable carriers, >=3 seeds, and finite RDI components. The regional soft gates, including weighted Stouffer diagnostics, are retained as information only. No new soft gate is added to locked selection.

High confidence is the conjunction of technical assessability, length>=1 Mb, carriers>=max(3,ceil(0.005N)), median raw HVR<=0.005, mean raw HRS>=0.5, and RDI_A>=0.7. Missing values never pass. No knee calculation, source-domain reselection, simulation-truth operation or AD-conditioned tuning is called.

## Association, intervals and multiplicity

Only high-confidence regions proceed to the configured Firth models. For ancestry a, centering L_a at 1 makes the ROH coefficient equal to the contrast beta_ROH+beta_ROH×LA in the uncentered model. The model includes the local main effect, interaction, age, sex, PC1–PC5, and K−1 global ancestry components. Participants are not classified by dominant ancestry. Minimum weighted carrier support is 3; local-ancestry support and LA=1 extrapolation are reported.

Firth estimation maximizes logistic log likelihood plus 0.5 log determinant(X'WX). The adjusted score is X'[y−p+h(0.5−p)], where h is the diagonal of the weighted hat matrix. The constrained null fit preserves the same full design and information determinant; the 1-df likelihood-ratio statistic compares the two penalized likelihoods. The reported intervals use the inverse information matrix and are explicitly Wald approximations. These are not profile-likelihood confidence intervals. [Firth regression implementation/reference documentation](https://cran.r-project.org/web/packages/logistf/logistf.pdf)

Rank-deficient designs, insufficient carrier support, single outcome classes, insufficient complete samples, timeouts and nonconvergence produce explicit statuses. No failed fit enters nominal/FDR findings. BH correction and Bonferroni values are reported within the documented families. Tests use standard independent-subject logistic modeling; a kinship-aware analysis would be a separate design.

FROH summaries use union lengths within each subject/chromosome to avoid counting overlapping segments twice, divided by the sum of GRCh38 reference chromosome lengths for the configured autosomes. The HC burden is explicitly the union of member segment lengths for retained regions; it is distinct from the length of shared representative intervals.

## Provenance and real-data limits

The source functions and regional formulas were preserved in `scoring_math.py` and `regional_scoring.py`; relevant unmodified reference files and SHA-256 values are bundled. Runtime changes adapt string ID handling, cohort subsetting, strict input validation and real-data associations/reporting. The frozen configuration selects a deployable rule; it does not establish real-cohort sensitivity, precision, F1 or balanced accuracy. There are no observed truth labels in this real cohort, and AD association is not a substitute for ROH truth.
