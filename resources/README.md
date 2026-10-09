# Bundled annotation

`hg38_ucsc_refgene_genes.tsv.gz` is the preexisting GRCh38/UCSC RefGene gene-interval resource copied from the saved TRUROH_Ancestry_v3.2.5 package. Fields are CHR, BP1, BP2, gene and strand; BP1/BP2 are used as 1-based inclusive endpoints. It is provided for coordinate-overlap gene labels, with no claim of a newly downloaded annotation version. The catalog does not affect technical selection or association testing. Supply an alternative same-schema GRCh38 table in `gene_annotation_file` to use another documented annotation release.
