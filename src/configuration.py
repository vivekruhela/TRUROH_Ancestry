"""Cohort-independent configuration, explicit selection, and output ownership."""
from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path

from common import ROOT, LOCK, canonical, chromosomes, fingerprint

VERSION = (ROOT / 'VERSION').read_text().strip()
STANDARD_COVARIATES = ['AGE', 'SEX', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'APOE4']
RESOURCE_DEFAULTS = {
    'cpus': 4, 'memory': '64G', 'time': '2-00:00:00', 'max_concurrent': 4,
    'partition': '', 'account': '', 'constraint': '',
    'final_cpus': 4, 'final_memory': '32G', 'final_time': '1-00:00:00',
    'final_constraint': '',
}
ASSOCIATION_DEFAULTS = {
    'enabled': True, 'covariates': STANDARD_COVARIATES[:-1],
    'ancestries': ['NAM', 'EUR', 'AFR'], 'primary_ancestry': 'NAM',
    'global_reference': 'EUR', 'run_apoe_sensitivity': True,
    'min_carriers': 3, 'min_noncarriers': 3, 'min_weighted_carriers': 3.0,
    'max_iterations': 300, 'score_tolerance': 1e-5, 'alpha': .05,
    'max_fit_seconds': 60,
}


def _object(value, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be a JSON object')
    return value


def _unknown_keys(value, allowed, where):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {where} field(s): {", ".join(sorted(unknown))}')


def normalize_covariates(cfg):
    """Accept canonical names or explicitly mapped source headers; never guess coding."""
    columns = cfg['columns']
    extra = columns.get('covariates', {})
    aliases = {x.casefold(): x for x in STANDARD_COVARIATES}
    for key, target in [('age', 'AGE'), ('sex', 'SEX')]:
        source = columns[key].casefold()
        if source in aliases and aliases[source] != target:
            raise ValueError(f'columns.{key} conflicts with reserved covariate {columns[key]!r}')
        aliases[source] = target
    for target, source in extra.items():
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', target):
            raise ValueError(f'Invalid custom covariate name: {target!r}')
        if target.upper() in set(STANDARD_COVARIATES) | {'AD', 'FID', 'IID', 'ROH', 'LA_TARGET', 'LA_CENTERED', 'ROHXLA_CENTERED', 'ROHXAPOE4', 'INTERCEPT', 'SEX_FAM'} or target.upper().startswith(('Q_', 'LA_', 'BP_')):
            raise ValueError(f'Custom covariate {target!r} conflicts with a pipeline variable')
        for alias in [target.casefold(), source.casefold()]:
            if alias in aliases and aliases[alias] != target:
                raise ValueError(f'Ambiguous covariate mapping: {target!r} / {source!r}')
            aliases[alias] = target
    requested = cfg['association']['covariates']
    if not isinstance(requested, list) or any(not isinstance(x, str) for x in requested):
        raise ValueError('association.covariates must be a list of names')
    unsupported = [x for x in requested if x.casefold() not in aliases]
    if unsupported:
        raise ValueError(
            f'Unsupported association covariate(s): {unsupported}. Use {STANDARD_COVARIATES} '
            'or define additional numeric variables in columns.covariates.'
        )
    normalized = [aliases[x.casefold()] for x in requested]
    if len(normalized) != len(set(normalized)):
        raise ValueError('association.covariates contains duplicate variables after name normalization')
    if cfg['association']['run_apoe_sensitivity'] and 'APOE4' in normalized:
        raise ValueError('APOE4 is added by the APOE sensitivity models; remove it from the baseline covariates or disable run_apoe_sensitivity')
    return normalized


def analysis_config(cfg):
    # Scheduling and display changes do not invalidate biological outputs.
    return {k: v for k, v in cfg.items() if k not in {'resources', 'cohort_label', 'keep_intermediate_vcf'}}


def analysis_fingerprint(cfg):
    return fingerprint(analysis_config(cfg))


def read_config(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Configuration file not found: {path}')
    try:
        cfg = _object(json.loads(path.read_text()), 'Configuration')
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {path}, line {exc.lineno}: {exc.msg}') from exc
    allowed = {
        'project', 'cohort_label', 'genome_build', 'chromosomes', 'plink_prefix',
        'phenotype_file', 'pca_file', 'output_root', 'rfmix_msp_pattern',
        'rfmix_sample_map', 'gene_annotation_file', 'columns', 'outcome_map',
        'sex_map', 'missing_values', 'association', 'tools', 'resources',
        'keep_intermediate_vcf',
    }
    _unknown_keys(cfg, allowed, 'configuration')
    required = ['project', 'genome_build', 'chromosomes', 'plink_prefix', 'phenotype_file',
                'pca_file', 'output_root', 'rfmix_msp_pattern', 'columns', 'outcome_map', 'sex_map']
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f'Missing configuration fields: {missing}')
    if not isinstance(cfg['project'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', cfg['project']):
        raise ValueError('project must be 1–64 filename-safe letters, digits, dots, underscores or hyphens, starting with a letter or digit')
    cfg.setdefault('cohort_label', cfg['project'])
    if not isinstance(cfg['cohort_label'], str) or not cfg['cohort_label'].strip():
        raise ValueError('cohort_label must be a nonempty display name')
    cfg['cohort_label'] = cfg['cohort_label'].strip()
    if cfg['genome_build'] != 'GRCh38':
        raise ValueError('GRID_ID 1433 deployment currently requires GRCh38')
    if json.loads((ROOT / 'config/locked_grid1433.json').read_text()) != LOCK:
        raise ValueError('Locked GRID_ID 1433 was changed; restore config/locked_grid1433.json')
    cfg['chromosomes'] = ','.join(map(str, chromosomes(cfg['chromosomes'])))
    for key in ['rfmix_sample_map', 'gene_annotation_file']:
        cfg.setdefault(key, '')
    for key in ['plink_prefix', 'phenotype_file', 'pca_file', 'output_root', 'rfmix_msp_pattern', 'rfmix_sample_map', 'gene_annotation_file']:
        value = cfg[key]
        if not isinstance(value, str) or (not value.strip() and key not in {'rfmix_sample_map', 'gene_annotation_file'}):
            raise ValueError(f'{key} must be a nonempty path')
        if value:
            p = Path(value).expanduser()
            cfg[key] = str((p if p.is_absolute() else path.parent / p).resolve())
    pattern = cfg['rfmix_msp_pattern']
    if '{CHR}' not in pattern and '{chr}' not in pattern:
        raise ValueError('rfmix_msp_pattern must include {CHR} or {chr}')
    if re.search(r'[{}]', pattern.replace('{CHR}', '').replace('{chr}', '')):
        raise ValueError('rfmix_msp_pattern contains an unsupported placeholder')
    if Path(cfg['output_root']) == Path('/') or Path(cfg['output_root']) == ROOT:
        raise ValueError('output_root must be a dedicated analysis directory')
    columns = _object(cfg['columns'], 'columns')
    _unknown_keys(columns, {'fid', 'iid', 'outcome', 'age', 'sex', 'apoe_genotype', 'covariates'}, 'columns')
    for key in ['fid', 'iid', 'outcome', 'age', 'sex']:
        if not isinstance(columns.get(key), str) or not columns[key]:
            raise ValueError(f'columns.{key} must name an input column')
    if len({columns[key] for key in ['fid', 'iid', 'outcome', 'age', 'sex']}) != 5:
        raise ValueError('FID, IID, outcome, age and sex must map to distinct input columns')
    columns.setdefault('apoe_genotype', '')
    if not isinstance(columns['apoe_genotype'], str):
        raise ValueError('columns.apoe_genotype must be a source header or an empty string')
    columns.setdefault('covariates', {})
    extra = _object(columns['covariates'], 'columns.covariates')
    if any(not isinstance(v, str) or not v for v in extra.values()):
        raise ValueError('Each columns.covariates value must be an input header')
    for key in ['outcome_map', 'sex_map']:
        mapping = _object(cfg[key], key)
        if not mapping or any(type(v) is not int or v not in {0, 1} for v in mapping.values()) or set(mapping.values()) != {0, 1}:
            raise ValueError(f'{key} must explicitly map source codes to both integer values 0 and 1')
    cfg.setdefault('missing_values', ['', 'NA', 'N/A', 'NaN', 'nan', '-9', '.', 'None'])
    if not isinstance(cfg['missing_values'], list) or any(not isinstance(x, str) for x in cfg['missing_values']):
        raise ValueError('missing_values must be a list of strings')
    association = _object(cfg.setdefault('association', {}), 'association')
    _unknown_keys(association, ASSOCIATION_DEFAULTS, 'association')
    cfg['association'] = {**copy.deepcopy(ASSOCIATION_DEFAULTS), **association}
    association = cfg['association']
    for key in ['enabled', 'run_apoe_sensitivity']:
        if type(association[key]) is not bool:
            raise ValueError(f'association.{key} must be true or false')
    if not isinstance(association['ancestries'], list) or not association['ancestries']:
        raise ValueError('association.ancestries must be a nonempty list')
    for value in association['ancestries'] + [association['primary_ancestry'], association['global_reference']]:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_]+', value):
            raise ValueError(f'Invalid ancestry label: {value!r}')
    association['ancestries'] = [canonical(x) for x in association['ancestries']]
    association['primary_ancestry'] = canonical(association['primary_ancestry'])
    association['global_reference'] = canonical(association['global_reference'])
    if len(set(association['ancestries'])) != len(association['ancestries']):
        raise ValueError('association.ancestries contains duplicate ancestry aliases')
    if association['primary_ancestry'] not in association['ancestries']:
        raise ValueError('primary_ancestry must be listed in association.ancestries')
    for key in ['min_carriers', 'min_noncarriers', 'max_iterations']:
        if type(association[key]) is not int or association[key] < 1:
            raise ValueError(f'association.{key} must be a positive integer')
    for key in ['min_weighted_carriers', 'score_tolerance', 'alpha', 'max_fit_seconds']:
        value = association[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'association.{key} must be positive and finite')
    if not 0 < association['alpha'] < 1:
        raise ValueError('association.alpha must lie strictly between 0 and 1')
    association['covariates'] = normalize_covariates(cfg)
    resources = _object(cfg.setdefault('resources', {}), 'resources')
    _unknown_keys(resources, RESOURCE_DEFAULTS, 'resources')
    cfg['resources'] = {**RESOURCE_DEFAULTS, **resources}
    resources = cfg['resources']
    for key in ['cpus', 'final_cpus', 'max_concurrent']:
        if type(resources[key]) is not int or resources[key] < (0 if key == 'max_concurrent' else 1):
            raise ValueError(f'resources.{key} must be a positive integer (max_concurrent may be 0)')
    for key in set(resources) - {'cpus', 'final_cpus', 'max_concurrent'}:
        if not isinstance(resources[key], str) or '\n' in resources[key] or '\x00' in resources[key]:
            raise ValueError(f'resources.{key} must be a single-line string')
    for key in ['memory', 'time', 'final_memory', 'final_time']:
        if not resources[key].strip():
            raise ValueError(f'resources.{key} must not be empty')
    tools = _object(cfg.setdefault('tools', {}), 'tools')
    _unknown_keys(tools, {'plink', 'bcftools'}, 'tools')
    cfg['tools'] = {'plink': 'plink', 'bcftools': 'bcftools', **tools}
    for key, value in cfg['tools'].items():
        if not isinstance(value, str) or not value:
            raise ValueError(f'tools.{key} must name an executable')
        if '/' in value:
            p = Path(value).expanduser()
            cfg['tools'][key] = str((p if p.is_absolute() else path.parent / p).resolve())
    cfg.setdefault('keep_intermediate_vcf', False)
    if type(cfg['keep_intermediate_vcf']) is not bool:
        raise ValueError('keep_intermediate_vcf must be true or false')
    return cfg


def guard_output(cfg, *, claim=False):
    """Reject a foreign output directory before any reports or status files are written."""
    root = Path(cfg['output_root'])
    owner = root / 'run_identity.json'
    if owner.exists():
        existing = json.loads(owner.read_text()).get('project')
        if existing != cfg['project']:
            raise ValueError(f'output_root belongs to project {existing!r}, not {cfg["project"]!r}; choose a separate output_root')
    else:
        resolved = root / 'config.resolved.json'
        if resolved.exists():
            existing = json.loads(resolved.read_text()).get('project')
            if existing != cfg['project']:
                raise ValueError(f'Existing output_root belongs to project {existing!r}; choose a separate output_root')
        if claim:
            root.mkdir(parents=True, exist_ok=True)
            # Publish a complete owner record atomically; competing projects cannot replace it.
            import tempfile
            descriptor, temporary = tempfile.mkstemp(prefix='.identity-', dir=root)
            try:
                with os.fdopen(descriptor, 'w') as handle:
                    json.dump({'project': cfg['project'], 'package_version': VERSION}, handle)
                    handle.write('\n')
                try:
                    os.link(temporary, owner)
                except FileExistsError:
                    pass
            finally:
                Path(temporary).unlink(missing_ok=True)
            guard_output(cfg)
    return root
