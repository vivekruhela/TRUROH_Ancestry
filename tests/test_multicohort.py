"""Regression coverage for explicit config routing and isolated cohort runs."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from test_pipeline import fixture, finish_fixture, ROOT
from common import atomic_json, file_hash, load_config, read_table, LOCK, carrier_min
from configuration import guard_output
from pipeline import check_manifest, config_signature, finalize, preflight, submit
from phenotypes import prepare_phenotypes
from ancestry import collapse_ancestry


def save_config(cfg, destination):
    destination = Path(destination)
    destination.write_text(json.dumps(cfg, indent=2) + '\n')
    return load_config(destination)


def ch_fixture(directory):
    cfg, _, _ = fixture(directory)
    p = pd.read_csv(cfg['phenotype_file'], sep='\t', dtype=str, keep_default_na=False)
    p = p.rename(columns={'fam': 'FID', 'id': 'IID', 'sex': 'Sex',
                          'Dx_binary': 'AD', 'Age Data Analysis': 'Age', 'apoe_wgs': 'APOE_reported'})
    p['AD'] = p.AD.map({'0': '1', '1': '2'})
    p.to_csv(cfg['phenotype_file'], sep='\t', index=False)
    cfg.update(project='CH', cohort_label='Caribbean Hispanics')
    cfg['columns'] = dict(fid='FID', iid='IID', age='Age', sex='Sex', outcome='AD', apoe_genotype='APOE_reported')
    cfg['outcome_map'] = {'1': 0, '2': 1}
    cfg['association']['covariates'] = ['Age', 'Sex', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    path = Path(directory) / 'ch.json'
    return save_config(cfg, path), path


class MultiCohortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='truroh cohorts ')
        self.base = Path(self.tmp.name)
        self.cfg, self.path = ch_fixture(self.base / 'CH')

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, args, env=None, cwd=None):
        runtime = os.environ.copy()
        runtime.pop('TRUROH_CONFIG', None)
        runtime.pop('TRUROH_MODULES', None)
        runtime.update(PYTHON_BIN=sys.executable, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if env:
            runtime.update(env)
        return subprocess.run(['bash', str(ROOT / 'run.sh'), *args], cwd=cwd or self.base,
                              env=runtime, capture_output=True, text=True)

    def test_ch_header_covariates_and_mapping(self):
        self.assertEqual(self.cfg['association']['covariates'][:2], ['AGE', 'SEX'])
        _, prepared, audit = prepare_phenotypes(self.cfg)
        self.assertEqual(audit['AD_controls'], 20)
        self.assertEqual(audit['AD_cases'], 20)
        self.assertEqual(audit['M1_complete_N'], 39)
        self.assertEqual(audit['cohort'], 'Caribbean Hispanics')
        self.assertIn('AGE', prepared)
        self.assertEqual(carrier_min(2278), 12)

    def test_explicit_custom_numeric_covariate(self):
        raw = json.loads(self.path.read_text())
        raw['columns']['covariates'] = {'EDUCATION': 'Years of education'}
        raw['association']['covariates'].append('Years of education')
        p = pd.read_csv(raw['phenotype_file'], sep='\t', dtype=str, keep_default_na=False)
        p['Years of education'] = [str(i % 15) for i in range(len(p))]
        p.loc[0, 'Years of education'] = 'NA'
        p.to_csv(raw['phenotype_file'], sep='\t', index=False)
        cfg = save_config(raw, self.path)
        _, prepared, audit = prepare_phenotypes(cfg)
        self.assertEqual(cfg['association']['covariates'][-1], 'EDUCATION')
        self.assertTrue(pd.isna(prepared.EDUCATION.iloc[0]))
        self.assertEqual(audit['M1_complete_N'], 38)

    def test_no_default_cohort_in_either_launcher(self):
        for script, args in [('run.sh', ['preflight']), ('submit.sh', [])]:
            env = dict(os.environ, PYTHON_BIN=sys.executable)
            env.pop('TRUROH_CONFIG', None)
            r = subprocess.run(['bash', str(ROOT / script), *args], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('No config is selected by default', r.stderr)
            self.assertNotIn('Missing input files', r.stderr)
        self.assertFalse(Path(self.cfg['output_root']).exists())

    def test_flag_before_after_action_and_environment(self):
        forms = [(['--config', str(self.path), 'config'], {}),
                 (['config', '--config', str(self.path)], {}),
                 (['config'], {'TRUROH_CONFIG': str(self.path)}),
                 (['config', '--config', str(self.path)], {'TRUROH_CONFIG': '/bad/wrong.json'})]
        for args, env in forms:
            r = self.cli(args, env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)['project'], 'CH')
            self.assertIn(str(self.path), r.stderr)

    def test_relative_paths_are_config_relative_and_snapshots_absolute(self):
        raw = json.loads(self.path.read_text())
        for key in ['plink_prefix', 'phenotype_file', 'pca_file', 'rfmix_msp_pattern', 'output_root']:
            raw[key] = os.path.relpath(raw[key], self.path.parent)
        cfg = save_config(raw, self.path)
        self.assertEqual(cfg, self.cfg)
        snapshot = self.base / 'elsewhere' / 'frozen.json'
        snapshot.parent.mkdir()
        self.assertEqual(save_config(cfg, snapshot), cfg)

    def test_invalid_covariate_is_actionable(self):
        raw = json.loads(self.path.read_text())
        raw['association']['covariates'] = ['Age', 'NotMapped']
        with self.assertRaisesRegex(ValueError, 'NotMapped.*columns.covariates'):
            save_config(raw, self.path)

    def test_foreign_output_fails_without_overwriting_report_or_status(self):
        preflight(self.cfg, check_external=False)
        root = Path(self.cfg['output_root'])
        before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
        wrong = copy.deepcopy(self.cfg);wrong.update(project='GAPP', cohort_label='GAPP Peruvians')
        wrong_path = self.base / 'wrong.json';save_config(wrong, wrong_path)
        r = self.cli(['--config', str(wrong_path), 'preflight'])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('belongs to project', r.stderr)
        self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_scientific_changes_block_reuse_scheduling_changes_do_not(self):
        preflight(self.cfg, check_external=False)
        cfg = copy.deepcopy(self.cfg);cfg['resources']['cpus'] = 12
        self.assertEqual(config_signature(cfg), config_signature(self.cfg))
        check_manifest(cfg)
        cfg['association']['covariates'] = ['AGE']
        self.assertNotEqual(config_signature(cfg), config_signature(self.cfg))
        with self.assertRaisesRegex(ValueError, 'Configuration or source changed'):
            check_manifest(cfg)

    def test_two_cohorts_have_separate_outputs(self):
        gapp, _, _ = fixture(self.base / 'GAPP')
        gapp_path = self.base / 'GAPP/gapp.json';gapp = save_config(gapp, gapp_path)
        for cfg in [gapp, self.cfg]:
            manifest = preflight(cfg, check_external=False)
            self.assertEqual(manifest['project'], cfg['project'])
            self.assertEqual(manifest['cohort'], cfg['cohort_label'])
        self.assertNotEqual(config_signature(gapp), config_signature(self.cfg))

    def test_submission_freezes_config_and_propagates_resources(self):
        self.cfg['resources'].update(constraint='cpu8mem64a', final_constraint='cpu16mem128a')
        calls = []
        def fake_sbatch(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=f'{9000 + len(calls)}\n', stderr='')
        with patch('pipeline.check_tools', return_value={'status': 'fixture'}), patch('pipeline.shutil.which', return_value='/fake/sbatch'), patch('pipeline.subprocess.run', side_effect=fake_sbatch):
            record = submit(self.cfg, config_path=self.path)
        self.assertEqual(len(calls), 2)
        frozen = Path(record['frozen_config'])
        self.assertIn('/submissions/', str(frozen))
        self.assertEqual(file_hash(frozen), record['config_sha256'])
        for command, stage in zip(calls, ['chromosome', 'finalize']):
            self.assertEqual(command[-4:], [str(ROOT), str(frozen), stage, record['config_sha256']])
            self.assertIn('TRUROH_CH_1433' + ('_report' if stage == 'finalize' else ''), command)
            self.assertNotIn('gapp.json', ' '.join(command))
        self.assertIn('cpu8mem64a', calls[0]);self.assertIn('cpu16mem128a', calls[1])
        self.assertIn('afterany:9001', calls[1])
        self.path.write_text('{"project":"GAPP"}\n')
        self.assertEqual(load_config(frozen)['project'], 'CH')
        check_manifest(load_config(frozen))

    def test_dry_run_never_calls_sbatch(self):
        with patch('pipeline.check_tools', return_value={'status': 'fixture'}), patch('pipeline.subprocess.run') as runner:
            record = submit(self.cfg, dry_run=True)
        runner.assert_not_called()
        self.assertTrue(record['dry_run'])
        self.assertIsNone(record['array_job_id'])
        self.assertTrue((Path(self.cfg['output_root']) / 'submission.plan.json').exists())

    def test_frozen_file_tampering_fails_before_output_write(self):
        digest = file_hash(self.path)
        self.path.write_text(self.path.read_text() + '\n')
        r = self.cli(['--config', str(self.path), '--expected-config-sha256', digest, 'preflight'])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('checksum mismatch', r.stderr)
        self.assertFalse(Path(self.cfg['output_root']).exists())

    def test_actual_worker_shell_forwards_frozen_config_and_stage(self):
        capture = self.base / 'captured.json'
        fake_python = self.base / 'capture_python'
        fake_python.write_text('#!' + sys.executable + '\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE_PATH"]).write_text(json.dumps({"args":sys.argv[1:],"cwd":os.getcwd()}))\n')
        fake_python.chmod(0o755)
        env = dict(os.environ, PYTHON_BIN=str(fake_python), CAPTURE_PATH=str(capture), TRUROH_CONFIG='/bad/gapp.json')
        for stage in ['chromosome', 'finalize']:
            command = ['bash', str(ROOT / 'slurm/job.sbatch'), str(ROOT), str(self.path), stage, file_hash(self.path)]
            r = subprocess.run(command, cwd=self.base, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            captured = json.loads(capture.read_text())
            self.assertEqual(captured['args'], [str(ROOT / 'src/pipeline.py'), '--config', str(self.path), '--expected-config-sha256', file_hash(self.path), stage])
            self.assertEqual(captured['cwd'], str(ROOT))

    def test_submit_shell_forwards_flag_and_legacy_explicit_argument(self):
        capture = self.base / 'submit_args.json'
        fake_python = self.base / 'capture_submit'
        fake_python.write_text('#!' + sys.executable + '\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE_PATH"]).write_text(json.dumps(sys.argv[1:]))\n')
        fake_python.chmod(0o755)
        env = dict(os.environ, PYTHON_BIN=str(fake_python), CAPTURE_PATH=str(capture))
        for form in [['--config', str(self.path), '--dry-run'], [str(self.path), '--dry-run']]:
            r = subprocess.run(['bash', str(ROOT / 'submit.sh'), *form], cwd=self.base, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(capture.read_text()), [str(ROOT / 'src/pipeline.py'), 'submit', '--config', str(self.path), '--dry-run'])

    def test_ch_full_pipeline_report_and_dynamic_primary_ancestry(self):
        self.cfg['association'].update(primary_ancestry='EUR', ancestries=['EUR'])
        root = finish_fixture(self.cfg)
        finalize(self.cfg)
        html_path = root / 'report/TRUROH_CH_report.html'
        document = html_path.read_text()
        self.assertIn('Caribbean Hispanics', document)
        self.assertIn('TRUROH_CH_report.pdf', document)
        self.assertNotIn('GAPP', document)
        self.assertTrue((root / 'report/TRUROH_CH_report.pdf').is_file())
        self.assertFalse((root / 'report/TRUROH_GAPP_report.pdf').exists())
        self.assertTrue((root / 'results/primary_EUR_associations.tsv').is_file())
        self.assertFalse((root / 'results/primary_NAM_associations.tsv').exists())
        summary = json.loads((root / 'report/summary.json').read_text())
        self.assertEqual(summary['state'], 'COMPLETE')
        self.assertEqual(summary['locked'], LOCK)
        self.assertEqual(read_table(root / 'results/regions_scored.tsv').HIGH_CONFIDENCE.tolist(), [1, 1])

    def test_auto_primary_ancestry_and_nominal_only_outputs(self):
        self.cfg['association'].update(primary_ancestry='auto', ancestries=['NAM', 'EUR', 'AFR'])
        root = finish_fixture(self.cfg)
        finalize(self.cfg)
        sel = json.loads((root / 'results/primary_ancestry_selection.json').read_text())
        q = collapse_ancestry(read_table(root / 'results/global_ancestry.tsv'), 'Q_')
        means = {c[2:]: q[c].mean() for c in q if c[2:] in ['NAM', 'EUR', 'AFR']}
        self.assertEqual(sel['method'], 'auto')
        self.assertEqual(sel['selected'], max(means, key=means.get))
        self.assertTrue((root / f"results/primary_{sel['selected']}_associations.tsv").is_file())
        self.assertEqual(json.loads((root / 'results/association_models.json').read_text())['primary_ancestry'], sel['selected'])
        assoc = read_table(root / 'results/associations.tsv')
        self.assertFalse({'Q_VALUE', 'BONFERRONI_P', 'FDR_ASSOCIATED'} & set(assoc.columns))
        self.assertFalse((root / 'results/AD_associated_FDR.tsv').exists())
        self.assertNotIn('FDR', (root / 'report/TRUROH_CH_report.html').read_text())

    def test_explicit_primary_ancestry_is_kept_and_noted(self):
        self.cfg['association'].update(primary_ancestry='AFR', ancestries=['NAM', 'EUR', 'AFR'])
        root = finish_fixture(self.cfg)
        finalize(self.cfg)
        sel = json.loads((root / 'results/primary_ancestry_selection.json').read_text())
        self.assertEqual((sel['method'], sel['selected']), ('explicit', 'AFR'))
        self.assertEqual(bool(sel['notes']), sel['majority'] != 'AFR')

    def test_auto_primary_ancestry_config_validation(self):
        cfg = json.loads(self.path.read_text())
        cfg['association']['primary_ancestry'] = 'AUTO'
        path = self.base / 'auto.json'; path.write_text(json.dumps(cfg))
        self.assertEqual(load_config(path)['association']['primary_ancestry'], 'auto')
        cfg['association']['primary_ancestry'] = 'SAS'; path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'auto'):
            load_config(path)

    def test_missing_ch_input_generates_ch_diagnostic_report(self):
        Path(self.cfg['phenotype_file']).unlink()
        r = self.cli(['preflight', '--config', str(self.path)])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Missing input files', r.stderr)
        self.assertNotIn('Diagnostic report error', r.stderr)
        root = Path(self.cfg['output_root'])
        self.assertTrue((root / 'report/TRUROH_CH_report.html').is_file())
        self.assertTrue((root / 'report/TRUROH_CH_report.pdf').is_file())
        summary = json.loads((root / 'report/summary.json').read_text())
        self.assertEqual(summary['project'], 'CH')
        self.assertEqual(summary['state'], 'INCOMPLETE')
        self.assertIsNone(summary['high_confidence_ROH'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
