"""Import and command-line smoke tests (every documented command runs; invalid input is rejected with a message)."""
import csv
import json
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ['info', 'demo', 'split', 'score', 'fit-references', 'cohort', 'windows', 'fit-speclr', 'fit-rawcnn', 'predict']


def run(args, env, cwd, ok=True):
    p = subprocess.run([sys.executable, '-m', 'heval', *args], env=env, cwd=cwd, capture_output=True, text=True, timeout=900)
    assert (p.returncode == 0) == ok, p.stdout + p.stderr
    return p


def test_all_modules_import_without_data(tmp_path, cli_env):
    import heval
    names = [m.name for m in pkgutil.walk_packages(heval.__path__, 'heval.') if m.name != 'heval.__main__']
    assert len(names) >= 18
    code = 'import importlib,sys;[importlib.import_module(n) for n in sys.argv[1:]]'
    subprocess.run([sys.executable, '-c', code, *names], env=cli_env, cwd=tmp_path, check=True, timeout=300)   # foreign cwd


def test_sources_have_no_workstation_paths_or_legacy_imports():
    pattern = re.compile('|'.join(['/' + 'home/', '/' + 'media/', '/' + 'mnt/', 'scripts' + '/decision', '.local' + '/h_eval',
                                   'gate' + r'_e\d', 'repair' + '_r', 'cth' + '_v', 'spec_from_' + 'file_location']))
    hits = [f'{p.relative_to(ROOT)}:{i}' for p in [*ROOT.glob('src/**/*.py'), *ROOT.glob('tests/*.py'), ROOT / 'README.md']
            for i, line in enumerate(p.read_text().splitlines(), 1) if pattern.search(line)]
    assert hits == []


@pytest.mark.parametrize('cmd', COMMANDS)
def test_every_command_help(cmd, tmp_path, cli_env):
    assert 'usage' in run([cmd, '--help'], cli_env, tmp_path).stdout


def test_info_and_demo(tmp_path, cli_env):
    info = json.loads(run(['info'], cli_env, tmp_path).stdout)
    assert info['rawcnn_parameters'] == 114_817
    run(['demo', '--persons', '8', '--epochs', '1', '--boot', '200', '--out', 'demo.json'], cli_env, tmp_path)
    demo = json.loads((tmp_path / 'demo.json').read_text())
    assert demo['note'].startswith('SYNTHETIC') and set(demo['models']) == {'CONSTANT', 'SpecLR', 'RawCNN', 'Stage-only', 'Stage+time'}
    c = demo['stage_contribution']['CONSTANT->SpecLR']
    assert sum(c['absolute'].values()) == pytest.approx(c['delta_ww_all'], abs=1e-15)


def test_split_score_and_references(tmp_path, cli_env):
    (tmp_path / 'subjects.txt').write_text('\n'.join([f'SCsyn{i:02d}' for i in range(78)] + [f'STsyn{i:02d}' for i in range(17)]))
    split = json.loads(run(['split', '--subjects-file', 'subjects.txt'], cli_env, tmp_path).stdout)
    assert split['counts'] == {'SC_PROTECTED_MODEL_TEST': 16, 'SC_TRAIN': 46, 'SC_VALIDATION': 16, 'ST_EXTERNAL_RESERVED': 17}
    from heval.synthetic import synthetic_nights
    d = synthetic_nights(n_persons=10, seed=4, with_raw=False)
    with (tmp_path / 'pred.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['person', 'elapsed_s', 'stage', 'y', 'p_const', 'p_model'])
        for i in range(len(d['y'])):
            w.writerow([d['person'][i], d['elapsed_s'][i], d['stage'][i], d['y'][i], 0.3, 0.9 if d['stage'][i] == '1' else 0.1])
    out = json.loads(run(['score', '--predictions', 'pred.csv', '--ref', 'p_const', '--cand', 'p_model', '--stage-col', 'stage',
                          '--boot', '500'], cli_env, tmp_path).stdout)
    assert 'R_ci95' in out['contrast'] and set(out['by_stage']) == {'W', '1'}
    assert sum(out['stage_contribution']['absolute'].values()) == pytest.approx(out['contrast']['delta_ww'], abs=1e-12)
    refs = json.loads(run(['fit-references', '--train-csv', 'pred.csv'], cli_env, tmp_path).stdout)
    assert set(refs) >= {'A', 'B', 'B_selection'} and 0 < refs['A']['p_wake'] < refs['A']['p_stage1'] < 1


def test_window_files_fit_and_predict(tmp_path, cli_env, synthetic_psg):
    psg, _, _ = synthetic_psg
    with (tmp_path / 'cohort.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['recording', 'subject_key', 'status', 'psg_path', 'lights_off_rel_s', 'first_n2_after_off_s'])
        w.writerow(['SYN01', 'SYN0', 'ELIGIBLE_ORIGIN_INDEPENDENT', psg.name, 300.0, 1800.0])
        w.writerow(['SYN02', 'SYN0', 'NO_STAGES', psg.name, '', ''])
    run(['windows', '--data-root', str(tmp_path), '--cohort-csv', 'cohort.csv', '--recording', 'SYN01', '--out', 'w.npz'], cli_env, tmp_path)
    z = np.load(tmp_path / 'w.npz')
    assert z['raw'].shape == (49, 2, 3000) and z['spectral'].shape == (49, 10) and int(z['y'].sum()) == 10
    run(['fit-speclr', '--windows', 'w.npz', '--out', 'speclr.json'], cli_env, tmp_path)
    run(['fit-rawcnn', '--windows', 'w.npz', '--out-dir', 'cnn', '--epochs', '1'], cli_env, tmp_path)
    assert sorted(p.name for p in (tmp_path / 'cnn').iterdir()) == ['rawcnn_config.json', 'rawcnn_seed0.pt', 'rawcnn_seed1.pt', 'rawcnn_seed2.pt']
    run(['predict', '--windows', 'w.npz', '--speclr', 'speclr.json', '--rawcnn-dir', 'cnn', '--out', 'p.csv'], cli_env, tmp_path)
    rows = list(csv.DictReader((tmp_path / 'p.csv').open()))
    assert len(rows) == 49 and all(0 < float(r['p_rawcnn']) < 1 and 0 < float(r['p_speclr']) < 1 for r in rows)
    err = run(['windows', '--data-root', str(tmp_path), '--cohort-csv', 'cohort.csv', '--recording', 'SYN02', '--out', 'x.npz'],
              cli_env, tmp_path, ok=False).stderr
    assert 'not eligible' in err


def test_invalid_inputs_are_rejected_with_messages(tmp_path, cli_env):
    env = {k: v for k, v in cli_env.items() if k != 'HEVAL_SLEEP_EDF_ROOT'}
    assert 'give --data-root' in run(['cohort', '--out-dir', 'o'], env, tmp_path, ok=False).stderr
    assert 'not found' in run(['cohort', '--data-root', str(tmp_path), '--out-dir', 'o'], env, tmp_path, ok=False).stderr
    assert 'not found' in run(['score', '--predictions', 'missing.csv', '--ref', 'a', '--cand', 'b'], env, tmp_path, ok=False).stderr
    (tmp_path / 'bad.csv').write_text('person,y,a\nP1,1,0.5\n')
    assert 'lacks column' in run(['score', '--predictions', 'bad.csv', '--ref', 'a', '--cand', 'b'], env, tmp_path, ok=False).stderr
    (tmp_path / 'subjects.txt').write_text('XX01\n')
    assert 'must start with SC' in run(['split', '--subjects-file', 'subjects.txt'], env, tmp_path, ok=False).stderr
    (tmp_path / 'few.txt').write_text('SC01\nST01\n')
    assert 'Too few groups' in run(['split', '--subjects-file', 'few.txt'], env, tmp_path, ok=False).stderr
    assert 'not found' in run(['fit-speclr', '--windows', 'none.npz'], env, tmp_path, ok=False).stderr
