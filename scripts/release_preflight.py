#!/usr/bin/env python3
"""Side-effect-free layout/syntax/private-input preflight: no model or dataset loaded."""
import argparse
import ast
import json
from pathlib import Path

REQUIRED_SOURCE = (
 'src/data/eligibility.py','src/data/xls_stdlib.py','src/data/person_split.py',
 'src/data/eeg_windows.py','src/models/raw_cnn.py','src/models/final_cnn_training.py',
 'src/evaluation/internal_test.py','src/evaluation/external_test.py',
 'src/analysis/wake_to_n1_evaluation.py','src/analysis/w2s_eval.py',
 'src/analysis/w2s_neural.py','src/analysis/w2s_p0a_diagnostic.py',
 'scripts/decision/h_eval/gate_e2/xls_stdlib.py',
 'scripts/decision/h_eval/gate_e6/gate_e6.py',
 'scripts/decision/h_eval/gate_e7b/gate_e7b.py',
 'scripts/decision/h_eval/gate_e7b/gate_e7b_v1_2.py',
 'scripts/decision/h_eval/cth_v1/cth_experiment.py',
 'scripts/decision/h_eval/cth_v2/cth_final_v2.py',
 'scripts/decision/h_eval/gate_t1/gate_t1.py',
 'scripts/decision/h_eval/gate_x1/gate_x1.py',
)
NEEDED_TO_REPLAY = (
 'data/raw/sleep_edfx_v1_0_0/metadata/SHA256SUMS.txt',
 'results/decision/h_eval/gate_e2/sleep_edf_origin_inventory.csv',
 'results/decision/h_eval/gate_e3/gate_e3_sc_group_split.json',
 '.local/h_eval/cth_v1/dataset_v1.npz',
 'docs/decision/h_eval/cth_v1/cth_v1_protocol.md',
 'results/decision/h_eval/gate_e7b/frozen_models_v1.json',
 'results/decision/h_eval/cth_v2/frozen_m2_46_final',
)

def inspect(root, tier):
    root = Path(root).resolve()
    parsed = {}
    for base in ('src', 'scripts/decision/h_eval'):
        folder = root / base
        if folder.is_dir():
            for file in sorted(folder.rglob('*.py')):
                rel = file.relative_to(root).as_posix()
                if file.is_symlink():
                    parsed[rel] = 'SYMLINK_NOT_ALLOWED'
                    continue
                try:
                    ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
                    parsed[rel] = 'OK'
                except (SyntaxError, UnicodeError, OSError) as e:
                    parsed[rel] = type(e).__name__
    missing = [f for f in REQUIRED_SOURCE if not (root / f).is_file()]
    problems = {k:v for k,v in parsed.items() if v != 'OK'}
    inputs = [f for f in NEEDED_TO_REPLAY if not (root / f).exists()] if tier == 'replay' else []
    if missing or problems:
        verdict = 'SOURCE_BLOCKED'
    elif inputs:
        verdict = 'REPLAY_BLOCKED_PRIVATE_ARTIFACTS'
    elif tier == 'replay':
        verdict = 'REPLAY_INPUTS_PRESENT_NOT_VALIDATED'
    else:
        verdict = 'SOURCE_LAYOUT_AND_SYNTAX_PASS_ONLY'
    return dict(verdict=verdict, tier=tier, checked_py_files=len(parsed),
                missing_source=missing, python_parse_failures=problems,
                missing_replay_inputs=inputs,
                limitations=['No research module imported','No EEG loaded',
                             'No model training/evaluation','No frozen results reproduced'])

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a.add_argument('--tier', choices=('source','replay'), default='source')
    args = a.parse_args()
    out = inspect(args.root,args.tier)
    print(json.dumps(out,indent=2,sort_keys=True))
    return 2 if out['verdict'] in ('SOURCE_BLOCKED','REPLAY_BLOCKED_PRIVATE_ARTIFACTS') else 0

if __name__ == '__main__':
    raise SystemExit(main())
