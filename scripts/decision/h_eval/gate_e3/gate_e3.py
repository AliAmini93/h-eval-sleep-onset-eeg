#!/usr/bin/env python3
"""H-EVAL Gate E3: source-verified stage audit, SC subject split, development-only time controls.

Only official cached Sleep-EDF hypnograms and prior result CSV/JSON files are used.
No EEG data, no external packages, no GPU, no protected Muse test, and NO SC test predictions.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 'H-EVAL-GateE3-stage-audit-SC-development-v1-2026-10-22'
SALT = 'H-EVAL-SC-subject-groups-20261022-v1'
HORIZON = 300
LOOKBACK = 1200
BINS = (300., 600., 1200., 1800., 3600., 7200.)
ELIGIBLE = 'ELIGIBLE_ORIGIN_INDEPENDENT'

def hash_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def immutable_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing artifact differs; refusing overwrite: ' + str(path))
    else:
        with path.open('xb') as f:
            f.write(data)

def csv_blob(rows):
    if not rows:
        raise ValueError('empty rows')
    keys = sorted({k for r in rows for k in r})
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode('utf-8')

def import_gate_e(repo):
    path = repo/'scripts/decision/h_eval/gate_e/gate_e.py'
    spec = importlib.util.spec_from_file_location('heval_gate_e_read_only', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_inputs(repo):
    evidence = repo/'results/decision/h_eval/gate_e2'
    summary_path = evidence/'sleep_edf_anchor_eligibility.json'
    inventory_path = evidence/'sleep_edf_origin_inventory.csv'
    if not summary_path.is_file() or not inventory_path.is_file():
        raise ValueError('Gate E2 official audit missing')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('origin_independent_eligible_recordings_n') != 172 or summary.get('origin_independent_subject_groups_n') != 94:
        raise ValueError('Prior E2 eligibility changed: independent review needed')
    with inventory_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 197 or len({r['recording'] for r in rows}) != 197:
        raise ValueError('E2 recording inventory not unique and complete')
    if dict(Counter(x['status'] for x in rows)) != summary.get('status_counts'):
        raise ValueError('Gate E2 summary status counts do not match rows')
    if sum(x['status'] == ELIGIBLE for x in rows) != 172:
        raise ValueError('Gate E2 eligible row count mismatch')
    groups = {r['subject_key'] for r in rows if r['status'] == ELIGIBLE}
    if len(groups) != 94:
        raise ValueError('E2 group count mismatch')
    sums = import_gate_e(repo).sums_parse((repo/'data/raw/sleep_edfx_v1_0_0/metadata/SHA256SUMS.txt').read_bytes())
    return rows, summary, sums, {'gate_e2_json': hash_file(summary_path), 'gate_e2_inventory': hash_file(inventory_path)}

def stage_at(events, t):
    if t is None or not math.isfinite(t):
        return 'NO_ORIGIN'
    active = [lab.lower().removeprefix('sleep stage ').strip() for x,d,lab in events
              if lab.lower().startswith('sleep stage ') and x-1.e-7 <= t < x+d-1.e-7]
    return active[0].upper() if len(active) == 1 else ('AMBIGUOUS' if active else 'UNSCORED')

def gap_details(events, off, end):
    if off is None or end is None or not off < end:
        return {'issue':'NO_VALID_ORIGIN_OR_TARGET'}
    stages = sorted(((x, x+d,lab) for x,d,lab in events if lab.lower().startswith('sleep stage ')),key=lambda z:z[0])
    current=off
    for x,b,lab in stages:
        left=max(x,off);right=min(b,end)
        if right <= left:continue
        if left > current+0.05:
            marker = next((nm for k,d,nm in events if not nm.lower().startswith('sleep stage ') and k < left and k+d > current), None)
            return {'issue':'GAP','problem_start_s':round(current,3),'problem_end_s':round(left,3),
                    'preceding_nonstage_annotation':marker or ''}
        if left < current-0.05:
            return {'issue':'OVERLAP','problem_start_s':round(left,3),'problem_end_s':round(min(current,right),3)}
        if lab.lower() in ('sleep stage ?', 'sleep stage m', 'sleep stage unknown'):
            return {'issue':'UNKNOWN_STAGE','problem_start_s':round(left,3),'problem_end_s':round(right,3)}
        current=max(current,right)
    if current < end-0.05:
        return {'issue':'TRAILING_GAP','problem_start_s':round(current,3),'problem_end_s':round(end,3)}
    return {'issue':'CONTIGUOUS'}

def diagnostics(rows, repo, sums, decoder):
    out=[]
    raw_root=repo/'data/raw/sleep_edfx_v1_0_0'
    for row in rows:
        hyp=raw_root/row['hypnogram_path']
        expected=sums.get(row['hypnogram_path'])
        if not hyp.is_file() or expected is None or hash_file(hyp)!=expected:
            raise ValueError('Cached official hypnogram missing/changed: '+row['recording'])
        ev,_=decoder.decode_edfplus_annotations(hyp.read_bytes())
        off = float(row['lights_off_rel_s']) if row.get('lights_off_rel_s') else None
        first = float(row['first_n2_after_off_s']) if row.get('first_n2_after_off_s') else None
        o={'recording':row['recording'],'study':row['study'],'subject_key':row['subject_key'],
           'e2_status':row['status'],'stage_at_lights_off':stage_at(ev,off),
           'lights_off_rel_s':row.get('lights_off_rel_s',''),'first_n2_after_off_s':row.get('first_n2_after_off_s',''),
           'e2_eligible':row['status']==ELIGIBLE}
        if row['status']=='SCORING_GAP_OR_OVERLAP':
            o.update(gap_details(ev,off,first))
        else:
            o['issue'] = 'NONE' if row['status']==ELIGIBLE else row['status']
        out.append(o)
    return out

def group_split(eligible_rows):
    by_study=defaultdict(set)
    for r in eligible_rows:
        by_study[r['study']].add(r['subject_key'])
    if len(by_study['sleep-cassette']) < 60 or len(by_study['sleep-telemetry']) < 10:
        raise ValueError('Too few groups for predeclared exploratory allocation')
    ordered=sorted(by_study['sleep-cassette'],key=lambda s:hashlib.sha256((SALT+'|'+s).encode()).hexdigest())
    n=len(ordered);n_test=round(n*.20);n_val=round(n*.20)
    if not n>n_test+n_val+25:raise ValueError('Insufficient development SC groups')
    split={s:'SC_PROTECTED_MODEL_TEST' for s in ordered[:n_test]}
    split.update({s:'SC_VALIDATION' for s in ordered[n_test:n_test+n_val]})
    split.update({s:'SC_TRAIN' for s in ordered[n_test+n_val:]})
    for s in by_study['sleep-telemetry']:
        if s in split:raise ValueError('SC-ST subject ID collision')
        split[s]='ST_EXTERNAL_RESERVED'
    return split

def make_landmarks(row):
    off=float(row['lights_off_rel_s']);first=float(row['first_n2_after_off_s'])
    first_k=math.ceil((off+30.)/30.-1.e-9)
    last_k=math.floor(first/30.+1.e-9)
    landmarks=[]
    for k in range(first_k,last_k+1):
        t=k*30.
        if t>=first-1.e-8:continue
        landmarks.append((t-off,int(first<=t+HORIZON+1.e-8),first-t<=LOOKBACK+1.e-8))
    if (len(landmarks),sum(a for _,_,a in landmarks),sum(y for _,y,_ in landmarks))!=\
       (int(row['landmark_n']),int(row['anchored_lb1200_n']),int(row['positive_h300_n'])):
        raise ValueError('E2 risk landmark mismatch: '+row['recording'])
    return landmarks

def fixed_bin(elapsed):
    return sum(elapsed>=p for p in BINS)

def fit(landmarks, support):
    data=[(t,y) for t,y,anch in landmarks if support=='FULL' or anch]
    if not data:raise ValueError('Empty SC training support')
    p=(1+sum(y for _,y in data))/(2+len(data))
    bins=[[0,0] for _ in range(len(BINS)+1)]
    for t,y in data:
        v=bins[fixed_bin(t)];v[0]+=y;v[1]+=1
    return {'CONSTANT':p,'TIME_BIN':[(pos+5*p)/(num+5) for pos,num in bins]}

def losses(p,y):
    p=max(1.e-6,min(1-1.e-6,p))
    return (p-y)**2,-(y*math.log(p)+(1-y)*math.log1p(-p))

def score_sc_development(rows,split):
    eligible=[r for r in rows if r['status']==ELIGIBLE and r['study']=='sleep-cassette']
    by_participant=defaultdict(list)
    for r in eligible:
        by_participant[split[r['subject_key']]].append(r)
    train=by_participant['SC_TRAIN'];val=by_participant['SC_VALIDATION']
    if not train or not val:raise ValueError('Empty development partition')
    if {r['subject_key'] for r in train}&{r['subject_key'] for r in val}:
        raise ValueError('Subject-group leakage')
    # The protected model-test and ST rows are deliberately not accessed here.
    per=[]
    for trsupport in ('FULL','ANCHOR'):
        tr=[(t,y,a) for r in train for t,y,a in make_landmarks(r)]
        model=fit(tr,trsupport)
        for evsupport in ('FULL','ANCHOR'):
            for r in val:
                landmarks=[(t,y) for t,y,a in make_landmarks(r) if evsupport=='FULL' or a]
                if not landmarks:continue
                for kind in ('CONSTANT','TIME_BIN'):
                    sc=[losses(model[kind] if kind=='CONSTANT' else model[kind][fixed_bin(t)],y) for t,y in landmarks]
                    per.append({'subject_key':r['subject_key'],'recording':r['recording'],
                                'train_support':trsupport,'eval_support':evsupport,'model':kind,
                                'n_landmarks':len(landmarks),'brier':sum(z[0] for z in sc)/len(sc),
                                'log_loss':sum(z[1] for z in sc)/len(sc)})
    metrics=[]
    for tr in ('FULL','ANCHOR'):
        for ev in ('FULL','ANCHOR'):
            for kind in ('CONSTANT','TIME_BIN'):
                subset=[r for r in per if r['train_support']==tr and r['eval_support']==ev and r['model']==kind]
                groups=defaultdict(list)
                for r in subset:groups[r['subject_key']].append(r)
                if len(groups)<5:raise ValueError('Too few validation subjects')
                # Equal-participant mean of within-participant mean-recording Brier.
                eq=lambda k:statistics.mean(statistics.mean(r[k] for r in rs) for rs in groups.values())
                metrics.append({'train_support':tr,'eval_support':ev,'model':kind,
                      'n_validation_people':len(groups),'n_validation_recordings':len(subset),
                      'n_validation_landmarks':sum(r['n_landmarks'] for r in subset),
                      'brier_subject_equal':eq('brier'),'log_loss_subject_equal':eq('log_loss'),
                      'brier_window_weighted':sum(r['brier']*r['n_landmarks'] for r in subset)/sum(r['n_landmarks'] for r in subset)})
    return per,metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    args=ap.parse_args()
    repo=args.repo.resolve()
    if not (repo/'.git').is_dir():ap.error('Expected existing Git repository')
    rows,e2,sums,sha=read_inputs(repo)
    decoder=import_gate_e(repo)
    inspected=diagnostics(rows,repo,sums,decoder)
    eligible=[r for r in rows if r['status']==ELIGIBLE]
    split=group_split(eligible)
    if len(split)!=e2['origin_independent_subject_groups_n']:
        raise ValueError('Split group count does not match E2')
    for r in eligible:make_landmarks(r)  # uses metadata and annotation counts only, no signal
    per,metrics=score_sc_development(rows,split)
    by_status=Counter(r['e2_status'] for r in inspected)
    off_stages={study:dict(Counter(r['stage_at_lights_off'] for r in inspected if r['study']==study and r['e2_eligible'])) for study in ['sleep-cassette','sleep-telemetry']}
    gap_reason=dict(Counter(r['issue'] for r in inspected if r['e2_status']=='SCORING_GAP_OR_OVERLAP'))
    sc_w=off_stages['sleep-cassette'].get('W',0)
    sc_all=sum(off_stages['sleep-cassette'].values())
    split_sizes=dict(Counter(split.values()))
    out={'version':VERSION,'source_gate_e2_sha256':sha,'input_eligible_recordings':len(eligible),
        'input_eligible_people':len(split),'e2_status_counts':dict(by_status),
        'stage_at_lights_off_among_eligible':off_stages,'gap_diagnostic_counts':gap_reason,
        'sc_wake_at_origin_recordings':sc_w,'sc_wake_at_origin_fraction':sc_w/sc_all if sc_all else None,
        'sc_group_split_sizes':split_sizes,'sc_development_metrics':metrics,
        'sc_test_results':'NOT_COMPUTED','st_external_results':'NOT_COMPUTED',
        'baseline_scope':'SC training/validation subjects only; first annotated Stage 2 target, horizon 300s, retrospective data',
        'limitations':['No EEG signals inspected','No no-N2 nights evaluated','No clinical prospective inference',
                       'ST condition differs and is reserved','Subject-group split is model-test protected, not fully outcome-blinded',
                       'At-off W does not prove W-only preceding context','Time-only bin boundaries are fixed exploratory design'],
        'training_gpu_hours':0,'muse_test_accessed':False,'primary_claim_confirmed':False,
        'decision':('STAGE_AUDIT_AND_SC_DEV_COMPLETED_NOT_GPU' if sc_w/sc_all>=.80 else 'REVIEW_WAKENESS_AT_ORIGIN_BEFORE_EEG')}
    manifest={'version':VERSION,'salt':SALT,
      'note':'Frozen for subsequent development; never use SC_PROTECTED_MODEL_TEST or ST_EXTERNAL_RESERVED for model selection',
      'by_subject':dict(sorted(split.items())),'input_gate_e2_inventory_sha256':sha['gate_e2_inventory']}
    artefacts={'gate_e3_summary.json':(json.dumps(out,indent=2,ensure_ascii=False)+'\n').encode(),
               'gate_e3_stage_diagnostics.csv':csv_blob(inspected),
               'gate_e3_sc_group_split.json':(json.dumps(manifest,indent=2)+'\n').encode(),
               'gate_e3_sc_validation_scores.csv':csv_blob(per)}
    for name,data in artefacts.items():immutable_write(args.outdir/name,data)
    print(json.dumps({'status':'PASS_DEVELOPMENT','eligible_recordings':len(eligible),'eligible_people':len(split),
       'SC_person_split':split_sizes,'stage_at_off':off_stages,'gap_issue_counts':gap_reason,
       'SC_time_only_validation_model_configs':len(metrics),'SC_test_scored':False,
       'ST_external_scored':False,'gpu_used':False},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
