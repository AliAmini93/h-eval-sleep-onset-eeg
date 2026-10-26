#!/usr/bin/env python3
"""Gate E5: SC_TRAIN-only six-person leave-one-person-out EEG mechanism smoke test.

EXPLORATORY, not confirmatory; no network, no GPU, no validation/test/ ST.
EEG features depend solely on the 30 s pre-landmark signal. No packages.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 'H-EVAL-E5-six-TRAIN-LOPO-v1-2026-10-23'
BINS = (300.,600.,1200.,1800.,3600.,7200.)
LABEL_HORIZON=300
LOOKBACK=1200
FEATURES=('fpz_log_delta','fpz_log_theta','fpz_log_alpha','fpz_log_beta',
          'pz_log_delta','pz_log_theta','pz_log_alpha','pz_log_beta',
          'fpz_log_rms','pz_log_rms')
MODELS=('CONSTANT','TIME_BIN','TIME_LOGIT','EEG_LOGIT','TIME_EEG_LOGIT')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def import_py(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ValueError('Could not import '+str(path))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def get_inputs(repo):
    evidence=repo/'results/decision/h_eval'
    files={
        'e2_inventory':evidence/'gate_e2/sleep_edf_origin_inventory.csv',
        'e3_split':evidence/'gate_e3/gate_e3_sc_group_split.json',
        'e3_summary':evidence/'gate_e3/gate_e3_summary.json',
        'e4_qc':evidence/'gate_e4/gate_e4_signal_qc.json',
        'e4_windows':evidence/'gate_e4/gate_e4_signal_windows.csv',
        'e4_code':repo/'scripts/decision/h_eval/gate_e4/v3/gate_e4.py',
        'sums':repo/'data/raw/sleep_edfx_v1_0_0/metadata/SHA256SUMS.txt',
    }
    for p in files.values():
        if not p.is_file():raise ValueError('Missing verified upstream input: '+str(p))
    with files['e2_inventory'].open(newline='') as f: inventory={x['recording']:x for x in csv.DictReader(f)}
    with files['e4_windows'].open(newline='') as f: checks=list(csv.DictReader(f))
    split=json.loads(files['e3_split'].read_text())['by_subject']
    qc=json.loads(files['e4_qc'].read_text())
    if len(inventory)!=197 or len(checks)!=60 or len(qc['sampled_train_recordings'])!=6:
        raise ValueError('Gate E2/E4 counts no longer match')
    if len({x['subject_key'] for x in qc['sampled_train_recordings']})!=6:
        raise ValueError('E4 six unique training subjects required')
    if set(x['recording'] for x in checks)!={x['recording'] for x in qc['sampled_train_recordings']}:
        raise ValueError('E4 sample QC and selected recordings differ')
    if any(split.get(r['subject_key'])!='SC_TRAIN' or not r['psg_path'].startswith('sleep-cassette/')
           for r in qc['sampled_train_recordings']):
        raise ValueError('PROTECTED SPLIT VIOLATION: refuse to read any signal')
    if any(r['status']!='ELIGIBLE_ORIGIN_INDEPENDENT' for r in
           (inventory[x['recording']] for x in qc['sampled_train_recordings'])):
        raise ValueError('E4 subject has invalid E2 eligibility')
    if any(x['samples']!='3000' or x['non_finite_samples']!='0' for x in checks):
        raise ValueError('Original QC reported unusable sample count')
    sig=import_py(files['e4_code'],'heval_e4_readonly')
    sums=sig.manifest_hashes(files['sums'])
    for r in qc['sampled_train_recordings']:
        if sums.get(r['psg_path'])!=r['source_sha256']:
            raise ValueError('Official source SHA mismatch '+r['recording'])
        if abs(float(inventory[r['recording']]['lights_off_rel_s'])-r['lights_off_rel_s'])>0.001:
            raise ValueError('LightsOff source mismatch')
        if abs(float(inventory[r['recording']]['first_n2_after_off_s'])-r['first_n2_after_off_s'])>0.001:
            raise ValueError('First N2 source mismatch')
    return qc['sampled_train_recordings'],inventory,sums,sig,{k:digest(p) for k,p in files.items()}


def fft_real(x):
    n=len(x)
    if n==0 or n & (n-1):raise ValueError('FFT length must be a power of 2')
    out=[complex(v) for v in x]
    j=0
    for i in range(1,n):
        bit=n>>1
        while j & bit:
            j^=bit;bit>>=1
        j^=bit
        if i<j:out[i],out[j]=out[j],out[i]
    length=2
    while length<=n:
        wlen=complex(math.cos(-2*math.pi/length),math.sin(-2*math.pi/length))
        half=length>>1
        for i in range(0,n,length):
            w=1+0j
            for j in range(i,i+half):
                u=out[j];v=out[j+half]*w
                out[j]=u+v;out[j+half]=u-v;w*=wlen
        length*=2
    return out


def channel_features(samples):
    if len(samples)!=3000 or any(not math.isfinite(x) for x in samples):
        raise ValueError('Invalid fixed 30s, 100Hz EEG window')
    # Five fixed 4-second Hann frames spaced evenly across 30 seconds.
    # Unfiltered descriptive spectral features; nothing depends on label or time-to-event.
    bpower=[0.,0.,0.,0.]
    w=[.5-.5*math.cos(2*math.pi*i/399) for i in range(400)]
    for start in (0,600,1200,1800,2400):
        frame=samples[start:start+400]
        mean=statistics.fmean(frame)
        x=[(v-mean)*wi for v,wi in zip(frame,w)]+[0.]*112
        f=fft_real(x)
        for k in range(1,154):
            hz=k*100./512.
            idx=0 if 0.5<=hz<4 else 1 if 4<=hz<8 else 2 if 8<=hz<13 else 3 if 13<=hz<30 else -1
            if idx>=0:bpower[idx]+=f[k].real*f[k].real+f[k].imag*f[k].imag
    avg=statistics.fmean(samples)
    rms=math.sqrt(statistics.fmean((v-avg)**2 for v in samples))
    return [math.log(1.+p) for p in bpower]+[math.log(1.+rms)]


def make_landmarks(off,n2,r):
    if not math.isfinite(off) or not math.isfinite(n2) or not off+30<n2:
        raise ValueError('Unexpected LightsOff/N2 timing')
    first=math.ceil((off+30)/30-1e-9)
    last=math.floor(n2/30+1e-9)
    tt=[k*30. for k in range(first,last+1) if k*30<n2-1e-8]
    if len(tt)!=int(r['landmark_n']):
        raise ValueError('E2 Full-support landmark count disagreement: '+r['recording'])
    an=sum(n2-t<=LOOKBACK+1.e-8 for t in tt)
    pos=sum(n2-t<=LABEL_HORIZON+1.e-8 for t in tt)
    if an!=int(r['anchored_lb1200_n']) or pos!=int(r['positive_h300_n']):
        raise ValueError('E2 anchor or label reconstruction mismatch: '+r['recording'])
    return tt


def extract(six,inventory,repo,sig):
    out=[];diagnostics=[]
    for r in six:
        raw=repo/'data/raw/sleep_edfx_v1_0_0'/r['psg_path']
        if not raw.is_file() or raw.stat().st_size!=int(r['source_bytes']) or digest(raw)!=r['source_sha256']:
            raise ValueError('Missing or source-SHA-invalid full EDF; no download in E5: '+r['recording'])
        row=inventory[r['recording']]
        off=float(row['lights_off_rel_s']); n2=float(row['first_n2_after_off_s'])
        landmarks=make_landmarks(off,n2,row)
        bad=0;maxclip=0.0
        with raw.open('rb') as f:
            h=sig.edf_header(f,raw.stat().st_size)
            if len(h['eeg_channels'])!=2 or any(abs(ch['rate_hz']-100.)>1e-8 for _,ch in h['eeg_channels']):
                raise ValueError('EEG channels invalid')
            for t in landmarks:
                x=[]
                for labelch in h['eeg_channels']:
                    samples,digital=sig.read_window(f,h,labelch,t-30,t)
                    clip=sum(v in (labelch[1]['dig_min'],labelch[1]['dig_max']) for v in digital)/len(digital)
                    maxclip=max(maxclip,clip)
                    if clip>.05:bad+=1
                    x.extend(channel_features(samples))
                out.append({'recording':r['recording'],'subject_key':r['subject_key'],
                            'elapsed_s':t-off,'landmark_end_s':t,
                            'anchor':bool(n2-t<=LOOKBACK+1e-8),
                            'y':int(n2-t<=LABEL_HORIZON+1e-8),'features':x})
        diagnostics.append({'recording':r['recording'],'subject_key':r['subject_key'],
                            'full_windows':len(landmarks),'anchor_windows':sum(n2-t<=LOOKBACK+1e-8 for t in landmarks),
                            'nonfatal_high_clip_windows_per_channel':bad,
                            'largest_clip_fraction':round(maxclip,5)})
    return out,diagnostics


def bin_idx(t):return sum(t>=b for b in BINS)

def time_vector(t):return [1. if bin_idx(t)==i else 0. for i in range(7)]

def preprocess(train,held):
    # Train-fold feature-only z-score, fixed scale clipping. No subject/test leakage.
    means=[statistics.fmean(v['features'][j] for v in train) for j in range(10)]
    std=[math.sqrt(statistics.fmean((v['features'][j]-means[j])**2 for v in train)) for j in range(10)]
    def transform(rows):
        result=[]
        for v in rows:
            z=[max(-5.,min(5.,(v['features'][j]-means[j])/max(std[j],1.e-6))) for j in range(10)]
            result.append((v,z))
        return result
    return transform(train),transform(held)


def sigmoid(x):
    if x>=0:
        z=math.exp(-min(60.,x));return 1/(1+z)
    z=math.exp(max(-60.,x));return z/(1+z)

def logistic_fit(x,y,steps=350,penalty=.025,lr=.3):
    if not x or len(x)!=len(y) or any(len(row)!=len(x[0]) for row in x):
        raise ValueError('Logistic feature matrix invalid')
    d=len(x[0]);n=len(x)
    p=(sum(y)+1)/(n+2)
    w=[0.]*(d+1);w[0]=math.log(p/(1-p))
    for step in range(steps):
        grad=[0.]*(d+1)
        for row,yy in zip(x,y):
            pred=sigmoid(w[0]+sum(a*b for a,b in zip(w[1:],row)))
            err=pred-yy
            grad[0]+=err
            for j,v in enumerate(row):grad[j+1]+=err*v
        eta=lr/(1.+step/200.)
        w[0]-=eta*grad[0]/n
        for j in range(d):w[j+1]-=eta*(grad[j+1]/n+penalty*w[j+1])
    return w

def predict_logit(w,x):return max(1e-6,min(1-1e-6,sigmoid(w[0]+sum(a*b for a,b in zip(w[1:],x)))))

def calc_score(p,y):return ((p-y)**2,-(y*math.log(p)+(1-y)*math.log1p(-p)))

def score_cv(rows):
    persons=sorted({x['subject_key'] for x in rows})
    if len(persons)!=6:raise ValueError('Only 6 SC_TRAIN subjects supported by E5')
    per=[]
    for held in persons:
        for training in ('FULL','ANCHOR'):
            train=[r for r in rows if r['subject_key']!=held and (training=='FULL' or r['anchor'])]
            if {r['subject_key'] for r in train}!={p for p in persons if p!=held}:
                raise ValueError('Incomplete LOPO training subjects')
            prior=(sum(r['y'] for r in train)+1)/(len(train)+2)
            buckets=defaultdict(lambda:[0.,0.])
            for v in train:
                b=buckets[bin_idx(v['elapsed_s'])];b[0]+=v['y'];b[1]+=1
            tsmooth={k:(b[0]+5*prior)/(b[1]+5) for k,b in buckets.items()}
            hold=[r for r in rows if r['subject_key']==held]
            trn,hld=preprocess(train,hold)
            models={}
            for kind in ('TIME_LOGIT','EEG_LOGIT','TIME_EEG_LOGIT'):
                def v(r,z):
                    return time_vector(r['elapsed_s']) if kind=='TIME_LOGIT' else z if kind=='EEG_LOGIT' else time_vector(r['elapsed_s'])+z
                models[kind]=logistic_fit([v(r,z) for r,z in trn],[r['y'] for r,z in trn])
            for evaluation in ('FULL','ANCHOR'):
                ev=[(r,z) for r,z in hld if evaluation=='FULL' or r['anchor']]
                if not ev:raise ValueError('Empty held-out support '+held)
                for kind in MODELS:
                    scores=[]
                    for r,z in ev:
                        if kind=='CONSTANT':p=prior
                        elif kind=='TIME_BIN':p=tsmooth.get(bin_idx(r['elapsed_s']),prior)
                        else:
                            xx=time_vector(r['elapsed_s']) if kind=='TIME_LOGIT' else z if kind=='EEG_LOGIT' else time_vector(r['elapsed_s'])+z
                            p=predict_logit(models[kind],xx)
                        scores.append(calc_score(p,r['y']))
                    per.append({'held_subject':held,'train_support':training,'eval_support':evaluation,
                                'model':kind,'n_eval_windows':len(ev),
                                'n_train_windows':len(train),
                                'positive_eval':sum(r['y'] for r,z in ev),
                                'brier':statistics.fmean(z[0] for z in scores),
                                'logloss':statistics.fmean(z[1] for z in scores)})
    result=[]
    for training in ('FULL','ANCHOR'):
        for ev in ('FULL','ANCHOR'):
            for kind in MODELS:
                rr=[r for r in per if r['train_support']==training and r['eval_support']==ev and r['model']==kind]
                if len(rr)!=6:raise ValueError('LOPO report missing folds')
                result.append({'train_support':training,'eval_support':ev,'model':kind,
                               'n_subjects':6,'total_eval_windows':sum(r['n_eval_windows'] for r in rr),
                               'brier_subject_equal':statistics.fmean(r['brier'] for r in rr),
                               'logloss_subject_equal':statistics.fmean(r['logloss'] for r in rr),
                               'brier_window_weighted':sum(r['brier']*r['n_eval_windows'] for r in rr)/sum(r['n_eval_windows'] for r in rr)})
    return per,result

def to_csv(rows):
    keys=list(rows[0]);s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=keys);w.writeheader();w.writerows(rows);return s.getvalue().encode()

def immut(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Existing result differs, refusing overwrite '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(argv)
    repo=a.repo.resolve()
    if not (repo/'.git').is_dir():raise ValueError('Must run inside existing original repository')
    six,inventory,sums,sig,provenance=get_inputs(repo)
    rows,diag=extract(six,inventory,repo,sig)
    per,aggregate=score_cv(rows)
    summary={'version':VERSION,'status':'TRAIN_ONLY_EXPLORATORY_NOT_CONFIRMATORY',
        'n_people':6,'n_recordings':6,'n_landmarks':len(rows),
        'n_anchored_landmarks':sum(r['anchor'] for r in rows),
        'n_positive_h300_full':sum(r['y'] for r in rows),
        'n_train_only_lopo_fold_scores':len(per),'method':'six-fold leave-one-person-out, deterministic fixed logistic L2=0.025, 350 steps, full batch',
        'features':'two channels raw 100Hz, five 4s Hann windows per 30s landmark, delta/theta/alpha/beta log power + log AC RMS',
        'provenance':provenance,'qc_counts':diag,
        'metrics':aggregate,
        'primary_matched_support':'TIME_LOGIT vs TIME_EEG_LOGIT, training FULL, evaluation FULL, identical six held subjects and horizons',
        'limitations':['TRAIN-ONLY selected by N2 delay; no independent model validation',
          'LOPO on six purposively selected subjects is NOT a protected test or manuscript evidence',
          'Time/EEG model scale and regularization fixed for diagnostic purposes; not tuned',
          'Retrospective event-only cohort; neither censoring nor clinical performance established',
          'No protected SC test, no SC validation, no ST external, no Muse data examined',
          'No automated window exclusion based on observed amplitude; QC is descriptive']}
    exports={'gate_e5_summary.json':(json.dumps(summary,indent=2,ensure_ascii=False)+'\n').encode(),
             'gate_e5_fold_scores.csv':to_csv(per),
             'gate_e5_qc_diagnostics.csv':to_csv(diag)}
    a.out.mkdir(parents=True,exist_ok=True)
    for nm,blob in exports.items():
        f=a.out/nm
        if f.exists() and f.read_bytes()!=blob:raise ValueError('Preflight result conflict '+str(f))
    for nm,blob in exports.items():immut(a.out/nm,blob)
    print(json.dumps({'status':summary['status'],'n_people':6,'n_landmarks':len(rows),
      'n_anchored_landmarks':summary['n_anchored_landmarks'],
      'n_fold_scores':len(per),'model_configs':len(aggregate),
      'SC_validation_or_test_scored':False,'ST_or_Muse_scored':False,'GPU':False,
      'report':str(a.out/'gate_e5_summary.json')},indent=2),flush=True)
    return summary
if __name__=='__main__':main()
