#!/usr/bin/env python3
"""H-EVAL Gate E4: bounded, SHA-verified TRAIN-only Sleep-EDF signal QC pilot.

Source: official Sleep-EDF Expanded v1.0.0 files; no models, no held-out
or external data processing. Uses only the Python standard library.
"""
import argparse
import csv
import hashlib
import json
import math
import http.client
import os
import re
import shutil
import statistics
import struct
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

VERSION='H-EVAL-GateE4-train-signal-QC-v3-2026-10-22'
# (paper-only release) an unused constant holding a local workstation path was removed here.
BASE='https://physionet.org/files/sleep-edfx/1.0.0/'
SOURCE_PATH='data/raw/sleep_edfx_v1_0_0'
COUNT=6
EACH_CAP=70_000_000
TOTAL_CAP=420_000_000
WINDOW_SECONDS=30


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(p):return json.loads(p.read_text(encoding='utf-8'))


def manifest_hashes(path):
    out={}
    for raw in path.read_text().splitlines():
        x=re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(\S+)',raw.strip())
        if x:
            key=x[2].removeprefix('./')
            if key in out:raise ValueError('Duplicate SHA manifest path')
            out[key]=x[1].lower()
    return out


def entries(repo):
    base=repo/'results/decision/h_eval'
    e2=base/'gate_e2/sleep_edf_origin_inventory.csv'
    e3=base/'gate_e3/gate_e3_sc_group_split.json'
    e3sum=base/'gate_e3/gate_e3_summary.json'
    diag=base/'gate_e3/gate_e3_stage_diagnostics.csv'
    root=repo/SOURCE_PATH
    sums=manifest_hashes(root/'metadata/SHA256SUMS.txt')
    manifest_file=root/'metadata/RECORDS'
    if sha(manifest_file)!=sums.get('RECORDS'):
        raise ValueError('Official RECORDS checksum mismatch')
    from csv import DictReader
    with e2.open(newline='',encoding='utf8') as f:rows=list(DictReader(f))
    with diag.open(newline='',encoding='utf8') as f:diags={r['recording']:r for r in DictReader(f)}
    spl=read_json(e3)
    ev=read_json(e3sum)
    e2_summary=read_json(base/'gate_e2/sleep_edf_anchor_eligibility.json')
    if len(rows)!=197 or len({r['recording'] for r in rows})!=197 or ev['input_eligible_recordings']!=172:
        raise ValueError('Upstream eligibility mismatch')
    if Counter(r['status'] for r in rows)!=Counter(e2_summary['status_counts']) or sum(r['status']=='ELIGIBLE_ORIGIN_INDEPENDENT' for r in rows)!=172:
        raise ValueError('E2 inventory eligibility status changed')
    if ev.get('source_gate_e2_sha256',{}).get('gate_e2_inventory')!=sha(e2):
        raise ValueError('E3 summary provenance mismatch')
    if spl['input_gate_e2_inventory_sha256']!=sha(e2):
        raise ValueError('E3 split no longer matches E2 inventory')
    allowed={'SC_TRAIN','SC_VALIDATION','SC_PROTECTED_MODEL_TEST','ST_EXTERNAL_RESERVED'}
    if set(spl['by_subject'].values())!=allowed:
        raise ValueError('Unexpected E3 split')
    if Counter(spl['by_subject'].values())!=Counter({'SC_TRAIN':47,'SC_VALIDATION':15,'SC_PROTECTED_MODEL_TEST':15,'ST_EXTERNAL_RESERVED':17}):
        raise ValueError('E3 subject-group split count changed')
    eligible=[]
    for row in rows:
        who=row['subject_key']; split=spl['by_subject'].get(who)
        if row['study']!='sleep-cassette' or row['status']!='ELIGIBLE_ORIGIN_INDEPENDENT' or split!='SC_TRAIN':continue
        if diags[row['recording']]['stage_at_lights_off']!='W':continue
        duration=float(row['latency_s']); n=int(row['landmark_n'])
        if not math.isfinite(duration) or duration<360 or n<10:continue
        source=row['psg_path']
        if not re.fullmatch(r'sleep-cassette/SC\d{4}[A-Za-z0-9]{2}-PSG\.edf',source):raise ValueError('Unsafe official PSG path')
        if source not in sums:raise ValueError('Source PSG absent from official checksum manifest')
        row=dict(row)
        row.update(source_sha256=sums[source],expected_bytes=int(row['psg_size_bytes']))
        if row['expected_bytes']<=0 or row['expected_bytes']>EACH_CAP:continue
        eligible.append(row)
    # Spread evenly across latency distribution while enforcing distinct people.
    eligible.sort(key=lambda r:(float(r['latency_s']),r['recording']))
    if len({r['subject_key'] for r in eligible})<COUNT:raise ValueError('Insufficient distinct training subjects')
    chosen=[];used=set()
    for k in range(COUNT):
        target=round((len(eligible)-1)*(k+0.5)/COUNT)
        cand=sorted(range(len(eligible)),key=lambda i:(abs(i-target),i))
        row=next(eligible[i] for i in cand if eligible[i]['subject_key'] not in used)
        chosen.append(row);used.add(row['subject_key'])
    assert len(chosen)==COUNT and all(spl['by_subject'][r['subject_key']]=='SC_TRAIN' for r in chosen)
    if sum(r['expected_bytes'] for r in chosen)>TOTAL_CAP:raise ValueError('Pilot download budget exceeded')
    provenance={
      'gate_e2_inventory_sha256':sha(e2),
      'gate_e2_summary_sha256':sha(base/'gate_e2/sleep_edf_anchor_eligibility.json'),
      'gate_e3_split_sha256':sha(e3),
      'gate_e3_summary_sha256':sha(e3sum),
      'gate_e3_stage_diagnostics_sha256':sha(diag),
      'official_sha256s_sha256':sha(root/'metadata/SHA256SUMS.txt'),
      'official_records_sha256':sha(manifest_file)}
    return chosen, provenance


def resume_download(url, target, expected_sha, expected_bytes, chunk=256 << 10,
                    timeout=30, max_consecutive_stalls=8, max_reconnects=120):
    """Safely resume an official EDF in one invocation despite transient socket timeouts.

    Never truncates an existing partial. Each nonzero resume must be an exact 206
    byte range. A final official SHA256 is required before atomic publication.
    Repeated failures WITHOUT any new bytes eventually stop with the part intact.
    """
    if target.is_symlink():
        raise ValueError('Refusing symlink completed EDF: ' + str(target))
    if target.exists():
        if target.stat().st_size != expected_bytes or sha(target) != expected_sha:
            raise ValueError('EXISTING PSG FAILED SHA: refusal to overwrite ' + str(target))
        print(f'[E4] Reusing verified EDF: {target.name}', flush=True)
        return 'already_verified'
    if not 0 < expected_bytes <= EACH_CAP:
        raise ValueError('EDF download exceeds per-file safety budget')
    if chunk < 1 or chunk > 1 << 20:
        raise ValueError('Unsafe network buffer size')
    if max_consecutive_stalls < 1 or max_reconnects < 1:
        raise ValueError('Invalid bounded retry settings')
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.edf.e4-part')
    if tmp.is_symlink():
        raise ValueError('Refusing symlink partial file: ' + str(tmp))
    consecutive_stalls = 0
    reconnects = 0
    reported_mark = 0
    while True:
        offset = tmp.stat().st_size if tmp.exists() else 0
        if offset > expected_bytes:
            raise ValueError('Part file exceeds published EDF size')
        if offset == expected_bytes:
            break
        if reconnects >= max_reconnects:
            raise RuntimeError('Network reconnect safety budget exhausted; .e4-part preserved')
        reconnects += 1
        if offset:
            print(f'[E4] Resuming {target.name}: {offset:,}/{expected_bytes:,} bytes (connection {reconnects})', flush=True)
        else:
            print(f'[E4] Starting {target.name}: {expected_bytes:,} bytes (connection {reconnects})', flush=True)
        headers = {'User-Agent':'H-EVAL E4 resilient range download/2026-10-22',
                   'Accept-Encoding':'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-{expected_bytes-1}'
        req = urllib.request.Request(url, headers=headers)
        start_offset = offset
        issue = 'short HTTP response'
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if offset:
                    expected_cr = f'bytes {offset}-{expected_bytes-1}/{expected_bytes}'
                    if resp.status != 206 or resp.headers.get('Content-Range') != expected_cr:
                        raise ValueError('Range resume refused or unexpected; preserve .e4-part and stop')
                elif resp.status != 200:
                    raise ValueError('Unexpected HTTP status for full EDF download')
                if resp.headers.get('Content-Encoding', 'identity').lower() not in ('identity', ''):
                    raise ValueError('Unexpected compressed response; preserve partial')
                mode = 'ab' if tmp.exists() else 'xb'
                with tmp.open(mode) as f:
                    if f.tell() != offset:
                        raise ValueError('Partial file changed concurrently; abort without overwrite')
                    n = offset
                    while n < expected_bytes:
                        blob = resp.read(min(chunk, expected_bytes-n))
                        if not blob:
                            break
                        n += len(blob)
                        if n > expected_bytes:
                            raise ValueError('Response exceeds expected EDF byte count; preserve partial')
                        f.write(blob)
                        milestone = n // (4 << 20)
                        if milestone > reported_mark:
                            print(f'[E4] {target.name}: {n:,}/{expected_bytes:,} bytes', flush=True)
                            reported_mark = milestone
            current = tmp.stat().st_size
            if current == expected_bytes:
                break
            if current <= start_offset:
                issue = 'connection ended without new bytes'
            else:
                issue = 'connection ended before expected file length'
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, OSError) as exc:
            issue = f'{type(exc).__name__}: {str(exc)[:180]}'
            current = tmp.stat().st_size if tmp.exists() else 0
        if current > start_offset:
            consecutive_stalls = 0
        else:
            consecutive_stalls += 1
        if consecutive_stalls >= max_consecutive_stalls:
            raise RuntimeError(f'No download progress for {consecutive_stalls} consecutive '
                               f'attempts, .e4-part retained at {current:,}/{expected_bytes:,} '
                               f'bytes. Latest issue: {issue}')
        wait = min(12, 2 * max(1, consecutive_stalls))
        print(f'[E4] Network interruption ({issue}). Retained {current:,}/{expected_bytes:,} '
              f'bytes; retry in {wait}s; consecutive no-progress={consecutive_stalls}/{max_consecutive_stalls}',
              flush=True)
        time.sleep(wait)
    if not tmp.is_file() or tmp.stat().st_size != expected_bytes:
        raise ValueError('Incomplete EDF, retained partial')
    print(f'[E4] Downloaded full length. Checking official SHA256: {target.name}', flush=True)
    if sha(tmp) != expected_sha:
        raise ValueError('Downloaded EDF SHA FAILED, retained partial for inspection')
    # Atomic publish with hard link; refuses collision, never replaces a target.
    os.link(tmp, target)
    tmp.unlink()
    return 'downloaded_sha_verified'


def edf_header(f,physical_bytes):
    fixed=f.read(256)
    if len(fixed)!=256 or fixed[:8].strip()!=b'0':raise ValueError('Not EDF fixed header')
    def num(a,b,type_=int):return type_(fixed[a:b].decode('ascii').strip())
    n=num(252,256);hdr=num(184,192);nr=num(236,244);dr=num(244,252,float)
    if not 1<=n<=32 or hdr!=256*(n+1) or nr<=0 or dr<=0 or not math.isfinite(dr):raise ValueError('Unexpected EDF record structure')
    extra=f.read(hdr-256)
    if len(extra)!=hdr-256:raise ValueError('Truncated EDF signal header')
    pos=0;blocks={}
    for name,width in [('label',16),('transducer',80),('unit',8),('phys_min',8),('phys_max',8),('dig_min',8),('dig_max',8),('prefilter',80),('samples',8),('reserved',32)]:
        blocks[name]=[extra[pos+i*width:pos+(i+1)*width].decode('latin1').strip() for i in range(n)]
        pos+=n*width
    if pos!=len(extra):raise ValueError('Signal field lengths incorrect')
    info=[]
    for i in range(n):
        a={name:blocks[name][i] for name in blocks}
        for k in ('phys_min','phys_max'):a[k]=float(a[k])
        for k in ('dig_min','dig_max','samples'):a[k]=int(a[k])
        if not (a['phys_max']>a['phys_min'] and a['dig_max']>a['dig_min'] and a['samples']>0):raise ValueError('Invalid EDF signal calibration')
        a.update(rate_hz=a['samples']/dr,byte_offset=2*sum(int(x) for x in blocks['samples'][:i]))
        info.append(a)
    record_bytes=2*sum(int(x) for x in blocks['samples'])
    size=hdr+nr*record_bytes
    if size!=physical_bytes:raise ValueError('EDF physical size mismatch: '+str((size,physical_bytes)))
    selected=[]
    for lbl in ('EEG Fpz-Cz','EEG Pz-Oz'):
        match=[(i,s) for i,s in enumerate(info) if s['label'].lower()==lbl.lower()]
        if len(match)!=1:raise ValueError('Expected exactly one '+lbl+' EEG channel')
        i,s=match[0]
        if s['unit'].lower() not in ('uv','µv','μv'):raise ValueError('Unexpected EEG physical unit '+s['unit'])
        if abs(s['rate_hz']-100)>1e-7:raise ValueError('Unexpected EEG sampling rate '+str(s['rate_hz']))
        selected.append((i,s))
    return {'n_signals':n,'header_bytes':hdr,'n_records':nr,'data_record_seconds':dr,
            'record_bytes':record_bytes,'duration_s':dr*nr,'eeg_channels':selected,'info':info}


def read_window(f,header,channel,t1,t2):
    i,ch=channel;dt=header['data_record_seconds'];samp=ch['samples']
    total=header['duration_s']
    if t1<0 or t2<=t1 or t2>total+1e-8:raise ValueError('EEG window outside EDF')
    start=int(round(t1*ch['rate_hz']))
    end=int(round(t2*ch['rate_hz']))
    # Slice data by the EDF data-record grain to preserve per-signal interleaving.
    out=[];idx=start
    while idx<end:
        rec=idx//samp;ofs=idx%samp;take=min(end-idx,samp-ofs)
        if rec>=header['n_records']:raise ValueError('Requested samples exceed EDF data records')
        f.seek(header['header_bytes']+rec*header['record_bytes']+ch['byte_offset']+ofs*2)
        raw=f.read(take*2)
        if len(raw)!=take*2:raise ValueError('Truncated raw signal at record '+str(rec))
        out.extend(struct.unpack('<'+str(take)+'h',raw))
        idx+=take
    # EDF scaling: digital value maps linearly onto physical extrema.
    gain=(ch['phys_max']-ch['phys_min'])/(ch['dig_max']-ch['dig_min'])
    return [ch['phys_min']+(x-ch['dig_min'])*gain for x in out],out


def qc_window(f,h,channel,t1,t2):
    physical,digital=read_window(f,h,channel,t1,t2)
    n=len(physical)
    if n<100:raise ValueError('EEG window too short')
    m=statistics.fmean(physical)
    ac=[x-m for x in physical]
    rms=math.sqrt(statistics.fmean(x*x for x in ac))
    p2p=max(physical)-min(physical)
    repeats=sum(digital[i]==digital[i-1] for i in range(1,n))/(n-1)
    clipped=sum(x<=channel[1]['dig_min'] or x>=channel[1]['dig_max'] for x in digital)/n
    return {'samples':n,'physical_unit':channel[1]['unit'],
            'rms_ac_uV':round(rms,7),'peak_to_peak_uV':round(p2p,7),
            'adc_limit_fraction':round(clipped,7),
            'adjacent_identical_fraction':round(repeats,7),
            'non_finite_samples':sum(not math.isfinite(x) for x in physical)}


def choose_landmarks(row):
    off=float(row['lights_off_rel_s']);n2=float(row['first_n2_after_off_s'])
    first=math.ceil((off+WINDOW_SECONDS)/30-1e-9);last=math.floor(n2/30+1e-9)
    times=[k*30 for k in range(first,last+1) if k*30<n2-1e-8]
    if len(times)<10:raise ValueError('No valid ten landmarks')
    return [times[round((len(times)-1)*q)] for q in (0,.25,.5,.75,1)]


def run(repo,out,offline=False):
    if not (repo/'.git').is_dir():raise ValueError('Target must be existing Git repo')
    chosen,prov=entries(repo)
    planned=sum(r['expected_bytes'] for r in chosen)
    space=shutil.disk_usage(repo).free
    if space<max(650_000_000,int(planned*1.5)):
        raise ValueError('Insufficient free disk space for bounded pilot')
    root=repo/SOURCE_PATH
    results=[];details=[]
    for row in chosen:
        rid=row['recording'];path=root/row['psg_path']
        print(f'[E4] Checking TRAIN recording {rid} ({row["expected_bytes"]:,} bytes)',flush=True)
        if offline:
            if not path.exists():raise FileNotFoundError('Offline pilot PSG not present: '+str(path))
            if sha(path)!=row['source_sha256']:raise ValueError('Offline PSG hash mismatch')
            state='already_verified'
        else:
            state=resume_download(BASE+row['psg_path'],path,row['source_sha256'],row['expected_bytes'])
        print(f'[E4] Source checksum verified: {rid} ({state})',flush=True)
        if sha(path)!=row['source_sha256'] or path.stat().st_size!=row['expected_bytes']:
            raise ValueError('Post-download verification failed for '+rid)
        with path.open('rb') as f:
            h=edf_header(f,path.stat().st_size)
            if float(row['first_n2_after_off_s'])>h['duration_s']:
                raise ValueError('Annotation outside EEG raw signal '+rid)
            landmarks=choose_landmarks(row)
            for t in landmarks:
                for _,ch in h['eeg_channels']:
                    met=qc_window(f,h,(_,ch),t-WINDOW_SECONDS,t)
                    details.append({'recording':rid,'subject_key':row['subject_key'],
                        'channel':ch['label'],'t_start_s':t-WINDOW_SECONDS,'t_end_s':t,
                        'label_h300':int(float(row['first_n2_after_off_s'])<=t+300),**met})
        results.append({'recording':rid,'subject_key':row['subject_key'],
                        'psg_path':row['psg_path'],'source_sha256':row['source_sha256'],
                        'source_bytes':row['expected_bytes'],'full_psg_sha256_verified':True,
                        'lights_off_rel_s':float(row['lights_off_rel_s']),
                        'first_n2_after_off_s':float(row['first_n2_after_off_s']),
                        'latency_s':float(row['latency_s']),
                        'eeg_channel_labels':[s['label'] for _,s in h['eeg_channels']],
                        'sample_rates_hz':[s['rate_hz'] for _,s in h['eeg_channels']],
                        'windows_examined':len(landmarks)})
    output={'version':VERSION,'status':'TRAIN_ONLY_SIGNAL_QC_REVIEW',
        'dataset':'PhysioNet Sleep-EDF Expanded v1.0.0','provenance':prov,
        'selection':'six unique TRAIN subjects, W at LightsOff, spread over observed TRAIN latency',
        'sampled_train_recordings':results,
        'total_download_bytes_budgeted':planned,
        'examined_eeg_windows':len(details),
        'signals_sha256_verified':all(r['source_sha256'] for r in results),
        'split_guardrail':'Only SC_TRAIN; do not access SC_VALIDATION or SC_PROTECTED_MODEL_TEST signal or ST_EXTERNAL_RESERVED',
        'no_model_training':True,'gpu_hours':0,
        'limitations':['Six selected training recordings are a QC spot-check, not an unbiased random sample',
                       '5 chosen landmarks per record depend on event onset FOR QC SAMPLING ONLY',
                       'No single recording guarantees general signal quality',
                       'EEG window suitability thresholds are not inferred from this QC',
                       'No protected test or external scoring performed',
                       'All nights have N2 in the audited cohort; no-event coverage missing']}
    import io
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    artifacts={'gate_e4_signal_qc.json':(json.dumps(output,ensure_ascii=False,indent=2)+'\n').encode(),
               'gate_e4_signal_windows.csv':buf.getvalue().encode()}
    out.mkdir(parents=True,exist_ok=True)
    # Preflight conflicts before any artifact commit.
    for fn,blob in artifacts.items():
        f=out/fn
        if f.exists() and f.read_bytes()!=blob:
            raise ValueError('Existing result differs; no overwrite: '+str(f))
    for fn,blob in artifacts.items():
        f=out/fn
        if not f.exists():f.write_bytes(blob)
    print(json.dumps({'result':'SUCCESS_TRAIN_ONLY_QC','n_training_subjects':len(results),
                      'n_training_recordings':len(results),'n_eeg_channel_windows':len(details),
                      'all_psg_sha256_verified':True,'no_gpu':True,
                      'result_dir':str(out)},ensure_ascii=False,indent=2))
    return output,details


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--repo',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--offline',action='store_true')
    z=a.parse_args()
    run(z.repo,z.out,z.offline)

if __name__=='__main__':main()
