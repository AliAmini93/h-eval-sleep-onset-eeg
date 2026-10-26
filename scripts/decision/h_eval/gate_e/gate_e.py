#!/usr/bin/env python3
"""Sleep-EDF Expanded v1.0.0 metadata-only eligibility audit for H-EVAL.

Downloads only official RECORDS, SHA256SUMS and 197 tiny EDF+ hypnograms.
Never downloads PSG signals, never makes sleep-start proxies from PSG onset.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE='https://physionet.org/files/sleep-edfx/1.0.0/'
VERSION='H-EVAL-GateE-sleep-edf-eligibility-v2-2026-10-22'
PSG=re.compile(r'^(sleep-cassette/SC|sleep-telemetry/ST)([0-9A-Za-z]{6})-PSG\.edf$')
HYPNO=re.compile(r'^(sleep-cassette/SC|sleep-telemetry/ST)([0-9A-Za-z]{6})-Hypnogram\.edf$')
NIGHTS={'sleep-cassette':153,'sleep-telemetry':44}


def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def fetch(url, tries=3):
    for i in range(tries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'H-EVAL GateE metadata-audit/1.0','Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=35) as r:
                if r.status!=200:raise ValueError('Unexpected HTTP status '+str(r.status))
                b=r.read(2*1024*1024+1)
                if len(b)>2*1024*1024:raise ValueError('Oversized metadata response')
                return b
        except Exception:
            if i+1==tries:raise
            time.sleep(1.2*(i+1))


def sums_parse(raw):
    result={}
    for line in raw.decode('ascii').splitlines():
        m=re.fullmatch(r'([a-fA-F0-9]{64})\s+\*?(.+)',line.strip())
        if not m:raise ValueError('Malformed official SHA256SUMS line')
        name=m[2].removeprefix('./')
        if name in result:raise ValueError('Duplicate official hash '+name)
        result[name]=m[1].lower()
    return result


def records_parse(raw):
    paths=[line.strip() for line in raw.decode('ascii').splitlines() if line.strip()]
    if len(paths)!=197 or len(set(paths))!=197:raise ValueError('Expected 197 unique official PSG recordings')
    if not all(PSG.fullmatch(path) for path in paths):raise ValueError('Unexpected PSG path in RECORDS')
    cnt=collections.Counter(p.split('/')[0] for p in paths)
    if cnt != NIGHTS:raise ValueError('Unexpected study counts '+str(cnt))
    return paths


def pairs_create(paths,sums):
    candidates=collections.defaultdict(list)
    for name in sums:
        if HYPNO.fullmatch(name):candidates[(name.split('/')[0],Path(name).name[:6])].append(name)
    out=[]
    for p in paths:
        study=p.split('/')[0];fname=Path(p).name;key=fname[:6]
        hyp=candidates.get((study,key),[])
        if len(hyp)!=1:raise ValueError('Missing/ambiguous hypnogram for '+p+': '+str(hyp))
        if p not in sums:raise ValueError('No official PSG SHA '+p)
        # SC4ssNEO, ST7ssNJ0: 2-night participant ID is first 5 letters.
        out.append({'recording':key,'subject_key':key[:5],
                    'study':study,'night':key[5],
                    'psg_path':p,'hypnogram_path':hyp[0]})
    if len({x['recording'] for x in out})!=197:raise ValueError('Duplicate recording identity')
    return sorted(out,key=lambda x:x['recording'])


def parse_edf_header(data):
    def field(a,b):return data[a:b].decode('ascii','replace').strip()
    if len(data)<256 or field(0,8)!='0':raise ValueError('Not an EDF+ file')
    n=int(field(252,256));header=int(field(184,192));nrec=int(field(236,244));dt=float(field(244,252))
    if not 1<=n<=32 or header!=256*(n+1) or nrec<1 or not math.isfinite(dt) or dt<0:
        raise ValueError('EDF+ header structure invalid')
    if len(data)<header:raise ValueError('Truncated EDF+ header')
    labels=[data[256+16*i:256+16*(i+1)].decode('ascii','replace').strip() for i in range(n)]
    pos=256+n*216
    samples=[int(data[pos+8*i:pos+8*(i+1)].decode('ascii','replace').strip()) for i in range(n)]
    if any(v<=0 for v in samples):raise ValueError('Invalid sample count')
    full=header+2*sum(samples)*nrec
    if full!=len(data):raise ValueError('EDF+ byte size mismatch: '+str((full,len(data))))
    anns=[i for i,l in enumerate(labels) if l.lower()=='edf annotations']
    if not anns:raise ValueError('EDF+ has no EDF Annotations channel')
    # Official Sleep-EDF hypnogram SC4001EC has nrec=1, dt=0, a single
    # EDF Annotations channel, and a fully valid on-disk TAL payload.
    # The zero-duration edge case is legal here; it does NOT give us a
    # PSG time origin, recording duration, or legitimate lights-off time.
    if dt==0 and not (nrec==1 and n==1 and len(anns)==1 and data[192:236].strip().startswith(b'EDF+')):
        raise ValueError('Zero EDF data-record duration invalid outside single-record EDF+ annotation-only file')
    return header,nrec,dt,samples,anns


def decode_edfplus_annotations(data):
    header,nrec,dt,samples,anns=parse_edf_header(data)
    record_size=2*sum(samples)
    offsets=[2*sum(samples[:i]) for i in range(len(samples))]
    result=[]
    for j in range(nrec):
        start=header+j*record_size
        for i in anns:
            blob=data[start+offsets[i]:start+offsets[i]+2*samples[i]]
            for item in blob.split(b'\0'):
                if not item.strip(b'\x00\x14\x15'):continue
                pieces=item.split(b'\x14')
                if not pieces:continue
                timepart=pieces[0].decode('ascii','replace')
                parts=timepart.split('\x15')
                try:
                    onset=float(parts[0]);dur=float(parts[1]) if len(parts)>1 and parts[1] else 0.
                except ValueError:raise ValueError('Bad TAL onset/duration')
                if not math.isfinite(onset) or not math.isfinite(dur) or dur<0:
                    raise ValueError('Invalid TAL time')
                for title in pieces[1:]:
                    label=title.decode('latin-1').strip()
                    if label:result.append((onset,dur,label))
    if not result:raise ValueError('No EDF+ annotations')
    return sorted(result), nrec*dt if dt>0 else None


def annotation_profile(data):
    a,duration=decode_edfplus_annotations(data)
    stages=[(t,d,label) for t,d,label in a if label.casefold().startswith('sleep stage ')]
    n2=[t for t,d,label in stages if label.casefold()=='sleep stage 2']
    off=[t for t,d,label in a if label.strip().casefold() in ('lights off','lights-out','lights out','light off')]
    on=[t for t,d,label in a if label.strip().casefold() in ('lights on','lights-on','light on')]
    labels=collections.Counter(label for _,_,label in a)
    # Recording origin is NOT a lights-off marker; no implicit approximation.
    explicit=(len(off)==1 and len(on)==1 and off[0]<on[0] and bool(n2))
    first_n2_after_off=min((t for t in n2 if t>=off[0] and t<on[0]),default=None) if explicit else None
    return {'annotation_count':len(a),'stage_count':len(stages),'n2_present':bool(n2),
            'first_n2_from_psg_origin_s':min(n2) if n2 else None,
            'min_stage_time_s':min((t for t,_,_ in stages),default=None),
            'max_stage_end_s':max((t+d for t,d,_ in stages),default=None),
            'has_explicit_lights_off':len(off)==1,'has_explicit_lights_on':len(on)==1,
            'first_n2_after_lights_off_s':first_n2_after_off,
            'strict_independent_origin_candidate':first_n2_after_off is not None,
            'annotation_labels':dict(labels),'edf_annotation_record_duration_total_s':duration}


def safe_store_bytes(path,raw,expected):
    if sha_bytes(raw)!=expected:raise ValueError('Official SHA256 mismatch '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if sha_file(path)!=expected:raise ValueError('Existing source differs (no overwrite): '+str(path))
        return
    with path.open('xb') as f:f.write(raw)


def audit_item(item,rawdir,sums):
    path=rawdir/item['hypnogram_path']
    base=dict(item)
    try:
        expected=sums[item['hypnogram_path']]
        if not path.is_file():safe_store_bytes(path,fetch(BASE+item['hypnogram_path']),expected)
        if sha_file(path)!=expected:raise ValueError('SHA check failed local '+str(path))
        info=annotation_profile(path.read_bytes())
        return base|{'status':'PARSED_SHA_VERIFIED','hypnogram_sha256':expected}|info
    except Exception as err:
        return base|{'status':'FAILED','reason':str(err)[:300]}


def summarize(rows):
    valid=[r for r in rows if r['status']=='PARSED_SHA_VERIFIED']
    subjects=collections.defaultdict(list)
    for r in rows:subjects[r['subject_key']].append(r['recording'])
    gaps=[r['recording'] for r in rows if r['status']!='PARSED_SHA_VERIFIED']
    labels=collections.Counter()
    for r in valid:labels.update(r['annotation_labels'])
    eligible=sum(r['strict_independent_origin_candidate'] for r in valid)
    groupcounts=collections.Counter(len(v) for v in subjects.values())
    return {'version':VERSION,'source':BASE,'expected_psg_recordings':197,
       'official_count_by_study':dict(collections.Counter(r['study'] for r in rows)),
       'hypnograms_parsed_sha_verified_n':len(valid),'file_parse_failures':gaps,
       'number_subject_groups_from_official_filename_schema':len(subjects),
       'number_recordings_per_subject_group_distribution':dict(sorted(groupcounts.items())),
       'annotated_n2_present_n':sum(r.get('n2_present',False) for r in valid),
       'hypnograms_with_explicit_lights_off_n':sum(r.get('has_explicit_lights_off',False) for r in valid),
       'hypnograms_with_explicit_lights_on_n':sum(r.get('has_explicit_lights_on',False) for r in valid),
       'strict_independent_origin_candidates_n':eligible,
       'stage_annotation_labels':dict(labels),
       'no_psg_signal_downloaded':True,'no_gpu_used':True,
       'claim_guardrail':'Filename patient IDs allow grouping. PSG-recording start must NOT be treated as lights-off without evidence.',
       'decision':'REVIEW_ANCHOR_FEASIBILITY_MANUALLY' if eligible==0 else 'REVIEW_EXPLICIT_MARKERS_AND_STUDY_CONFOUNDING',
       'notes':'Sleep-EDF SC/ ST have different acquisition contexts and R&K labels. A within-person repeated-night design is possible, but origin integrity and censoring must be verified before modeling.'}


def audit(rawdir,outputdir,workers=6,gate_d_sha=None):
    manifest=rawdir/'metadata';manifest.mkdir(parents=True,exist_ok=True)
    docs={}
    for name in ('RECORDS','SHA256SUMS.txt'):
        path=manifest/name
        if not path.is_file():
            b=fetch(BASE+name)
            path.write_bytes(b)
        docs[name]=path.read_bytes()
    sums=sums_parse(docs['SHA256SUMS.txt'])
    if sums.get('RECORDS')!=sha_bytes(docs['RECORDS']):raise ValueError('RECORDS SHA mismatch with official SHA256SUMS')
    files=records_parse(docs['RECORDS'])
    rows=pairs_create(files,sums)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outcomes=list(ex.map(lambda item:audit_item(item,rawdir,sums),rows))
    s=summarize(outcomes)
    s['input_gate_d_preflight_sha256']=gate_d_sha
    s['official_manifest_sha256']={'RECORDS':sha_bytes(docs['RECORDS']),'SHA256SUMS.txt':sha_bytes(docs['SHA256SUMS.txt'])}
    # Only immutable output after complete verified parse. Partial sources remain local for safe retry.
    if s['hypnograms_parsed_sha_verified_n']!=197:
        outputdir.mkdir(parents=True,exist_ok=True)
        failures=[{'recording':r['recording'],'hypnogram_path':r['hypnogram_path'],
                   'reason':r.get('reason','UNKNOWN')} for r in outcomes if r['status']!='PARSED_SHA_VERIFIED']
        diagnose={'version':VERSION,'passed_n':len(outcomes)-len(failures),
                  'failed_n':len(failures),'failures':failures}
        diag_blob=(json.dumps(diagnose,sort_keys=True)+'\n').encode('utf-8')
        diag_hash=hashlib.sha256(diag_blob).hexdigest()[:12]
        diagnostic=outputdir/('gate_e_v2_diagnostics_'+diag_hash+'.json')
        if diagnostic.is_file():
            if json.loads(diagnostic.read_text())!=diagnose:
                raise ValueError('Existing diagnostic differs (no overwrite): '+str(diagnostic))
        else:
            diagnostic.write_text(json.dumps(diagnose,indent=2)+'\n')
        print(json.dumps({'status':'INCOMPLETE','parsed_n':len(outcomes)-len(failures),
                         'failures':failures[:12], 'full_diagnostics':str(diagnostic)},indent=2),file=sys.stderr)
        raise ValueError('Incomplete hypnogram audit; see full per-file failure reasons in '+str(diagnostic))
    outputdir.mkdir(parents=True,exist_ok=True)
    csv_bytes=make_csv(outcomes)
    res={'sleep_edf_eligibility.json':(json.dumps(s,ensure_ascii=False,indent=2)+'\n').encode(),
         'sleep_edf_recording_inventory.csv':csv_bytes}
    for name,data in res.items():
        p=outputdir/name
        if p.exists():
            if p.read_bytes()!=data:raise ValueError('Existing result differs: refusal to overwrite '+str(p))
        else:
            with p.open('xb') as f:f.write(data)
    print(json.dumps({k:s[k] for k in ('hypnograms_parsed_sha_verified_n',
           'number_subject_groups_from_official_filename_schema','annotated_n2_present_n',
           'hypnograms_with_explicit_lights_off_n','strict_independent_origin_candidates_n',
           'decision')},indent=2))
    return s


def make_csv(rows):
    import io
    fields=['recording','subject_key','study','night','psg_path','hypnogram_path','status',
      'n2_present','first_n2_from_psg_origin_s','min_stage_time_s','max_stage_end_s',
      'has_explicit_lights_off','has_explicit_lights_on','strict_independent_origin_candidate',
      'stage_count','annotation_count','hypnogram_sha256']
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    return f.getvalue().encode()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--outdir',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be 1..8')
    repo=args.repo.resolve()
    if not (repo/'.git').exists():parser.error('Not a Git repo')
    b=repo/'results/decision/h_eval/gate_d/gate_d_preflight.json'
    if not b.is_file():parser.error('Gate D evidence missing')
    upstream=json.loads(b.read_text())
    if upstream.get('all_151_header_compatible') is not True:parser.error('Gate D did not pass headers')
    s=audit(repo/'data/raw/sleep_edfx_v1_0_0',args.outdir,args.workers,sha_file(b))
    if s['hypnograms_parsed_sha_verified_n']!=197:raise SystemExit(2)

if __name__=='__main__':main()
