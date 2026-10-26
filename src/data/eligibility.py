#!/usr/bin/env python3
"""H-EVAL Gate E2 (2026-10-22): source-authenticated Sleep-EDF lights-off and origin audit.

Uses SHA-verified PhysioNet subject XLS and cached hypnograms, and HTTP-ranged
PSG headers only. No signal data, no model training, no reliance on sleep onset
or PSG recording start as a proxy for bedtime. Requires only Python stdlib.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import datetime, time as dtime
from pathlib import Path
from xls_stdlib import read_sheets

BASE='https://physionet.org/files/sleep-edfx/1.0.0/'
VERSION='H-EVAL-SleepEDF-GateE2-MetadataAnchors-v1-2026-10-22'
PSG_HEADER_LIMIT=64*1024
HEADER_SIGNALS_MAX=100
MILESTONES=(120,300,600)

def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for z in iter(lambda:f.read(1<<20),b''):h.update(z)
    return h.hexdigest()

def fetch(url,max_bytes=1048576,retries=3):
    e=None
    for n in range(retries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'H-EVAL GateE2 metadata 2026-10-22', 'Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=35) as res:
                if res.status!=200:raise ValueError('Unanticipated HTTP status')
                data=res.read(max_bytes+1)
                if len(data)>max_bytes:raise ValueError('Metadata download exceeded size cap')
                return data
        except Exception as ex:
            e=ex
            if n+1<retries:time.sleep(n+1)
    raise RuntimeError('Metadata download failed: '+str(e))

def range_read(url,start,end,retries=2):
    if start<0 or end<start or end-start+1>PSG_HEADER_LIMIT:
        raise ValueError('Unsafe HTTP Range size')
    e=None
    for k in range(retries):
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}',
                'Accept-Encoding':'identity','User-Agent':'H-EVAL GateE2 header-only 2026-10-22'})
            with urllib.request.urlopen(req,timeout=28) as res:
                if res.status!=206:raise ValueError('Refused HTTP Range; full PSG download forbidden')
                cr=res.headers.get('Content-Range','')
                m=re.fullmatch(r'bytes\s+(\d+)-(\d+)/(\d+)',cr)
                if not m or int(m[1])!=start or int(m[2])!=end:raise ValueError('Bad range response')
                b=res.read(end-start+2)
                if len(b)!=end-start+1:raise ValueError('Short/oversize ranged response')
                return b,int(m[3])
        except Exception as ex:
            e=ex
            if k+1<retries:time.sleep(0.4)
    raise RuntimeError('HTTP header probe failed: '+str(e))

def save_verified(root,name,sums):
    hash_expected=sums.get(name)
    if not hash_expected:raise ValueError('No official SHA for '+name)
    p=root/'metadata'/name
    if p.exists():
        if sha_file(p)!=hash_expected:raise ValueError('Official metadata local mismatch, refusing overwrite '+str(p))
    else:
        raw=fetch(BASE+name, 1024*1024)
        if sha(raw)!=hash_expected:raise ValueError('Official XLS SHA256 mismatch '+name)
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(raw)
    return p

def norm(s):return re.sub(r'[^a-z0-9]+','_',str(s).strip().lower()).strip('_')

def nint(x):
    if isinstance(x,bool):return None
    try:
        f=float(str(x).strip())
        if math.isfinite(f) and f==int(f) and f>=0:return int(f)
    except (TypeError,ValueError):pass
    return None

def parse_clock(x):
    # XLS may store date/time as text (00:38, 00:38h, 22:15:00) or Excel day fraction.
    if isinstance(x,dtime):return x.hour*3600+x.minute*60+x.second
    if isinstance(x,datetime):return x.hour*3600+x.minute*60+x.second
    if isinstance(x,(float,int)) and not isinstance(x,bool):
        if not math.isfinite(x) or x<0:return None
        frac=x%1
        if frac==0 and 0<=x<24 and isinstance(x,int):return None # ambiguity: integer as timestamp
        return round(frac*86400,3)
    s=str(x).strip()
    if not s or s.lower() in ('none','nan','missing','?','-'):return None
    m=re.fullmatch(r'\s*(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?\s*(?:h|hr|hrs)?\s*',s,re.I)
    if m:
        hh=int(m[1]);mm=int(m[2]);ss=float(m[3]) if m[3] else 0
        if hh>24 or (hh==24 and (mm or ss)) or mm>=60 or ss>=60:return None
        return (hh%24)*3600+mm*60+ss
    try:
        # Treat unformatted numeric floats from BIFF as Excel date/time fractions only.
        x=float(s)
        if not math.isfinite(x) or x<0:return None
        return round((x%1)*86400,3)
    except ValueError:
        return None

def identify_sc(sheets):
    possible=[]
    for sh,grid in sheets.items():
        for ir,row in enumerate(grid[:18]):
            cols={norm(x):idx for idx,x in enumerate(row)}
            sub=next((v for k,v in cols.items() if k in ('subject','subject_number','subject_nr')),None)
            night=next((v for k,v in cols.items() if k in ('night','night_nr','night_number')),None)
            off=next((v for k,v in cols.items() if k in ('lightsoff','lights_off','lights_out')),None)
            if all(v is not None for v in (sub,night,off)):
                possible.append((sh,grid,ir,sub,night,off))
    if len(possible)!=1:raise ValueError('Ambiguous/missing SC columns: subject, night, LightsOff: '+str([x[0] for x in possible]))
    sh,grid,header,subcol,nightcol,offcol=possible[0]
    rows={}; seen=0
    for row in grid[header+1:]:
        def col(i):return row[i] if i<len(row) else ''
        subject=nint(col(subcol)); night=nint(col(nightcol))
        if subject is None or night is None:continue
        if not 0<=subject<=99 or night not in (1,2):continue
        seen+=1;key=f'SC4{subject:02}{night}'
        if key in rows:raise ValueError('Duplicate subject-night key in SC metadata: '+key)
        rows[key]={'anchor_clock_seconds':parse_clock(col(offcol)),
                   'anchor_source':'SC-subjects.xls:LightsOff',
                   'raw_anchor_repr':str(col(offcol))[:70],'condition':'home_cassette'}
    if seen<150:raise ValueError('SC metadata parsed too few subject/night rows: '+str(seen))
    return rows,{'sheet':sh,'column_headers_at_row':header+1,'metadata_rows':seen}

def identify_st(sheets):
    candidates=[]
    for sh,grid in sheets.items():
        for r in range(min(16,len(grid)-1)):
            first=[norm(x) for x in grid[r]]; second=[norm(x) for x in grid[r+1]]
            group=[];last=''
            for v in first:
                if v:last=v
                group.append(last)
            cols=[]
            for i,v in enumerate(second):
                g=group[i] if i<len(group) else ''
                cols.append((g,v))
            def match(group_part,sub_parts):
                return [i for i,(g,v) in enumerate(cols) if group_part in g and all(p in v for p in sub_parts)]
            sub=match('subject',['nr'])
            p_night=match('placebo',['night','nr']) or match('placebo',['nr'])
            p_off=match('placebo',['lights','off'])
            t_night=match('temazepam',['night','nr']) or match('temazepam',['nr'])
            t_off=match('temazepam',['lights','off'])
            if all(len(x)==1 for x in (sub,p_night,p_off,t_night,t_off)):
                candidates.append((sh,grid,r,[x[0] for x in (sub,p_night,p_off,t_night,t_off)]))
    if len(candidates)!=1:raise ValueError('Missing/ambiguous ST two-level Placebo/Temazepam lights-off columns: '+str([x[0] for x in candidates]))
    sh,grid,h,cols=candidates[0]
    rows={};seen=0
    for row in grid[h+2:]:
        def v(i):return row[i] if i<len(row) else ''
        subject=nint(v(cols[0]))
        if subject is None or subject<0 or subject>99:continue
        seen+=1
        for night_i,off_i,condition in [(cols[1],cols[2],'placebo'),(cols[3],cols[4],'temazepam')]:
            night=nint(v(night_i))
            if night not in (1,2):continue
            key=f'ST7{subject:02}{night}'
            if key in rows:raise ValueError('Duplicate ST subject/night key: '+key)
            rows[key]={'anchor_clock_seconds':parse_clock(v(off_i)),
                       'anchor_source':'ST-subjects.xls:'+condition+'_lights_off',
                       'raw_anchor_repr':str(v(off_i))[:70], 'condition':condition}
    if seen<20:raise ValueError('ST metadata parsed too few subject rows: '+str(seen))
    return rows,{'sheet':sh,'column_headers_at_row':h+1,'metadata_rows':seen}

def read_metadata_book(path,reader,repo):
    try:
        parsed,about=reader(read_sheets(path))
        about['parser_backend']='stdlib_ole_biff'
        return parsed,about
    except Exception as first_error:
        # A dependency-free alternate path for uncommon BIFF variants; never install/upgrade software.
        executable=shutil.which('libreoffice') or shutil.which('soffice')
        if executable is None:
            raise ValueError('Legacy XLS parser rejected official workbook; no LibreOffice fallback installed. Original error: '+str(first_error))
        temp_parent=repo/'.local/h_eval/gate_e2'
        temp_parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent,prefix='.lo-metadata-') as work:
            work=Path(work);output=work/'output';output.mkdir()
            profile=(work/'lo-user').as_uri()
            cmd=[executable,'-env:UserInstallation='+profile,'--headless','--convert-to','csv','--outdir',str(output),str(path)]
            done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=35)
            csv_path=output/(path.stem+'.csv')
            if done.returncode!=0 or not csv_path.exists():
                raise ValueError('Both XLS parsing methods failed. Stdlib: '+str(first_error)+'; LibreOffice: '+done.stderr[:250])
            with csv_path.open(newline='',encoding='utf-8-sig') as f:
                grid=list(csv.reader(f))
            parsed,about=reader({'converted-first-sheet':grid})
            about['parser_backend']='headless_libreoffice_fallback'
            return parsed,about

def edf_header(raw,size):
    def field(a,b):return raw[a:b].decode('ascii','replace').strip()
    if len(raw)<256 or field(0,8)!='0':raise ValueError('Invalid PSG EDF first 256 bytes')
    n=int(field(252,256));hb=int(field(184,192));nr=int(field(236,244));duration=float(field(244,252))
    if n<1 or n>HEADER_SIGNALS_MAX or hb!=256*(1+n) or len(raw)!=hb or nr<=0 or not math.isfinite(duration) or duration<=0:
        raise ValueError('Invalid PSG EDF header layout')
    labels=[raw[256+i*16:256+(i+1)*16].decode('latin1').strip() for i in range(n)]
    # Standard EDF signal header stores samples per record after 216*n bytes.
    start=256+216*n
    ns=[int(raw[start+8*i:start+8*(i+1)].decode('ascii','replace').strip()) for i in range(n)]
    if any(x<=0 for x in ns):raise ValueError('Bad EDF sample count')
    expected=hb+nr*sum(ns)*2
    if expected!=size:raise ValueError('Remote EDF byte size disagrees with parsed signal layout')
    if not any('EEG' in v.upper() for v in labels):raise ValueError('PSG lacks EEG channel label')
    hm=re.fullmatch(r'(\d{2})\.(\d{2})\.(\d{2})',field(176,184))
    if not hm or int(hm[1])>23 or int(hm[2])>59 or int(hm[3])>59:
        raise ValueError('Invalid PSG EDF clock start time')
    start_clock=int(hm[1])*3600+int(hm[2])*60+int(hm[3])
    return {'psg_clock_start':field(176,184),'psg_clock_start_s':start_clock,
      'psg_date':field(168,176),'psg_duration_s':nr*duration,'psg_channels':labels,
      'psg_sample_rates_hz':[x/duration for x in ns],
      'header_sha256':sha(raw),'psg_size_bytes':size}

def probe_header(path):
    url=BASE+path
    first,total=range_read(url,0,255)
    n=int(first[252:256].decode('ascii').strip());hb=256*(1+n)
    if not 1<=n<=HEADER_SIGNALS_MAX or hb>PSG_HEADER_LIMIT:raise ValueError('PSG signal count out of bound')
    buf,total2=range_read(url,0,hb-1)
    if total!=total2:raise ValueError('Remote length changed between two header requests')
    return edf_header(buf,total)

def match_time(header,anchor):
    if anchor is None:return None
    # A clock time identifies a unique PSG-relative instant only for recordings shorter than 24h.
    if header['psg_duration_s']>=86400:return None
    t=(anchor-header['psg_clock_start_s'])%86400
    if t>header['psg_duration_s']-30:return None
    return t

def stage_risk_profile(events,off,recording_duration_s,horizons=MILESTONES):
    stages=sorted([(t,t+d,label) for t,d,label in events if label.lower().startswith('sleep stage ')], key=lambda z:z[0])
    if not stages:return 'NO_STAGES',{},None
    if any(not math.isfinite(t) or not math.isfinite(b) or b<=t for t,b,_ in stages):return 'INVALID_STAGE_EPOCH',{},None
    if off<0 or off>=recording_duration_s:return 'OFF_OUT_OF_PSG',{},None
    if any(t<=off<b and label.lower()=='sleep stage 2' for t,b,label in stages):return 'N2_ACTIVE_AT_OFF',{},None
    n2=[t for t,b,label in stages if label.lower()=='sleep stage 2' and off<t<recording_duration_s]
    if not n2:return 'NO_STAGE2_AFTER_OFF',{},None
    first=min(n2)
    # Require preevent interval covered by scored W/N1/other known stages, not unknown/M.
    cur=off
    for t,b,label in stages:
        left=max(t,off);right=min(b,first)
        if right<=left:continue
        if label.lower() in ('sleep stage ?','sleep stage m','sleep stage unknown'):
            return 'UNKNOWN_STAGE_IN_RISK_SET',{},first
        if left>cur+0.05 or left<cur-0.05:
            return 'SCORING_GAP_OR_OVERLAP',{},first
        cur=max(cur,right)
    if cur<first-0.05:return 'SCORING_GAP_OR_OVERLAP',{},first
    if first-off<30:return 'NO_30S_HISTORY_BEFORE_EVENT',{},first
    # Strictly causal 30-s history end, 30-s stride; horizon labels only when time before N2.
    kmin=math.ceil((off+30)/30-1e-9);kmax=math.floor(first/30+1e-9)
    landmarks=[k*30 for k in range(kmin,kmax+1) if k*30<first-1e-8]
    if not landmarks:return 'NO_LANDMARKS',{},first
    summary={'landmark_n':len(landmarks),'anchored_lb1200_n':sum(first-t<=1200 for t in landmarks),
             'latency_s':first-off}
    for h in horizons:
        summary[f'positive_h{h}_n']=sum(first<=t+h+1e-8 for t in landmarks)
        summary[f'negative_h{h}_n']=len(landmarks)-summary[f'positive_h{h}_n']
    return 'ELIGIBLE_ORIGIN_INDEPENDENT',summary,first

def load_gate_e(repo):
    p=repo/'scripts/decision/h_eval/gate_e/gate_e.py'
    spec=importlib.util.spec_from_file_location('previous_gate_e_local',p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def analyze_item(row,meta,repo,old):
    rid=row['recording'];mp=meta.get(rid[:6]);out={k:row[k] for k in ('recording','subject_key','study','night','psg_path','hypnogram_path')}
    if not mp:
        return out|{'status':'NO_MATCHED_SUBJECT_NIGHT','detail':'Official subject spreadsheet entry absent'}
    out.update(metadata_source=mp['anchor_source'],condition=mp['condition'],
               lights_off_clock_s=mp['anchor_clock_seconds'])
    if mp['anchor_clock_seconds'] is None:return out|{'status':'LIGHTS_OFF_MISSING_OR_INVALID'}
    try:
        hdr=probe_header(row['psg_path'])
        off=match_time(hdr,mp['anchor_clock_seconds'])
        out.update(psg_start_clock=hdr['psg_clock_start'],psg_duration_s=hdr['psg_duration_s'],
                   psg_header_sha256=hdr['header_sha256'],psg_size_bytes=hdr['psg_size_bytes'],
                   lights_off_rel_s=off,signal_has_EEG=True)
        if off is None:return out|{'status':'OFF_CLOCK_OUTSIDE_PSG'}
        p=repo/'data/raw/sleep_edfx_v1_0_0'/row['hypnogram_path']
        events,_=old.decode_edfplus_annotations(p.read_bytes())
        status,extra,first=stage_risk_profile(events,off,hdr['psg_duration_s'])
        out.update(status=status,first_n2_after_off_s=first,**extra)
    except Exception as ex:
        out.update(status='ERROR',detail=(type(ex).__name__+': '+str(ex))[:300])
    return out

def write_immutable(path,content):
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('Existing file differs (no overwrite): '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(content)

def execute(repo,outdir,workers=6):
    root=repo/'data/raw/sleep_edfx_v1_0_0'
    old=load_gate_e(repo)
    el=repo/'results/decision/h_eval/gate_e/sleep_edf_eligibility.json'
    inv=repo/'results/decision/h_eval/gate_e/sleep_edf_recording_inventory.csv'
    gate_d=repo/'results/decision/h_eval/gate_d/gate_d_preflight.json'
    for p in (el,inv,gate_d):
        if not p.is_file():raise ValueError('Upstream provenance missing: '+str(p))
    v=json.loads(el.read_text())
    if v.get('hypnograms_parsed_sha_verified_n')!=197 or v.get('hypnograms_with_explicit_lights_off_n')!=0:
        raise ValueError('Prior Gate E inventory contract changed')
    with inv.open(newline='',encoding='utf-8') as f:
        source=list(csv.DictReader(f))
    if len(source)!=197 or len(set(x['recording'] for x in source))!=197:raise ValueError('Gate E inventory not 197 unique')
    if any(x.get('status')!='PARSED_SHA_VERIFIED' for x in source):raise ValueError('Gate E includes unverified hypnograms')
    manifest=root/'metadata/SHA256SUMS.txt';records=root/'metadata/RECORDS'
    sums=old.sums_parse(manifest.read_bytes())
    if sha_file(records)!=sums.get('RECORDS'):raise ValueError('Upstream official RECORDS SHA mismatch')
    if len(old.records_parse(records.read_bytes()))!=197:raise ValueError('Source RECORDS count changed')
    # Validate cached hypnograms again; do not assume the previous on-disk SHA state persisted.
    for row in source:
        hyp=root/row['hypnogram_path']
        expected=sums.get(row['hypnogram_path'])
        if expected is None or not hyp.is_file() or sha_file(hyp)!=expected:
            raise ValueError('Cached hypnogram changed after Gate E: '+row['recording'])
    sources={};xlsx_checks={}
    for tag,fn,reader in [('SC','SC-subjects.xls',identify_sc),('ST','ST-subjects.xls',identify_st)]:
        p=save_verified(root,fn,sums)
        rec,desc=read_metadata_book(p,reader,repo)
        sources.update(rec)
        xlsx_checks[tag]={'filename':fn,'sha256':sha_file(p),'rows_parsed':len(rec),**desc}
    if len(sources)<190:raise ValueError('Metadata insufficient: '+str(len(sources))+' rows')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(lambda r:analyze_item(r,sources,repo,old),source))
    results.sort(key=lambda r:r['recording'])
    counts=collections.Counter(r['status'] for r in results)
    succ=[r for r in results if r['status']=='ELIGIBLE_ORIGIN_INDEPENDENT']
    eligible_subjects=len({r['subject_key'] for r in succ})
    sample_coverage=collections.Counter(r['study'] for r in succ)
    agg={'version':VERSION, 'source_dataset':'PhysioNet Sleep-EDF Expanded 1.0.0',
         'recording_inventory_n':len(results),'metadata_file_sha256':xlsx_checks,
         'matched_metadata_n':sum(r['status']!='NO_MATCHED_SUBJECT_NIGHT' for r in results),
         'lights_off_fields_parsed_n':sum(r.get('lights_off_clock_s') is not None for r in results),
         'origin_independent_eligible_recordings_n':len(succ),
         'origin_independent_subject_groups_n':eligible_subjects,
         'eligible_by_study':dict(sample_coverage),'status_counts':dict(counts),
         'eligible_landmark_totals':{k:sum(r.get(k,0) for r in succ) for k in ('landmark_n','anchored_lb1200_n','positive_h300_n','negative_h300_n')},
         'provenance':{'gate_e_json_sha256':sha_file(el),'gate_e_inventory_sha256':sha_file(inv),
           'gate_d_preflight_sha256':sha_file(gate_d),'metadata_source_sha256_checked':True,
           'hypnograms_verified_in_gate_e':197,'psg_headers_metadata_only':True,
           'full_psg_signal_sha256_checked':False},
         'flags':{'no_gpu':True,'no_full_PSG_download':True,'training_authorized':False,
          'clinical_prospective_claim':False,'no_event_cohort_verified':False,
          'sc_vs_st_must_be_separate':True,'medication_condition_confounding':True},
         'decision':('CANDIDATE_FOR_PREREGISTRATION_NOT_GPU' if len(succ)>=100 and eligible_subjects>=50
                     else 'REVIEW_ANCHOR_AND_ELIGIBILITY_GAPS'),
         'note':'Metadata lights-off is independent of future stage-2 labels, but a retrospective cohort remains not a prospective clinical trial. SC/ST not exchangeable.'}
    import io
    keys=sorted({k for row in results for k in row})
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(results)
    artefacts={'sleep_edf_anchor_eligibility.json':(json.dumps(agg,indent=2,ensure_ascii=False)+'\n').encode('utf-8'),
               'sleep_edf_origin_inventory.csv':f.getvalue().encode('utf-8')}
    for name,b in artefacts.items():write_immutable(outdir/name,b)
    print(json.dumps({'version':VERSION,'status':'COMPLETED','recordings':len(results),
         'matched_metadata_n':agg['matched_metadata_n'],
         'lights_off_fields_parsed_n':agg['lights_off_fields_parsed_n'],
         'eligible_recordings':len(succ),'eligible_subject_groups':eligible_subjects,
         'eligible_by_study':dict(sample_coverage),'reason_counts':dict(counts),
         'decision':agg['decision'],'no_gpu':True},indent=2))
    return agg

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--outdir',required=True,type=Path)
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.workers<1 or a.workers>8:p.error('workers must be 1..8')
    if not (a.repo/'.git').is_dir():p.error('No Git repository at expected path')
    execute(a.repo,a.outdir,a.workers)
if __name__=='__main__':main()
