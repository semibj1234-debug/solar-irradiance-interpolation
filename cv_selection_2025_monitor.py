"""Read-only model monitoring; only writes progress records, never starts fits."""
import argparse,json,time,gzip,os
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import pandas as pd
import psutil
from cv_selection_2025 import read,save

def snapshot(out):
    cachepath=out/'monitor_checkpoint_index.json';cache=read(cachepath) if cachepath.exists() else {}
    manifests={f.stem:read(f) for f in (out/'cv_fold_manifest').glob('*.json')}
    expected={}
    for key,m in manifests.items():
        for method,info in m['methods'].items():
            for k in range(1,17):
                for fold in range(len(info['folds'])):expected[f'{key}/{method}/M{k*k:03d}/fold{fold:03d}.json.gz']=(key,method,k*k,fold)
    assert len(expected)==122528
    # Pilot records were already independently audited; retain their exact recorded status.
    if not cache:
        for r in pd.read_csv(out/'pilot/cv_fold_results.csv').to_dict('records'):
            name=f"{r['key']}/{r['method']}/M{int(r['M']):03d}/fold{int(r['fold']):03d}.json.gz"
            cache[name]=dict(status=r['status'],source='pilot_saved_output_audit')
    files=list((out/'cv_fold_results').glob('*/*/*/fold*.json.gz'));present={f.relative_to(out/'cv_fold_results').as_posix():f for f in files}
    extra=set(present)-set(expected)
    if extra:raise RuntimeError(f'Unexpected checkpoint identities: {list(extra)[:5]}')
    if set(cache)-set(present):raise RuntimeError('Previously saved checkpoints disappeared')
    for name,f in present.items():
        if name in cache:continue
        with gzip.open(f,'rt',encoding='utf-8') as stream:r=json.load(stream)
        key,method,M,fold=expected[name]
        assert (r['key'],r['method'],r['M'],r['fold'])==(key,method,M,fold)
        cache[name]=dict(status=r['status'],source='checkpoint',seconds=r['seconds'])
    save(cachepath,cache)
    counts=Counter(r['status'] for r in cache.values());now=time.time();workflow=read(out/'RUN_STATUS.json')
    pid=workflow['pid'];alive=False
    try:alive=psutil.Process(pid).is_running() and 'cv_selection_2025_finish.py' in ' '.join(psutil.Process(pid).cmdline())
    except psutil.Error:pass
    duplicate=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=' '.join(proc.info['cmdline'] or [])
            if proc.info['name'] and 'python' in proc.info['name'].lower() and 'cv_selection_2025_finish.py' in cmd:duplicate.append(proc.pid)
        except psutil.Error:pass
    previous=read(out/'PROGRESS.json') if (out/'PROGRESS.json').exists() else None
    elapsed=now-previous['unix_time'] if previous else 0
    rate=(len(cache)-previous['completed'])/elapsed if elapsed>0 else None
    recent=sorted(files,key=lambda f:f.stat().st_mtime)[-12:]
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),unix_time=now,target=122528,completed=len(cache),remaining=122528-len(cache),statuses=dict(counts),workflow_alive=alive,workflow_state=workflow['state'],workflow_stage=workflow['stage'],workflow_pids=duplicate,recent_fits_per_second=rate,estimated_remaining_hours=(122528-len(cache))/rate/3600 if rate and rate>0 else None,last_checkpoint_utc=datetime.fromtimestamp(max(f.stat().st_mtime for f in files),timezone.utc).isoformat(),recent_conditions=sorted(set(str(f.parent.relative_to(out/'cv_fold_results')) for f in recent)),log_updated_utc=datetime.fromtimestamp((out/'full_workflow.log').stat().st_mtime,timezone.utc).isoformat())
    save(out/'PROGRESS.json',result)
    with (out/'progress_history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        r=snapshot(a.output)
        if a.once or r['workflow_state'] in ['completed','failed'] or not r['workflow_alive']:break
        time.sleep(60)
