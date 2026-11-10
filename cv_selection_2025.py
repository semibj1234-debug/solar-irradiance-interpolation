"""SR-2025 CV resolution selection; real refit per fold, immutable checkpoints.
python cv_selection_2025.py --stage pilot --workers 12
python cv_selection_2025.py --stage full --output <pilot directory> --workers 12
"""
import sys,os
sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse,gzip,json,time,traceback,shutil,io,unittest
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import sr
from data import read,sha,child_seed
from thinning_2025 import save,clean,subset_arrays,fit_only,ROOT,SOURCE,NS,KS
from basis_sensitivity_2025 import settings_for,FIT_SEED
from audit import audit_fit

PREVIOUS=ROOT/'results/thinning_2025_20261013T053000Z'
METHODS=('random','spatial','loo')
CV_SEED=20261014
CORE=['sr.py','data.py','audit.py','config.json','thinning_2025.py','basis_sensitivity_2025.py','single_2025.py']


def partitions(x,method,seed):
    """Coordinates (n,2) only -> validation local-index lists and geometry metadata.
    Spatial: five contiguous vertical slabs, near-equal counts; no KMeans.
    Midpoint boundaries use x only. First four slabs right-open; last right-closed.
    Fixed equal-width slabs were rejected for empty folds in the frozen manifests.
    """
    n=len(x)
    if n<6:raise ValueError('Need at least six positions')
    if method=='loo':return [np.array([i]) for i in range(n)],dict(type='leave_one_out')
    if method=='random':
        permutation=np.random.default_rng(seed).permutation(n)
        return [np.sort(v) for v in np.array_split(permutation,5)],dict(type='random',seed=seed)
    if method!='spatial':raise ValueError(method)
    order=np.argsort(x[:,0],kind='stable');groups=np.array_split(order,5)
    cuts=[]
    for a,b in zip(groups[:-1],groups[1:]):
        left,right=x[a[-1],0],x[b[0],0]
        if not left<right:raise ValueError('Coordinate tie prevents a contiguous slab boundary')
        cuts.append(float((left+right)/2))
    assignment=np.searchsorted(cuts,x[:,0],side='right')
    folds=[np.flatnonzero(assignment==j) for j in range(5)]
    return folds,dict(type='coordinate_quantile_slabs',axis=0,boundaries=[0.,*cuts,1.],
        origin=[0.,0.],height=1.,widths=np.diff([0.,*cuts,1.]).tolist(),
        boundary='left closed/right open except final right closed',empty_policy='quantile slabs guarantee nonempty when coordinate1 distinct')


def candidate_summary(records,n,M):
    """Observation-weighted MAE; SE=sample SD of fold MAE/sqrt(folds).
    For LOO fold MAEs are individual absolute errors, exactly the requested heuristic.
    """
    valid=[r for r in records if r['status']=='ok']
    complete=len(valid)==len(records) and sum(r['n_validation'] for r in valid)==n
    row=dict(M=M,status='complete_cv' if complete else 'incomplete_cv',folds=len(records),
             failed_folds=len(records)-len(valid),valid_observations=sum(r['n_validation'] for r in valid),CV_MAE=None,SE=None)
    if complete:
        losses=np.concatenate([r['absolute_errors'] for r in records])
        foldmeans=np.array([np.mean(r['absolute_errors']) for r in records])
        row.update(CV_MAE=float(losses.mean()),SE=float(foldmeans.std(ddof=1)/np.sqrt(len(records))),fold_MAE=foldmeans.tolist())
    return row


def select(candidates):
    """No external responses. Incomplete candidates never enter either selection rule."""
    valid=[c for c in candidates if c['status']=='complete_cv']
    if not valid:return dict(status='selection_failed',minimum=None,one_se=None,threshold=None)
    best=min(valid,key=lambda c:(c['CV_MAE'],c['M']))
    threshold=best['CV_MAE']+best['SE']
    return dict(status='selected',minimum=best['M'],one_se=min(c['M'] for c in valid if c['CV_MAE']<=threshold),
                minimum_error=best['CV_MAE'],minimum_SE=best['SE'],threshold=threshold,eligible_count=len(valid))


def write_checkpoint(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temporary,'wt',encoding='utf-8') as f:json.dump(clean(value),f,ensure_ascii=False,allow_nan=False)
    temporary.replace(path)


def load_checkpoint(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def fold_fit(training,validation_x,settings):
    """Only fold-training responses enter this boundary; validation responses absent.
    Returns fit, predictions and independent audit using unchanged SR equations.
    """
    domain=read(PREVIOUS/'domain.json')
    x,z,tr=subset_arrays(training,domain['coordinate_min'],domain['coordinate_span'])
    xx=tr.coordinates(validation_x);diagnostics={}
    fit=fit_only(x,z,settings,diagnostics)
    pred=sr.predict_sr(fit,x,z,xx,settings)
    audit=audit_fit('SR',fit,x,z,xx,pred,settings)
    return fit,tr.inverse_prediction(pred),audit,tr,diagnostics


def run_candidate(job):
    """One candidate task with per-fold atomic checkpoints; no heldout file access."""
    out,key,method,k=job;out=Path(out);directory=out/'cv_fold_results'/key/method/f'M{k*k:03d}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'candidate.json').exists():return read(directory/'candidate.json')
    manifest=read(out/'cv_fold_manifest'/f'{key}.json')
    full=pd.read_parquet(SOURCE/'training.parquet')
    rows=np.asarray(manifest['row_indices']);folds=manifest['methods'][method]['folds'];records=[]
    settings=settings_for(k)
    with threadpool_limits(1):
        for i,validation_local in enumerate(folds):
            path=directory/f'fold{i:03d}.json.gz'
            if path.exists():records.append(load_checkpoint(path));continue
            validation_rows=rows[validation_local];training_rows=np.setdiff1d(rows,validation_rows)
            record=dict(key=key,replicate=manifest['replicate'],n_train=len(rows),M=k*k,method=method,fold=i,
                n_fit=len(training_rows),n_validation=len(validation_rows),training_rows=training_rows.tolist(),
                validation_rows=validation_rows.tolist(),status='fit_failed',fit_seed=FIT_SEED,
                manifest_hash=sha(out/'cv_fold_manifest'/f'{key}.json'),fit_success=False,audit_success=False)
            started=time.perf_counter();diagnostics={};fit=None;stage='fit'
            try:
                # Explicit boundary: heldout observations are not loaded, fold labels read after audit.
                domain=read(PREVIOUS/'domain.json')
                x,z,tr=subset_arrays(full.iloc[training_rows],domain['coordinate_min'],domain['coordinate_span'])
                xx=tr.coordinates(full.iloc[validation_rows][['coord1','coord2']].to_numpy(float))
                begin=time.perf_counter();fit=fit_only(x,z,settings,diagnostics)
                record.update(fit_success=True,fit_seconds=time.perf_counter()-begin,fit=fit,
                    transform=dict(tr.manifest(),coordinate_fit='frozen full-88 domain; fold-training response only'),
                    condition_number=fit['condition_number'],converged_starts=sum(s['converged'] and s['error'] is None for s in fit['starts']))
                stage='prediction';record['status']='prediction_failed'
                pred=sr.predict_sr(fit,x,z,xx,settings);inv=tr.inverse_prediction(pred)
                record['predictions']=inv
                stage='audit';record['status']='audit_failed'
                record['audit']=audit_fit('SR',fit,x,z,xx,pred,settings);record['audit_success']=True
                truth=full.iloc[validation_rows].nins.to_numpy(float)
                errors=abs(truth-inv['mean'])
                if not np.isfinite(errors).all():raise FloatingPointError('Nonfinite validation loss')
                record.update(status='ok',truth=truth,absolute_errors=errors)
            except Exception as exc:
                record.update(error=f'{type(exc).__name__}: {exc}',failure_stage=stage,traceback=traceback.format_exc())
                if hasattr(exc,'starts'):record.update(starts=exc.starts,trace=exc.trace)
            record.update(seconds=time.perf_counter()-started,diagnostics=diagnostics)
            write_checkpoint(path,record);records.append(clean(record))
    result=candidate_summary(records,len(rows),k*k)
    result.update(key=key,replicate=manifest['replicate'],n_train=len(rows),method=method,
                  seconds=sum(r['seconds'] for r in records),manifest_hash=sha(out/'cv_fold_manifest'/f'{key}.json'))
    save(directory/'candidate.json',result)
    return result


def key_for(b,n):return 'full88' if n==88 else f'b{b:02d}_n{n:02d}'


def prepare(out):
    out.mkdir(parents=True);(out/'cv_fold_manifest').mkdir();(out/'code').mkdir()
    frozen=read(PREVIOUS/'manifest.json')
    for name in CORE:
        assert sha(ROOT/name)==frozen['hashes'][name]
    train=pd.read_parquet(SOURCE/'training.parquet');domain=read(PREVIOUS/'domain.json')
    x=(train[['coord1','coord2']].to_numpy()-domain['coordinate_min'])/domain['coordinate_span']
    manifests=[]
    for b,n in [(-1,88)]+[(b,n) for b in range(20) for n in NS]:
        source=None if n==88 else PREVIOUS/f'subsets/b{b:02d}_n{n:02d}.json'
        rows=list(range(88)) if n==88 else read(source)['row_indices'];key=key_for(b,n)
        manifest=dict(key=key,replicate=b,n_train=n,row_indices=rows,thinning_manifest=str(source),
                      thinning_hash=None if source is None else sha(source),methods={})
        for method in METHODS:
            seed=child_seed(CV_SEED,b+1,n)
            folds,geometry=partitions(x[rows],method,seed)
            assert sorted(np.concatenate(folds).tolist())==list(range(n))
            assert all(0<len(f)<n-1 for f in folds)
            manifest['methods'][method]=dict(folds=[f.tolist() for f in folds],geometry=geometry)
        save(out/'cv_fold_manifest'/f'{key}.json',manifest);manifests.append(manifest)
    names=CORE+['cv_selection_2025.py','cv_selection_2025_tests.py']
    for name in names:shutil.copy2(ROOT/name,out/'code'/name)
    save(out/'manifest.json',dict(CV_SEED=CV_SEED,FIT_SEED=FIT_SEED,methods=METHODS,settings=settings_for(16),
         hashes={n:sha(ROOT/n) for n in names},training_hash=sha(SOURCE/'training.parquet'),
         previous_manifest_hash=sha(PREVIOUS/'manifest.json'),fold_manifest_hashes={m['key']:sha(out/'cv_fold_manifest'/f"{m['key']}.json") for m in manifests},
         expected_thinning_fits=120960,expected_full88_fits=1568,expected_pilot_fits=7776,
         spatial_design='coord1 quantile contiguous slabs; fixed equal-width 5 strips empirically have empty folds; no label-based tuning'))
    from cv_selection_2025_tests import suite
    stream=io.StringIO()
    with threadpool_limits(1):r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite())
    (out/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    save(out/'tests.json',dict(passed=r.wasSuccessful(),tests=r.testsRun))
    if not r.wasSuccessful():raise RuntimeError(stream.getvalue())
    maps(out,x)


def maps(out,x):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    directory=out/'spatial_maps';directory.mkdir(exist_ok=True)
    manifests=[read(f) for f in sorted((out/'cv_fold_manifest').glob('*.json'))]
    with PdfPages(directory/'all_spatial_folds.pdf') as pdf:
        for start in range(0,len(manifests),12):
            fig,axes=plt.subplots(3,4,figsize=(12,10))
            for ax,m in zip(axes.flat,manifests[start:start+12]):
                xx=x[m['row_indices']];method=m['methods']['spatial']
                for j,f in enumerate(method['folds']):ax.scatter(xx[f,0],xx[f,1],s=14,color=plt.get_cmap('tab10')(j),label=f'F{j+1}: {len(f)}')
                for edge in method['geometry']['boundaries'][1:-1]:ax.axvline(edge,c='gray',lw=.6)
                ax.set(xlim=(0,1),ylim=(0,1),title=m['key'],xlabel='normalized coord1',ylabel='normalized coord2')
                ax.legend(fontsize=5,loc='lower left')
            for ax in axes.flat[len(manifests[start:start+12]):]:ax.axis('off')
            fig.suptitle('Five contiguous spatial validation slabs; each color held out in turn')
            fig.tight_layout();pdf.savefig(fig)
            if start==0:fig.savefig(directory/'spatial_examples.png',dpi=180)
            plt.close(fig)
    points=[]
    for m in manifests:
        for method in METHODS:
            for j,f in enumerate(m['methods'][method]['folds']):
                for i in f:points.append(dict(key=m['key'],method=method,fold=j,row=m['row_indices'][i],coord1=x[m['row_indices'][i],0],coord2=x[m['row_indices'][i],1]))
    pd.DataFrame(points).to_csv(directory/'fold_positions.csv',index=False)


def summarize(out,keys,stage):
    directory=out/stage;directory.mkdir(exist_ok=True);candidates=[];selections=[];foldrows=[]
    for key in keys:
        m=read(out/'cv_fold_manifest'/f'{key}.json')
        for method in METHODS:
            group=[]
            for k in KS:
                d=out/'cv_fold_results'/key/method/f'M{k*k:03d}';r=read(d/'candidate.json');group.append(r);candidates.append(r)
                for i in range(len(m['methods'][method]['folds'])):
                    f=load_checkpoint(d/f'fold{i:03d}.json.gz')
                    foldrows.append({name:f.get(name) for name in ['key','replicate','n_train','M','method','fold','n_fit','n_validation','status','fit_success','audit_success','condition_number','converged_starts','seconds','fit_seconds','error']})
            start=time.perf_counter();selection=select(group);post=time.perf_counter()-start
            for rule,column in [('minimum','minimum'),('one_se','one_se')]:
                selections.append(dict(key=key,replicate=m['replicate'],n_train=m['n_train'],method=method,rule=rule,
                    selected_M=selection[column],selection_status=selection['status'],threshold=selection.get('threshold'),
                    minimum_error=selection.get('minimum_error'),minimum_SE=selection.get('minimum_SE'),eligible_count=selection.get('eligible_count',0),
                    selection_postprocessing_seconds=post))
    pd.DataFrame(candidates).to_csv(directory/'cv_candidate_summary.csv',index=False)
    pd.DataFrame(foldrows).to_csv(directory/'cv_fold_results.csv',index=False)
    pd.DataFrame(selections).to_csv(directory/'selection_results.csv',index=False)
    failures=pd.DataFrame(foldrows);failures[failures.status!='ok'].to_csv(directory/'numerical_failures.csv',index=False)
    runtime=pd.DataFrame(foldrows).groupby(['method','n_train']).agg(fits=('status','size'),seconds=('seconds','sum'),failures=('status',lambda x:int((x!='ok').sum()))).reset_index()
    runtime.to_csv(directory/'runtime_summary.csv',index=False)
    save(directory/'cv_complete.json',dict(fits=len(foldrows),candidates=len(candidates),selectors=len(selections),failed_folds=int((failures.status!='ok').sum())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','full'],required=True);p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();out=a.output or ROOT/'results'/('cv_selection_2025_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    print('OUTPUT',out,flush=True)
    if not out.exists():prepare(out)
    else:
        m=read(out/'manifest.json');assert all(sha(ROOT/n)==h for n,h in m['hashes'].items())
        assert sha(SOURCE/'training.parquet')==m['training_hash']
        assert all(sha(out/'cv_fold_manifest'/f'{key}.json')==h for key,h in m['fold_manifest_hashes'].items())
        assert read(out/'tests.json')['passed']
    if a.stage=='full':assert read(out/'pilot_review.json')['continue_full']
    keys=[key_for(b,n) for b in range(3) for n in [77,44,11]] if a.stage=='pilot' else ['full88']+[key_for(b,n) for b in range(20) for n in NS]
    jobs=[(str(out),key,method,k) for key in keys for method in METHODS for k in KS if not (out/'cv_fold_results'/key/method/f'M{k*k:03d}'/'candidate.json').exists()]
    start=time.perf_counter();print('NEW CANDIDATES',len(jobs),'WORKERS',a.workers,flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_candidate,job) for job in jobs]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result()
            if i%8==0:print('DONE',i,'/',len(jobs),r['key'],r['method'],r['M'],r['status'],'elapsed',round(time.perf_counter()-start),flush=True)
    summarize(out,keys,a.stage)
    save(out/(a.stage+'_timing.json'),dict(new_candidates=len(jobs),seconds=time.perf_counter()-start,workers=a.workers))
    print('CV STAGE COMPLETE',a.stage,flush=True)

if __name__=='__main__':main()
