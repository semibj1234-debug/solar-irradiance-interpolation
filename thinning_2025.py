"""Nested Monte Carlo thinning, unchanged SR-2025 equations.
python thinning_2025.py --stage pilot
python thinning_2025.py --stage full --output <pilot directory>
A completed result (including failure) is immutable; interrupted attempts are retained.
"""
import sys
sys.dont_write_bytecode=True
import os
for _name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[_name]='1'
import argparse,io,json,math,time,traceback,unittest,shutil,platform
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import sr
from data import TrainingTransform,child_seed,read,sha
from single_2025 import ROOT,SOURCE,SETTINGS,load_and_check,training_arrays
from basis_sensitivity_2025 import settings_for,initial_values,monitored_fit,FIT_SEED
from audit import audit_fit,audit_metrics

REFERENCE=ROOT/'results/basis_sensitivity_2025_20261013T024747Z'
NS=(77,66,55,44,33,22,11)
KS=tuple(range(1,17))
SAMPLING_SEED=20261013
FILES=['sr.py','data.py','audit.py','config.json','single_2025.py','single_2025_tests.py','tests.py',
       'basis_sensitivity_2025.py','basis_sensitivity_2025_tests.py','thinning_2025.py','thinning_2025_tests.py']


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


def save(path,value):
    """Atomic JSON checkpoint; nonfinite diagnostics are explicit strings, not repairs."""
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def subsets(b):
    """No coordinates or responses accepted. Independent stream per b; sorted fit order."""
    seed=child_seed(SAMPLING_SEED,2025,b)
    permutation=np.random.default_rng(seed).permutation(88)
    return seed,permutation,{n:np.sort(permutation[:n]) for n in NS}


def subset_arrays(frame,coordinate_min,coordinate_span):
    """Only selected rows accepted. x(n,2), z(n,), transform.
    Domain uses full-88 coordinates only; mean/sd use selected n responses, ddof=0.
    """
    y=frame.nins.to_numpy(float)
    mean=float(y.mean());sd=float(y.std(ddof=0))
    if not np.isfinite(y).all() or sd<=1e-12:raise ValueError('Invalid subset response scale')
    tr=TrainingTransform(list(coordinate_min),list(coordinate_span),mean,sd)
    return tr.coordinates(frame[['coord1','coord2']].to_numpy(float)),tr.response(y),tr


def scores(y,mean,variance,mask):
    """Original-unit observation metrics; empty support is NA, one site is descriptive."""
    mask=np.asarray(mask,bool);count=int(mask.sum())
    if not count:return dict(count=0,MAE=None,RMSE=None,NLPD=None,coverage=None,reason='no fixed targets inside thinned hull')
    e=np.asarray(y)[mask]-np.asarray(mean)[mask];v=np.asarray(variance)[mask]
    if not np.isfinite(e).all() or not np.isfinite(v).all() or (v<=0).any():raise FloatingPointError('Invalid predictive moments; no clipping')
    r=dict(count=count,MAE=float(abs(e).mean()),RMSE=float(np.sqrt(np.mean(e*e))),
           NLPD=float(np.mean(.5*(np.log(2*np.pi*v)+e*e/v))),
           coverage=float(np.mean(abs(e)<=norm.ppf(.975)*np.sqrt(v))),reason=None)
    audit_metrics(y,mean,variance,mask,r)
    return r


def fit_only(x,z,s,diagnostics):
    """Train-only boundary; logs failed Cholesky/nonfinite state calls without changing values."""
    original=sr.state
    diagnostics.update(cholesky_failures=0,nonfinite_state_inputs=0,state_errors=[])
    def observed(A,y,q,noise):
        if not np.isfinite(q).all() or not np.isfinite(noise):diagnostics['nonfinite_state_inputs']+=1
        try:return original(A,y,q,noise)
        except Exception as exc:
            if isinstance(exc,np.linalg.LinAlgError):diagnostics['cholesky_failures']+=1
            diagnostics['state_errors'].append(f'{type(exc).__name__}: {exc}')
            raise
    sr.state=observed
    try:return monitored_fit(x,z,s,diagnostics)
    finally:sr.state=original


def one_model(job):
    """One isolated process job. Never receives external response arrays."""
    out,b,n,k=job;out=Path(out);d=out/f'fits/b{b:02d}/n{n:02d}/M{k*k:03d}'
    d.mkdir(parents=True,exist_ok=True)
    if (d/'result.json').exists():return read(d/'result.json')
    # Retain partial attempts after an interruption, never confuse them with completed fits.
    if any(d.iterdir()):
        old=d/('interrupted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));old.mkdir()
        for f in list(d.iterdir()):
            if f.is_file():f.replace(old/f.name)
    row=dict(replicate=b,n_train=n,M=k*k,status='fit_failed',fit_success=False,prediction_success=False,
             audit_success=False,metric_success=False,fixed_count=21,fit_seed=FIT_SEED)
    diagnostic={};start=time.perf_counter();stage='fit'
    try:
        subset=read(out/f'subsets/b{b:02d}_n{n:02d}.json')
        full=pd.read_parquet(SOURCE/'training.parquet')
        selected=full.iloc[subset['row_indices']]
        domain=read(out/'domain.json')
        x,z,tr=subset_arrays(selected,domain['coordinate_min'],domain['coordinate_span'])
        target=pd.read_csv(out/'fixed_targets.csv',float_precision='round_trip')
        xx=tr.coordinates(target[['coord1','coord2']].to_numpy(float))
        inside=Delaunay(x).find_simplex(xx,tol=1e-12)>=0
        row.update(subset_hash=sha(out/f'subsets/b{b:02d}_n{n:02d}.json'),inside_count=int(inside.sum()),outside_count=int((~inside).sum()))
        manifest=tr.manifest();manifest['coordinate_fit']='fixed full-88 coordinates only';manifest['response_fit']='selected subset only'
        save(d/'transform.json',manifest)
        s=settings_for(k);save(d/'settings.json',s)
        q0,noise0,seeds=initial_values(x,s);np.savez(d/'initial_values.npz',weights=q0,noise=noise0,restart_seeds=seeds)
        with threadpool_limits(1):
            fit_begin=time.perf_counter()
            fit=fit_only(x,z,s,diagnostic)
            row['fit_seconds']=time.perf_counter()-fit_begin
            trace=pd.DataFrame(fit.pop('trace'));trace.to_csv(d/'trace.csv.gz',index=False,compression='gzip')
            save(d/'fit.json',fit);pd.DataFrame(fit['starts']).to_csv(d/'starts.csv',index=False)
            q=np.asarray(fit['weights']);np.save(d/'basis_variances.npy',q)
            t=trace[trace.restart==fit['selected_start']];pos=q[q>0]
            row.update(fit_success=True,converged_starts=sum(v['converged'] and v['error'] is None for v in fit['starts']),
                valid_starts=sum(v['error'] is None and np.isfinite(v['log_marginal_likelihood']) for v in fit['starts']),
                selected_start=fit['selected_start'],iterations=int(t.iteration.iloc[-1]),log_likelihood=fit['log_marginal_likelihood'],
                last_ll_change=float(t.ll.iloc[-1]-t.ll.iloc[-2]),condition_number=fit['condition_number'],
                noise_variance=fit['noise_variance'],noise_variance_original=fit['noise_variance']*tr.response_sd**2,
                zero_count=int((q==0).sum()),positive_count=len(pos),zero_ratio=float((q==0).mean()),
                min_positive=float(pos.min()) if len(pos) else None,max_q=float(q.max()),sum_q=float(q.sum()),
                maximum_ll_decrease=max(v['maximum_ll_decrease'] for v in fit['starts']))
            stage='prediction';row['status']='prediction_failed'
            pred=sr.predict_sr(fit,x,z,xx,s)
            np.savez(d/'prediction_standardized.npz',**pred)
            inv=tr.inverse_prediction(pred);p=target.copy()
            for key,value in inv.items():p[key]=value
            p['inside_thinned_hull']=inside
            p.to_csv(d/'predictions_unscored.csv',index=False)
            row['prediction_success']=True;stage='audit';row['status']='audit_failed'
            audit_begin=time.perf_counter()
            a=audit_fit('SR',fit,x,z,xx,pred,s)
            save(d/'audit.json',a)
            row.update(audit_success=True,audit_seconds=time.perf_counter()-audit_begin,reference_precision=a['reference_precision'])
        # Labels first enter here, after successful estimation, prediction and audit.
        stage='metrics';row['status']='metrics_failed'
        truth=pd.read_parquet(SOURCE/'heldout.parquet').set_index('pv_id').loc[target.pv_id,'nins'].to_numpy(float)
        p['nins']=truth;p['absolute_error']=abs(truth-p['mean']);p['squared_error']=(truth-p['mean'])**2
        p.to_csv(d/'predictions.csv',index=False)
        for prefix,mask in [('fixed',np.ones(21,bool)),('inside',inside)]:
            for key,value in scores(truth,p['mean'],p['observation_variance'],mask).items():row[prefix+'_'+key]=value
        row.update(status='ok',metric_success=True)
    except Exception as exc:
        row.update(error=f'{type(exc).__name__}: {exc}',failure_stage=stage)
        if hasattr(exc,'starts'):
            save(d/'failed_starts.json',exc.starts)
            pd.DataFrame(exc.starts).to_csv(d/'starts.csv',index=False)
            pd.DataFrame(exc.trace).to_csv(d/'trace.csv.gz',index=False,compression='gzip')
            row['converged_starts']=sum(v['converged'] and v['error'] is None for v in exc.starts)
            row['valid_starts']=sum(v['error'] is None and np.isfinite(v['log_marginal_likelihood']) for v in exc.starts)
        save(d/'failure.json',dict(stage=stage,error=row['error'],traceback=traceback.format_exc()))
    finally:
        row['total_seconds']=time.perf_counter()-start
        row.setdefault('fit_seconds',row['total_seconds'] if stage=='fit' else None)
        save(d/'numerical_diagnostics.json',diagnostic)
        for key in ['cholesky_failures','nonfinite_state_inputs','positive_to_zero','below_subnormal_to_zero','zero_b_to_zero','nonfinite_update']:
            row[key]=diagnostic.get(key)
        save(d/'result.json',row)
    return row


def tests(out):
    from single_2025_tests import suite
    from basis_sensitivity_2025_tests import VariableGridTests
    from thinning_2025_tests import ThinningTests
    alltests=unittest.TestSuite([suite(),unittest.defaultTestLoader.loadTestsFromTestCase(VariableGridTests),unittest.defaultTestLoader.loadTestsFromTestCase(ThinningTests)])
    stream=io.StringIO()
    with threadpool_limits(1):r=unittest.TextTestRunner(stream=stream,verbosity=2).run(alltests)
    (out/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    save(out/'tests.json',dict(passed=r.wasSuccessful(),count=r.testsRun))
    if not r.wasSuccessful():raise RuntimeError(stream.getvalue())


def prepare(out):
    train,target=load_and_check(out)
    x,z,tr=training_arrays(train)
    domain=dict(coordinate_min=tr.coordinate_min,coordinate_span=tr.coordinate_span)
    save(out/'domain.json',domain)
    xx=tr.coordinates(target[['coord1','coord2']].to_numpy(float))
    hull=Delaunay(x).find_simplex(xx,tol=1e-12)>=0;defined=((xx>=0)&(xx<=1)).all(axis=1)
    assert hull.sum()==21
    target.loc[hull].to_csv(out/'fixed_targets.csv',index=False)
    (out/'subsets').mkdir()
    for b in range(20):
        seed,permutation,sets=subsets(b)
        save(out/f'subsets/b{b:02d}_permutation.json',dict(replicate=b,seed=seed,permutation=permutation))
        for n,indices in sets.items():
            save(out/f'subsets/b{b:02d}_n{n:02d}.json',dict(replicate=b,n_train=n,row_indices=indices,
                 location_ids=train.iloc[indices].location_id.tolist(),pv_ids=train.iloc[indices].pv_id.tolist()))
    # Reuse all 16 baseline fits, re-evaluate with original 23-target batch for exact equality.
    baseline=[]
    assert read(REFERENCE/'manifest.json')['settings']==SETTINGS
    for key in ['sr.py','data.py','audit.py','config.json']:
        assert sha(ROOT/key)==read(REFERENCE/'manifest.json')['hashes'][key]
    with threadpool_limits(1):
        for k in KS:
            d=REFERENCE/f'M{k*k:03d}';fit=read(d/'fit.json');old=read(d/'result.json')
            pred=sr.predict_sr(fit,x,z,xx[defined],settings_for(k))
            saved=np.load(d/'prediction_standardized.npz')
            for key in pred:np.testing.assert_array_equal(pred[key],saved[key])
            a=audit_fit('SR',fit,x,z,xx[defined],pred,settings_for(k))
            p=pd.read_csv(d/'predictions_all.csv',float_precision='round_trip').loc[hull]
            metric=scores(p.nins,p.prediction_mean,p.observation_variance,np.ones(21,bool))
            np.testing.assert_array_equal([metric['MAE'],metric['RMSE']],[old['MAE'],old['RMSE']])
            row=dict(replicate=-1,n_train=88,M=k*k,status='ok',reused=True,fit_success=True,prediction_success=True,audit_success=True,metric_success=True,
                     converged_starts=8,valid_starts=8,condition_number=fit['condition_number'],zero_ratio=old['zero_ratio'],
                     zero_count=old['zero_count'],positive_count=old['positive_count'],noise_variance=fit['noise_variance'],
                     fit_seconds=old['fit_seconds'],inside_count=21,outside_count=0,reference=str(d),reference_precision=a['reference_precision'])
            for prefix in ['fixed','inside']:
                row.update({prefix+'_'+key:value for key,value in metric.items()})
            baseline.append(row)
    save(out/'baseline.json',dict(passed=True,records=baseline,exact_predictions=True,exact_metrics=True,
         note='16 existing fits reused once; no MC standard deviation for n=88'))


def summarize(out,B,label):
    directory=out/label;directory.mkdir(exist_ok=True)
    rows=read(out/'baseline.json')['records']
    paths=[]
    for b in range(B):
        for n in NS:
            for k in KS:
                p=out/f'fits/b{b:02d}/n{n:02d}/M{k*k:03d}/result.json'
                if not p.exists():raise RuntimeError(f'Missing result {p}')
                rows.append(read(p));paths.append(p)
    df=pd.DataFrame(rows);df.to_csv(directory/'individual_fits.csv',index=False)
    df[df.status!='ok'].to_csv(directory/'failures.csv',index=False)
    summaries=[]
    for (n,M),g in df.groupby(['n_train','M']):
        good=g[g.status=='ok'];r=dict(n_train=n,M=M,attempts=len(g),successes=len(good),
          success_rate=len(good)/len(g),fit_success_rate=float(g.fit_success.mean()),
          audit_success_rate=float(g.audit_success.mean()),prediction_success_rate=float(g.prediction_success.mean()),
          failure_rate=float((g.status!='ok').mean()),all_starts_converged_rate=float((g.converged_starts==8).mean()),
          fit_seconds_mean=float(g.fit_seconds.mean()),inside_count_min=int(g.inside_count.min()),inside_count_max=int(g.inside_count.max()),
          aggregation='fixed baseline' if n==88 else 'conditional on successful fits; failures reported separately')
        for prefix in ['fixed','inside']:
            for metric in ['MAE','RMSE','NLPD','coverage']:
                v=good[prefix+'_'+metric].dropna()
                r[prefix+'_'+metric+'_mean']=float(v.mean()) if len(v) else None
                r[prefix+'_'+metric+'_sd']=float(v.std(ddof=1)) if n!=88 and len(v)>1 else None
                r[prefix+'_'+metric+'_median']=float(v.median()) if len(v) else None
                r[prefix+'_'+metric+'_count']=len(v)
        for col in ['condition_number','zero_ratio']:
            v=pd.to_numeric(g[col],errors='coerce').dropna()
            for stat in ['median','max','min']:
                r[col+'_'+stat]=float(getattr(v,stat)()) if len(v) else None
            r[col+'_q25']=float(v.quantile(.25)) if len(v) else None
            r[col+'_q75']=float(v.quantile(.75)) if len(v) else None
        summaries.append(r)
    summary=pd.DataFrame(summaries);summary.to_csv(directory/'summary.csv',index=False)
    paired=[]
    for (b,n),g in df[df.n_train!=88].groupby(['replicate','n_train']):
        reference=g[g.M==256].iloc[0]
        for _,r in g.iterrows():
            valid=r.status=='ok' and reference.status=='ok'
            paired.append(dict(replicate=b,n_train=n,M=int(r.M),paired_valid=valid,
                 delta_MAE=float(r.fixed_MAE-reference.fixed_MAE) if valid else None,
                 reason=None if valid else f"M status={r.status}; M256 status={reference.status}"))
    paired=pd.DataFrame(paired);paired.to_csv(directory/'paired_differences.csv',index=False)
    pairsummary=paired.groupby(['n_train','M']).delta_MAE.agg(['count','mean','std','median','min','max']).reset_index()
    pairsummary['attempts']=B;pairsummary.to_csv(directory/'paired_summary.csv',index=False)
    save(directory/'completion.json',dict(B=B,thinning_fits=len(paths),baseline_fits=16,successes=int((df.status=='ok').sum()),
         failures=int((df.status!='ok').sum()),fit_seconds_sum=float(df[df.n_train!=88].fit_seconds.sum()),
         full_B20_estimated_worker_seconds=float(df[df.n_train!=88].fit_seconds.sum()*20/B)))
    return df,summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','full'],required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();B=3 if args.stage=='pilot' else 20
    out=args.output or ROOT/'results'/('thinning_2025_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    print('OUTPUT',out,flush=True)
    if not out.exists():
        out.mkdir(parents=True)
        code=out/'code';code.mkdir()
        for name in FILES:shutil.copy2(ROOT/name,code/name)
        save(out/'manifest.json',dict(sampling_seed=SAMPLING_SEED,fit_seed=FIT_SEED,settings=SETTINGS,ns=NS,ks=KS,
              hashes={n:sha(ROOT/n) for n in FILES},data_hashes={n:sha(SOURCE/n) for n in ['training.parquet','heldout.parquet']},
              policy='independent permutation per replicate; sorted prefixes; fixed full-88 domain; subset response scaling; same fit seed for all models',
              pilot_in_full=True,workers=args.workers,python=platform.python_version(),numpy=np.__version__))
        tests(out);prepare(out)
    else:
        manifest=read(out/'manifest.json')
        assert all(sha(ROOT/n)==h for n,h in manifest['hashes'].items()),'Code changed: resume requires an explicit version migration'
        assert all(sha(SOURCE/n)==h for n,h in manifest['data_hashes'].items())
        assert read(out/'tests.json')['passed'] and read(out/'baseline.json')['passed']
    if args.stage=='full':
        assert (out/'pilot/completion.json').exists(),'Complete and review pilot before full stage'
        assert read(out/'pilot_review.json')['approved_to_continue'],'Pilot review required'
    jobs=[(str(out),b,n,k) for b in range(B) for n in NS for k in KS
          if not (out/f'fits/b{b:02d}/n{n:02d}/M{k*k:03d}/result.json').exists()]
    print('NEW JOBS',len(jobs),'WORKERS',args.workers,flush=True)
    start=time.perf_counter();completed=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one_model,job):job for job in jobs}
        for future in as_completed(futures):
            result=future.result();completed+=1
            if completed%16==0 or result['status']!='ok':
                print('DONE',completed,'/',len(jobs),'b',result['replicate'],'n',result['n_train'],'M',result['M'],result['status'],
                      'cond',result.get('condition_number'),'elapsed',round(time.perf_counter()-start,1),flush=True)
    summarize(out,B,args.stage)
    save(out/(args.stage+'_timing.json'),dict(new_jobs=len(jobs),elapsed_seconds=time.perf_counter()-start))
    print('STAGE COMPLETE',args.stage,flush=True)

if __name__=='__main__':main()
