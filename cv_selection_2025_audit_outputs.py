"""Independent saved-output validation; streaming restart and prediction tables."""
import argparse,csv,gzip,time
from pathlib import Path
import numpy as np,pandas as pd
from cv_selection_2025 import read,save,load_checkpoint,SOURCE

def audit(out,stage):
    begin=time.perf_counter();d=out/stage;candidates=pd.read_csv(d/'cv_candidate_summary.csv');train=pd.read_parquet(SOURCE/'training.parquet');diagnostics=[];count=0;restarts=0
    with gzip.open(d/'validation_predictions.csv.gz','wt',encoding='utf-8',newline='') as pf,gzip.open(d/'all_restarts.csv.gz','wt',encoding='utf-8',newline='') as rf:
        pw=csv.DictWriter(pf,fieldnames=['key','method','M','fold','row','truth','mean','latent_variance','observation_variance','absolute_error']);pw.writeheader()
        rw=csv.DictWriter(rf,fieldnames=['key','method','M','fold','restart','converged','iterations','log_marginal_likelihood','error','noise_variance','zero_weights','maximum_ll_decrease','runtime_seconds']);rw.writeheader()
        for candidate in candidates.to_dict('records'):
            key=candidate['key'];method=candidate['method'];M=int(candidate['M']);manifest=read(out/'cv_fold_manifest'/f'{key}.json');outer=np.array(manifest['row_indices']);folds=manifest['methods'][method]['folds'];all_losses=[];fold_means=[];failed=0
            for i,local in enumerate(folds):
                r=load_checkpoint(out/'cv_fold_results'/key/method/f'M{M:03d}'/f'fold{i:03d}.json.gz');count+=1
                expected_val=outer[local];expected_train=np.setdiff1d(outer,expected_val)
                np.testing.assert_array_equal(r['validation_rows'],expected_val);np.testing.assert_array_equal(r['training_rows'],expected_train)
                fit=r.get('fit',{});starts=fit.get('starts',r.get('starts',[]));assert len(starts)==8
                identity=dict(key=key,method=method,M=M,fold=i)
                for start in starts:rw.writerow(identity|start);restarts+=1
                row=identity|dict(status=r['status'],converged_starts=sum(s['converged'] and s['error'] is None for s in starts),valid_starts=sum(s['error'] is None for s in starts),condition_number=r.get('condition_number'),noise_variance=fit.get('noise_variance'),fit_seconds=sum(s['runtime_seconds'] for s in starts))
                diag=r['diagnostics']
                for name in ['cholesky_failures','nonfinite_state_inputs','positive_to_zero','below_subnormal_to_zero','zero_b_to_zero','nonfinite_update']:row[name]=diag.get(name)
                if fit:
                    q=np.asarray(fit['weights']);assert q.shape==(M,);assert np.isfinite(q).all() and (q>=0).all();assert fit['noise_variance']>0
                    row.update(selected_start=fit['selected_start'],log_likelihood=fit['log_marginal_likelihood'],zero_count=int((q==0).sum()),min_positive=float(q[q>0].min()) if (q>0).any() else None)
                    selected=[v for v in fit['trace'] if v['restart']==fit['selected_start']]
                    row['last_ll_change']=abs(selected[-1]['ll']-selected[-2]['ll']) if len(selected)>1 else None
                    transform=r['transform'];y=train.iloc[expected_train].nins.to_numpy()
                    np.testing.assert_allclose(transform['response_mean'],y.mean(),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(transform['response_sd'],y.std(ddof=0),rtol=0,atol=1e-12)
                diagnostics.append(row)
                if r['status']=='ok':
                    assert r['audit_success'];pred=r['predictions'];truth=train.iloc[expected_val].nins.to_numpy()
                    np.testing.assert_array_equal(truth,r['truth']);errors=np.abs(truth-np.asarray(pred['mean']));np.testing.assert_array_equal(errors,r['absolute_errors'])
                    assert (np.asarray(pred['observation_variance'])>0).all();all_losses.extend(errors);fold_means.append(float(errors.mean()))
                    for j,global_row in enumerate(expected_val):pw.writerow(identity|dict(row=int(global_row),truth=truth[j],mean=pred['mean'][j],latent_variance=pred['latent_variance'][j],observation_variance=pred['observation_variance'][j],absolute_error=errors[j]))
                else:failed+=1
            assert int(candidate['failed_folds'])==failed
            if failed:assert candidate['status']=='incomplete_cv' and pd.isna(candidate['CV_MAE'])
            else:
                assert len(all_losses)==len(outer);np.testing.assert_allclose(candidate['CV_MAE'],np.mean(all_losses),rtol=1e-14)
                np.testing.assert_allclose(candidate['SE'],np.std(fold_means,ddof=1)/np.sqrt(len(folds)),rtol=1e-14,atol=1e-14)
    selections=pd.read_csv(d/'selection_results.csv')
    for (key,method),group in candidates.groupby(['key','method']):
        eligible=group[group.status=='complete_cv'].sort_values(['CV_MAE','M']);s=selections[(selections.key==key)&(selections.method==method)].set_index('rule')
        if eligible.empty:assert (s.selection_status=='selection_failed').all()
        else:
            best=eligible.iloc[0];threshold=best.CV_MAE+best.SE;one=eligible[eligible.CV_MAE<=threshold].M.min()
            assert s.loc['minimum','selected_M']==best.M;assert s.loc['one_se','selected_M']==one
    pd.DataFrame(diagnostics).to_csv(d/'fold_numerical_diagnostics.csv',index=False)
    save(d/'saved_output_audit.json',dict(passed=True,folds=count,restarts=restarts,candidates=len(candidates),selections=len(selections),seconds=time.perf_counter()-begin))
    print('SAVED OUTPUT AUDIT PASSED',count,restarts,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--stage',required=True);a=p.parse_args();audit(a.output,a.stage)
