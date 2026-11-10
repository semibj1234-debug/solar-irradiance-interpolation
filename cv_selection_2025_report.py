"""Post-selection external evaluation. Never imported by the CV fitting runner."""
import argparse,itertools,time,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
from cv_selection_2025 import PREVIOUS,ROOT,SOURCE,read,sha,save,load_checkpoint
from thinning_2025 import scores

def summarize_pairs(frame,columns):
    rows=[]
    for key,g in frame.groupby(columns,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        v=g.difference.dropna();count=len(v);sd=v.std(ddof=1) if count>1 else np.nan
        se=sd/np.sqrt(count);half=t.ppf(.975,count-1)*se if count>1 else np.nan
        rows.append(dict(zip(columns,key))|dict(count=count,mean=v.mean(),sd=sd,ci_low=v.mean()-half,ci_high=v.mean()+half))
    return pd.DataFrame(rows)

def figures(d,external,paired):
    dest=d/'figures';dest.mkdir(exist_ok=True)
    valid=external[external.final_status=='ok'];methods=sorted(external.selector.unique())
    ns=sorted(external.n_train.unique());colors=plt.get_cmap('tab10')
    specs=[('selected_M','Selected initial M','figure1_selected_M'),('fixed_MAE','External MAE','figure2_external_MAE'),('regret','Post-hoc oracle regret (MAE)','figure3_oracle_regret'),('fixed_NLPD','NLPD using observation variance','figure5a_NLPD'),('fixed_coverage','95% observation interval coverage','figure5b_coverage')]
    for metric,title,name in specs:
        frame=external[external.selection_status=='selected'] if metric=='selected_M' else valid
        agg=frame.groupby(['n_train','selector'])[metric].agg(['mean','std','median','count']).reset_index()
        agg.to_csv(dest/(name+'.csv'),index=False)
        fig,ax=plt.subplots(figsize=(10,5))
        for j,m in enumerate(methods):
            a=agg[agg.selector==m].sort_values('n_train');ax.plot(a.n_train,a['mean'],'.-',label=m,color=colors(j))
            a=a[a['count']>1];ax.fill_between(a.n_train,a['mean']-a['std'],a['mean']+a['std'],alpha=.08,color=colors(j))
        if metric=='fixed_MAE':
            reference=pd.read_csv(d/'fixed256_evaluation.csv');reference=reference[reference.status=='ok']
            a=reference.groupby('n_train').fixed_MAE.agg(['mean','std','count']).reset_index()
            a.to_csv(dest/'fixed256_MAE_reference.csv',index=False);ax.plot(a.n_train,a['mean'],'k--',label='fixed M=256 (valid only)')
        if metric=='fixed_coverage':ax.axhline(.95,color='gray',ls='--')
        ax.set(xlabel='Training locations n',ylabel=title,title=title+' | bands: MC SD; n=88 is one network')
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(dest/(name+'.'+ext),dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7),squeeze=False)
    for ax,n in zip(axes.flat,ns):
        groups=[external.loc[(external.n_train==n)&(external.selector==m),'selected_M'].dropna().to_numpy() for m in methods]
        ax.boxplot(groups,tick_labels=[m.replace('minimum','min').replace('one_se','1SE') for m in methods],showmeans=True)
        ax.set(title=f'n={n}'+(' (one fixed network)' if n==88 else ''),ylabel='Selected M',ylim=(0,270));ax.tick_params(axis='x',rotation=70,labelsize=7)
    for ax in axes.flat[len(ns):]:ax.axis('off')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/('figure1b_selected_M_distribution.'+ext),dpi=180)
    plt.close(fig)
    effects=[]
    for (key,method),g in external.groupby(['key','method']):
        if set(g.rule)=={'minimum','one_se'}:
            a=g.set_index('rule');effects.append(dict(key=key,method=method,n_train=int(a.n_train.iloc[0]),M_reduction=a.loc['minimum','selected_M']-a.loc['one_se','selected_M'],MAE_difference=a.loc['one_se','fixed_MAE']-a.loc['minimum','fixed_MAE']))
    e=pd.DataFrame(effects);e.to_csv(dest/'figure4_one_se_effect.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method,g in e.groupby('method'):
        a=g.groupby('n_train')[['M_reduction','MAE_difference']].mean()
        for ax,col in zip(axes,['M_reduction','MAE_difference']):ax.plot(a.index,a[col],'.-',label=method);ax.set(xlabel='n_train',ylabel=col);ax.axhline(0,c='gray',lw=.6);ax.legend()
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/('figure4_one_se_effect.'+ext),dpi=180)
    plt.close(fig)
    runtime=pd.read_csv(d/'runtime_summary.csv');runtime['failure_rate']=runtime.failures/runtime.fits
    runtime.to_csv(dest/'figure6_cv_failure_cost.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method,g in runtime.groupby('method'):
        g=g.sort_values('n_train')
        for ax,col in zip(axes,['failure_rate','seconds']):ax.plot(g.n_train,g[col],'.-',label=method);ax.set(xlabel='n_train',ylabel=col);ax.legend()
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/('figure6_cv_failure_cost.'+ext),dpi=180)
    plt.close(fig)
    # Actual random and spatial partitions for the three pilot densities.
    failures=pd.read_csv(d/'final_failure_summary.csv');failures['failure_rate']=failures.final_failed/failures.selections
    failures.to_csv(dest/'figure6b_final_failures.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    for label,g in failures.groupby('selector'):ax.plot(g.n_train,g.failure_rate,'.-',label=label)
    ax.set(xlabel='n_train',ylabel='Final selection/evaluation failure fraction');ax.legend(fontsize=7,ncol=2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/('figure6b_final_failures.'+ext),dpi=180)
    plt.close(fig)
    positions=pd.read_csv(d.parent/'spatial_maps/fold_positions.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,method in enumerate(['random','spatial']):
        for col,n in enumerate([77,44,11]):
            ax=axes[row,col];g=positions[(positions.key==f'b00_n{n}')&(positions.method==method)]
            for fold,h in g.groupby('fold'):ax.scatter(h.coord1,h.coord2,s=20,label=f'F{fold+1}')
            if method=='spatial':
                m=read(d.parent/'cv_fold_manifest'/f'b00_n{n}.json')
                for edge in m['methods'][method]['geometry']['boundaries'][1:-1]:ax.axvline(edge,c='gray',lw=.7)
            ax.set(xlim=(0,1),ylim=(0,1),title=f'{method}, n={n}',xlabel='normalized coord1',ylabel='normalized coord2');ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/('fold_examples.'+ext),dpi=180)
    plt.close(fig)

def revalidate_selected(out,key,b,n,M,r):
    """Recompute predictions and independent audit from frozen parameters; no refit."""
    from cv_selection_2025 import subset_arrays,sr,audit_fit
    from threadpoolctl import threadpool_limits
    folder=Path(r['reference']) if n==88 else PREVIOUS/'fits'/f'b{b:02d}'/f'n{n:02d}'/f'M{M:03d}'
    checkpoint=out/'revalidation'/f'{key}_M{M:03d}.json';checkpoint.parent.mkdir(exist_ok=True)
    fit_hash=sha(folder/'fit.json')
    if checkpoint.exists():
        previous=read(checkpoint);assert previous['fit_hash']==fit_hash;return previous
    start=time.perf_counter();result=dict(key=key,M=M,fit_hash=fit_hash,passed=False)
    try:
        frame=pd.read_parquet(SOURCE/'training.parquet')
        rows=list(range(88)) if n==88 else read(PREVIOUS/'subsets'/f'b{b:02d}_n{n:02d}.json')['row_indices']
        domain=read(PREVIOUS/'domain.json');x,z,tr=subset_arrays(frame.iloc[rows],domain['coordinate_min'],domain['coordinate_span'])
        saved=pd.read_csv(folder/('predictions_all.csv' if n==88 else 'predictions.csv'))
        if n==88:saved=saved[saved.prediction_defined].copy()
        target=tr.coordinates(saved[['coord1','coord2']].to_numpy(float));fit=read(folder/'fit.json');settings=read(folder/'settings.json')
        with threadpool_limits(1):
            pred=sr.predict_sr(fit,x,z,target,settings);result['audit']=audit_fit('SR',fit,x,z,target,pred,settings)
        original=tr.inverse_prediction(pred)
        for col in ['mean','latent_variance','observation_variance']:
            old=saved['prediction_mean' if col=='mean' and n==88 else col].to_numpy()
            np.testing.assert_allclose(original[col],old,rtol=1e-7,atol=1e-7)
        result['passed']=True
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
    result['seconds']=time.perf_counter()-start;save(checkpoint,result);return result

def evaluate(out,stage):
    start=time.perf_counter();d=out/stage
    # Selection is immutable input. All external information is loaded only here.
    selections=pd.read_csv(d/'selection_results.csv');selection_hash=sha(d/'selection_results.csv')
    frozen=read(PREVIOUS/'manifest.json');manifest=read(out/'manifest.json')
    assert sha(PREVIOUS/'manifest.json')==manifest['previous_manifest_hash']
    for name,h in frozen['data_hashes'].items():assert sha(SOURCE/name)==h
    baseline={int(r['M']):r for r in read(PREVIOUS/'baseline.json')['records']}
    fixed=pd.read_csv(PREVIOUS/'fixed_targets.csv');assert len(fixed)==21
    records=[];provenance=[];cache={}
    def retrieve(key,b,n,M):
        identity=(key,M)
        if identity in cache:return cache[identity]
        if n==88:
            r=baseline[M].copy();folder=Path(r['reference']);source=PREVIOUS/'baseline.json'
        else:
            folder=PREVIOUS/'fits'/f'b{b:02d}'/f'n{n:02d}'/f'M{M:03d}';source=folder/'result.json';r=read(source)
            sub=PREVIOUS/'subsets'/f'b{b:02d}_n{n:02d}.json'
            assert r['subset_hash']==sha(sub)
            assert read(sub)['row_indices']==read(out/'cv_fold_manifest'/f'{key}.json')['row_indices']
            assert r['fit_seed']==manifest['FIT_SEED']
        assert read(folder/'settings.json')==__import__('basis_sensitivity_2025').settings_for(int(np.sqrt(M)))
        if r['status']=='ok':
            assert r['audit_success']
            if n==88:
                pred=pd.read_csv(folder/'evaluation.csv')
                # Original baseline has different column labels; metrics were independently verified by thinning baseline audit.
            else:
                pred=pd.read_csv(folder/'predictions.csv');assert pred.pv_id.tolist()==fixed.pv_id.tolist()
                for prefix,mask in [('fixed',np.ones(21,dtype=bool)),('inside',pred.inside_thinned_hull.to_numpy(bool))]:
                    computed=scores(pred.nins.to_numpy(),pred['mean'].to_numpy(),pred.observation_variance.to_numpy(),mask)
                    for metric in ['MAE','RMSE','NLPD','coverage']:
                        expected=r.get(prefix+'_'+metric)
                        if expected is not None:np.testing.assert_allclose(computed[metric],expected,rtol=1e-12,atol=1e-12)
            provenance.append(dict(key=key,M=M,result_path=str(source),result_hash=sha(source),fit_path=str(folder/'fit.json'),fit_hash=sha(folder/'fit.json'),prediction_path=str(folder/('evaluation.csv' if n==88 else 'predictions.csv')),prediction_hash=sha(folder/('evaluation.csv' if n==88 else 'predictions.csv')),status=r['status']))
        else:provenance.append(dict(key=key,M=M,result_path=str(source),result_hash=sha(source),status=r['status']))
        cache[identity]=r;return r
    oracle=[];fixed_rows=[]
    for (key,b,n),g in selections.groupby(['key','replicate','n_train']):
        old=[retrieve(key,int(b),int(n),k*k) for k in range(1,17)]
        eligible=[r for r in old if r['status']=='ok' and r['audit_success']]
        best=min(eligible,key=lambda r:(r['fixed_MAE'],r['M'])) if eligible else None
        fixed256=old[-1];fixed_rows.append(dict(key=key,**fixed256))
        oracle.append(dict(key=key,replicate=b,n_train=n,oracle_M=best['M'] if best else None,oracle_MAE=best['fixed_MAE'] if best else None,valid_candidates=len(eligible)))
        for s in g.to_dict('records'):
            row=s|dict(selector=s['method']+'_'+s['rule'],final_status='selection_failed',reused=True,new_final_fit_seconds=0.,new_final_prediction_seconds=0.)
            if s['selection_status']=='selected':
                r=retrieve(key,int(b),int(n),int(s['selected_M']));row.update(r);row['final_status']=r['status']
                if r['status']=='ok':
                    validation=revalidate_selected(out,key,int(b),int(n),int(s['selected_M']),r)
                    row['reuse_revalidation_passed']=validation['passed'];row['reuse_revalidation_seconds']=validation['seconds']
                    if not validation['passed']:row['final_status']='reuse_revalidation_failed';row['reuse_error']=validation['error']
            if row['final_status']=='ok' and best:
                row['regret']=row['fixed_MAE']-best['fixed_MAE']
                row['difference_fixed256']=row['fixed_MAE']-fixed256['fixed_MAE'] if fixed256['status']=='ok' else None
            records.append(row)
    external=pd.DataFrame(records);external.to_csv(d/'external_evaluation.csv',index=False)
    pd.DataFrame(oracle).to_csv(d/'oracle_comparison.csv',index=False)
    pd.DataFrame(fixed_rows).to_csv(d/'fixed256_evaluation.csv',index=False)
    pd.DataFrame(provenance).to_csv(d/'reused_artifacts.csv',index=False)
    pairs=[]
    for (key,n),g in external.groupby(['key','n_train']):
        for a,b in itertools.combinations(g.to_dict('records'),2):
            pairs.append(dict(key=key,n_train=n,replicate=a['replicate'],method_a=a['selector'],method_b=b['selector'],pair_valid=a['final_status']=='ok' and b['final_status']=='ok',difference=a['fixed_MAE']-b['fixed_MAE'] if a['final_status']=='ok' and b['final_status']=='ok' else np.nan))
    pairs=pd.DataFrame(pairs);pairs.to_csv(d/'paired_comparison.csv',index=False)
    summarize_pairs(pairs,['n_train','method_a','method_b']).to_csv(d/'paired_summary.csv',index=False)
    successful=external[external.final_status=='ok']
    cols=['selected_M','fixed_MAE','fixed_RMSE','fixed_NLPD','fixed_coverage','inside_MAE','inside_RMSE','inside_NLPD','inside_coverage','regret','difference_fixed256']
    aggregate=successful.groupby(['n_train','selector'])[cols].agg(['mean','std','median','count']);aggregate.columns=['_'.join(c) for c in aggregate.columns];aggregate.reset_index().to_csv(d/'external_summary.csv',index=False)
    external.groupby(['n_train','selector']).agg(selections=('selection_status','size'),selection_failed=('selection_status',lambda x:int((x!='selected').sum())),final_failed=('final_status',lambda x:int((x!='ok').sum()))).reset_index().to_csv(d/'final_failure_summary.csv',index=False)
    external.groupby(['n_train','selector','selected_M'],dropna=False).size().rename('count').reset_index().to_csv(d/'selected_M_distribution.csv',index=False)
    external[['key','replicate','n_train','selector','final_status','condition_number','noise_variance','converged_starts','audit_success']].to_csv(d/'final_numerical_status.csv',index=False)
    fixed_pairs=external[['key','replicate','n_train','selector','difference_fixed256']].rename(columns={'difference_fixed256':'difference'})
    summarize_pairs(fixed_pairs,['n_train','selector']).to_csv(d/'paired_fixed256_summary.csv',index=False)
    figures(d,external,pairs)
    assert sha(d/'selection_results.csv')==selection_hash
    save(d/'evaluation_complete.json',dict(selection_hash=selection_hash,external_rows=len(external),unique_reused_fits=len(cache),seconds=time.perf_counter()-start,report_source_hash=sha(Path(__file__)),all_reuse_checks_passed=bool(external.reuse_revalidation_passed.dropna().all()),artifact_integrity_checks_passed=True))
    report(out,stage,external)
    return external

def report(out,stage,e):
    d=out/stage;c=pd.read_csv(d/'cv_candidate_summary.csv');f=pd.read_csv(d/'cv_fold_results.csv');r=pd.read_csv(d/'runtime_summary.csv')
    lines=['# 초기 DCT 해상도 선택 비교 — '+stage,'','## 실행 및 검증',f'- CV 적합 {len(f):,}개, 독립 검증까지 통과 {(f.status=="ok").sum():,}개, 실패 {(f.status!="ok").sum():,}개.',f'- 완전 평가 후보 {(c.status=="complete_cv").sum():,}/{len(c):,}. 일부 fold 실패 후보는 선택에서 제외하고 원인을 보존했다.',f'- 외부 평가 성공 {(e.final_status=="ok").sum()}/{len(e)}개 선택 결과. 중복 선택의 최종 모델은 기존 적합을 재사용했다.','- CV와 외부 평가를 분리했다. 외부 지표는 selection_results.csv 저장 이후 별도 모듈에서 읽었다.','- 모든 fold에서 기존 fit_sr의 8개 초기값을 실제 재적합했다. 조건수에 따라 기존 독립 double/70자리 검증을 수행했다.','- CV 계산 시간은 runtime_summary.csv의 작업별 시간 합이며 병렬 벽시계 시간과 구별한다. 기존 최종 적합 시간은 재사용 비용에 합산하지 않는다.','','## 공간 분할 및 선택 규칙','고정 전체 88개 좌표 영역을 유지한다. 동일 폭 5개 수직 블록은 빈 블록을 만들므로 모든 부분집합에 coord1 순위 기반 연속 수직 슬랩 5개를 적용했다. 원점 (0,0), 높이 1, 폭과 중점 경계는 각 fold manifest에 저장했다. 왼쪽 경계 포함, 오른쪽 경계 제외(최종 블록은 포함)다. 이는 좌표만 이용한 적용 선택이며 공간 의존성을 완전히 제거하지 않는다.','CV 평균은 관측치별 절대오차 평균이다. 5-fold SE는 fold MAE 표본표준편차/√5, LOO SE는 관측별 절대오차 표본표준편차/√n이다. 1-SE는 신뢰구간이 아닌 선택 휴리스틱이다. Fold 크기가 다를 때 CV 평균은 관측치 가중 평균이지만 SE는 동일 가중 fold 평균 변동성의 근사이며, 작은 n에서 특히 불안정하다.','','## 관찰 결과']
    for n,g in e.groupby('n_train',sort=False):
        good=g[g.final_status=='ok'];lines.append(f'### n={n}')
        if len(good):
            a=good.groupby('selector').agg(M_median=('selected_M','median'),MAE=('fixed_MAE','mean'),MAE_SD=('fixed_MAE','std'),NLPD=('fixed_NLPD','mean'),Coverage=('fixed_coverage','mean'),Regret=('regret','mean'),success=('fixed_MAE','size'))
            lines.append(a.to_markdown())
    lines+=['','## 해석 범위','표의 성능은 성공한 적합에 조건부다. 실패율은 final_failure_summary.csv 및 numerical_failures.csv로 함께 판단해야 한다. 방법 간 비교는 동일 부분집합의 paired_summary.csv를 사용하며 평균 차이와 MC 변동성을 함께 확인한다. n=88은 단일 관측망이며 표준편차나 신뢰구간을 생성하지 않는다.','단일 날짜, 동일 88개 원 관측망에서 추출한 반복, 고정 외부 21개 위치의 결과다. M과 격자 해상도가 동시에 변한다. 고정 외부 집합에는 thinning hull 바깥 위치가 있으며 내부 위치 지표는 inside_* 열에 별도로 저장했다. Spatial 결과는 수직 슬랩 구성에 의존한다. Oracle은 외부 MAE를 이용한 사후 비교 기준이며 선택에 사용하지 않았다.','','## 재현 경로','cv_selection_2025.py → fold manifest → run_candidate → subset_arrays → fit_only → sr.fit_sr → sr.predict_sr → audit_fit → candidate_summary → select. 이후 cv_selection_2025_report.py가 외부 평가를 수행한다.','각 fold의 압축 JSON에 학습/검증 행 번호, 모수, 8개 시작값, 전체 trace, 원단위 예측 및 오차, 독립 검증, 실패 원인을 저장했다. reused_artifacts.csv는 최종 예측과 기저 분산을 포함한 기존 적합 경로와 해시를 추적한다.']
    lines+=['','## 연구 질문별 확인','1. 관측 수와 선택 M: 아래 선택 M 중앙값을 비교한다. 작은 n일수록 반드시 단조 감소한다고 가정하지 않는다.',e.groupby(['n_train','selector']).selected_M.median().unstack().to_markdown(),'2. Spatial 대 Random/LOO: paired_summary.csv의 동일 관측망 차이와 구간을 우선한다. 성공한 적합만의 평균 순위는 실패 비율과 분리해 해석한다.','3. 1-SE: 동일 CV 결과의 후처리이므로 선택 M은 minimum보다 커질 수 없다. 실제 감소 빈도와 폭은 figure4_one_se_effect.csv에 기록한다.','4. 작은 M과 성능: 같은 파일의 MAE_difference와 외부 NLPD/Coverage를 함께 비교한다. M 감소 자체를 성능 개선으로 해석하지 않는다.','5. Oracle: 다음은 성공한 최종 적합의 평균 regret이다.',e[e.final_status=='ok'].groupby(['n_train','selector']).regret.mean().unstack().to_markdown(),'6. 안정성과 비용: CV 내부 실패는 numerical_failures.csv, 최종 실패는 final_failure_summary.csv, 실제 CV 시간은 runtime_summary.csv를 구분한다. 1-SE 추가 학습 비용은 0이다.']
    lines+=['','## 질문에 대한 관찰 요약']
    good=e[e.final_status=='ok']
    for n,g in good.groupby('n_train'):
        a=g.groupby('selector').fixed_MAE.agg(['mean','std','count']);winner=a['mean'].idxmin();v=a.loc[winner]
        oracle_mean=g.groupby('selector').regret.mean();ow=oracle_mean.idxmin()
        lines.append(f'n={n}: 성공한 평가의 최소 평균 MAE는 {winner}, {v["mean"]:.3f} (MC SD {v["std"]:.3f}, 유효 반복 {int(v["count"])}). 평균 regret 최소는 {ow} ({oracle_mean[ow]:.3f}). 이 순위만으로 우수성을 확정하지 않으며 paired 차이의 변동성과 실패율을 함께 본다.')
    effect=pd.read_csv(d/'figures/figure4_one_se_effect.csv')
    for method,g in effect.groupby('method'):
        finite=g.M_reduction.dropna();delta=g.MAE_difference.dropna()
        lines.append(f'{method}: 1-SE로 M이 감소한 관측망은 {(finite>0).sum()}/{len(finite)}, 평균 M 감소 {finite.mean():.2f}. 양쪽 외부 평가가 유효한 경우 1-SE minus minimum MAE 차이 평균 {delta.mean():.3f}, SD {delta.std(ddof=1):.3f}. 이 전체 요약은 여러 n을 포함하므로 독립 표본 신뢰구간을 계산하지 않는다; n별 paired 표를 사용한다.')
    lines.append('계산 비용은 1-SE와 minimum이 같은 CV 적합을 공유한다. 다음 표는 CV 방법별 실제 작업 시간 합과 실패 fold 수다.')
    lines.append(r.groupby('method')[['fits','seconds','failures']].sum().to_markdown())
    (d/'REPORT_KO.md').write_text('\n\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--stage',required=True,choices=['pilot','full']);a=p.parse_args();evaluate(a.output,a.stage)
