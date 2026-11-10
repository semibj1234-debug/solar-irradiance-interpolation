"""Read saved thinning results, independently check metrics, write poster figures/report.
Usage: python thinning_2025_report.py <run directory> --stage full
No fitting or parameter selection occurs in this module.
"""
import sys
sys.dont_write_bytecode=True
import argparse,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thinning_2025 import read,save,sha,ROOT,SOURCE,NS,KS,subsets,subset_arrays


def verify(out,df,stage):
    """Recompute from saved original-unit predictions; check source/data/subset lineage."""
    manifest=read(out/'manifest.json')
    assert all(sha(ROOT/n)==h for n,h in manifest['hashes'].items())
    assert all(sha(SOURCE/n)==h for n,h in manifest['data_hashes'].items())
    full=pd.read_parquet(SOURCE/'training.parquet')
    truth=pd.read_parquet(SOURCE/'heldout.parquet').set_index('pv_id').nins
    target=pd.read_csv(out/'fixed_targets.csv',float_precision='round_trip')
    domain=read(out/'domain.json');checks=[];restart_frames=[]
    for r in df[df.n_train!=88].to_dict('records'):
        b,n,M=int(r['replicate']),int(r['n_train']),int(r['M'])
        d=out/f'fits/b{b:02d}/n{n:02d}/M{M:03d}'
        sm=read(out/f'subsets/b{b:02d}_n{n:02d}.json')
        seed,p,sets=subsets(b)
        np.testing.assert_array_equal(sm['row_indices'],sets[n])
        assert r['subset_hash']==sha(out/f'subsets/b{b:02d}_n{n:02d}.json')
        x,z,tr=subset_arrays(full.iloc[sets[n]],domain['coordinate_min'],domain['coordinate_span'])
        savedtr=read(d/'transform.json')
        assert savedtr['response_mean']==tr.response_mean and savedtr['response_sd']==tr.response_sd
        assert savedtr['coordinate_min']==domain['coordinate_min'] and savedtr['coordinate_span']==domain['coordinate_span']
        if (d/'starts.csv').exists():
            starts=pd.read_csv(d/'starts.csv',float_precision='round_trip')
            assert len(starts)==8
            starts.insert(0,'M',M);starts.insert(0,'n_train',n);starts.insert(0,'replicate',b)
            restart_frames.append(starts)
        if (d/'fit.json').exists():
            fit=read(d/'fit.json');q=np.load(d/'basis_variances.npy')
            np.testing.assert_array_equal(q,fit['weights']);assert len(q)==M
            assert np.isfinite(q).all() and (q>=0).all()
            assert int((q==0).sum())==r['zero_count']
        if r['status']=='ok':
            p=pd.read_csv(d/'predictions.csv',float_precision='round_trip')
            np.testing.assert_array_equal(p.pv_id,target.pv_id)
            np.testing.assert_array_equal(p.nins,truth.loc[p.pv_id].to_numpy())
            assert len(p)==21 and read(d/'audit.json')['passed']
            for prefix,mask in [('fixed',np.ones(21,bool)),('inside',p.inside_thinned_hull.to_numpy(bool))]:
                pp=p.loc[mask];N=len(pp)
                assert N==r[prefix+'_count']
                if not N:continue
                errors=[float(a)-float(b) for a,b in zip(pp.nins,pp['mean'])]
                v=pp.observation_variance.to_numpy();assert np.isfinite(v).all() and (v>0).all()
                values=[math.fsum(abs(e) for e in errors)/N,math.sqrt(math.fsum(e*e for e in errors)/N),
                    -math.fsum(norm.logpdf(a,loc=b,scale=math.sqrt(c)) for a,b,c in zip(pp.nins,pp['mean'],v))/N,
                    sum(abs(e)<=norm.ppf(.975)*math.sqrt(c) for e,c in zip(errors,v))/N]
                np.testing.assert_allclose(values,[r[prefix+'_'+m] for m in ['MAE','RMSE','NLPD','coverage']],rtol=1e-12,atol=1e-13)
        else:assert (d/'failure.json').exists()
        checks.append(dict(replicate=b,n_train=n,M=M,status=r['status'],saved_output_check=True))
    for (b,n),g in df[df.n_train!=88].groupby(['replicate','n_train']):
        assert len(g)==16 and g.subset_hash.nunique()==1 and g.inside_count.nunique()==1
    s=pd.read_csv(out/stage/'summary.csv')
    assert s[s.n_train==88].fixed_MAE_sd.isna().all()
    pd.DataFrame(checks).to_csv(out/stage/'saved_output_checks.csv',index=False)
    pd.concat(restart_frames,ignore_index=True).to_csv(out/stage/'all_restarts.csv',index=False)
    save(out/stage/'saved_output_audit.json',dict(passed=True,checked=len(checks),
        baseline_reused_once=True,training_response_isolation=True,unchanged_source_hashes=True))



def precision_details(out,stage,df):
    """Diagnose every returned high-condition/failed fit plus extremes at each n.
    Existing acceptance decisions are never changed by this supplementary audit.
    """
    from audit import high_precision_sr
    import sr
    from threadpoolctl import threadpool_limits
    frame=df[(df.n_train!=88)&df.fit_success.astype(bool)].copy()
    numeric=pd.to_numeric(frame.condition_number,errors='coerce')
    indices=set(frame[(numeric>1e10)|(frame.status!='ok')].index)
    for n,g in frame.groupby('n_train'):
        indices.add(g.condition_number.idxmax());indices.add(g.condition_number.idxmin())
    full=pd.read_parquet(SOURCE/'training.parquet')
    target=pd.read_csv(out/'fixed_targets.csv',float_precision='round_trip')
    domain=read(out/'domain.json');rows=[]
    with threadpool_limits(1):
        for index in sorted(indices):
            r=frame.loc[index];b,n,M=int(r.replicate),int(r.n_train),int(r.M);k=int(round(np.sqrt(M)))
            d=out/f'fits/b{b:02d}/n{n:02d}/M{M:03d}'
            record=dict(replicate=b,n_train=n,M=M,original_status=r.status,condition_number=r.condition_number)
            try:
                selected=read(out/f'subsets/b{b:02d}_n{n:02d}.json')['row_indices']
                x,z,tr=subset_arrays(full.iloc[selected],domain['coordinate_min'],domain['coordinate_span'])
                xx=tr.coordinates(target[['coord1','coord2']].to_numpy(float))
                C1=np.cos(np.pi*(np.arange(k)[:,None]+.5)*np.arange(k)[None,:]/k)*np.sqrt(2/k)
                C1[:,0]=np.sqrt(1/k);B=np.kron(C1,C1)
                A=sr.observation_operator(x,k,k)@B;H=sr.observation_operator(xx,k,k)@B
                fit=read(d/'fit.json');q=np.load(d/'basis_variances.npy');noise=fit['noise_variance']
                if r.condition_number>1e10:
                    ll,mean,latent=high_precision_sr(A,H,q,noise,z);record['precision']='70 decimal digits'
                else:
                    C=(A*q)@A.T+noise*np.eye(n);cross=(H*q)@A.T
                    ll=-.5*(np.linalg.slogdet(C)[1]+z@np.linalg.solve(C,z)+n*np.log(2*np.pi))
                    mean=cross@np.linalg.solve(C,z)
                    latent=np.sum(H*H*q,axis=1)-np.einsum('ij,ji->i',cross,np.linalg.solve(C,cross.T))
                    record['precision']='53 bits direct covariance'
                record['ll_difference']=float(abs(ll-fit['log_marginal_likelihood']))
                record['ll_pass']=bool(np.allclose(ll,fit['log_marginal_likelihood'],rtol=1e-7,atol=1e-7))
                if (d/'prediction_standardized.npz').exists():
                    pred=np.load(d/'prediction_standardized.npz')
                    for key,value in [('mean',mean),('latent_variance',latent),('observation_variance',latent+noise)]:
                        record[key+'_difference']=float(np.max(abs(pred[key]-value)))
                        record[key+'_pass']=bool(np.allclose(pred[key],value,rtol=1e-7,atol=1e-7))
                else:record['prediction_comparison']='not available: original prediction failed'
            except Exception as exc:record['error']=repr(exc)
            rows.append(record)
    pd.DataFrame(rows).to_csv(out/stage/'precision_details.csv',index=False)
    scope=df[['replicate','n_train','M','fit_success','prediction_success','audit_success','reference_precision','status']].copy()
    scope['supplementary_reference']=False
    for r in rows:
        scope.loc[(scope.replicate==r['replicate'])&(scope.n_train==r['n_train'])&(scope.M==r['M']),'supplementary_reference']=True
    scope.to_csv(out/stage/'audit_scope.csv',index=False)
    print('Supplementary precision cases',len(rows),flush=True)


def heatmap(s,column,title,path,percent=False,cmap='viridis'):
    ns=[88,*NS];ms=[k*k for k in KS]
    pivot=s.pivot(index='n_train',columns='M',values=column).reindex(index=ns,columns=ms)
    pivot.to_csv(path.with_suffix('.csv'))
    fig,ax=plt.subplots(figsize=(15,6))
    values=pivot.to_numpy(float);masked=np.ma.masked_invalid(values)
    cm=plt.get_cmap(cmap).copy();cm.set_bad('#c9c9c9')
    log_color=column=='fixed_NLPD_mean' and np.nanmin(values)>0
    if log_color:
        from matplotlib.colors import LogNorm
        im=ax.imshow(masked,aspect='auto',cmap=cm,norm=LogNorm(vmin=np.nanmin(values),vmax=np.nanmax(values)))
        title+='\nLog color scale; labels in original NLPD units'
    else:
        im=ax.imshow(masked,aspect='auto',cmap=cm,vmin=0 if percent else None,vmax=1 if percent else None)
    failures=s.pivot(index='n_train',columns='M',values='failure_rate').reindex(index=ns,columns=ms).to_numpy()
    mid=np.sqrt(np.nanmin(values)*np.nanmax(values)) if log_color else (np.nanmin(values)+np.nanmax(values))/2
    for i in range(8):
        for j in range(16):
            v=values[i,j];text='NA' if not np.isfinite(v) else (f'{v:.0%}' if percent else f'{v:.1e}' if abs(v)>=1e4 else f'{v:.1f}')
            if failures[i,j]>0:text+='*'
            ax.text(j,i,text,ha='center',va='center',fontsize=8,color='black' if not np.isfinite(v) or v>mid else 'white')
    ax.set_xticks(range(16),ms);ax.set_yticks(range(8),['88 (fixed)']+[str(n) for n in NS])
    if column.startswith('fixed_'):
        note='* some fits failed; means conditional on successful fits; gray = unavailable'
    elif column=='zero_ratio_median':
        note='All returned fits included; * audit failures present; gray = unavailable'
    else:
        note='Rates over all scheduled fits; * failures present; gray = unavailable'
    ax.set(xlabel='M = k squared; grid resolution changes together',ylabel='n_train',title=title+'\n'+note)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.02);fig.subplots_adjust(left=.10,right=.95,bottom=.13,top=.81)
    fig.savefig(path.with_suffix('.png'),dpi=220);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


def figures(out,stage,df,s):
    directory=out/stage/'figures';directory.mkdir(exist_ok=True)
    for col,title,file,percent in [
        ('fixed_MAE_mean','Figure 1 | Mean MAE on 21 fixed sites','figure1_mae',False),
        ('fixed_NLPD_mean','Figure 2 | Mean NLPD, observation variance','figure2_nlpd',False),
        ('fixed_coverage_mean','Figure 3 | 95% prediction interval coverage','figure3_coverage',True),
        ('fit_success_rate','Figure 4a | At least one converged valid restart','figure4a_convergence',True),
        ('failure_rate','Figure 4b | Fit, prediction, or audit failure rate','figure4b_failures',True),
        ('audit_success_rate','Figure 4c | Independent audit success rate','figure4c_audit',True),
        ('zero_ratio_median','Figure 6 | Median exactly-zero stored variance fraction','figure6_sparsity',True)]:
        heatmap(s,col,title,directory/file,percent)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
    for ax,n in zip(axes.flat,[88,*NS]):
        g=s[s.n_train==n].sort_values('M');x=g.M.to_numpy();y=g.fixed_MAE_mean.to_numpy()
        ax.plot(x,y,'o-',markersize=3)
        if n!=88:
            sd=g.fixed_MAE_sd.to_numpy();ax.fill_between(x,y-sd,y+sd,alpha=.2)
        ax.set(title=f'n={n}'+(' | single fixed network' if n==88 else ' | mean +/- MC SD'),xlabel='M',ylabel='MAE')
        ax.grid(alpha=.25)
    fig.suptitle('Figure 5 | Fixed 21-site MAE; variability over network thinning, not a confidence interval')
    fig.tight_layout();fig.savefig(directory/'figure5_performance.png',dpi=220);fig.savefig(directory/'figure5_performance.pdf');plt.close(fig)
    s[['n_train','M','fixed_MAE_mean','fixed_MAE_sd','successes','attempts']].to_csv(directory/'figure5_performance.csv',index=False)
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for ax,n in zip(axes.flat,[88,*NS]):
        values=[pd.to_numeric(df[(df.n_train==n)&(df.M==k*k)].condition_number,errors='coerce').dropna().to_numpy() for k in KS]
        ax.boxplot(values,positions=range(16),widths=.6,showfliers=True)
        ax.set_xticks(range(16),[str(k*k) for k in KS],rotation=90,fontsize=7)
        ax.set(yscale='log',title=f'n={n}'+(' | one value per M' if n==88 else ''),xlabel='M',ylabel='Condition number')
        ax.grid(axis='y',alpha=.25)
    fig.suptitle('Figure 4d | Condition numbers of returned fits (including subsequent audit failures)')
    fig.tight_layout();fig.savefig(directory/'figure4d_condition.png',dpi=220);fig.savefig(directory/'figure4d_condition.pdf');plt.close(fig)
    df[['replicate','n_train','M','status','condition_number','zero_ratio']].to_csv(directory/'figure4d_condition.csv',index=False)
    df[['replicate','n_train','M','status','zero_ratio']].to_csv(directory/'figure6_sparsity_distribution.csv',index=False)


def report(out,stage,df,s):
    directory=out/stage
    pair=pd.read_csv(directory/'paired_summary.csv')
    best=[]
    for n in [88,*NS]:
        g=s[s.n_train==n];a=g.loc[g.fixed_MAE_mean.idxmin()];r=g.loc[g.fixed_RMSE_mean.idxmin()]
        p=g.loc[g.fixed_NLPD_mean.idxmin()];c=g.loc[(g.fixed_coverage_mean-.95).abs().idxmin()]
        baseline=g[g.M==256].iloc[0]
        d=pair[(pair.n_train==n)&(pair.M==a.M)]
        best.append(dict(n_train=n,observed_MAE_min_M=int(a.M),MAE_mean=a.fixed_MAE_mean,MAE_sd=a.fixed_MAE_sd,
            best_successes=int(a.successes),M256_successes=int(baseline.successes),RMSE_min_M=int(r.M),NLPD_min_M=int(p.M),coverage_nearest95_M=int(c.M),MAE_M256=baseline.fixed_MAE_mean,
            paired_delta_mean=float(d['mean'].iloc[0]) if len(d) else None,paired_delta_sd=float(d['std'].iloc[0]) if len(d) else None,
            paired_count=int(d['count'].iloc[0]) if len(d) else None))
    best=pd.DataFrame(best);best.to_csv(directory/'observed_minima.csv',index=False)
    def table(frame):
        text='|'+'|'.join(frame.columns)+'|\n|'+'|'.join(['---']*len(frame.columns))+'|\n'
        for _,r in frame.iterrows():text+='|'+'|'.join('NA' if pd.isna(v) else f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v) for v in r)+'|\n'
        return text
    failures=df[df.status!='ok'];thinned=df[df.n_train!=88]
    convergence=thinned.groupby('n_train').agg(fits=('status','size'),successes=('metric_success','sum'),converged_models=('fit_success','sum'),
        audit_passed=('audit_success','sum'),inside_min=('inside_count','min'),inside_max=('inside_count','max'),condition_max=('condition_number','max')).reset_index()
    convergence.to_csv(directory/'status_by_n.csv',index=False)
    summary=read(directory/'completion.json')
    observed_sequence=', '.join(f"n={int(r.n_train)}: M={int(r.observed_MAE_min_M)}" for _,r in best.iterrows())
    pairs=pair[(pair.M!=256)&(pair['count']>0)]
    difference_range=f"{pairs['mean'].min():.3f} to {pairs['mean'].max():.3f}"
    sd_range=f"{pairs['std'].min():.3f} to {pairs['std'].max():.3f}"
    failure_groups=failures.groupby('n_train').size().to_dict()
    matching=best[(best.observed_MAE_min_M==best.NLPD_min_M)&(best.observed_MAE_min_M==best.coverage_nearest95_M)].n_train.tolist()
    q256=s[s.M==256].sort_values('n_train',ascending=False)
    zero_sequence=', '.join(f"n={int(r.n_train)}: {r.zero_ratio_median:.1%}" for _,r in q256.iterrows())
    text=f"""# 2025 SR-Kriging Monte Carlo Thinning — {stage}

{summary['B']}회 중첩 thinning, 고유 thinning 적합 {len(thinned)}개, n=88 기존 적합 16개를 한 번씩 재사용했다. 총 완료 평가 {int((df.status=='ok').sum())}개, 실패 {len(failures)}개다. 기준 결과의 재계산 예측과 MAE/RMSE는 정확히 일치했다.

## 관측된 결과

{table(best[['n_train','observed_MAE_min_M','MAE_mean','MAE_sd','MAE_M256','paired_delta_mean','paired_delta_sd','paired_count']])}

{table(best[['n_train','RMSE_min_M','NLPD_min_M','coverage_nearest95_M','best_successes','M256_successes']])}

이 표의 최소값은 외부 평가 결과를 기술한 것이며 자동 M 선택·새 모델의 성능 검증이 아니다. 성공한 적합만의 평균이므로 실패율과 paired 표본 수를 함께 읽어야 한다. Coverage 기준은 0.95와 평균 coverage의 거리이며 단순 최대 coverage가 아니다.

{table(convergence)}

## 연구 질문에 대한 답

1. 관측 최소 MAE 위치는 {observed_sequence}이다. 관측 수가 달라질 때 최소점도 달라졌다. 그러나 중첩 관측망의 결과를 서로 독립적인 표본으로 취급하지 않는다.
2. M=256 자기 비교를 제외한 전체 (n,M)의 paired 평균 차이는 {difference_range}, 반복 간 SD는 {sd_range} 범위였다. paired_delta_mean/sd는 같은 반복·같은 관측망에서 MAE(M)-MAE(256)를 계산한 후 반복 단위로 요약했다. 두 적합이 모두 성공한 반복만 포함한다. 전체 M의 평균·표준편차·중앙값·범위 및 유효 쌍 수는 paired_summary.csv, 반복별 값과 제외 이유는 paired_differences.csv에 있다. n=88에는 Monte Carlo SD나 CI를 생성하지 않았다.
3. 실패의 n별 개수는 {failure_groups}이다. 전체 반환 적합의 최대 조건수는 {pd.to_numeric(df.condition_number,errors='coerce').max():.6g}였다. status_by_n.csv와 failure 표는 수렴, 예측, 독립 검증을 구분한다. 조건수는 적합 결과가 반환된 모델에 대해 요약하며, 이후 검증 실패도 조건수 분포에 포함한다. 큰 M의 안정성은 Figure 4의 M별 실패율과 함께 판단한다. 실제 실패 상세는 failures.csv 및 각 failure.json에 보존했다.
4. M=256에서 정확한 0 비율 중앙값은 {zero_sequence}이다. 모든 M의 비율 분포는 summary.csv 및 Figure 6에 있다. 기저 분산이 반환된 모든 적합을 포함하며, 이후 독립 검증에 실패한 적합도 이 수치적 현상 집계에는 남겨두었다. 이 변화가 n 또는 M에 대해 반드시 단조적이라고 가정하지 않는다. 양수→0 로그에서 float 최소 subnormal보다 작은 곱을 구분했지만 모든 underflow 사건을 포괄 계수한 것은 아니다. q=0 개수를 통계적 유효 기저 수나 공분산 rank로 부르지 않는다.
5. MAE, NLPD, 0.95에 가까운 coverage의 세 기준에서 같은 M이 기록된 n은 {matching}이다. 세 기준이 항상 일치하지 않는다. 동률은 표 순서상 첫 M을 표시하며 전체 값은 summary.csv에 있다. MAE만으로 불확실성의 적절성을 판정하지 않는다. NLPD 및 coverage는 측정오차를 포함한 관측분산과 원단위 역변환을 사용한다.
6. 이 결과에서 M=256이 최소 평균 MAE를 기록한 n 조건은 {best.loc[best.observed_MAE_min_M==256,"n_train"].tolist()}이다. M=256의 평균 MAE와 동일 관측망 paired 차이를 함께 제시했다. 이 한 날짜·한 원 관측망에서의 결과를 다른 날짜나 지역의 일반적 우수성으로 확대하지 않는다.

## 실험 통제와 구현

기존 sr.py, data.py, audit.py 및 기존 실험 파일은 변경하지 않았다. 핵심 수식, 8 starts, maxiter=10000, |ΔLL|<1e-8, 기존 독립 검증 허용오차를 유지했다. 기저 수 M=k²와 격자 해상도 k×k가 동시에 달라지므로 독립적인 기저 수 인과효과로 해석할 수 없다. k=1은 기존 상수 기저 처리다.

각 b에 child_seed(20261013,2025,b)로 독립 순열을 생성하고 앞 n개를 선택한 다음 기존 행 순서로 정렬했다. 위치 선택에 반응값이나 테스트 오차를 사용하지 않는다. 20개 순열·140개 부분집합 manifest를 결과 확인 전에 저장했다. 같은 (b,n)의 모든 M에 동일한 입력을 제공한다.

좌표 영역은 원 88개 위치의 최소·최대 좌표로 고정했다. 기존 TrainingTransform.fit을 부분집합에 그대로 호출하면 영역도 바뀌므로 새 subset_arrays에서 좌표 영역과 반응 표준화를 분리했다. 평균과 ddof=0 표준편차는 선택된 n개 반응만 사용한다. 제외된 위치와 외부 테스트 반응은 표준화에 사용하지 않는다. 표본 선택 후 원자료 단위를 바꾸지 않는다.

모든 적합은 기존 공통 fit seed 1448741610과 기존 초기값 생성 규칙을 사용한다. M개 q를 뽑은 뒤 noise를 뽑는 순서를 유지하므로 다른 M의 초기 noise 값 자체는 다를 수 있다. 새로운 랜덤 재시작 예산을 추가하지 않았다.

주 분석은 원 hull 내부 21개 고정 위치의 성능이며 thinning hull 밖 예측을 포함할 수 있다. 이를 순수 내부 보간 성능이라고 부르지 않는다. 보조 분석은 같은 21개 중 thinning hull 내부만 사용하며 개수·평가 불가 사유를 저장했다. 내부 위치 0개는 NA, 1개 이상은 기술적 지표를 계산하지만 작은 표본의 불안정성을 인정한다.

## 검증 범위와 저장·재개

기존 및 추가 테스트는 tests.txt/tests.json에 있다. 모든 반환된 적합에서 예측을 시도하고, 예측에 성공하면 기존 audit_fit으로 독립 코사인 기저·직접 공분산을 검증했다. 조건수 >1e10이면 기존 70자리 고정밀 참조 계산을 사용한다. 학습이나 예측 자체가 실패한 경우 검증 통과로 간주하지 않았다. 이 실행의 검증 범위는 audit_scope.csv와 각 audit.json에 있다. 추가로 조건수>1e10인 반환된 모든 적합, 실패한 반환 적합, n별 조건수 양끝 사례를 precision_details.csv에서 검사했다. 실패한 경우도 likelihood·예측평균·잠재/관측 분산의 각 오차를 분리했다. 추가 진단으로 기존 성공/실패 판정을 바꾸지 않았다.

saved_output_audit.json은 원본 레이블·고정21개 ID·표본 재현·변환값·q 배열·저장된 예측으로 재계산한 지표를 검사한다. 모든 모델은 개별 폴더에 즉시 원자적 result.json을 저장하며, 실패도 완료된 기록으로 남겨 임의 재시도하지 않는다. 중단된 불완전 파일은 interrupted 폴더에 보존한다. 코어·실행 소스 또는 데이터 hash가 다르면 재개를 차단한다.

파일럿 3회는 본실험 20회 중 b=0,1,2와 동일하므로 그대로 포함했다. 20회 동안 n=88을 반복 학습하지 않았다. 기본 실행은 4개 프로세스, 각 선형대수 스레드 1개다. fit_seconds에는 underflow 진단을 위한 추가 계산이 포함된다. 전체 진행 시간은 pilot_timing.json/full_timing.json, 계산량 합계는 completion.json에 있다.

실행: `python thinning_2025.py --stage pilot`; 파일럿 점검 뒤 같은 경로로 `python thinning_2025.py --stage full --output <run directory>`. 후처리: `python thinning_2025_report.py <run directory> --stage full`. pilot_review.json은 사용자 재승인 요구가 아니라 파일럿 검증 결과를 기록하는 실행 단계 점검이다.

포스터 그림은 figures/의 PNG와 PDF, 각 그림 원자료는 같은 이름 CSV로 저장했다. summary.csv는 주·보조 지표를 별도 접두어로 구분한다. 개별 위치 예측, q 원본 npy, 초기값, 8개 재시작 결과, gzip trace, 오류를 fits/bXX/nXX/MXXX/에 저장했다. 원본 n=88 배열·예측은 baseline.json이 가리키는 기존 디렉터리를 재사용한다.
"""
    (directory/'REPORT_KO.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--stage',choices=['pilot','full'],default='full')
    args=parser.parse_args();out=args.output
    df=pd.read_csv(out/args.stage/'individual_fits.csv',float_precision='round_trip')
    s=pd.read_csv(out/args.stage/'summary.csv',float_precision='round_trip')
    verify(out,df,args.stage);precision_details(out,args.stage,df);figures(out,args.stage,df,s);report(out,args.stage,df,s)
    print('Verified and reported',out/args.stage,flush=True)

if __name__=='__main__':main()
