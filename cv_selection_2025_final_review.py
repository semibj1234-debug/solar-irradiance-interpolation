"""Final artifact completeness and reporting review; never modifies model fits."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cv_selection_2025 import ROOT,PREVIOUS,SOURCE,read,sha,save
out=ROOT/'results/cv_selection_2025_20261013_run1';d=out/'full';fig=d/'figures'
m=read(out/'manifest.json')
assert all(sha(ROOT/name)==h for name,h in m['hashes'].items())
assert sha(SOURCE/'training.parquet')==m['training_hash']
assert sha(PREVIOUS/'manifest.json')==m['previous_manifest_hash']
expected=set()
for key,h in m['fold_manifest_hashes'].items():
    path=out/'cv_fold_manifest'/f'{key}.json';assert sha(path)==h
    manifest=read(path)
    if manifest['n_train']!=88:
        assert sha(Path(manifest['thinning_manifest']))==manifest['thinning_hash']
        assert read(Path(manifest['thinning_manifest']))['row_indices']==manifest['row_indices']
    for method,info in manifest['methods'].items():
        for k in range(1,17):
            for i in range(len(info['folds'])):expected.add(f'{key}/{method}/M{k*k:03d}/fold{i:03d}.json.gz')
actual={p.relative_to(out/'cv_fold_results').as_posix() for p in (out/'cv_fold_results').glob('*/*/*/fold*.json.gz')}
assert actual==expected and len(expected)==122528
fold=pd.read_csv(d/'cv_fold_results.csv');candidate=pd.read_csv(d/'cv_candidate_summary.csv');selection=pd.read_csv(d/'selection_results.csv');e=pd.read_csv(d/'external_evaluation.csv')
assert not fold.duplicated(['key','method','M','fold']).any()
assert {f"{r.key}/{r.method}/M{r.M:03d}/fold{r.fold:03d}.json.gz" for r in fold.itertuples()}==expected
assert len(candidate)==6768 and not candidate.duplicated(['key','method','M']).any()
assert len(selection)==846 and not selection.duplicated(['key','method','rule']).any()
assert (selection.n_train==88).sum()==6
assert e.fixed_count.eq(21).all() and e.reuse_revalidation_passed.all()
assert read(d/'saved_output_audit.json')['passed']
assert sha(d/'selection_results.csv')==read(d/'evaluation_complete.json')['selection_hash']
levels=[('CV fold',len(fold),int((fold.status!='ok').sum())),('CV candidate',len(candidate),int((candidate.status!='complete_cv').sum())),('Selection',len(selection),int((selection.selection_status!='selected').sum())),('Final independent validation',len(e),int((~e.reuse_revalidation_passed.fillna(False)).sum()))]
fail=pd.DataFrame(levels,columns=['level','total','failed']);fail['failure_rate']=fail.failed/fail.total;fail.to_csv(d/'four_level_failure_summary.csv',index=False)
shutil.copy2(d/'selected_M_distribution.csv',fig/'figure1b_selected_M_distribution.csv')
shutil.copy2(out/'spatial_maps/fold_positions.csv',fig/'fold_examples.csv')
# Keep the original linear-scale NLPD figure; add a clearly labelled log view without clipping.
a=pd.read_csv(fig/'figure5a_NLPD.csv');a.to_csv(fig/'figure5c_NLPD_log.csv',index=False)
f,ax=plt.subplots(figsize=(10,5))
for label,g in a.groupby('selector'):
    g=g.sort_values('n_train');assert (g['mean']>0).all();ax.plot(g.n_train,g['mean'],'.-',label=label)
ax.set(yscale='log',xlabel='Training locations n',ylabel='Mean NLPD (log axis; unchanged values)',title='Observation NLPD: extreme under-dispersion remains visible')
ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);f.tight_layout()
for ext in ['png','pdf']:f.savefig(fig/('figure5c_NLPD_log.'+ext),dpi=180)
plt.close(f)
fail.to_csv(fig/'figure6c_four_failure_levels.csv',index=False)
f,ax=plt.subplots(figsize=(9,4));ax.bar(fail.level,100*fail.failure_rate);ax.set(ylabel='Failure rate (%)',title='Four distinct failure denominators')
for i,r in enumerate(fail.itertuples()):ax.text(i,100*r.failure_rate+.03,f'{r.failed}/{r.total}',ha='center',fontsize=9)
ax.tick_params(axis='x',labelsize=8);f.tight_layout()
for ext in ['png','pdf']:f.savefig(fig/('figure6c_four_failure_levels.'+ext),dpi=180)
plt.close(f)
# Independently recompute the worst selected NLPD cases from original-unit observation variances.
provenance=pd.read_csv(d/'reused_artifacts.csv');extremes=[]
for r in e.nlargest(5,'fixed_NLPD').itertuples():
    source=provenance[(provenance.key==r.key)&(provenance.M==r.selected_M)].iloc[0]
    q=pd.read_csv(source.prediction_path);v=q.observation_variance.to_numpy();error=q.nins.to_numpy()-q['mean'].to_numpy();assert np.isfinite(v).all() and (v>0).all()
    nlpd=float(np.mean(.5*(np.log(2*np.pi*v)+error**2/v)));np.testing.assert_allclose(nlpd,r.fixed_NLPD,rtol=1e-12)
    extremes.append(dict(key=r.key,selector=r.selector,M=r.selected_M,NLPD=nlpd,min_observation_variance=v.min(),max_absolute_error=np.abs(error).max(),noise_variance_original=r.noise_variance_original))
pd.DataFrame(extremes).to_csv(d/'extreme_NLPD_review.csv',index=False)
for png in fig.glob('*.png'):
    assert png.read_bytes().startswith(b'\x89PNG')
    assert png.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    assert png.with_suffix('.csv').exists(),str(png)
# Preserve the generated report before adding the final scientific review.
report=d/'REPORT_KO.md';backup=d/'REPORT_KO_generated.md'
if not backup.exists():shutil.copy2(report,backup)
paired=pd.read_csv(d/'paired_fixed256_summary.csv');summary=e.groupby(['n_train','selector']).agg(MAE=('fixed_MAE','mean'),SD=('fixed_MAE','std'),NLPD=('fixed_NLPD','mean'),Coverage=('fixed_coverage','mean'),M_median=('selected_M','median')).reset_index()
winners=summary.loc[summary.groupby('n_train').MAE.idxmin()].copy();winners.to_csv(d/'observed_lowest_mean_MAE.csv',index=False)
effect=pd.read_csv(fig/'figure4_one_se_effect.csv')
lines=['# 최종 완료 검토 — SR-Kriging CV 기저 차원 선택','',
'전체 122,528개 내부 적합이 저장됐다. 122,326개 정상, 78개 미수렴, 124개 독립 검증 실패다. 실패를 정상 CV 점수로 바꾸지 않았으며 151개 불완전 후보는 선택에서 제외했다. 846개 선택 모두 성공했고 최종 선택 모델의 재검증도 모두 통과했다. 동일한 최종 모델을 여러 선택 규칙이 선택한 경우 중복 학습하지 않았다.',
fail.to_markdown(index=False),
'## 관측 수별 최소 평균 MAE (사후 관찰)',
winners.round(3).to_markdown(index=False),
'표는 각 n에서 관찰된 평균의 순위다. 방법의 일반적 우수성이나 통계적 유의성을 뜻하지 않는다. n=11~77은 각각20회, n=88은 한 번이다. SD는 MC 반복 간 표준편차이며 n=88에는 정의하지 않는다.',
'## 연구 질문에 대한 답',
'**작은 M 선택:** Spatial 1-SE의 선택 M 중앙값은 모든 n에서 1이었다. Random 1-SE도 n<=55에서 대부분 중앙값1을 선택했다(n=11 중앙값2.5). LOOCV minimum은 n=22~77에서 중앙값49~100으로 대체로 더 큰 M을 선택했다. 공간 슬랩 CV는 coord1 방향으로 연속 영역을 제외하는 설계이며, 이 방향성을 일반적 Spatial CV 전체의 성질로 확대하지 않는다.',
'**외부 MAE:** n=11,44에서는 Spatial 1-SE, n=22,33,55에서는 Random 1-SE, n=66에서는 Random minimum, n=77에서는 Spatial minimum의 평균이 가장 낮았다. n=88에서는 Random minimum과 LOOCV 두 규칙이 M49, MAE247.332로 같았다. 모든 n에서 우세한 단일 방법은 관찰되지 않았다.',
'**1-SE 효과:** 1-SE는 선택 M을 늘리지 않았지만 MAE를 항상 유지하지는 않았다. 예를 들어 Random CV의 n=66에서는 minimum보다 평균 MAE가11.964 높았고, n=88에서는51.279 높았다. n=11~55에서는 세 CV 방식 모두 1-SE의 평균 MAE가 minimum보다 낮았지만, 많은 paired 95% 구간은0을 포함한다. 따라서 비열등성을 검증한 결과라고 주장할 수 없다.',
'**불확실성:** n=11에서 LOOCV minimum의 평균 NLPD는 약1.003e10, Coverage는0.486이었다. Spatial 1-SE는 NLPD8.462, Coverage0.888로 덜 극단적이었지만 명목0.95에 못 미쳤다. 극단 NLPD 상위5개를 원단위 관측 예측분산으로 재계산하여 일치를 확인했다. 작은 양의 예측분산과 큰 오차가 큰 NLPD를 만들며, 수치 검증 통과가 확률 예측의 보정을 보장하지 않는다. 선형 축 그림은 보존하고 값의 clipping 없이 로그 축 보조 그림을 추가했다.',
'**수치 안정성:** 실패는 n=11(155/6720)과 n=22(47/10240)에 집중됐다. 전체 CV fold 실패율은 Random46/11280, Spatial46/11280, LOO110/99968이다. 절대 실패 수와 분모가 다르므로 단순 개수로 비교하지 않는다. 최종 선택 모델은 여섯 방법 모두 재검증에 통과하여 이 기준만으로 안정성 우열을 정할 수 없다.',
'**고정 M256 대비:** 모든 n 및 방법의 유효 paired 평균 MAE 차이는 음수였다. 그러나 n=11의 M256은3개 적합이 실패하여 비교는17쌍에 조건부다. 예를 들어 n=11 Spatial 1-SE minus M256은 평균-154.943, SD219.011, 95% t구간[-267.548,-42.338]이다. n=66 Random minimum은 평균-88.281, SD72.273, 20쌍이다. 일부 다른 조건의 구간은0을 포함하므로 전체 설정에서 유의한 개선이라고 단정하지 않는다.',
'**계산 비용과 실용성:** Random/Spatial은 각각11,280회, LOO는99,968회 적합했다. 작업별 경과시간 합은 각각 약5.34/5.07/43.14시간이며 병렬 벽시계 시간이 아니다. 1-SE는 같은 CV 결과의 후처리이므로 추가 학습 비용이 없다. 이 스냅샷에서는 Random/Spatial 5-fold와 1-SE가 비용·평균 MAE·불확실성 측면의 후보이나, 단일 보편적 추천 규칙을 확정한 것은 아니다.',
'## 해석 및 완료 범위',
'본실험 workflow는 2026-10-15 02:42 KST에 완료됐고, 파일럿 이후 벽시계 시간은약4시간47분35초다. 파일럿 계산·요약 약27분28초는 별도다. 원자료의 날짜는2024-08-01 12:00 KST 단일 스냅샷이다. 동일88개 원 관측망에서 만든20개 부분집합을 반복 단위로 사용했다. M과 격자 해상도는 함께 바뀌고 외부 평가는21개 위치로 제한된다. 고정 외부 집합은 thinning hull 밖 위치를 포함할 수 있으며 inside_* 지표는 별도로 저장했다. 1-SE의 SE는 공간 의존성을 엄밀히 반영한 신뢰구간이 아닌 휴리스틱이다. paired t구간은 동일 원 관측망의 MC 부분집합 변동성을 요약하며 여러 n과 방법의 반복 비교에 대한 보정된 검정이 아니다.',
'전체 실제 체크포인트 집합과 기대 조합 집합이 정확히 일치하고, 중복 CSV 키가 없음을 재확인했다. 핵심 소스/데이터/fold/thinning 해시가 보존됐고 n=88 선택 결과는 정확히6행이다. 최종 예측은 모두21개 위치이며 기존 독립 검사와 선택 결과 해시가 일치한다. 추가 연구 단계는 실행하지 않았다.',
'## 기존 상세 결과 및 실행 경로','']
report.write_text('\n\n'.join(lines)+backup.read_text(encoding='utf-8'),encoding='utf-8')
save(d/'final_review.json',dict(passed=True,expected_and_actual_checkpoints=122528,duplicate_fold_keys=0,source_data_manifest_hashes_match=True,candidates=6768,selections=846,n88_selection_rows=6,unique_selected_models=int(e[['key','selected_M']].drop_duplicates().shape[0]),all_final_targets_21=True,extreme_NLPD_recomputed=5,png_pdf_csv_sets=len(list(fig.glob('*.png'))),review_source_hash=sha(Path(__file__)),report_hash=sha(report)))
print(read(d/'final_review.json'))
