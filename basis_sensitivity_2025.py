"""Fixed N=88, k=1..16 grid/basis sensitivity. Run: python basis_sensitivity_2025.py.
Reuses 2025 equations; no CV, pruning, clipping, or new estimator.
All output goes to a new timestamped directory; existing runs are read-only.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import ast
import io
import json
import math
import platform
import shutil
import time
import traceback
import unittest
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import Delaunay
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import sr
from audit import audit_fit, audit_metrics
from data import child_seed, read, sha
from single_2025 import ROOT, SOURCE, SETTINGS, SEED, load_and_check, training_arrays, save

REFERENCE = ROOT / "results/single_2025_20261013T015544Z"
KS = tuple(range(1,17))
FIT_SEED = child_seed(SEED,88,2025)


def settings_for(k):
    """Full k-by-k DCT grid; M=k*k. Never truncate a fixed 16-by-16 grid."""
    if k not in KS:
        raise ValueError("k must be in 1..16")
    s = dict(SETTINGS,grid_nx=k,grid_ny=k,basis_count=k*k)
    s.pop("basis_nx",None); s.pop("basis_ny",None)
    return s


def initial_values(x, settings):
    """Input x(N,2); return q0(8,M), noise0(8,), restart seeds(8,).
    Same seed and generator rule for every k. Noise is drawn AFTER M weights,
    as in the baseline, so noise draws need not be equal across different M.
    """
    A = sr.design(x,settings)
    base = 1/np.mean(np.sum(A*A,axis=1))
    qs, noises, seeds = [], [], []
    for restart in range(settings["starts"]):
        seed = child_seed(FIT_SEED,8101,restart)
        rng = np.random.default_rng(seed)
        qs.append(base*rng.uniform(*settings["initial_weight_multiplier"],size=A.shape[1]))
        noises.append(np.exp(rng.uniform(*np.log(settings["initial_noise_variance"]))))
        seeds.append(seed)
    return np.asarray(qs), np.asarray(noises), seeds


def monitored_fit(x,z,settings,diagnostics):
    """Train-only inputs x(N,2), z(N,). Return existing fit_sr dictionary.
    A wrapper observes unchanged update outputs. If positive q becomes zero,
    log(q)+2log(abs(b))-log(h) diagnoses a below-subnormal exact product using
    the same float b,h. It is diagnostic only and never replaces model values.
    b==0 transitions are kept separate: exact cancellation/earlier underflow
    cannot be distinguished by this monitor. Suppressed subnormal rounding is
    not exhaustively trapped; no claim that all underflow events are counted.
    """
    original = sr.update
    diagnostics.update(update_calls=0,positive_to_zero=0,below_subnormal_to_zero=0,
                       zero_b_to_zero=0,other_to_zero=0,nonfinite_update=False,
                       underflow_monitor="positive-to-zero only; nonzero subnormal rounding not exhaustively trapped")
    def observed(A,y,q,noise):
        nq, nn = original(A,y,q,noise)
        diagnostics["update_calls"] += 1
        diagnostics["nonfinite_update"] |= bool(not np.isfinite(nq).all() or not np.isfinite(nn))
        hit = (q>0)&(nq==0)
        if hit.any():
            diagnostics["positive_to_zero"] += int(hit.sum())
            s = sr.state(A,y,q,noise)
            b,h = s["b"][hit],s["h"][hit]
            nonzero = b!=0
            tiny = np.log(q[hit][nonzero])+2*np.log(abs(b[nonzero]))-np.log(h[nonzero]) < np.log(np.nextafter(0.,1.))
            diagnostics["below_subnormal_to_zero"] += int(tiny.sum())
            diagnostics["zero_b_to_zero"] += int((~nonzero).sum())
            diagnostics["other_to_zero"] += int(nonzero.sum()-tiny.sum())
        return nq,nn
    sr.update = observed
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always",RuntimeWarning)
            try:
                return sr.fit_sr(x,z,settings,FIT_SEED)
            finally:
                diagnostics["runtime_warnings"] = sorted(set(str(w.message) for w in caught))
    finally:
        sr.update = original


def run_setting(k,x,z,xx,target,hull,defined,tr,out):
    """Fit without external labels; return status row and optional prediction frame.
    x(N,2),z(N,),xx(27,2),hull/defined(27,); predictions use all 23 defined sites.
    """
    directory=out/f"M{k*k:03d}"
    directory.mkdir()
    settings=settings_for(k)
    save(directory/"settings.json",settings)
    q0,n0,seeds=initial_values(x,settings)
    np.savez(directory/"initial_values.npz",weights=q0,noise=n0,restart_seeds=seeds)
    row=dict(k=k,M=k*k,status="failed",stage="fit",audit_passed=False)
    diagnostics={}; begun=time.perf_counter()
    try:
        fit=monitored_fit(x,z,settings,diagnostics)
        row["fit_seconds"]=time.perf_counter()-begun
        trace=pd.DataFrame(fit.pop("trace"))
        trace.to_csv(directory/"likelihood_trace.csv",index=False)
        save(directory/"fit.json",fit)
        pd.DataFrame(fit["starts"]).to_csv(directory/"starts.csv",index=False)
        q=np.asarray(fit["weights"])
        np.save(directory/"basis_variances.npy",q)
        pd.DataFrame(dict(index=np.arange(k*k),frequency_x=np.arange(k*k)//k,
                          frequency_y=np.arange(k*k)%k,variance_standardized=q,
                          variance_original=q*tr.response_sd**2)).to_csv(directory/"basis_variances.csv",index=False)
        selected=trace[trace.restart==fit["selected_start"]]
        positive=q[q>0]
        errors=[s["error"] for s in fit["starts"] if s["error"]]
        row.update(converged_starts=sum(s["converged"] and s["error"] is None for s in fit["starts"]),
                   selected_start=fit["selected_start"],selected_iterations=int(selected.iteration.iloc[-1]),
                   log_likelihood=fit["log_marginal_likelihood"],last_absolute_ll_change=float(abs(selected.ll.diff().iloc[-1])),
                   condition_number=fit["condition_number"],noise_variance_standardized=fit["noise_variance"],
                   noise_variance_original=fit["noise_variance"]*tr.response_sd**2,
                   zero_count=int((q==0).sum()),positive_count=len(positive),zero_ratio=float((q==0).mean()),
                   min_positive=float(positive.min()) if len(positive) else None,max_variance=float(q.max()),
                   sum_variance=float(q.sum()),subnormal_positive_count=int(((q>0)&(q<np.finfo(float).tiny)).sum()),
                   cholesky_failure=any("positive definite" in e.lower() or "cholesky" in e.lower() for e in errors),
                   restart_errors=json.dumps(errors),maximum_ll_decrease=max(s["maximum_ll_decrease"] for s in fit["starts"]))
        row["stage"]="prediction"
        pred=sr.predict_sr(fit,x,z,xx[defined],settings)
        np.savez(directory/"prediction_standardized.npz",**pred)
        inv=tr.inverse_prediction(pred)
        frame=target.copy()
        frame["in_training_hull"]=hull;frame["prediction_defined"]=defined
        for column,key in [("prediction_mean","mean"),("latent_variance","latent_variance"),("observation_variance","observation_variance")]:
            frame[column]=np.nan;frame.loc[defined,column]=inv[key]
        frame.to_csv(directory/"predictions_unscored.csv",index=False)
        row["stage"]="independent_audit"
        audit=audit_fit("SR",fit,x,z,xx[defined],pred,settings)
        save(directory/"prediction_audit.json",audit)
        row.update(status="predicted",audit_passed=True,stage="awaiting_evaluation")
        return row,frame
    except Exception as exc:
        row.update(error=f"{type(exc).__name__}: {exc}")
        if "fit_seconds" not in row: row["fit_seconds"]=time.perf_counter()-begun
        if hasattr(exc,"starts"):
            pd.DataFrame(exc.starts).to_csv(directory/"starts.csv",index=False)
            pd.DataFrame(exc.trace).to_csv(directory/"likelihood_trace.csv",index=False)
            row["converged_starts"]=sum(s["converged"] and not s["error"] for s in exc.starts)
        save(directory/"failure.json",dict(error=row["error"],traceback=traceback.format_exc()))
        return row,None
    finally:
        save(directory/"numerical_diagnostics.json",diagnostics)
        row.update(positive_to_zero=diagnostics.get("positive_to_zero"),
                   below_subnormal_to_zero=diagnostics.get("below_subnormal_to_zero"),
                   zero_b_to_zero=diagnostics.get("zero_b_to_zero"),
                   nonfinite_update=diagnostics.get("nonfinite_update"))
        save(directory/"status.json",row)


def verify_baseline(out,frame,tr):
    """Exact saved float checks before any other M. Runtime is not compared."""
    d=out/"M256"
    baseline=read(REFERENCE/"fit.json"); current=read(d/"fit.json")
    for key in ["weights","noise_variance","log_marginal_likelihood","selected_start","condition_number"]:
        np.testing.assert_array_equal(current[key],baseline[key])
    np.testing.assert_array_equal(np.load(d/"basis_variances.npy"),np.load(REFERENCE/"basis_variances.npy"))
    previous=pd.read_csv(REFERENCE/"predictions_all.csv",float_precision="round_trip")
    for col in ["pv_id","in_training_hull","prediction_defined","prediction_mean","latent_variance","observation_variance"]:
        np.testing.assert_array_equal(frame[col].to_numpy(),previous[col].to_numpy())
    assert tr.manifest()==read(REFERENCE/"transform.json")
    for col in ["restart","iteration","ll","noise","zero_weights","min_weight","max_weight"]:
        a=pd.read_csv(d/"likelihood_trace.csv",float_precision="round_trip")[col]
        b=pd.read_csv(REFERENCE/"likelihood_trace.csv",float_precision="round_trip")[col]
        np.testing.assert_array_equal(a,b)
    save(out/"baseline_comparison.json",dict(passed=True,reference=str(REFERENCE),
         comparison="exact float equality including all restart traces, q, noise, likelihood, prediction batch and transform"))


def score_saved(row,out,truth):
    """Evaluate AFTER ALL fits. Read saved prediction CSV, join fixed target IDs.
    Returns metrics on the same 21 rows; no fitted parameters can be changed here.
    """
    d=out/f"M{row['M']:03d}"
    f=pd.read_csv(d/"predictions_unscored.csv",float_precision="round_trip")
    np.testing.assert_array_equal(f.pv_id,truth.pv_id)
    f["nins"]=truth.nins.to_numpy()
    mask=f.in_training_hull.to_numpy(bool)
    assert mask.sum()==21
    error=f.nins.to_numpy()[mask]-f.prediction_mean.to_numpy()[mask]
    variance=f.observation_variance.to_numpy()[mask]
    if not np.isfinite(variance).all() or (variance<=0).any():
        raise FloatingPointError("Invalid observation variance; no clipping")
    metrics=dict(MAE=float(abs(error).mean()),RMSE=float(np.sqrt(np.mean(error**2))),
                 NLPD=float(np.mean(.5*(np.log(2*np.pi*variance)+error**2/variance))),
                 coverage95=float(np.mean(abs(error)<=norm.ppf(.975)*np.sqrt(variance))))
    audit_metrics(f.nins,f.prediction_mean,f.observation_variance,mask,metrics)
    independent_coverage=sum(abs(float(a))<=float(norm.ppf(.975))*math.sqrt(float(v)) for a,v in zip(error,variance))/21
    assert independent_coverage==metrics["coverage95"]
    f["absolute_error"]=np.where(mask,abs(f.nins-f.prediction_mean),np.nan)
    f["latent_sd"]=np.sqrt(f.latent_variance);f["observation_sd"]=np.sqrt(f.observation_variance)
    f.to_csv(d/"predictions_all.csv",index=False)
    f.loc[mask].to_csv(d/"evaluation.csv",index=False)
    row.update(metrics,status="completed",stage="completed",metric_audit_passed=True,evaluation_count=21)
    save(d/"result.json",row); save(d/"status.json",row)


def run_tests(out):
    from single_2025_tests import suite
    from basis_sensitivity_2025_tests import VariableGridTests
    counts={}
    for name,tests in [("existing_tests",suite()),("variable_grid_tests",unittest.defaultTestLoader.loadTestsFromTestCase(VariableGridTests))]:
        stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
        (out/(name+".txt")).write_text(stream.getvalue(),encoding="utf-8")
        counts[name]=dict(passed=result.wasSuccessful(),tests=result.testsRun)
        save(out/"tests.json",counts)
        print(name,counts[name],flush=True)
        if not result.wasSuccessful():raise RuntimeError(stream.getvalue())


def plots_and_report(out,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df=pd.DataFrame(rows).sort_values("M")
    df.to_csv(out/"metrics.csv",index=False)
    failures=df[df.status!="completed"]
    failures.to_csv(out/"failures.csv",index=False)
    good=df[df.status=="completed"]
    plots=[("MAE","MAE (original nins units)",False),("RMSE","RMSE (original nins units)",False),
           ("condition_number","Condition number of observed covariance",True),
           ("noise_variance_standardized","Measurement variance (standardized)",False),
           ("selected_iterations","Iterations of selected restart",False),("fit_seconds","Fit + numerical monitoring (seconds)",False)]
    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for ax,(col,label,log) in zip(axes.flat,plots):
        ax.plot(good.M,good[col],"o-",markersize=4)
        if log:ax.set_yscale("log")
        ax.set(xlabel="M = k squared (grid resolution also changes)",ylabel=label)
        ax.grid(alpha=.25)
        for m in failures.M:ax.axvline(m,color="red",linestyle=":",alpha=.6)
    fig.suptitle("SR-Kriging 2025 | N=88, 21 fixed evaluation sites | failures marked red")
    fig.tight_layout();fig.savefig(out/"performance_stability.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(good.M,good.positive_count,"o-",label="Strictly positive stored variances")
    ax.plot(good.M,good.zero_count,"s-",label="Exactly zero stored variances")
    ax.set(xlabel="M = k squared",ylabel="Number of basis variances",title="Stored floating-point values; no pruning threshold or rank claim")
    for m in failures.M:ax.axvline(m,color="red",linestyle=":")
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/"variance_counts.png",dpi=180);plt.close(fig)
    columns=["M","MAE","RMSE","converged_starts","selected_iterations","condition_number","noise_variance_standardized","positive_count","zero_count","status"]
    table="|"+"|".join(columns)+"|\n|"+"|".join(["---"]*len(columns))+"|\n"
    for _,r in df.iterrows():
        table+="|"+"|".join(f"{r[c]:.6g}" if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+"|\n"
    best=good.loc[good.MAE.idxmin()]
    report=f"""# 2025 SR-Kriging 기저·격자 민감도 실험

총 {len(df)}개 설정 중 {len(good)}개 완료, {len(failures)}개 실패. 모든 평가는 같은 21개 위치를 사용했다.
M=256은 기존 기준 실행과 q, 측정오차 분산, likelihood, 전체 재시작 trace, 예측값, 변환값이 정확히 일치했다.

## 구현과 실험 통제

- N=88, 원래 27개 외부 위치 중 convex hull 내부 21개 평가. 예측은 동일한 사각형 내부 23개에 계산하며 나머지 4개는 정의하지 않는다.
- sr.state, sr.update, sr.predict_sr, sr.design의 AST가 변경 전 백업과 동일하다. Eq.16 기저 갱신 후 새 기저 분산으로 Eq.20 측정오차 분산을 갱신한다. 최종 분산은 고정 모수 조건부 분산이며 Gibbs 모수 불확실성이 아니다.
- sr.grid_basis는 k=1을 허용하고 observation_operator는 B=[1], Psi=ones(N,1)로 모든 위치에 상수 공간장을 연결한다. 이는 application-specific choice이며 1×1에서 bilinear 보간을 실행하지 않는다. k>=2의 계산은 유지했다.
- fit_sr의 변경은 전체 재시작 실패 시 기존 trace/starts를 예외에 첨부하는 기록 기능뿐이다. 원본은 basis_sensitivity_source_backup/sr.py와 실행 폴더 code에 보존했다.
- k×k 전체 DCT를 사용한다. 기저 수와 격자 해상도가 함께 변하며 고정 16×16 기저의 일부만 선택한 실험이 아니다.
- 시작 seed={SEED}, fit seed={FIT_SEED}; 모든 k에 동일한 fit seed 및 child_seed(seed,8101,restart). q 초기값은 기존 base × Uniform(0.5,1.5), 이어 log-uniform noise(0.01,0.5)를 뽑는다. M개 q를 먼저 뽑으므로 M 사이 noise 초기값 자체는 달라질 수 있다. 각 배열을 initial_values.npz에 저장했다.
- 8 starts, maxiter=10000, |ΔLL|<1e-8 유지. 학습 likelihood로 재시작을 선택하며 외부 레이블은 모든 M의 학습·예측 완료 후 평가에만 사용했다. 기존 loader의 외부값 유한성 사전 점검은 그대로다.

## 결과

{table}

모든 수치와 NLPD, 관측분산 기반 95% coverage, 최종 ΔLL, 시간, 최소 양수 q 및 합계는 metrics.csv에 있다.

## 연구 질문에 대한 답

이 스냅샷에서 MAE 범위는 {good.MAE.min():.6f}–{good.MAE.max():.6f}, RMSE 범위는 {good.RMSE.min():.6f}–{good.RMSE.max():.6f}이다. 관측된 최소 MAE는 M={int(best.M)}의 {best.MAE:.6f}이다. 이 값은 외부 평가 결과의 사후 기술이며 새 M 선택 규칙이나 최적 하이퍼파라미터로 사용하지 않았다.

조건수 범위는 {good.condition_number.min():.6g}–{good.condition_number.max():.6g}이다. 증가 방향의 단조성은 {bool(np.all(np.diff(good.condition_number)>=0))}이다. 수렴·독립검증·오류를 함께 해석해야 하며 조건수 하나로 안정성을 판정하지 않는다.

정확한 0 개수의 M에 대한 단조 증가 여부는 {bool(np.all(np.diff(good.zero_count)>=0))}이다. 값이 0으로 저장된 현상은 통계적 기저 선택을 뜻하지 않는다. numerical_diagnostics.json은 모든 재시작의 양수→0 변화와 그중 계산상 subnormal 최솟값 미만 곱을 따로 기록한다. b=0의 경우 정확한 상쇄와 선행 underflow를 구별할 수 없다. 양수 subnormal 상태의 모든 반올림 사건을 포괄적으로 계수한 것은 아니다.

M=256이 항상 더 우수하다는 증거는 이 실험에서 제시할 수 없다. 결과표의 오차 비교는 한 스냅샷·한 학습망·21개 외부 위치에만 해당한다. M과 격자 해상도의 효과를 분리하지 못하며, 정확한 0 개수로 공분산 유효 rank를 확정하지 않는다.

## 검증과 재실행

기존 수학 테스트와 추가 가변 격자 테스트는 tests.json 및 두 txt 로그 참조. 각 M의 prediction_audit.json은 독립 코사인 기저와 직접 공분산 계산을 이용하며 기존 rtol=atol=1e-7을 유지한다. 저장된 예측으로 다시 계산한 MAE/RMSE/NLPD/coverage 검증도 수행했다. clipping, noise floor, pruning, 정규화 항을 추가하지 않았다.

실행: `python basis_sensitivity_2025.py` (이 저장소 루트). 결과는 새 UTC 이름 폴더에 생성된다. 기존 기준 결과가 없거나 정확 비교가 실패하면 이후 M 실험을 중단한다. 한 M의 학습/검증 실패는 failure.json과 failures.csv에 남기고 다른 M은 계속한다.

함수 연결: sr.grid_basis(k,k) → B(M,M), sr.observation_operator(x(N,2),k,k) → Psi(N,M), sr.design → A(N,M), sr.state(A,z(N,),q(M,),noise) → C(N,N) 및 likelihood, sr.fit_sr → 모수와 trace, sr.predict_sr → 평균/잠재분산/관측분산(T,), score_saved → 21개 위치 지표. 새로운 runner는 이 함수를 호출하며 추정식을 재구현하지 않는다.

학습 시간에는 양수→0 진단용 추가 행렬 계산이 포함되며 기기 부하에 영향을 받는다. 전체 시간은 completion.json 참조. 실패 표: failures.csv (없어도 빈 표 저장). 시각화: performance_stability.png, variance_counts.png.
"""
    (out/"REPORT_KO.md").write_text(report,encoding="utf-8")


def main():
    out=ROOT/"results"/("basis_sensitivity_2025_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True,exist_ok=False)
    print("OUTPUT",out,flush=True)
    begun=time.perf_counter();rows=[]
    try:
        code=out/"code";code.mkdir()
        names=["sr.py","data.py","audit.py","config.json","single_2025.py","single_2025_tests.py","tests.py","basis_sensitivity_2025.py","basis_sensitivity_2025_tests.py"]
        for name in names:shutil.copy2(ROOT/name,code/name)
        shutil.copy2(ROOT/"basis_sensitivity_source_backup/sr.py",code/"sr_before.py")
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n,ast.FunctionDef)}
        before,after=functions(code/"sr_before.py"),functions(code/"sr.py")
        unchanged={name:before[name]==after[name] for name in ["state","update","predict_sr","design"]}
        assert all(unchanged.values())
        save(out/"manifest.json",dict(settings=SETTINGS,ks=KS,seed=SEED,fit_seed=FIT_SEED,
             reference=str(REFERENCE),unchanged_math_ast=unchanged,hashes={n:sha(ROOT/n) for n in names},
             python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
             numerical_threads=1,scope="fixed N=88; grid and full basis vary together; no model selection"))
        with threadpool_limits(limits=1):
            run_tests(out)
            train,target=load_and_check(out)
            x,z,tr=training_arrays(train);save(out/"transform.json",tr.manifest())
            xx=tr.coordinates(target[["coord1","coord2"]].to_numpy())
            hull=Delaunay(x).find_simplex(xx,tol=1e-12)>=0
            defined=((xx>=0)&(xx<=1)).all(axis=1)
            assert hull.sum()==21 and defined.sum()==23 and not np.any(hull&~defined)
            np.savez(out/"fixed_masks.npz",hull=hull,defined=defined,target_ids=target.pv_id.to_numpy())
            row,frame=run_setting(16,x,z,xx,target,hull,defined,tr,out);rows.append(row)
            if frame is None:raise RuntimeError("M256 baseline fit/audit failed")
            verify_baseline(out,frame,tr)
            print("M256 BASELINE EXACT MATCH",flush=True)
            for k in KS[:-1]:
                row,_=run_setting(k,x,z,xx,target,hull,defined,tr,out);rows.append(row)
                print("M",k*k,row["status"],"converged",row.get("converged_starts"),"seconds",round(row["fit_seconds"],2),flush=True)
            # No target responses are used for fitting, initialization, or start selection.
            truth=pd.read_parquet(SOURCE/"heldout.parquet")
            for row in rows:
                if row["status"]=="predicted":
                    try:score_saved(row,out,truth)
                    except Exception as exc:
                        row.update(status="failed",stage="evaluation",error=repr(exc))
                        save(out/f"M{row['M']:03d}"/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
                        save(out/f"M{row['M']:03d}"/"status.json",row)
            assert len(rows)==16 and {r["M"] for r in rows}=={k*k for k in KS}
            plots_and_report(out,rows)
        assert all(sha(ROOT/n)==h for n,h in read(out/"manifest.json")["hashes"].items())
        save(out/"completion.json",dict(settings_recorded=16,completed=sum(r["status"]=="completed" for r in rows),
             failed=sum(r["status"]!="completed" for r in rows),wall_seconds=time.perf_counter()-begun,source_unchanged_during_run=True))
        print(pd.DataFrame(rows).sort_values("M")[["M","status","MAE","RMSE","condition_number","zero_count"]].to_string(index=False),flush=True)
    except Exception as exc:
        save(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc(),partial_rows=rows))
        raise

if __name__=="__main__":
    main()
