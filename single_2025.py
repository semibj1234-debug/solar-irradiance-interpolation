from pathlib import Path
CORE_PROJECT = Path('E:/04_Project/01_OIBC 2025/code_paper/refactored')

import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(CORE_PROJECT))

from datetime import datetime, timezone
import argparse
import io
import json
import platform
import time
import traceback
import unittest

import numpy as np
import pandas as pd
import scipy
from scipy.spatial import Delaunay
from threadpoolctl import threadpool_limits

from shared_partition_gp.data import TrainingTransform, child_seed
from sr_kriging_benchmark import sr
from sr_kriging_benchmark.audit import audit_fit
from sr_kriging_benchmark.data import ROOT as CORE_ROOT, read, sha
ROOT = Path(__file__).resolve().parent

# Application choices, frozen before looking at external prediction errors.
SEED = 20261011
SNAPSHOT = "2024-08-01 12:00:00+09:00"
SOURCE = CORE_PROJECT / "shared_partition_gp/results/poster_v1/data"
SETTINGS = dict(read(CORE_ROOT / "config.json")["sr"])
PAPER = Path("E:/04_Project/01_OIBC 2025/선행 연구 조사/miao-wang-2024-interpolation-of-sparse-measurements-and-quantification-of-measurement-error-using-enhanced-kriging-with.pdf")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def training_arrays(train):
    """Train-only transform. Input DataFrame (n, >=3); output x(n,2), z(n,), transform.
    Global centering/scaling is an application choice; not a fitted spatial trend.
    No evaluation frame or evaluation response is accepted.
    """
    xraw = train[["coord1", "coord2"]].to_numpy(float)
    y = train.nins.to_numpy(float)
    transform = TrainingTransform.fit(xraw, y)
    return transform.coordinates(xraw), transform.response(y), transform


def evaluate(truth, prediction, mask):
    """Input truth/prediction (k,), mask(k,); output hull-only MAE/RMSE and count."""
    error = np.asarray(truth)[mask] - np.asarray(prediction)[mask]
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Empty or nonfinite evaluation")
    return dict(count=len(error), MAE=float(np.mean(np.abs(error))),
                RMSE=float(np.sqrt(np.mean(error**2))))


def load_and_check(out):
    """Read fixed snapshot only, verify (88,2)/(27,2), aggregation and no overlap.
    Returns training frame and target coordinates only; labels cannot enter fitting.
    """
    meta = read(SOURCE / "complete.json")
    assert meta["timestamp"] == SNAPSHOT and meta["aggregation"] == "mean"
    train = pd.read_parquet(SOURCE / "training.parquet")
    target = pd.read_parquet(SOURCE / "heldout.parquet", columns=["pv_id", "coord1", "coord2"])
    raw = pd.read_parquet(SOURCE / "raw_snapshot.parquet")
    assert len(train) == 88 and len(target) == 27 and len(raw) == 183
    for f in (train, target):
        assert not f.duplicated(["coord1", "coord2"]).any()
        assert not f.pv_id.duplicated().any()
        assert np.isfinite(f[["coord1", "coord2"]].to_numpy()).all()
    assert np.isfinite(train.nins).all() and np.isfinite(raw.nins).all()
    assert not raw.duplicated(["time", "pv_id"]).any()
    assert (pd.to_datetime(raw.time) == pd.Timestamp(SNAPSHOT)).all()
    assert train.merge(target, on=["coord1", "coord2"]).empty
    raw_ids = set(raw.pv_id)
    assert not raw_ids.intersection(target.pv_id)
    reconstructed = raw.groupby("location_id").nins.mean().reindex(train.location_id).to_numpy()
    np.testing.assert_allclose(reconstructed, train.nins, rtol=1e-13, atol=1e-13)
    report = dict(snapshot=SNAPSHOT, training_count=len(train), target_count=len(target),
                  raw_training_count=len(raw), aggregation="coordinate mean",
                  missing_or_nonfinite=0, duplicate_training_coordinates=0,
                  duplicate_target_coordinates=0, overlapping_coordinates=0,
                  training_coordinate_min=train[["coord1","coord2"]].min().tolist(),
                  training_coordinate_max=train[["coord1","coord2"]].max().tolist(),
                  training_nins_min=float(train.nins.min()), training_nins_max=float(train.nins.max()),
                  coordinate_units="not established from source metadata; not labeled km",
                  response_units="original nins units; physical unit not established",
                  source_count_distribution={str(k):int(v) for k,v in train.source_count.value_counts().items()},
                  aggregation_recomputed=True,
                  hashes={name:sha(SOURCE/name) for name in
                          ["training.parquet","heldout.parquet","raw_snapshot.parquet","complete.json"]})
    save(out / "data_audit.json", report)
    train.to_csv(out / "training_used.csv", index=False)
    target.to_csv(out / "target_coordinates.csv", index=False)
    return train, target


def smoke(x, z, out):
    """20 actual-data update pairs, all 88 sites/256 bases. Diagnostic, not convergence.
    Eq.16 then Eq.20 via unchanged sr.update; no model from this test is selected.
    """
    A = sr.design(x, SETTINGS)
    q = np.full(A.shape[1], 1 / np.mean(np.sum(A*A, axis=1)))
    noise = 0.1
    rows = []
    for it in range(21):
        s = sr.state(A,z,q,noise)
        rows.append(dict(iteration=it,ll=s["ll"],noise_variance=noise))
        if it < 20:
            q, noise = sr.update(A,z,q,noise)
    fit = dict(weights=q.tolist(),noise_variance=noise,
               log_marginal_likelihood=s["ll"],condition_number=float(np.linalg.cond(s["C"])))
    p = sr.predict_sr(fit,x,z,x,SETTINGS)
    audit = audit_fit("SR",fit,x,z,x,p,SETTINGS)
    pd.DataFrame(rows).to_csv(out / "smoke_trace.csv",index=False)
    save(out / "smoke.json",dict(passed=True,iterations=20,convergence_claim=False,audit=audit))


def run(out):
    """Tests -> data validation -> actual-data smoke -> single full 88-site fit."""
    from single_2025_tests import suite
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite())
    (out/"tests.txt").write_text(stream.getvalue(),encoding="utf-8")
    save(out/"tests.json",dict(passed=result.wasSuccessful(),tests=result.testsRun,
                               failures=len(result.failures),errors=len(result.errors)))
    if not result.wasSuccessful():
        raise RuntimeError("Mathematical tests failed")
    print(stream.getvalue(),flush=True)
    train, target = load_and_check(out)
    x,z,tr = training_arrays(train)
    save(out/"transform.json",tr.manifest())
    xx = tr.coordinates(target[["coord1","coord2"]].to_numpy())
    hull = Delaunay(x).find_simplex(xx,tol=1e-12)>=0
    defined = ((xx>=0)&(xx<=1)).all(axis=1)
    assert not np.any(hull & ~defined)
    smoke(x,z,out)
    print("SMOKE PASSED; full fit 88 sites, 256 bases, 8 starts",flush=True)
    seed = child_seed(SEED,88,2025)
    started = time.perf_counter()
    fit = sr.fit_sr(x,z,SETTINGS,seed)
    fit_seconds = time.perf_counter()-started
    trace = pd.DataFrame(fit.pop("trace"))
    trace.to_csv(out/"likelihood_trace.csv",index=False)
    save(out/"fit.json",dict(fit,seed=seed))
    pd.DataFrame(fit["starts"]).to_csv(out/"starts.csv",index=False)
    q = np.asarray(fit["weights"])
    np.save(out/"basis_variances.npy",q)
    basis = pd.DataFrame(dict(index=np.arange(len(q)),frequency_x=np.arange(len(q))//16,
                             frequency_y=np.arange(len(q))%16,
                             variance_standardized=q,variance_original=q*tr.response_sd**2))
    basis.to_csv(out/"basis_variances.csv",index=False)
    # No target labels have been loaded before this point.
    pred = sr.predict_sr(fit,x,z,xx[defined],SETTINGS)
    audit = audit_fit("SR",fit,x,z,xx[defined],pred,SETTINGS)
    save(out/"prediction_audit.json",audit)
    inv = tr.inverse_prediction(pred)
    frame = target.copy()
    frame["in_training_hull"] = hull
    frame["prediction_defined"] = defined
    for name,key in [("prediction_mean","mean"),("latent_variance","latent_variance"),
                     ("observation_variance","observation_variance")]:
        frame[name] = np.nan
        frame.loc[defined,name] = inv[key]
    truth = pd.read_parquet(SOURCE/"heldout.parquet")
    np.testing.assert_array_equal(truth.pv_id,target.pv_id)
    assert np.isfinite(truth.nins).all()
    frame["nins"] = truth.nins.to_numpy()
    frame["absolute_error"] = np.where(hull,abs(frame.nins-frame.prediction_mean),np.nan)
    frame["latent_sd"] = np.sqrt(frame.latent_variance)
    frame["observation_sd"] = np.sqrt(frame.observation_variance)
    frame.to_csv(out/"predictions_all.csv",index=False)
    frame.loc[hull].to_csv(out/"evaluation.csv",index=False)
    score = evaluate(frame.nins,frame.prediction_mean,hull)
    # Independent scalar recomputation of evaluation arithmetic.
    errors = [abs(float(a)-float(b)) for a,b in zip(frame.loc[hull,"nins"],frame.loc[hull,"prediction_mean"])]
    np.testing.assert_allclose(score["MAE"],sum(errors)/len(errors),rtol=1e-13)
    selected = trace[trace.restart == fit["selected_start"]]
    record = dict(status="completed",training_count=88,target_count=27,basis_count=256,
                  grid=[16,16],evaluation_count=int(hull.sum()),excluded_count=int((~hull).sum()),
                  metrics=score,noise_variance_standardized=fit["noise_variance"],
                  noise_variance_original=fit["noise_variance"]*tr.response_sd**2,
                  noise_sd_original=np.sqrt(fit["noise_variance"])*tr.response_sd,
                  log_likelihood_standardized=fit["log_marginal_likelihood"],
                  log_likelihood_original=fit["log_marginal_likelihood"]-88*np.log(tr.response_sd),
                  converged=fit["converged"],selected_start=fit["selected_start"],
                  selected_iterations=int(selected.iteration.iloc[-1]),
                  last_absolute_ll_change=float(abs(selected.ll.diff().iloc[-1])),
                  converged_starts=sum(s["converged"] and s["error"] is None for s in fit["starts"]),
                  fit_seconds=fit_seconds,condition_number=fit["condition_number"],
                  machine_zero_weights=int((q==0).sum()),minimum_variance=float(q.min()),
                  positive_minimum_variance=float(q[q>0].min()),maximum_variance=float(q.max()),
                  variance_quantiles=np.quantile(q,[0,.25,.5,.75,1]).tolist(),
                  jitter=0.0,pruned_basis_count=0,audit=audit,
                  source_code_unchanged=all(sha(CORE_ROOT/n)==h for n,h in read(out/"manifest.json")["core_hashes"].items()))
    assert record["source_code_unchanged"]
    save(out/"result.json",record)
    print(json.dumps(record,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args()
    out=args.output or ROOT/"results"/("single_2025_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(settings=SETTINGS,seed=SEED,paper_path=str(PAPER),paper_sha256=sha(PAPER),
                  core_hashes={n:sha(CORE_ROOT/n) for n in ["sr.py","data.py","audit.py","tests.py","config.json","README.md","run.py"]},
                  new_code_hashes={n:sha(ROOT/n) for n in ["single_2025.py","single_2025_tests.py"]},
                  versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__),
                  scope="one full-training snapshot; no CV, adaptive M or model comparisons",
                  paper_verification="2025 PDF pp2-5, equations 1-3,8-12,16,20; pp4-5 rendered and visually checked")
    save(out/"manifest.json",manifest)
    print("OUTPUT",out,flush=True)
    try:
        with threadpool_limits(limits=1):
            run(out)
    except Exception as exc:
        save(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=="__main__":
    main()

