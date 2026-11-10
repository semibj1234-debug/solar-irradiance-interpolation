"""Enhanced SR-kriging equations with explicitly documented 2D operators.

Miao & Wang, DOI 10.1061/AJRUA6.RUENG-1403. Eq (16) updates basis
precision; Eq (20) updates iid measurement variance using the NEW precision.
We compute the same equations in observation space (Woodbury identities),
retaining every grid basis column. No test responses are accepted by this API.
"""
import time
import numpy as np
from scipy.fft import dct
from scipy.linalg import cholesky, cho_solve
from data import child_seed


def grid_basis(nx,ny,active_nx=None,active_ny=None):
    if min(nx,ny)<1:raise ValueError("At least one node per axis required")
    active_nx=nx if active_nx is None else int(active_nx)
    active_ny=ny if active_ny is None else int(active_ny)
    if not (1<=active_nx<=nx and 1<=active_ny<=ny):
        raise ValueError("Active DCT block must fit inside the latent grid")
    # B[j,r] is the r-th orthonormal cosine evaluated at node j. Columns are
    # ordered as (frequency_x, frequency_y), so low-frequency blocks nest.
    bx=dct(np.eye(nx),type=2,norm="ortho",axis=0).T
    by=dct(np.eye(ny),type=2,norm="ortho",axis=0).T
    full=np.kron(bx,by)
    indices=[i*ny+j for i in range(active_nx) for j in range(active_ny)]
    # Preserve the C-contiguous layout of the original full basis. In the
    # ill-conditioned high-M SR updates, a different BLAS traversal order can
    # move the iterative endpoint even when entries are mathematically equal.
    return np.ascontiguousarray(full[:,indices])


def observation_operator(x,nx,ny):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all():raise ValueError("Invalid 2D coordinates")
    if (x<0).any() or (x>1).any():raise ValueError("Location outside the train-only grid rectangle")
    # Application-specific k=1 limit: one constant latent value over the
    # training rectangle. B=[1], Psi=ones(n,1); observation noise stays iid.
    if nx == ny == 1:
        return np.ones((len(x),1))
    if min(nx,ny)<2:
        raise ValueError("Mixed singleton axes are not supported")
    u=x*np.array([nx-1,ny-1])
    lo=np.floor(u).astype(int)
    # At the endpoint use the last cell at interpolation fraction one.
    lo=np.minimum(lo,[nx-2,ny-2]);f=u-lo
    op=np.zeros((len(x),nx*ny))
    for dx in (0,1):
        for dy in (0,1):
            weight=(f[:,0] if dx else 1-f[:,0])*(f[:,1] if dy else 1-f[:,1])
            op[np.arange(len(x)),(lo[:,0]+dx)*ny+lo[:,1]+dy]=weight
    return op


def design(x,settings):
    nx,ny=settings["grid_nx"],settings["grid_ny"]
    B=grid_basis(nx,ny,settings.get("basis_nx",nx),settings.get("basis_ny",ny))
    psi=observation_operator(x,nx,ny)
    return psi@B


def state(A,y,q,noise):
    """Exact observation-space likelihood and coefficient posterior moments."""
    A,y,q=np.asarray(A),np.asarray(y),np.asarray(q)
    if not np.isfinite(q).all() or (q<0).any() or not np.isfinite(noise) or noise<=0:
        raise FloatingPointError("Invalid coefficient or noise variance; no clipping")
    C=(A*q)@A.T+noise*np.eye(len(y))
    L=cholesky(C,lower=True)
    cy=cho_solve((L,True),y);ca=cho_solve((L,True),A)
    h=np.einsum("ij,ij->j",A,ca)
    b=A.T@cy;mu=q*b;gamma=q*h
    ll=-.5*(y@cy+2*np.log(np.diag(L)).sum()+len(y)*np.log(2*np.pi))
    return {"C":C,"L":L,"mu":mu,"gamma":gamma,"h":h,"b":b,"ll":float(ll)}


def update(A,y,q,noise):
    old=state(A,y,q,noise)
    # (16): alpha_new = gamma / mu^2, hence q_new = q*b^2/h.
    # Avoid 1-alpha*Sigma subtraction and 0/0 for vanishing basis weights.
    if (old["h"]<=0).any():raise FloatingPointError("Unidentified zero design column")
    with np.errstate(under="ignore"):
        newq=q*(old["b"]**2/old["h"])
    # Paper p5: update noise using the most recently updated alpha values.
    interim=state(A,y,newq,noise)
    residual=y-A@interim["mu"]
    # tr(A Sigma A') = noise * sum(gamma), exactly, without forming Sigma.
    newnoise=float((residual@residual+noise*interim["gamma"].sum())/len(y))
    return newq,newnoise


def fit_sr(x,z,settings,seed):
    started=time.perf_counter();A=design(x,settings)
    base=1/np.mean(np.sum(A*A,axis=1))
    records=[];solutions=[];traces=[]
    for restart in range(settings["starts"]):
        begin=time.perf_counter();rng=np.random.default_rng(child_seed(seed,8101,restart))
        q=base*rng.uniform(*settings["initial_weight_multiplier"],size=A.shape[1])
        noise=float(np.exp(rng.uniform(*np.log(settings["initial_noise_variance"]))))
        previous=state(A,z,q,noise)["ll"];converged=False;error=None
        trace=[{"restart":restart,"iteration":0,"ll":previous,"noise":noise,"zero_weights":0}]
        try:
            for iteration in range(1,settings["maxiter"]+1):
                q,noise=update(A,z,q,noise);s=state(A,z,q,noise)
                trace.append({"restart":restart,"iteration":iteration,"ll":s["ll"],"noise":noise,
                              "zero_weights":int(np.sum(q==0)),"min_weight":float(q.min()),"max_weight":float(q.max())})
                difference=s["ll"]-previous
                if abs(difference)<settings["likelihood_absolute_tolerance"]:
                    converged=True;break
                previous=s["ll"]
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
        ll=trace[-1]["ll"]
        records.append({"restart":restart,"converged":converged,"iterations":len(trace)-1,
                        "log_marginal_likelihood":ll,"error":error,"noise_variance":noise,
                        "zero_weights":int(np.sum(q==0)),"maximum_ll_decrease":float(max(0,-np.diff([t["ll"] for t in trace]).min())) if len(trace)>1 else 0,
                        "runtime_seconds":time.perf_counter()-begin})
        traces.extend(trace);solutions.append((q,noise))
    valid=[r for r in records if r["converged"] and r["error"] is None]
    if not valid:
        failure = RuntimeError(f"No converged SR restart: {records}")
        failure.starts, failure.trace = records, traces
        raise failure
    best=max(valid,key=lambda r:r["log_marginal_likelihood"]);q,noise=solutions[best["restart"]]
    final=state(A,z,q,noise)
    # Full objective scores for active and tiny weights; zeros are machine limits,
    # logged explicitly. No column has been removed from A or from the grid.
    alpha_score=.5*(1-(1-final["gamma"]+np.divide(final["mu"]**2,q,out=np.zeros_like(q),where=q>0)))
    cy=cho_solve((final["L"],True),z)
    precision=cho_solve((final["L"],True),np.eye(len(z)))
    noise_score=.5*noise*(cy@cy-np.trace(precision))
    return {"model":"SR","weights":q.tolist(),"noise_variance":noise,
            "log_marginal_likelihood":best["log_marginal_likelihood"],"converged":True,
            "selected_start":best["restart"],"starts":records,"trace":traces,
            "condition_number":float(np.linalg.cond(final["C"])),
            "cholesky_min_diagonal":float(np.diag(final["L"]).min()),"jitter":0.0,
            "score_max_abs":float(max(np.max(np.abs(alpha_score)),abs(noise_score))),
            "grid_nx":settings["grid_nx"],"grid_ny":settings["grid_ny"],"basis_count":A.shape[1],
            "explicit_pruned_count":0,"machine_zero_weight_count":int(np.sum(q==0)),
            "dense_basis_bytes":A.shape[1]**2*8,"design_bytes":A.nbytes,
            "runtime_seconds":time.perf_counter()-started}


def predict_sr(fit,x,z,target,settings):
    A=design(x,settings);H=design(target,settings);q=np.asarray(fit["weights"]);noise=fit["noise_variance"]
    s=state(A,z,q,noise);cross=(H*q)@A.T
    mean=cross@cho_solve((s["L"],True),z)
    latent=np.sum(H*H*q,axis=1)-np.einsum("ij,ji->i",cross,cho_solve((s["L"],True),cross.T))
    if not np.isfinite(latent).all() or (latent<0).any():raise FloatingPointError("Negative/nonfinite SR latent variance; no clipping")
    return {"mean":mean,"latent_variance":latent,"observation_variance":latent+noise}
