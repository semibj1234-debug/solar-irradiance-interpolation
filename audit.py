"""Independent dense prediction and metric audits of saved benchmark fits."""
import numpy as np
from scipy.spatial.distance import cdist
from numpy.testing import assert_allclose
from sr import observation_operator


def high_precision_sr(A,H,q,noise,z):
    """Independent 70-decimal covariance algebra; no tolerance relaxation.

    Needed when roundoff in forming B diag(q) B' can swamp a very small noise
    variance. Basis/operators supplied here are independently computed above.
    """
    import mpmath as mp
    with mp.workdps(70):
        am,hm=mp.matrix(A.tolist()),mp.matrix(H.tolist())
        qm=[mp.mpf(float(v)) for v in q]
        aw=mp.matrix([[am[i,j]*qm[j] for j in range(len(q))] for i in range(len(A))])
        hw=mp.matrix([[hm[i,j]*qm[j] for j in range(len(q))] for i in range(len(H))])
        C=aw*am.T+mp.mpf(noise)*mp.eye(len(z));y=mp.matrix(z.tolist())
        inv=C**-1;cross=hw*am.T;mu=cross*inv*y
        latent=hw*hm.T-cross*inv*cross.T
        ll=-mp.mpf('.5')*(mp.log(mp.det(C))+(y.T*inv*y)[0]+len(z)*mp.log(2*mp.pi))
        return float(ll),np.array(list(mu),float),np.array([latent[i,i] for i in range(len(H))],float)


def audit_fit(model,fit,x,z,target,prediction,settings):
    if model=="RBF":
        signal,noise,ell=np.exp(fit["log_parameters"])
        C=signal*np.exp(-cdist(x,x,"sqeuclidean")/(2*ell**2))+(noise+settings["jitter"])*np.eye(len(x))
        cross=signal*np.exp(-cdist(target,x,"sqeuclidean")/(2*ell**2))
        diagonal=np.full(len(target),signal)
    elif model=="SR":
        nx,ny=settings["grid_nx"],settings["grid_ny"]
        # Explicit DCT cosine formula, independent of scipy.fft used by model.
        def cosine(n):
            value=np.cos(np.pi*(np.arange(n)[:,None]+.5)*np.arange(n)[None,:]/n)*np.sqrt(2/n)
            value[:,0]=np.sqrt(1/n);return value
        basis_nx=settings.get("basis_nx",nx);basis_ny=settings.get("basis_ny",ny)
        full=np.kron(cosine(nx),cosine(ny))
        indices=[i*ny+j for i in range(basis_nx) for j in range(basis_ny)]
        B=full[:,indices];q=np.asarray(fit["weights"])
        cf=(B*q)@B.T;psi=observation_operator(x,nx,ny);psit=observation_operator(target,nx,ny)
        noise=fit["noise_variance"];C=psi@cf@psi.T+noise*np.eye(len(x))
        cross=psit@cf@psi.T;diagonal=np.diag(psit@cf@psit.T)
        assert_allclose(cf,cf.T,atol=1e-12)
        if np.linalg.eigvalsh(cf).min() < -1e-9*max(1,q.max()):raise AssertionError("Non-PSD SR covariance")
    else:raise ValueError(model)
    logdet=np.linalg.slogdet(C)
    if logdet[0]!=1:raise AssertionError("Nonpositive covariance determinant")
    ll=-.5*(logdet[1]+z@np.linalg.solve(C,z)+len(z)*np.log(2*np.pi))
    mean=cross@np.linalg.solve(C,z)
    latent=diagonal-np.einsum("ij,ji->i",cross,np.linalg.solve(C,cross.T))
    double_difference=float(abs(ll-fit["log_marginal_likelihood"]))
    precision=53
    if model=="SR" and fit["condition_number"]>1e10:
        ll,mean,latent=high_precision_sr(psi@B,psit@B,q,noise,z)
        precision="70 decimal digits"
    assert_allclose(ll,fit["log_marginal_likelihood"],rtol=1e-7,atol=1e-7)
    assert_allclose(prediction["mean"],mean,rtol=1e-7,atol=1e-7)
    assert_allclose(prediction["latent_variance"],latent,rtol=1e-7,atol=1e-7)
    assert_allclose(prediction["observation_variance"],latent+noise,rtol=1e-7,atol=1e-7)
    return {"passed":True,"reference_precision":precision,"double_precision_likelihood_difference":double_difference,
            "likelihood_difference":float(abs(ll-fit["log_marginal_likelihood"])),
            "prediction_max_difference":float(np.max(np.abs(mean-prediction["mean"]))),
            "variance_max_difference":float(np.max(np.abs(latent-prediction["latent_variance"])))}


def audit_metrics(y,mean,variance,mask,result):
    y,mean=np.asarray(y)[mask],np.asarray(mean)[mask]
    if not len(y):return
    error=y-mean
    assert_allclose(result["MAE"],np.sum(abs(error))/len(y),rtol=1e-13)
    assert_allclose(result["RMSE"],np.linalg.norm(error)/np.sqrt(len(y)),rtol=1e-13)
    if variance is not None:
        from scipy.stats import norm
        nlpd=-norm.logpdf(y,loc=mean,scale=np.sqrt(np.asarray(variance)[mask])).mean()
        assert_allclose(result["NLPD"],nlpd,rtol=1e-12)
