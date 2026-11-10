"""Variable-grid tests against direct cosine, node weights and Gaussian algebra."""
import sys
sys.dont_write_bytecode=True
import inspect
import unittest
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
import sr
from basis_sensitivity_2025 import settings_for,initial_values,monitored_fit,FIT_SEED,run_setting


def cosine(n):
    j=np.arange(n)[:,None];r=np.arange(n)[None,:]
    B=np.cos(np.pi*(j+.5)*r/n)*np.sqrt(2/n)
    B[:,0]=1/np.sqrt(n)
    return B


def direct_operator(x,k):
    if k==1:return np.ones((len(x),1))
    nodes=np.linspace(0,1,k)
    # Tensor hat functions provide an independent expression, including endpoints.
    weights=np.maximum(0,1-abs(x[:,:,None]-nodes)*(k-1))
    return (weights[:,0,:,None]*weights[:,1,None,:]).reshape(len(x),k*k)


class VariableGridTests(unittest.TestCase):
    def test_all_grid_shapes_orthogonality_and_indices(self):
        rng=np.random.default_rng(444)
        for k in range(1,17):
            with self.subTest(k=k):
                B=sr.grid_basis(k,k);b=cosine(k)
                self.assertEqual(B.shape,(k*k,k*k))
                assert_allclose(B.T@B,np.eye(k*k),atol=1e-14)
                # Explicit [ix,iy,fx,fy] tensor verifies C-order row/column indexing.
                reference=np.einsum('ac,bd->abcd',b,b).reshape(k*k,k*k)
                assert_allclose(B,reference,atol=1e-14)
                field=rng.normal(size=(k,k))
                assert_allclose((B@field.ravel()).reshape(k,k),b@field@b.T,atol=1e-13)

    def test_all_observation_operators(self):
        x=np.vstack([[[0,0],[0,1],[1,0],[1,1]],np.random.default_rng(413).uniform(size=(7,2))])
        for k in range(1,17):
            with self.subTest(k=k):
                P=sr.observation_operator(x,k,k)
                assert_allclose(P,direct_operator(x,k),atol=5e-15)
                assert_allclose(P.sum(axis=1),1,atol=1e-14)
                self.assertTrue((P>=0).all())
                nodes=np.array([(i/(k-1),j/(k-1)) for i in range(k) for j in range(k)]) if k>1 else np.array([[.5,.5]])
                assert_allclose(sr.observation_operator(nodes,k,k),np.eye(k*k),atol=5e-15)
                self.assertEqual(sr.design(x,settings_for(k)).shape,(len(x),k*k))

    def test_covariance_likelihood_prediction_at_multiple_sizes(self):
        rng=np.random.default_rng(65)
        x=rng.uniform(size=(7,2));target=rng.uniform(size=(3,2));y=rng.normal(size=7)
        for k in [1,2,3,5,9,16]:
            with self.subTest(k=k):
                B=np.kron(cosine(k),cosine(k));q=rng.uniform(.1,2,k*k);noise=.3
                cf=(B*q)@B.T
                assert_allclose(cf,cf.T,atol=1e-14)
                self.assertGreaterEqual(np.linalg.eigvalsh(cf).min(),-1e-12)
                P,T=direct_operator(x,k),direct_operator(target,k)
                C=P@cf@P.T+noise*np.eye(7);cross=T@cf@P.T
                expected_mean=cross@np.linalg.solve(C,y)
                expected_var=np.diag(T@cf@T.T-cross@np.linalg.solve(C,cross.T))
                fit=dict(weights=q,noise_variance=noise)
                result=sr.predict_sr(fit,x,y,target,settings_for(k))
                assert_allclose(result['mean'],expected_mean,rtol=1e-11,atol=1e-12)
                assert_allclose(result['latent_variance'],expected_var,rtol=1e-11,atol=1e-12)
                assert_allclose(result['observation_variance'],expected_var+noise,rtol=1e-11,atol=1e-12)
                ll=-.5*(np.linalg.slogdet(C)[1]+y@np.linalg.solve(C,y)+7*np.log(2*np.pi))
                assert_allclose(sr.state(sr.design(x,settings_for(k)),y,q,noise)['ll'],ll,rtol=1e-12)

    def test_constant_case_analytic_prediction_and_domain(self):
        x=np.array([[0,0],[0,1],[1,0],[1,1]],float);z=np.array([-1.,1.,-1.,1.])
        q=.8;noise=.4;targets=np.array([[.1,.2],[.9,.8]])
        result=sr.predict_sr(dict(weights=[q],noise_variance=noise),x,z,targets,settings_for(1))
        assert_allclose(result['mean'],q*z.sum()/(noise+4*q),atol=1e-14)
        assert_allclose(result['latent_variance'],q*noise/(noise+4*q),atol=1e-14)
        with self.assertRaises(ValueError):sr.observation_operator(np.array([[1.1,.5]]),1,1)

    def test_initial_policy_and_monitor_do_not_change_fit(self):
        x=np.array([[0,0],[0,1],[1,0],[1,1]],float);z=np.array([-1.,1.,-1.,1.])
        s=settings_for(1)
        before=sr.fit_sr(x,z,s,FIT_SEED);diagnostics={}
        after=monitored_fit(x,z,s,diagnostics)
        for key in ['weights','noise_variance','log_marginal_likelihood','trace','selected_start']:
            self.assertEqual(before[key],after[key])
        for k in [1,4,16]:
            a,b,seeds=initial_values(x,settings_for(k));a2,b2,seeds2=initial_values(x,settings_for(k))
            assert_array_equal(a,a2);assert_array_equal(b,b2);self.assertEqual(seeds,seeds2)
            self.assertEqual(a.shape,(8,k*k))

    def test_full_failure_retains_restart_records(self):
        x=np.array([[0,0],[0,1],[1,0],[1,1]],float);z=np.array([-1.,1.,-1.,1.])
        s=dict(settings_for(2),maxiter=1)
        with self.assertRaises(RuntimeError) as caught:sr.fit_sr(x,z,s,FIT_SEED)
        self.assertEqual(len(caught.exception.starts),8)
        self.assertEqual(len(caught.exception.trace),16)

    def test_training_response_interface_and_scope(self):
        self.assertEqual(list(inspect.signature(monitored_fit).parameters),['x','z','settings','diagnostics'])
        self.assertNotIn('truth',inspect.getsource(run_setting))
        import basis_sensitivity_2025 as runner
        source=inspect.getsource(runner.main)
        self.assertLess(source.index('for k in KS[:-1]'),source.index('truth=pd.read_parquet'))
        for k in range(1,17):
            s=settings_for(k)
            self.assertEqual((s['grid_nx'],s['grid_ny'],s['basis_count']),(k,k,k*k))
            self.assertEqual(s['starts'],8);self.assertEqual(s['maxiter'],10000)
            self.assertEqual(s['likelihood_absolute_tolerance'],1e-8)

if __name__=='__main__':unittest.main(verbosity=2)
