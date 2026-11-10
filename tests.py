"""Unchanged SR-only direct-matrix tests extracted from the previous project."""
import unittest
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from sr import grid_basis,observation_operator,design,state,update,predict_sr,fit_sr
from audit import high_precision_sr

class Equations(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(391)
        self.x=rng.uniform(.05,.95,(7,2));self.y=rng.normal(size=7)
        self.t=rng.uniform(.1,.9,(3,2));self.cfg={"grid_nx":4,"grid_ny":5}

    def test_basis_and_operator(self):
        B=grid_basis(4,5);assert_allclose(B.T@B,np.eye(20),atol=1e-14)
        nodes=np.array([(x,y) for x in np.linspace(0,1,4) for y in np.linspace(0,1,5)])
        assert_allclose(observation_operator(nodes,4,5),np.eye(20),atol=1e-14)
        psi=observation_operator(self.x,4,5)
        self.assertEqual(psi.shape,(7,20));assert_allclose(psi.sum(axis=1),1)
        self.assertTrue((psi>=0).all());self.assertTrue(((psi>0).sum(axis=1)<=4).all())
        # A bilinear operator must reproduce an arbitrary affine field exactly.
        assert_allclose(psi@(2+3*nodes[:,0]-4*nodes[:,1]),2+3*self.x[:,0]-4*self.x[:,1],atol=1e-14)
        with self.assertRaises(ValueError):observation_operator(np.array([[-.01,.4]]),4,5)

    def test_sr_covariance_psd_and_paper_likelihood(self):
        B=grid_basis(4,5);q=np.linspace(.2,2.,20);noise=.13
        A=design(self.x,self.cfg);cf=(B*q)@B.T;C=(A*q)@A.T+noise*np.eye(7)
        assert_allclose(cf,cf.T,atol=1e-14);self.assertGreater(np.linalg.eigvalsh(cf).min(),0)
        s=state(A,self.y,q,noise)
        M=np.diag(1/q)+A.T@A/noise;m=A.T@self.y/noise
        # Eq 12 is -ln|D| = +sum ln q (D=diag(1/q)).
        ll=-.5*(np.log(q).sum()+7*np.log(noise)+np.linalg.slogdet(M)[1]+self.y@self.y/noise-m@np.linalg.solve(M,m)+7*np.log(2*np.pi))
        assert_allclose(s["C"],C);assert_allclose(s["ll"],ll,atol=1e-11)

    def test_sr_equations_16_20_against_dense_parameter_space(self):
        A=design(self.x,self.cfg);q=np.linspace(.2,2.,20);noise=.13
        covariance=np.linalg.inv(np.diag(1/q)+A.T@A/noise)
        mu=covariance@A.T@self.y/noise
        alpha_new=(1-np.diag(covariance)/q)/mu**2
        qnew=1/alpha_new
        covnew=np.linalg.inv(np.diag(alpha_new)+A.T@A/noise)
        munew=covnew@A.T@self.y/noise
        noise_new=(np.linalg.norm(self.y-A@munew)**2+np.trace(covnew@A.T@A))/len(self.y)
        qfast,nfast=update(A,self.y,q,noise)
        assert_allclose(qfast,qnew,rtol=1e-11,atol=1e-12);assert_allclose(nfast,noise_new,rtol=1e-11)

    def test_sr_prediction_equations_8_9_and_coefficient_posterior(self):
        q=np.linspace(.2,2.,20);noise=.13;A=design(self.x,self.cfg);H=design(self.t,self.cfg)
        p=predict_sr({"weights":q,"noise_variance":noise},self.x,self.y,self.t,self.cfg)
        Sigma=np.linalg.inv(np.diag(1/q)+A.T@A/noise);mu=Sigma@A.T@self.y/noise
        assert_allclose(p["mean"],H@mu,atol=1e-12)
        assert_allclose(p["latent_variance"],np.diag(H@Sigma@H.T),atol=1e-12)
        assert_allclose(p["observation_variance"],p["latent_variance"]+noise,atol=1e-14)

    def test_high_precision_reference(self):
        q=np.linspace(.2,2.,20);noise=.13;A=design(self.x,self.cfg);H=design(self.t,self.cfg)
        ll,mean,latent=high_precision_sr(A,H,q,noise,self.y)
        expected=state(A,self.y,q,noise)
        p=predict_sr({"weights":q,"noise_variance":noise},self.x,self.y,self.t,self.cfg)
        assert_allclose(ll,expected["ll"],atol=1e-12)
        assert_allclose(mean,p["mean"],atol=1e-12)
        assert_allclose(latent,p["latent_variance"],atol=1e-12)

