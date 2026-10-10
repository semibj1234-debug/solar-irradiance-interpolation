"""Independent direct algebra tests; existing SR-only tests plus application checks."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path("E:/04_Project/01_OIBC 2025/code_paper/refactored")))
import inspect
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.fft import dct
from sr_kriging_benchmark import sr
from sr_kriging_benchmark.tests import Equations


class Single2025Tests(unittest.TestCase):
    def test_1d_and_rectangular_kronecker_order(self):
        nx,ny=3,5
        bx=dct(np.eye(nx),type=2,norm="ortho",axis=0).T
        by=dct(np.eye(ny),type=2,norm="ortho",axis=0).T
        assert_allclose(bx.T@bx,np.eye(nx),atol=1e-14)
        assert_allclose(by.T@by,np.eye(ny),atol=1e-14)
        B=sr.grid_basis(nx,ny)
        for ix in range(nx):
            for iy in range(ny):
                for fx in range(nx):
                    for fy in range(ny):
                        self.assertAlmostEqual(B[ix*ny+iy,fx*ny+fy],bx[ix,fx]*by[iy,fy],places=14)
        # Paper 2026 uses column-stacking: permutation, not a different kernel.
        perm=np.arange(nx*ny).reshape(nx,ny).ravel(order="F")
        assert_allclose(B[np.ix_(perm,perm)],np.kron(by,bx),atol=1e-14)
        coeff=np.arange(nx*ny,dtype=float).reshape(nx,ny)
        assert_allclose((B@coeff.ravel()).reshape(nx,ny),bx@coeff@by.T,atol=1e-13)

    def test_actual_bilinear_weights_independently(self):
        psi=sr.observation_operator(np.array([[.25,.125]]),3,5)
        expected=np.zeros((1,15))
        expected[0,[0,1,5,6]]=.25
        assert_allclose(psi,expected,atol=1e-15)
        # Grid noise covariance must remain I at the observation level,
        # not Psi Psi.T when using off-grid bilinear weights.
        self.assertNotAlmostEqual(float((psi@psi.T)[0,0]),1.)

    def test_multiple_update_pairs_and_noise_recomputation(self):
        rng=np.random.default_rng(812)
        A=rng.normal(size=(6,9));y=rng.normal(size=6);q=np.linspace(.7,1.5,9);eta=.6
        for _ in range(4):
            S=np.linalg.inv(np.diag(1/q)+A.T@A/eta)
            mu=S@A.T@y/eta
            qnew=mu**2/(1-np.diag(S)/q)
            Snew=np.linalg.inv(np.diag(1/qnew)+A.T@A/eta)
            munew=Snew@A.T@y/eta
            enew=(np.sum((y-A@munew)**2)+np.trace(A@Snew@A.T))/len(y)
            fastq,faste=sr.update(A,y,q,eta)
            assert_allclose(fastq,qnew,rtol=1e-10,atol=1e-12)
            assert_allclose(faste,enew,rtol=1e-10,atol=1e-12)
            q,eta=fastq,faste

    def test_no_threshold_or_noise_floor(self):
        A=np.array([[1.,.2],[.3,1.],[.6,.8]])
        s=sr.state(A,np.array([1.,2.,.5]),np.array([1.,0.]),.1)
        self.assertTrue(np.isfinite(s["ll"]))
        with self.assertRaises(FloatingPointError):
            sr.state(A,np.ones(3),np.ones(2),0.)

    def test_training_only_and_label_independent_prediction(self):
        from single_2025 import training_arrays,evaluate
        train=pd.DataFrame(dict(coord1=[0.,1.,0.,1.],coord2=[0.,0.,1.,1.],nins=[1.,2.,3.,4.]))
        x,z,t=training_arrays(train)
        settings=dict(grid_nx=3,grid_ny=4)
        fit=dict(weights=np.ones(12),noise_variance=.2)
        targets=np.array([[.2,.4],[.7,.8]])
        before=sr.predict_sr(fit,x,z,targets,settings)
        ytest=np.array([20.,30.])
        evaluate(ytest,before["mean"],np.ones(2,bool))
        ytest[:]=1e9
        after=sr.predict_sr(fit,x,z,targets,settings)
        for key in before: assert_array_equal(before[key],after[key])
        self.assertEqual(list(inspect.signature(sr.fit_sr).parameters),["x","z","settings","seed"])
        self.assertEqual(list(inspect.signature(training_arrays).parameters),["train"])
        assert_allclose(z.mean(),0,atol=1e-14)
        assert_allclose(z.std(),1,atol=1e-14)


def suite():
    names=["test_basis_and_operator","test_sr_covariance_psd_and_paper_likelihood",
           "test_sr_equations_16_20_against_dense_parameter_space",
           "test_sr_prediction_equations_8_9_and_coefficient_posterior",
           "test_high_precision_reference","test_hull_mask_and_extrapolation_exclusion"]
    return unittest.TestSuite([Equations(name) for name in names]+
                              list(unittest.defaultTestLoader.loadTestsFromTestCase(Single2025Tests)))

if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(not result.wasSuccessful())

