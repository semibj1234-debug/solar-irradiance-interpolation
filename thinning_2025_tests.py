"""Sampling, fixed domain, response isolation, metric and checkpoint tests."""
import unittest,inspect,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal,assert_allclose
import thinning_2025 as mc
import sr

class ThinningTests(unittest.TestCase):
    def test_nested_unique_reproducible_subsets(self):
        permutations=[]
        for b in range(20):
            seed,p,sets=mc.subsets(b);seed2,p2,sets2=mc.subsets(b)
            assert_array_equal(p,p2);self.assertEqual(seed,seed2)
            self.assertEqual(len(set(p)),88);previous=set()
            for n in sorted(mc.NS):
                self.assertEqual(len(sets[n]),n);self.assertEqual(len(set(sets[n])),n)
                self.assertTrue(previous.issubset(set(sets[n])))
                assert_array_equal(sets[n],sets2[n]);assert_array_equal(sets[n],np.sort(p[:n]))
                previous=set(sets[n])
            permutations.append(tuple(p))
        self.assertEqual(len(set(permutations)),20)

    def test_domain_fixed_and_response_subset_only(self):
        full=pd.DataFrame(dict(coord1=[0.,1.,.2,.4],coord2=[0.,1.,.7,.8],nins=[1.,9.,3.,5.]))
        chosen=full.iloc[[2,3]].copy()
        x,z,tr=mc.subset_arrays(chosen,[0.,0.],[1.,1.])
        assert_array_equal(x,chosen[['coord1','coord2']].to_numpy())
        assert_allclose(z,[-1.,1.]);self.assertEqual(tr.response_mean,4);self.assertEqual(tr.response_sd,1)
        full.loc[[0,1],'nins']=[1e30,-1e30]
        x2,z2,tr2=mc.subset_arrays(full.iloc[[2,3]],[0.,0.],[1.,1.])
        assert_array_equal(x,x2);assert_array_equal(z,z2);self.assertEqual(tr,tr2)

    def test_same_arrays_all_M_and_prediction_shapes(self):
        rng=np.random.default_rng(2);x=rng.uniform(size=(11,2));z=rng.normal(size=11)
        before=x.copy();responses=z.copy();target=rng.uniform(size=(21,2))
        for k in mc.KS:
            settings=mc.settings_for(k);A=sr.design(x,settings)
            self.assertEqual(A.shape,(11,k*k))
            p=sr.predict_sr(dict(weights=np.ones(k*k),noise_variance=.7),x,z,target,settings)
            for v in p.values():self.assertEqual(v.shape,(21,))
            assert_array_equal(x,before);assert_array_equal(z,responses)

    def test_fit_interface_and_label_read_order(self):
        self.assertEqual(list(inspect.signature(mc.subsets).parameters),['b'])
        self.assertEqual(list(inspect.signature(mc.fit_only).parameters),['x','z','s','diagnostics'])
        source=inspect.getsource(mc.one_model)
        self.assertLess(source.index('fit=fit_only'),source.index("truth=pd.read_parquet"))
        self.assertLess(source.index("a=audit_fit"),source.index("truth=pd.read_parquet"))
        self.assertNotIn('heldout',inspect.getsource(mc.subset_arrays))

    def test_metrics_empty_singleton_and_invalid_variance(self):
        y=np.array([1.,3.]);mu=np.array([0.,1.]);v=np.array([2.,4.])
        r=mc.scores(y,mu,v,[True,True])
        self.assertEqual(r['MAE'],1.5);self.assertAlmostEqual(r['RMSE'],np.sqrt(2.5))
        self.assertEqual(mc.scores(y,mu,v,[False,False])['MAE'],None)
        self.assertEqual(mc.scores(y,mu,v,[True,False])['count'],1)
        with self.assertRaises(FloatingPointError):mc.scores(y,mu,[0.,4.],[True,True])

    def test_variance_backtransform(self):
        f=pd.DataFrame(dict(coord1=[0.,1.,0.],coord2=[0.,0.,1.],nins=[2.,6.,10.]))
        x,z,tr=mc.subset_arrays(f,[0.,0.],[1.,1.])
        r=tr.inverse_prediction(dict(mean=np.array([.5]),latent_variance=np.array([.2]),observation_variance=np.array([.7])))
        assert_allclose(r['observation_variance'],.7*np.var(f.nins))
        assert_allclose(r['mean'],np.mean(f.nins)+.5*np.std(f.nins))

    def test_record_roundtrip_and_resume_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);d=out/'fits/b00/n11/M001';d.mkdir(parents=True)
            r=dict(status='audit_failed',error='intentional test',replicate=0,n_train=11,M=1)
            mc.save(d/'result.json',r)
            self.assertEqual(mc.one_model((str(out),0,11,1)),r)

if __name__=='__main__':unittest.main(verbosity=2)
