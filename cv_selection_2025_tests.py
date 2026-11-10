"""Independent selection arithmetic and fold-contract tests."""
import unittest,inspect,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import cv_selection_2025 as cv
class CVTests(unittest.TestCase):
    def setUp(self):
        self.x=np.column_stack([np.linspace(0,1,11),np.sin(np.arange(11))*.4+.5])
    def test_partitions(self):
        for method in cv.METHODS:
            fs,g=cv.partitions(self.x,method,123)
            self.assertEqual(sorted(np.concatenate(fs).tolist()),list(range(11)))
            self.assertTrue(all(0<len(f)<10 for f in fs))
            self.assertEqual(len(fs),11 if method=='loo' else 5)
            if method=='loo':self.assertTrue(all(len(f)==1 for f in fs))
    def test_spatial_geometry(self):
        fs,g=cv.partitions(self.x,'spatial',1)
        for i,f in enumerate(fs):
            self.assertTrue(np.all(self.x[f,0]>=g['boundaries'][i]))
            self.assertTrue(np.all(self.x[f,0]<=g['boundaries'][i+1]))
        self.assertAlmostEqual(sum(g['widths']),1.)
    def test_reproducible_and_no_M_argument(self):
        for method in cv.METHODS:
            a,_=cv.partitions(self.x,method,42);b,_=cv.partitions(self.x,method,42)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(list(inspect.signature(cv.partitions).parameters),['x','method','seed'])
    def test_weighted_mae_and_se(self):
        r=[dict(status='ok',n_validation=2,absolute_errors=[1.,3.]),dict(status='ok',n_validation=1,absolute_errors=[8.])]
        c=cv.candidate_summary(r,3,4)
        self.assertEqual(c['CV_MAE'],4.)
        self.assertAlmostEqual(c['SE'],3.)
    def test_loo_se(self):
        r=[dict(status='ok',n_validation=1,absolute_errors=[v]) for v in [1.,2.,6.]]
        c=cv.candidate_summary(r,3,4)
        self.assertAlmostEqual(c['SE'],np.std([1,2,6],ddof=1)/np.sqrt(3))
    def test_selection_and_ties(self):
        c=[dict(M=m,CV_MAE=e,SE=2.,status='complete_cv') for m,e in [(1,11.),(4,10.),(9,10.),(16,14.)]]
        r=cv.select(c);self.assertEqual(r['minimum'],4);self.assertEqual(r['one_se'],1)
    def test_incomplete_and_all_failed(self):
        r=[dict(status='ok',n_validation=1,absolute_errors=[1.]),dict(status='audit_failed',n_validation=1)]
        c=cv.candidate_summary(r,2,1)
        self.assertEqual(c['status'],'incomplete_cv');self.assertIsNone(c['CV_MAE'])
        self.assertEqual(cv.select([c])['status'],'selection_failed')
    def test_fold_scaling_and_label_boundary(self):
        f=pd.DataFrame(dict(coord1=[0.,.3,.6,1.],coord2=[0.,1.,0.,1.],nins=[1.,3.,5.,99999.]))
        a=cv.subset_arrays(f.iloc[:3],[0,0],[1,1]);f.loc[3,'nins']=-99999.
        b=cv.subset_arrays(f.iloc[:3],[0,0],[1,1]);assert_allclose(a[1],b[1]);self.assertAlmostEqual(a[2].response_mean,3.)
        self.assertNotIn('heldout.parquet',inspect.getsource(cv.run_candidate))
        self.assertEqual(list(inspect.signature(cv.fold_fit).parameters),['training','validation_x','settings'])
    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'r.json.gz';cv.write_checkpoint(p,dict(status='audit_failed',q=[1.,0.]))
            self.assertEqual(cv.load_checkpoint(p),dict(status='audit_failed',q=[1.,0.]))
def suite():
    from single_2025_tests import suite as base
    from basis_sensitivity_2025_tests import VariableGridTests
    from thinning_2025_tests import ThinningTests
    s=base()
    for cls in [VariableGridTests,ThinningTests,CVTests]:s.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    return s
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):r=unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(not r.wasSuccessful())
