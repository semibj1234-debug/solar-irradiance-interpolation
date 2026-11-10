"""Adversarial end-to-end fold response isolation, using actual fit_sr calls."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from cv_selection_2025 import *
class LeakageTest(unittest.TestCase):
    def test_validation_perturbation_does_not_change_its_fold_model(self):
        original=pd.read_parquet(SOURCE/'training.parquet')
        rows=read(PREVIOUS/'subsets/b00_n11.json')['row_indices']
        domain=read(PREVIOUS/'domain.json')
        xx=(original.iloc[rows][['coord1','coord2']].to_numpy()-domain['coordinate_min'])/domain['coordinate_span']
        folds,geometry=partitions(xx,'random',42)
        manifest=dict(key='b00_n11',replicate=0,n_train=11,row_indices=rows,methods={'random':dict(folds=[f.tolist() for f in folds],geometry=geometry)})
        changed=original.copy();changed.loc[np.asarray(rows)[folds[0]],'nins']+=123456.
        outputs=[]
        with tempfile.TemporaryDirectory() as tmp:
            for version,frame in enumerate([original,changed]):
                out=Path(tmp)/str(version);(out/'cv_fold_manifest').mkdir(parents=True)
                save(out/'cv_fold_manifest/b00_n11.json',manifest)
                def reader(path):
                    self.assertEqual(Path(path).name,'training.parquet')
                    return frame.copy()
                with patch('cv_selection_2025.pd.read_parquet',side_effect=reader):run_candidate((str(out),'b00_n11','random',1))
                outputs.append(load_checkpoint(out/'cv_fold_results/b00_n11/random/M001/fold000.json.gz'))
        a,b=outputs
        self.assertEqual(a['status'],'ok');self.assertEqual(b['status'],'ok')
        def mathematical(value):
            if isinstance(value,dict):return {k:mathematical(v) for k,v in value.items() if 'time' not in k and 'seconds' not in k}
            if isinstance(value,list):return [mathematical(v) for v in value]
            return value
        self.assertEqual(mathematical(a['fit']),mathematical(b['fit']))
        self.assertEqual(a['transform'],b['transform'])
        self.assertEqual(a['predictions'],b['predictions'])
        self.assertNotEqual(a['absolute_errors'],b['absolute_errors'])
if __name__=='__main__':unittest.main(verbosity=2)
