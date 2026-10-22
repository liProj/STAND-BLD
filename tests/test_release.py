"""Small CPU checks for released manifests, projection and prediction paths."""
import sys, unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from validate_data import validate
from evaluate_reference import evaluate,EXPECTED,COLS
from predict import load_heads,predict_features

class ReleaseChecks(unittest.TestCase):
    def test_manifest_separation(self):
        self.assertTrue(validate(ROOT)['photographer_disjoint'])

    def test_main_table_from_predictions(self):
        for row in evaluate(ROOT).to_dict('records'):
            np.testing.assert_allclose(np.round([row[c] for c in COLS],2),EXPECTED[row['method']],atol=.001)

    def test_classifier_prediction_is_independent_of_batch_size(self):
        heads=load_heads(ROOT/'models/stand_heads.npz');rng=np.random.default_rng(7)
        X=rng.normal(size=(7,768));P=rng.normal(size=(7,12,768)).astype(np.float16)
        z1=predict_features(X,P,heads,batch=1);z2=predict_features(X,P,heads,batch=7)
        np.testing.assert_allclose(z1,z2,rtol=1e-5,atol=1e-5)
        self.assertTrue(np.isfinite(z1).all())

    def test_projection_bases_are_orthonormal(self):
        heads=load_heads(ROOT/'models/stand_heads.npz')
        for k in ['cls_V','mil_V']:
            V=heads[k]
            np.testing.assert_allclose(V@V.T,np.eye(len(V)),atol=2e-6)

if __name__=='__main__':unittest.main()
