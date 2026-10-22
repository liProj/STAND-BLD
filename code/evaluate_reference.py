"""Recalculate the manuscript's main table from saved per-image predictions."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

EXPECTED={
    'Inception-v3':[71.17,87.74,57.34,67.00,85.24],
    'MobileNetV3-Large':[69.31,89.80,51.13,63.73,84.46],
    'ResNet-50':[71.26,90.82,53.06,66.82,85.10],
    'DINOv2-B/14':[77.59,99.18,59.67,74.51,95.71],
    'SigLIP 2-B/16':[71.34,89.59,54.07,67.44,85.17],
    'EfficientNetV2-S (released BLD checkpoint)':[81.30,81.00,86.16,83.50,89.64],
    'STAND':[91.38,93.87,90.19,91.99,97.25],
}
COLS=['accuracy','precision','recall','f1','auc']

def score(y,z):
    pred=z>0
    return np.array([accuracy_score(y,pred),precision_score(y,pred,zero_division=0),
                     recall_score(y,pred,zero_division=0),f1_score(y,pred,zero_division=0),roc_auc_score(y,z)])*100

def evaluate(root):
    root=Path(root)
    ev=pd.read_csv(root/'data/inat/eval.csv');test=ev[ev.part=='test'][['file','y']]
    assert len(test)==1856
    def one(path):
        df=pd.read_csv(path)
        if 'set' in df:df=df[df['set']=='inat']
        elif 'protocol' in df:df=df[df.protocol=='inat']
        df=test.merge(df[['file','y','logit']],on='file',validate='one_to_one',suffixes=('_manifest','_prediction'))
        assert len(df)==1856 and (df.y_manifest==df.y_prediction).all()
        assert np.isfinite(df.logit).all()
        return score(df.y_manifest.to_numpy(),df.logit.to_numpy())
    pre=root/'reference_results/predictions';recent=root/'reference_results/recent_baselines'
    out=[]
    for name,arch,year in [('Inception-v3','inception_v3',2016),('MobileNetV3-Large','mobilenet_v3_large',2019),('ResNet-50','resnet50',2016)]:
        values=np.mean([one(pre/f'ref_random_s{s}__{arch}.csv') for s in range(3)],axis=0)
        out.append(dict(method=name,year=year,runs=3,**dict(zip(COLS,values))))
    for name,path,year in [('DINOv2-B/14',recent/'dinov2_b14_predictions.csv',2024),('SigLIP 2-B/16',recent/'siglip2_b16_predictions.csv',2025),('EfficientNetV2-S (released BLD checkpoint)',pre/'author_ckpt.csv',2026),('STAND',pre/'distill_stand.csv',2026)]:
        out.append(dict(method=name,year=year,runs=1,**dict(zip(COLS,one(path)))))
    return pd.DataFrame(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--verify',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results/main_table.csv'))
    a=p.parse_args();df=evaluate(a.root)
    if a.verify:
        for row in df.to_dict('records'):
            for col,value in zip(COLS,EXPECTED[row['method']]):assert abs(round(row[col],2)-value)<.001,(row['method'],col,row[col],value)
    a.output.parent.mkdir(parents=True,exist_ok=True);df.to_csv(a.output,index=False,float_format='%.8f')
    print(df.round(2).to_string(index=False));print(json.dumps({'verified':a.verify,'test_images':1856,'output':str(a.output)}))
