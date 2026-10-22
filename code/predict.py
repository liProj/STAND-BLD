"""STAND prediction with the exported frozen-feature classifier heads."""
from pathlib import Path
import argparse, csv
import numpy as np

def load_heads(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def predict_features(global_features,patch_features,heads,batch=32):
    X=np.asarray(global_features,dtype=np.float64)
    P=np.asarray(patch_features)
    if X.ndim!=2 or P.ndim!=3 or len(X)!=len(P):raise ValueError('Expected global [N,D] and patch [N,T,D] arrays')
    if X.shape[1]!=heads['cls_mu'].size or P.shape[-1]!=heads['mil_w'].size:raise ValueError('Feature dimension differs from the released DINOv3 heads')
    A=(X-heads['cls_mu'])/heads['cls_sd']
    V=heads['cls_V'];A=A-A@V.T@V
    global_z=(A@heads['cls_coef'].T+heads['cls_intercept']).reshape(-1)
    patch_z=[];tau=float(heads['tau'])
    for i in range(0,len(P),batch):
        A=(P[i:i+batch].astype(np.float32)-heads['mil_mu'])/heads['mil_sd']
        V=heads['mil_V'];A=A-(A@V.T)@V
        scores=A@heads['mil_w']+heads['mil_b']
        scaled=scores*tau;maximum=scaled.max(axis=1)
        value=(maximum+np.log(np.exp(scaled-maximum[:,None]).sum(axis=1))-np.log(scores.shape[1]))/tau
        patch_z.append(value)
    return .5*global_z+.5*np.concatenate(patch_z)

def predict_images(paths,heads,device):
    import timm,torch
    from PIL import Image
    from torchvision.transforms import v2 as T
    model=timm.create_model('vit_base_patch16_dinov3.lvd1689m',pretrained=True,num_classes=0,dynamic_img_size=True).to(device).eval()
    cfg=model.pretrained_cfg
    transform=T.Compose([T.Resize((384,384),antialias=True),T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize(cfg['mean'],cfg['std'])])
    values=[]
    with torch.no_grad():
        for start in range(0,len(paths),16):
            xs=[]
            for p in paths[start:start+16]:
                with Image.open(p) as im:xs.append(transform(im.convert('RGB')))
            x=torch.stack(xs).to(device)
            if str(device).startswith('cuda'):
                with torch.autocast('cuda',dtype=torch.bfloat16):tokens=model.forward_features(x).float()
            else:tokens=model.forward_features(x).float()
            c=tokens[:,0].cpu().numpy()
            p=tokens[:,model.num_prefix_tokens:].cpu().numpy().astype(np.float16)
            values.extend(predict_features(c,p,heads).tolist())
    return np.asarray(values)

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads',type=Path,default=root/'models/stand_heads.npz')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--features',type=Path,help='NPZ with files, cls, patch extracted by code/feats.py')
    mode.add_argument('--images',type=Path,nargs='+')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,default=Path('results/predictions.csv'))
    a=p.parse_args();heads=load_heads(a.heads)
    if a.features:
        with np.load(a.features,allow_pickle=False) as f:
            z=predict_features(f['cls'],f['patch'],heads);names=f['files'].astype(str).tolist()
    else:names=[str(q) for q in a.images];z=predict_images(a.images,heads,a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['file','logit','prediction'])
        for name,value in zip(names,z):w.writerow([name,float(value),'symptomatic' if value>0 else 'asymptomatic'])
    print(f'{len(z)} predictions written to {a.output}')

if __name__=='__main__':main()
