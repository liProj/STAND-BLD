"""Frozen-backbone features for an image list.

    python feats.py <set> <model> <res> [patch]

<set>   bld  -> data/bld/meta.csv   (column file, root data/bld/img)
        inat -> data/inat/eval.csv  (column file, root data/inat/img)
<model> a timm name (DINOv3 / DINOv2 ViTs and ConvNeXts), or tv:<torchvision arch> for the
        ImageNet backbones the article fine-tunes (used frozen, as a linear-probe control)
<res>   square input side; images are squashed to it exactly as the authors' loader does
patch   also store the patch-token grid (float16) for the localisation analyses

Output results/feats/<set>__<model>__<res>.npz : files, cls, avg  (+ patch).
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

from common import BASE
SETS = {"bld": ("data/bld/meta.csv", "data/bld/img"),
        "inat": ("data/inat/eval.csv", "data/inat/img"),
        "pool": ("data/inat/pool.csv", "data/inat/img")}


class DS(Dataset):
    def __init__(self, root, files, tf):
        self.root, self.files, self.tf = root, list(files), tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(f"{self.root}/{self.files[i]}").convert("RGB"))


def load(name):
    if name.startswith("tv:"):
        m = getattr(models, name[3:])(weights="DEFAULT")
        if hasattr(m, "classifier"):
            m.classifier = torch.nn.Identity()
        else:
            m.fc = torch.nn.Identity()
        return m, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), None
    m = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True) \
        if "vit" in name else timm.create_model(name, pretrained=True, num_classes=0)
    cfg = m.pretrained_cfg
    return m, cfg["mean"], cfg["std"], getattr(m, "num_prefix_tokens", None)


@torch.no_grad()
def main():
    sset, name, res = sys.argv[1], sys.argv[2], int(sys.argv[3])
    want_patch = len(sys.argv) > 4 and sys.argv[4] == "patch"
    out = f"{BASE}/results/feats/{sset}__{name.replace(':', '_').replace('/', '_')}__{res}.npz"
    if os.path.exists(out) and not os.environ.get("FORCE"):
        print("exists", out)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    csv, root = SETS[sset]
    files = pd.read_csv(f"{BASE}/{csv}").file.tolist()
    m, mean, std, npre = load(name)
    m = m.cuda().eval()
    tf = T.Compose([T.Resize((res, res), antialias=True), T.ToImage(),
                    T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])
    dl = DataLoader(DS(f"{BASE}/{root}", files, tf), batch_size=32, num_workers=8)
    cls, avg, patch = [], [], []
    for i, x in enumerate(dl):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = x.cuda()
            if npre is not None:                      # ViT: prefix tokens then the patch grid
                t = m.forward_features(x).float()
                c, p = t[:, 0], t[:, npre:]
                a = p.mean(1)
            else:                                     # CNN: pooled embedding only
                f = m.forward_features(x).float() if hasattr(m, "forward_features") else None
                if f is not None and f.ndim == 4:
                    p = f.flatten(2).transpose(1, 2)
                    c = a = p.mean(1)
                else:
                    c = a = m(x).float()
                    p = None
        cls.append(c.cpu().numpy())
        avg.append(a.cpu().numpy())
        if want_patch and p is not None:
            patch.append(p.cpu().numpy().astype(np.float16))
        if i % 10 == 0:
            print(name, sset, (i + 1) * 32, "/", len(files), flush=True)
    kw = dict(files=np.array(files), cls=np.concatenate(cls), avg=np.concatenate(avg))
    if patch:
        kw["patch"] = np.concatenate(patch)
    np.savez(out, **kw)
    print("DONE feats", out, kw["cls"].shape, flush=True)


if __name__ == "__main__":
    main()
