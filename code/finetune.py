"""Re-run of the article's fine-tuning recipe (Waldo et al. 2026, J. Nematol. 58) on any split.

Stated in the article and followed here: torchvision ImageNet weights; input 384 / 299 / 224 /
224 px for EfficientNetV2-S / InceptionV3 / MobileNetV3-L / ResNet50; one-logit head with
BCE-with-logits; AdamW; backbone frozen for the first 3 epochs, then learning rates 1e-4
(backbone) and 1e-3 (head); batch 16; at most 100 epochs; early stopping after 5 epochs without
validation-loss improvement; gradient clipping; augmentation = random resized crop, horizontal
flip, random affine, random grayscale, brightness/contrast/saturation jitter.

Not stated (the supplemental notes are not in the PMC record), so chosen here and kept fixed for
every model: crop scale 0.6-1.0, affine +-15 deg / 10% shift, grayscale p 0.1, jitter 0.2,
weight decay 1e-2, clip norm 1.0, no scheduler, best-validation-loss weights restored.

    python finetune.py <arch> <split.csv> <tag> [seed]

split.csv has columns file,part (train/val/test). Per-image probabilities for val, test and
every extra evaluation set are written to results/preds/<tag>.csv; the curve to
results/curves/<tag>.csv.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

from common import BASE
SIZE = {"efficientnet_v2_s": 384, "inception_v3": 299, "mobilenet_v3_large": 224,
        "resnet50": 224}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build(arch):
    m = getattr(models, arch)(weights="DEFAULT")
    if arch in ("efficientnet_v2_s", "mobilenet_v3_large"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
        head = list(m.classifier[-1].parameters())
    else:
        m.fc = nn.Linear(m.fc.in_features, 1)
        head = list(m.fc.parameters())
        if arch == "inception_v3":
            m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, 1)
            head += list(m.AuxLogits.fc.parameters())
    ids = {id(p) for p in head}
    body = [p for p in m.parameters() if id(p) not in ids]
    return m, body, head


class DS(Dataset):
    def __init__(self, root, files, y, tf):
        self.root, self.files, self.y, self.tf = root, list(files), list(y), tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(f"{self.root}/{self.files[i]}").convert("RGB")
        return self.tf(im), torch.tensor(float(self.y[i]))


def tfs(size):
    norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        T.RandomGrayscale(p=0.1),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] + norm)
    test = T.Compose([T.Resize((size, size), antialias=True)] + norm)   # as in the authors' app
    return train, test


@torch.no_grad()
def predict(m, dl, dev):
    m.eval()
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last))
        out.append(z.float().squeeze(1).cpu())
    return torch.cat(out).numpy()


def main():
    arch, split, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    out = f"{BASE}/results/preds/{tag}.csv"
    if os.path.exists(out) and not os.environ.get("FORCE"):
        print("exists, skip", out)
        return
    os.makedirs(f"{BASE}/results/preds", exist_ok=True)
    os.makedirs(f"{BASE}/results/curves", exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda"
    sp = pd.read_csv(split)
    sp["y"] = (sp.file.str.split("/").str[0] == "BLD").astype(int)      # 1 = symptomatic
    size = SIZE[arch]
    ttf, etf = tfs(size)
    root = f"{BASE}/data/bld/img"
    tr, va, te = (sp[sp.part == p] for p in ("train", "val", "test"))
    kw = dict(num_workers=6, pin_memory=True, persistent_workers=True)
    dl_tr = DataLoader(DS(root, tr.file, tr.y, ttf), batch_size=16, shuffle=True,
                       drop_last=True, **kw)
    ev = {"val": (root, va), "test": (root, te)}
    ext = f"{BASE}/data/inat/eval.csv"                                  # public external set
    if os.path.exists(ext):
        e = pd.read_csv(ext)
        ev["inat"] = (f"{BASE}/data/inat/img", e)
    dls = {k: DataLoader(DS(r, d.file, d.y, etf), batch_size=32, shuffle=False, **kw)
           for k, (r, d) in ev.items()}

    m, body, head = build(arch)
    m = m.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}],
                            weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best, bad, state, curve, t0 = 1e9, 0, None, [], time.time()
    for ep in range(100):
        frozen = ep < 3
        for p in body:
            p.requires_grad_(not frozen)
        m.train()
        if frozen:                       # frozen backbone: keep its BatchNorm statistics too
            for mod in m.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        tl, tc, n = 0.0, 0, 0
        for x, y in dl_tr:
            x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m(x)
                if arch == "inception_v3" and isinstance(z, tuple):
                    loss = lossf(z[0].float().squeeze(1), y) + 0.4 * lossf(
                        z[1].float().squeeze(1), y)
                    z = z[0]
                else:
                    loss = lossf(z.float().squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(y)
            tc += ((z.float().squeeze(1) > 0) == (y > 0.5)).sum().item()
            n += len(y)
        zv = predict(m, dls["val"], dev)
        yv = va.y.values
        vl = float(np.mean(np.logaddexp(0, zv) - yv * zv))
        va_acc = float(((zv > 0) == (yv > 0)).mean())
        curve.append(dict(epoch=ep + 1, train_loss=tl / n, train_acc=tc / n, val_loss=vl,
                          val_acc=va_acc, sec=time.time() - t0))
        print(f"{tag} ep{ep + 1} tl {tl / n:.4f} ta {tc / n:.4f} vl {vl:.4f} va {va_acc:.4f} "
              f"{time.time() - t0:.0f}s", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    m.load_state_dict(state)
    rows = []
    for k, (r, d) in ev.items():
        z = predict(m, dls[k], dev)
        rows.append(pd.DataFrame(dict(set=k, file=d.file.values, y=d.y.values, logit=z)))
    pd.concat(rows).assign(arch=arch, seed=seed, tag=tag).to_csv(out, index=False)
    pd.DataFrame(curve).assign(arch=arch, seed=seed, tag=tag).to_csv(
        f"{BASE}/results/curves/{tag}.csv", index=False)
    if os.environ.get("SAVE_CKPT"):
        os.makedirs(f"{BASE}/results/ckpt", exist_ok=True)
        torch.save(state, f"{BASE}/results/ckpt/{tag}.pt")
    print(f"DONE {tag} epochs {len(curve)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
