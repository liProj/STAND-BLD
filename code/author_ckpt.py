"""Score the authors' released EfficientNetV2-S checkpoint exactly as their Gradio app does.

Transform, head and label convention are copied from the Space's app.py / labels_info.json:
Resize((384, 384)) -> ToTensor -> ImageNet normalisation; one logit; sigmoid(logit) = P(BLD).

    python author_ckpt.py            # source dataset (all 1,934 images) and the iNaturalist set
Outputs results/preds/author_ckpt.csv (set, file, y, logit).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s

from common import BASE
CK = f"{BASE}/data/hf_space/efficientnet_final_model.pth"


class DS(Dataset):
    def __init__(self, root, files, tf):
        self.root, self.files, self.tf = root, list(files), tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(f"{self.root}/{self.files[i]}").convert("RGB"))


@torch.no_grad()
def main():
    m = efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    state = torch.load(CK, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    res = m.load_state_dict(state, strict=False)
    print("missing", len(res.missing_keys), "unexpected", len(res.unexpected_keys))
    assert not res.missing_keys and not res.unexpected_keys
    m = m.cuda().eval()
    tf = T.Compose([T.Resize((384, 384)), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    sets = {"bld": (f"{BASE}/data/bld/meta.csv", f"{BASE}/data/bld/img"),
            "inat": (f"{BASE}/data/inat/eval.csv", f"{BASE}/data/inat/img")}
    rows = []
    for k, (csv, root) in sets.items():
        d = pd.read_csv(csv)
        y = d.y if "y" in d else (d.label == "BLD").astype(int)
        z = torch.cat([m(x.cuda()).squeeze(1).float().cpu()
                       for x in DataLoader(DS(root, d.file, tf), batch_size=32, num_workers=8)])
        rows.append(pd.DataFrame(dict(set=k, file=d.file, y=y, logit=z.numpy())))
        acc = ((z.numpy() > 0) == (y.values > 0)).mean()
        print(k, len(d), "accuracy", round(100 * acc, 2), flush=True)
    os.makedirs(f"{BASE}/results/preds", exist_ok=True)
    pd.concat(rows).to_csv(f"{BASE}/results/preds/author_ckpt.csv", index=False)
    print("DONE author_ckpt", flush=True)


if __name__ == "__main__":
    main()
