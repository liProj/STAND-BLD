"""SigLIP 2 image embeddings for both image sets, and text embeddings for the symptom prompts.

    python feats_siglip.py <hf-name>     e.g. google/siglip2-base-patch16-224

Output results/feats/{bld,inat}__<name>.npz (files, emb: L2-normalised image embeddings) and
results/feats/text__<name>.npz (prompts, group, emb).

The prompts describe the symptom in words a field guide would use; none was tuned on any image.
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from common import BASE
POS = ["a beech leaf with dark green bands between the veins",
       "beech leaves with dark interveinal banding caused by beech leaf disease",
       "a photo of beech leaf disease",
       "beech leaves with dark stripes between the veins seen against the light",
       "thickened, leathery, curled beech leaves with dark bands",
       "a beech leaf infected by the nematode Litylenchus crenatae"]
NEG = ["a healthy green beech leaf",
       "healthy beech leaves with evenly coloured blades",
       "a photo of healthy American beech foliage",
       "normal beech leaves with no banding between the veins",
       "a beech leaf with uniform colour seen against the light",
       "a photo of an asymptomatic beech leaf"]


def emb(o):
    return torch.nn.functional.normalize(o if torch.is_tensor(o) else o.pooler_output, dim=-1)


@torch.no_grad()
def main():
    name = sys.argv[1]
    tag = name.split("/")[-1]
    model = AutoModel.from_pretrained(name).cuda().eval()
    proc = AutoProcessor.from_pretrained(name)
    ti = proc(text=POS + NEG, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
    np.savez(f"{BASE}/results/feats/text__{tag}.npz", prompts=np.array(POS + NEG),
             group=np.array([1] * len(POS) + [0] * len(NEG)),
             emb=emb(model.get_text_features(**ti)).float().cpu().numpy(),
             scale=float(model.logit_scale.exp()), bias=float(model.logit_bias))
    for sset, csv, root in (("bld", "data/bld/meta.csv", "data/bld/img"),
                            ("inat", "data/inat/eval.csv", "data/inat/img")):
        files = pd.read_csv(f"{BASE}/{csv}").file.tolist()
        out = []
        for i in range(0, len(files), 48):
            ims = [Image.open(f"{BASE}/{root}/{f}").convert("RGB") for f in files[i:i + 48]]
            px = proc(images=ims, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(emb(model.get_image_features(**px).float()
                               if torch.is_tensor(model.get_image_features(**px))
                               else model.get_image_features(**px)).float().cpu().numpy())
            if i % 480 == 0:
                print(tag, sset, i, "/", len(files), flush=True)
        np.savez(f"{BASE}/results/feats/{sset}__{tag}.npz", files=np.array(files),
                 emb=np.concatenate(out))
    print("DONE feats_siglip", tag, flush=True)


if __name__ == "__main__":
    main()
