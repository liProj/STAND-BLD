# Running comparisons

Commands below assume the image manifests and original image folders are prepared. The fixed split files are supplied; regenerating them with a different library version may change the split.

## CNN baselines

The following example reproduces one Inception-v3 source fit and its external predictions.

```bash
python code/finetune.py inception_v3 data/bld/splits/random_s0.csv ref_random_s0__inception_v3 0
```

Repeat seeds 0, 1, and 2 for `inception_v3`, `mobilenet_v3_large`, and `resnet50`. The external test predictions are the `set=inat` rows. Main-table evaluation joins them to `data/inat/eval.csv` and keeps `part=test`.

For the authors' released checkpoint, download it as described in `DATA.md`, then run:

```bash
python code/author_ckpt.py
```

## Frozen DINOv2 and SigLIP 2 baselines

```bash
python code/feats.py bld vit_base_patch14_dinov2.lvd142m 392
python code/feats.py inat vit_base_patch14_dinov2.lvd142m 392
python code/feats_siglip.py google/siglip2-base-patch16-224
python code/run_recent_baselines_20261022.py
```

The fitting script performs fivefold source capture-group cross-validation over `C = 0.001, 0.01, 0.1, 1, 10`. It retains both candidates regardless of their external scores. Its output directory must not already exist, preventing accidental overwrite of a completed run.

## Frozen global/local teacher components

```bash
python code/run_ours.py L1 k=0 mil=0 protocols=session,lodo,inat
python code/run_ours.py L2 k=8 mil=0 protocols=session,lodo,inat
python code/run_ours.py L3 k=8 mil=1 protocols=session,lodo,inat
```

`k=auto` is also supported for source-only grouped selection in the original experiment driver. Consult the per-image prediction columns (`k`, `n_train`) and manuscript supplementary tables for the settings associated with each recorded result.

## Student configurations

The main entry point is `python code/train.py --config configs/stand.json`. The original `distill.py` interface exposes `rounds`, `mil`, `nap`, `tstd`, `npool`, `share`, `seed`, `teacher`, `k`, and `C`. Recorded combinations are in `reference_results/distill/*.json`. A tag identifies the output filename; choose a new tag for a new run.

`share` changes retrieval-source composition for sensitivity experiments. It is not a known disease prevalence. `conf=rank0.3` selects equal counts from both score tails, while `conf=adapt0.6` uses fixed class-specific quotas from the initial teacher. Numeric confidence thresholds invoke a Gaussian-mixture comparison and are not the main STAND selection rule.
