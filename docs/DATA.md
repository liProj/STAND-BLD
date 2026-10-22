# Data and pretrained models

## Source photographs

The source data accompany Waldo et al., *Beech leaf disease symptom detection using deep learning and computer vision tools*, Journal of Nematology (2026), [doi:10.2478/jofnem-2026-0011](https://doi.org/10.2478/jofnem-2026-0011).

- [USDA dataset record, doi:10.15482/USDA.ADC/30850214.v1](https://doi.org/10.15482/USDA.ADC/30850214.v1)
- [US Government catalog record](https://catalog.data.gov/dataset/data-from-beech-leaf-disease-symptom-detection-using-deep-learning-and-computer-vision-too)
- [Deposited ZIP download](https://ndownloader.figshare.com/files/60816103)

The source catalog lists CC0 for this deposit. Download the archive separately and save it as `data/bld_zip/BLD_dataset.zip`. `code/prep.py` preserves the original pixel orientation, resizes the long side to 768 pixels, and writes `data/bld/img/<label>/<stem>.jpg`.

The released manifest fixes the 1,934 images used by this study. If the deposit changes or a file is unavailable, keep track of the difference rather than silently changing the evaluated cohort. `code/validate_data.py` checks counts and order against the saved capture groups.

## Public adaptation, development and test photographs

`data/inat/eval.csv` lists 527 development and 1,856 test photographs. `data/inat/pool.csv` lists 2,976 adaptation photographs. These are fixed snapshots of public iNaturalist photo URLs. `obs` identifies the original observation, available at `https://www.inaturalist.org/observations/<obs>`.

The `license` column records the photo license at collection. Check the original observation and its attribution before redistributing a photo. Raw photographs are not included in the repository. The downloader retrieves only the listed files; deleted or inaccessible entries are reported as missing.

Photographer IDs were replaced with shared anonymous group identifiers. The mapping is consistent across the adaptation/development/test manifests so their separation can be verified. Geographic coordinates, place descriptions, and account names are not included.

The evaluation label `y=1` denotes the positive symptom category under the study's weak-label rule. It is not an independently verified clinical or lesion-mask annotation. The pool's `source` value describes the retrieval source and supports composition diagnostics. It is not used as a symptom training label.

## Expected layout

```text
data/
  bld_zip/BLD_dataset.zip        # downloaded separately
  bld/meta.csv
  bld/groups.csv
  bld/splits/*.csv
  bld/img/BLD/*.jpg
  bld/img/Asymptomatic/*.jpg     # use the exact paths in meta.csv
  inat/eval.csv
  inat/pool.csv
  inat/img/pos/*.jpg
  inat/img/neg/*.jpg
  inat/img/pool/*.jpg
```

Use manifest paths verbatim; the schematic folder names above do not override them.

## Model sources

- [DINOv3](https://github.com/facebookresearch/dinov3), consumed through `timm` as `vit_base_patch16_dinov3.lvd1689m`.
- [DINOv2](https://github.com/facebookresearch/dinov2), frozen baseline `vit_base_patch14_dinov2.lvd142m` at 392 pixels.
- [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224), frozen baseline.
- [Waldo et al.'s released BLD model](https://huggingface.co/spaces/bdwaldo/BLD), evaluated by `code/author_ckpt.py`. Obtain `efficientnet_final_model.pth` separately and place it in `data/hf_space/`.

Obtain foundation-model/checkpoint access from the original providers and follow their terms. The repository contains the STAND student classification heads only. Their exact source-checkpoint and exported-array hashes are recorded in `models/stand_heads.json`.
