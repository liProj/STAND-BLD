# Protocol and implementation details

The source cohort has 1,934 images. Capture sessions (采集批次) group consecutive photographs from the same device with adjacent intervals no longer than 300 seconds. Source class labels and these capture groups define the teacher projections. The term `session` in the original code denotes these acquisition groups.

STAND uses DINOv3 ViT-B/16 at 384 pixels. Global and patch branches use separate rank-eight bases estimated from weighted within-class capture-group mean deviations. The global head is a class-balanced logistic regression with `C=0.1`. The local head shares one linear scorer across image patches and uses log-mean-exp pooling with `tau=1`. Their logits receive equal weight.

The initial teacher predicts the independent unlabeled pool. The fraction of positive logits, clipped to `[0.05, 0.95]`, determines positive and negative quotas with `rho=0.6`. For the recorded main run these are 789 positive and 996 negative pseudo-labels. Both quotas remain fixed for two rounds; current scores determine which images are selected. Source projection bases and the encoder remain fixed.

## Normalization in the recorded implementation

The global student uses full-pool statistics when refitting and scoring. The local head is fitted with statistics from selected pseudo-labeled images, then uses full-pool statistics for scoring, without refitting its parameters. This distinction is retained in the released code and heads. It should not be described as identical normalization in both branches. Alternative normalization runs are retained in the experiment records.

## Labels and evaluation history

- The adaptation pool provides no symptom training labels.
- The 527-image target development set is labeled and used for configuration selection.
- The 1,856-image external test set uses observation-derived weak labels.
- Adaptation, development, and test sets are disjoint by photographer.
- The external test was inspected during development. Later checks and baseline additions use the same set and are not newly blinded confirmation experiments.

The code includes a separate transductive capture-fold diagnostic in `distill_session.py`. It adapts to the unlabeled held-out fold and evaluates that fold. It is not the main external STAND protocol, which adapts on an independent photo pool before predicting test photos individually.

## Comparisons and saved results

The CNN entries in the main table average three seed-specific metrics. They are not an ensemble. DINOv2 and SigLIP 2 use source-only linear probes with source-grouped hyperparameter selection. The BLD EfficientNetV2-S row uses the authors' released model. STAND uses its recorded single seed. Pretraining sources and fitting budgets differ, as described in the manuscript.

All retained per-image predictions and student settings are under `reference_results/`. Source random-test rows from the old exploratory protocol are omitted from this release. CNN filenames still contain `random_s<seed>` because that is the source training split used before their external evaluation.

## Release validation

The release checks validate file/label alignment, photographer separation, capture-group separation, the rounded seven-row main table, exported projection orthonormality, and batch-independent head prediction. Preparation also compares predictions from exported heads against a small cached-feature slice from the actual original experiment. Recorded validation details are in `verification.json`.

The core functions in `methods.py`, `distill.py`, and `run_ours.py` were checked for syntax-tree equivalence after removing documentation. Changes to publication copies concern configurable paths, online model-download defaults, and explanatory text. Original experiment files were left intact. Full GPU retraining and image-encoder inference were not rerun as part of preparing this upload.
