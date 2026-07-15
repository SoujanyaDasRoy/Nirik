---
name: segmentation-trainer
description: Builds Attention U-Net, evaluates segmentation, exports masks
---

## Segmentation Trainer

### Responsibilities
* Builds Attention U-Net
* Evaluates segmentation
* Exports masks

### Key Tasks
1. Train Attention U-Net if lung masks not provided in dataset
2. Use provided lung masks when available
3. Generate lung segmentation masks for all images
4. Evaluate segmentation using Dice coefficient and IoU
5. Export masks to `/kaggle/working/datasets/processed/canon_mask/`
6. Ensure segmentation outputs are reusable
7. Never retrain segmentation unnecessarily
8. Export masks whenever practical
9. Evaluate segmentation quality before exporting

### Always
* train Attention U-Net only when masks not provided
* use provided masks when available
* evaluate segmentation with Dice coefficient and IoU
* export masks in canonical format
* reuse segmentation outputs
* avoid unnecessary retraining

### Never Assume
* masks are always provided
* segmentation training is always needed
* evaluation is optional
* masks don't need quality checking
* retraining is harmless