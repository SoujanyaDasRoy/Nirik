---
name: model-trainer
description: Builds EfficientNetV2-M (teacher) and NirikNet (student), performs transfer learning, performs knowledge distillation, exports trained models
---

## Model Trainer

### Responsibilities
* Builds EfficientNetV2-M (teacher model)
* Builds NirikNet (student model)
* Performs transfer learning
* Performs knowledge distillation
* Exports trained models

### Key Tasks
1. Load or create EfficientNetV2-M teacher model with ImageNet pretrained weights
2. Load or create NirikNet student model with specified architecture
3. Freeze teacher backbone initially, then unfreeze top 20% (excluding BatchNorm)
4. Implement knowledge distillation with:
   * Temperature = 4
   * Alpha = 0.5 (balance between hard and soft loss)
   * Class weights for imbalance
   * Gradient clipping (clipnorm=1.0)
   * Mixed Precision (mixed_float16)
   * AdamW optimizer
   * Warm-up cosine decay learning rate
5. Train models using preprocessed data from Notebook 2
6. Evaluate models on validation set
7. Save best models based on validation AUC
8. Export trained models:
   * attention_unet.keras
   * teacher_efficientnetv2m.keras
   * niriknet_best.keras
   * niriknet.keras (final model)
9. Generate training history and metrics files
10. Ensure teacher fine-tunes only upper layers initially
11. Implement proper layer-wise unfreezing strategies for teacher

### Always
* use mixed precision (float16)
* apply gradient clipping (clipnorm=1.0)
* use AdamW optimizer
* implement warm-up cosine decay learning rate
* set temperature = 4 for knowledge distillation
* set alpha = 0.5 for loss balancing
* freeze teacher backbone initially then unfreeze top 20% (excluding BatchNorm)
* save best models based on validation AUC
* export all required model files
* generate training history and metrics

### Never Assume
* preprocessing matches between training and inference
* knowledge distillation hyperparameters are optional
* teacher backbone should be fully trainable from start
* student architecture can vary from specification
* evaluation can skip validation set
* model Export doesn't require history/metrics