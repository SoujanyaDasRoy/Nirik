# Nirikshon TB Detection Project -- Claude Handoff

## Project Overview

This project is an AI-assisted pulmonary tuberculosis screening system.

Pipeline:

1.  Lung segmentation using an Attention U-Net.
2.  Image preprocessing (CLAHE + lung crop).
3.  Teacher model: ResNet-50.
4.  Student model: Custom CNN trained via knowledge distillation.
5.  High-resolution Grad-CAM++ / LayerCAM / EigenCAM explanations.
6.  Frontend (Next.js/Vercel) + backend (Flask on local PC).

Goal: - Produce a clinically useful TB classifier with explainability. -
Later integrate an LLM that interprets the CNN output and Grad-CAM for
doctors.

------------------------------------------------------------------------

# Current Dataset

Datasets:

-   Chest X-rays Tuberculosis from India
-   Montgomery
-   Shenzhen
-   TB Chest Radiography Database
-   TBX11K Simplified
-   Chest Xray Masks and Labels

Current issue:

Cell 2 is inferring labels from folder names.

This is incorrect.

The datasets already provide metadata CSVs.

Use:

-   jaypee_metadata.csv
-   montgomery_metadata.csv
-   shenzhen_metadata.csv
-   data.csv (TBX11K)

Build lookup dictionaries:

    filename -> label

and only use folder names for the TB Chest Radiography Database.

Expected result:

Unknown (-1) images should reduce dramatically.

------------------------------------------------------------------------

# Cell-by-cell changes

## Cell 2 (Highest priority)

Rewrite completely.

Tasks:

-   Read all metadata CSVs.
-   Create lookup dictionaries.
-   Match images using filenames.
-   Remove the current infer_label() implementation.
-   Keep mask pairing logic.
-   Keep dataframe format identical so later cells continue working.

------------------------------------------------------------------------

## Cell 3

Keep preprocessing.

Ensure:

-   CLAHE
-   Lung crop
-   8.5% padding

These SAME operations must also be used during inference.

------------------------------------------------------------------------

## Cell 5

No major architecture changes.

Verify:

-   8.5% crop padding
-   Mask pairing
-   Dataset statistics

------------------------------------------------------------------------

## Cell 6

Fix mixed precision.

Problem:

float16 × float32 Dice loss crash.

Solution:

-   Cast y_true and y_pred to float32 inside dice() and dice_bce().
-   Output layer should use dtype="float32".
-   Train U-Net under float32 policy.
-   Restore mixed_float16 afterwards.

------------------------------------------------------------------------

## Cell 7

Teacher model:

ResNet50.

Use:

tf.keras.applications.resnet50.preprocess_input

No rescaling to \[0,1\].

------------------------------------------------------------------------

## Cell 8

Student CNN.

Architecture:

Conv64

↓

128

↓

256

↓

256

↓

512

Classifier:

GlobalAveragePooling

↓

Dense(512)

↓

BN

↓

Dropout

↓

Dense(256)

↓

BN

↓

Dropout

↓

Dense(2)

Teacher and student MUST use identical preprocessing.

Grad-CAM helper:

Fix broadcasting bug by removing batch dimension from conv_output before
Grad-CAM++ calculations.

------------------------------------------------------------------------

## Cell 9

Knowledge distillation.

Keep:

-   Cosine LR schedule
-   AdamW
-   Early stopping
-   Checkpointing

Fix:

-   Use validation set for Youden threshold.
-   Never compute threshold on test set.
-   Use TensorFlow CategoricalFocalCrossentropy(from_logits=True)
    instead of custom focal loss if possible.
-   Keep ROC, PR, confusion matrix, AUC, sensitivity, specificity, F1.

------------------------------------------------------------------------

## Cell 10

Inference.

Must exactly match training:

-   CLAHE
-   Lung crop
-   8.5% padding
-   preprocess_input()

No preprocessing mismatch.

------------------------------------------------------------------------

# Remaining bugs

1.  Cell 2 metadata-driven labeling.
2.  Cell 6 mixed precision Dice loss.
3.  Cell 8 Grad-CAM++ broadcasting bug.
4.  Verify TBX11K image discovery.
5.  Verify final unknown labels are near zero.

------------------------------------------------------------------------

# Final objective

Produce a publication-quality TB screening pipeline suitable for a
final-year B.Tech project.

Target outputs:

-   AUC
-   ROC
-   PR Curve
-   Confusion Matrix
-   Sensitivity
-   Specificity
-   Precision
-   Recall
-   F1-score
-   Grad-CAM++ / LayerCAM / EigenCAM
-   Saved best student model
-   Deployment-ready inference pipeline with preprocessing identical to
    training.
