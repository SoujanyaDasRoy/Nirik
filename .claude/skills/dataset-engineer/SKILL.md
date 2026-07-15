# Dataset Engineer

## Purpose
The Dataset Engineer skill ensures that all datasets used in the NIRikhshon project are discoverable, validated, cleaned, versioned, and prepared according to the canonical architecture and engineering principles. This skill guarantees that downstream notebooks (lung segmentation, TB classification, explainability) receive reproducible, leak-proof, and scientifically justified dataset splits.

## Mission
To establish a rigorous, auditable, and reproducible dataset preparation pipeline that adheres to medical AI research standards, prevents data leakage, and supports explainability validation by maintaining dataset provenance and integrity throughout the project lifecycle.

## Responsibilities
- Discover and inventory all raw datasets according to their canonical purpose (lung segmentation, TB classification, explainability validation, external evaluation).
- Verify dataset integrity through checksums, file readability, and format validation when available.
- Detect and handle duplicate images and duplicate patients (where patient-level metadata exists) to prevent data leakage.
- Identify and log corrupted images, missing files, and invalid formats without silent ignoring.
- Generate comprehensive metadata including image properties, labels, dataset origin, and preprocessing history.
- Compute dataset statistics (class balance, resolution distribution, aspect ratios) to inform experimental design.
- Perform label verification and normalization to ensure consistency across datasets (e.g., mapping all TB-positive labels to 'TB', normal to 'Normal').
- Create train/validation/test splits that prevent leakage across patients and datasets, using fixed random seeds for reproducibility.
- Implement cross-dataset strategy where appropriate (e.g., using Shenzhen for both segmentation and classification while recording provenance).
- Export standardized CSV manifests and JSON version files for traceability.
- Log all operations for auditability and reproducibility.
- Enforce configuration-driven parameters (image size, split ratios) rather than hardcoded values.
- Validate that exported datasets meet quality gates before handoff to downstream notebooks.

## Responsibilities Explicitly Out of Scope
- Model training or architecture modification (U-Net, DenseNet, etc.)
- Backend API development or frontend UI implementation
- Grad-CAM generation or explainability computation
- Clinical report generation or observation creation
- Any implementation involving neural networks, loss functions, or optimization algorithms
- Modification of existing notebook code beyond dataset preparation inputs

## Primary Notebook
- Notebook 1 (Dataset Preparation)

## Secondary Notebooks
- Notebook 2 (Segmentation) - provides segmentation metadata
- Notebook 3 (Classification) - provides classification metadata
- Notebook 4 (Explainability & Evaluation) - provides explainability metadata

## When This Skill Should Be Used
- At the start of any experiment requiring dataset preparation
- When new datasets are added to the project
- Before running Notebook 1 (Dataset Preparation) to ensure raw data is ready
- When reproducing prior experiments to verify dataset integrity
- When updating dataset versions or applying new preprocessing pipelines
- During external evaluation setup to prevent training set contamination
- Prior to any cross-dataset validation or generalization testing

## Required Inputs
- Root directory path to the project (containing `datasets/` folder)
- Dataset source information (e.g., Kaggle, local storage, provided paths)
- Expected dataset names and their canonical purposes (from CLAUDE.md)
- Configuration parameters (image size, train/val/test split ratios, random seed)
- Optional: checksum files or hashes for download verification
- Optional: patient-level metadata files for duplicate patient detection

## Expected Outputs
- `datasets/processed/master_metadata.csv`: Comprehensive metadata for all images
- `datasets/processed/dataset_statistics.csv`: Per-dataset and aggregate statistics
- `datasets/processed/segmentation_metadata.csv`: Metadata for lung segmentation datasets
- `datasets/processed/heatmap_metadata.csv`: Metadata for explainability validation (TBX11K)
- `datasets/splits/train.csv`, `validation.csv`, `test.csv`: Split files with columns: `image_path,label,dataset_origin,patient_id (if available)`
- `datasets/manifest/dataset_manifest.json`: File-level manifest with hashes and metadata
- `datasets/manifest/dataset_version.json`: Version tracking including sources, dates, preprocessing steps
- `logs/dataset_preparation_<timestamp>.log`: Detailed log of all operations
- `docs/dataset_preparation_report.json`: Summary report for audit

## Dependencies
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, Pillow/OpenCV (for image validation), hashlib, json, csv, os, sys
- Access to raw dataset storage (local or remote)
- Git for version tracking of preparation scripts (optional but recommended)

## Workflow
1. **Initialization**
   - Load configuration (split ratios, image size, random seed, dataset paths)
   - Create necessary output directories (`datasets/processed/`, `datasets/splits/`, `datasets/manifest/`, `logs/`, `docs/`)
   - Initialize log file with timestamp and configuration dump

2. **Dataset Discovery**
   - Scan `datasets/raw/` for expected datasets (Montgomery, Shenzhen, Jaypee, TBX11K)
   - For each dataset, verify presence of images and labels (if separate)
   - Record dataset origin, download date, and source URL in version manifest
   - If checksums available, verify file integrity; log mismatches and halt if critical

3. **Dataset Validation**
   - For each image in raw datasets:
     - Attempt to read image file; log and skip if corrupted/unreadable
     - Validate image format (PNG, JPG, TIFF, DICOM as per dataset specs)
     - Check image dimensions and aspect ratio; flag extreme outliers
     - Verify label existence and readability (if labels in separate files)
     - Normalize labels to canonical set: {'TB', 'Normal'} for classification, {'lung_mask', 'no_lung_mask'} for segmentation (if applicable)
   - Generate per-image metadata: path, filename, dataset, label, width, height, aspect ratio, file size, checksum (if computed)

4. **Dataset Cleaning**
   - Remove entries for corrupted/unreadable images from metadata
   - Log all skipped images with reason
   - Detect duplicate images via perceptual hashing (or exact hash if lossless) within and across datasets
   - Where patient metadata exists (e.g., filename contains patient ID), detect duplicate patients across datasets and splits
   - For duplicates: keep one instance (prefer higher quality or first occurrence), log all removals

5. **Metadata Generation**
   - Aggregate per-image metadata into `master_metadata.csv`
   - Compute dataset-level statistics: count per label, resolution distribution, file type breakdown
   - Save `dataset_statistics.csv` with columns: `dataset,label,count,mean_width,mean_height,std_width,std_height,min_area,max_area`
   - Generate specialization metadata:
     - `segmentation_metadata.csv`: only for Montgomery and Shenzhen (lung segmentation purpose)
     - `heatmap_metadata.csv`: only for TBX11K (explainability validation purpose)

6. **Dataset Versioning**
   - Create `dataset_version.json` containing:
     - `version_id`: UTC timestamp of preparation
     - `datasets`: list of dictionaries with `name, source, download_date, checksum_verified, image_count, label_distribution`
     - `preprocessing_steps`: list of applied steps (e.g., ['resize_to_224x224', 'normalize_0_1'])
     - `random_seed`: used for splits
     - `configuration_snapshot`: copy of input config
   - Save manifest with per-file hashes for future change detection

7. **Label Verification**
   - Cross-check labels against source documentation (e.g., JP2000 dataset labels)
   - Ensure consistent mapping: all TB-positive -> 'TB', all normal -> 'Normal'
   - For segmentation datasets, verify lung masks are binary and non-empty where expected
   - Log any label inconsistencies or missing labels for manual review

8. **Train Validation Test Split**
   - Shuffle images using fixed random seed from configuration
   - Stratify split by label and dataset origin to maintain distribution
   - Enforce patient-level separation: if patient IDs available, ensure no patient appears in more than one split
   - For external evaluation: hold out entire Jaypee test set if specified, or use Montgomery as unseen
   - Generate split CSV files with columns: `image_path,label,dataset_origin,patient_id (if available)`
   - Verify split integrity: no overlap, correct proportions, no leakage

9. **Cross Dataset Strategy**
   - For datasets used in multiple purposes (e.g., Shenzhen in segmentation and classification):
     - Record dual usage in version manifest
     - Ensure splits are independent per purpose (different random seeds or stratified separately)
     - Never use the same image for both segmentation training and classification testing without explicit design

10. **Data Leakage Prevention**
    - Verify no image appears in more than one split (train/val/test)
    - Verify no patient appears in more than one split (if patient IDs available)
    - Ensure external evaluation sets (Jaypee hold-out, Montgomery) are completely excluded from training
    - Log any potential leakage detected during split generation

11. **Class Balance**
    - Compute class ratios per split and overall
    - Log imbalance ratios; recommend stratified sampling or weighting in downstream notebooks
    - Never apply oversampling/undersampling at dataset preparation stage (to avoid leakage); leave to training stage with proper validation

12. **Folder Structure**
    - Organize processed datasets by purpose:
      - `datasets/processed/segmentation/` (for U-Net training)
      - `datasets/processed/classification/` (for DenseNet training)
      - `datasets/processed/explainability/` (for TBX11K Grad-CAM validation)
    - Maintain original raw data untouched in `datasets/raw/`
    - Splits stored in `datasets/splits/`
    - Manifests and versions in `datasets/manifest/`

13. **File Naming Standards**
    - Use descriptive, lowercase, underscore-separated names
    - Include dataset name and purpose: `shenzhen_train.csv`, `montgomery_validation.csv`
    - Avoid spaces, special characters, or non-alphanumeric except underscores and hyphens
    - Timestamp logs: `dataset_preparation_YYYYMMDD_HHMMSS.log`

14. **Reproducibility**
    - Fix random seed for all stochastic operations (shuffling, splitting)
    - Record exact versions of dependencies in configuration
    - Ensure same input raw data + same configuration = same output splits
    - Never rely on implicit ordering of file systems; always sort file lists

15. **Logging**
    - Log every major step with timestamps
    - Record counts: total images, valid images, corrupted images, duplicates removed
    - Log split statistics per dataset and label
    - Save logs to both console and file for audit

16. **Configuration**
    - Store parameters in JSON/YAML: `config/dataset_preparation_config.json`
    - Example parameters:
      ```json
      {
        "image_size": [224, 224],
        "split_ratios": [0.7, 0.15, 0.15],
        "random_seed": 42,
        "datasets": {
          "montgomery": {"purpose": ["segmentation", "external_evaluation"]},
          "shenzhen": {"purpose": ["segmentation", "classification"]},
          "jaypee": {"purpose": ["classification", "external_evaluation_holdout"]},
          "tbx11k": {"purpose": ["explainability_validation"]}
        },
        "normalize_to": [0, 1],
        "check_scan_for_corrupted": true,
        "remove_duplicates": true,
        "patient_id_extraction_regex": "([A-Za-z0-9]+)_"
      }
      ```
    - Never hardcode values in scripts; always read from configuration

17. **Error Handling**
    - Fail fast on critical errors (e.g., missing raw dataset directory) with clear error messages
    - Warn and skip non-critical issues (single corrupted image) but log extensively
    - Ensure pipeline continues after non-fatal errors where possible
    - Require manual intervention for label inconsistencies or missing patient metadata when critical for leakage prevention

18. **Performance Considerations**
    - Use incremental hashing for large datasets to avoid memory overload
    - Process images in batches for validation
    - Cache image properties (dimensions) to avoid repeated I/O
    - Parallelize independent operations (e.g., per-image validation) where safe
    - Avoid loading entire datasets into memory; use iterators and generators

19. **Documentation Requirements**
    - Update `docs/DATASET_PREPARATION.md` with:
      - Purpose and scope
      - Configuration used
      - Step-by-step procedure
      - Known issues and assumptions
      - Output file descriptions
    - Reference this skill in notebook preconditions
    - Ensure all output files are self-describing (CSV headers, JSON schemas)

20. **Quality Checklist** (Must pass before handoff)
    - [ ] No duplicate images in master metadata
    - [ ] No duplicate patients across train/test splits (if patient IDs available)
    - [ ] No corrupted or unreadable images in processed sets
    - [ ] All labels verified and normalized to canonical set
    - [ ] Train/val/test splits sum to 100% with correct stratification
    - [ ] Random seed recorded and fixed
    - [ ] All expected output files generated and non-empty
    - [ ] Configuration saved with outputs
    - [ ] Log file complete without critical errors
    - [ ] Manifest hashes match current file states
    - [ ] Dataset statistics computed and saved

21. **Common Mistakes**
    - Assuming labels are correct without verification against source documentation
    - Silently ignoring corrupted images leading to training failures
    - Using different preprocessing for training vs inference (violates pipeline contract)
    - Hardcoding image paths or split ratios
    - Merging datasets without recording provenance in version manifest
    - Creating splits that leak patients across train and test
    - Using timestamps or non-deterministic operations in split generation
    - Forgetting to stratify splits leading to severe class imbalance in one split
    - Not validating image readability before attempting to load in training

22. **Never Do**
    - Never hardcode dataset paths, image sizes, or split ratios
    - Never silently skip corrupted images without logging
    - Never merge datasets without explicit provenance tracking
    - Never assume labels from different datasets are compatible without verification
    - Never use non-fixed random seeds for splits
    - Never allow the same patient to appear in both training and test sets
    - Never modify raw data; always work on copies or derive processed versions
    - Never report accuracy alone; always prepare for full metric computation
    - Never skip label verification step
    - Never exclude dataset preparation from version control (scripts and configs should be tracked)

23. **Deliverables**
    The Dataset Engineer skill delivers a fully prepared, validated, and versioned dataset pipeline that enables:
    - Notebook 1 to verify its inputs are ready
    - Notebook 2 to load segmentation-ready images and masks without further cleaning
    - Notebook 3 to load classification-ready segmented images with consistent labels
    - Notebook 4 to validate Grad-CAM against a known, clean dataset
    - External evaluators to reproduce exact splits using configuration and seed
    - Auditors to trace every image from raw source to final split via manifest
    - Researchers to trust that reported metrics are not compromised by leakage or poor preparation

    Specifically, the skill provides:
    - Reproducible train/validation/test splits
    - Comprehensive metadata for debugging and analysis
    - Verified dataset integrity free of duplicates and corruption
    - Clear documentation of preprocessing and configuration
    - Leakage-proof design adhering to medical AI research standards