# Segmentation Engineer

## Purpose
The Segmentation Engineer skill ensures the design, training, validation, and deployment of lung segmentation models that accurately identify lung fields in chest X-rays while preserving anatomical boundaries, producing high-quality segmentation masks suitable for downstream tuberculosis classification tasks, and adhering to medical AI principles of correctness, reproducibility, and explainability.

## Mission
To establish a rigorous, auditable, and scientifically justified lung segmentation pipeline that develops models capable of robust lung field extraction, minimizes shortcut learning by focusing on anatomical structures rather than image artifacts, and provides reliable inputs for TB classification and explainability components, thereby supporting accurate and clinically useful AI-assisted tuberculosis screening.

## Responsibilities
- Design and implement U-Net and Attention U-Net architectures with justification for encoder/decoder choices, skip connections, and bottleneck design based on lung segmentation literature
- Select and justify segmentation loss functions (Dice, Binary Cross Entropy, Tversky, Focal Tversky) based on class imbalance, boundary precision, and overlap optimization requirements
- Develop training strategies including transfer learning from medical imaging datasets, learning rate scheduling, early stopping, and model checkpointing with reproducibility controls
- Implement validation strategies that compute segmentation-specific metrics (Dice coefficient, IoU, pixel accuracy, boundary accuracy) and visualize failure cases
- Generate and post-process lung masks using connected component analysis, morphological operations, and lung-preserving refinement to ensure anatomical integrity
- Export segmentation models in standardized format (keras) with accompanying metadata, training history, and configuration for reproducibility
- Ensure masks are saved in lossless format with consistent naming and directory structure for seamless integration with Notebook 3 (TB Classification)
- Validate segmentation quality against anatomical plausibility (lung shape, symmetry, diaphragm position) and quantitative metrics
- Prevent data leakage by ensuring strict separation of training, validation, and test sets at the patient level
- Log all training hyperparameters, augmentation parameters, and experimental configurations for auditability
- Collaborate with Dataset Engineer to receive validated, split datasets and with Computer Vision Engineer to ensure preprocessing consistency
- Never perform TB classification, explainability computation, medical diagnosis, backend/frontend development, or dataset preparation tasks
- Never hardcode architecture parameters, image sizes, or loss function weights; always use configuration-driven, justified values
- Never skip validation steps or accept poor segmentation without investigation and documentation
- Never allow downstream models to consume invalid masks without explicit warnings and quality flags

## Responsibilities Explicitly Out of Scope
- Tuberculosis classification (DenseNet training, evaluation, or inference)
- Gradient-weighted Class Activation Mapping (Grad-CAM) or explainability computations
- Medical diagnosis, report generation, or clinical decision making
- Backend API development (Flask, Django, etc.)
- Frontend UI development (React, Next.js, etc.)
- Dataset discovery, validation, cleaning, or metadata generation (handled by Dataset Engineer)
- Core image processing operations (resize, normalization, CLAHE) – guided by but not the responsibility of this role
- Any operation involving neural network weights for classification or explainability

## Primary Notebook
- Notebook 2 (Segmentation)

## Secondary Notebooks
- Notebook 1 (Dataset Preparation) - provides preprocessed images and splits
- Notebook 3 (Classification) - consumes segmentation masks
- Notebook 4 (Explainability & Evaluation) - uses masks for localization

## When This Skill Should Be Used
- When initiating lung segmentation model development for the first time
- When reproducing prior segmentation experiments to verify consistency
- When evaluating new architectures (e.g., Attention U-Net) against baseline U-Net with justification
- When preparing segmentation masks for Notebook 3 (TB Classification) input
- When validating segmentation quality for external evaluation datasets (Montgomery, Jaypee hold-out)
- When investigating degradation in downstream TB classification performance potentially linked to segmentation errors
- When deploying segmentation models to ensure training/inference preprocessing consistency
- When documenting segmentation methodology for research papers or technical reports
- When updating segmentation configuration based on validation findings or literature review

## Required Inputs
- Split dataset outputs from Dataset Engineer:
  * `datasets/splits/train.csv`, `validation.csv`, `test.csv` with columns: `image_path, label, dataset_origin, patient_id (if available)`
  * `datasets/processed/segmentation/` directory containing preprocessed chest X-ray images (from Computer Vision Engineer)
  * Optional: `datasets/processed/segmentation/masks/` directory containing ground truth lung masks (if available from dataset)
- Configuration file (`config/segmentation_config.json`) specifying:
  * Model architecture (U-Net or Attention U-Net) with justification
  * Encoder backbone (if using transfer learning) with justification and source
  * Input image dimensions (width, height) with justification (must match Computer Vision Engineer output)
  * Output mask dimensions (should match input unless justified otherwise)
  * Loss function(s) and weights with justification for choice and weighting
  * Optimizer type (AdamW, SGD, etc.) with learning rate and justification
  * Learning rate scheduler (ReduceLROnPlateau, CosineAnnealing, etc.) with justification
  * Batch size with justification based on GPU memory and convergence behavior
  * Number of epochs with justification and early stopping patience
  * Data augmentation parameters (rotation, zoom, etc.) with justification and anatomical constraints
  * Post-processing thresholds (connected component size, morphological kernel size) with justification
  * Random seeds for reproducibility (numpy, tensorflow, python hash)
  * Model export path and naming convention
  * Validation metrics to compute and threshold for acceptable performance
- Optional: pre-trained encoder weights (if using transfer learning) with source and justification
- Optional: ground truth lung masks for validation if not included in dataset splits
- Hardware specifications (GPU memory, CPU cores) for batch size justification

## Expected Outputs
- Trained segmentation model: `exports/segmentation/unet_best_model.keras` (or attention_unet variant)
- Training history: `exports/segmentation/training_history.csv` (loss, metrics per epoch)
- Validation metrics: `exports/segmentation/validation_metrics.json` (Dice, IoU, etc. per dataset split)
- Segmentation masks: `exports/segmentation/predicted_masks/` directory containing:
  * `[original_filename]_mask.png` for each input image in validation/test sets
  * Saved as lossless PNG with consistent naming
- Mask metadata: `exports/segmentation/mask_metadata.csv` with columns: `image_path, mask_path, dice_score, iou_score, pixel_accuracy, lung_area_pixels, lung_symmetry_score, diaphragm_position_flag, connected_components_count, largest_component_ratio`
- Configuration snapshot: `exports/segmentation/segmentation_config_used.json` copy of input config with any runtime additions
- Log file: `logs/segmentation_training_<timestamp>.log` detailing:
  * Epoch-level training and validation metrics
  * Learning rate changes
  * Early stopping triggers
  * Checkpoint saving events
  * Any warnings or errors (NaN loss, gradient explosion)
- Experiment manifest: `exports/segmentation/experiment_manifest.json` containing:
  * Git commit hash of training code (if applicable)
  * Dataset versions used (from Dataset Engineer manifests)
  * Preprocessing pipeline version (from Computer Vision Engineer manifests)
  * Random seeds and hardware information
  * Training start/end timestamps
- Quality report: `docs/segmentation_quality_report.json` summarizing:
  * Mean and std of Dice/IoU across validation/test
  * Failure case analysis (examples of poor segmentation with lung boundary errors)
  * Comparison to baseline U-Net if Attention U-Net tested
  * Recommendations for architecture or hyperparameter adjustments

## Dependencies
- Python 3.8+
- Libraries:
  * TensorFlow >= 2.8.0 or PyTorch >= 1.10.0 (specify in config; Skill assumes TensorFlow/Keras unless justified otherwise)
  * NumPy >= 1.20.0
  * Pandas >= 1.3.0
  * scikit-image >= 0.19.0 (for metrics, morphological operations)
  * OpenCV >= 4.5.0 (for connected component analysis, masking)
  * tqdm >= 4.60.0 for progress logging
  * matplotlib >= 3.5.0 for visualization (optional but recommended)
  * yaml or json for configuration
  * logging for structured logs
  * hashlib for file checksums
  * gitpython (optional) for commit tracking
- Access to processed images and split CSV files from prior skills
- Write access to `exports/segmentation/`, `logs/`, `docs/` directories
- GPU with sufficient memory for batch size (or configuration for CPU fallback)

## Workflow
1. **Initialization**
   - Load segmentation configuration file with justification for every parameter
   - Create output directory structure: `exports/segmentation/`, `exports/segmentation/predicted_masks/`, `logs/`, `docs/`
   - Initialize training log with timestamp, configuration dump, TensorFlow/PyTorch version, and hardware info
   - Set random seeds for numpy, tensorflow, and python hash for reproducibility
   - Load and verify dataset split CSV files and preprocessed image directories from Dataset Engineer and Computer Vision Engineer outputs

2. **Data Loading and Preparation**
   - Read train/validation/test CSV files to get image paths and metadata
   - Verify that all listed images exist in the preprocessed segmentation directory (`datasets/processed/segmentation/`)
   * If ground truth masks are provided in dataset paths (e.g., from Montgomery/Shenzhen mask datasets), load them; otherwise, note that masks must be generated separately (not in scope for this skill)
   * Create TensorFlow/PyTorch data generators or datasets that:
     * Yield batches of preprocessed images and corresponding ground truth masks
     * Apply configured data augmentation ONLY during training (never validation/test)
     * Ensure patient-level separation: if patient_id available in CSV, verify no patient appears in more than one split
   - Log dataset sizes, class balance (lung vs background pixels), and any missing files

3. **Model Architecture Definition**
   - **U-Net Baseline**:
     * Encoder: Series of convolutional blocks (typically 2x3x3 conv per level) with max pooling for downsampling
       * Justification: Captures hierarchical features; doubling filters per level is standard for feature richness
     * Bottleneck: Two convolutional blocks at lowest resolution
       * Justification: Processes most abstract features before upsampling
     * Decoder: Series of upsampling (transpose conv or upsample+conv) followed by convolutional blocks
       * Justification: Recovers spatial resolution while combining contextual and local information
     * Skip Connections: Concatenate encoder feature maps to decoder at corresponding resolution
       * Justification: Preserves fine-grained spatial details lost in downsampling, critical for boundary precision
     * Final Layer: 1x1 convolution with sigmoid activation for binary mask prediction
       * Justification: Produces per-pixel probability of lung membership
   - **Attention U-Net Enhancement** (if justified in config):
     * Adds attention gates between encoder and decoder skip connections
       * Justification: Focuses decoder on relevant encoder features, suppresses irrelevant regions (e.g., background noise), improves segmentation of small/low-contrast structures
     * Attention Gate Computation: 
       * `alpha = sigma(W_g * g + W_x * x + b)` where g is gating signal (decoder upsampled), x is skip connection
       * Output: `alpha * x` to modulate skip connection
       * Justification: Learns to weigh skip connection features by relevance to current decoder layer
   - **Encoder Backbone Options** (if transfer learning justified):
     * If using pretrained encoder (e.g., from ImageNet or medical imaging dataset):
       * Freeze early layers, train later layers
       * Justification: Leverages learned features while adapting to lung-specific textures
     * Must validate that pretrained weights are compatible with input channels (grayscale -> repeat or adapt first layer)
   - **Input Handling**:
     * Grayscale images: ensure single channel input; if Computer Vision Engineer output is 3-channel (e.g., for visualization), convert to grayscale or use first channel
     * Normalize input to [0,1] or z-score as per configuration (must match preprocessing pipeline to avoid leakage)
   - Log architecture diagram (textual) with layer names, output shapes, and parameter counts

4. **Loss Function Selection and Implementation**
   - **Binary Cross Entropy (BCE)**:
     * Formula: `- [y * log(p) + (1-y) * log(1-p)]`
     * Justification: Standard for binary classification; penalizes confident wrong predictions
     * Limitation: Sensitive to class imbalance (lung typically <50% of image)
   - **Dice Loss**:
     * Formula: `1 - (2 * intersection + smooth) / (|pred| + |gt| + smooth)`
     * Justification: Directly optimizes overlap metric; robust to class imbalance
     * Note: Non-divergent when pred and gt are empty; smooth term prevents division by zero
   - **Combined Dice + BCE**:
     * Formula: `weights[0] * BCE + weights[1] * Dice_Loss`
     * Justification: Combines region-based (Dice) and pixel-based (BCE) optimization for better boundary and overlap
     * Weights justified (e.g., 0.5 each) or tuned via validation
   - **Tversky Loss** (for imbalance sensitivity):
     * Formula: `1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)`
     * where TP, FP, FN are true positives, false positives, false negatives
     * alpha, beta control FP/FN penalty; alpha=beta=0.5 reduces to Dice
     * Justification: Allows tuning for precision/recall tradeoff; useful when false lung inclusions (FP) are more detrimental than exclusions (FN) or vice versa
   - **Focal Tversky Loss** (for hard example focus):
     * Formula: `(1 - Tversky)^gamma`
     * gamma > 0 focuses on hard-to-segment examples
     * Justification: Down-weights easy examples, focuses learning on challenging boundaries
   - Log chosen loss function(s) with mathematical formula, justification, and any hyperparameters (smooth, alpha, beta, gamma)
   - Implement loss as Keras custom loss or PyTorch loss module; verify reduction over batch

5. **Training Strategy**
   - **Optimizer**:
     * AdamW: Adam with weight decay decoupled from gradient update
       * Justification: Better regularization than standard Adam; weight decay applied correctly
     * Log learning rate, beta1, beta2, epsilon, weight decay with justification
   - **Learning Rate Scheduler**:
     * ReduceLROnPlateau: reduce LR when validation metric plateaus
       * Justification: Prevents overshooting minimum, allows finer convergence
     * CosineAnnealing: cosine annealing from max to min LR over epochs
       * Justification: Smooth LR reduction, can escape local minima
     * Log scheduler type, parameters, and justification
   - **Early Stopping**:
     * Monitor validation Dice coefficient (or loss) with patience (e.g., 10-20 epochs)
     * Justification: Prevents overfitting, saves best model
     * Log monitor metric, patience, and min_delta
   - **Model Checkpointing**:
     * Save model weights when validation Dice improves
     * Justification: Preserves best performing model; avoids saving overfitted epochs
     * Log checkpoint frequency and metric used
   - **Mixed Precision Training** (if justified and hardware supports):
     * Use TensorFlow mixed precision policy or PyTorch AMP (Automatic Mixed Precision)
       * Justification: Speeds up training, reduces memory usage, maintains numerical stability
   - **Data Augmentation** (training only):
     * Geometric: rotation (±15°), zoom (0.9-1.1), translation (±10% width/height)
       * Justification: Simulates patient positioning variation; constrained to avoid anatomical impossibility
     * Intensity: brightness (±0.1), contrast (±0.1) in normalized range
       * Justification: Simulates exposure variation; avoid clipping
     * *Never*: horizontal flip (laterality significant), vertical flip (inverts anatomy), shear/elastic distortion (creates impossible anatomy)
     * Log augmentation parameters and confirmation they are applied only to training batches
   - Log epoch-level training loss, validation loss, validation Dice, validation IoU, learning rate

6. **Validation Strategy**
   - **Per-Epoch Validation**:
     * Run validation set after each training epoch (or every N epochs)
     * Compute metrics without augmentation
     * Log validation loss, Dice coefficient, IoU, pixel accuracy, boundary accuracy
   - **Metric Definitions**:
     * Dice Coefficient: `2 * |A ∩ B| / (|A| + |B|)` where A=prediction, B=ground truth
       * Justification: Standard overlap metric for segmentation; sensitive to false positives/negatives
     * IoU (Jaccard Index): `|A ∩ B| / |A ∪ B|`
       * Justification: Measures overlap relative to union; stricter than Dice
     * Pixel Accuracy: `(TP + TN) / (TP + TN + FP + FN)`
       * Limitation: misleading with high class imbalance; log but not primary
     * Boundary Accuracy: average distance prediction boundary to ground truth boundary
       * Justification: Captures edge precision; requires skeletonization or distance transform
   - **Failure Case Analysis**:
     * Save validation images where Dice < threshold (e.g., 0.8) for manual review
     * Overlay prediction and ground truth to visualize errors (under-segmentation, over-segmentation, boundary shift)
     * Log common failure patterns (e.g., missed apical regions, diaphragm confusion, heart shadow inclusion)
   - **Anatomical Plausibility Checks** (optional but recommended):
     * Lung Symmetry: compare left/right lung area ratio; flag if >1.5 or <0.67 (indicating possible laterality error or severe pathology)
     * Diaphragm Position: estimate diaphragm curve; flag if abnormally high/low
     * Lung Area: compare to expected range based on image size and typical lung fraction
     * Connected Components: lungs should be one or two components (left/right); flag excessive fragments
   - After training, evaluate final best model on test set (if available) with same metrics
   - Log test set metrics and save predictions to `exports/segmentation/predicted_masks/`

7. **Mask Post-processing**
   - **Connected Component Analysis**:
     * Label connected components in binary prediction (using threshold 0.5)
     * Remove components smaller than min_size (e.g., 100 pixels) as likely noise
       * Justification: Eliminates spurious small predictions not corresponding to lungs
     * Keep largest N components (typically N=2 for left/right lungs)
       * Justification: Handles cases where prediction includes extra fragments; preserves anatomical expectation
   - **Morphological Operations**:
     * Optional: opening (erosion then dilation) to remove small false positives
       * Kernel size justified (e.g., 2x2 or 3x3) based on pixel spacing and expected noise scale
     * Optional: closing (dilation then erosion) to fill small holes in lung mask
       * Justification: Accounts for partial volume effects or minor prediction gaps
     * Log operations and kernel sizes with justification
   - **Lung-Specific Refinement**:
     * If ground truth available during development, learn post-processing thresholds via validation
     * Never apply post-processing that could remove genuine lung pathology (e.g., large nodules)
     * Validate that post-processing does not significantly alter Dice/IoU on validation set
   - Save final post-processed masks as lossless PNG in `exports/segmentation/predicted_masks/`
   - Generate mask metadata CSV with lung area, symmetry, component count, etc.

8. **Model Export and Documentation**
   - Save best model (by validation Dice) in Keras format: `exports/segmentation/unet_best_model.keras`
   - Include model architecture, weights, and training configuration (if using `model.save()`)
   - Export training history CSV with columns: epoch, loss, val_loss, dice, val_dice, iou, val_iou, lr
   - Export validation/test metrics JSON with per-image and aggregate scores
   - Export configuration used (copy of input config) for reproducibility
   - Create manifest linking model to dataset versions, preprocessing version, and random seeds
   - Write quality report summarizing performance, failure cases, and recommendations

## Engineering Principles
- **Anatomical Fidelity First**: Every architectural decision (encoder depth, skip connections, loss function) must be judged by its ability to preserve lung boundaries and anatomical plausibility, not just by raw metric scores.
- **Reproducibility**: Fixed random seeds, versioned dependencies, and configuration-driven experiments ensure same input yields same output.
- **Leakage Prevention**: Strict patient-level split separation; augmentation applied only to training; normalization statistics computed from training set only.
- **Modularity**: Architecture, loss, optimizer, and training loop are separable components; each can be independently tested or replaced.
- **Evidence-Based Design**: Choices of U-Net over alternatives, loss functions, and hyperparameters must be grounded in medical image segmentation literature or internal validation.
- **Transparency**: All hyperparameters, augmentation parameters, and experimental conditions are logged for audit.
- **Clinical Safety**: Never accept segmentation that removes genuine lung tissue or includes excessive background without investigation; always validate masks anatomically.
- **Resource Awareness**: Justify batch size, encoder depth, and image resolution based on available hardware and convergence behavior.
- **Iterative Improvement**: Use validation failure cases to inform architecture or hyperparameter adjustments; never treat first model as final.

## Lung Segmentation Principles
- **Purpose of Lung Segmentation**: 
  * Reduce shortcut learning in TB classification by focusing model on lung parenchyma rather than background (collars, scapulae, image artifacts)
  * Enable region-of-interest analysis for feature extraction and explainability
  * Normalize input to TB classifier by providing consistent lung-only field of view
  * Facilitate anatomical localization of TB findings (e.g., upper lobe preference)
- **Why U-Net?**:
  * Symmetric encoder-decoder structure captures context at multiple scales
  * Skip connections preserve spatial detail critical for boundary segmentation
  * Proven effective in biomedical image segmentation (original paper: Ronneberger et al., 2015)
  * Relatively lightweight compared to alternatives; trains well on moderate dataset sizes
  * Flexible: easy to modify depth, filter count, or add attention mechanisms
- **Why Skip Connections?**
  * Encoder loses spatial resolution through pooling; decoder recovers it via upsampling but loses fine details
  * Skip connections reroute high-resolution encoder features to decoder, combining semantic context with spatial precision
  * Critical for lung segmentation where boundary accuracy (e.g., diaphragm contour, costophrenic angles) is essential
  * Without skip connections, U-Net becomes a simple encoder-decoder with blurred boundaries
- **Encoder Design Justification**:
  * Progressive downsampling (max pool or stride-2 conv) captures hierarchical features: edges → textures → object parts → lung fields
  * Doubling filters per depth (e.g., 32→64→128→256) increases representational capacity for complex structures
  * Number of depths (typically 4) justified by input size and desired bottleneck resolution (e.g., 224→112→56→28→14)
- **Bottleneck Design**:
  * Processes most abstract features; should be sufficient complexity to distinguish lung vs non-lung
  * Two conv layers at bottleneck allow non-linear combination of multi-scale context
  * Too small bottleneck loses information; too large increases parameters without gain
- **Decoder Design**:
  * Upsampling (transpose conv or nearest+conv) increases spatial resolution
  * Convolution after upsampling refines features using skip connection information
  * Symmetric filter reduction mirrors encoder (e.g., 256→128→64→32) for feature integration
- **Final Activation**:
  * Sigmoid for binary segmentation: outputs probability of lung membership per pixel
  * Threshold at 0.5 for binary mask; can be adjusted based on precision/recall tradeoff from validation

## U-Net Architecture Details
- **Input Layer**: Grayscale image (H, W, 1) - must match preprocessed output from Computer Vision Engineer
- **Encoder Block** (repeated for each depth):
  * Conv2D (filters, 3x3, padding='same', activation='relu')
  * Conv2D (filters, 3x3, padding='same', activation='relu')
  * MaxPooling2D (2x2) - except at bottleneck
  * Justification: Two convs per level allow feature integration; pooling reduces resolution
- **Bottleneck Block**:
  * Conv2D (filters*2, 3x3, padding='same', activation='relu')
  * Conv2D (filters*2, 3x3, padding='same', activation='relu')
  * Justification: Highest feature depth; processes most abstract representation
- **Decoder Block** (repeated for each depth):
  * UpSampling2D (2x2) or Conv2DTranspose (filters, 3x3, strides=2, padding='same')
  * Concatenate with skip connection from encoder
  * Conv2D (filters, 3x3, padding='same', activation='relu')
  * Conv2D (filters, 3x3, padding='same', activation='relu')
  * Justification: Upsampling increases resolution; skip connection provides high-res details; convs refine
- **Output Layer**:
  * Conv2D (1, 1x1, padding='same', activation='sigmoid')
  * Justification: Single channel for lung probability; sigmoid ensures [0,1] output
- **Attention U-Net Addition**:
  * Attention Gate between encoder skip connection and decoder upsampled feature
  * Computes attention coefficients to modulate skip connection
  * Justification: Improves focus on relevant lung regions, suppresses background distractions
- **Parameter Justification**:
  * Initial filters (e.g., 32) based on input channel count and desired feature richness
  * Number of depths (typically 4) based on input size and target bottleneck resolution
  * Filter growth factor (typically 2x) justified by feature complexity increase per downsampling
  * Kernel size (3x3) standard for capturing local context without excessive parameters
  * Activation (ReLU) justified by sparsity and non-linearity benefits; alternatives (LeakyReLU, ELU) require justification
- **Padding Strategy**:
  * 'same' padding ensures spatial dimensions preserved after conv (except pooling)
  * Justification: Avoids information loss at boundaries; critical for maintaining lung edge details
- **Upsampling Method**:
  * Transpose convolution (learned) vs. nearest neighbor + conv (fixed)
  * Justification for transpose conv: learns upsampling filters, reduces checkerboard artifacts with proper initialization
  * Justification for nearest+conv: avoids checkerboard artifacts entirely; may be preferred if transpose conv unstable
  * Log choice with justification
- **Skip Connection Type**:
  * Concatenation (default) vs. addition
  * Justification for concatenation: preserves both encoder and decoder feature information
  * Justification for addition: assumes feature compatibility; less common in segmentation
  * Log choice with justification

## Segmentation Loss Functions
- **Binary Cross Entropy (BCE)**:
  * **Mathematical Intuition**: Measures probability error; minimizes surprise of true label given prediction
  * **Why Appropriate**: Foundation for probabilistic binary classification; well-understood gradients
  * **When to Avoid**: Severe class imbalance without balancing (lung <50% pixels); use weighting or alternative loss
  * **Justification for Use**: Often combined with Dice to capture both region and pixel-level accuracy
- **Dice Loss**:
  * **Mathematical Intuition**: Directly optimizes F1-score surrogate; maximizes overlap between prediction and ground truth
  * **Why Appropriate**: Robust to class imbalance; correlates with overlap metric used for evaluation
  * **Limitations**: Can be unstable when both pred and gt are near zero; mitigated by smooth term
  * **Justification for Use**: Primary choice for segmentation tasks due to overlap optimization
- **Dice + BCE Combination**:
  * **Mathematical Intuition**: Weighted sum of region-based (Dice) and pixel-based (BCE) objectives
  * **Why Appropriate**: Balances global overlap (Dice) with local classification accuracy (BCE)
  * **Weight Justification**: Typically 0.5 each; can be tuned based on validation (e.g., if boundaries poor, increase BCE weight)
- **Tversky Loss**:
  * **Mathematical Intuition**: Generalization of Dice and IoU; allows asymmetric weighting of FP vs FN
  * **Why Appropriate**: When false positives (e.g., including background as lung) are more costly than false negatives (missing lung) or vice versa
  * **Parameter Justification**: alpha, beta in [0,1]; alpha=beta=0.5 => Dice; alpha>beta penalizes FP more; typical values (0.3,0.7) for precision/recall tradeoff
  * **When to Avoid**: Without domain-specific cost ratio; requires justification for alpha/beta values
- **Focal Tversky Loss**:
  * **Mathematical Intuition**: Applies modulating factor (1-Tversky)^gamma to down-weight well-segmented examples
  * **Why Appropriate**: Focuses training on hard-to-segment examples (e.g., boundaries, low-contrast regions)
  * **Gamma Justification**: gamma>0 focuses on hard examples; gamma=0 reduces to Tversky; typical values 0.5-2.0
  * **When to Avoid**: If dataset is already well-segmented; may slow convergence on easy examples
- **Loss Logging**:
  * Record exact loss function formula, parameters, and weighting strategy
  * Justify why chosen loss aligns with segmentation goals (boundary precision, overlap, class imbalance handling)

## Mask Generation and Post-processing
- **Binary Mask Creation**:
  * Apply threshold (default 0.5) to probability map to obtain binary lung prediction
  * Threshold justification: Balances precision and recall; can be optimized on validation set for specific metric (e.g., maximize Dice)
  * Log threshold value and justification
- **Connected Component Analysis (CCA)**:
  * **Purpose**: Remove spurious small predictions and enforce lung anatomical expectation (usually two components)
  * **Algorithm**: 
    * Label connected components in binary image (4-connectivity or 8-connectivity; justify choice)
    * Sort components by area (pixel count)
    * Remove components below min_size threshold (e.g., 50-500 pixels justified by expected noise size)
    * Keep top N components (typically N=2 for left/right lungs)
  * **Justification**: 
    * min_size: based on pixel spacing and expected minimum lung feature size (e.g., 5mm nodule => min pixels = (5/mm)^2)
    * N=2: anatomical prior; unless pathology known to cause fusion/collapse (rare in screening)
  * **Logging**: Record min_size, connectivity, N, and components removed/kept per image
- **Morphological Operations**:
  * **Opening** (erosion followed by dilation):
    * Purpose: Remove small false positive protrusions (noise) while preserving lung shape
    * Kernel Justification: Size based on pixel spacing and expected noise scale (e.g., 2x2 kernel for 0.2mm/pixel => 0.4mm scale)
  * **Closing** (dilation followed by erosion):
    * Purpose: Fill small holes in lung mask (e.g., due to prediction gaps in fissures)
    * Kernel Justification: Similar to opening; based on expected gap scale
  * **Never**: Arbitrary large kernels that distort lung boundaries; justify size relative to anatomical features
  * **Log**: Operation type, kernel size, shape (square, ellipse), and justification
- **Lung-Specific Refinement**:
  * **Lung Symmetry Enforcement** (optional, use with caution):
    * Only if dataset expected symmetric lungs (e.g., healthy screening population)
    * Method: Force left/right lung area ratio to be within bounds (e.g., 0.8-1.2) by scaling smaller lung
    * **Critical**: Never apply if pathology (e.g., lung collapse, effusion) is expected; validate with radiologist
    * Log if applied and justification
  * **Diaphragm Position Constraint** (optional):
    * Estimate diaphragm curve from lung mask bottom boundary
    * Flag if position deviates excessively from expected range (e.g., >2 standard deviations from mean)
    * Never automatically correct; use for quality flagging only
  * **Lung Area Boundaries**:
    * Flag if lung area too small (<10% image) or too large (>90% image) indicating probable failure
    * Justification: Based on typical lung-to-thorax ratio in chest X-rays
  * **Record**: All post-processing steps applied and justification in mask metadata

## Error Handling
- **Critical Errors (halt training, log and alert)**:
  * GPU out of memory: reduce batch size, log recommendation
  * NaN loss: check learning rate, numerical stability; halt and investigate
  * Data loader failure: missing files or corrupt images; log and skip batch
  * Shape mismatch: input image vs model expectation; halt and verify preprocessing consistency
- **Recoverable Errors** (log warning, continue):
  * Validation image missing: skip and log
  * Post-processing component count unexpected: log warning, proceed with available components
  * Learning rate scheduler NaN: fallback to fixed LR, log
  * Checkpoint save failure: retry, log if persistent
- **Logging Requirements**:
  * Every error/warning must include timestamp, epoch (if applicable), batch, and description
  * Continue processing after logging; never silently skip
  * Never expose raw stack traces to user; provide contextual message
- **Validation Checks**:
  * After each epoch, verify validation metrics are finite numbers
  * After training, verify best model file exists and is loadable
  * After mask generation, verify output masks are non-empty and same dimensions as input
  * Log success/failure of each step with counts

## Logging
- **Training Log** (`logs/segmentation_training_<timestamp>.log`):
  * Header: timestamp, git commit (if available), TF/PyTorch version, GPU/CPU info
  * Configuration dump: all parameters with justifications
  * Per-epoch: train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, learning_rate, epoch_time
  * Early stopping: metric, patience, best epoch
  * Checkpoint: saved when validation metric improves
  * Warnings: data loading issues, shape mismatches, NaN losses
  * Final: best validation/test metrics, model path
- **Experiment Manifest** (`exports/segmentation/experiment_manifest.json`):
  * `experiment_id`: timestamp or UUID
  * `dataset_versions`: references to Dataset Engineer manifests used
  * `preprocessing_version`: reference to Computer Vision Engineer manifest
  * `random_seeds`: numpy, tensorflow, python
  * `hardware`: GPU model, CPU cores, RAM
  * `git_info`: {commit, branch, dirty} if available
  * `start_time`, `end_time`, `duration_seconds`
  * `best_model_path`: relative to exports/
- **Quality Report** (`docs/segmentation_quality_report.json`):
  * `aggregate_metrics`: {val: {dice_mean, dice_std, iou_mean, iou_std}, test: {...}}
  * `failure_analysis`: 
    * `low_dice_cases`: list of {image_path, dice, error_type, description}
    * `common_error_types`: [apical_miss, diaphragm_confusion, heart_inclusion, ...]
  * *recommendations*: [e.g., "Increase encoder depth to 5 for better context", "Try attention gates"]
  * `post_processing_stats`: {mean_lung_area, symmetry_rate, diaphragm_flag_rate, avg_components}

## Performance Considerations
- **Memory Optimization**:
  * Use TensorFlow Dataset API or PyTorch DataLoader with prefetching
  * Cache preprocessed images in memory if dataset fits; else use disk cache
  * Release batch tensors after forward/backward pass
- **Computation Optimization**:
  * Use built-in operations (tf.nn.conv2d, torch.conv2d) rather than custom loops
  * Enable XLA (TensorFlow) or TorchScript (PyTorch) for acceleration if justified
  * Log training time per epoch; identify bottlenecks (data loading vs computation)
- **Mixed Precision**:
  * If using, log loss scaling method and verify no numerical instability
  * Fallback to float32 if instability detected
- **Batch Size Justification**:
  * Log GPU memory usage per batch; adjust if OOM
  * Never exceed memory that causes swapping or training failure
- **Epoch Count Justification**:
  * Log validation metric convergence; stop when plateaus (early stopping handles this)
  * Avoid excessive epochs that overfit without validation improvement
- **Saving Frequency**:
  * Save checkpoints only on improvement to reduce I/O
  * Log checkpoint count and disk usage

## Documentation Requirements
- **Model Documentation**:
  * Every exported model must be accompanied by:
    * Architecture description (depth, filters, attention)
    * Training configuration (optimizer, LR, augmentation)
    * Dataset versions used
    * Random seeds
    * Preprocessing pipeline version
    * Intended use (lung segmentation for TB classification pipeline)
- **Code Documentation**:
  * Every function/class must have docstring explaining purpose, inputs, outputs, and justification for non-obvious choices
  * Reference specific sections of CLAUDE.md or lung segmentation literature for key decisions
  * Example: `def build_unet(...): """Builds U-Net with skip connections. Justification: Preserves spatial detail for boundary segmentation (Ronneberger et al., 2015)."""`
- **Configuration Documentation**:
  * Every parameter in `segmentation_config.json` must include:
    * `value`: the parameter value
    * `justification`: evidence-based reason (literature, internal validation, physiological constraint)
    * `range`: acceptable values if applicable (e.g., learning_rate: 1e-5 to 1e-3)
    * `type`: (float, int, string, bool)
  * Maintain separate JSON schema for validation
- **Process Documentation**:
  * Maintain `SEGMENTATION_TRAINING_PROTOCOL.md` detailing:
    * Step-by-step pipeline from data loading to model export
    * Justification for each step (e.g., "Why apply augmentation only to training?")
    * Expected input formats and output schemas
    * Quality control checkpoints (e.g., "Validate mask anatomical plausibility")
  * Update `docs/segmentation_guide.md` with:
    * How to run training
    * How to interpret logs and output files
    * Troubleshooting common issues (OOM, NaN loss, poor convergence)
- **Output Documentation**:
  * CSV files: clear headers, units where applicable (pixels for area, mm for distances if pixel spacing known)
  * JSON manifests: include schema or link to JSON Schema file
  * Log files: standard format with timestamp, level, message; consider JSON lines for machine parsing
  * Model files: Keras format includes architecture and weights; document loading instructions
- **Versioning**:
  * Include software versions (TF/PyTorch, Python, CUDA) in logs and manifest
  * Record dependency versions (via `pip freeze` or equivalent) in manifest for environment reconstruction
  * Tag exports with experiment ID or date for traceability
  * Never overwrite previous experiments; always create new folder under `exports/segmentation/` or use experiment manifest

## Interaction with Other Skills
- **Consumes From Dataset Engineer**:
  * Split CSV files guaranteeing patient-level separation
  * Preprocessed image directory from Computer Vision Engineer (consistent normalization, resizing)
  * Validation that images are readable, labels correct, duplicates removed
- **Consumes From Computer Vision Engineer**:
  * Guarantee that input images have undergone scientifically justified preprocessing (CLAHE, normalization, resizing)
  * Assurance that aspect ratio preserved via padding, no geometric distortion
  * Confirmation that pixel values are in expected range (e.g., [0,1] float)
- **Consumes From Medical Imaging Engineer**:
  * Validation that image orientation and laterality are correct (prevents label inversion in segmentation)
  * Assurance that pixel spacing is present or justified assumption
  * Confirmation that no burned-in text obscures lung fields (critical for mask accuracy)
  * Verification that image completeness includes full lung fields
- **Produces For Classification Engineer**:
  * Lung-only images (original image masked with segmentation prediction) for Notebook 3 input
  * Segmentation masks in `exports/segmentation/predicted_masks/` with consistent naming
  * Mask metadata indicating segmentation quality (Dice, symmetry, etc.) for filtering low-quality inputs
  * Export of best segmentation model for potential use in ensemble or feature extraction
- **Produces For Explainability Engineer**:
  * Anatomically consistent lung regions for Grad-CAM localization (ensures explanations are over lung, not background)
  * Reliable mask-removed background reduces false activations in non-lung regions
  * Segmentation quality metrics to weight explainability validation (trust masks more when segmentation good)
- **Collaborates With Research Engineer**:
  * Provides segmentation methodology and justification for literature review
  * Shares failure case analysis to inform limitations and future work sections
  * Validates that segmentation experiments follow reproducibility requirements
- **Collaborates With MLOps Engineer**:
  * Ensures model export format compatible with serving infrastructure (TensorFlow SavedModel, TorchScript)
  * Validates that preprocessing pipeline for inference matches training (via configuration hashes)
  * Provides metadata for model card (intended use, training data, performance)

## Quality Checklist (Must Pass Before Handoff to Downstream)
* [ ] Every architecture decision (depth, filters, attention) has justification documented in config or code
* [ ] No hardcoded image sizes, batch sizes, or learning rates; all configuration-driven
* [ ] Fixed random seeds for reproducibility; same input + same config = same model
* [ ] Normalization statistics (if used) derived exclusively from training set
* [ ] Augmentation applied only during training (never validation/test)
* [ ] Patient-level separation verified: no patient appears in more than one split
* [ ] All training hyperparameters logged (optimizer, LR, scheduler, epsilon, etc.)
* [ ] Validation metrics computed per epoch (loss, Dice, IoU, etc.)
* [ ] Best model saved by validation Dice (or justified alternative metric)
* [ ] Segmentation masks saved as lossless PNG with consistent naming
* [ ] Mask metadata includes lung area, symmetry, component count, etc.
* [ ] Experiment manifest links to dataset and preprocessing versions
* [ ] Log file contains no unhandled exceptions that halted training
* [ ] Quality report includes failure analysis and recommendations
* [ ] Configuration file archived with outputs for reproducibility
* [ ] Output directory structure matches expectations for Classification Engineer (`exports/segmentation/predicted_masks/`)
* [ ] No vertical flips applied (would invert anatomy)
* [ ] No horizontal flips applied unless laterality proven irrelevant (extremely rare in chest radiographs)
* [ ] All post-processing steps justified and logged
* [ ] Anatomical plausibility checks performed and flagged if concerning
* [ ] Test set evaluation performed if available; log metrics
* [ ] Model file is loadable and produces expected output shape
* [ ] Every configuration parameter includes justification field
* [ ] Code follows modular principles: architecture, loss, data loading, training loop separable
* [ ] Documentation updated: PROTOCOL.md, GUIDE.md reflect current pipeline

## Common Mistakes
* **Using global average pooling instead of skip connections**: Loses spatial detail, produces blurred boundaries
* **Ignoring class imbalance in loss function**: Leads to degenerate model predicting background everywhere
* **Applying data augmentation to validation/test**: Invalidates evaluation by artificially improving performance
* **Using horizontal flip in chest X-rays**: Creates mirror-image heart (apparent dextrocardia) which is pathological
* **Nearest-neighbor resizing for masks**: Causes blocky artifacts that misalign with lung boundaries
* **Assuming all images are same orientation**: Results in upside-down or rotated lungs
* **Zero-padding during resizing**: Creates artificial black edges mistaken for pathology (e.g., pleural effusion)
* **Not validating mask anatomical plausibility**: Allows implausible lungs (e.g., diagonal, fragmented) to pass
* **Using arbitrary threshold (e.g., 0.3) without justification**: Shifts precision-recall tradeoff unpredictably
* **Failing to verify patient-level split separation**: Leads to data leakage and overoptimistic performance
* **Using test set for early stopping or hyperparameter tuning**: Invalidates final test performance
* **Applying global histogram equalization as preprocessing**: Amplifies noise and alters anatomy shading
* **Assuming pixel spacing is uniform without checking**: Leads to incorrect size estimates if variable
* **Not logging augmentation parameters**: Makes reproducibility impossible
* **Using elastic deformation for augmentation**: Creates anatomically impossible spinal or rib deformations
* **Treating quantum noise as removable via denoising**: Noise is inherent; over-denoising removes diagnostic texture
* **Ignoring laterality markers or anatomical heuristics**: Results in left/right label swaps
* **Applying windowing to pixel data for analysis**: Alters raw relationship to attenuation
* **Using sigmoid activation with threshold 0.5 without validation**: May not optimize for Dice or IoU
* **Not saving training history**: Prevents analysis of convergence and overfitting
* **Using batch normalization without verifying it helps**: Must justify based on internal validation
* **Assuming all lungs are symmetric**: Pathology (effusion, collapse) breaks symmetry; validate before enforcing
* **Using large morphological kernels**: Over-smooths boundaries, removes genuine lung features
* **Failing to verify output mask dimensions match input**: Causes misalignment in downstream tasks
* **Ignoring Computed Vision Engineer's output format**: Leads to mismatch (e.g., expecting grayscale but getting RGB)
* **Not validating that loss decreases during training**: Early sign of bugs (learning rate too high, etc.)
* **Using Adam without weight decay justification**: May overfit; AdamW preferred for decoupled decay
* **Ignoring gradient clipping justification**: Prevents explosion; justify norm value (e.g., 1.0)
* **Assuming all datasets have ground truth masks**: If not, clarify evaluation strategy (e.g., semi-supervised, proxy metrics)
* **Not checking for NaN in loss or metrics**: Leads to silent training failure
* **Using non-deterministic order for data loading**: Leads to different splits between runs
* **Applying Confidence-weighted averaging without basis**: May blur edges and reduce diagnostic sharpness
* **Treating the image as if it were natural photography**: Chest X-rays have specific physics and anatomy; generic CV assumptions fail
* **Assuming U-Net is always best without validation**: Must compare to alternatives (e.g., FCN, PSPNet) with justification
* **Using transposed convolution without checking for checkerboard artifacts**: May require initialization fix or alternative upsampling
* **Not validating that skip connections are concatenated (not added) if justified**: Changes feature fusion behavior
* **Post-processing that removes genuine pathology**: e.g., removing large nodules as "noise"
* **Using hardcoded paths for data or exports**: Breaks portability and reproducibility
* **Failing to update documentation when pipeline changes**: Leads to outdated instructions
* **Not preserving random seeds in manifest**: Prevents exact replication
* **Using validation loss for early stopping without justifying why it correlates with generalization**
* **Logging only final metrics**: Misses opportunity to detect overfitting early
* **Sharing model without preprocessing configuration**: Guarantees failure in deployment due to pipeline mismatch

## Never Do
* **Never** hardcode image dimensions, batch sizes, or learning rates; always derive from configuration with justification
* **Never** apply image processing steps (rotation, flip, scaling, intensity transform) without documenting their mathematical and clinical justification
* **Never** use a random seed based on system time (e.g., `time.time()`) for any stochastic process; always fix the seed for reproducibility
* **Never** compute normalization statistics (mean, std) from the entire dataset including validation/test splits
* **Never** apply data augmentation (rotation, scaling, etc.) to validation or test datasets
* **Never** use transformations that create anatomically impossible structures (e.g., >30° rotation, extreme shear that warps spine)
* **Never** silently skip corrupted, unreadable, or missing images; always log with reason (file not found, unsupported format, corrupt header)
* **Never** assume labels from different acquisition protocols (e.g., PA vs AP) are interchangeable without validation
* **Never** apply histogram matching without validating that the reference distribution represents your target population and does not map pathological intensities to non-pathological ranges
* **Never** use interpolation methods that cause severe artifacts for smooth intensity gradients (e.g., nearest-neighbor for gradual lung opacity changes)
* **Never** apply gamma correction as a default preprocessing step; require explicit justification based on detector response analysis
* **Never** modify raw image data; always work on copies and preserve originals in `datasets/raw/` for audit
* **Never** skip image validation step; garbage in, garbage out ruins model training and can produce dangerous false confidence
* **Never** use hardcoded intensity thresholds (e.g., divide by 255) without first verifying actual data range per image via bits stored
* **Never** apply vertical flips; they invert anatomical relationships (e.g., diaphragm positioned above lungs)
* **Never** use elastic deformation without strong justification and validation that deformed anatomy remains physiologically plausible
* **Never** process images in non-deterministic order (e.g., `os.listdir()` without sorting) leading to non-reproducible train/test splits
* **Never** omit justification for any configuration parameter; "it works better" or "it's common practice" is insufficient
* **Never** process DICOM images without consulting the DICOM standard for correct pixel data interpretation (endianness, signedness, padding)
* **Never** assume all grayscale images are 8-bit; always check BitsAllocated and BitsStored and scale appropriately
* **Never** apply windowing as a default step; it is a display operation, not a preprocessing step for analysis
* **Never** use training/test splits that share patients if patient IDs are available; always enforce patient-level separation
* **Never** ignore aspect ratio when resizing; always pad to preserve original ratio or provide justification for stretching
* **Never** use batch processing non-deterministically (e.g., shuffling without fixed seed) for reproducible splits
* **Never** validate augmentation by looking only at loss; visually inspect augmented samples for anatomical plausibility
* **Never** treat quantum noise as something to remove; it is inherent to the imaging modality and may contain texture information
* **Never** assume laterality is irrelevant in chest X-rays without validation via cardiac position or laterality markers
* **Never** process images without logging the exact parameters used for each image (e.g., rotation angle, padding value)
* **Never** use image processing libraries without understanding their default behaviors (e.g., OpenCV vs PIL coordinate systems, origin at top-left)
* **Never** assume DICOM images are little-endian; always check (0002,0010) Transfer Syntax UID and convert if necessary
* **Never** apply intensity clipping without justification; it can remove subtle pathological signatures (e.g., faint ground-glass opacity)
* **Never** use interpolation methods that cause ringing artifacts (e.g., sinc) without windowing or filtering justification
* **Never** apply preprocessing steps that are not at least approximately invertible for debugging purposes (when possible)
* **Never** process DICOM files without checking for compression and attempting decompression
* **Never** ignore the PhotometricInterpretation tag; assuming MONOCHROME1 when it's MONOCHROME2 inverts the image
* **Never** assume PixelRepresentation is unsigned; always check and interpret signed data correctly
* **Never** assume Rescale Slope is 1 and Intercept is 0; apply if present to convert to meaningful units
* **Never** assume all images have the same orientation; check Patient Orientation or use anatomical heuristics
* **Never** assume laterality markers are always present; have fallback anatomical heuristics with confidence logging
* **Never** assume all images are free of burned-in annotations; always scan and log presence/location
* **Never** assume image completeness without checking for cropping that cuts off lung apices or costophrenic angles
* **Never** assume absence of artifacts; always check for common ones (grid lines, motion, metal) and log findings
* **Never** process images without recording software and library versions for reproducibility
* **Never** release validation outputs without a manifest that allows tracing back to original files
* **Never** modify the validation criteria mid-experiment without documenting the change and re-running validation on all data
* **Never** allow validation to proceed without a comprehensive log that enables audit
* **Never** treat the validation step as optional; it is essential for ensuring data quality and reproducibility

## Deliverables
The Segmentation Engineer skill delivers a production-ready lung segmentation pipeline that provides:
1. **Anatomically Accurate Lung Masks**:
   * Scientifically justified U-Net or Attention U-Net architecture preserving lung boundaries via skip connections
   * Loss function optimized for overlap (Dice) and class imbalance robustness
   * Patient-level split separation preventing data leakage
   * Post-processing that enforces anatomical expectation (two lungs) without removing genuine pathology
   * Validation against quantitative metrics (Dice, IoU) and anatomical plausibility (symmetry, diaphragm position)
2. **Reproducible and Auditable Training**:
   * Fixed random seeds, versioned dependencies, and configuration-driven experiments
   * Comprehensive training log with epoch-level metrics, learning rate, and events
   * Experiment manifest linking model to dataset versions, preprocessing pipeline, and hardware
   * Best model saved by validation performance with clear versioning
3. **Downstream Model Readiness**:
   * Lung-only images (original image masked with segmentation prediction) ready for TB classification
   * Segmentation masks in standardized format and location for Notebook 3 consumption
   * Mask metadata providing quality indicators (Dice, symmetry, etc.) for filtering or weighting inputs
   * Guaranteed no leakage from preprocessing parameters (especially normalization stats)
   * Deterministic output enabling reproducible experiments and fair comparison
4. **Documentation and Compliance**:
   * Self-describing output files (CSV headers, JSON schemas, log formats)
   * Processing protocol document justifying every architectural and training decision
   * Configuration file enabling exact replication of training pipeline
   * Quality report providing performance summary, failure case analysis, and improvement recommendations
   * Adherence to medical AI principles: no altered anatomy, no misrepresented laterality, no unjustified information loss
5. **Risk Mitigation**:
   * Early detection of systemic issues (e.g., consistent apical misses) before model deployment
   * Guidance on architecture or hyperparameter adjustments based on failure case analysis
   * Documentation of assumptions and limitations for transparent reporting
   * Support for ethical and regulatory review via traceable metadata handling and de-identification logs

In summary, the Segmentation Engineer skill ensures that every lung segmentation model entering the AI pipeline has been developed with the same rigor and justification as the model training itself, eliminating a major source of variability and artifact in medical AI research and enabling trustworthy, explainable AI-assisted tuberculosis screening through accurate lung field extraction.