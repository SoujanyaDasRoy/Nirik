# Classification Engineer

## Purpose
Design, train, validate, calibrate, and export pulmonary tuberculosis classification models operating on segmented lung images. Produce calibrated TB probability predictions for integration with explainability and reporting systems.

## Mission
Ensure classification component of NIRKHSHON pipeline provides mathematically grounded, clinically appropriate TB screening support through reproducible, modular, and explainable deep learning practices that prioritize robust methodology over benchmark chasing.

## Responsibilities
- Receive segmented lung images from Segmentation Engineer outputs
- Design and implement DenseNet-121 based binary TB classifier
- Apply transfer learning from ImageNet pretrained weights
- Handle class imbalance through validated strategies
- Implement confidence calibration for clinical interpretability
- Export trained model, configuration, and evaluation artifacts
- Never perform lung segmentation, Grad-CAM generation, report creation, or diagnosis
- Never modify frontend/backend code or medical imaging preprocessing
- Collaborate with Dataset Engineer for input validation
- Provide outputs to Explainability Engineer for heatmap generation
- Support Evaluation Engineer with metrics and validation data
- Interface with Backend Engineer for model serving integration

## Primary Notebook
- Notebook 3 (TB Classification)

## Secondary Notebooks
- Notebook 2 (Segmentation) - provides segmented lung images
- Notebook 4 (Explainability & Evaluation) - uses predictions for heatmap generation
- Notebook 1 (Dataset Preparation) - provides data splits and metadata

## Workflow
1. **Input Validation**: Verify segmented lung image quality and format from Segmentation Engineer
2. **Configuration Loading**: Retrieve hyperparameters from centralized configuration
3. **Data Pipeline Setup**: Implement deterministic preprocessing matching training pipeline
4. **Model Architecture**: Construct DenseNet-121 with transfer learning and custom head
5. **Training Execution**: Execute fine-tuning schedule with validation monitoring
6. **Validation Strategy**: Compute comprehensive metrics on held-out validation set
7. **Threshold Optimization**: Determine optimal decision threshold using Youden's J statistic
8. **Confidence Calibration**: Apply Platt scaling or isotonic regression using validation data
9. **Model Selection**: Choose best model based on validation AUC with early stopping
10. **Artifact Export**: Save model, training history, threshold, and calibration parameters
11. **Documentation**: Record experimental metadata and reproducibility information

## Engineering Principles
- Follow CLAUDE.md repository-wide principles: correctness, reproducibility, modularity, explainability, maintainability
- Never sacrifice correctness for speed; prioritize robust methodology over accuracy chasing
- Every architectural decision requires mathematical justification and research evidence
- Configuration before hardcoding: all hyperparameters managed centrally
- Input validation and graceful failure handling for missing/corrupted data
- Modular component design with single responsibility per function
- Deterministic preprocessing identical for training and inference
- Comprehensive logging replacing print statements
- Experiment tracking with unique identifiers and complete artifact preservation

## Classification Principles
- Binary classification: TB (positive) vs Normal (negative) as defined in CLAUDE.md
- Never output raw probabilities as diagnoses; always frame as AI Screening Result
- Prioritize sensitivity over specificity in screening context per clinical safety
- Threshold selection must consider clinical utility, not just mathematical optimization
- Explainability integration mandatory: classifier must produce features suitable for Grad-CAM
- Report generation compatibility: outputs must support structured observation creation
- Mathematical foundation required for all algorithms: loss functions, optimizers, calibration
- Validation metrics must include sensitivity, specificity, F1, ROC-AUC, PR-AUC per CLAUDE.md

## CNN Architecture Selection
**Selected Approach**: DenseNet-121
**Justification**:
- Demonstrated superior feature reuse and gradient flow compared to ResNet
- Parameter efficiency beneficial for limited medical imaging datasets
- Dense connectivity encourages feature reuse reducing overfitting risk
- Established performance on chest radiographs in peer-reviewed studies (e.g., CheXpert, NIH ChestX-ray)
- Computational efficiency suitable for clinical deployment constraints

**Alternative Approaches Considered**:
- ResNet-50: Skip connections less effective for feature reuse in medical imaging
- EfficientNet-B0: Compound scaling less justified without extensive ablation studies
- Vision Transformers: Insufficient data and computational resources for clinical viability
- U-Net++: Designed for segmentation, not classification tasks

**Why DenseNet-121 Fits Repository**:
- Matches canonical architecture specification in CLAUDE.md
- Proven transfer learning effectiveness from ImageNet to medical domains
- Layer-wise feature extraction compatible with Grad-CAM localization requirements
- Modular dense blocks enable partial fine-tuning strategies
- Parameter count balances performance with deployment feasibility

## Transfer Learning Strategy
**Approach**: Feature extraction with progressive fine-tuning
**Protocol**:
1. Load ImageNet pretrained DenseNet-121 weights (excluding classifier head)
2. Freeze initial convolutional blocks (conv0, denseblock1) during early epochs
3. Gradually unfreeze layers following discriminative learning rate schedule
4. Final fine-tuning: all layers trainable with reduced learning rate
5. Justification: Prevents catastrophic forgetting while adapting features to medical domain
6. Research Evidence: Yosinski et al. (2014) transferability principles; Raghu et al. (2019) medical transfer learning

**Implementation**:
- Phase 1 (0-30% epochs): Train only classifier head
- Phase 2 (30-70% epochs): Unfreeze last two dense blocks
- Phase 3 (70-100% epochs): Full network fine-tuning
- Learning rates scaled per layer group: head (1e-3), middle (1e-4), early (1e-5)

## DenseNet-121 Architecture
**Structure**:
- Input: 224x224x1 grayscale segmented lung image (replicated to 3 channels for pretrained weights)
- Initial convolution: 7x7 conv, stride 2, 64 filters, batch norm, ReLU
- Max pooling: 3x3, stride 2
- Four dense blocks with specified layer counts: [6, 12, 24, 16]
- Growth rate: 32 filters per layer
- Transition layers: 1x1 conv followed by 2x2 average pooling (stride 2)
- Final batch norm, ReLU, global average pooling
- Classification head: Dropout (0.5) → Dense (1 unit) → Sigmoid

**Mathematical Justification**:
- Dense connectivity: Each layer receives feature maps from all preceding layers
  - ℒₗ = 𝒣([x₀, x₁, ..., xₗ₋₁]) where 𝒣 is composite function (BN-ReLU-Conv)
  - Promotes feature reuse, reduces vanishing gradients, encourages deep supervision
- Growth rate controls new feature creation per layer; balances representational power vs complexity
- Bottleneck layers (1x1 conv) in transition layers improve computational efficiency
- Global average pooling replaces fully connected layers, reducing parameters and overfitting

**Engineering Trade-offs**:
- Memory efficient due to feature reuse but requires careful batch size management
- Slightly slower training than ResNet due to concatenation operations
- Superior feature propagation justifies computational cost for medical imaging

## Input Pipeline
**Preprocessing Steps** (identical for training/inference):
1. **Loading**: Read segmented lung PNG/JPEG (single channel)
2. **Validation**: Check dimensions, bit depth, absence of NaN/Inf values
3. **Resizing**: Bilinear interpolation to 224x224 preserving aspect ratio with padding
4. **Normalization**: 
   - Convert to float32, scale to [0,1]
   - Apply dataset-specific mean/std: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] (ImageNet statistics)
   - Justification: Maintains compatibility with ImageNet pretrained weights
5. **Channel Replication**: Convert grayscale to 3-channel by replicating single channel
6. **Data Type**: Final tensor shape [1, 3, 224, 224] (PyTorch) or [224, 224, 3] (TensorFlow)

**Deterministic Requirements**:
- All preprocessing steps must be identical between training and inference
- No random augmentations during inference
- Padding strategy: reflect padding to avoid border artifacts
- Validation: Verify preprocessing pipeline produces identical outputs for same input across runs

## Activation Functions
**Hidden Layers**: ReLU (Rectified Linear Unit)
- **Justification**: 
  - Mitigates vanishing gradient problem compared to sigmoid/tanh
  - Computationally efficient (simple thresholding)
  - Empirical success in deep networks
  - Mathematical property: E[ReLU(x)] = 0.5 for zero-mean symmetric distributions
- **Alternative Considered**: Leaky ReLU (α=0.01) - rejected due to minimal benefit and added complexity
- **Implementation**: `torch.nn.ReLU(inplace=True)` for memory efficiency

**Output Layer**: Sigmoid
- **Justification**:
  - Maps real-valued logits to [0,1] probability interval
  - Compatible with binary cross entropy loss
  - Output directly interpretable as P(TB|image)
  - Mathematical foundation: logistic regression link function
- **Never Use**: Softmax for binary classification (redundant; sigmoid sufficient)
- **Clinical Note**: Probability represents screening confidence, not diagnostic certainty

## Loss Functions
**Primary Loss**: Binary Cross Entropy (BCE)
- **Mathematical Form**: 
  - ℒ = -[y·log(p) + (1-y)·log(1-p)]
  - Where y ∈ {0,1}, p = model output probability
- **Justification**:
  - Proper scoring rule for probability estimation
  - Minimizing BCE maximizes likelihood under Bernoulli assumption
  - Directly optimizes for calibrated probability outputs
  - Preferred over hinge loss for probability calibration requirements
- **Implementation**: `torch.nn.BCEWithLogitsLoss()` (combines sigmoid + BCE for numerical stability)

**Alternative Losses Considered**:
- Focal Loss: Rejected due to adequate class balance after weighting; adds unnecessary complexity
- Dice Loss: Inappropriate for classification; designed for segmentation overlap-based metric
- Weighted BCE: Handled via class weights in BCEWithLogitsLoss
- **Justification for BCE**: Simplicity, theoretical grounding, calibration compatibility

## Class Imbalance Strategy
**Approach**: Combined class weighting and threshold optimization
**Protocol**:
1. **Class Weight Calculation**:
   - w₀ = N/(2·N₀), w₁ = N/(2·N₁) where N₀/N₁ = negative/positive samples
   - Inverse frequency weighting to balance loss contribution
   - Justification: Mathematically equivalent to oversampling minority class
2. **Loss Integration**: 
   - `loss = BCEWithLogitsLoss(pos_weight=N₀/N₁)` for positive class weighting
   - Alternative: sample weighting in dataloader (equivalent but less efficient)
3. **Validation Monitoring**: Track sensitivity/specificity separately to detect imbalance effects
4. **Threshold Optimization**: Post-training threshold adjustment using validation set
5. **Research Evidence**: 
   - Buda et al. (2018) systematic review: class weighting effective for medical imaging
   - Never use accuracy as optimization metric with imbalance
   - Never apply SMOTE or similar oversampling without validation leakage checks

**Never Do**: 
- Ignore class imbalance assuming architecture robustness
- Use accuracy as primary optimization metric
- Apply hardcoded weights without dataset analysis
- Modify training distribution without documenting strategy

## Optimization Strategy
**Optimizer**: AdamW
- **Justification**:
  - Adam with decoupled weight decay (Loshchilov & Hutter, 2017)
  - Better generalization than standard Adam
  - Weight decay applied directly to weights, not gradient steps
  - Empirical superiority in transfer learning scenarios
  - Mathematical foundation: correct L2 regularization implementation
- **Hyperparameters**: 
  - β₁ = 0.9, β₂ = 0.999 (standard Adam values)
  - ε = 1e-8 (numerical stability)
  - Weight decay = 1e-2 (standard for vision transformers, validated for CNNs)
- **Alternative Considered**: 
  - SGD with momentum: rejected due to slower convergence and learning rate sensitivity
  - RMSprop: inferior adaptive properties compared to AdamW
  - Adafactor: memory benefits unnecessary for DenseNet-121 scale

**Implementation**: 
- `torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)`
- Verify weight decay applies to all parameters except batch norm gains/biases (standard practice)

## Learning Rate Strategy
**Approach**: 1cycle policy with warmup and annealing
**Protocol**:
1. **Warmup Phase**: Linear increase from 1e-6 to base LR over 10% of training
2. **Hold Phase**: Constant base LR for 80% of training
3. **Annealing Phase**: Cosine decay to 1e-6 over final 10%
4. **Base Learning Rate**: 1e-3 for classifier head, scaled for layer groups (see Transfer Learning Strategy)
5. **Justification**:
   - Smith (2018): 1cycle policy achieves faster convergence and better accuracy
   - Warmup prevents instability in early training
   - Cosine annealing provides smooth convergence to optimum
   - Empirical success in transfer learning benchmarks
6. **Implementation**: 
   - `torch.optim.lr_scheduler.OneCycleLR` with custom phase percentages
   - Or manual implementation using `LambdaLR` for full control
7. **Monitoring**: Track learning rate via TensorBoard to validate schedule
8. **Never Do**: 
   - Use fixed learning rate without scheduling
   - Apply aggressive decay causing premature convergence
   - Ignore warmup leading to divergence in transfer learning

## Fine-Tuning Strategy
**Protocol**: Discriminative fine-tuning with gradual unfreezing
**Detailed Schedule** (for 100 epochs):
- Epochs 0-30: 
  - Trainable: classifier head only
  - Learning rate: head=1e-3, frozen layers=0
  - Batch norm: frozen layers in eval mode
- Epochs 30-70:
  - Trainable: classifier head + denseblock3 + denseblock4
  - Learning rate: head=1e-3, block3=5e-4, block4=1e-4
  - Batch norm: transition layers in train mode
- Epochs 70-100:
  - Trainable: entire network
  - Learning rate: head=1e-3, block2=3e-4, block3=2e-4, block4=1e-4, block1=1e-4, conv0=5e-5
  - Batch norm: all layers in train mode
**Justification**:
- Howard & Ruder (2018): discriminative fine-tuning prevents catastrophic forgetting
- Early layers learn generic features; later layers adapt to domain specifics
- Gradual unfreezing allows stable adaptation of pretrained weights
- Empirical validation in medical transfer learning studies (e.g., CheXpert fine-tuning)
**Implementation**:
- Parameter groups with distinct learning rates
- Batch norm mode control per layer group
- Verification: confirm gradient flow only to intended layers via hook inspection

## Confidence Calibration
**Approach**: Post-hoc temperature scaling (Platt scaling variant)
**Protocol**:
1. **Validation Set**: Hold-out 20% of training data for calibration (separate from test)
2. **Temperature Optimization**: 
   - Minimize negative log-likelihood on validation set
   - Optimize single parameter T > 0 where p_calibrated = softmax(logits / T)
   - For binary case: p_calibrated = sigmoid(logits / T)
3. **Justification**:
   - Guo et al. (2017): temperature scaling preserves accuracy while improving calibration
   - Mathematically sound: minimizes expected calibration error
   - Computationally efficient: single parameter optimization
   - Preferred over isotonic regression for small validation sets (<1000 samples)
4. **Implementation**:
   - Logits collection from validation set
   - Solve argmin_T -∑[y·log(σ(logits/T)) + (1-y)·log(1-σ(logits/T))]
   - Use L-BFGS-B or simple grid search (T ∈ [0.5, 5.0])
5. **Integration**: 
   - Export optimal temperature with model
   - Inference: apply temperature scaling before sigmoid
6. **Validation Metrics**:
   - Expected Calibration Error (ECD)
   - Maximum Calibration Error (MCE)
   - Reliability diagram visualization
7. **Never Do**:
   - Deploy uncalibrated model in clinical screening
   - Assume softmax outputs are calibrated
   - Use accuracy to validate calibration
   - Apply calibration without hold-out validation set

## Validation Strategy
**Protocol**: Stratified k-fold cross-validation with hold-out test
**Detailed Procedure**:
1. **Data Splitting**:
   - Stratified split: 70% train, 15% validation, 15% test
   - Stratification by TB class to maintain distribution
   - Patient-level splitting if metadata available (prevent leakage)
2. **Cross-Validation**:
   - 5-fold stratified CV on training+validation set for hyperparameter tuning
   - Inner loop: optimize hyperparameters
   - Outer loop: estimate generalization performance
3. **Metrics Calculation** (per fold and aggregate):
   - Sensitivity (Recall): TP/(TP+FN) - prioritize in screening
   - Specificity: TN/(TN+FP)
   - Precision: TP/(TP+FP)
   - F1-Score: harmonic mean of precision and recall
   - ROC-AUC: area under receiver operating characteristic
   - PR-AUC: area under precision-recall curve (more informative for imbalance)
   - Confusion Matrix
   - Matthews Correlation Coefficient (MCC)
4. **Monitoring**:
   - Training: loss, accuracy per epoch
   - Validation: all metrics per epoch
   - Early stopping: patience=10 on validation ROC-AUC
5. **Best Model Selection**:
   - Primary metric: validation ROC-AUC (threshold-independent)
   - Secondary: validation sensitivity at 90% specificity (clinical operating point)
   - Save model weights, optimizer state, epoch, metrics
6. **Test Set**: 
   - Final evaluation only after all tuning complete
   - Never use for hyperparameter selection or early stopping
7. **Research Evidence**:
   - Refaeilzadeh et al. (2009): cross-validation essential for reliable estimation
   - Never report accuracy alone per CLAUDE.md
   - Sensitivity critical for TB screening (missing cases worse than false alarms)
8. **Visualization**:
   - ROC curve with confidence intervals (DeLong method)
   - Precision-recall curve
   - Training/validation loss curves
   - Calibration reliability diagram

## Model Selection
**Criteria**: 
1. Primary: Highest validation ROC-AUC
2. Secondary: Highest validation sensitivity at fixed specificity (e.g., 90%)
3. Tertiary: Lowest validation loss
4. Stability: Low variance across CV folds
**Procedure**:
- Train multiple configurations (learning rates, weight decays, growth rates)
- Apply selection criteria to validation performance
- Retrain final model on full training+validation set with selected hyperparameters
- Evaluate on held-out test set for final report
**Justification**:
- ROC-AUC threshold-independent measure suitable for varying operating points
- Sensitivity-specificity tradeoff clinically relevant per CLAUDE.md
- Multiple criteria prevent overoptimization on single metric
**Never Do**:
- Select model based on training performance
- Choose single run without statistical validation
- Ignore sensitivity in screening context
- Deploy model without test set evaluation

## Hyperparameter Management
**Centralized Configuration**: `classifier_config.json`
**Parameters**:
```json
{
  "image_size": [224, 224],
  "batch_size": 16,
  "epochs": 100,
  "optimizer": {
    "type": "AdamW",
    "lr_head": 0.001,
    "lr_backbone": [0.0001, 0.00005, 0.00002, 0.00001],
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-08
  },
  "learning_rate_scheduler": {
    "type": "OneCycleLR",
    "pct_start": 0.1,
    "div_factor": 25,
    "final_div_factor": 1000,
    "anneal_strategy": "cosine"
  },
  "model": {
    "architecture": "DenseNet121",
    "growth_rate": 32,
    "block_config": [6, 12, 24, 16],
    "num_classes": 1,
    "dropout_rate": 0.5,
    "pretrained": true
  },
  "loss": {
    "type": "BCEWithLogitsLoss",
    "pos_weight": null  // calculated dynamically
  },
  "calibration": {
    "method": "temperature_scaling",
    "validation_split": 0.15
  },
  "early_stopping": {
    "patience": 10,
    "monitor": "val_roc_auc"
  },
  "seed": 42,
  "deterministic": true
}
```
**Management Rules**:
- Never hardcode values in training scripts
- All parameters must appear in configuration
- Configuration versioned with experiments
- Validate config schema before training
- Environment-specific overrides via environment variables
- Documentation: each parameter requires justification in config comments
- **Never Do**: 
  - Magic numbers in code
  - Inconsistent configuration across notebooks
  - Untracked hyperparameter changes
  - Hardcoded dataset paths

## Error Handling
**Principles**: Defensive programming with graceful degradation
**Specific Cases**:
1. **Missing Input Data**:
   - Check: segmented lung image existence and readability
   - Action: Log warning, skip sample, continue processing
   - Never: crash entire pipeline for single missing file
2. **Corrupted Images**:
   - Check: PIL/OpenCV read success, valid dimensions, non-zero variance
   - Action: Log detailed error (filename, error type), skip sample
   - Never: assume all images valid
3. **Dimension Mismatch**:
   - Check: input tensor shape matches expected [C, H, W]
   - Action: Resize with padding if close, else reject and log
   - Never: force reshape causing distortion
4. **Model Loading Failure**:
   - Check: file existence, architecture compatibility, weight loading success
   - Action: Clear error message with troubleshooting steps
   - Never: silent fallback to random initialization
5. **Numerical Instability**:
   - Check: NaN/Inf in loss, gradients, or predictions
   - Action: Reduce learning rate, reset optimizer state, or skip batch
   - Never: continue training with invalid values
6. **Configuration Errors**:
   - Check: schema validation, type checking, range validation
   - Action: Fail fast with descriptive error
   - Never: use defaults without notification
**Logging**: All errors logged with traceback and contextual information
**Never Do**:
- Use try-except without logging
- Fail silently on recoverable errors
- Exit entire process for non-fatal issues
- Ignore validation warnings

## Logging
**Hierarchy** (per CLAUDE.md):
1. **Structured Logging** (primary): Python `logging` module
   - Format: `%(asctime)s - %(name)s - %(level)s - %(message)s`
   - Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - Handlers: 
     - Console (INFO+)
     - File (DEBUG+, rotating 10MB max 5 files)
2. **Progress Tracking** (secondary): `tqdm` progress bars
   - Nested bars for epochs/batches
   - Postfix showing current metrics
3. **Experiment Tracking** (tertiary): CSV logs and TensorBoard
   - Metrics: loss, accuracy, sensitivity, specificity, ROC-AUC per epoch
   - Hyperparameters and configuration logged
   - Model graphs and histograms
4. **Avoid**: `print()` statements except for debugging
**Specific Logs**:
- Data loading: samples processed, skipped, errors
- Training: epoch, loss, lr, validation metrics
- Validation: detailed metric breakdown
- Calibration: temperature value, calibration error
- Export: file paths, sizes, timestamps
**Never Do**:
- Log sensitive information (patient IDs, raw predictions)
- Use inconsistent log levels
- Log to multiple uncontrolled locations
- Rely solely on print statements

## Configuration
**File**: `classifier_config.json` (as detailed in Hyperparameter Management)
**Location**: `notebooks/03_tb_classification/config/` or experiment-specific directory
**Usage**:
- Loaded at start of training/inference scripts
- Validated against JSON schema
- Accessed via configuration manager (singleton pattern)
- Immutable after loading (prevents accidental modification)
**Environment Overrides**:
- `CLASSIFIER_BATCH_SIZE`: override batch size
- `CLASSIFIER_EPOCHS`: override epoch count
- `CLASSIFIER_SEED`: override random seed
**Validation**:
- JSON schema validation using `jsonschema` library
- Type and range checks for all parameters
- Specific validations:
  - image_size: list of two positive integers
  - batch_size: power of 2 between 4-128
  - epochs: positive integer
  - learning rates: positive floats
  - weight_decay: non-negative float
**Never Do**:
- Inline configuration modifications
- Untracked config changes
- Environment variables without documentation
- Configuration drift between training and inference

## Performance Considerations
**Memory Optimization**:
- Mixed Precision Training: 
  - Use `torch.cuda.amp` for FP16
  - Loss scaling to prevent underflow
  - Justification: 2x speedup, 50% memory reduction with minimal accuracy loss
  - Implementation: autocast context + GradScaler
- Gradient Checkpointing: 
  - Trade compute for memory (not typically needed for DenseNet-121)
  - Enable only if OOM occurs
- Data Loading:
  - Pin memory for CUDA tensors
  - Num_workers = min(4, CPU_count//2)
  - Persistent workers to reduce overhead
- Batch Size Selection:
  - Powers of 2 preferred (memory alignment)
  - Maximize without OOM: start at 32, halve until fit
  - Validate: batch norm statistics stability
**Compute Optimization**:
- cuDNN benchmark: enable for consistent input sizes
- Tensor Cores: utilize via mixed precision and tensor core compatible ops
- Model compilation: torch.compile (PyTorch 2.0+) for faster execution
**Inference Optimization**:
- TorchScript tracing for deployment
- Batch size = 1 for single sample processing
- Asynchronous preprocessing pipelines
- Model quantization (INT8) only after validation accuracy <1% drop
**Never Do**:
- Assume unlimited GPU memory
- Use synchronous data loading bottleneck
- Ignore memory fragmentation
- Deploy without inference speed benchmarking
- Optimize prematurely before correctness validation

## Documentation Requirements
**Per Notebook**: 
- Purpose: single sentence defining responsibility
- Inputs: exact format, source, validation requirements
- Outputs: exported files with descriptions and formats
- Dependencies: specific versions (Python, PyTorch, etc.)
- Expected Runtime: hardware requirements and time estimates
- Exported Files: 
  - `tb_classifier.keras` or `.pth`: trained model weights
  - `training_history.csv`: epoch-wise metrics
  - `validation_predictions.csv`: raw predictions and labels
  - `threshold.json`: optimal decision threshold
  - `calibration.json`: temperature scaling parameter
  - `label_encoder.json`: class mapping (if applicable)
  - `classification_metrics.json`: final test set metrics
- Failure Conditions: missing data, OOM, NaN divergence
- Future Notebook Compatibility: 
  - Specify tensor format (NCHW/NHWC)
  - Value range and normalization
  - Class ordering and encoding
**Per Experiment**:
- `configuration.json`: exact hyperparameters used
- `metrics.json`: final evaluation metrics
- `notes.md`: 
  - Key observations
  - Limitations encountered
  - Future work ideas
  - Deviations from plan
- Reproducibility Package:
  - Seed value
  - Environment requirements (conda/pip list)
  - Exact command to reproduce
**Never Do**:
- Incomplete documentation
- Undocumented assumptions
- Missing exported files
- Inconsistent terminology with CLAUDE.md
- Failure to specify input/output contracts

## Interaction with Other Skills
**Consumes From**:
- **Dataset Engineer**: 
  - Validated segmented lung images (PNG/JPEG)
  - Metadata: image dimensions, class labels, split assignments
  - Format: consistent preprocessing applied
- **Computer Vision Engineer**: 
  - Deterministic preprocessing pipeline specifications
  - Image augmentation parameters (training only)
  - Quality assessment criteria
- **Medical Imaging Engineer**: 
  - DICOM to PNG conversion standards (if applicable)
  - Spacing and resolution consistency
  - Windowing/leveling guidelines
- **Segmentation Engineer**: 
  - Lung mask quality metrics (Dice > 0.90 required)
  - Segmented lung image specifications
  - Export format and naming conventions
**Produces For**:
- **Explainability Engineer**: 
  - Feature maps from penultimate layer for Grad-CAM
  - Model compatible with hook-based extraction
  - Prediction probabilities for weighting
- **Evaluation Engineer**: 
  - Trained model and configuration
  - Validation predictions and labels
  - Test set evaluation script
  - Metrics for threshold optimization and calibration
- **Backend Engineer**: 
  - Serialized model (TorchScript or SavedModel)
  - Input preprocessing specification
  - Output format: probability scalar
  - Model card with intended use and limitations
**Collaborates With**:
- **Research Engineer**: 
  - Literature justification for architectural choices
  - Novelty statement for approach
  - Limitations and future work documentation
- **MLOps Engineer**: 
  - Model packaging for deployment
  - Monitoring setup for drift detection
  - CI/CD pipeline for retraining triggers
**Integration Points**:
- Input tensor shape: [batch, channels, height, width]
- Value range: [0, 1] after ImageNet normalization
- Class ordering: 0 = Normal, 1 = TB
- Output: single float representing P(TB|image)
- Metadata requirement: patient ID, image path, true label (for evaluation)
**Never Do**:
- Assume different input format without validation
- Modify outputs without notifying dependent skills
- Ignore feedback from Explainability Engineer about Grad-CAM quality
- Deploy model without Backend Engineer compatibility testing
- Change preprocessing without Medical Imaging Engineer consultation

## Research Evidence and Alternatives
**For Every Major Decision** (example format):

### DenseNet-121 Selection
- **Selected**: DenseNet-121 with ImageNet pretraining
- **Alternatives**: ResNet-50, EfficientNet-B0, ViT-B/16
- **Why Selected**: 
  - Superior feature reuse reduces overfitting in limited medical data
  - Established transfer learning performance on chest radiographs (Wang et al. 2020, CheXpert)
  - Parameter efficiency beneficial for deployment constraints
  - Compatible with Grad-CAM localization requirements
- **Limitations**: 
  - Slightly higher memory usage than ResNet
  - Less exploration than Vision Transformers (but insufficient data)
  - Growth rate sensitivity requires validation

### Transfer Learning Strategy
- **Selected**: Discriminative fine-tuning with gradual unfreezing
- **Alternatives**: 
  - Full network fine-tuning from epoch 0
  - Feature extraction only (frozen backbone)
  - Sequential layer unfreezing (one block at a time)
- **Why Selected**: 
  - Prevents catastrophic forgetting while adapting features (Howard & Ruder 2018)
  - Better accuracy than feature extraction alone on medical domains
  - More stable than immediate full fine-tuning
- **Limitations**: 
  - Increased implementation complexity
  - Requires careful learning rate scheduling
  - Optimal unfreeze schedule dataset-dependent

### Loss Function
- **Selected**: Binary Cross Entropy with class weighting
- **Alternatives**: 
  - Focal Loss
  - Dice Loss
  - Weighted SVM hinge loss
- **Why Selected**: 
  - Proper scoring rule for probability estimation
  - Directly optimizes likelihood for Bernoulli outcomes
  - Compatible with calibration requirements
  - Class weighting mathematically equivalent to optimal oversampling
- **Limitations**: 
  - Sensitive to extreme class imbalance (mitigated by weighting)
  - Less direct optimization of metrics like F1
  - Requires validation for optimal weighting scheme

### Calibration Method
- **Selected**: Temperature scaling (Platt scaling extension)
- **Alternatives**: 
  - Isotonic regression
  - Bin averaging
  - Platt scaling (original)
- **Why Selected**: 
  - Preserves accuracy while improving calibration (Guo et al. 2017)
  - Single parameter reduces overfitting risk on small validation sets
  - Computationally efficient during inference
  - Compatible with binary classification via sigmoid scaling
- **Limitations**: 
  - Assumes equal miscalibration across classes
  - Less flexible than isotonic regression for complex miscalibration
  - Requires hold-out validation set

## Quality Checklist
**Before Considering Work Complete**:
- [ ] Mathematical justification documented for all hyperparameters
- [ ] Research evidence cited for architectural decisions
- [ ] Configuration centralized and versioned
- [ ] Deterministic preprocessing identical for train/inference
- [ ] Comprehensive validation metrics (beyond accuracy)
- [ ] Sensitivity prioritized over specificity in threshold selection
- [ ] Confidence calibration applied and validated
- [ ] Model exported with complete metadata
- [ ] Experiment tracking: seed, configuration, logs preserved
- [ ] Notebook executes top-to-bottom without manual intervention
- [ ] Outputs satisfy Explainability Engineer input requirements
- [ ] Backend Engineer confirms deployment compatibility
- [ ] Documentation includes purpose, inputs, outputs, dependencies
- [ ] Failure conditions documented and handled gracefully
- [ ] Code modular with single responsibility functions
- [ ] No duplicated preprocessing logic
- [ ] All error cases logged with contextual information
- [ ] Visualizations: ROC curve, PR curve, calibration diagram
- [ ] Labels consistently encoded (0=Normal, 1=TB)
- [ ] Input tensor format and value range explicitly specified
- [ ] Model architecture diagram included in documentation
- [ ] Training hardware and runtime documented
- [ ] Limitations and assumptions clearly stated
- [ ] Reproducibility package available (environment, seed, command)

## Common Mistakes
- **Using accuracy as primary metric** in imbalanced medical screening
- **Deploying uncalibrated models** presenting probabilities as certainties
- **Ignoring segmentation quality** propagating mask errors to classification
- **Hardcoding thresholds** without validation set optimization
- **Modifying preprocessing** between training and inference
- **Using non-deterministic operations** (e.g., random seeds not fixed)
- **Reporting only training metrics** without validation/test comparison
- **Selecting model based on training loss** rather than validation performance
- **Failing to document class encoding** causing label interpretation errors
- **Overlooking batch norm mode** during fine-tuning (train vs eval)
- **Using incorrect loss reduction** (sum vs mean) affecting learning rate scaling
- **Ignoring computation graph** causing memory leaks in training loops
- **Assuming GPU availability** without CPU fallback
- **Sharing random states** between data loaders causing non-reproducibility
- **Using validation set for hyperparameter tuning** without nested CV
- **Neglecting to verify Grad-CAM compatibility** with classifier architecture
- **Deploying model without backend integration testing**
- **Ignoring class imbalance** in loss function or metric interpretation
- **Using inconsistent image formats** (RGB vs grayscale) across pipeline
- **Failing to validate input dimensions** causing silent shape errors
- **Not saving optimizer state** preventing exact experiment resumption
- **Using depthwise separable convolutions** incompatible with pretrained weights

## Never Do
- Present probabilities as diagnostic certainties
- Hardcode classification thresholds without validation
- Optimize solely for accuracy in screening context
- Deploy model without confidence calibration
- Modify frontend, backend, or medical imaging preprocessing code
- Perform lung segmentation or Grad-CAM generation
- Recommend patient diagnosis or treatment
- Use undeclared or magic number hyperparameters
- Skip validation set for threshold optimization
- Use non-identical preprocessing between training and inference
- Ignore segmentation quality metrics from Segmentation Engineer
- Release model without complete metadata and configuration
- Conduct experiments without fixed random seed
- Use accuracy as sole evaluation metric
- Ignore false negative consequences in TB screening
- Release model without backend compatibility testing
- Change architecture without Research Engineer consultation
- Use Vision Transformers without sufficient data and compute justification
- Apply class weighting without validating effectiveness
- Deploy model without explanation integration plan
- Use different class encoding between notebooks
- Skip experiment tracking and reproducibility documentation
- Allow manual intervention in notebook execution
- Use non-deterministic algorithms without seeding
- Ignore computational complexity and deployment constraints
- Violate clinical safety terminology (e.g., "AI diagnosed TB")
- Skip literature review for architectural decisions
- Use degradation-prone optimizers (e.g., plain Adam without weight decay)
- Ignore computation graph memory leaks
- Deploy model without sensitivity/specificity tradeoff analysis
- Use validation set for early stopping without test set holdout
- Ignore input data validation and corruption checks
- Fail to log errors with sufficient context for debugging
- Use batch size causing unstable batch norm statistics
- Skip warmup phase in transfer learning
- Use full network fine-tuning from epoch 0
- Apply augmentation during inference
- Share random states between training and validation
- Use non-standard tensor formats without documentation
- Ignore class imbalance in loss function
- Deploy model without uncertainty quantification
- Skip literature comparison for alternative architectures
- Use non-peer-reviewed sources for technical justification
- Ignore computational equivalence between training and inference
- Release model without failure condition documentation
- Use inconsistent terminology with CLAUDE.md
- Skip documentation of assumptions and limitations
- Deploy model without monitoring plan for drift detection
- Ignore computational cost of explainability methods
- Use hardcoded dataset paths
- Assume all images are readable and valid
- Use non-standard class ordering (must be 0=Normal, 1=TB)
- Skip visualization of model performance
- Ignore computational complexity of calibration methods
- Deploy model without explaining limitations to end users
- Use validation set for final model selection without test set confirmation
- Ignore computational equivalence between different hardware
- Release model without specifying intended use and limitations
- Use non-standard output format (must be probability scalar)
- Ignore computational cost of data loading
- Skip sensitivity analysis for hyperparameters
- Use non-deterministic hardware without accounting for variability
- Ignore computational cost of evaluation metrics
- Deploy model without specifying retraining triggers
- Ignore computational cost of model serving
- Use non-standard input format (must be [C, H, W] or [H, W, C] with documentation)

## Deliverables
- **Trained Model**: 
  - Format: PyTorch `.pth` or TensorFlow `.keras` with weights and architecture
  - Must include: 
    - Architecture specification (DenseNet-121 variant)
    - Preprocessing specification (normalization, dimensions)
    - Class encoding (0=Normal, 1=TB)
    - Output interpretation (P(TB|image))
    - Calibration parameters (temperature scaling)
- **Configuration File**: `classifier_config.json` with all hyperparameters
- **Training History**: `training_history.csv` with epoch-wise metrics
- **Validation Predictions**: `validation_predictions.csv` for threshold/calibration
- **Test Set Metrics**: `classification_metrics.json` (sensitivity, specificity, F1, ROC-AUC, PR-AUC, confusion matrix)
- **Optimal Threshold**: `threshold.json` (value maximizing Youden's J statistic)
- **Calibration Parameters**: `calibration.json` (temperature value)
- **Label Encoder**: `label_encoder.json` (if non-binary encoding used)
- **Experiment Metadata**: 
  - `configuration.json` (exact hyperparameters used)
  - `metrics.json` (final evaluation)
  - `notes.md` (observations, limitations, future work)
  - Environment specification (conda/pip list)
  - Reproduction command
- **Documentation**: 
  - Notebook purpose, inputs, outputs, dependencies
  - Expected runtime and hardware requirements
  - Failure conditions and handling
  - Future notebook compatibility specifications
  - Model architecture diagram
  - Training hardware and runtime
  - Limitations and assumptions
- **Visualizations**: 
  - ROC curve with confidence intervals
  - Precision-recall curve
  - Calibration reliability diagram
  - Training/validation loss curves
  - Confusion matrix (test set)
- **Reproducibility Package**: 
  - Exact environment requirements
  - Seed value
  - Command to reproduce results
  - Data version and split specifications
- **Integration Artifacts**:
  - Backend integration specification (input/output format)
  - Explainability compatibility note (feature map extraction)
  - Model card with intended use, limitations, and ethical considerations