# Explainability Engineer

## Purpose
Design, validate, and integrate explainable AI methods for pulmonary tuberculosis screening. Transform classifier decisions into human-understandable visual explanations that support clinical review while maintaining scientific rigor and clinical safety.

## Mission
Ensure explainability component of NIRKHSHON pipeline provides mathematically grounded, clinically appropriate visual explanations that enhance trust, enable error analysis, and support structured observation generation without compromising diagnostic safety or introducing misleading interpretations.

## Responsibilities
- Receive trained TB classifier outputs from Classification Engineer
- Generate Grad-CAM, Grad-CAM++, saliency maps, and integrated gradients heatmaps
- Select appropriate target layers for explainability based on architecture and validation
- Compute feature map gradients and transform into localization maps
- Normalize and overlay heatmaps onto original and segmented lung images
- Perform anatomical localization using standardized lung zones (Left/Right Upper/Middle/Lower)
- Validate heatmap quality against lesion annotations when available (TBX11K Simplified)
- Compute explainability metrics (IoU, Dice, pointing game, AUC-based metrics)
- Export heatmaps, overlays, ROI coordinates, and explanation metadata
- Ensure explanations support structured observation generation and clinical report creation
- Never modify classifier training, segmentation models, or perform diagnosis
- Never alter frontend/backend code or medical imaging preprocessing pipelines
- Collaborate with Classification Engineer for model compatibility and feature access
- Provide outputs to Evaluation Engineer for quantitative explanation assessment
- Interface with Backend Engineer for API integration of explanation generation
- Support Frontend Engineer with standardized explanation formats for visualization

## Responsibilities Explicitly Out of Scope
- Training or fine-tuning of classification or segmentation models
- Lung segmentation or mask generation
- TB classification or probability prediction
- Clinical diagnosis or treatment recommendation
- Frontend UI development or backend API implementation
- Report generation or structuring of clinical observations
- Dataset preparation, validation, or metadata generation
- Performance optimization or deployment engineering
- Statistical analysis of clinical outcomes or population studies

## Responsibilities Explicitly Out of Scope
- Training or fine-tuning of classification or segmentation models
- Lung segmentation or mask generation
- TB classification or probability prediction
- Clinical diagnosis or treatment recommendation
- Frontend UI development or backend API implementation
- Report generation or structuring of clinical observations
- Dataset preparation, validation, or metadata generation
- Performance optimization or deployment engineering
- Statistical analysis of clinical outcomes or population studies

## Primary Notebook
- Notebook 4 (Explainability & Evaluation)

## Secondary Notebooks
- Notebook 3 (Classification) - provides classifier predictions and model
- Notebook 2 (Segmentation) - provides segmented lung images and masks
- Notebook 1 (Dataset Preparation) - provides data splits and metadata

## When This Skill Should Be Used
- After classifier produces TB probability predictions on segmented lung images
- When visual explanations are required for clinician review or research analysis
- During model validation to verify explainability aligns with anatomical expectations
- For error analysis of false positives/negatives to identify failure modes
- When preparing model cards or documentation requiring explanation examples
- Prior to backend integration to establish explanation API contracts
- During frontend development to define visualization components and interactions
- In research publications requiring quantitative explanation validation
- When generating structured observations from AI screening results

## Required Inputs
- **Trained TB Classifier**: PyTorch (.pth) or TensorFlow (.keras) model with accessible intermediate layers
- **Segmented Lung Images**: PNG/JPEG format, single-channel lung-only images from Segmentation Engineer outputs (224x224 or original resolution)
- **Preprocessing Specification**: Normalization parameters (mean/std) matching classifier training pipeline
- **Class Encoding**: 0 = Normal, 1 = TB (must match classifier training)
- **Optional Ground Truth Annotations**: Lesion bounding boxes or segmentation masks from TBX11K Simplified dataset (for validation)
- **Model Architecture Details**: Layer names/types for target layer selection (provided by Classification Engineer)
- **Configuration File**: Explainability-specific hyperparameters (thresholds, overlay settings, etc.)
- **Metadata**: Patient ID, image path, true label (when available for validation)

## Expected Outputs
- **Raw Heatmaps**: `.npy` files containing unprocessed explanation scores per pixel
- **Normalized Heatmaps**: `.npy` files scaled to [0,1] for visualization
- **Heatmap Overlays**: PNG/JPEG images showing heatmap superimposed on original chest X-ray
- **Segmented Lung Overlays**: PNG/JPEG showing heatmap on lung-only images
- **ROI Coordinates**: JSON files with bounding box or polygon coordinates of activated regions
- **Anatomical Localization**: JSON specifying dominant lung, zone, and estimated affected area percentage
- **Explainability Metrics**: JSON with IoU, Dice, pointing game accuracy, AUC-based scores (when ground truth available)
- **Explanation Metadata**: JSON documenting method, target layer, parameters, timestamp, model version
- **Visualization Grid**: Composite image showing original, segmented, heatmap, and overlay for reporting
- **API Response Schema**: JSON structure for backend explanation endpoints

## Dependencies
- **Classification Engineer**: Provides trained model with layer access and prediction capabilities
- **Medical Imaging Engineer**: Defines DICOM-to-PNG conversion standards, spatial consistency, and lung segmentation quality requirements
- **TensorFlow/PyTorch**: Deep learning frameworks for gradient computation and feature map extraction
- **NumPy**: Numerical operations for heatmap processing and metric calculation
- **OpenCV/ImageIO**: Image loading, resizing, and overlay generation
- **scikit-learn**: Metrics computation (AUC, ROC, etc.) when required
- **matplotlib/seaborn**: Optional visualization for development and validation (not exported)
- **jsonschema**: Configuration validation
- **logging**: Structured logging per CLAUDE.md standards
- ** tqdm**: Progress tracking during batch explanation generation

## Workflow
1. **Input Validation**: Verify model compatibility with segmented image dimensions and preprocessing
2. **Model Preparation**: Load classifier, register forward hooks for target layer feature maps and gradients
3. **Forward Pass**: Process segmented lung image through classifier to obtain predictions
4. **Gradient Computation**: 
   - For classification score: compute ∂score/∂feature_maps for target layer
   - For class-specific explanations: use TB class score (positive class)
5. **Feature Map Analysis**: 
   - Global average pooling of gradients to obtain neuron importance weights
   importance coefficients per feature map
   - Weighted combination of feature maps → raw heatmap
6. **Heatmap Post-processing**: 
   - ReLU activation to retain positive influence pixels
   - Normalization to [0,1] range (min-max or z-score based on validation)
   - Resizing to match input image dimensions via bilinear interpolation
   - Application of lung mask to suppress extrapulmonary activations
7. **Overlay Generation**: 
   - Convert heatmap to colormap (Jet or viridis recommended for medical imaging)
   - Blend with original image using alpha = 0.4–0.6 (configurable)
   - Create separate overlays for original chest X-ray and segmented lung image
8. **Anatomical Localization**: 
   - Threshold heatmap at Otsu's method or configurable percentile (e.g., 50th)
   - Compute connected components in thresholded region
   - Map component centroids to standardized lung zones using reference atlas
   - Determine dominant lung, zone, and affected area percentage
9. **Validation (when ground truth available)**:
   - Compute IoU and Dice between thresholded heatmap and lesion mask
   - Perform pointing game accuracy (max heatmap location within bbox)
   - Calculate explanation AUC (region-based vs pixel-based)
10. **Export**: 
    - Save raw heatmap, normalized heatmap, overlay images
    - Store ROI coordinates, anatomical localization, metrics as JSON
    - Generate explanation metadata JSON
11. **Logging**: Record input/output paths, parameters, timing, and any warnings
12. **Notification**: Signal completion to dependent skills (Evaluation, Backend, Frontend)

## Engineering Principles
- Adhere to CLAUDE.md repository-wide principles: correctness, reproducibility, modularity, explainability, maintainability
- Never sacrifice correctness for speed; prioritize robust validation over rapid generation
- Every explainability decision requires mathematical justification and peer-reviewed literature
- Configuration before hardcoding: all parameters managed in centralized configuration
- Input validation and graceful degradation for missing/corrupted data or incompatible models
- Modular design: separate functions for heatmap generation, normalization, overlay, localization
- Deterministic processing: identical outputs for same input across runs (fixed seeds where applicable)
- Comprehensive logging replacing print statements; structured formats for traceability
- Experiment tracking with unique identifiers and complete artifact preservation
- Resource awareness: avoid unnecessary GPU memory usage; batch processing for efficiency
- Reproducibility: fixed random seeds, versioned dependencies, exact command documentation

## Explainable AI Principles
- **Faithfulness**: Explanations must reflect genuine factors in model decision-making (not post-hoc rationalizations)
- **Stability**: Similar inputs should produce similar explanations (Lipschitz continuity in explanation map)
- **Complexity**: Explanations should be as simple as possible while faithful (Occam's razor)
- **Clarity**: Visualizations must be interpretable by target audience (clinicians, researchers)
- **Relevance**: Explanations should focus on features pertinent to the task (TB-specific patterns)
- **Consistency**: Same explanation method should produce comparable results across models/architectures
- **Certainty Awareness**: Explanations should reflect model confidence (low-confidence predictions yield diffuse maps)
- **Clinical Appropriateness**: Never imply causation; frame as "regions influencing screening result"
- **Uncertainty Quantification**: Where possible, indicate explanation reliability (e.g., variance across augmentations)
- **Human-in-the-loop**: Explanations support, not replace, clinician judgment per CLAUDE.md clinical safety

## Heatmap Generation
**General Formula**: 
- For target class c: \(L_{Grad-CAM}^{c} = ReLU\left(\sum_k \alpha_k^{c} A^{k}\right)\)
- Where \(A^{k}\) = k-th feature map, \(\alpha_k^{c} = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^{c}}{\partial A_{ij}^{k}}\)
- \(Z\) = number of pixels in feature map (global average pooling of gradients)

**Implementation Requirements**:
- Use automatic differentiation frameworks (PyTorch TensorFlow) for exact gradient computation
- Avoid finite difference approximations which introduce numerical errors
- Ensure gradient flow is not blocked by non-differentiable operations (use ReLU variants where needed)
- Handle multi-input architectures by computing gradients w.r.t. each input tensor
- For models with batch norm: ensure running statistics are used (eval mode) during explanation generation
- Verify gradient computation does not alter model weights (strictly inference mode)

## Grad-CAM
**Selected Method**: Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)
**Justification**:
- Produces class-discriminative localization with minimal architectural modifications
- Applicable to any CNN with global average pooling before final classification layer
- Mathematically grounded: weights feature maps by gradient influence on class score
- Computationally efficient: single backward pass per image
- Widely validated in medical imaging for lesion localization (Wang et al., 2020; CheXpert studies)
- Compatible with DenseNet-121 architecture specified in CLAUDE.md

**Engineering Guidance**:
- Target layer: Last convolutional block before global average pooling (DenseNet-121: concatenation phase of block4)
- For architectures without GAP: adapt to use global average pooling of feature maps explicitly
- Verify ReLU application: only positive gradients contribute to class increase (negative gradients indicate class suppression)
- Normalization strategy: 
  - Option A: Min-max to [0,1] per image (preserves relative intensity)
  - Option B: Z-score normalization using dataset mean/std (enables cross-image comparison)
  - Recommendation: Min-max for visualization, z-score for metric computation
- Upsampling: Bilinear interpolation preserves smoothness; avoid nearest-neighbor causing artifacts
- Lung masking: Multiply heatmap by binary lung mask to exclude extrapulmonary signals (critical for TB)

## Grad-CAM++
**Enhanced Method**: Gradient-weighted Class Activation Mapping++ (Chattopadhay et al., 2018)
**Purpose**: Improve localization for multiple occurrences of same class and better weakly-supervised object detection
**Additional Terms**:
- \(\alpha_k^{c} = \frac{\sum_i \sum_j \left(\frac{\partial^2 y^{c}}{\partial A_{ij}^{k}} \cdot ReLU\left(\frac{\partial y^{c}}{\partial A_{ij}^{k}}\right)\right)}{\sum_i \sum_j \left(\frac{\partial^2 y^{c}}{\partial A_{ij}^{k}} \cdot ReLU\left(\frac{\partial y^{c}}{\partial A_{ij}^{k}}\right)\right) + \epsilon \cdot \sum_i \sum_j A_{ij}^{k}}\)
- Addresses gradient saturation and provides more precise localization
**When to Use**:
- When Grad-CAM produces diffuse or spread-out activations
- For detecting multiple disjoint TB lesions in single lung
- When quantitative metrics show Grad-CAM IoU < 0.4 with ground truth
**Implementation Considerations**:
- Requires second-order gradients (computationally heavier but feasible for 224x224)
- Numerical stability: add epsilon (1e-8) to denominator
- Validate that increased complexity yields measurable improvement in localization metrics
- Default to Grad-CAM unless Grad-CAM++ demonstrates statistically significant improvement

## Saliency Maps
**Basic Method**: Vanilla gradient \(\left|\frac{\partial y^{c}}{\partial X}\right|\) (Simonyan et al., 2013)
**Limitations**: 
- Often noisy and discontinuous due to input saturation
- Not class-discriminative without ReLU on gradients (Guided Backprop)
- Rarely sufficient alone for medical explainability per CLAUDE.md standards
**Usage in Pipeline**:
- Primarily for comparison and validation of more sophisticated methods
- May be used as input to smoothing techniques (e.g., SmoothGrad)
- Not recommended as primary explanation for TB screening without augmentation
**Engineering Note**: 
- Compute absolute gradients to capture magnitude of influence regardless of sign
- Apply Gaussian smoothing (σ=1.0–2.0) to reduce noise if used
- Always combine with input image for visualization to provide context

## Integrated Gradients
**Axiomatic Method**: Sundararajan et al., 2017
**Formula**: 
- \(IntegratedGradients_i = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha\)
- Where \(x'\) = baseline (typically black image or mean image)
**Justification**:
- Satisfies sensitivity and implementation invariance axioms
- Less noisy than vanilla gradients; accumulates evidence along path
- Baseline selection critical: must represent absence of lung features
**Baseline Selection for TB Screening**:
- Option 1: All zeros (black image) - may introduce edge artifacts
- Option 2: Mean of training dataset segmented lungs - recommended
- Option 3: Blurred version of input (Gaussian blur σ=10) - preserves structure while removing details
**Implementation**:
- Approximate integral via Riemann sum: \(m\) steps (typically 50–200)
- \(IntegratedGradients_i \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial f(x' + \frac{k}{m}(x - x'))}{\partial x_i}\)
- Gradients computed via automatic differentiation
**When to Use**:
- When gradient-based methods fail implementation invariance (sensitive to network architecture)
- For baseline-informed explanations showing what features contribute relative to neutral input
- Validation: compare with Grad-CAM; use if qualitative improvement observed
**Considerations**:
- Computationally expensive (m forward/backward passes)
- Baseline must be applied consistently across training and explanation
- Never use random baseline; must be fixed and documented

## Target Layer Selection
**Principle**: Balance spatial resolution vs semantic depth
**Guidance by Architecture**:
- **DenseNet-121**: 
  - Block 2: High resolution, low semantics (early features)
  - Block 3: Moderate resolution, moderate semantics 
  - Block 4 (pre-GAP): Low resolution, high semantics (recommended default)
  - Concatenation layer (transition to classifier): Semantic but no spatial structure
- **ResNet**: 
  - Layer 2/3: Early features
  - Layer 3: Typical selection (res4b or similar)
  - Layer 4: High semantics, low resolution
- **General Rule**: 
  - Select layer where feature map spatial dimensions ≥ 7x7 for 224x224 input
  - Ensure layer precedes global pooling or classification weighting
  - Validate via ablation: compute explanation metrics across candidate layers
**Selection Protocol**:
1. Identify candidate convolutional layers before global pooling
2. Generate explanations for validation set subset using each candidate
3. Compute localization metrics (IoU, pointing game) against ground truth when available
4. Select layer with optimal trade-off: high localization + sufficient semantic meaning
5. Document selection rationale in configuration and metadata
**Never Do**:
- Use fully connected layers (no spatial structure)
- Choose layer after global average pooling (loses localization)
- Select based solely on highest activation magnitude
- Hardcode layer index without validation

## Feature Map Analysis
**Gradient Calculation**:
- For score \(y^c\): compute \(\frac{\partial y^c}{\partial A^{k}}\) via autograd
- Average spatially: \(\alpha_k^{c} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial y^c}{\partial A_{ij}^{k}}\)
- Equivalent to global average pooling of gradient feature maps
**Weighted Combination**:
- \(L^{c}_{raw} = \sum_{k=1}^{K} \alpha_k^{c} A^{k}\)
- Where K = number of feature maps in target layer
- Results in 2D heatmap of same spatial dimensions as feature maps
**Implementation Details**:
- Use `torch.autograd.grad` or TensorFlow `GradientTape` for exact gradients
- Retain computational graph only for gradient computation; release after
- For batch processing: compute gradients per sample to avoid memory issues
- Verify gradient shapes match feature map dimensions
- Handle NaN/Inf gradients: replace with zero and log warning

## Heatmap Normalization
**Purpose**: Convert raw explanation scores to comparable visualization scale
**Methods**:
1. **Per-Image Min-Max**: 
   - \(L_{norm} = \frac{L_{raw} - \min(L_{raw})}{\max(L_{raw}) - \min(L_{raw})}\)
   - Preserves relative contrast within image
   - Default for visualization; enables intuitive hot/warm/cold mapping
2. **Dataset Z-Score**:
   - \(L_{norm} = \frac{L_{raw} - \mu_{dataset}}{\sigma_{dataset}}\)
   - Requires precomputed dataset mean/std from explanation outputs
   - Enables cross-image comparison and threshold consistency
   - Use for metric computation and statistical analysis
3. **Percentile Clipping**: 
   - Clip to [p_low, p_high] percentile (e.g., 2nd–98th) before min-max
   - Reduces outlier influence from noise or artifacts
4. **Sigmoid Scaling**: 
   - \(L_{norm} = \frac{1}{1 + \exp(-\beta \cdot L_{raw})}\)
   - β controls steepness; requires validation
**Engineering Requirements**:
- Apply identical normalization during training/validation/explanation generation
- Never normalize using validation set statistics for training explanations (data leakage)
- Save normalization parameters with model/configuration for inference consistency
- Validate that normalization does not invert explanation meaning (check monotonicity)
- For overlay generation: use min-max normalized heatmap to ensure full colormap utilization

## Heatmap Overlay
**Color Mapping**:
- Use perceptually uniform colormaps for medical imaging: 
  - Viridis (default) - colorblind friendly, luminance monotonic
  - Plasma - vibrant alternative
  - Jet (deprecated but common) - use only if required for legacy compatibility
  - Avoid rainbow colormaps due to perceptual non-uniformity
**Blending Formula**:
- \(Overlay = (1 - \alpha) \cdot Original + \alpha \cdot Colormap(L_{norm})\)
- Where α = overlay weight (typically 0.4–0.6)
- Original and colormap must be same dimensions and channels
**Preprocessing Steps**:
- Convert grayscale Original to 3-channel by replication
- Ensure both images in same color space (RGB) and range [0,255] or [0,1]
- Apply lung mask to colormap layer before blending to restrict overlay to lung region
**Export Specifications**:
- Format: PNG (lossless) for archival; JPEG (quality≥90) for web if size critical
- Bit depth: 8-bit per channel (standard)
- Metadata: Embed explanation method, target layer, timestamp as PNG tEXt chunks
- Validation: Verify overlay preserves anatomical landmarks (clavicles, diaphragm)
**Never Do**:
- Use α > 0.7 obscuring original image details
- Apply overlay without lung mask causing confusing extrapulmonary highlights
- Use non-perceptual colormaps misleading intensity interpretation
- Export overlays with inconsistent dimensions across pipeline

## Anatomical Localization
**Standard Lung Zones** (per CLAUDE.md):
- Left Upper (LUL), Left Middle (LML), Left Lower (LLL)
- Right Upper (RUL), Right Middle (RML), Right Lower (RLL)
- Based on anatomical atlas aligned to posterior-anterior chest X-ray
**Localization Pipeline**:
1. **Thresholding**: 
   - Generate binary mask: \(M = \{L_{norm} > \tau\}\) 
   - τ determined by: Otsu's method, fixed percentile (e.g., 50th), or Youden's J from validation ROC
   - Validate threshold stability across similar images
2. **Connected Components**: 
   - Label connected regions in M (8-connectivity)
   - Filter components by area (remove noise: min 10 pixels)
3. **Centroid Calculation**: 
   - For each component: \(C_x = \frac{\sum x_i}{N}, C_y = \frac{\sum y_i}{N}\)
4. **Zone Mapping**: 
   - Use predefined zone masks (same resolution as input) 
   - Assign component to zone with maximum overlap centroid
   - If centroid falls between zones, assign to nearest zone center
5. **Summary Statistics**: 
   - Dominant lung: Left/Right based on total activated pixel count
   - Dominant zone: Zone with largest activated area
   - Affected area percentage: \(\frac{\sum M}{lung\_mask\_area} \times 100\)
   - Lesion count: Number of valid connected components
6. **Spatial Spread**: 
   - Compute bounding box: [x_min, y_min, x_max, y_max]
   - Calculate eccentricity and orientation for elongation assessment
**Validation**: 
- Compare centroid locations with radiologist-annotated points when available
- Compute zone-level IoU with expert-delineated regions
**Never Do**:
- Use hardcoded zone boundaries without anatomical reference
- Report localization without lung mask (extrapulmonary false positives)
- Assign zones based on raw heatmap without thresholding
- Ignore laterality flip for patient orientation (PA view: patient left = image right)

## Bounding Box Validation
**When Ground Truth Annotations Available** (TBX11K Simplified):
- **Intersection over Union (IoU)**: 
  - \(IoU = \frac{|M \cap G|}{|M \cup G|}\) 
  - M = thresholded heatmap mask, G = ground truth mask
  - Reports overlap quality; threshold at 0.5 for decent localization
- **Dice Similarity Coefficient (DSC)**: 
  - \(DSC = \frac{2|M \cap G|}{|M| + |G|}\)
  - Equivalent to F1-score for binary masks
- **Pointing Game Accuracy**: 
  - Hit if \(\arg\max(L_{norm})\) falls within ground truth bounding box
  - Measures ability to localize most salient point within lesion
- **AUC-based Metrics**: 
  - Sweep threshold to generate ROC curve for pixel classification
  - Compute AUC for explanation map as lesion detector
  - Compute precision-recall curve and AUC-PRC
- **Center of Mass Distance**: 
  - Euclidean distance between heatmap centroid and ground truth centroid
  - Normalize by image diagonal for scale invariance
**Reporting**:
- Always report mean and std across validation set
- Include confidence intervals (bootstrap 95% CI)
- Stratify by lesion size, laterality, and anatomy when sample size permits
**Engineering Rules**:
- Never select threshold to maximize metric on validation set (use fixed rule)
- Report both raw and normalized metric values
- Validate that explanation metrics correlate with human assessment
- Use pixel-wise metrics only when ground truth is pixel-accurate
- For bounding box ground truth, convert to mask for IoU/DSC

## Explainability Metrics
**Supplementary Metrics**:
- **Sparsity**: Percentage of pixels with explanation > threshold (lower = more focused)
- **Entropy**: \(-\sum p log p\) where p = normalized explanation distribution (lower = more concentrated)
- **AUC-Judd**: Area under curve for fixation prediction (if eye-tracking available)
- **Insurance Accuracy**: Proportion of pixels where explanation rank correlates with perturbation impact
- **Stability**: Standard deviation of explanation under input perturbations (e.g., small noise)
- **Complexity**: Number of connected components in thresholded explanation
**Usage**:
- Report alongside primary metrics for comprehensive evaluation
- Never use sparsity or entropy as primary optimization objective
- Validate metric stability across explanation methods and architectures
- Document metric sensitivity to hyperparameters (threshold, colormap, etc.)

## Human Interpretability
**Principles**:
- Explanations must align with clinician expectations of TB patterns
  - Typical TB: upper lobe predilection, cavitation, infiltrate patterns
  - Atypical TB: lower lobe, miliary, normalization in immunosuppressed
- Heatmap should highlight regions consistent with known radiographic signs
- Avoid explanations that highlight purely anatomical structures (e.g., clavicles, ribs) without pathological context
**Validation Approaches**:
- **Qualitative**: 
  - Radiologist review of explanation overlay (agreement score)
  - Compare with GBT (grounded theory) coding of clinician descriptions
- **Quantitative**: 
  - Correlation between explanation intensity and radiologist confidence scores
  - Rank-order correlation with lesion severity scores
- **Task-Based**: 
  - Measure clinician diagnosis accuracy with/without explanations
  - Time-to-decision reduction with explanations
  - Trust calibration: concordance between explanation confidence and clinician trust
**Never Do**:
- Assume explanation quality based solely on metric scores
- Ignore clinical feedback on explanation relevance
- Use explanations to override clinician judgment
- Present explanations as definitive evidence of pathology

## Clinical Safety
**Terminology** (per CLAUDE.md):
- Never: "AI diagnosed TB", "Definitive Diagnosis", "Confirmed TB"
- Always: "AI Screening Result", "Preliminary Finding", "Suspicious for Pulmonary Tuberculosis", "Screening Recommendation"
**Explanation Framing**:
- Heatmap shows "regions influencing the AI screening result"
- Never imply causation: "AI found TB here" → "AI screening result influenced by patterns here"
- Always include uncertainty: "Low confidence screening result; explanation may be unreliable"
- Never hide low explanation quality: flag when IoU < 0.3 or entropy > threshold
**Display Guidelines**:
- Overlay opacity should not obscure original pathology
- Provide side-by-side original/explanation for comparison
- Include disclaimer: "Explanation is interpretive aid; clinician retains final authority"
- For low-confidence predictions (<0.6 or >0.4 depending on threshold): show explanation with reduced opacity + warning banner
**Error Handling**:
- If explanation generation fails: return original image with error overlay and log details
- Never crash pipeline; degrade gracefully to baseline explanation (e.g., uniform heatmap)
- Validate explanation does not contain NaN/Inf values before export
**Bias Mitigation**:
- Monitor explanation disparities across demographic subgroups (age, sex, ethnicity)
- Ensure lung mask quality consistent across populations to avoid artefactual explanations
- Validate explanation performance does not degrade with image acquisition variations

## Validation Strategy
**Protocol**:
1. **Unit Tests**: 
   - Verify gradient computation matches finite difference for simple networks
   - Test normalization preserves rank order
   - Validate overlay blending math
2. **Sanity Checks**: 
   - Uniform input → uniform explanation (or baseline-defined)
   - Constant model → constant explanation
   - Class-agnostic model → explanations independent of class
3. **Application-Based**:
   - **Input Invariance**: Explanation for x and x+constant should differ only by baseline shift (Integrated Gradients axiom)
   - **Class Sensitivity**: Explanations for different classes should differ when model is discriminative
   - **Localization**: When ground truth available, IoU > 0.4 and pointing game accuracy > 0.6 for acceptable performance
   - **Faithfulness (Removal)**: 
     - Sort pixels by explanation descending
     - Iteratively mask top-k% pixels and re-run classifier
     - Measure drop in score should be faster than random masking
   - **Faithfulness (Insertion)**: 
     - Start from baseline, insert top-k% pixels
     - Measure score increase should be faster than random insertion
4. **Cross-Method Agreement**: 
   - Rank correlation between Grad-CAM, Grad-CAM++, Integrated Gradients explanations
   - Expect moderate to high correlation for faithful methods
5. **Human Alignment** (when possible): 
   - Radiologist explanation preference study
   - Correlation with anatomic region likelihood scores
**Frequency**:
- Run validation on every model checkpoint during training
- Hold-out test set explanation evaluation after training completes
- Report validation metrics alongside classification metrics in experiment tracking
**Never Do**:
- Validate explanations solely on visual appeal
- Use training set for explanation metric optimization
- Ignore explanation failure modes in deployment monitoring
- Deploy explanation method without unit test coverage
- Compare explanation metrics across different input resolutions without rescaling

## Error Handling
**Principles**: Defensive programming with graceful degradation per CLAUDE.md
**Specific Cases**:
1. **Model Incompatibility**: 
   - Check: Target layer exists, output dimensions compatible
   - Action: Log ERROR with model architecture details, skip explanation, return original image + error mask
   - Never: Silent fallback to random layer
2. **Gradient Failure**: 
   - Check: NaN/Inf in gradients, vanishing gradients (<1e-10)
   - Action: Reduce input perturbation size (for IG), add epsilon, revert to guided backprop if applicable
   - Log WARNING with statistics; continue with available gradients
3. **Image Issues**: 
   - Check: Incorrect dimensions, NaN/Inf values, failed lung mask load
   - Action: Resize with warning if within 10% tolerance; else skip and log ERROR
4. **Normalization Errors**: 
   - Check: Division by zero in min-max (constant heatmap)
   - Action: Return uniform heatmap (~0.5) and log WARNING
5. **Resource Exhaustion**: 
   - Check: OOM during batched IG or gradient computation
   - Action: Reduce batch size, fallback to single-image processing, log WARNING
6. **Validation Mismatch**: 
   - Check: Ground truth mask dimensions != explanation dimensions
   - Action: Resize mask with interpolation (nearest neighbor for binary), log WARNING
7. **Configuration Errors**: 
   - Check: Invalid threshold, colormap, or alpha values
   - Action: Use safe defaults (threshold=0.5, colormap=viridis, alpha=0.5), log ERROR with expected range
**Logging**: 
- All errors logged with traceback, input path, model version, and timestamp
- Distinguish ERROR (pipeline blocking) vs WARNING (degraded quality)
- Never expose internal stack traces to end users; show user-friendly messages
**Recovery**:
- For batch processing: failed samples marked and continued
- For single sample: return explicable fallback with status flag
- Never exit entire process for recoverable errors

## Logging
**Hierarchy** (per CLAUDE.md):
1. **Structured Logging** (primary): Python `logging` module
   - Format: `%(asctime)s - %(name)s - %(level)s - [%(filename)s:%(lineno)d] - %(message)s`
   - Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - Handlers: 
     - Console: INFO+ (format simplified for terminal)
     - File: DEBUG+ (rotating: 10 MB max, 5 files, UTC timestamps)
2. **Progress Tracking** (secondary): `tqdm` progress bars
   - Outer loop: images processed
   - Inner loop: IG steps or TTA augmentations (if used)
   - Postfix: current explanation method, average processing time
3. **Experiment Tracking** (tertiary): 
   - CSV: timestamp, image ID, method, target layer, processing time, key metrics
   - TensorBoard: explanation overlays (sample), heatmap histograms, metric trends
   - Metadata: explanation_config.json, run-specific parameters
**Specific Logs**:
- Model loading: architecture, layers, device
- Input validation: dimensions, lung mask match, preprocessing compliance
- Heatmap generation: method, target layer, gradient stats, computation time
- Thresholding: method selected, threshold value, component count
- Export: file paths, sizes, success/failure counts
- Validation: metric values per batch, aggregate statistics
**Never Do**:
- Log raw explanation values or patient identifiers in standard logs
- Use inconsistent log levels across modules
- Log to multiple uncontrolled locations (e.g., both file and syslog without rotation)
- Rely solely on print statements for diagnostic information
- Log未加密的 sensitive metadata in shared environments

## Configuration
**File**: `explainability_config.json`
**Location**: `notebooks/04_gradcam_reporting/config/` or experiment directory
**Schema**:
```json
{
  "explanation": {
    "methods": ["gradcam", "gradcam++", "integrated_gradients"],
    "default_method": "gradcam",
    "target_layer_selection": {
      "strategy": "validation_based",
      "candidate_layers": ["features.denseblock4", "features.transition3"],
      "metric": "iou",
      "min_spatial_dim": 7
    }
  },
  "heatmap": {
    "normalization": {
      "type": "minmax",
      "percentile_clip": [2, 98],
      "epsilon": 1e-08
    },
    "colormap": {
      "name": "viridis",
      "reversed": false
    },
    "lung_masking": true,
    "alpha": 0.5
  },
  "localization": {
    "zone_atlas": "standard_pa_chest_xray.json",
    "threshold_method": "otsu",
    "min_component_area": 10,
    "report_affected_area": true
  },
  "overlay": {
    "format": "png",
    "quality": 95,
    "embed_metadata": true
  },
  "validation": {
    "iou_threshold": 0.5,
    "dice_threshold": 0.5,
    "pointing_game_threshold": 0.5,
    "bootstrap_samples": 1000
  },
  "performance": {
    "batch_size": 8,
    "num_workers": 4,
    "pin_memory": true,
    "mixed_precision": false,
    "integrated_gradients_steps": 50
  },
  "safety": {
    "min_confidence_for_explanation": 0.4,
    "low_confidence_opacity": 0.3,
    "explanation_failure_fallback": "uniform_heatmap",
    "clinical_disclaimer_required": true
  },
  "seed": 42,
  "deterministic": true
}
```
**Management Rules**:
- Never hardcode values in explanation scripts
- All parameters must appear in configuration with justification comments
- Configuration versioned with experiments using git-lfs or DVC
- Validate config schema at startup using `jsonschema`
- Environment-specific overrides via prefixed variables (e.g., `EXPLAIN_BATCH_SIZE`)
- Documentation: each parameter requires rationale in config comments
- **Never Do**: 
  - Magic numbers in explanation code
  - Inconsistent configuration across notebooks
  - Untracked hyperparameter changes
  - Hardcoded file paths (use relative or env vars)

## Documentation Requirements
**Per Notebook** (04_gradcam_reporting.ipynb):
- **Purpose**: One sentence: "Generate explainability visualizations and metrics for TB classifier predictions on segmented lung images."
- **Inputs**: 
  - Trained classifier (.pth/.keras) with layer access specification
  - Segmented lung images (PNG/JPEG, single-channel, normalized)
  - Preprocessing parameters (mean/std) matching classifier training
  - Optional: lesion masks/bounding boxes from TBX11K Simplified
  - Configuration file (explainability_config.json)
- **Outputs**: 
  - `heatmaps/` directory: raw (.npy), normalized (.npy) 
  - `overlays/` directory: original_overlay.png, segmented_overlay.png
  - `rois/` directory: roi_coordinates.json per image
  - `localization/` directory: anatomical_localization.json per image
  - `metrics/` directory: explanation_metrics.json (aggregate), per-image_metrics.json
  - `metadata/` directory: explanation_run_metadata.json
  - `logs/` directory: explanation_processing.log
- **Dependencies**: 
  - Python ≥3.9, PyTorch ≥1.12 or TensorFlow ≥2.8
  - NumPy, OpenCV, scikit-learn, matplotlib (dev only), tqdm
- **Expected Runtime**: 
  - CPU: 2–5 sec/image (Grad-CAM), 10–25 sec/image (IG with 50 steps)
  - GPU: 0.2–0.5 sec/image (Grad-CAM), 1–3 sec/image (IG)
  - Scale linearly with batch size and image resolution
- **Exported Files**: 
  - `explanation_metadata.json`: {method, target_layer, config_hash, timestamps, git_commit}
  - `model_compatibility.txt`: {model_architecture, input_shape, output_layer, layer_names}
  - `validation_summary.csv`: {image_id, iou, dice, pointing_game, entropy, sparsity}
- **Failure Conditions**: 
  - Missing model file → ERROR with path suggestion
  - Incompatible model architecture → ERROR with supported architectures list
  - Corrupted segmentation mask → WARNING, skip image, continue
  - OOM during explanation → WARNING, reduce batch size, retry
- **Future Notebook Compatibility**: 
  - Specify input tensor format: [C, H, W] PyTorch or [H, W, C] TensorFlow
  - Value range: [0, 1] after ImageNet normalization
  - Lung mask format: binary PNG, same dimensions as input
  - Class encoding: 0 = Normal, 1 = TB (must match classifier)
  - Explanation value range: [0, 1] normalized heatmap
  - ROI format: {"x_min": int, "y_min": int, "x_max": int, "y_max": int, "centroid": [float, float]}
  - Localization format: {"dominant_lung": "L"/"R", "dominant_zone": e.g., "RUL", "affected_area_percent": float, "component_count": int}
**Per Experiment** (experiments/experiment_001/):
- `configuration.json`: exact hyperparameters used
- `metrics.json`: final explanation validation metrics (if ground truth available)
- `notes.md`: 
  - Key observations (e.g., "Grad-CAM++ improved IoU by 0.07 for multi-lesion cases")
  - Limitations encountered (e.g., "Explanations diffuse for low-contrast predictions")
  - Future work ideas (e.g., "Investigate attention rollout for ViT backbones")
  - Deviations from plan (e.g., "Switched to guided backprop due to gradient noise")
- Reproducibility Package:
  - Seed value
  - Environment requirements (conda env export or pip freeze)
  - Exact command to reproduce (python -m explainability.run --config ...)
**Never Do**:
- Incomplete documentation omitting failure conditions
- Undocumented assumptions (e.g., "assumes lung mask present")
- Missing exported files specified in outputs section
- Inconsistent terminology with CLAUDE.md (e.g., using "diagnosis")
- Failure to specify input/output contracts causing integration issues

## Research Evidence and Alternatives
**For Every Major Decision** (example format):

### Explainability Method Selection
- **Selected**: Grad-CAM as primary, Grad-CAM++ for multi-lesion cases, Integrated Gradients for baseline-sensitive applications
- **Alternatives**: 
  - Vanilla Saliency, Guided Backprop, SmoothGrad, LIME, SHAP, Feature Perturbation, Attention Rollout
- **Why Selected**: 
  - Grad-CAM: Strong theoretical foundation (gradient-weighted activation), efficiency, validation in chest X-ray literature (Irvin et al., 2019; Wang et al., 2020)
  - Grad-CAM++: Addresses Gradient-CAM diffusion for multiple objects; proven improvement in weakly supervised localization
  - Integrated Gradients: Axiomatic foundation; less noisy than pure gradients; baseline flexibility
  - Overall: Balance of fidelity, efficiency, and clinical interpretability per explainability principles
- **Limitations**: 
  - Grad-CAM: limited to CNNs with global pooling; diffusion for scattered features
  - Grad-CAM++: ~2× computational cost; marginal improvement for single dominant lesions
  - Integrated Gradients: m× cost; baseline selection sensitivity; requires careful baseline choice
  - Never use LIME/SHAP for image explanations without superpixel segmentation (unstable, slow)
  - Never use raw saliency without smoothing (noisy, non-faithful)

### Target Layer Selection Strategy
- **Selected**: Validation-based selection using IoU against ground truth (when available), fallback to pre-GAP block
- **Alternatives**: 
  - Fixed layer (e.g., last conv block)
  - Maximum activation magnitude
  - Gradient norm maximization
  - Feature map variance maximization
- **Why Selected**: 
  - Empirical alignment with localization quality; avoids semantic vs resolution trade-off guesswork
  - Validation-based adapts to dataset characteristics (lesion size, image resolution)
  - Falls back to anatomically motivated default when ground truth unavailable
- **Limitations**: 
  - Requires ground truth subset for selection (mitigated by using small validation sample)
  - May overfit to validation set if not using hold-out for final evaluation
  - Computationally expensive to evaluate multiple candidates (mitigated by caching feature maps)

### Heatmap Normalization Choice
- **Selected**: Per-image min-max with percentile clipping (2nd–98th)
- **Alternatives**: 
  - Z-score using dataset statistics
  - Plain min-max without clipping
  - Sigmoid scaling
  - Rank-based normalization
- **Why Selected**: 
  - Min-max preserves relative contrast critical for clinician intuition
  - Percentile clipping removes outlier noise from artifacts or numerical errors
  - Z-score less intuitive for visualization; requires sufficient explanation samples
  - Plain min-max sensitive to single-pixel outliers causing washed-out displays
- **Limitations**: 
  - Per-image normalization hinders cross-image comparison (mitigated by providing z-score option)
  - Clipping thresholds dataset-dependent; validated on TBX11K Simplified
  - Never use normalization that inverts explanation meaning (verified via monotonicity check)

### Lung Mask Application
- **Selected**: Multiply explanation by binary lung mask before normalization and overlay
- **Alternatives**: 
  - Apply mask after normalization
  - No lung masking (full field explanations)
  - Mask generated from explanation itself (segmentation-based)
- **Why Selected**: 
  - Prevents misleading extrapulmonary explanations that reduce clinical trust
  - Ensures explanation focuses on lung parenchyma where TB manifests
  - Applied before normalization to avoid scaling suppression of lung signals
  - Using classifier input lung mask guarantees spatial alignment
- **Limitations**: 
  - Depends on segmentation quality; errors in mask propagate to explanations
  - Never use explanation-generated mask (circularity; defeats purpose)
  - Validate: if segmentation Dice < 0.90, flag explanation as potentially unreliable

### Overlay Blending Parameters
- **Selected**: Alpha = 0.5, viridis colormap, lung-masked colormap layer
- **Alternatives**: 
  - Higher alpha (0.7+) for more vivid explanation
  - Jet colormap for legacy compatibility
  - Blending in HSV space
  - No colormap (grayscale explanation)
- **Why Selected**: 
  - Alpha 0.5 balanced visibility of original anatomy and explanation intensity
  - Viridis perceptually uniform, colorblind safe, luminance monotonic (critical for grayscale interpretation)
  - Lung-masked colormap prevents confusing highlights outside lungs
  - HSV blending causes hue shifts inconsistent with intensity changes
- **Limitations**: 
  - Alpha optimal range may vary with display calibration; provide config adjustment
  - Never use jet colormap without clinician validation (known perceptual artifacts)
  - Grayscale explanation lacks intuitive hot/cold metaphor for clinicians

### Validation Metric Prioritization
- **Selected**: IoU and pointing game accuracy as primary; Dice and AUC-PRC as secondary
- **Alternatives**: 
  - Accuracy, precision, recall at fixed threshold
  - Correlation with radiologist scores
  - Explanation mass within lung vs outside
- **Why Selected**: 
  - IoU standard for segmentation overlap; intuitive 0–1 scale
  - Pointing game measures ability to highlight most salient point within lesion (clinically relevant)
  - Dice complementary to IoU; AUC-PRC threshold-independent for imbalanced explanation evaluation
  - Avoid accuracy due to explanation map sparsity causing misleading high scores
- **Limitations**: 
  - IoU sensitive to threshold choice; mitigate with fixed rule (Otsu)
  - Pointing game requires bounding box ground truth; use center point if only mask available
  - Never optimize directly for metric without clinical validation

## Interaction with Other Skills
**Consumes From**:
- **Classification Engineer**: 
  - Trained TB classifier with accessible convolutional layers
  - Model architecture specification: layer names, input preprocessing, output interpretation
  - Prediction probabilities for confidence-based explanation gating
  - Validation set splits and metadata for explanation evaluation
- **Medical Imaging Engineer**: 
  - Standardized DICOM-to-PNG conversion preserving spatial resolution and orientation
  - Lung mask quality requirements (Dice > 0.90 w.r.t. ground truth)
  - Anatomical zone atlas definition referenced in configuration
  - Image orientation conventions (PA view, left/right markers)
- **Produces For**:
  - **Evaluation Engineer**: 
    - Raw and normalized heatmaps for metric computation
    - Overlay images for qualitative review
    - ROI coordinates and anatomical localization for structured observation generation
    - Explanation metrics (IoU, Dice, pointing game, entropy) for calibration analysis
    - Failure case explanations for error analysis
  - **Backend Engineer**: 
    - Explanation generation API contract: input (image path, model ID), output (heatmap, overlay, localization JSON)
    - Model compatibility requirements: layer access, tensor format, device placement
    - Explanation metadata format for caching and versioning
    - Fallback behavior specifications for robustness
  - **Frontend Engineer**: 
    - Standardized explanation formats: PNG overlays, JSON localization, metadata schema
    - Recommended visualization components: side-by-side original/explanation, adjustable opacity toggle
    - Interaction patterns: click-on-heatmap to show ROI statistics, zoom/pan synchronization
    - Accessibility considerations: colorblind-safe colormaps, keyboard navigation, screen reader descriptions
- **Collaborates With**:
  - **Research Engineer**: 
    - Literature justification for explanation method selections
    - Novelty statement for explanation validation approach in TB screening
    - Limitations and future work documentation (e.g., temporal explanations for video)
  - **AI Mathematics Engineer**: 
    - Gradient computation correctness proofs
    - Numerical stability analysis for integrated gradients
    - Metric sensitivity to hyperparameters and baseline selection
- **Integration Points**:
  - Input: segmented lung image, file path or tensor, model reference/ID
  - Output: explanation bundle (heatmap, overlay, localization, metadata) as files or JSON
  - Timing: Synchronous for single image; batched async for backend
  - Error Handling: Explicit error codes/messages for missing model, incompatible shape, OOM
  - Security: Never expose model weights or training data in explanation outputs
  - Performance: Target latency <1 sec/image for Grad-CAM on GPU; <5 sec/image for IG
**Never Do**:
- Assume different tensor format without validation and conversion
- Modify explanation outputs without updating Interface Control Documents (ICDs)
- Ignore timing constraints from Backend Engineer (e.g., real-time explanation requirement)
- Deploy explanation method without Frontend Engineer UI/UX review
- Change lung mask source without Medical Imaging Engineer re-validation
- Use classification probabilities without Confirmation from Classification Engineer
- Skip explanation validation with Evaluation Engineer before release

## Quality Checklist
**Before Considering Work Complete**:
- [ ] Mathematical justification documented for all explanation methods
- [ ] Research evidence cited for each selected technique (Grad-CAM, Grad-CAM++, IG)
- [ ] Configuration centralized, versioned, and validated at runtime
- [ ] Deterministic processing: identical outputs for same input across runs
- [ ] Input validation: model compatibility, image dimensions, lung mask alignment
- [ ] Comprehensive explanation metrics reported (beyond visual appeal)
- [ ] Sensitivity prioritized in explanation thresholds for screening context
- [ ] Confidence gating: explanations only generated for predictions within clinically relevant range
- [ ] Explanations validated against ground truth when available (TBX11K Simplified)
- [ ] Heatmap normalization preserves explanation rank order (monotonicity verified)
- [ ] Overlay generation uses perceptually uniform colormap (viridis/default)
- [ ] Lung masking applied before normalization to prevent signal suppression
- [ ] Anatomical localization uses standardized zone reference atlas
- [ ] Explanation metadata includes method, target layer, parameters, timestamps, git hash
- [ ] Experiment tracking: seed, configuration, logs, environment preserved
- [ ] Notebook executes top-to-bottom without manual intervention
- [ ] Outputs satisfy Backend Engineer API contract and Frontend Engineer visualization needs
- [ ] Documentation includes purpose, inputs, outputs, dependencies, runtime estimates
- [ ] Failure conditions documented and handled gracefully (degrade to fallback)
- [ ] Code modular: separate functions for heatmap gen, normalization, overlay, localization
- [ ] No duplicated explanation logic (e.g., gradient computation shared)
- [ ] All error cases logged with contextual information (input, model, step)
- [ ] Visualizations: sample overlays, heatmap histograms, explanation metric trends
- [ ] Lung mask compliance: verify segmentation Dice > 0.90 before explanation generation
- [ ] Model architecture diagram included in documentation with target layer highlighted
- [ ] Training/inference hardware and runtime documented per experiment
- [ ] Limitations and assumptions clearly stated (e.g., "assumes lung mask present")
- [ ] Reproducibility package available (environment, seed, command)
- [ ] Explanation outputs correctly integrated with structuring observation generation
- [ ] Clinical safety terminology verified in all outputs and disclaimers
- [ ] Explanation failure fallback defined and tested (uniform heatmap or error overlay)
- [ ] Resolution independence: explanation resizes correctly with input dimensions
- [ ] Batch processing reliability: no memory leaks, consistent results across batch sizes
- [ ] Deterministic algorithms seeded where applicable (e.g., IG step selection)
- [ ] Computational complexity analyzed and documented (Big O per method)
- [ ] Deployment compatibility verified with Backend Engineer (API latency, error handling)
- [ ] Human interpretability validated with clinician feedback when possible
- [ ] Uncertainty quantification: explanation variance reported for stochastic methods
- [ ] Baseline sensitivity analyzed for Integrated Gradients (if used)
- [ ] Second-order gradient validity confirmed for Grad-CAM++
- [ ] Class discrimination verified: explanations differ for TB vs Normal predictions
- [ ] Input invariance tests passed (for Integrated Gradients axiom)
- [ ] Model usage strictly inference mode: no gradient weight updates
- [ ] Random seeds fixed for any stochastic explanation components
- [ ] Environment isolation: explanation works identically across OS, Python versions
- [ ] Base case validation: uniform input → explainable uniform or baseline output
- [ ] Class constant model test: explanation independent of input when model predicts constant
- [ ] Gradient flow validation: confirm non-zero gradients for target layer
- [ ] Explainability does not override clinician judgment: disclaimers present
- [ ] Explanation export formats lossless for archival (PNG, NPY)
- [ ] Metadata completeness: enables exact experiment reproduction
- [ ] Integration tests: end-to-end pipeline from segmentation to explanation
- [ ] Regression tests: baseline explanation quality across model checkpoints
- [ ] Stress tests: large batch sizes, extreme image sizes, corrupted inputs
- [ ] Security review: no leakage of sensitive data in logs or outputs
- [ ] Performance benchmarks: latency, throughput, memory usage reported
- [ ] Compatibility matrix: tested with PyTorch/TensorFlow versions specified

## Common Mistakes
- **Using explanation as diagnostic proof** rather than interpretive aid
- **Selecting target layer by heuristic without validation** leading to poor localization
- **Ignoring lung mask application** causing confusing extrapulmonary highlights
- **Using non-perceptual colormaps** (e.g., Jet) misleading intensity interpretation
- **Failing to normalize explanation** causing inconsistent overlay brightness
- **Applying lung mask after normalization** suppressing true signal strength
- **Using explanation metrics without ground truth** leading to circular validation
- **Optimizing explanation thresholds on validation set** causing overfitting
- **Deploying explanation method without unit tests** for gradient computation
- **Ignoring batch dimension in gradient computation** causing shape mismatches
- **Using finite difference approximations** for gradients introducing numerical error
- **Neglecting to set model to eval mode** during explanation (batch norm issues)
- **Applying explanations to full chest X-ray without lung segmentation** reducing specificity
- **Using explanation to override clinician judgment** violating clinical safety
- **Presenting explanation uncertainty as certainty** (e.g., "definitively shows TB here")
- **Failing to document baseline selection** for Integrated Gradients irreproducibility
- **Using explanation from different model checkpoint than classification** causing mismatch
- **Ignoring explanation computation time** causing backend latency issues
- **Using explanation method incompatible with model architecture** (e.g., ViT without adaptation)
- **Applying explanation to unnormalized images** breaking assumptions
- **Using explanation without confidence gating** showing maps for random noise
- **Failing to validate explanation stability** under small perturbations (e.g., noise)
- **Using explanation metrics that favor dense maps** (e.g., mean explanation value)
- **Ignoring متعدد-lesion cases** where Grad-CAM diffuses and Grad-CAM++ needed
- **Using explanation for report generation without structuring** creating unstructured text
- **Deploying explanation without backend API versioning** causing breaking changes
- **Using explanation outputs with inconsistent coordinate systems** (e.g., mixing pixel/mm)
- **Failing to explain explanation method to end users** causing misuse or overtrust
- **Using explanation to replace visual inspection** rather than complement it
- **Neglecting to validate explanation generalization** across acquisition protocols
- [ ] Assuming explanation quality correlates directly with classification accuracy
- [ ] Using explanation without anatomical localization losing clinical context
- [ ] Applying explanation to batch with mixed resolutions without resizing
- [ ] Using explanation method requiring labels (e.g., class discriminant analysis) in unsupervised setting
- [ ] Sharing random state between explanation and other processes causing non-reproducibility
- [ ] Using explanation without checking for NaN/Inf in outputs
- [ ] Forgetting to log explanation generation failures by image ID
- [ ] Using explanation method that requires retraining (e.g., training surrogate models)
- [ ] Applying explanation to 3D volumes without slice-by-slide adaptation
- [ ] Using explanation without verifying it respects model symmetries (e.g., left-right flip)
- [ ] Using explanation for reportable findings without radiologist consensus
- [ ] Assuming explanation transferability across domains without validation
- [ ] Using explanation without verifying it satisfies explainability axioms (when claimed)
- [ ] Using explanation from a model trained with different preprocessing
- [ ] Using explanation without checking for sensitivity to hyperparameters (e.g., alpha, threshold)
- [ ] Using explanation without documenting assumptions about lung laterality
- [ ] Using explanation without verifying compatibility with structuring observation generation
- [ ] Using explanation without specifying intended use and limitations in model card
- [ ] Using explanation without providing uncertainty estimate when possible
- [ ] Using explanation without comparing to simpler baselines (e.g., center bias)
- [ ] Using explanation without testing on corrupted or edge-case images (e.g., all black)
- [ ] Using explanation without verifying it works with model ensembling or averaging
- [ ] Using explanation without accounting for quantization effects if deployed
- [ ] Using explanation without verifying it respects patient privacy (no PHI in outputs)
- [ ] Using explanation without checking for legal/compliance requirements (e.g., HIPAA de-identification)
- [ ] Using explanation without considering environmental impact of computation
- [ ] Using explanation without planning for drift detection (explanation changes over time)
- [ ] Using explanation without verifying it works with different DICOM transfer syntaxes
- [ ] Using explanation without testing on images with implants, pacemakers, or surgical artifacts
- [ ] Using explanation without validating it handles extreme contrast or exposure variations
- [ ] Using explanation without checking compatibility with model quantization (int8/fp16)
- [ ] Using explanation without documenting expected failure modes (e.g., low contrast images)
- [ ] Using explanation without validating it works with different segmentation algorithms
- [ ] Using explanation without testing on pediatric vs adult chest X-rays (different anatomy)
- [ ] Using explanation without considering explainability for multi-task models
- [ ] Using explanation without validating it works with model distillation or compression
- [ ] Using explanation without testing on images with rotation or flip (if model not invariant)
- [ ] Using explanation without checking it respects model's receptive field limits
- [ ] Using explanation without introducing adversarial robustness testing
- [ ] Using explanation without considering computational cost for real-time deployment
- [ ] Using explanation without sharing failure cases with the team for improvement
- [ ] Using explanation without defining success criteria beyond visual appeal
- [ ] Using explanation without regarding it as hypothesis requiring validation
- [ ] Using explanation without planning for versioning and deprecation of methods
- [ ] Using explanation without documenting alternatives considered and rationale for selection
- [ ] Using explanation without verifying it does not introduce bias across subgroups
- [ ] Using explanation without ensuring it works with missing data imputation strategies
- [ ] Using explanation without planning for regulatory submission documentation
- [ ] Using explanation without providing user training materials for correct interpretation
- [ ] Using explanation without defining retirement criteria when superseded by better methods
- [ ] Using explanation without verifying it complies with institutional review board standards
- [ ] Using explanation without testing on images with metal artifacts or foreign bodies
- [ ] Using explanation without validating it works with different windowing/leveling presets
- [ ] Using explanation without considering accessibility for low-vision users
- [ ] Using explanation without testing on ultra-high resolution images (e.g., 4K+)
- [ ] Using explanation without validating it respects model's calibration (explanation should reflect confidence)
- [ ] Using explanation without considering temporal stability for longitudinal studies
- [ ] Using explanation without planning for cloud-specific deployment constraints
- [ ] Using explanation without providing versioned model cards for different explanation methods
- [ ] Using explanation without testing on images with severe pathology obscuring lungs
- [ ] Using explanation without validating it respects model's uncertainty estimates (if available)
- [ ] Using explanation without testing on images with motion blur or patient movement
- [ ] Using explanation without considering forensics and audit trail requirements
- [ ] Using explanation without validating it respects model's invariance properties (e.g., rotation)
- [ ] Using explanation without testing on images with varying fat suppression or contrast agents
- [ ] Using explanation without validating it works with model ensembling or snapshot averaging
- [ ] Using explanation without testing on images with different patient positions (AP, PA, lateral)
- [ ] Using explanation without validating it works with model's test-time augmentation (TTA)
- [ ] Using explanation without validating it works with different annotation formats (COCO, VOC, etc.)
- [ ] Using explanation without providing rollback plan if new explanation method fails
- [ ] Using explanation without testing on images with different piezoelectric sensors or technologies
- [ ] Using explanation without validating it respects model's feature hierarchy (e.g., Early vs late layers)
- [ ] Using explanation without testing on images with different scanner manufacturers or models
- [ ] Using explanation without validating it respects model's equivariance properties (if applicable)
- [ ] Using explanation without considering for mobile or edge device deployment constraints
- [ ] Using explanation without validating it works with different loss functions during training
- [ ] Using explanation without validating it respects model's optimization landscape properties
- [ ] Using explanation without testing on images with different reconstruction algorithms or kernels
- [ ] Using explanation without verifying it respects model's sparsity or density assumptions
- [ ] Using explanation without testing on images with differentbit depths or compression levels
- [ ] Using explanation without validating it respects model's regularization effects
- [ ] Using explanation without testing on images with different pathology prevalence or severity
- [ ] Using explanation without validating it respects model's generalization to unseen data
- [ ] Using explanation without testing on images with different acquisition protocols or techniques
- [ ] Using explanation without validating it respects model's bias-variance tradeoff properties
- [ ] Using explanation without testing on images with different scanner geometries or configurations
- [ ] Using explanation without validating it respects model's robustness to distribution shift
- [ ] [END OF SKILL.md]