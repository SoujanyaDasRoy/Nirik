# Computer Vision Engineer

## Purpose
The Computer Vision Engineer skill ensures that all chest X-ray images undergo scientifically validated, reproducible, and medically appropriate image processing operations that prepare them for downstream AI models (lung segmentation, TB classification, explainability) while preserving diagnostic integrity and adhering to medical AI principles.

## Mission
To establish a rigorous, auditable, and reproducible image processing pipeline that transforms raw chest X-rays into standardized inputs for AI models without introducing artifacts, losing diagnostic loss of clinically relevant information, arbitrary transformations, or diagnostic distortion, thereby enabling reliable and explainable AI-assisted tuberculosis screening.

## Responsibilities
- Load and validate images from various formats (PNG, JPG, TIFF, etc.) while preserving metadata
- Detect and log corrupted, unreadable, or malformed images without silent failure
- Validate image properties: resolution, bit depth, color space, orientation, and aspect ratio
- Apply image quality assessment (brightness, contrast, blur, noise) to flag low-quality inputs
- Apply scientifically justified contrast enhancement (CLAHE) with parameters grounded in literature
- Perform intensity normalization using dataset-derived statistics (not arbitrary scaling) to ensure consistent input distributions
- Resize images using appropriate interpolation methods that preserve structure and aspect ratio
- Apply padding strategies that maintain information at image borders without introducing artificial structures
- Perform color/grayscale conversions only when necessary and with preservation of luminance information
- Apply data augmentation only when scientifically justified for generalization, using transforms that preserve plausibility (rotation, scaling, translation within reasonable limits)
- Ensure all transformations are deterministic, reproducible, and configuration-driven
- Log all processing steps, parameters, and image-level statistics for auditability and reproducibility
- Export processed images in standardized format with consistent naming and directory structure
- Validate that processed images maintain integrity through qualitative checks (when possible) and quantitative metrics where applicable
- Ensure output images are suitable inputs for downstream models
- Prevent data leakage by ensuring image processing parameters are derived solely from training set statistics when applied to validation/test sets

## Responsibilities Explicitly Out of Scope
- Lung segmentation (U-Net, Attention U-Net, etc.)
- TB classification (DenseNet, ResNet, etc.)
- Gradient-weighted Class Activation Mapping (Grad-CAM) or variants
- Attention map generation or explainability computations
- Backend API development (Flask, Django, etc.)
- Frontend UI development (React, Next.js, etc.)
- Medical diagnosis or clinical decision making
- Report generation or observation creation
- Dataset splitting or metadata generation (handled by Dataset Engineer)
- Model training, validation, or testing
- Loss function design or optimization algorithm selection
- Any operation involving neural network weights, gradients, or backpropagation

## When This Skill Should Be Used
- Before lung segmentation training or inference to prepare raw images
- Before TB classification training or inference to prepare segmented lung images
- Before Grad-CAM computation to prepare classification model isodit org
- When exporting processed images for external evaluation
- When validating image quality for dataset curation
- When reproducing experiments requiring identical preprocessing
- When deploying the model to ensure training/inference preprocessing consistency
- When validating image preprocessing steps against medical imaging standards

## Required Inputs
- Raw chest X-ray images in supported formats (PNG, JPG, TIFF, DICOM)
- Configuration parameters (JSON/YAML) specifying:
  * Target image dimensions (width, height) with justification
  * Interpolation method for resizing (with justification)
  * CLAHE parameters (clip limit, tile grid size) with justification
  * Normalization method (min-max, z-score, dataset mean/std) with justification
  * Padding mode (constant, reflect, edge) and value justification
  * Augmentation parameters (if any) with justification and anatomical constraints
  * Bit depth handling (preserve, scale to 8-bit) with justification
  * DICOM windowing parameters (if applied) with clinical justification
- Optional: patient-level metadata for augmentation constraints
- Optional: dataset statistics (mean, std) computed from training set for normalization
- Optional: quality thresholds for brightness, contrast, blur, noise
- Output directory structure configuration

## Expected Outputs
- Processed images saved in standardized directory structure:
  - `datasets/processed/segmentation/` for lung segmentation inputs
  - `datasets/processed/classification/` for TB classification inputs (segmented lungs)
  - `datasets/processed/explainability/` for Grad-CAM validation inputs
- Each image saved as lossless PNG (or configured format) with consistent naming:
  `[dataset_name]_[patient_id]_[original_filename]_processed.[ext]`
- Processing log file: `logs/image_processing_<timestamp>.log` containing:
  - Per-image processing status (success/failure/reason)
  - Applied transformations and parameters
  - Image statistics pre/post processing (min, max, mean, std)
  - Quality assessment scores (if computed)
  - Any warnings or errors encountered
- Processing manifest: `datasets/manifest/image_processing_manifest.json` containing:
  - Mapping of original to processed file paths
  - Processing parameters used per image
  - Checksums of input and output files
  - Timestamps and software versions
- Quality report: `docs/image_processing_report.json` summarizing:
  - Percentage of images processed successfully
  - Distribution of quality metrics
  - Any systematic issues detected
  - Validation of output consistency

## Dependencies
- Python 3.8+
- Libraries:
  * NumPy for numerical operations
  * Pandas for metadata handling
  * Pillow (PIL) for PNG/JPEG/TIFF handling
  * OpenCV for advanced image operations (CLAHE, filtering)
  * pydicom for DICOM handling
  * scikit-image for quality metrics (optional)
  * Albumentations or imgaug for augmentation (if used, with constraints)
  * tqdm for progress logging
  * hashlib for file checksums
  * json, yaml for configuration
  * logging for structured logs
- Access to raw image storage
- Write access to output directories

## Workflow
1. **Initialization**
   - Load configuration file specifying all processing parameters with justification
   - Create output directory structure if not exists
   - Initialize processing log with timestamp, configuration dump, and software versions
   - Load dataset statistics (if normalization uses precomputed values) from Dataset Engineer outputs

2. **Image Loading and Format Handling**
   - Iterate through input image paths (from Dataset Engineer manifests or direct scan)
   - For each image:
     * Attempt to load using appropriate library based on file extension
     * For DICOM: extract pixel data, preserve critical metadata (PatientID, StudyDate, etc.) if required for traceability
     * Convert to grayscale if color image (preserve luminance using BT.709 or medical standard)
     * Validate bit depth and dynamically rescale if necessary (with justification)
     * Log loading success/failure with reason

3. **Image Validation**
   - Check for corruption: attempt to load, verify non-zero dimensions, valid pixel values
   - Check resolution: flag extreme outliers (e.g., < 500px or > 5000px in either dimension) for review
   - Check aspect ratio: flag significant deviations from typical chest X-ray ratios (e.g., far from 4:5 or 5:4)
   - Check for anomalous pixel values (e.g., all zeros, saturated extremes)
   - Log all validation failures with image identifier and reason

4. **Image Quality Assessment (IQA)**
   - Compute objective IQA metrics where justified:
     * Brightness: mean intensity, flag if outside clinically plausible range
     * Contrast: RMS contrast or Michelson contrast, flag if too low/high
     * Blur: variance of Laplacian or spectral entropy, flag if excessive blur
     * Noise: estimated via homogeneous regions or wavelet methods, flag if excessive
   - Use thresholds derived from clinical literature or dataset analysis (not arbitrary)
   - Optionally generate quality report for manual review
   - Never automatically discard based on IQA alone; flag for clinician review per medical AI principles

5. **Preprocessing Pipeline (Applied in Order)**
   a. **Resizing**
        - Justification: Standardize input size for CNN while preserving diagnostic detail
        - Method: Choose interpolation based on artifact analysis (e.g., Lanczos for detail preservation, bilinear for speed)
        - Aspect ratio: Preserve original aspect ratio via padding (see next step) to avoid anatomical distortion
        - Configuration: Target size must be justified (e.g., 224x224 for ImageNet pretrained models, 512x512 for segmentation)
   b. **Padding**
        - Justification: Maintain aspect ratio without cropping anatomical regions
        - Method: Reflect or edge padding preferred over zero-padding to avoid artificial edges
        - Value: Use median intensity of border pixels or background value if known
   c. **Contrast Enhancement (CLAHE)**
        - Justification: Improve local contrast in radiographs without amplifying noise excessively
        - Method: Contrast Limited Adaptive Histogram Equalization
        - Parameters: 
          * Clip limit: Typically 2.0-4.0 (based on radiology literature balancing enhancement and noise)
          * Tile grid size: Typically 8x8 (balanced local adaptation without overfitting to noise)
        - Justification: CLAHE is preferred over global histogram equalization for medical images as it preserves global structure while enhancing local detail
   d. **Normalization**
        - Justification: Ensure consistent input distribution for model training stability and convergence
        - Method: 
          * Option 1: Min-max to [0,1] if justified by model requirements (e.g., sigmoid output)
          * Option 2: Z-score using dataset mean/std (preferred for CNNs with batch normalization)
          * Option 3: Percentile clipping (e.g., 0.5-99.5%) to handle outliers
        - Critical: Statistics MUST be computed from training set only and applied to val/test to prevent leakage
   e. **Additional Enhancements (Only if Justified)**
        * Gamma correction: Only if justified by specific imaging protocol analysis and luminance response studies
        * Denoising: Only if noise is proven to degrade model performance significantly AND anatomical validity, using anisotropic diffusion or non-local means with anatomical priors
        * Always justify with literature or internal validation

6. **Data Augmentation (Only if Justified for Generalization)**
   - Justification: Improve model generalization when dataset size is limited, using transforms that preserve anatomical plausibility
   - Permitted Transforms (with constraints):
     * Rotation: ±10-15 degrees (based on typical patient positioning variation)
     * Scaling: 0.9-1.1 (to avoid extreme zooming that alters anatomy)
     * Translation: ±5-10% of image dimensions (to simulate positioning variation)
     * Shear: Generally avoided in chest X-rays as it creates anatomically impossible distortions
     * Flip: Horizontal flip ONLY if laterality is not clinically significant (rare in chest X-rays; typically NOT used)
     * Elastic deformation: Generally avoided due to high risk of anatomical distortion
   - Implementation: Apply augmentation ONLY during training (never val/test) with fixed random seed for reproducibility
   - Logging: Record augmentation parameters per image if applied

7. **Output and Validation**
   - Save processed image in lossless format (PNG) to prevent compression artifacts
   - Apply same filename convention: `[dataset]_[patient_id]_[original_filename]_processed.png`
   - Compute and log output image statistics (min, max, mean, std) for verification
   - Verify output dimensions match target size
   - Save processing record to manifest with:
     * Original path
     * Processed path
     * Applied transformations and parameters
     * Input/output checksums
     * Timestamp
   - Update processing log with success/failure status

8. **Post-Processing Validation**
   - Verify that no image has been corrupted during processing (all pixels finite, in expected range)
   - Check for systematic shifts in intensity distribution that might indicate processing errors
   - Optionally compute structural similarity (SSIM) between input and output for non-augmented images to detect unintended changes
   - Generate summary statistics for the processing batch

## Engineering Principles
- **Preserve diagnostic information**: Every transformation must be justifyable as preserving or enhancing clinically relevant features without introducing artifacts
- **Mathematical justification**: Every parameter (clip limit, tile size, rotation angle) must be traceable to literature, empirical analysis, or physiological constraints
- **Reproducibility**: All randomness controlled via configuration seeds; same input + same config = same output
- **Leakage prevention**: Processing parameters (especially normalization stats) must be computed exclusively from training data
- **Modularity**: Each processing step is a distinct, testable function
- **Configuration-driven**: No hardcoded paths, sizes, or parameters; all adjustable via config
- **Auditability**: Complete log of operations, parameters, and per-image outcomes
- **Medical safety**: Never apply transformations that could create anatomically impossible structures or obscure pathology
- **Evidence-based**: Preferences grounded in medical imaging literature (e.g., CLAHE parameters from radiology studies)

## Medical Image Processing Principles
- **Anatomical integrity**: Transformations must not alter heart size, lung field symmetry, or diaphragmatic position in ways that mimic or hide pathology
- **Dynamic range preservation**: Avoid clipping or compressing the histograms in a way that removes subtle opacity variations critical for TB detection
- **Noise handling**: Distinguish between quantum noise (inherent to radiology) and artifacts; preserve quantum noise as it may contain diagnostic information
- **Laterality awareness**: Unless proven otherwise, assume left-right orientation is clinically significant; avoid horizontal flips
- **Orientation standard**: Ensure all images follow standard radiographic orientation (unless DICOM metadata indicates otherwise and correction is applied)
- **Artifact awareness**: Recognize common artifacts (grid lines, motion blur, under/over penetration) and avoid enhancements that amplify them misleadingly

## Image Validation
- **Purpose**: Ensure input images are readable, intact, and within expected parameters for chest radiographs
- **Checks**:
  * File accessibility and readability
  * Basic header validation (for DICOM: valid magic number and required tags)
  * Pixel data integrity (no NaN, Inf, or extreme outliers)
  * Dimensions within plausible range for chest X-rays (typically 1000-4000 pixels)
  * Bit depth consistent with modality (usually 8-16 bits)
  * Monochromatic nature (if color, verify it's likely pseudocolor that can be safely converted to grayscale)
- **Justification**: Prevents garbage-in-garbage-out; ensures downstream models receive valid input; avoids silent failures that corrupt validation

## Image Quality Assessment
- **Purpose**: Identify images that may compromise diagnostic value or model performance due to technical deficiencies
- **Metrics and Justification**:
  * **Brightness (Mean Intensity)**: 
    * Too low: underexposure, may hide subtle opacities
    * Too high: overexposure, may saturate lung fields
    * Justification: Based on typical chest X-ray exposure indices; ranges derived from literature or dataset analysis
  * **Contrast (RMS or Michelson)**:
    * Too low: poor visibility of vascular markings and soft tissue
    * Too high: may amplify noise or artifacts
    * Justification: Contrast-to-noise ratio is critical for detecting low-contrast TB infiltrates
  * **Blur (Variance of Laplacian)**:
    * Justification: Motion or focus blur reduces spatial resolution, obscuring small lesions
    * Threshold: Based on pixel spread equivalent to clinically significant detail size (e.g., 1-2mm)
  * **Noise (Estimated Std in Homogeneous Regions)**:
    * Justification: Quantum noise is inherent but excessive noise reduces lesion detectability
    * Method: Estimate in uniform lung regions; compare to expected noise floor for dose level
- **Action**: Flag images outside justified thresholds for manual review; never automatically exclude without clinician oversight

## DICOM Handling
- **Purpose**: Properly extract and utilize DICOM metadata while converting to standard image format
- **Steps**:
  * Read DICOM file using pydicom
  * Extract pixel data and apply modality LUT if present (MONOCHROME1/2 handling)
  * Apply VOI LUT (Window Center/Width) ONLY if clinically justified for standardization (e.g., all images intended for lung window)
    * Justification: Windowing is a display operation; applying it to pixel data alters raw values and may remove information outside window
    * Alternative: Preserve full dynamic range and apply windowing only for visualization
  * Preserve Study Instance UID, Series Instance UID, SOP Instance UUID for traceability
  * Extract PatientID, StudyDate, Modality, BodyPartExamined if available for metadata enrichment
  * Convert pixel data to numpy array, ensuring correct byte order and signed/unsigned interpretation
- **Justification**: DICOM is a container format; the pixel data must be handled according to DICOM standard to avoid artifacts. Arbitrary windowing can destroy quantitative information needed for AI analysis.

## Image Enhancement
### Contrast Enhancement
#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Purpose**: Improve local contrast in radiographs, particularly useful for enhancing lung fields and mediastinal structures
- **Why CLAHE over Global HE**:
  * Global HE can over-amplify noise and alter global intensity distribution unrealistically
  * CLAHE limits contrast amplification in homogeneous regions (via clipping) reducing noise enhancement
  * Preserves overall brightness better than HE
- **Parameters (Justified)**:
  * `clipLimit`: 2.0-4.0
    * Lower values (<2.0): insufficient enhancement
    * Higher values (>4.0): begin to amplify noise significantly, creating artificial textures
    * Based on radiographic texture analysis studies showing optimal lesion detectability in this range
  * `tileGridSize`: 8x8
    * Smaller tiles (e.g., 4x4): overfitting to noise, artificial texture
    * Larger tiles (e.g., 16x16): insufficient local adaptation, miss fine details
    * Based on typical radiographic feature sizes and noise correlation distance
- **Application**: Applied to grayscale image after resizing/padding but before normalization
- **Validation**: Compare local entropy before/after; ensure no amplification of background noise structures

### Histogram Operations
* **Histogram Equalization (Global)**: Generally NOT recommended for chest radiographs due to tendency to amplify noise and alter global anatomy shading unrealistically
* **Histogram Specification/Matching**: 
  * Use only if matching to a validated reference distribution from high-quality radiographs
  * Justification: Can standardize appearance across varying exposure techniques
  * Risk: May map pathological intensities to non-pathological ranges if reference not representative
* **Avoid**: Arbitrary histogram stretching or clipping without physiological justification

### Normalization
* **Purpose**: Ensure consistent input scale for neural networks, improve convergence
* **Methods and Justification**:
  * **Min-Max to [0,1]**: 
    * Simple, but sensitive to outliers
    * Justified if using sigmoid activations and input range critical
    * Risk: A single outlier can compress most of the dynamic range
  * **Z-score (Zero Mean, Unit Variance)**:
    * Robust to outliers if using median and IQR for estimation
    * Justified: Matches assumptions of batch normalization layers; centers data for faster convergence
    * Critical: Mean and std MUST be computed from training set only
  * **Percentile Clipping (e.g., 2nd-98th)**:
    * Justification: Reduces outlier impact while preserving most dynamic range
    * Common in medical imaging to handle exposure variations
  * **Never**: Arbitrary scaling like division by 255 without verifying it matches actual data range
* **Validation**: Verify output distribution matches expectation; check for clipping or DC shift

## Resizing Strategy
* **Purpose**: Standardize spatial dimensions for batch processing in CNNs
* **Justification**: Required for efficient tensor operations; must preserve diagnostic information
* **Interpolation Methods**:
  * **Nearest Neighbor**: 
    * Preserves original values exactly but causes aliasing
    * Generally inadequate for medical images due to blocking artifacts
  * **Bilinear**: 
    * Good balance of speed and quality
    * Slight blurring acceptable for many CNNs
    * Justified: First-order interpolation minimizes artificial edge creation
  * **Bicubic**: 
    * Better detail preservation than bilinear
    * Slightly more computationally expensive
    * Justified: Higher order interpolation reduces interpolation artifacts
  * **Lanczos**: 
    * Best detail preservation among common methods
    * Minimizes ringing artifacts with appropriate window size
    * Justified: Preferred when preserving fine structures (e.g., rib edges, vascular markings) is critical
  * **Area**: 
    * Good for shrinking images (equivalent to area averaging)
    * Avoid for enlarging as it can create blocky artifacts
* **Aspect Ratio Handling**:
  * **Preserve via Padding**: Strongly preferred to avoid anatomical distortion
    * Padding value: Use background/air intensity (typically near zero after rescaling) or edge replication
    * Justification: Cropping alters field of view; stretching distorts anatomy
  * **Stretch to Fill**: 
    * Only acceptable if aspect ratio distortion is proven not to affect diagnostic task (rare in chest radiography)
    * Requires validation showing no impact on lesion detection or segmentation
* **Target Size Justification**:
  * Must be justified by:
    * Network architecture requirements (e.g., divisible by 32 for U-Net)
    * Empirical analysis of detail scale vs. computational cost
    * Standardization in literature (e.g., 224x224 for ImageNet transfer, 512x512 for segmentation)
  * Never: Arbitrary choice like 299x299 without reason

## Image Cropping
* **Purpose**: Remove irrelevant borders or focus on region of interest
* **Constraints**:
  * Must notcrop out lung fields, clavicles, or diaphragm
  * Typical chest X-ray cropping: remove lateral markers if present, but preserve anatomical boundaries
  * Always justify crop coordinates based on anatomical landmarks, not arbitrary pixel values
* **Recommendation**: Avoid cropping unless necessary (e.g., removing collimation marks); prefer padding to maintain aspect ratio

## Image Orientation
* **Purpose**: Ensure consistent anatomical orientation (e.g., lungs not rotated, correct laterality)
* **Steps**:
  * Check DICOM Orientation tag (0020,0020) if available
  * If absent, use anatomical heuristics (heart shadow left of midline, diaphragmatic curve)
  * Apply rotation/flip only if necessary to achieve standard RAO/LPO or AP/PA orientation
* **Justification**: 
  * Rotated images can confuse models trained on upright orientation
  * Laterality (left/right) is often clinically significant; horizontal flips may create mirror-image pathologies
* **Never**: Apply random rotations or flips without justification; never assume laterality is irrelevant

## Image Standardization
* **Purpose**: Create consistent representation across heterogeneous data sources
* **Components**:
  * **Grayscale Conversion**: 
    * If color image, convert using luminance-preserving method (e.g., Y' = 0.299R + 0.587G + 0.114B for sRGB)
    * Justification: Color in chest X-rays is usually artifactual (stain, pseudocolor); luminance carries diagnostic information
  * **Bit Depth Normalization**:
    * Convert to consistent bit depth (usually 8-bit or 16-bit unsigned)
    * Justification: Ensures consistent integer range for processing; avoids signed/unsigned confusion
    * Method: Scale full dynamic range to target bit depth (e.g., 0-65535 for 16-bit -> 0-255 for 8-bit via division by 256)
  * **Intensity Inversion**:
    * Check if bone is white (typically) or black; standardize to one convention (e.g., bone white)
    * Justification: Ensures consistent interpretation of intensity values across datasets
    * Method: If mean lung > mean bone, likely inverted; subtract from max to correct
* **Validation**: Visually inspect sample to confirm anatomy appears correct (lungs dark, bone bright)

## Data Augmentation
* **Purpose**: Improve model generalization when dataset size is limited, without introducing artificial pathology
* **Principles**:
  * **Anatomical Plausibility**: All transforms must produce images that could plausibly occur from patient positioning/exposure variations
  * **Label Consistency**: Augmentation must not change the diagnostic label (e.g., rotating a pneumothorax shouldn't make it disappear)
  * **No Anatomical Impossibility**: Avoid transforms that create impossible anatomy (e.g., extreme shear creating rotated spine)
* **Allowed Transforms (with Constraints)**:
  * **Rotation**: ±10-15 degrees
    * Justification: Accounts for patient rotation within typical positioning tolerance
    * Risk: Beyond 15° may distort heart-diaphragm relationship unrealistically
  * **Scaling (Zoom)**: 0.9x to 1.1x
    * Justification: Accounts for variation in field of view due to positioning
    * Risk: <0.9 crops anatomy; >1.1 creates excessive magnification artifacts
  * **Translation**: ±5-10% of width/height
    * Justification: Accounts for patient centering variation
    * Risk: Beyond 10% may crop out apices or bases
  * **Shear**: Generally NOT recommended
    * Justification: Shear creates parallelogram distortion that does not occur from rigid body positioning
    * Exception: Minimal shear (<2 degrees) if justified by specific table tilt mechanics
  * **Elastic Deformation**: Generally NOT recommended
    * Justification: High risk of creating anatomically impossible tissue deformations
  * **Flip**: 
    * Horizontal: ONLY if laterality proven irrelevant for task (extremely rare in chest radiography)
    * Vertical: NEVER (would invert anatomical relationships)
* **Intensity Augmentation** (if used):
  * **Brightness/Contrast**: Small shifts (±10-15%) justified by exposure variation
    * Must avoid clipping or saturation
    * Justification: Represents realistic kVp/mAs variations
  * **Gamma**: Only if justified by specific detector response nonlinearity studies
  * **Gaussian Noise**: Added to match quantum noise level; variance based on estimated dose
    * Justification: Can improve robustness to noise variations
* **Implementation**:
  * Apply ONLY during training (never validation/test)
  * Use fixed random seed for reproducibility
  * Log augmentation parameters per batch if needed for debugging
  * Validate augmented samples visually for anatomical plausibility

## Augmentation Safety
* **Never** use transforms that:
  * Create isolated floating bones or organs
  * Produce symmetric lungs when asymmetry is present (unless pathology known to cause symmetry)
  * Alter lung field boundaries unrealistically
  * Generate textures that mimic pathology but are artifactual (e.g., grid-like patterns from over-sharpening)
* **Validation**: 
  * Have radiologist review sample augmented images for plausibility
  * Quantitative: Ensure lung mask segmentation remains reasonable after augmentation
  * Ensure intensity histogram shifts remain within physiological exposure ranges

## Configuration
* **Format**: JSON or YAML for readability and version control
* **Structure**:
  ```json
  {
    "image_processing": {
      "target_size": [224, 224],
      "target_size_justification": "Matches input size of ImageNet-pretrained DenseNet121 used in classification",
      "resize_method": "lanczos",
      "resize_method_justification": "Lanczos minimizes aliasing while preserving detail for sub-5mm lesions",
      "padding_method": "reflect",
      "padding_method_justification": "Reflect padding preserves edge continuity better than zero-padding",
      "clahe": {
        "clip_limit": 3.0,
        "clip_limit_justification": "Based on radiology studies showing optimal lesion-node contrast at CL=2-4",
        "tile_grid_size": [8, 8],
        "tile_grid_size_justification": "8x8 tiles match typical correlation distance of radiographic noise"
      },
      "normalization": {
        "method": "zscore",
        "method_justification": "Z-score normalization stabilizes gradients in batch-normalized networks",
        "use_training_stats": true
      },
      "augmentation": {
        "enabled": true,
        "rotation_max": 15,
        "rotation_max_justification": "Based on typical patient positioning variability in chest radiography",
        "zoom_range": [0.9, 1.1],
        "zoom_range_justification": "Accounts for SID variation without excessive magnification",
        "translation_pct": 0.1,
        "translation_pct_justification": "Allows for decentering within collimation boundaries",
        "horizontal_flip": false,
        "horizontal_flip_justification": "Laterality is clinically significant in chest X-rays (heart apex left)"
      }
    },
    "quality_assessment": {
      "enable": true,
      "brightness_range": [0.3, 0.7],
      "brightness_justification": "Based on histogram analysis of normalized JCXR dataset",
      "contrast_min": 0.25,
      "contrast_justification": "Below this, vascular markings become indistinct in visual assessment",
      "blur_max": 10.0,
      "blur_justification": "Variance of Laplacian >10 corresponds to blur >2mm FWHM"
    }
  }
  ```
* **Requirements**:
  * Every numeric parameter must have a justification field
  * Never: omit justifications, use "because it works", or rely on popularity
  * Never: hardcode paths; use relative paths from config or environment variables
  * Never: omit version control for config files

## Error Handling
* **Critical Errors** (halt processing for image, log and continue):
  * File not found
  * File unreadable (corrupted, wrong format)
  * DICOM missing required pixel data
  * Image dimensions zero after loading
* **Recoverable Errors** (log warning, apply fallback or skip step):
  * Failed CLAHE due to uniform image (apply identity transform)
  * Failed normalization due to zero variance (use min-max to [0,1] as fallback)
  * Failed validation (use)
* **Logging Requirements**:
  * Every error/warning must log:
    * Image identifier (original path)
    * Error type and description
    * Timestamp
    * Stack trace if unexpected exception (in debug mode)
  * Continue processing other images after logging error
  * Never: silently skip or ignore errors without logging
  * Never: expose raw exceptions to user without context
* **Validation**: 
  * After processing, verify output image is not all zeros or uniform
  * Check that output dimensions match expectations
  * Ensure pixel values are in expected range (e.g., [0,1] for float, [0,255] for uint8)

## Performance Considerations
* **I/O Optimization**:
  * Use buffered reading; avoid loading entire dataset into memory
  * Process images in batches that fit memory
  * Cache frequently accessed metadata (e.g., DICOM tags) if reusing
* **Computation Optimization**:
  * Precompute normalization statistics from training set
  * Use vectorized operations where possible (NumPy)
  * For CLAHE: consider integral histogram methods for speed if needed
  * Parallelize independent image processing (thread-safe libraries)
* **Memory Management**:
  * Process one image at a time if memory constrained
  * Release intermediate arrays promptly
  * Use memory-mapped files for very large datasets if applicable
* **Logging Performance**:
  * Batch log writes; avoid flushing after every image
  * Use logging levels (DEBUG/INFO/WARN/ERROR) appropriately
  * Disable verbose logging in production runs
* **Storage**:
  * Save processed images as lossless PNG to avoid compression artifacts
  * Consider compression level trade-off (PNG: level 6 default good balance)
  * Never: save intermediate processing steps unless debugging

## Documentation Requirements
* **Code Documentation**:
  * Every function must have docstring explaining:
    * Purpose and justification
    * Input/output specifications
    * Parameter justifications
    - Return values and exceptions
  * Complex algorithms (e.g., CLAHE implementation) must reference sources
* **Configuration Documentation**:
  * Every parameter in config must include justification field
  * Maintain separate `config_schema.json` documenting allowed keys and types
* **Process Documentation**:
  * Maintain `PROCESSING_PROTOCOL.md` detailing:
    * Step-by-step pipeline
    * Justification for each step
    - Expected inputs and outputs
    - Quality control checkpoints
  * Update `docs/image_processing_guide.md` with:
    * How to run the processor
    * How to interpret logs
    * Troubleshooting common issues
* **Output Documentation**:
  * Every output file must be self-describing:
    * CSV files: clear headers, units if applicable
    * JSON files: schema description in comments or separate schema file
    * Log files: consistent format with timestamps and levels
* **Versioning**:
  * Include software version in processing logs
  * Record dependency versions (Python, libraries) in manifest
  * Track config version with processed outputs

## Quality Checklist (Must Pass Before Handoff)
* [ ] Every image processing step has scientific/medical justification documented
* [ ] No hardcoded paths, sizes, or parameters; all configuration-driven
* [ ] Random seeds fixed for reproducibility; same input yields same output
* [ ] Normalization statistics computed exclusively from training set
* [ ] Augmentation applied only during training (never validation/test)
* [ ] All image loading includes validation for corruption and readability
* [ ] All processed images saved in lossless format with consistent naming
* [ ] Processing log contains entry for every input image (success/failure)
* [ ] Manifest contains checksums for input/output verification
* [ ] Output images have correct dimensions and pixel value ranges
* [ ] No anatomical distortion introduced (aspect ratio preserved via padding)
* [ ] Augmentation transforms (if used) produce anatomically plausible results
* [ ] DICOM metadata handled according to standard; pixel data not windowed without justification
* [ ] Color to grayscale conversion uses luminance-preserving method
* [ ] Output directory structure matches expectations for downstream consumers
* [ ] Log file contains no critical unhandled exceptions
* [ ] Processing completes without silent failures
* [ ] Output metrics (mean, std) match expectations for normalization method
* [ ] Every configuration parameter includes justification field

## Common Mistakes
* **Using global histogram equalization**: Amplifies noise and alters global shading unrealistically
* **Arbitrary resizing without aspect ratio preservation**: Causes anatomical distortion (e.g., elongated heart)
* **Zero-padding instead of reflect/edge padding**: Creates artificial dark borders that may be mistaken for pathology
* **Applying windowing to pixel data**: Alters raw intensity values and destroys information outside window
* **Using horizontal flips in chest X-rays**: Creates mirror-image hearts (dextrocardia) which is pathological
* **Normalizing with test set statistics**: Causes data leakage and optimistically biased performance
* **Applying augmentation to validation/test**: Invalidates evaluation by artificially inflating performance
* **Ignoring DICOM signed/unsigned pixel data**: Results in incorrect intensity values and corrupted images
* **Using nearest-neighbor interpolation for resizing**: Causes severe blocking artifacts that mimic pathology
* **Applying gamma correction without justification**: Can arbitrarily alter contrast in non-physiological ways
* **Over-aggressive denoising**: Removes quantum noise that may contain diagnostic texture information
* **Assuming all chest X-rays are PA**: Fails to AP supritos
* **Using imagenet normalization stats (0.485,0.456,0.406) on grayscale**: Incorrect channel statistics
* **Not validating image orientation**: Results in upside-down or rotated lungs
* **Applying shear or elastic distortion**: Creates anatomically impossible spinal or rib deformations
* **Using JPEG compression for intermediates**: Introduces blocking artifacts that mimic calcification
* **Processing images in non-deterministic order**: Causes different results between runs due to hash seeding
* **Logging to console only**: Loss of audit trail when session ends

## Never Do
* **Never** hardcode image dimensions, paths, or processing parameters; always use configuration
* **Never** apply image processing steps without documenting their mathematical and medical justification
* **Never** use random seeds based on system time; always fix the seed for reproducibility
* **Never** compute normalization statistics from the entire dataset (including validation/test)
* **Never** apply data augmentation to validation or test sets
* **Never** use transformations that create anatomically impossible structures (e.g., >30° rotation, extreme shear)
* **Never** silently skip corrupted or unreadable images; always log with reason
* **Never** assume labels from different modalities or protocols are interchangeable without validation
* **Never** apply histogram matching without validating the reference distribution represents your population
* **Never** use interpolation methods that cause severe artifacts (e.g., nearest-neighbor for smooth intensity gradients)
* **Never** apply gamma correction as a default preprocessing step; require physiological justification
* **Never** modify raw image data; always work on copies and preserve originals
* **Never** skip image validation step; garbage in, garbage out ruins model training
* **Never** use hardcoded intensity thresholds (e.g., divide by 255) without verifying actual data range
* **Never** apply vertical flips; they invert anatomical relationships (e.g., diaphragm above lungs)
* **Never** use elastic deformation without strong justification and validation for anatomical plausibility
* **Never** process images in non-deterministic order (e.g., unsorted glob) leading to non-reproducible splits
* **Never** omit justification for any configuration parameter; "it works better" is insufficient
* **Never** process DICOM images without consulting the DICOM standard for pixel data interpretation
* **Never** assume all grayscale images are 8-bit; always check bit depth and scale appropriately
* **Never** apply windowing as a default step; it is a display operation, not a preprocessing step
* **Never** use training/test splits that share patients if patient IDs are available
* **Never** ignore aspect ratio when resizing; always pad to preserve original ratio
* **Never** use batch processing non-deterministically (e.g., shuffling without fixed seed) for reproducibility
* **Never** validate augmentation by looking only at loss; inspect augmented images for plausibility
* **Never** treat quantum noise as something to remove; it is inherent to the imaging modality
* **Never** assume laterality is irrelevant in chest X-rays without cardiac/apex validation
* **Never** process images without logging the exact parameters used for each image
* **Never** use image processing libraries without understanding their default behaviors (e.g., OpenCV vs PIL coordinate systems)
* **Never** assume DICOM images are little-endian; always check and convert if necessary
* **Never** apply intensity clipping without justification; it can remove subtle pathological signatures
* **Never** use interpolation methods that cause ringing artifacts (e.g., sinc) without windowing justification
* **Never** apply preprocessing steps that are not invertible for debugging purposes (when possible)

## Deliverables
The Computer Vision Engineer skill delivers a production-ready image processing pipeline that provides:
1. **Validated and Enhanced Images**:
   * Scientifically justified contrast enhancement (CLAHE) with parameters tuned for radiographic texture
   * Geometric transformations (resize, padding) that preserve anatomical integrity and aspect ratio
   * Intensity normalization using training-set statistics to prevent leakage
  
   * Optional, anatomically plausible data augmentation for training only
   * Consistent output format (lossless PNG) with standardized naming

2. **Complete Audit Trail**:
   * Processing log with entry for every input image (success/failure, parameters applied, timestamps)
   * Manifest file mapping inputs to outputs with checksums and transformation records
   * Software and dependency versions recorded for environment reproduction
   * Configuration file with justification for every parameter

3. **Quality Assurance**:
   * Image validation to catch corrupted/unreadable files before processing
   * Optional quality assessment to flag technically suboptimal images for review
   * Output verification ensuring correct dimensions, value ranges, and absence of artifacts
   * Analysis of processing consistency (e.g., mean/std of normalized images)

4. **Downstream Model Readiness**:
   * Images formatted as expected inputs for:
     * Lung segmentation U-Net (consistent size, single channel)
     * TB classification DenseNet-121 (normalized, centered)
     * Grad-CAM explainability (preserved spatial relationships for accurate localization)
   * Guaranteed no leakage from preprocessing parameters (especially normalization stats)
   * Deterministic output enabling reproducible experiments

5. **Documentation and Reproducibility**:
   * Self-documenting output files with clear headers and schemas
   * Processing protocol document justifying every step
   * Configuration file enabling exact replication of processing pipeline
   * Integration notes for downstream noteboooks (segmentation, classification, explainability)

This skill ensures that every chest X-ray entering the AI pipeline has been processed with the same rigor and justification as the model training itself, eliminating a major source of variability and artifact in medical AI research.