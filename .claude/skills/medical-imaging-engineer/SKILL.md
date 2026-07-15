# Medical Imaging Engineer

## Purpose
The Medical Imaging Engineer skill ensures that every chest X-ray entering the AI pipeline adheres to medical imaging standards, preserves diagnostic integrity, and is correctly interpreted according to radiology best practices. This role provides domain expertise in DICOM standards, chest radiography fundamentals, image acquisition parameters, and clinical safety to prevent misinterpretation or degradation of medically relevant information before downstream AI processing.

## Mission
To establish a rigorous, auditable, and clinically sound medical image validation and preparation framework that guarantees all images used for training, validation, and testing of AI models are anatomically accurate, properly oriented, correctly calibrated, and free from artifacts that could introduce bias or reduce model generalizability, thereby supporting reliable and explainable AI-assisted tuberculosis screening.

## Responsibilities
- Validate DICOM file integrity and extract critical metadata (PatientID, StudyDate, Modality, BodyPartExamined, etc.)
- Verify correct interpretation of pixel data including PhotometricInterpretation (MONOCHROME1/MONOCHROME2), PixelRepresentation (signed/unsigned), and BitsStored/BitsAllocated
- Apply DICOM-standard grayscale transformation pipeline: Modality LUT → VOI LUT (if applicable and justified) → Presentation LUT (conceptual for display)
- Determine and correct image orientation based on Patient Orientation (0020,0020) and Image Laterality (0020,0062) or anatomical heuristics
- Validate and utilize Pixel Spacing (0028,0030) for physical scale awareness; flag missing or anomalous spacing
- Distinguish between projection types (PA, AP, Lateral) using View Position (0018,5101) and Patient Position (0018,5100); advise on implications for AI models
- Detect and flag burned-in annotations (text, arrows, overlay icons) that may obscure pathology or confound AI analysis
- Identify and document laterality markers (L/R markers) and assess their presence/correctness
- Assess image completeness: verify field of view includes full lung fields, clavicles, and diaphragms; flag excessive collimation or cropping
- Evaluate image quality artifacts: motion blur, grid lines, under/over penetration, quantum noise, and detector artifacts
- Confirm correct laterality (left/right lung placement) using cardiac silhouette or aortic knob as reference
- Ensure no clinically relevant anatomy is cropped, obscured, or altered by preprocessing steps
- Validate that pixel spacing is consistent across datasets when physical measurements are required (e.g., lesion size estimation)
- Provide guidance on handling irreducible artifacts (e.g., implanted devices, surgical clips) that may mimic pathology
- Ensure metadata (especially patient identifiers) is preserved or de-identified according to privacy requirements without losing traceability
- Advise on appropriate windowing/level settings for visualization without altering raw pixel data used by AI models
- Collaborate with Computer Vision Engineer to ensure image processing steps (CLAHE, normalization, resizing) are medically justified and do not introduce artifacts
- Log all metadata extraction, validation results, and any corrections applied for auditability and reproducibility
- Never alter pixel data in a way that changes anatomical relationships or introduces synthetic structures
- Never assume uniform acquisition protocols across datasets; always validate and document variations
- Never ignore or discard DICOM metadata without justification; preserve traceability to original acquisition

## Responsibilities Explicitly Out of Scope
- Training or fine-tuning of neural networks (U-Net, DenseNet, etc.)
- Image segmentation or classification algorithms
- Gradient-weighted Class Activation Mapping (Grad-CAM) or explainability computations
- Backend API development (Flask, Django, etc.)
- Frontend UI development (React, Next.js, etc.)
- Medical diagnosis, report generation, or clinical decision making
- Dataset splitting or metadata generation (handled by Dataset Engineer)
- Core image processing operations (resize, normalization, CLAHE) – guided but not executed by this role
- Any operation involving neural network weights, gradients, or backpropagation

## Primary Notebook
- Notebook 1 (Dataset Preparation) - ensures medical image validity for dataset creation

## Secondary Notebooks
- Notebook 2 (Segmentation) - validated images used for segmentation training
- Notebook 3 (Classification) - validated images used for classification training
- Notebook 4 (Explainability & Evaluation) - validated images used for explainability and evaluation

## When This Skill Should Be Used
- Before ingesting any DICOM or image dataset into the pipeline to validate integrity and extract metadata
- When setting up the Dataset Engineer to ensure raw data passes medical imaging validation
- When configuring the Computer Vision Engineer to ensure processing steps (e.g., orientation correction, spacing awareness) are medically sound
- When reviewing processed images for artifacts that could affect segmentation or classification (e.g., burned-in text over lung fields)
- When preparing external evaluation datasets (e.g., Montgomery, Jaypee hold-out) to ensure they meet chest radiography standards
- When investigating model failures that may stem from image quality, orientation, or artifact issues
- When documenting acquisition protocol variations across datasets for reproducibility
- When ensuring de-identification preserves research utility while protecting patient privacy
- When validating that laterality markers are correctly interpreted to avoid left-right confusion in models
- When assessing whether pixel spacing is sufficient for intended analysis (e.g., measurement of opacities)
- When preparing documentation for regulatory or ethical review concerning medical image handling

## Required Inputs
- Raw image files in DICOM, PNG, JPG, TIFF format located in `datasets/raw/` or specified paths
- Configuration file specifying:
  * Expected modalities (CR, DX, etc.)
  * Required DICOM tags for validation (PatientID, StudyDate, Modality, etc.)
  * Allowed projection types (PA, AP, Lateral) and handling instructions for each
  * Pixel spacing validation thresholds (e.g., min/max plausible values for chest radiography)
  * Orientation handling policy (use DICOM tags, fallback to anatomical heuristics)
  * Burned-in annotation tolerance (zero tolerance for overlays on lung fields; log elsewhere)
  * Laterality marker requirements (must be present, must be correct)
  * Image completeness criteria (percent of lung field that must be visible)
  * Artifact detection thresholds (motion blur, penetration)
  * De-identification rules (which tags to remove/hash vs. keep for research)
  * Software/tool versions for reproducibility (pydicom version, etc.)
- Optional: radiology protocol documents or acquisition protocol sheets for each dataset source
- Optional: expert radiologist feedback on sample images for validation

## Expected Outputs
- Enhanced Dataset Engineer outputs with added medical imaging validation columns:
  * `master_metadata.csv` includes: `patient_id, study_date, modality, body_part_examined, view_position, patient_position, pixel_spacing_x, pixel_spacing_y, rows, columns, bits_allocated, bits_stored, pixel_representation, photometric_interpretation, laterality_marker_present, laterality_marker_correct, burned_in_annotation_present, burned_in_annotation_location, image_complete, motion_blur_suspected, unterexposed, overexposed, grid_artifact_present, implant_or_device_present, laterality_correct_anatomically, orientation_corrected_applied, metadata_source (DICOM/header/heuristic)`
  * `dataset_statistics.csv` augmented with: distribution of modalities, view positions, pixel spacing, laterality correctness rates, artifact prevalence
  * `segmentation_metadata.csv` and `heatmap_metadata.csv` include same medical validation fields
- Medical imaging validation log: `logs/medical_imaging_validation_<timestamp>.log` containing:
  * Per-file DICOM parsing success/failure
  * Metadata extraction results
  * Orientation correction applied (if any)
  * Laterality determination method and result
  * Burned-in annotation detection outcome
  * Image completeness assessment
  * Artifact flags with descriptions
  * Any deviations from expected protocol
- Medical imaging manifest: `datasets/manifest/medical_imaging_manifest.json` mapping each image to:
  * Original DICOM tags extracted
  * Derived orientation and laterality
  * Validation flags (pass/fail per criterion)
  * Recommended handling (use as-is, orient correction, flag for review)
  * Checksum of original file for integrity
- Summary report: `docs/medical_imaging_report.json` with:
  * Percentage of images passing each validation criterion
  * Cohort demographics (age, sex if available via safe hashing)
  - Protocol distribution (PA/AP/Lateral counts per dataset)
  - Laterality correctness rate
  - Artifact frequency and types
  - Recommendations for dataset usage (e.g., "suitable for training after orientation correction", "exclude due to burned-in text over lungs")

## Dependencies
- Python 3.8+
- Libraries:
  * pydicom >= 2.3.0 for DICOM parsing
  * Pillow (PIL) >= 9.0.0 for non-DICOM image handling
  * NumPy >= 1.20.0 for numerical operations
  * Pandas >= 1.3.0 for metadata handling
  * scikit-image >= 0.19.0 for optional artifact detection (e.g., blur via Laplacian variance)
  * OpenCV >= 4.5.0 for advanced image analysis (optional)
  * tqdm >= 4.60.0 for progress logging
  * hashlib, json, csv, os, sys (standard library)
- Access to DICOM standard documentation (via pydicom or local reference)
- Optional: radiology textbook or peer-reviewed references for chest radiography standards

## Workflow
1. **Initialization**
   - Load configuration specifying validation criteria and handling policies
   - Initialize log file with timestamp, configuration dump, and software versions
   - Prepare output directories for logs, manifests, reports

2. **File Discovery and Format Identification**
   - Recursively scan input directories for image files
   - Determine file type by extension and, if ambiguous, by magic bytes (DICOM preamble)
   - Route each file to appropriate handler (DICOM vs. raster image)

3. **DICOM-Specific Processing**
   For each DICOM file:
   a. **Read and Validate Dataset**
      - Attempt to read with `pydicom.dcmread()`; catch and log any `InvalidDicomError`
      - Verify presence of mandatory SOP Common tags (SOPClassUID, SOPInstanceUID, StudyDate, etc.)
      - Log and flag if critical tags missing (PatientID, StudyDate, Modality, etc.)
   b. **Extract Pixel Data Properties**
      - Rows (0028,0010), Columns (0028,0011)
      - BitsAllocated (0028,0100), BitsStored (0028,0101), HighBit (0028,0102)
      - PixelRepresentation (0028,0103): 0 = unsigned, 1 = signed 2's complement
      - PhotometricInterpretation (0028,0004): MONOCHROME1 (higher value = brighter) or MONOCHROME2 (higher value = darker)
      - PlanarConfiguration (0028,0006) if applicable (should be 0 or absent for grayscale)
   c. **Modality LUT Application (if present)**
      - Check for Modality LUT Sequence (0028,3000) or LUT Data (0028,3006)
      - If present, apply to convert modality-specific values to linear attenuation values (per DICOM standard)
      - Log application; note that most chest X-rays do not contain modality LUT
   d. **VOI LUT Application (Conditional and Justified)**
      - Examine VOI LUT Sequence (0028,3010) or WindowCenter (0028,1050) / WindowWidth (0028,1051)
      - **Do NOT apply VOI LUT to pixel data** if the goal is to preserve original linear attenuation values for AI analysis
      - Record Window Center/Width for potential use in visualization only
      - Exception: If all images in a dataset are intended for a specific window (e.g., lung window) and raw values are consistently offset, application may be justified with documentation; otherwise preserve raw
   e. **Pixel Data Extraction and Conversion**
      - Extract pixel array from PixelData (7FE0,0010) or related fragments
      - Apply correct byte order based on (0028,0101) and system endianness
      - If PixelRepresentation = 1 (signed), interpret as signed integer; else unsigned
      - Convert to numpy array with appropriate dtype (uint8, uint16, int16)
   f. **Grayscale Standardization Check**
      - Determine if image needs intensity inversion to achieve standard where higher pixel value = more radiodense (bone white, lung black)
        - Use PhotometricInterpretation: MONOCHROME1 means higher value = more radiodense (already standard); MONOCHROME2 means higher value = less radiodense (needs inversion)
      - If MONOCHROME2, compute max pixel value based on bit depth and subtract from max to invert
      - Log inversion applied if performed
   g. **Pixel Spacing Extraction and Validation**
      - Pixel Spacing (0028,0030): row spacing, column spacing in mm
      - If missing, attempt to derive from ImagerPixelSpacing (0018,1164) or nominal values; log assumption
      - Validate against plausible range for chest radiography (typically 0.1–0.5 mm/pixel); flag outliers
   h. **Orientation and Laterality Determination**
      - Patient Orientation (0020,0020): row direction (first two values), column direction (second two); values like 'A\P' (anterior-posterior), 'R\L' (right-left)
      - Image Laterality (0020,0062): 'L', 'R', 'B' (both), or empty
      - If missing or ambiguous, use anatomical heuristics:
        * Locate cardiac silhouette (typically left of midline in PA view)
        * Check aortic knob prominence
        * Verify diaphragm contour
        * Presence of laterality marker (if present, trust unless contradicted by anatomy)
      - Determine if rotation/flip correction needed to achieve standard orientation (e.g., upright, patient not rotated)
      - Log any correction applied (e.g., "rotated 90° CCW based on Patient Orientation")
   i. **View Position and Patient Position**
      - View Position (0018,5101): 'PA', 'AP', 'LL', 'L' (lateral left), etc.
      - Patient Position (0018,5100): 'HFP', 'HFP', 'HFDL', etc.
      - Log values; note that AP vs PA affects cardiac magnification and scapular position
   j. **Burned-in Annotation Detection**
      - Convert image to grayscale uint8 if needed
      - Use template matching or connected component analysis to detect high-contrast stray pixels resembling text or lines
      - Focus detection on non-lung regions first; flag any annotations overlapping lung fields (determined via rough lung threshold or bounded box)
      - Record location and approximate content (if readable via OCR, optional)
   k. **Laterality Marker Presence and Correctness**
      - Detect 'L' and 'R' markers typically in corner of image
      - Verify that marker matches laterality determined from anatomy or Image Laterality tag
      - Log correctness and position
   l. **Image Completeness Assessment**
      - Define region of interest (e.g., central 80% of image) that should contain lung fields
      - Check for excessive black borders (collimation) that cut off anatomy:
        * Compute projection histograms; identify tight cropping
      - Flag if lung apices or costophrenic angles are likely cut off
   m. **Artifact Detection**
      - Motion Blur: estimate via variance of Laplacian in uniform regions; compare to threshold
      - Grid Lines: detect periodic high-frequency bursts in Fourier transform or via template
      - Under/Overexposure: analyze histogram; clip at extremes; note if >X% pixels at min/max
      - Metallic Implants/Devices: detect high-intensity compact structures; log presence and location
   n. **Metadata Preservation and De-identification Planning**
      - Identify which tags contain PHI (PatientID, PatientName, etc.)
      - Per configuration, either:
        * Retain for research traceability (if IRB allows) with plan to hash later
        * Replace with pseudonyms
        * Remove entirely
      - Log decision and action taken per tag category
   o. **Output Preparation**
      - Save processed pixel array (after any orientation correction and grayscale standardization) as lossless PNG or retain as numpy array for next step
      - Filename convention: `[dataset]_[patientID]_[studyID]_[SOPInstanceUID]_validated.[ext]`
      - Store all extracted metadata and validation flags in a temporary structure for merging into Dataset Engineer outputs

4. **Non-DICOM Image Handling (PNG, JPG, TIFF)**
   - Attempt to read with Pillow; log failure if unreadable
   - Extract basic metadata: size, mode (L, RGB, etc.), bits per pixel
   - Attempt to infer laterality/orientation from filename heuristics if configured (e.g., contains '_L_', '_R_')
   - Since DICOM tags missing, mark fields as `unknown` or `heuristic_only`
   - Perform same burned-in annotation, laterality marker, completeness, artifact checks as possible
   - Note limitations: no pixel spacing, no formal view position; rely on configuration defaults or filename patterns

5. **Aggregation and Reporting**
   - Aggregate per-image validation results into DataFrames
   - Merge with Dataset Engineer metadata outputs (if run sequentially) or produce standalone CSV files
   - Generate summary statistics:
     * Count per modality, view position, laterality correctness
     * Prevalence of each artifact type
     * Distribution of pixel spacing
     * Rate of burned-in annotations over lung fields
   - Write CSV outputs: `master_metadata.csv`, `dataset_statistics.csv`, `specialized_metadata.csv` files
   - Write JSON manifest with detailed per-image records
   - Write log file with verbose per-step records
   - Write summary report JSON for stakeholder review

6. **Feedback and Iteration**
   - Provide validation report to Dataset Engineer and Computer Vision Engineer for pipeline adjustment
   - If systematic issues found (e.g., 30% of images have burned-in text over lungs), recommend exclusion or manual cleanup
   - Update configuration based on findings (e.g., adjust pixel spacing tolerance, update laterality heuristic)

## Engineering Principles
- **Preserve Diagnostic Fidelity**: Every validation step must ask: does this action preserve or enhance the ability to detect pathology? Never alter image in a way that could hide or mimic disease.
- **Evidence-Based Validation**: All thresholds (pixel spacing, artifact detection, orientation) must be justified by radiology literature, DICOM standards, or empirical analysis of the dataset.
- **Metadata Integrity**: Preserve as much original DICOM metadata as possible for traceability; only anonymize when required and with documented strategy.
- **Explicit Assumptions**: When metadata is missing (e.g., pixel spacing), state assumptions clearly and log them; avoid silent defaults.
- **Reproducibility**: Same input DICOM + same configuration = same validation output and derived corrections.
- **Modularity**: Separate concerns: DICOM parsing, metadata extraction, validation rules, correction application, logging.
- **Configuration-Driven**: No hardcoded thresholds, paths, or policies; all adjustable via configuration file.
- **Auditability**: Comprehensive logs enable tracing why an image was flagged or corrected.
- **Clinical Safety First**: When in doubt, retain original image and flag for expert review rather than applying automatic correction that could be wrong.
- **Resource Awareness**: Balance thoroughness with performance; lightweight validation first, deeper analysis only if needed.
- **Error Logging**: Never swallow exceptions; log with context (filename, tag being processed) and continue with other images.
- **Separation of Concerns**: Validation and correction logging are distinct; never apply correction without logging what was done and why.

## Medical Imaging Principles
- **Radiograph Physics**: Chest X-rays represent differential absorption of X-rays; pixel values relate linearly to log attenuation (after detector correction). Arbitrary nonlinear transformations (e.g., gamma) alter this relationship and may affect quantitative analysis.
- **Grayness Standardization**: In medical displays, higher pixel values typically represent greater radiodensity (bone white, air black). The DICOM Grayscale Standard Display Function (GSDF) ensures perceptual consistency; however, for machine learning, preserving linear proportionality to attenuation is often preferred unless justified.
- **Laterality Significance**: Left/right asymmetry is clinically significant (e.g., aortic arch left, liver right). Incorrect laterality flips can create mirror-image pathologies (e.g., dextrocardia apparent normal).
- **Projection Differences**: PA (posteroanterior) reduces heart magnification compared to AP (anteroposterior); scapulae position differs. Models trained on one may not generalize to the other without explicit handling.
- **Pixel Spacing Meaning**: Enables conversion of pixel measurements to millimeters; critical for any size-based feature (e.g., cardiothoracic ratio, lesion diameter).
- **Image Completeness**: Missing apices can hide apical tuberculosis; missing bases can miss pleural effusion. Field of view must include relevant anatomy.
- **Artifacts vs. Pathology**: Grid lines, motion blur, and metallic implants can mimic or obscure nodules, infiltrates, or cavities. Must be identified and either corrected or accounted for.
- **Burned-in Annotations**: Text, arrows, or overlay icons permanently baked into pixels obscure underlying anatomy and can be mistaken for pathology; must be flagged or avoided.
- **Quality Factors**: Exposure (mAs, kVp), patient motion, and detector quality affect noise and contrast; these are inherent to the acquisition and should not be "corrected" unless justified and documented.
- **DICOM Conformance**: Adherence to DICOM standard ensures interoperability and correct interpretation; private tags should be noted but not relied upon for core functionality.

## Chest Radiography Fundamentals
- **Standard Projections**: Posteroanterior (PA) and anteroposterior (AP) are most common for chest; lateral (left lateral or right lateral) used additionally.
- **Patient Position**: Typically standing; may be supine in ICU. Affects diaphragm position and lung volumes.
- **Technical Factors**: 
  * kVp (kilovolt peak): influences penetration; typical chest 110-125 kVp
  * mAs (milliampere-seconds): influences exposure/motion noise; typical 2-5 mAs
  * Source-to-Image Distance (SID): usually 180 cm (72 inches)
- **Anatomical Landmarks**:
  * Clavicles: symmetry suggests no rotation
  * Scapulae: should be outside lung fields in proper PA
  * Diaphragm: right hemidiaphragm typically higher due to liver
  * Cardiac silhouette: left lung field, apex left
  * Aortic knob: left upper mediastinum
- **Normal Variants**: Rib notching, fetal nipple shadows, etc.; awareness prevents false positives.
- **Common Pathologies for TB**: Upper lobe infiltrates, cavitation, lymphadenopathy, miliary pattern.
- **Typical Artifacts**:
  * Grid Lines: from anti-scatter grid; appear as linear densitometric patterns
  * Motion Blur: patient motion during exposure
  * Underexposure: quantum mottle appears as noisy appearance
  * Overexposure: saturation leads to loss of detail in dense regions
  * Ghosting: from patient movement or detector artifacts

## DICOM Standards
- **File Structure**: Preamble (132 bytes) + DICOM prefix ("DICM") + Data Set (tags)
- **Tag Structure**: (group, element) each 32-bit; VR (Value Representation) defines data type
- **Endianness**: Explicit VR Little Endian default; must check (0002,0010) Transfer Syntax UID
- **Pixel Data**: 
  * May be encapsulated or native
  * May be compressed (jpeg, jpeg2000, rle) – must decompress before pixel access
  * PlanarConfiguration 0 = chunky (RGBRGB...), 1 = planar (RRR...GGG...BBB...) – rarely used for grayscale
- **Modality LUT**: Converts modality-specific values to Hounsfield units (CT) or optical density (radiography); often identity for DX.
- **VOI LUT**: Window Center/Width for display; applying to pixel data alters raw values.
- **Presentation LUT**: Conceptual; not stored in dataset.
- **Pixel Spacing**: Physical distance between centers of adjacent pixels; critical for calibration.
- **Patient Orientation**: Direction of patient rows and columns relative to anatomy (anterior, posterior, left, right, superior, inferior).
- **Image Laterality**: Explicit marker of which side of body is imaged (L, R, B).
- **Burned-in Annotation**: Text or graphics overlay burned into pixel data; identified by high-frequency components or OCR.
- **Private Tags**: (gggg,eeee) where gggg odd; may contain institution-specific info; handle per policy.

## Medical Image Metadata
- **Patient Identification**: 
  * Patient Name (0010,0010)
  * Patient ID (0010,0020)
  * Patient Birth Date (0010,0030)
  * Patient Sex (0010,0040)
- **Study Information**:
  * Study Date (0008,0020)
  * Study Time (0008,0030)
  * Study Instance UID (0020,000D)
  * Study Description (0008,1030)
- **Series Information**:
  * Series Date (0008,0021)
  * Series Time (0008,0031)
  * Series Instance UID (0020,000E)
  * Modality (0008,0060) – should be 'DX', 'CR', etc.
  * Series Description (0008,103E)
- **Image Information**:
  * Instance Number (0020,0013)
  * SOP Instance UID (0008,0018)
  * SOP Class UID (0008,0016)
  * Rows (0028,0010)
  * Columns (0028,0011)
  * Pixel Spacing (0028,0030)
  * Image Orientation (Patient) (0020,0037) – direction cosines for rows and columns
  * Image Position (Patient) (0020,0032) – not always present in 2D
  * Instance Creation Date (0008,0012)
  * Instance Creation Time (0008,0013)
- **Acquisition Context**:
  * KVP (0018,0060)
  * Exposure Time (0018,1150)
  * X-ray Tube Current (0018,1151)
  * Exposure (0018,1152)
- **Device Information**:
  * Manufacturer (0008,0070)
  * Institution Name (0008,0080)
  * Station Name (0008,1010)
- **Positioning and Laterality**:
  * Patient Position (0018,5100) – e.g., HFP (head first prone)
  * View Position (0018,5101) – PA, AP, LL, etc.
  * Image Laterality (0020,0062) – L, R, B
- **Pixel Characteristics**:
  * SamplesPerPixel (0028,0002) – 1 for grayscale
  * PhotometricInterpretation (0028,0004) – MONOCHROME1/2
  * PlanarConfiguration (0028,0006)
  * Rows, Columns, BitsAllocated, BitsStored, HighBit, PixelRepresentation
- **Window Center/Width** (0028,1050, 0028,1051): for display; not to be applied to pixel data for analysis unless justified.
- **Rescale Intercept/Slope** (0028,1052, 0028,1053): convert modality units to Hounsfield (CT) or optical density (DX); often 0/1 for DX.

## Image Orientation
- **Patient Orientation (0020,0020)**: String of 2 or 4 characters: 
  * First two: row direction (left-to-right in image)
  * Last two: column direction (top-to-bottom in image)
  * Each pair: anatomical direction (A=anterior, P=posterior, R=right, L=left, H=head (superior), F=foot (inferior))
  * Example: 'ANPO' = rows anterior-to-posterior, columns superior-to-inferior (uncommon)
- **Image Position (Patient) (0020,0032)**: x, y, z coordinates of first pixel (top-left) in patient coordinate system (mm from origin).
- **Image Orientation (Patient) (0020,0037)**: Row direction cosines (x,y,z) and column direction cosines (x,y,z) relative to patient.
- **Interpretation**: Together, Image Position and Image Orientation define the voxel grid in patient space.
- **Common Chest X-ray Orientations**:
  * PA upright: Typically 'AP' or 'PA' depending on convention; need to check. 
    * Actually, for PA: X-ray enters posterior, exits anterior => Beam direction posterior->anormal.
      Row direction (left-right) is usually Left-to-Right => 'RL'
      Column direction (top-bottom) is usually Superior-to-Inferior => 'FI'
      So Patient Orientation might be 'RLFI' (rows R->L, columns F->I) but varies.
    * Many systems use 'RL' for rows, 'AP' for columns? Need to verify.
  * Standard: Many PACS assume rows left-to-right (RL), columns top-to-bottom (PI? Actually superior-inferior is SI). 
  * Due to variability, rely on DICOM tags and/or anatomical heuristics.
- **Heuristic Method**:
  * Determine approximate location of heart shadow (should be left of midline in PA view)
  * Determine diaphragm contour (should be smooth, right higher than left)
  * If contradictions, suspect incorrect orientation or rotation.
- **Correction**: If orientation suggests patient rotated or flipped, apply rotation/flip to achieve standard anatomical orientation (e.g., feet superior? Usually we want feet inferior). 
  * Typical goal: have patient upright, not rotated, with left side of patient on left side of image (unless marker says otherwise). 
  * Apply transformation (rotate, flip) and update metadata accordingly; log change.

## Patient Position Verification
- **Patient Position (0018,5100)**: Describes patient orientation relative to gravity:
  * HFP: head first prone (lying face down)
  * HFDL: head first decubitus left (lying on left side)
  * HFPU: head first prone up? Actually standard: 
    * HFP: head first prone
    * HFDP: head first decubitus prone
    * HFDL: head first decubitus left
    * HFDR: head first decubitus right
    * HFS: head first supine (lying face up)
    * HFDP: head first decubitus prone
    * FFP: feet first prone
    * etc.
- **Impact on Anatomy**:
  * Supine (HFS): diaphragm may be more pleural; lung bases posterior
  * Prone (HFP): diaphragm anterior; lung posterior
  * Decubitus: dependent lung may be compressed
- **Validation**: Check consistency with View Position and known acquisition protocols; flag if implausible (e.g., PA with patient prone unlikely).

## Projection Types (PA vs AP vs Lateral)
- **View Position (0018,5101)**: 
  * PA: Posteroanterior (X-ray enters posterior, exits anterior)
  * AP: Anteroposterior (enters anterior, exits posterior)
  * LL: Left lateral
  * RL: Right lateral
  * RLD: Right lateral decubitus
  * LLD: Left lateral decubitus
  * etc.
- **Anatomical Implications**:
  * **Heart Magnification**: In AP, heart is closer to anode, appears larger due to divergence; in PA, heart farther from anode, less magnified.
  * **Scapulae Position**: In PA, scapulae should be outside lung fields; in AP, scapulae often overlap lung fields.
  * **Clavicle Equality**: In PA, clavicles should be at similar density equal height; asymmetry may indicate rotation.
  * **Diaphragm**: Should be relatively smooth; blunting may indicate effusion.
- **AI Model Considerations**:
  * Models trained on PA may not generalize to AP without explicit inclusion or domain adaptation.
  * Always record View Position; consider stratifying analysis by projection type.
  * If mixing PA and AP, recommend either:
    * Train separate models
    * Include View Position as feature
    * Apply correction (e.g., estimate heart size ratio) – requires validation
  * Never assume all images are same projection without verification.

## Pixel Spacing
- **Definition**: Physical distance between centers of adjacent pixels in mm, row spacing and column spacing.
- **Source**: DICOM tag (0028,0030) – may be Image Pixel Spacing or Referenced Frame of Reference Sequence.
- **Validation**:
  * Must be present for calibration; if missing, attempt to derive from:
    * ImagerPixelSpacing (0018,1164) – spacing at detector plane
    * Distance Source to Detector (0018,1110) and Distance Source to Patient (0018,1111) to estimate magnification
    * Nominal values from protocol (e.g., 0.14 mm/pixel for typical chest dx)
  * Log assumption and uniformity across dataset.
- **Plausible Range**: 
  * Digital chest radiography: typically 0.1–0.5 mm/pixel
  * Computed radiography (CR): similar range
  * Film digitized: depends on dpi; e.g., 50 µm ≈ 0.05 mm/pixel
  * Flag values outside 0.01–2.0 mm/pixel as suspect.
- **Usage**:
  * Enables conversion of pixel measurements to mm (e.g., cardiothoracic ratio, lesion size in mm)
  * Critical for any size-based feature; if absent, size features must be in pixels only with caveat.
- **Consistency**: 
  * For longitudinal studies, pixel spacing should be consistent per patient; flag large changes.
  * For multi-center studies, expect variation; report distribution.

## Window Width & Window Level
- **Window Center (0028,1050)**: Center of grayscale range to display
- **Window Width (0028,1051)**: Width of grayscale range to display
- **Display Mapping**: 
  * Values below (center - width/2) mapped to black
  * Values above (center + width/2) mapped to white
  * Intermediate values linearly mapped to grayscale
- **Critical Note**: Applying window center/width to pixel data alters the raw relationship between pixel value and X-ray attenuation; should NOT be done for analysis unless:
  * All images are intended to be viewed with same window (e.g., lung window) AND
  * The goal is to simulate that viewing condition for consistency AND
  * This is documented and justified.
- **Typical Windows for Chest Radiography**:
  * Lung Window: Width ~1500, Level ~-600 (to see lucencies)
  * Mediastinal Window: Width ~350, Level ~40 (to see soft tissue, heart)
  * Bone Window: Width ~2000, Level ~400 (to see ribs, spine)
- **Recommendation**: 
  * Preserve raw pixel values (or after rescale slope/intercept) for analysis.
  * Use window/level only for visualization or if explicitly simulating a standard viewing condition with documentation.

## Bit Depth
- **Bits Allocated (0028,0100)**: Number of bits allocated per pixel sample (usually 8 or 16)
- **Bits Stored (0028,0101)**: Number of bits actually used for pixel data (≤ Bits Allowed)
- **High Bit (0028,0102)**: Most significant bit position (should be BitsStored-1)
- **Pixel Representation (0028,0103)**: 0 = unsigned, 1 = signed (2's complement)
- **Interpretation**:
  * Most modern DX: BitsAllocated=16, BitsStored=16, PixelRepresentation=0 (unsigned 16-bit)
  * Some older systems: BitsStored=12 or 10, padded to 16
  * Rarely signed; if signed, must interpret correctly to avoid negative values where none expected.
- **Validation**:
  * Ensure BitsStored ≤ BitsAllocated
  * HighBit = BitsStored - 1
  * If PixelRepresentation=1, check that values are plausible (e.g., not negative large magnitudes unless offset)
- **Conversion for Processing**:
  * Extract pixel values as integers per correct signedness
  * If needed, convert to float for processing; maintain scale
  * Common practice: scale to [0,1] or [-1,1] after applying rescale slope/intercept if present

## Grayscale Interpretation
- **PhotometricInterpretation (0028,0004)**:
  * MONOCHROME1: higher pixel value = more radiodense (whiter = more attenuation) – considered standard for radiographic image storage
  * MONOCHROME2: higher pixel value = less radiodense (whiter = less attenuation) – requires inversion for standard interpretation
  * Other values (RGB, YBR_FULL, etc.) should not appear for chest x-ray unless color image (unlikely)
- **Determining Need for Inversion**:
  * If PhotometricInterpretation = MONOCHROME2, invert: `pixel_value = max_possible - pixel_value`
    * max_possible = 2^BitsStored - 1
  * If MONOCHROME1, use as-is (after considering signedness)
- **Verification**:
  * After potential inversion, check that air (lung) appears dark (low values) and bone appears bright (high values)
  * Simple heuristic: mean of lung region (lower percentile) < mean of bone region (upper percentile)
  * If opposite, likely still inverted or other issue; log and investigate.

## Burned-in Annotations
- **Definition**: Text, arrows, overlay icons, or other graphics that have been burned into the pixel data, becoming part of the image intensity values.
- **Sources**:
  * Hardcoded annotations from modalities or PACS
  * User-added measurements or labels
  * Institution-specific markers
- **Detection Methods**:
  * OCR (Optical Character Recognition) for text regions
  * Connected component analysis on high-gradient or thresholded regions
  * Frequency domain analysis: abrupt high-frequency spikes suggesting lines/text
  * Template matching for common symbols (arrows, boxes)
- **Impact on AI**:
  * Can obscure pathology (e.g., text over lung nodule)
  * Can be mistaken for pathology (e.g., straight line resembling linear opacity)
  * Introduces site-specific bias if annotations correlate with site
- **Handling**:
  * Ideal: obtain clean images without burned-in annotations
  * If unavoidable: flag images with annotations over lung fields for exclusion or manual review
  * Annotations outside lung fields may be less harmful but still should be logged
  * Never attempt to "remove" via inpainting without validation; may create artifacts

## Laterality Markers
- **Definition**: Physical markers (usually 'L' and 'R') placed on the patient to indicate left and right sides.
- **Location**: Typically adjacent to image, often in corners; may appear as radiopaque letters.
- **Importance**:
  * Prevents laterality errors (swapping left/right)
  * Critical for asymmetric pathology (e.g., unilateral infiltrate, pneumothorax)
  * Required by radiology best practices
- **Detection**:
  * Template matching for 'L' and 'R' characters
  * Connected component analysis of high-intensity regions matching expected size/font
  * Use known approximate location from acquisition protocol (e.g., always upper left)
- **Validation**:
  * Confirm marker matches laterality determined from anatomy or Image Laterality tag
  * If marker missing, flag for review
  * If marker present but contradicted by anatomy, investigate possible marker placement error or image flip
- **Impact of Missing/Incorrect Marker**:
  * Model may learn inverse associations (e.g., left lung patterns labeled as right)
  * Could degrade performance and introduce systematic error
- **Recommendation**:
  * Require laterality marker presence for training data unless robust anatomical laterality verification is in place
  * Log presence and correctness per image
  * Consider using marker as ground truth for laterality if anatomy ambiguous

## Medical Image Validation
- **File Integrity**:
  * Confirm file is not truncated: check for valid DICOM preamble and correct length
  * For raster images: attempt to load; catch exceptions
- **Pixel Data Integrity**:
  * After loading, verify no NaN or Inf values (if floating)
  * Check that all values are within expected bit depth range
  * Flag uniform images (all same value) – may indicate corruption
- **Metadata Consistency**:
  * Cross-check related tags (e.g., Rows with actual pixel array height)
  * Verify that Pixel Spacing, if present, matches expected dimensions and physical size if known
  * Check that Instance Number, etc., are sequential if expected
- **Anomaly Detection**:
  * Outlier intensities: excessive clipping at min or max may indicate over/under exposure
  * Histogram bimodality: expected for lung/air vs soft tissue/bone
  * Very low entropy: may indicate missing data or corruption
  * Grid detection: periodic patterns in Fourier transform
- **Patient Identity Verification**:
  * If multiple studies, ensure Patient ID consistent across series unless known change
  * Guard against mix-ups: if Patient ID changes unexpectedly, log for investigation
- **Study/Series Consistency**:
  * Verify Study Date, Modality, etc., are consistent within a reasonable for chest x-ray
  * Flag if Modality is not DX/CR for a chest x-ray dataset
- **Laterality and Orientation Verification**:
  * As described: use tags, anatomical heuristics, laterality markers
  * Log method used and confidence
- **Image Completeness**:
  * Estimate lung field bounding box via simple threshold or known proportions
  * Check if significant portion of expected lung region is missing (e.g., >20% clipped)
- **Artifact Flags**:
  * Motion Blur: low frequency blur metric
  * Grid Lines: periodic detection
  * Metallic Implants: high intensity localized objects
  * Jewelry: similar to implants but often superficial
- **Decision Logic**:
  * Define pass/fail criteria per project (e.g., "must have correct laterality, no burned-in text over lungs, pixel spacing within 0.1-0.5 mm")
  * Images failing critical criteria may be excluded from training but retained for testing with flag
  * Less critical flags (e.g., minor artifact outside lung) may warrant warning but not exclusion

## Quality Assurance
- **Automated Checks**:
  * Percentage of images with valid DICOM parsing
  * Percentage with correct laterality (by marker or anatomy)
  * Percentage with plausible pixel spacing (0.1-0.5 mm)
  * Percentage without burned-in annotations over lung fields
  * Percentage with adequate lung field visibility (e.g., <20% clipping)
  * Distribution of View Position (PA vs AP)
  * Average image entropy (sudden drops may indicate corruption)
- **Manual Review Triggers**:
  * Any image flagged for burned-in text over lungs
  * Images with conflicting laterality (marker vs anatomy)
  * Images with extreme pixel spacing outliers
  * Images with visible gross corruption
  * Subset random sample for radiologist review
- **Documentation**:
  * Maintain validation log with timestamps and reasons for flags
  * Generate summary report for stakeholders
  * Record any manual overrides or exclusions with justification
- **Reproducibility**:
  * Same input data + same validation config = same output flags
  * Archive exact versions of validation scripts and configuration
  * If manual decisions made, document rationale

## Clinical Safety
- **Never Alter Anatomy**: 
  * Do not crop, rotate, or flip in ways that violate anatomical plausibility without strong evidence and documentation
  * Always verify that lung fields, heart, diaphragm remain intact and correctly oriented
- **Never Misrepresent Laterality**:
  * Incorrect laterality can lead to life-threatening misinterpretation (e.g., missing left pneumothorax)
  * Always prioritize laterality marker; if missing, use anatomical heuristics with caution
- **Never Remove Diagnostic Information Without Justification**:
  * Do not aggressively denoise to the point of removing quantum noise that may contain texture information
  * Do not apply windowing that removes information outside window without documenting loss
- **Never Assume Uniform Acquisition**:
  * Always validate pixel spacing, view position, laterality per image; do not assume consistency across dataset
- **Never Ignore Artifacts That Mimic Pathology**:
  * Grid lines can mimic linear opacities; motion blur can mimic infiltrates; metallic implants can mimic calcifications
  * Flag and either exclude or ensure model learns to ignore via diverse training
- **Never De-identify Without Traceability Plan**:
  * If removing PatientID, ensure study-level identifiers remain for grouping or provide hashable pseudonyms
  * Maintain ability to link back to source for audit if needed (under IRB)
- **Never Process Without Logging**:
  * Every validation step, assumption, and correction must be logged for traceability
- **Never Release Model Trained on Questionable Data**:
  * If validation reveals systemic issues (e.g., 40% images have burned-in text over lungs), address before training
- **Never Override Radiologist Judgment**:
  * If a radiologist flags an image as unsuitable for AI training due to artifact, respect that unless override justified with evidence

## Image Standardization
- **Goal**: Create consistent representation while preserving diagnostic information.
- **Steps (Order Matters)**:
  1. **Load and Validate**: As per workflow above
  2. **Apply Modality LUT (if present and justified)**: Rare for DX
  3. **Photometric Interpretation Correction**: Invert if MONOCHROME2 to achieve standard (higher value = more radiodense)
  4. **Apply Rescale Slope/Intercept (if present)**: Convert to optical density (OD) or Hounsfield-like units
     * NewPixel = Pixel * RescaleSlope + RescaleIntercept
     * For DX: often slope=1, intercept=0; but check
  5. **Optional: Apply VOI LUT (Only If Justified for Standardized Viewing)**:
     * Not recommended for analysis; use only if simulating a standard window and documented
  6. **Orientation Correction**: Rotate/flip to achieve standard anatomical orientation (e.g., feet inferior, no rotation, laterality correct)
  7. **Resizing**: 
     * Preserve aspect ratio via padding (preferred) or justify stretching
     * Use interpolation method justified for detail preservation (e.g., Lanczos)
  8. **Padding**: 
     * Use reflection or edge replication to avoid artificial boundaries
     * Avoid zero-padding that creates false dark edges
  9. **Normalization**: 
     * Apply using statistics from TRAINING SET ONLY to prevent leakage
     * Method justified (e.g., z-score for batch norm compatibility)
  10. **Optional: Anatomically Plausible Augmentation (Training Only)**:
       * Rotation ±15°, scaling 0.9-1.1, translation ±10%
       * No horizontal flip (laterality significant)
       * No shear or elastic distortion
  * **Never**:
      * Apply arbitrary gamma correction
      * Use global histogram equalization
      * Apply windowing to pixel data for analysis
      * Flip vertically (inverts anatomy)
      * Use nearest-neighbor resize causing blocking artifacts
      * Assume all images are same orientation or laterality without verification
      * Apply non-rigid deformations without strong validation

## Metadata Preservation
- **Which Metadata to Keep**:
  * Study and Series UIDs for traceability
  * Acquisition parameters (kVp, mAs) if available for potential stratification
  * View Position and Patient Position for laterality/orientation analysis
  * Pixel Spacing for calibration
  * Photometric Interpretation and BitsStored for correct pixel interpretation
  * Manufacturer and Station Name for site effect analysis
  * Instance Number for ordering if temporal
  * Any private tags deemed relevant after review (e.g., radiation dose)
- **Which Metadata to Remove/De-identify**:
  * Patient Name (0010,0010)
  * Patient ID (0010,0020) – consider pseudonymization if needed
  * Patient Birth Date (0010,0030) – may keep year only for age calculation if needed
  * Other overtly PHI tags per Safe Harbor or expert determination
- **Methods**:
  * Replace with hashes or remove entirely
  * Keep audit trail linking original to de-identified if required
  * Never remove metadata that is necessary for scientific validity (e.g., Study Date for longitudinal)
- **Logging**:
  * Record which tags were stripped, hashed, or retained
  * Justify each decision (e.g., "Patient ID hashed to allow subject-level aggregation while protecting identity")

## Error Handling
- **Critical Errors (Stop Processing Image, Log and Continue)**:
  * File not found or inaccessible
  * File not a valid DICOM or supported raster format
  * DICOM missing required PixelData and cannot generate pixel array
  * Pixel data dimensions zero or negative after loading
  * Irrecoverable corruption (e.g., cannot read past header)
- **Recoverable Errors (Log Warning, Apply Fallback or Skip Step)**:
  * Missing optional tag (e.g., Pixel Spacing): log assumption, use default if configured, flag
  * Inconsistent tags (Rows vs actual height): use actual pixel array dimensions, log discrepancy
  * PhotometricInterpretation unknown: assume MONOCHROME1, log warning
  * Unable to determine laterality via tags or anatomy: mark as unknown, flag for review
  * Burned-in annotation detection fails: log and continue
  * Artifact detection algorithm fails: log and continue with other checks
- **Logging Requirements**:
  * Every error/warning must include:
    * Filename or SOP Instance UID
    * Specific tag or operation that failed
    * Error message or description
    * Timestamp
    * Stack trace if unexpected exception (in verbose/debug mode)
  * Continue processing remaining images after logging
  * Never exit entire batch due to single image failure unless configured to halt on critical errors
- **Recovery Strategies**:
  * If pixel data load fails, attempt to read as raster image with Pillow as fallback
  * If orientation tags missing/unclear, attempt anatomical heuristic and log confidence
  * If pixel spacing missing, use configured nominal value (e.g., 0.14 mm) and flag as assumed

## Performance Considerations
- **I/O**:
  * Stream files sequentially; avoid loading all paths into memory if list huge
  * Use buffered reading; consider memory-mapped files for very large DICOMs if needed
  * Cache parsing results if re-running validation on same set (unlikely but possible)
- **Computation**:
  * Vectorize metadata extraction where possible (e.g., process batches of tags with pandas)
  * For pixel-level operations (artifact detection, lateralityheuristic), process per image but avoid nested loops over pixels when possible
  * Use efficient OCR libraries (pytesseract) only if needed; consider lighter alternatives
  * Precompute constants (e.g., max pixel value per bit depth)
- **Memory**:
  * Process one image at a time if memory constrained
  * Release pixel arrays after metadata extraction and validation
  * Store only metadata and flags in memory; offload pixel arrays to disk if needed for next step
- **Logging**:
  * Buffer log writes; flush periodically (e.g., every 100 images or 5 seconds)
  * Use appropriate log levels (DEBUG for per-pixel timing, INFO for milestones)
  * In production, reduce verbosity to warnings and errors
- **Storage**:
  * Store validation outputs as compressed CSV (optional) or Parquet for efficiency
  * Keep logs rotated or purged per policy
  * If saving intermediate validated images, use lossless compression (PNG) with moderate compression level

## Documentation Requirements
- **Code Docstrings**:
  * Every function must describe purpose, inputs, outputs, and justification for any non-obvious steps
  * Reference DICOM standard sections or radiology literature for key assumptions
  * Note any assumptions made (e.g., "assumes Pixel Spacing present; if missing, uses 0.14 mm")
- **Configuration Documentation**:
  * Every parameter in config must include:
    * Description
    * Justification (e.g., "based on analysis of JCXR dataset pixel spacing distribution")
    * Acceptable values or range
    * Default value
  * Maintain separate schema file for validation
- **Process Documentation**:
  * Maintain `MEDICAL_IMAGING_VALIDATION_PROTOCOL.md` detailing:
    * Step-by-step validation pipeline
    * Justification for each validation criterion
    * Expected input formats and output schemas
    * Quality control checkpoints and exit criteria
  * Update `docs/medical_imaging_guide.md` with:
    * How to run the validator
    * How to interpret logs and output files
    * Troubleshooting common validation failures
- **Output Documentation**:
  * CSV files: clear headers, units where applicable (mm for spacing, counts for artifacts)
  * JSON manifests: include schema description or link to JSON Schema file
  * Log files: standard format with timestamp, level, message; consider JSON lines for machine parsing
- **Versioning**:
  * Include software used
  In output format version, etc.
- **Audit Trail**:
  * Record software versions (Python, pydicom, Pillow, etc.) in logs and manifest
  * Include configuration hash with outputs to detect config drift
  * Preserve raw validation logs for re-audit
- **Traceability**:
  * Every output row must trace back to exactly one input file via SOP Instance UID or filename
  * Enable reverse lookup from metadata to original raw file location

## Interaction with Other Skills
- **Consumes From Dataset Engineer**:
  * List of raw image files to validate
  * Initial metadata (if any) from prior dataset passes
  * Configuration pathways (where to find raw data, where to output processed data)
- **Provides Guidance To**:
  * **Segmentation Engineer**: 
    * Confirm lung fields are intact and correctly oriented for mask generation
    * Advise on laterality importance for symmetric vs asymmetric pathologies
    * Warn about artifacts that may be mistaken for lung boundaries (e.g., clipped diaphragms)
  * **Classification Engineer**:
    * Confirm image laterality is correct to avoid label inversion
    * Advise on View Position (PA vs AP) effects on cardiothoracic ratio and appearance
    * Flag images with burned-in text over lung fields that may obscure pathology
    * Recommend stratification by acquisition parameters if available
  * **Explainability Engineer**:
    * Ensure spatial relationships preserved for accurate Grad-CAM localization
    * Warn about artifacts that may attract saliency incorrectly (e.g., grid lines)
    * Confirm pixel spacing adherence if size-based explanations are desired
  * **Backend Engineer**:
    * Inform on metadata fields to preserve for API responses (patient ID hash, study date, laterality)
    * Advise on image orientation requirements for consistent display
    * Note any de-identification requirements for stored or transmitted images
  * **Frontend Engineer**:
    * Specify orientation corrections needed for consistent viewport display
    * Recommend laterality markers UI overlay if present in original
    * Indicate which annotations (burned-in) should be hidden or flagged in viewer
  * **Research Engineer**:
    * Provide metadata cohort description (age/sex distribution if available, modality mix)
    * Report validation pass rates per criterion to support methodology section
    * Document any exclusions and justification for reproducibility
  * **Evaluation Engineer**:
    * Ensure test set meets same validation criteria as training to avoid distribution shift
    * Verify no laterality or orientation leakage between splits
    * Confirm artifact distribution similar across sets (or document differences)

## Quality Checklist (Must Pass Before Handoff to Downstream)
* [ ] Every image file attempted for validation has a log entry (success or failure reason)
* [ ] No hardcoded thresholds, paths, or policy values; all configuration-driven
* [ ] Fixed random seed used for any stochastic validation step (if any)
* [ ] Normalization statistics (if computed) derived exclusively from training set
* [ ] Augmentation (if any) applied only during training, never validation/test
* [ ] All DICOM files parsed with pydicom; failures logged with specific error
* [ ] All non-DICOM files attempted with Pillow; failures logged
* [ ] Photometric interpretation correction applied and logged when needed
* [ ] Orientation correction applied only when justified by tags or anatomical heuristics; log method and confidence
* [ ] Laterality determination method recorded per image (tag, marker, heuristic, unknown)
* [ ] Pixel spacing validated; missing values logged with assumption source
* [ ] Burned-in annotation detection performed; location and lung overlap logged
* [ ] Laterality marker presence and correctness logged per image
* * Image completeness assessed; lung field clipping estimated and logged
* [ ] Artifact detection (motion blur, grid, implants) performed where feasible; results logged
* [ ] No anatomical alteration performed without logging justification (e.g., rotation >15° flagged)
* [ ] No vertical flips applied (would invert anatomy)
* [ ] No horizontal flips applied unless laterality proven irrelevant (extremely rare in chest radiographs)
* [ ] No windowing applied to pixel data for analysis unless justified and documented
* [ ] No global histogram equalization applied
* [ ] No gamma correction applied as default step
* [ ] All output files (CSV, manifest, log) generated and non-empty
* [ ] Manifest contains checksums for input files to detect future changes
* [ ] Output directory structure matches expectations for Dataset Engineer and Computer Vision Engineer
* [ ] Log file contains no unhandled exceptions that halted processing
* [ ] Summary report includes validation rates per criterion and recommendations
* [ ] Configuration file archived with outputs for reproducibility
* [ ] Patient IDs (if present) are either retained (with IRB) or consistently pseudonymized
* [ ] Study and Series UIDs preserved for traceability
* [ ] Any manual exclusions documented with justification in report

## Common Mistakes
* **Assuming PhotometricInterpretation is always MONOCHROME1**: Leads to inverted images where lungs appear bright
* **Ignoring PixelRepresentation (signed vs unsigned)**: Results in negative pixel values being interpreted as large positives
* **Applying VOI LUT Window Center/Width to pixel data**: Alters raw relationship to attenuation, destroying quantitative information
* **Using horizontal flip in chest X-rays**: Creates mirror-image heart (apparent dextrocardia) which is pathological
* **Zero-padding during resizing**: Creates artificial black edges that may be mistaken for pathology (e.g., pleural effusion)
* **Nearest-neighbor resizing**: Produces blocky artifacts that mimic calcifications or fibrosis
* **Assuming all images are PA**: Ignoring AP acquisition leads to heart size bias and scapular overlap confusion
* **Missing laterality marker and not verifying anatomically**: Results in left/right label swaps in training data
* **Using arbitrary normalization (e.g., divide by 255) without verifying actual bit depth**: Incorrect scaling if images are 12-bit packed in 16-bit
* **Applying global histogram equalization**: Amplifies noise and alters gross anatomy shading unrealistically
* **Failing to preserve aspect ratio via stretching**: Causes anatomical distortion (e.g., elongated heart, flattened lungs)
* **Using elasticity or shear augmentation**: Creates anatomically impossible spinal or rib deformations
* **Treating quantum noise as removable artifact**: Noise is inherent to the modality; over-denoising removes texture that may contain diagnostic information
* **Assuming pixel spacing is uniform across dataset without checking**: Leads to incorrect size estimates if variable
* **Ignoring View Position (PA vs AP)**: Models may inadvertently learn projection-specific artifacts as pathology
* **Not logging assumptions**: Makes reproducibility impossible and hides potential bias sources
* **Silently skipping corrupted images**: Results in silent data loss and potential bias if corruption is not random
* **Using training set statistics for normalization on validation/test**: Introduces data leakage and optimistically biased performance
* **Applying augmentation to validation or test sets**: Invalidates evaluation by artificially improving performance
* **Processing images in non-deterministic order (e.g., unsorted glob)**: Leads to different splits between runs due to hash seeding
* **Failing to verify DICOM transfer syntax and endianness**: Results in garbled pixel data (e.g., swapped bytes)
* **Assuming all grayscale images are 8-bit**: Fails to handle 12-bit or 16-bit data correctly, causing clipping or scaling errors
* **Applying lung window as default preprocessing**: Removes information outside window (e.g., supraclavicular structures, diaphragm contour)
* **Using rigid body transformations that exceed physiological limits**: E.g., >30° rotation creates impossible spine orientation
* **Not validating laterality with both marker and anatomy**: Increases risk of laterality errors
* **Overlooking burned-in annotations in lung fields**: Obscures pathology and can be mistaken for disease
* **Assuming metadata from filename is reliable without cross-check**: Filenames may be inconsistent or incorrect
* **Using OCR without validation on medical text region localization**: May misinterpret anatomy as text
* **Assuming all artifacts are benign**: Some (grid, metal) can mimic or obscure pathology
* **Neglecting to de-identify or inadequately de-identifying PHI**: Risk of privacy breach
* **Over-de-identifying and losing necessary traceability**: Hinders auditing and reproducibility
* **Not preserving Study/Series UIDs**: Breaks ability to group images by patient or study for longitudinal analysis
* **Assuming laterality is irrelevant for the task**: Laterality is almost always clinically relevant in chest radiographs
* **Using interpolation methods that cause ringing (e.g., sinc) without justification**: Introduces artificial oscillations near edges
* **Applying Confidence-weighted averaging without basis**: May blur edges and reduce diagnostic sharpness
* **Treating the image as if it were natural photography**: Chest X-rays have specific physics and anatomy; generic CV assumptions fail

## Never Do
* **Never** hardcode image dimensions, pixel spacing, or viewport settings; always derive from configuration or metadata
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
* **Never** apply windowing as a default preprocessing step; it is a display operation, not a preprocessing step for analysis
* **Never** use training/test splits that share patients if patient IDs are available; always enforce patient-level separation
* **Never** ignore aspect ratio when resizing; always pad to preserve original ratio or provide justification for stretching
* **Never** use batch processing non-deterministically (e.g., shuffling without fixed seed) for reproducible splits
* **Never** validate augmentation by looking only at loss; visually inspect augmented samples for anatomical plausibility
* **Never** treat quantum noise as something to remove; it is inherent to the imaging modality and may contain texture information
* **Never** assume laterality is irrelevant in chest X-rays without validation via cardiac position or laterality markers
* **Never** process images without logging the exact parameters used for each image (e.g., rotation angle, padding value)
* **Never** use image processing libraries without understanding their default behaviors (e.g., OpenCV vs PIL coordinate systems, origin at top-left)
* **Never** assume DICOM images are little-endian; always check (0002,0010) Transfer Syntax UID and convert if needed
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
The Medical Imaging Engineer skill delivers a rigorous, auditable, and clinically sound medical image validation and preparation layer that ensures:
1. **Correct Image Interpretation**:
   * Accurate extraction and validation of DICOM metadata (patient ID, study date, modality, view position, laterality, pixel spacing)
   * Proper handling of PhotometricInterpretation and PixelRepresentation to achieve standardized grayscale (higher value = more radiodense)
   * Correct orientation and laterality determination using DICOM tags, anatomical heuristics, and laterality markers
   * Preservation of Study and Series UIs for traceability and longitudinal analysis
2. **Artifact and Quality Awareness**:
   * Detection and logging of common chest radiography artifacts (grid lines, motion blur, metallic implants, under/over exposure)
   * Identification of burned-in annotations and their overlap with lung fields
   * Assessment of image completeness (lung field coverage, apices and bases visibility)
   * Laterality marker presence and correctness verification
   * Pixel spacing validation for calibration-dependent analyses
3. **Auditability and Traceability**:
   * Comprehensive processing log with entry for every input image (success/failure, applied corrections, timestamps)
   * Manifest file mapping each input to extracted metadata, validation flags, and any corrections applied
   * Preservation of original file checksums for integrity verification
   * Software and dependency versions captured for environment reconstruction
4. **Configuration-Driven and Reproducible**:
   * All validation thresholds, handling policies, and output paths specified in configuration file with justification
   * Fixed random seeds for any stochastic elements (if used)
   * Deterministic output: same input + same config = same validation results and derived corrections
   * Output directory structure consumable by downstream engineers (Dataset Engineer, Computer Vision Engineer)
5. **Downstream Model Readiness**:
   * Validated images (or metadata) that serve as reliable inputs for:
     * Lung segmentation U-Net (correct lung field integrity, orientation, laterality)
     * TB classification DenseNet-121 (consistent laterality, no label inversion, artifact awareness)
     * Explainability Engineer (Grad-CAM) (spatial fidelity preserved, no misleading artifacts)
   * Guaranteed no leakage from metadata handling (e.g., patient IDs not inadvertently shared across splits)
   * Clear flags for images that may require exclusion or special handling (e.g., burned-in text over lungs)
6. **Documentation and Compliance**:
   * Self-describing output files (CSV headers, JSON schemas, log formats)
   * Processing protocol document justifying every validation step and criterion
   * Configuration file enabling exact replication of validation pipeline
   * Summary report providing validation statistics and recommendations for dataset usage
   * Compliance with medical AI principles: no altered anatomy, no misrepresented laterality, no unjustified information loss
7. **Risk Mitigation**:
   * Early detection of systemic issues (e.g., 30% of images have burned-in text over lymphocytes) before model training
   * Guidance on exclusions or manual cleanup to improve data quality
   * Documentation of assumptions and limitations for transparent reporting
   * Support for ethical and regulatory review via traceable metadata handling and de-identification logs

In summary, the Medical Imaging Engineer skill ensures that every chest X-ray entering the AI pipeline has been scrutinized through the lens of radiology best practices, DICOM standards, and clinical safety, thereby eliminating a major source of bias, artifact, and misinterpretation in medical AI research and enabling trustworthy, explainable AI-assisted tuberculosis screening.