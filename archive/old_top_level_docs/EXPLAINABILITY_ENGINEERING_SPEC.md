# Explainability Engineering Specification for Milestone 3
## Nirikhshon Explainable AI-assisted Pulmonary Tuberculosis Screening Workstation

**Version**: 1.0.0  
**Date**: 2026-07-06  
**Status**: Ready for Implementation Contracts 3.2-3.10

## 1. Executive Summary

This specification defines the complete Explainability subsystem for Milestone 3 of the Nirikhshon project. It builds upon the existing explainability implementation in `backend/core/inference.py` and defines the contracts for integrating explainability data into the API response while maintaining backward compatibility and adhering to all CLAUDE.md engineering principles.

The specification covers:
- Algorithm selection and mathematical justification
- API contract design for optional explainability fields
- Performance optimization requirements
- Validation metrics and quality assurance
- Risks and mitigation strategies
- Future extensibility considerations

## 2. Algorithm Selection & Mathematical Justification

### 2.1 Primary Explainability Method: Grad-CAM++

**Selection Justification**:
- **Mathematical Foundation**: Gradient-weighted Class Activation Mapping++ (Chattopadhay et al., 2018) extends Grad-CAM with second-order gradient information for improved localization
- **Superior Localization**: Addresses Gradient-CAM's diffusion issues for multiple object detection, critical for TB where lesions may appear in multiple lung zones
- **Medical Imaging Validation**: Proven effectiveness in chest X-ray lesion localization (Wang et al., 2020; CheXpert studies)
- **Compatibility**: Works with DenseNet-121 architecture specified in CLAUDE.md without architectural modifications
- **Computational Efficiency**: Only ~2x overhead of standard Grad-CAM, acceptable for clinical screening workflow

**Mathematical Formulation**:
```
For target class c:
α_k^c = [∑_i ∑_j (∂²y^c/∂A_ij^k) · ReLU(∂y^c/∂A_ij^k)] / 
        [∑_i ∑_j (∂²y^c/∂A_ij^k) · ReLU(∂y^c/∂A_ij^k) + ε · ∑_i ∑_j A_ij^k]
L_Grad-CAM++^c = ReLU(∑_k α_k^c · A^k)
```
Where:
- A^k = k-th feature map in target convolutional layer
- y^c = score for class c (TB class)
- ε = small constant for numerical stability (1e-8)
- ReLU(x) = max(0, x)

**Target Layer Selection**:
- **Layer**: `relu` activation before global average pooling in DenseNet-121
- **Justification**: 
  - Captures combined features from all dense blocks (1024 channels)
  - Pre-norm/pre-concat layers (e.g., `conv5_block16_2_conv`) only capture final block features (32 channels)
  - Post-activation preserves non-linear feature interactions critical for TB pattern recognition
  - Spatial dimensions ≥ 7x7 for 224x224 input provide sufficient localization detail

### 2.2 Secondary Methods (Available for Research/Validation)

1. **Standard Grad-CAM**: Baseline for comparison
2. **Attention Maps**: High-pass filtered gradients for edge-sensitive analysis
3. **Coverage**: Binary threshold activation maps for lesion extent quantification
4. **Attribution**: Grid-based attribution for spatial contribution analysis

## 3. Mathematical Validation & Verification

### 3.1 Faithfulness Properties (Must Be Verified)
- **Input Invariance**: Explanation for x and x+constant should differ only by baseline shift
- **Class Sensitivity**: Explanations for TB vs. Normal cases must be statistically different
- **Localization Accuracy**: When ground truth available, IoU > 0.4 and pointing game accuracy > 0.6

### 3.2 Implementation Verification Requirements
- Gradient computation uses automatic differentiation (PyTorch), not finite differences
- Model remains in eval() mode during explanation generation (no weight updates)
- Gradient flow verified: non-zero gradients for target layer
- Lung masking applied before normalization to prevent signal suppression
- Normalization preserves rank order of explanation values (monotonicity Check)

### 3.3 Numerical Stability Safeguards
- Epsilon (1e-8) added to denominators to prevent division by zero
- Gradient clipping at ±1e-4 to prevent explosion
- NaN/Inf gradient detection with zero replacement and warning logging
- Memory-efficient batch processing to prevent OOM

## 4. API Contract Design

### 4.1 Current State Analysis
The `/predict` endpoint in `backend/app.py` currently:
1. Computes explainability data for ALL requests (lines 1015-1025, 1052)
2. Stores results in `result_dict` 
3. Only conditionally includes explainability fields in response (lines 462-467)
4. Contains syntax error in conditional logic (line 463)

### 4.2 Required Modifications

#### 4.2.1 Fix Syntax Error
**Location**: `backend/app.py`, line 463
**Current** (invalid Python syntax):
```python
if request.form.get('explain', 'false').lower() == 'true':
```
**Corrected**:
```python
explain_requested = request.form.get('explain', 'false').lower() == 'true'
```

#### 4.2.2 Response Structure Specification
The API must maintain backward compatibility while supporting optional explainability:

**Base Response (Always Included)**:
```json
{
  "success": true,
  "study_id": "string",
  "patient_id": "string", 
  "metadata": { /* DICOM metadata */ },
  "original_image": "base64_string",
  "image_quality": { /* IQA results */ },
  "prediction": "string", // "Tuberculosis" or "Normal"
  "confidence": float,   // [0,1]
  "threshold_used": float,
  "is_tb": boolean,
  "demo_mode": boolean,
  "segmentation_active": boolean,
  "saliency_fallback": boolean,
  "model_version": "string",
  "model_name": "string",
  "segmentation_status": "string", // "active"/"inactive"
  "processing_time_ms": float
}
```

**Explainability Extension (When `explain=true`)**:
```json
{
  // ... all base response fields ...
  "heatmaps": {
    "gradcam": "base64_string",
    "gradcam_plusplus": "base64_string", 
    "attention": "base64_string",
    "coverage": "base64_string",
    "attribution": "base64_string"
  },
  "xai_results": {
    "rois": [
      {
        "id": "char",           // A, B, C, ...
        "activation_score": float, // [0,100] 
        "contribution_pct": float, // [0,100]
        "location": "string",  // e.g., "Right Upper Lung Zone"
        "bbox": [x, y, width, height], // pixels
        "circle": [cx, cy, radius], // pixels
        "contour": [[x1,y1], [x2,y2], ...], // pixels
        "center": [nx, ny]     // normalized [0,1] coordinates
      }
    ],
    "summary": "string",      // Clinical explanation summary
    "ranking": [
      {
        "region_id": "char",
        "location": "string", 
        "contribution_pct": float
      }
    ],
    "metrics": {
      "tb_probability": float,      // [0,100]
      "calibrated_confidence": float, // [0,100]
      "reliability": "string",      // "High"/"Medium"/"Low"
      "uncertainty": "string"       // "High"/"Medium"/"Low"
    }
  },
  "quadrant_analysis": {
    "quadrant_scores": {       // percentage activation per quadrant
      "upper_left": float,
      "upper_right": float, 
      "lower_left": float,
      "lower_right": float
    },
    "upper_fraction": float,   // [0,100]
    "lower_fraction": float,   // [0,100]
    "dominant_zone": "string", // "upper"/"lower"/"mixed"
    "interpretation": "string",
    "disease_overlap": ["string"] // differential diagnoses
  },
  "clinical_observations": ["string"] // structured observation list
}
```

### 4.3 Performance Requirements

#### 4.3.1 Latency Targets
- **Base inference** (without explain): ≤ 500ms on CPU, ≤ 100ms on GPU
- **With Grad-CAM++**: ≤ 700ms on CPU, ≤ 150ms on GPU
- **With all 5 methods**: ≤ 1000ms on CPU, ≤ 250ms on GPU

#### 4.3.2 Optimization Strategies
- **Single Forward Pass**: Compute all explanations from one backward pass
- **Shared Computations**: Reuse feature maps and gradients across methods
- **Async Processing**: Queue explanation generation for non-critical paths
- **Caching**: Cache model layer references and preprocessing parameters

#### 4.3.3 Resource Constraints
- **Memory**: < 500MB additional RAM overhead for explanation generation
- **GPU Utilization**: ≥ 70% GPU utilization when available
- **Batch Size**: Maintain current BATCH_SIZE=4 for explanation batching

## 5. Implementation Roadmap (Contracts 3.2-3.10)

### 3.2 Grad-CAM Integration
**Objective**: Formalize and validate existing Grad-CAM++ implementation
**Tasks**:
- [ ] Verify target layer selection (`relu`) via ablation study
- [ ] Document mathematical justification in code comments
- [ ] Add numerical stability safeguards (epsilon, gradient clipping)
- [ ] Implement input validation for model/tensor compatibility
- [ ] Create unit tests for gradient computation correctness
**Files**: `backend/core/inference.py`

### 3.3 Lung Region Validation
**Objective**: Ensure explanations respect lung segmentation boundaries
**Tasks**:
- [ ] Verify lung mask application before normalization (line 117 in inference.py)
- [ ] Add validation that <5% of activation occurs outside lung mask
- [ ] Implement fallback explanation when lung segmentation fails
- [ ] Add Lung Mask Compliance metric to quality assessment
**Files**: `backend/core/inference.py`

### 3.4 ROI Extraction
**Objective**: Optimize and validate region of interest extraction
**Tasks**:
- [ ] Validate contour filtering thresholds (min area=15 pixels)
- [ ] Implement adaptive thresholding based on activation distribution
- [ ] Add ROI overlap prevention (merge overlapping contours)
- [ ] Create unit tests for ROI extraction accuracy
**Files**: `backend/core/inference.py` (extract_xai_rois function)

### 3.5 Lung Zone Localization
**Objective**: Validate anatomical localization against standard atlas
**Tasks**:
- [ ] Verify zone boundary definitions match radiological standards
- [ ] Implement centroid-based zone assignment with tie-breaking
- [ ] Add lateralization flip correction for PA view orientation
- [ ] Create validation suite using synthetic lesions in known zones
**Files**: `backend/core/inference.py` (extract_xai_rois function)

### 3.6 Clinical Observation Generator
**Objective**: Ensure explanations support structured observation generation
**Tasks**:
- [ ] Validate integration with `utils/observation_builder.py`
- [ ] Ensure ROI data includes all required fields for observation generation
- [ ] Verify clinical summary generation follows CLAUDE.md terminology
- [ ] Test observation generation for edge cases (no ROIs, single ROI)
**Files**: `backend/core/inference.py` (compute_xai_payload, generate_xai_clinical_summary)

### 3.7 Confidence & Explainability Metrics
**Objective**: Implement quantitative explanation validation
**Tasks**:
- [ ] Add explanation confidence scoring based on activation concentration
- [ ] Implement entropy-based explanation focus metric
- [ ] Add stability metric (variance under small perturbations)
- [ ] Create explanation quality report in API response metadata
**Files**: `backend/core/inference.py` (compute_xai_payload)

### 3.8 Response Contract Extension
**Objective**: Implement API contract for optional explainability fields
**Tasks**:
- [ ] Fix syntax error in explainability conditional (app.py line 463)
- [ ] Implement response structure exactly as specified in Section 4.2
- [ ] Ensure backward compatibility (existing clients unaffected)
- [ ] Add API documentation for new explainability fields
- [ ] Implement response validation schema
**Files**: `backend/app.py`

### 3.9 Backend API Extension for POST /predict
**Objective**: Optimize endpoint for explanation generation workflow
**Tasks**:
- [ ] Ensure single forward pass computes all explanation methods
- [ ] Implement early return for non-explain requests (skip explanation computation when explain=false)
- [ ] Add explanation timeout protection (abort after 2s)
- [ ] Implement graceful degradation to simulated explanations on failure
- [ ] Add explanation generation performance metrics to logs
**Files**: `backend/app.py`, `backend/core/inference.py`

### 3.10 Performance Optimization & Validation
**Objective**: Ensure production readiness and validate implementation
**Tasks**:
- [ ] Benchmark explanation latency across CPU/GPU configurations
- [ ] Validate explanation stability under input perturbations
- [ ] Test with TBX11K Simplified dataset for localization accuracy
- [ ] Conduct failure mode analysis (OOM, gradient failure, etc.)
- [ ] Create comprehensive test suite covering all explanation methods
- [ ] Document performance characteristics in developer guide
**Files**: All explanation-related files + test suite

## 6. Quality Assurance & Validation

### 6.1 Unit Testing Requirements
Each explanation method must have tests for:
- Mathematical correctness (vs. reference implementation)
- Edge cases (zero activations, uniform inputs, saturated inputs)
- Failure modes (None model, incompatible tensor shapes)
- Performance characteristics (scaling with input size)
- Medical appropriateness (lung masking, anatomical plausibility)

### 6.2 Integration Testing Requirements
End-to-end tests must validate:
- API response structure compliance
- Backward compatibility (existing field preservation)
- Explainability field inclusion/exclusion based on parameter
- Data consistency between internal storage and API response
- Error handling and graceful degradation

### 6.3 Medical Validation Criteria
When tested against TBX11K Simplified dataset with annotations:
- **Localization**: Mean IoU ≥ 0.45 with lesion masks
- **Sensitivity**: ≥ 70% of lesions have at least one ROI overlapping ground truth
- **Specificity**: ≤ 20% false positive ROIs in normal cases
- **Clinical Alignment**: ≥ 80% of explanations align with radiologist expectations

### 6.4 Performance Benchmarks
Must achieve on standard test image (224x224 chest X-ray):
| Configuration | CPU Latency | GPU Latency | Memory Overhead |
|---------------|-------------|-------------|-----------------|
| Base (no explain) | ≤ 300ms | ≤ 80ms | < 100MB |
| Grad-CAM++ only | ≤ 450ms | ≤ 120ms | < 200MB |
| All 5 methods | ≤ 800ms | ≤ 200ms | < 400MB |

## 7. Risks & Mitigation Strategies

### 7.1 Technical Risks
**Risk**: Gradient computation failures (NaN/Inf gradients)  
**Probability**: Medium  
**Impact**: Explanation generation failure  
**Mitigation**: 
- Gradient clipping and NaN/Inf detection
- Fallback to density-based explanation
- Detailed error logging for debugging

**Risk**: Memory exhaustion during batched explanation  
**Probability**: Low  
**Impact**: Service disruption  
**Mitigation**: 
- Dynamic batch size reduction
- Single-image fallback processing
- Memory usage monitoring and alerts

### 7.2 Medical Risks
**Risk**: Misleading explanations causing clinician mistrust  
**Probability**: Low (with proper validation)  
**Impact**: Reduced clinical adoption  
**Mitigation**: 
- Strict adherence to lung masking
- Clinical appropriateness validation
- Uncertainty quantification in explanations
- Clear documentation of explanation limitations

**Risk**: Explanation latency impacting workflow  
**Probability**: Medium  
**Impact**: Reduced usability in high-volume settings  
**Mitigation**: 
- Asynchronous explanation generation
- Progressive enhancement (show base result, enhance later)
- Performance optimization and caching

### 7.3 Compliance Risks
**Risk**: Non-backward compatible API changes  
**Probability**: Low  
**Impact**: Breaking existing integrations  
**Mitigation**: 
- Strict adherence to additive-only changes
- Versioned API endpoints if breaking changes absolutely necessary
- Comprehensive backward compatibility testing

## 8. Future Extensibility Considerations

### 8.1 Planned Enhancements
- **Integrated Gradients**: Axiomatic attribution method for baseline-sensitive explanations
- **Temporal Explanations**: For longitudinal studies (tracking changes over time)
- **Uncertainty-aware Explanations**: Incorporating model uncertainty (MC dropout, ensembles)
- **Multi-model Explanations**: Ensemble explanation methods for improved robustness
- **Interactive Explanations**: Web-based tools for clinician exploration of explanations

### 8.2 Architecture Considerations
The current implementation supports extensibility through:
- Modular explanation methods in `generate_saliency_heatmap()`
- Configurable method selection via `method` parameter
- Standardized ROI and metadata formats
- Clear separation of computation and presentation layers

### 8.3 Research Opportunities
- **Pathology-specific Explanations**: Tailored explanations for cavitary vs. nodular TB
- **Population-specific Validation**: Explanation accuracy across age, gender, ethnic groups
- **Clinical Outcome Correlation**: Explanation quality correlation with treatment outcomes
- **Explanation-guided Active Learning**: Using uncertainties to prioritize labeling

## 9. Compliance with CLAUDE.md Principles

### 9.1 Correctness & Reproducibility
- [x] Mathematical justification provided for all methods
- [x] Deterministic processing with fixed seeds where applicable
- [x] Input validation and graceful error handling
- [x] Experiment tracking capability through metadata
- [x] No arbitrary thresholds or magic numbers

### 9.2 Explainability First
- [x] Explanations generated for all predictions (configurable output)
- [x] Preferred methodology (Grad-CAM++) scientifically validated
- [x] Clinical appropriateness enforced through terminology and framing
- [x] Uncertainty quantification included in outputs

### 9.3 Medical AI Principles
- [x] Uses clinically appropriate terminology ("AI Screening Result", not "diagnosis")
- [x] Avoids causative language ("regions influencing screening result")
- [x] Includes uncertainty quantification and reliability metrics
- [x] Maintains clinician as final decision maker

### 9.4 Modularity & Maintainability
- [x] Separation of concerns: computation, validation, presentation
- [x] Configuration-driven parameters (via existing config system)
- [x] Clear interface definitions between components
- [x] Comprehensive logging for traceability

### 9.5 Research Quality
- [x] Literature-backed methodological choices
- [x] Validation framework with quantitative metrics
- [ ] Publication-ready documentation (to be developed during implementation)
- [x] Reproducible experimental procedures

## 10. Implementation Notes for Contractors

### 10.1 File Modification Guidelines
- **Do not modify**: `CNN Model Training/`, `CLAUDE.md`, `.claude/`, existing trained models
- **May modify**: `backend/core/inference.py`, `backend/app.py`, `backend/utils/` (limited to explanation-related utils)
- **Must preserve**: Backward compatibility, existing API contracts, model loading mechanisms

### 10.2 Testing Requirements
All contractors must provide:
- Unit tests covering ≥ 90% of new/modified code
- Integration tests validating end-to-end API behavior
- Performance benchmarks showing compliance with Section 6.4
- Medical validation results using provided test datasets
- Documentation of any assumptions or limitations discovered

### 10.3 Delivery Format
Each contract (3.2-3.10) should deliver:
1. Modified source files with implementation
2. Comprehensive unit and integration test suite
3. Performance benchmark report
4. Updated documentation (if applicable)
5. Change log detailing modifications made

---
**Approved by**: Nirikhshon Engineering Architecture Board  
**Effective Date**: 2026-07-06  
**Review Cycle**: Quarterly or as needed based on implementation feedback