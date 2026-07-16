# Nirikhshon Workstation: Heuristic Clinical Mapping Rules Spec

This document details the rule-based clinical findings and recommendations mapped from the Vision Model's Grad-CAM Region of Interest (ROI) activations and overall prediction status.

The clinical observation system is implemented in [observation_builder.py](file:///c:/Users/sdroy/OneDrive/Desktop/Documents/Final%20Year%20Project/backend/utils/observation_builder.py) and operates statelessly using deterministic rules to guarantee reproducibility and auditability.

---

## 🧭 Anatomical Localization Mapping

Each ROI returned by the XAI (consensus CAM) is parsed into **Laterality** (Right, Left, Bilateral) and **Lung Zone** based on string containment inside the location label:

| Location Substring | Zone Classification |
| ------------------ | ------------------- |
| `upper`, `apical`, `apex` | **Upper**           |
| `mid`, `middle`, `hilar`  | **Middle**          |
| `lower`, `basal`          | **Lower**           |
| `pleural`, `costophrenic` | **Pleural**         |

---

## 🔬 Rule Definitions

### 1. TB-Specific Saliency Rules (when prediction: `Tuberculosis` / `is_tb=True`)

These rules are triggered when the screening model classifies the image as suspicious for tuberculosis with high confidence.

| Zone | Finding Label | ICD-10 Code | Rule ID | Audit Source |
| ---- | ------------- | ----------- | ------- | ------------ |
| **Upper** | Focal apical consolidation | `A15.0` | `RULE-TB-UPPER` | `_TB_LABELS['Upper']` |
| **Middle** | Mid-zone parenchymal infiltrate | `A15.0` | `RULE-TB-MIDDLE` | `_TB_LABELS['Middle']` |
| **Lower** | Lower-zone consolidative opacity | `A15.0` | `RULE-TB-LOWER` | `_TB_LABELS['Lower']` |
| **Pleural** | Pleural thickening / effusion | `A15.6` | `RULE-TB-PLEURAL` | `_TB_LABELS['Pleural']` |

* **TB Follow-up Plan:**
  1. Sputum AFB smear microscopy x 3 (early-morning specimens)
  2. GeneXpert MTB/RIF or Truenat molecular assay
  3. Chest CT for cavitation and tree-in-bud assessment
  4. HIV co-infection testing and CD4 count
  5. Drug-susceptibility testing if AFB positive
  6. Contact tracing per NTEP / WHO guidelines
* **TB Differential Diagnoses:**
  - Typical pulmonary tuberculosis (*Mycobacterium tuberculosis*)
  - Atypical mycobacterial infection (*M. avium complex*, *M. kansasii*)
  - Fungal pneumonia (histoplasmosis, coccidioidomycosis)
  - Cavitating lung malignancy (squamous cell carcinoma)
  - Aspiration pneumonia with cavitation
  - Septic emboli with cavitary lesions

---

### 2. Non-TB / Normal Saliency Rules (when prediction: `Normal` / `is_tb=False`)

These rules are triggered when the screening model classifies the image as normal (not suspicious for active pulmonary tuberculosis).

| Zone | Finding Label | ICD-10 Code | Rule ID | Audit Source |
| ---- | ------------- | ----------- | ------- | ------------ |
| **Upper** | Mild apical opacity / vascular marking | `R91.1` | `RULE-NORMAL-UPPER` | `_NORMAL_LABELS['Upper']` |
| **Middle** | Mid-zone parenchymal texture | `R91.1` | `RULE-NORMAL-MIDDLE` | `_NORMAL_LABELS['Middle']` |
| **Lower** | Lower-zone vascular shadow | `R91.1` | `RULE-NORMAL-LOWER` | `_NORMAL_LABELS['Lower']` |
| **Pleural** | Costophrenic angle / pleural reflection | `R91.8` | `RULE-NORMAL-PLEURAL` | `_NORMAL_LABELS['Pleural']` |

* **Normal Follow-up Plan:**
  1. Clinical correlation with symptoms (cough, fever, weight loss)
  2. Repeat chest radiograph in 4-6 weeks if symptoms persist
  3. Sputum AFB smear only if TB risk factors present
  4. Consider chest CT only if clinical concern escalates
* **Normal Differential Diagnoses:**
  - Normal anatomical variant / vascular shadow
  - Mild atelectasis or subsegmental collapse
  - Early interstitial changes (non-specific)
  - Viral lower-respiratory-tract infection
  - Residual scarring from prior infection

---

## 📈 Score Indicators

- **Evidence Score:** A normalized value between `0` and `1`, calculated as:
  $$\text{Evidence Score} = \text{min}\left(1.0, \frac{\text{Activation}}{100} \times 0.7 + \frac{\text{Contribution}}{100} \times 0.3\right)$$
- **Clinical Significance:** 
  - **High:** Combined score $\ge 60.0$
  - **Moderate:** Combined score $\ge 30.0$
  - **Low:** Combined score $< 30.0$
  - *Combined score calculated as $\text{Activation} \times 0.6 + \text{Contribution} \times 0.4$*
- **Severity Classification:** Mapped from peak activation:
  - **Critical:** $\ge 90.0$
  - **Marked:** $\ge 70.0$
  - **Moderate:** $\ge 50.0$
  - **Mild:** $\ge 30.0$
  - **Background:** $< 30.0$
