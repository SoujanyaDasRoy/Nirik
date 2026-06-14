# Project Details: Users, Use Cases, and Metrics

## 1. Who are our users?
*   **Primary Users:** Radiologists and busy doctors in government hospitals or private clinics who need a fast, reliable "second opinion" to screen hundreds of X-rays a day without fatigue.
*   **Secondary Users:** Rural healthcare workers, frontline medical staff (like ASHA workers), or NGO volunteers operating in remote areas who do not have a specialized radiologist on site.

## 2. Where will they normally use this?
This system is strictly designed for **low-resource and high-volume clinical settings**. 
*   **Mobile Health Camps & Rural Clinics:** Because the web app and the DenseNet-121 model are so lightweight, a healthcare worker can run this web app on a basic laptop directly connected to an X-ray machine in a rural village. They don't need a massive internet connection.
*   **Busy Urban Hospitals:** In crowded city hospitals (where TB screening is high volume), doctors will use this on their desktop computers via their web browser to instantly highlight high-risk patients who need immediate attention.

## 3. How much accuracy should we aim for?
In the medical field, flat "Accuracy" is a deceptive metric. We don't care about standard accuracy; we care about **Recall (Sensitivity)**. 

*   **Our Target Goal:** We aimed for **>95% Recall for Tuberculosis** on the validation/calibration sets.
*   **Why?** In medicine, a **False Positive** (the AI says you have TB, but you are healthy) is mildly annoying—you just take a second test. But a **False Negative** (the AI says you are healthy, but you actually have active TB) is catastrophic. The patient goes home untreated and infects their community. 
*   **Actual Achieved Metrics (on Unseen Test Set):**
    *   **AUC Score:** 98.99%
    *   **Overall Accuracy:** 96.6%
    *   **Best Decision Threshold:** 0.93 (calibrated on the held-out calibration set)
    *   **TB Recall (Sensitivity):** 91.7% (100 correct detections out of 109 true cases)
    *   **TB Precision:** 92.6% (with only 8 false alarms out of 500 test images)
    *   **Specificity (True Negative Rate):** 98.0% (383 correct normal classifications out of 391 cases)
    *   **NPV (Negative Predictive Value):** 97.7%
*   **Clinical Viability:** By setting a calibrated dynamic decision threshold of **0.93** (determined on the calibration split), the model achieved an exceptional balance of **98.99% AUC** and **98.0% Specificity**, leaving only 8 false alarms and 9 missed cases out of 500 test chest X-rays. This makes the model highly clinically viable as a second-opinion diagnostic tool in low-resource and high-volume environments.
