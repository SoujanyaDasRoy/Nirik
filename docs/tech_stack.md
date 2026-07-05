# Tech Stack — AI-Based Tuberculosis Detection Web Application
> All versions verified as latest stable as of June 2026.

A detailed breakdown of every technology used in both the Frontend and Backend of this project, along with the reason each tool was chosen.

---

## BACKEND

The backend is the **AI brain** of the application. It handles all image processing, model inference, and API responses.

### 1. Python `3.13.x` *(Recommended Stable)*
- **What it is:** The core programming language for the entire backend.
- **Why `3.13` and not `3.14`?** While 3.14.5 is the latest release, Python 3.13 is the current **LTS-recommended** version with the widest compatibility across PyTorch, Flask, and all ML libraries. PyTorch 2.11 is fully tested on 3.13 but has limited testing on 3.14.
- **Why we chose it:** Python is the undisputed standard language for machine learning. Every major AI library (PyTorch, NumPy, scikit-learn) is built for Python first.

### 2. Flask `3.1.3`
- **What it is:** A lightweight Python web framework used to build the REST API.
- **Why we chose it:** Flask is minimal and lets us focus on the AI logic. It is fast to set up, easy to understand for academic projects, and integrates perfectly with PyTorch without compatibility issues.
- **Key Role:** Exposes a `/predict` POST endpoint that the Next.js frontend calls when a user uploads an X-ray.

### 3. Flask-CORS `6.0.2`
- **What it is:** A Flask extension that adds Cross-Origin Resource Sharing (CORS) headers to API responses.
- **Why we chose it:** Because Next.js (port 3000) and Flask (port 5000) run on different origins, browsers block requests by default. Flask-CORS whitelists the Next.js origin to allow communication between the two servers.
- **Note:** Version 6.0.3 was released but was subsequently yanked due to a compatibility bug. `6.0.2` is the safe, stable choice.

### 4. PyTorch `2.11.0`
- **What it is:** The deep learning framework used to build and run our ResNet-50 (Teacher) and DenseNet-121 (Student) models.
- **Why we chose it:** PyTorch is the industry and research standard for medical imaging AI. It provides dynamic computation graphs which makes debugging neural networks far easier than TensorFlow.

### 5. TorchVision `0.27.0`
- **What it is:** A PyTorch companion library for computer vision tasks.
- **Why we chose it:** Provides the pre-trained ImageNet weights for ResNet-50 and DenseNet-121 (Transfer Learning), and the `transforms` pipeline used to resize, crop, normalize, and augment X-ray images before feeding them into the model.
- **Compatibility:** TorchVision `0.27.0` is the official matching release for PyTorch `2.11.0`. They must always be installed together.

### 6. TorchScript (`.pt` file)
- **What it is:** A serialization format built into PyTorch to save and reload models.
- **Why we chose it:** Unlike a standard `.pth` state dictionary, TorchScript compiles the entire model into a static graph optimized for inference. This makes the model load faster and run faster at deployment time.
- **Our file:** `student_indian_tuned_torchscript.pt`

### 7. Pillow `12.2.0`
- **What it is:** Python's standard image processing library.
- **Why we chose it:** Flask receives an uploaded image file from the browser. Pillow handles opening it, converting it to RGB, and resizing it to 224×224 before passing it to the model.

### 8. pydicom `2.4.5`
- **What it is:** A Python library for reading DICOM files (the format used by real hospital X-ray machines).
- **Why we chose it:** The NIRT India dataset is entirely in DICOM format. pydicom extracts both pixel data and critical metadata tags (`WindowCenter`, `WindowWidth`) needed for medical-grade contrast windowing.

### 9. NumPy `2.4.6`
- **What it is:** The fundamental library for numerical computing in Python.
- **Why we chose it:** Used for array operations during image preprocessing and for calculating optimal clinical thresholds from the Precision-Recall curve.

### 10. scikit-learn `1.9.0`
- **What it is:** The standard Python machine learning library for classical algorithms and evaluation metrics.
- **Why we chose it:** Used for `GroupShuffleSplit` (patient-level data splitting to prevent leakage), and for computing `roc_auc_score`, `confusion_matrix`, `classification_report`, and `precision_recall_curve`.

---

## FRONTEND

The frontend is the **visual interface** that doctors and healthcare workers interact with.

### 1. Next.js `16.2.7` (App Router)
- **What it is:** A React-based framework for building web applications, created by Vercel.
- **Why we chose it:** Next.js 16 ships with **Turbopack as the default bundler**, making it significantly faster to build and hot-reload than previous versions. The App Router architecture provides a clean, modular component structure. It is the easiest framework to deploy for free on Vercel.

### 2. React `19.2.7`
- **What it is:** The underlying JavaScript UI library that Next.js 16 is built on.
- **Why we chose it:** React 19 introduced the new **React Compiler**, which automatically optimises re-renders without needing `useMemo` or `useCallback` manually. This makes our animated result cards and drag-and-drop zones buttery smooth.

### 3. Vanilla CSS (with CSS Custom Properties)
- **What it is:** Standard, plain CSS used for all styling.
- **Why we chose it:** We use CSS Custom Properties (CSS variables) to define our entire design system (colors, gradients, spacing, fonts) in one place. We deliberately avoid Tailwind CSS to have full, precise control over every animation and glassmorphism effect.
- **Key Design Elements:**
  - `backdrop-filter: blur()` — Glassmorphism panels
  - `@keyframes` — Smooth loading spinners and result animations
  - CSS Grid & Flexbox — Responsive layouts
  - `transition` & `transform` — Micro-animations on hover

### 4. Google Fonts (Inter)
- **What it is:** A modern, clean, highly legible sans-serif typeface.
- **Why we chose it:** Inter is specifically designed for computer screens and is the standard font used by major tech companies (Apple, GitHub, Linear). Using it makes the UI feel professional and trustworthy — critical for a medical application.

### 5. Fetch API (Vanilla JavaScript)
- **What it is:** The browser's built-in function for making HTTP requests.
- **Why we chose it:** When a user uploads an X-ray, Next.js packages it as `FormData` and sends it to the Flask `/predict` endpoint using `fetch()`. This is a dependency-free solution that avoids the overhead of Axios or other HTTP libraries.

---

## DEPLOYMENT STACK

### Frontend → Vercel
- **Why:** Created by the Next.js team. Connecting a GitHub repository and deploying takes under 2 minutes. Every push to `main` automatically triggers a new live deployment. Completely free for academic projects.

### Backend → Hugging Face Spaces
- **Why:** The industry-leading free platform for hosting AI/ML models. Natively understands Python and Flask. Provides sufficient free compute to run DenseNet-121 inference without needing a paid tier.

---

## Complete Version Reference Table

### Backend

| Package | Version | Purpose |
|---|---|---|
| Python | `3.13.x` | Core backend language |
| Flask | `3.1.3` | REST API server |
| Flask-CORS | `6.0.2` | Allow cross-origin requests from Next.js |
| PyTorch | `2.11.0` | Model training & inference |
| TorchVision | `0.27.0` | Transforms & pre-trained weights |
| Pillow | `12.2.0` | Open & preprocess uploaded images |
| pydicom | `2.4.5` | Read Indian hospital X-ray DICOM files |
| NumPy | `2.4.6` | Array math & threshold calculation |
| scikit-learn | `1.9.0` | AUC, F1, Confusion Matrix, GroupSplit |

### Frontend

| Package | Version | Purpose |
|---|---|---|
| Next.js | `16.2.7` | React web framework (Turbopack) |
| React | `19.2.7` | Component-based UI (React Compiler) |
| Vanilla CSS | — | Glassmorphism, animations, layout |
| Google Fonts (Inter) | — | Professional medical typography |
| Fetch API | — | Sends X-ray images to Flask backend |

### Deployment

| Layer | Platform | Cost |
|---|---|---|
| Frontend | Vercel | Free |
| Backend + Model | Hugging Face Spaces | Free |
