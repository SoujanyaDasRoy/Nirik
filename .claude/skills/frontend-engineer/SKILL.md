# Purpose

Build clinical frontend workstation for Nirikhshon TB screening system.

# Responsibilities

- React components and Next.js pages
- TypeScript interfaces and utilities
- Tailwind CSS styling and responsive design
- State management (context, hooks, or state library)
- API integration with Flask backend
- Clinical dashboard and workflow
- Chest X-ray and DICOM viewer components
- Heatmap visualization overlay
- Report generation interface
- Authentication and authorization UI
- Accessibility compliance (WCAG 2.1 AA)
- Performance optimization and lazy loading
- Error boundaries and loading states
- Vercel deployment configuration

## Primary Notebook
- Notebook 4 (Explainability & Reporting) - displays results, heatmaps, and reports

## Secondary Notebooks
- Notebook 1 (Dataset Preparation) - understands data structure for UI components
- Notebook 2 (Segmentation) - visualizes lung masks and segmentation results
- Notebook 3 (Classification) - displays classification results and confidence scores

# When to Use

Implement or modify frontend components for:
- New clinical workflow features
- UI enhancements for image viewing
- Report visualization improvements
- Accessibility fixes
- Performance optimizations
- Backend API contract changes
- Deployment configuration updates

# Architecture

Follows Next.js app router structure:
```
/app
  /(clinical)          # Protected routes
    dashboard/
    viewer/
    report/
  /api                 # Route handlers (proxy to backend)
  /components          # Reusable UI components
  /lib                 # Utilities, hooks, constants
  /types               # TypeScript definitions
  /styles              # Global CSS and Tailwind config
```

Components organized by feature:
- Layout: Header, Sidebar, Footer
- Viewer: ImageDisplay, DicomViewer, HeatmapOverlay
- Report: ObservationForm, RecommendationSection, PDFGenerator
- Forms: UploadForm, AuthForms, SettingsPanel

# Engineering Principles

Reference CLAUDE.md for:
- Medical terminology (avoid diagnostic language)
- Explainability requirements (Grad-CAM mandatory)
- Clinical safety (clinician final decision)
- Reproducibility (consistent UI states)
- Modularity (single responsibility components)
- Accessibility (keyboard navigable, ARIA labels)
- Performance (code splitting, image optimization)
- Maintainability (consistent naming, documented props)

State updates must trigger re-renders for explainability updates.
All image processing occurs in viewer components, never in layout.

# Component Architecture

Atomic design pattern:
- Atoms: Button, Input, Label, Icon
- Molecules: SearchBar, ImageToolbar, ReportCard
- Organisms: DicomViewer, ReportGenerator, Dashboard
- Templates: ClinicalLayout, ViewerPage
- Pages: /dashboard, /viewer/[id], /report/[id]

Props drilling avoided via context or state management.
Event handlers debounced for performance.
Image components use next/image for optimization.

# State Management

Hierarchical state:
- Local: useState for form toggles, tool selections
- Global: Context or Zustand for:
  - Current study/patient data
  - Segmentation masks
  - Classification results
  - Grad-CAM heatmaps
  - Report observations
  - User preferences

Immutable updates via immer or structured cloning.
Loading and error states managed per feature slice.

# API Integration

Backend endpoint: https://nirikshon-hf-space.hf.space (orvercel.hf.space
Service layer in /lib/api with:
- Typed request/response functions
- Automatic retry with exponential backoff
- Request cancellation on unmount
- Error normalization (network vs backend errors)
- Auth token handling (if implemented)

Endpoints:
- POST /predict: Image → {probability, heatmap, segmentation}
- GET /studies/:id: Metadata retrieval
- POST /reports: Save generated report
- GET /health: Service availability

# Clinical User Interface

Workflow flow:
1. Study list (dashboard) → 2. Image viewer → 3. Report generation
Viewer includes:
- Original X-ray with toggleable lung overlay
- Grad-CAM heatmap adjustable opacity
- Measurement tools (distance, angle)
- Window/level presets (lung, bone, soft tissue)
- Zoom/pan with mouse/touch
- Slice navigation for multi-frame

Report sections:
- Patient/study info auto-filled
- AI confidence score (with uncertainty indicator)
- Structured observations (checkboxes + free text)
- Anatomical localization (lung zone selector)
- Recommendation dropdown (routine/followup/urgent)
- Generate PDF button

# DICOM Viewer

Cornerstone.js integration:
- Lazy-loaded to avoid bundle bloat
- Viewport synchronized with heatmap canvas
- Tool activation via keyboard shortcuts
- Measurement persistence per session
- Supported modalities: CXR, DX, MG
- Fallback to JPEG/PNG for non-DICOM

Heatmap synchronization:
- Same dimensions as displayed image
- Real-time opacity slider (0-100%)
- Colormap options (jet, hot, viridis)
- Export as overlay PNG

# Report Interface

Structured observation templates:
- Lung zones (6 regions) with severity scale
- Finding types: opacity, cavity, nodule, effusion
- Modifier checkboxes: new vs old, stable vs changing
- Free-text observations field
- Auto-generated impression from selections

PDF generation:
- Uses jsPDF with html2canvas fallback
- Embeds original image, segmentation, heatmap
- Includes disclaimer: "Screening assist only - clinician review required"
- Formats: A4, Letter, portrait/landscape

# Accessibility

WCAG 2.1 AA compliance:
- Semantic HTML landmarks (main, nav, section)
- ARIA labels for all interactive components
- Keyboard navigation tab order
- Focus trap in modals
- Color contrast ratios ≥4.5:1
- Text scaling support (rem units)
- Skip to content link
- Image alt text with medical description
- Screen reader friendly dynamic regions

# Performance

Optimizations:
- Code splitting at route level
- Image optimization via next/image
- Heatmap computation web worker offload
- Virtualized lists for large study sets
- Prefetching adjacent study data
- Bundle analyzer reports in CI
- LCI metrics targeting:
  - LCP < 2.5s
  - FID < 100ms
  - CLS < 0.1

# Deployment

Vercel configuration:
- vercel.json with:
  - Rewrites for API proxy to backend
  - Headers for security (CSP, HSTS)
  - Environment variables for backend URL
- Build cache for node_modules
- Preview deployments on branch pushes
- Production deployment on main branch
- Environment variables:
  - NEXT_PUBLIC_BACKEND_URL
  - NEXT_PUBLIC_GA_ID (analytics)

# Interaction with Other Skills

Consumes:
- Backend Engineer: API contract specifications
- Explainability Engineer: Heatmap format and validation requirements
- Documentation Engineer: UI props documentation and usage examples

Produces:**
- Clinical frontend interface
- Visualization layer for AI outputs
- User-facing report generation

Collaborates with:
- Security Engineer: Auth implementation and audit logging
- DevOps Engineer: Deployment pipeline and monitoring

# Common Mistakes

- Hardcoding image dimensions instead of using container-relative sizing
- Modifying backend API responses in frontend (should use adapter pattern)
- Storing sensitive data in localStorage (use sessionStorage or HTTP-only cookies)
- Ignoring error boundaries leading to blank screens
- Using any type instead of defining proper TypeScript interfaces
- Not cleaning up event listeners in useEffect
- Blocking main thread with heavy image processing
- Forgetting to cancel API requests on component unmount
- Overlooking touch events for mobile accessibility
- Using div instead of button for interactive elements
- Missing lang attribute on html element
- Insufficient color contrast in heatmap overlays
- Not providing text alternatives for non-text content
- Using placeholder text as label substitute
- Not testing with screen reader software
- Ignoring prefers-reduced-motion media query
- Not validating image file types before upload
- Storing PHI in frontend logs or console

# Never Do

- Implement AI model logic (training, inference, preprocessing)
- Modify backbone CNN architectures
- Change Grad-CAM target layer selection
- Alter medical terminology to imply diagnosis
- Store patient data client-side beyond session
- Bypass backend authentication checks
- Use !important in CSS (use specificity instead)
- Block UI thread with synchronous operations
- Hardcode backend URLs (use env vars)
- Ignore TypeScript strict mode warnings
- Deploy without accessibility testing
- Commit secrets to repository
- Use client-side only validation for forms
- Modify dataset paths or names
- Create new directories outside established structure
- Override established Tailwind config without review
- Skip PropTypes or TypeScript prop validation
- Use console.log in production code
- Leave commented-out code in commits
- Modify CLAUDE.md principles without team consensus