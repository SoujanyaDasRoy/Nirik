# Purpose

Maintain comprehensive, accurate, and accessible documentation for Nirikhshon to ensure reproducibility, maintainability, and knowledge transfer.

# Responsibilities

- Create and update README, architecture diagrams, and contributor guides
- Document notebooks, datasets, models, APIs, and deployment procedures
- Ensure consistency between documentation and implementation
- Maintain Architecture Decision Records (ADRs) and changelogs
- Produce user guides, developer guides, and research documentation
- Review documentation for clarity, completeness, and correctness
- Standardize Markdown formatting, code comments, and docstrings
- Document security considerations, privacy practices, and configuration

## Primary Notebook
- Supports all notebooks (documents all notebooks, outputs, and workflows)

## Secondary Notebooks
- Notebook 1 (Dataset Preparation) - documents dataset discovery, validation, and preparation
- Notebook 2 (Segmentation) - documents U-Net training, evaluation, and mask export
- Notebook 3 (Classification) - documents DenseNet-121 training, evaluation, and model export
- Notebook 4 (Explainability & Reporting) - documents Grad-CAM, ROI localization, observation generation, and report generation

# When to Use

Update documentation when:
- Adding new features, modifying existing components, or fixing bugs
- Changing dataset versions, model architectures, or preprocessing pipelines
- Altering API contracts, authentication mechanisms, or deployment configurations
- Conducting experiments or publishing research findings
- Onboarding new contributors or transferring knowledge
- Preparing for releases, audits, or institutional reviews
- Receiving feedback on documentation clarity or completeness

# Documentation Philosophy

Reference CLAUDE.md for:
- Reproducibility: Document random seeds, dataset versions, and training parameters
- Explainability: Document Grad-CAM usage, validation methods, and limitations
- Research First: Prioritize methodological rigor over benchmark chasing
- Medical AI Principles: Use clinically appropriate terminology; avoid diagnostic language
- Configuration Before Hardcoding: Document configurable parameters and their sources
- Scientific Integrity: Never fabricate citations, results, or methodologies
- Implementation Philosophy: Document the full engineering lifecycle from literature review to validation

Ensure documentation answers: What, Why, How, When, Who, and Limitations.
Maintain single source of truth; avoid duplication.
Prioritize clarity and actionable guidance over completeness.
Document assumptions, known issues, and future work.

# Repository Documentation Architecture

```
/docs
  /architecture         # System diagrams, data flow, component relationships
  /api                  # REST API reference, endpoints, request/response schemas
  /datasets             # Dataset descriptions, provenance, preprocessing
  /deploy               # Deployment guides for HF Spaces, Vercel, Docker
  /notebooks            # Per-notebook documentation and workflow explanations
  /research             # Methodology, evaluation, novelty analysis, future work
  /security             # Threat model, privacy practices, incident response
  /user                 # Clinical workflow guides, interpretation help, FAQs
  /contributing         # Setup instructions, coding standards, PR process
```

Top-level documents:
- README.md: Project overview, vision, setup, and navigation
- CLAUDE.md: Engineering principles and rules (authoritative)
- Understanding.md: High-level project explanation (if present)
- CHANGELOG.md: Versioned release notes
- CONTRIBUTING.md: Contribution guidelines
- SECURITY.md: Security policy and reporting procedure

# README Standards

Include:
- Project name, vision, and one-sentence purpose
- Primary users and clinical objective
- System overview diagram (text or image)
- Setup instructions: prerequisites, installation, quick start
- Navigation guide: where to find notebooks, backend, frontend, docs
- Key features: image quality assessment, segmentation, classification, explainability, reporting
- Deployment targets: Hugging Face Spaces (backend), Vercel (frontend)
- Citation instructions if applicable
- License and contribution guidelines
- Badges: build status, license, etc. (optional)

# Notebook Documentation

Each notebook must have:
- Purpose: clear statement of responsibility per CLAUDE.md Notebook Contracts
- Inputs: required files, formats, and sources (with paths relative to repo root)
- Outputs: exported files, formats, and destinations
- Dependencies: Python packages, system requirements, GPU notes
- Expected Runtime: approximate time and hardware requirements
- Step-by-step explanation: what each section does and why
- Failure Conditions: common errors and how to resolve them
- Exported Files: exact filenames and locations for downstream notebooks
- Future Notebook Compatibility: note any breaking changes or version dependencies
- References: papers, tutorials, or external resources used

Use Markdown cells for explanations; code cells for executable steps only.
Avoid narrative in code cells; keep code focused on implementation.

# Code Documentation

- Functions: docstring with purpose, args, returns, raises, and examples
- Modules: module-level docstring describing purpose and usage
- Classes: class docstring with purpose, attributes, and methods
- Complex algorithms: inline comments explaining mathematical intuition
- Configuration blocks: comments explaining source and validation
- Public APIs: comprehensive docstrings; private functions: concise comments
- Type hints: use where practical (Python 3.8+)
- JSDoc: for JavaScript/TypeScript (frontend)
- Never: outdated comments, commented-out code, or TODOs without tracking
- Reference: follow PEP 257 (Python) and project-specific conventions

# API Documentation

Backend (Flask):
- OpenAPI/Swagger specification preferred; alternatively, Markdown reference
- Endpoints: HTTP method, path, description, parameters, request body, responses
- Authentication: required headers, token format, scopes
- Error Responses: standard error format, common status codes
- Examples: curl commands and sample JSON payloads/responses
- Versioning: include in URL (e.g., /api/v1/predict)
- Deprecation: clearly mark deprecated endpoints with removal timeline

Frontend (Next.js):
- API routes: document if used for proxying or utility functions
- Component libraries: document props, state, and events
- Custom hooks: document purpose, arguments, return values
- Utilities: document input/output types and side effects

# Architecture Documentation

- System Architecture: high-level diagram showing data flow and components
- Data Pipeline: detailed diagram of dataset engineering → training → inference
- Component Diagrams: frontend (pages, components, hooks) and backend (services, models)
- Sequence Diagrams: key workflows (upload → prediction → report generation)
- Deployment Diagram: show Hugging Face Spaces, Vercel, and external services
- Technology Stack: versions and justification for each choice
- Maintain: update when architectural changes occur; archive old diagrams

# Research Documentation

- Methodology: detailed description of experiments, controls, and variables
- Dataset Justification: why each dataset was selected for its purpose
- Model Justification: architecture choices, alternatives considered, trade-offs
- Evaluation Protocol: metrics, validation strategy, statistical tests
- Reproducibility: random seeds, hardware, software versions, preprocessing steps
- Limitations: known constraints, threats to validity, assumptions
- Future Work: planned improvements, open questions, scalability
- References: formatted bibliography of cited works
- Novelty Statement: clear articulation of project's contributions

# Dataset Documentation

Per dataset:
- Name, source, version, and access instructions
- Purpose in pipeline (per CLAUDE.md Canonical Dataset Architecture)
- Contents: number of images, labels, modalities, demographic info (if available)
- Preprocessing Steps: applied to raw data for training/inference
- Label Definitions: mapping of raw labels to TB/Normal (or other)
- Known Issues: corruption, duplicates, missing labels, biases
- Ethics Statement: IRB status, consent, privacy protections
- Citation: how to attribute the dataset
- Directory Structure: where raw/processed/metadata versions live
- Validation Scripts: if applicable, for checking integrity

# Model Documentation

Per model:
- Purpose: role in pipeline (segmentation, classification, etc.)
- Architecture: layers, hyperparameters, and schematic
- Training Details: dataset, epochs, batch size, optimizer, learning rate schedule
- Validation Metrics: accuracy, precision, recall, F1, ROC-AUC, etc. (per CLAUDE.md)
- Explainability Method: Grad-CAM target layer, validation approach
- File Format: .keras, .pth, etc., and loading instructions
- Versioning: hash or version number tied to training configuration
- Deployment Notes: preprocessing requirements, input shape, output format
- Limitations: known failure modes, input constraints
- References: papers or tutorials that inspired the architecture

# Deployment Documentation

Backend (HF Spaces):
- Hardware requirements: CPU/RAM/storage
- Environment variables: required and optional
- Build steps: Dockerfile or HF Spaces-specific instructions
- Model loading: where weights are stored, how they're loaded
- Scaling considerations: concurrency, request limits
- Monitoring: logs, metrics, health checks
- Rollback procedure: reverting to previous version
- Security: authentication, secrets management, CORS

Frontend (Vercel):
- Build and deploy settings: framework presets, environment variables
- Performance budgets: bundle size, LCP/CLS/FID targets
- CDN configuration: caching headers, image optimization
- Environment variables: backend URL, feature flags
- Preview deployments: access controls, expiration
- Custom domains: setup instructions
- Error monitoring: integration with Sentry or similar

General:
- Docker: base image, exposed volumes, port mappings
- Cloud Agnostic: notes for deploying to AWS/GCP/Azure
- Backup Strategy: database, model weights, uploads
- Disaster Recovery: RTO/RPO estimates
- Compliance: HIPAA/GDPR considerations for prototype

# User Documentation

- Clinical Workflow: step-by-step guide for using the workstation
- Image Preparation: supported formats, quality requirements, anonymization
- Interpretation Guide: 
  * How to read AI confidence score
  * How to validate Grad-CAM heatmap
  * 
  * How to localize findings to anatomical zones
  * When to recommend clinical correlation
- Report Generation: creating, saving, and exporting PDF reports
- Troubleshooting: common issues (upload failures, slow inference, display problems)
- FAQ: answers to anticipated user questions
- Feedback Mechanism: how to report issues or suggest improvements
- Disclaimer: prominent notice that AI assists but does not replace clinician

# Changelog Management

- Use Keep a Changelog format (https://keepachangelog.com/)
- Categories: Added, Changed, Deprecated, Removed, Fixed, Security
- Each version: 
  * Version number (SemVer)
  * Release date (YYYY-MM-DD)
  * Bullet points per category
- Unreleased section for upcoming changes
- Link to GitHub tag or release
- Avoid vague descriptions; be specific about what changed
- Deprecations: include migration guidance and removal timeline

# Versioning

- Follow Semantic Versioning (MAJOR.MINOR.PATCH)
- MAJOR: breaking changes to API or workflow
- MINOR: new features, backward compatible
- PATCH: bug fixes, backward compatible
- Tag releases in Git; link from changelog
- Document version in:
  * README (current stable version)
  * Model metadata files
  * API responses (version header)
  * Docker image tags
  * Notebook output filenames

# Documentation Reviews

- Treat documentation changes like code changes: review for correctness and clarity
- Checklist for reviews:
  * Accuracy: matches implementation
  * Completeness: covers all necessary details
  * Clarity: understandable to target audience
  * Consistency: follows style and terminology
  * Reproducibility: enables replication of results
  * Usability: actionable guidance, not just theory
  * Links: working and pointing to correct resources
  * Images: descriptive alt text, scale, and labels
- Automate: link checking, spell checking, linting (markdownlint)
- Schedule: major documentation review before releases
- Feedback: incorporate user reports of confusion or missing info

# Engineering Decision Records (ADRs)

- Location: /docs/architecture/adrs/ or similar
- Format: Markdown with:
  * Status: Proposed, Accepted, Superseded
  * Context: problem and constraints
  * Decision: what we will do
  * Consequences: positive, negative, and neutral impacts
  * Alternatives: other options considered with trade-offs
  * Implementation Notes: if applicable
- Trigger: significant architectural changes (data flow, API contracts, key components)
- Review: involve relevant stakeholders (research, engineering, security)
- Link: from affected documentation and code comments
- Archive: superseded ADRs remain for history

# Interaction with Other Skills

Consumes:
- Research Engineer: to understand experimental methodology and novelty claims
- Backend Engineer: to document API contracts, configuration, and deployment
- Frontend Engineer: to document component usage, state management, and UI workflows
- Security Engineer: to document authentication, authorization, and privacy controls
- MLOps Engineer: to document model tracking, versioning, and pipeline automation

Produces:
- Repository-wide documentation accessible to all stakeholders
- Clear onboarding materials for new contributors
- Reproducibility packages for experiments
- User guides for clinical end-users
- Maintenance guides for long-term sustainability

Collaborates with:
- Documentation Engineer (self): ensures consistency across all documentation
- All other skills: to gather information and validate technical accuracy
- Medical AI Reviewer: to verify clinical terminology and safety considerations
- Dataset Engineer: to validate dataset descriptions and preprocessing steps

# Common Mistakes

- Updating implementation without updating documentation (or vice versa)
- Using inconsistent terminology across documents (e.g., "TB classification" vs "pneumonia detection")
- Omitting setup instructions or assuming prior knowledge
- Documenting speculative features as if already implemented
- Leaving public APIs or complex functions undocumented
- Duplicating information in multiple places leading to inconsistencies
- Using passive voice or vague language ("it is recommended" instead of "do this")
- Forgetting to update version numbers in documentation after releases
- Not providing examples for complex configurations or code usage
- Ignoring accessibility in documentation (e.g., low-contrast diagrams, missing alt text)
- Storing documentation in multiple formats without a single source of truth
- Using images without captions or explanations of what they show
- Documenting internal implementation details irrelevant to users
- Failing to distinguish between current state and future plans
- Not defining audience for each document (user vs developer vs researcher)
- Overlooking to document known issues or limitations
- Using proprietary formats that require special software to read
- Not reviewing documentation for medical terminology compliance with CLAUDE.md

# Never Do

- Never duplicate information already documented elsewhere in the repository
- Never allow documentation to drift from the implementation (update in same PR)
- Never document features that don't exist or are only in planning stages
- Never omit assumptions, limitations, or known issues with the approach
- Never leave any public API endpoint, function, or component undocumented
- Never use documentation to work around poor code structure (fix the code instead)
- Never compromise on clarity for brevity; prefer clear, complete explanations
- Never use inconsistent naming for the same concept across documents
- Never include confidential information (patient data, secrets) in public documentation
- Never skip documenting error conditions and how to handle them
- Never assume the reader has access to undocumented tribal knowledge
- Never use documentation as a substitute for code comments where appropriate
- Never publish documentation with broken links, missing images, or syntax errors
- Never deviate from established documentation structure without team consensus
- Never document patient-identifiable information in any public-facing document
- Never allow documentation to become outdated; schedule regular reviews
- Never use documentation to justify non-reproducible experimental practices
- Never omit citing sources for third-party methods, datasets, or models
- Never use documentation to hide technical debt; document the debt and plan to fix it
- Never ignore feedback from users who find documentation confusing or incomplete