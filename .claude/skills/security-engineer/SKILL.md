# Purpose

Secure the Nirikhshon application across frontend, backend, and deployment layers.

# Responsibilities

- Authentication and session management
- Authorization and access controls
- API security (Flask)
- File upload validation (especially medical images/DICOM)
- Input validation and output sanitization
- Secret management and environment variables
- Database security (SQLite)
- Logging, audit trails, and privacy protection
- Dependency vulnerability management
- Deployment security (Hugging Face Spaces, Vercel)
- Security reviews and incident response guidance
- Ensuring compliance with medical data privacy principles

## Primary Notebook
- Supports all notebooks (implements security controls throughout pipeline)

## Secondary Notebooks
- Notebook 1 (Dataset Preparation) - secures data handling and validation
- Notebook 2 (Segmentation) - protects model integrity and validates inputs
- Notebook 3 (Classification) - secures classification pipeline and outputs
- Notebook 4 (Explainability & Reporting) - ensures secure report generation and access

# When to Use

Implement or review security controls when:
- Adding new API endpoints
- Modifying file upload handling
- Changing authentication mechanisms
- Updating dependencies
- Preparing for deployment
- Conducting security assessments
- Handling patient data or medical images
- Integrating with external services

# Security Philosophy

Reference CLAUDE.md for:
- Clinical safety: Never imply AI diagnosis; clinician retains final decision
- Explainability: Security must not interfere with Grad-CAM heatmap accessibility
- Reproducibility: Secure configurations must be versioned and documented
- Minimal attack surface: Disable unused features, enforce least privilege
- Defense in depth: Layer controls (network, application, data)
- Fail securely: Default to deny, handle errors without leaking information
- Privacy by design: Minimize patient data exposure, protect PHI

Consume guidance from Backend Engineer (API contracts) and Frontend Engineer (UI security considerations).

# Repository Security Architecture

Trust boundaries:
- Frontend (Vercel) → Internet
- API Gateway (Vercel rewrites) → Backend (Flask on HF Spaces)
- Backend → SQLite database, model files, upload storage
- External: Hugging Face Spaces platform, Vercel platform

Apply controls at each boundary:
- Network: Platform-provided DDoS, TLS
- Application: AuthN/Z, input validation, rate limiting
- Data: Encryption at rest (SQLite), access controls, audit logs

# Authentication

- Use platform-managed auth where possible (HF Spaces private gating, Vercel auth)
- If custom: Implement via Flask-Login or podobne with secure password hashing (bcrypt)
- Session cookies: Secure, HttpOnly, SameSite=Strict
- Multi-factor authentication recommended for admin access
- Token-based auth for API: Access tokens short-lived, refresh tokens rotated
- Never store passwords or tokens in frontend code or localStorage
- OAuth2/OpenID Connect preferred for federated login (institutional accounts)
- Public endpoints (health check) remain unauthenticated; all others require auth

# Authorization

- Role-based access control (RBAC): roles: viewer, clinician, admin
- Principle of least privilege: Users only access own studies unless explicit sharing
- Resource-level permissions: Check ownership on every database query
- Admin functions restricted to verified personnel
- Authorization failures logged and monitored
- Regular review of role assignments

# API Security

Flask-specific:
- Validate Content-Type for POST/PUT
- Enforce JSON schema validation for request bodies (e.g., using pydantic or marshmallow)
- Use Flask-Talisman for:
  - Secure HTTP headers (CSP, HSTS, Referrer-Policy, X-Frame-Options)
  - Force HTTPS in production
  - Session cookie security
- CORS: Restrict to frontend domain(s) only; avoid wildcards
- Rate limiting: Use Flask-Limiter (e.g., 100 requests/minute per IP, stricter on auth endpoints)
- API versioning: Include version in URL (/api/v1/predict)
- Error handling: Return generic messages in production; log details internally
- No debug mode or detailed error pages in production
- Use await/async patterns where appropriate to avoid blocking workers

# File Upload Security

Chest X-ray/DICOM uploads:
- Whitelist MIME types: application/dicom, image/jpeg, image/png
- Validate file signatures (magic bytes) in addition to extension
- Rename uploaded files to UUIDs; store original filename in metadata (separate)
- Limit file size: Maximum 50 MB per study (configurable via env)
- Scan for malware via ClamAV or similar if resources allow (optional for prototype)
- Store uploads outside web root; serve via secure backend endpoint only
- Delete temporary files immediately after processing
- Validate DICOM integrity using pydicom; reject corrupted files
- Strip or anonymize PHI from DICOM metadata before storage if not required for processing
- Log upload attempts (success/failure) with user ID, filename, size, IP
- Never execute or interpret uploaded files as code

# Input Validation

- Validate all inputs: query parameters, headers, body, cookies
- Use allowlists for known good values (e.g., enum parameters)
- Reject requests with unexpected fields or types
- Sanitize free-text fields (observations) for XSS: escape HTML or use safe rendering
- Limit lengths: Prevent DoS via overly long strings
- Validate numeric ranges (e.g., probability between 0 and 1)
- For DICOM tags: Validate presence of required tags; ignore private tags unless needed
- Use prepared statements or ORM for SQLite to prevent SQL injection
- Never concatenate user input into SQL or shell commands

# Output Validation

- Sanitize AI-derived outputs before returning to frontend:
  - Probability: ensure float 0-1
  - Heatmap: validate dimensions, normalize values
  - Segmentation mask: validate binary values, correct shape
- JSON responses: set Content-Type: application/json; prevent JSON hijacking (prefix while loop for GET if needed)
- Error responses: Do not include stack traces or system details in production
- Set security headers via Flask-Talisman:
  - Content-Security-Policy: default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'
  - X-Content-Type-Options: nosniff
  - X-Frame-Options: DENY
  - Referrer-Policy: strict-origin-when-cross-origin
  - Permissions-Policy: geolocation=(), microphone=()
- Ensure UTF-8 encoding to prevent encoding-based attacks

# Secret Management

- Never hardcode secrets in source code
- Use environment variables for:
  - Database connection strings (if external)
  - API keys for external services
  - Encryption keys
  - Secret keys for Flask sessions
- In development: Use .env.example; never commit .env
- In production (HF Spaces/ Vercel): Set secrets via platform UI
- Rotate secrets periodically
- Use separate secrets per environment (dev, staging, prod)
- Audit repository periodically for accidental commits (git-secrets, pre-commit hooks)
- Encrypt secrets at rest if stored (platforms typically handle this)

# Environment Variables

- BACKEND_URL: Frontend points to backend (relative in prod via rewrites)
- SECRET_KEY: Flask session secret
- JWT_SECRET_KEY: For token-based auth if used
- ENCRYPTION_KEY: For encrypting sensitive DB fields
- MAX_UPLOAD_SIZE: File size limit
- ALLOWED_ORIGINS: CORS origins
- RATE_LIMIT_DEFAULT: Requests per minute
- LOG_LEVEL: debug/info/warn/error
- NODE_ENV / FLASK_ENV: Set to production in prod
- Disable verbose logging in production
- Do not include patient identifiers in env vars

# Database Security

SQLite:
- File permissions: Restrict to backend process user (read/write only)
- Consider encrypted SQLite (SQLCipher) if PHI stored directly (prefer anonymization)
- Parameterized queries only; no string building
- Regular backups stored securely with access controls
- Validate data before storage: type, length, format
- Audit table: Log reads/writes to PHI if required by policy
- Connection pooling: Use sensible limits to prevent exhaustion
- Disable features like load_extension if not needed

# Logging & Audit Trails

- Log authentication attempts (success/failure)
- Log file uploads: user, filename, size, IP, timestamp
- Log API access: endpoint, method, status, user ID, response time
- Log errors: Itype and message only; no stack traces or PHI
- Ensure logs do not contain:
  - Patient names, IDs, DOB
  - Full file paths
  - Secret keys or tokens
  - Raw DICOM headers with PHI
- Use structured logging (JSON) for easier parsing
- Retain logs per institutional policy (minimally 6 months for prototype)
-Monitor logs for anomalies: repeated failed logins, unusual upload patterns
- Never log to console in production; use file or external service
- Log security events separately for alerting

# Privacy

- Apply minimum necessary principle: Only collect data essential for screening
- Anonymize/pseudonymize data where possible for research
- Store encountered PHI separately from AI results with strict access controls
- Provide mechanism for data deletion upon request (right to be forgotten)
- Ensure frontend does not cache or store medical images beyond session
- Use HTTPS exclusively; enforce HSTS
- Display clear privacy notice in UI
- Align with institutional IRB/data use agreements for research data
- Never sell or share patient data with third parties without explicit consent

# Error Handling

- Fail securely: Default to deny access on authentication/authorization error
- Return generic error messages to client (e.g., "Internal server error")
- Log detailed errors server-side for debugging
- Ensure error pages do not leak stack traces or system info
- Handle file upload errors gracefully: reject invalid files, inform user
- Validate that error responses themselves do not introduce XSS
- Use try/catch blocks; avoid broad exception handlers that hide bugs
- Return appropriate HTTP status codes (4xx for client errors, 5xx for server)

# Dependency Security

- Use lockfiles: requirements.txt (backend), package-lock.json (frontend)
- Regularly scan for vulnerabilities: Dependabot, npm audit, pip-audit
- Update dependencies promptly, especially security patches
- Avoid dependencies with known unpatched vulnerabilities
- Review new dependencies before adding: license, maintenance, security history
- Consider using SBOMs for provenance
- Remove unused dependencies to reduce attack surface
- For frontend: Audit bundle for unnecessary polyfills or large libs
- Prefer well-maintained, widely-used libraries (e.g., Cornerstone.js, next/image)

# Deployment Security

Backend (Hugging Face Spaces):
- Keep Space private or restricted to authorized users via HF authentication
- Disable community features if not needed (comments, likes)
- Set appropriate hardware (CPU/RAM) to avoid resource exhaustion
- Monitor Space logs for anomalies
- Ensure models stored in repo are trusted; verify integrity
- Use HF secrets for tokens, keys
- Enable HTTPS only (HF provides)
- Consider using HF Organizations for team management

Frontend (Vercel):
- Deploy Preview deployments only to internal/team domains if possible
- Enable password protection for preview branches if needed
- Set security headers via vercel.json: Content-Security-Policy, etc.
- Use environment variables for secrets; never expose in client code
- Disable unnecessary features: experimental flags, analytics if not needed
- Monitor bandwidth and function invocations for DoS signs
- Use Vercel's IP filtering or allowlist if available
- Ensure build step does not include secrets in output
- Leverage Vercel's Identity for SSO if applicable

General:
- Infrastructure as Code: Use platform-provided settings; avoid manual changes
- Conduct penetration testing before major releases
- Have rollback plan for faulty deployments
- Document deployment process and security checks
- After deployment, run smoke tests: health endpoint, auth flow, sample upload

# Incident Response

- Prepare plan: identify, contain, eradicate, recover, lessons learned
- Isolate affected instance: disable new deployments, revoke tokens if needed
- Preserve logs: do not alter until forensic copy made
- Notify stakeholders per protocol (IRB, institutional security)
- For data breach: Follow applicable regulations (HIPAA if covered entity, GDPR if EU data)
- Test plan annually
- Maintain contact list: security lead, devops, legal, PI
- Use version control to track changes; enable rollback
- After incident, update security controls based on findings

# Interaction with Other Skills

Consumes:
- Backend Engineer: API contract details to enforce validation and auth
- Frontend Engineer: UI considerations for secure token handling, CSP, etc.
- MLOps Engineer: MLOps considerations for model provenance)
- Documentation Engineer: Produce security checklists, architecture docs, consent forms

Produces:
- Security review checklists for PRs
- Deployment security verification steps
- Threat modeling documentation
- Privacy impact assessment guidance

Collaborates with:
- DevOps Engineer: For runtime monitoring, logging infrastructure, secret platforms
- Research Engineer: To ensure security does not impede scientific validity
- Medical AI Reviewer: To verify that security controls do not introduce diagnostic language or unsafe workflows

# Common Mistakes

- Hardcoding secrets in code or config files
- Using predictable session secrets or JWT keys
- Allowing file uploads based solely on extension
- Setting CORS to * or allowing null origin
- Returning stack traces or debug info in error responses
- Storing PHI in browser localStorage or sessionStorage
- Using HTTP instead of HTTPS in development (leads to prod mistakes)
- Overly permissive file permissions on SQLite or upload directories
- Forgetting to invalidate sessions on password change
- Using SMS for 2FA without backup methods (SIM swapping risk)
- Logging full request bodies including PHI
- Using deprecated or weak crypto algorithms (MD5, SHA1, DES)
- Skipping input validation on "internal" endpoints assuming frontend protection
- Not setting Secure flag on cookies when using HTTPS
- Using same secret key across environments
- Not updating dependencies for months
- Exposing admin endpoints to public internet without additional auth
- Allowing file uploads to web-accessible directories
- Forgetting to remove .env files from repository after commit
- Using eval or similar dangerous functions with user input
- Disabling CSRF protection for convenience
- Using iframe embedding without proper sandboxing
- Not validating redirect URLs (open redirect)
- Using weak password policies
- Forgetting to set HttpOnly on session cookies
- Storing encryption keys alongside encrypted data

# Never Do

- Never hardcode API keys, secrets, or passwords in source code
- Never trust files
- Never transmit patient data over unencrypted channels
- Never log PHI, patient identifiers, or full DICOM headers containing personal info
- Never bypass authentication for any endpoint, even during development
- Never disable security headers (CSP, HSTS) to troubleshoot UI issues
- Never use default or guessable credentials (admin/admin)
- Never store encryption keys in the same repository as encrypted data
- Never allow file execution from upload directories
- Never use client-side only validation for security-critical checks
- Never send passwords or tokens via email or insecure chat
- Never use the same password/service account across multiple systems
- Never modify security controls without review and testing
- Never ignore security scans or vulnerability alerts
- Never deploy with known critical vulnerabilities
- Never leave debug endpoints exposed in production
- Never use patient data for testing without proper anonymization and IRB approval
- Never assume openness equals security; always validate