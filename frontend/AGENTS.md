# AGENTS.md

## Purpose

This document defines the expected behavior, standards, workflows, and responsibilities for human and AI contributors working on this project.

All contributors should prioritize:

* Correctness
* Maintainability
* Security
* Performance
* Simplicity
* User Experience

---

# Core Principles

## 1. Understand Before Modifying

Before making any changes:

* Read relevant files completely.
* Understand the existing architecture.
* Identify dependencies and side effects.
* Avoid assumptions.

If requirements are unclear:

* Ask questions.
* Document assumptions.
* Avoid speculative implementations.

---

## 2. Preserve Existing Functionality

Changes must not:

* Break existing features.
* Introduce regressions.
* Remove functionality without approval.
* Alter behavior unexpectedly.

When modifying existing code:

* Verify affected workflows.
* Consider edge cases.
* Maintain backward compatibility whenever possible.

---

## 3. Prefer Simplicity

Choose solutions that are:

* Easy to understand
* Easy to maintain
* Easy to test

Avoid:

* Premature optimization
* Overengineering
* Unnecessary abstractions
* Excessive complexity

---

# Planning Process

For medium and large tasks:

## Step 1: Analysis

Identify:

* Problem statement
* Root cause
* Constraints
* Risks
* Dependencies

## Step 2: Planning

Create:

* Task breakdown
* Implementation strategy
* Testing strategy
* Rollback strategy

## Step 3: Execution

Implement incrementally.

Avoid large unreviewed changes.

## Step 4: Verification

Confirm:

* Build succeeds
* Tests pass
* No regressions introduced
* Requirements satisfied

---

# Coding Standards

## General

Write code that is:

* Readable
* Consistent
* Predictable
* Self-documenting

Prefer:

* Clear naming
* Small functions
* Single responsibility
* Explicit logic

Avoid:

* Magic numbers
* Dead code
* Unused imports
* Excessive comments explaining obvious behavior

---

## Naming

Use meaningful names.

Good:

```text
calculateInvoiceTotal()
validateUserInput()
isAuthenticated
```

Avoid:

```text
doStuff()
temp()
x()
```

---

## Error Handling

Never silently ignore errors.

Requirements:

* Handle expected failures.
* Log useful diagnostic information.
* Provide actionable error messages.
* Fail safely.

Avoid:

```javascript
catch (e) {}
```

---

## Security

Always evaluate:

### Input Validation

Validate:

* User input
* API requests
* Query parameters
* Uploaded files

### Authentication & Authorization

Verify:

* Access permissions
* User roles
* Session validity

### Secrets

Never:

* Hardcode credentials
* Commit API keys
* Store secrets in source code

Use:

* Environment variables
* Secret management systems

### Common Risks

Check for:

* SQL Injection
* XSS
* CSRF
* SSRF
* Command Injection
* Path Traversal
* Insecure Deserialization

---

# Bug Fixing Workflow

## 1. Reproduce

Document:

* Steps
* Environment
* Expected behavior
* Actual behavior

## 2. Diagnose

Identify:

* Root cause
* Impacted components
* Related systems

## 3. Fix

Implement the smallest correct fix.

Avoid unrelated refactoring unless necessary.

## 4. Validate

Confirm:

* Bug is resolved
* No regressions exist
* Tests cover the scenario

---

# Testing Requirements

Every significant change should include:

## Unit Tests

Verify:

* Individual functions
* Components
* Services

## Integration Tests

Verify:

* System interactions
* API flows
* Database operations

## Manual Testing

Verify:

* Critical user journeys
* Edge cases
* Error scenarios

---

# Code Review Checklist

Before approval verify:

## Correctness

* Requirements met
* Logic validated
* Edge cases handled

## Maintainability

* Code is understandable
* Naming is clear
* Complexity is reasonable

## Security

* Inputs validated
* Permissions enforced
* Secrets protected

## Performance

* No obvious bottlenecks
* Queries optimized
* Resource usage acceptable

## Testing

* Tests added or updated
* Existing tests pass

---

# Documentation Requirements

Update documentation whenever:

* Features change
* APIs change
* Configuration changes
* Workflows change

Documentation should include:

* Purpose
* Usage
* Examples
* Limitations

---

# Architecture Guidelines

Prefer:

* Modular design
* Clear separation of concerns
* Reusable components
* Loose coupling

Avoid:

* Circular dependencies
* Monolithic functions
* Hidden side effects

---

# Performance Guidelines

Consider:

* Algorithm complexity
* Database efficiency
* Memory usage
* Network overhead

Optimize only when:

* A bottleneck is identified
* Measurements justify optimization

---

# Git & Commit Standards

## Commit Messages

Format:

```text
type(scope): summary
```

Examples:

```text
feat(auth): add refresh token support
fix(api): resolve validation error handling
refactor(ui): simplify dashboard layout
docs(readme): update installation steps
```

Types:

* feat
* fix
* refactor
* docs
* test
* chore
* perf

---

# Pull Request Standards

Each PR should include:

## Summary

What changed?

## Reason

Why was it needed?

## Testing

How was it verified?

## Risks

Potential impact areas.

---

# AI Agent Responsibilities

When acting as an AI contributor:

## Always

* Read relevant files before editing.
* Explain reasoning when requested.
* Identify assumptions.
* Flag risks.
* Suggest tests.

## Never

* Invent APIs.
* Invent requirements.
* Assume behavior without evidence.
* Remove functionality without justification.
* Ignore warnings or failing tests.

## If Uncertain

State uncertainty explicitly and request clarification.

---

# Definition of Done

A task is complete only when:

* Requirements are satisfied.
* Code builds successfully.
* Tests pass.
* Documentation is updated.
* Security considerations reviewed.
* No critical warnings remain.
* Code review standards are met.

---

# Guiding Principle

Make the smallest change that fully solves the problem while improving the overall quality of the codebase.
