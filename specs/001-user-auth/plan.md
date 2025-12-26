# Implementation Plan: User Authentication

**Branch**: `001-user-auth` | **Date**: 2025-12-19 | **Spec**: [link](./spec.md)
**Input**: Feature specification from `/specs/001-user-auth/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of user authentication system with email/password signup/login, onboarding data collection, and protected routes using Better Auth, FastAPI, and Neon Postgres. The system will include frontend UI for authentication, route protection, backend token verification, and a `/auth/me` endpoint that returns user profile and onboarding data.

## Technical Context

**Language/Version**: Python 3.11 (FastAPI backend), Node.js 18+ (Better Auth service), TypeScript (Better Auth client)
**Primary Dependencies**: Better Auth, FastAPI, Neon Postgres, Docusaurus
**Storage**: Neon Postgres (user accounts, onboarding data, sessions)
**Testing**: pytest (FastAPI backend), Jest/React Testing Library (frontend) or NEEDS CLARIFICATION
**Target Platform**: Web application (Linux server for backend, browser for frontend)
**Project Type**: Web application (frontend + backend)
**Performance Goals**: <500ms for `/auth/me` endpoint, <2min for registration flow, <30s for login flow
**Constraints**: Token-based authentication using Better Auth, JWT verification for backend, secure password storage
**Scale/Scope**: Support 1000+ concurrent users, secure data persistence in Neon Postgres

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the constitution:
- Authentication MUST use better-auth (✓ - implemented with Better Auth service)
- Signup MUST collect software & hardware background (✓ - included in onboarding data: background field with software/hardware/other options)
- Neon Serverless Postgres MUST be used (✓ - implemented as shared database for auth and onboarding data)
- All AI responses MUST be grounded in retrieved content (N/A - not relevant to auth feature)
- Book MUST be built with Docusaurus (✓ - frontend framework implemented)
- Backend framework: FastAPI (✓ - implemented for backend API)
- All requirements align with constitution

## Project Structure

### Documentation (this feature)

```text
specs/001-user-auth/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── main.py          # FastAPI application entry point
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── dependencies.py  # Auth dependency for token verification
│   │   └── models.py    # Auth-related models
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py      # User and onboarding data models
│   │   └── database.py  # Database connection and session management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py      # Auth-related API endpoints (e.g., /auth/me)
│   │   └── router.py    # Main API router
│   └── config/
│       ├── __init__.py
│       └── settings.py  # Configuration settings
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
└── tests/
    ├── __init__.py
    ├── auth/
    │   ├── test_auth.py  # Auth endpoint tests
    │   └── test_dependencies.py  # Auth dependency tests
    └── conftest.py       # Test configuration

# Better Auth service files
.better-auth/
├── config.ts            # Better Auth configuration
├── index.ts             # Better Auth service entry point
└── types.ts             # Better Auth type definitions

# Docusaurus frontend files
docs/
├── src/
│   ├── pages/
│   │   ├── login.js     # Login page component
│   │   ├── signup.js    # Signup page component
│   │   └── protected.js # Protected page example
│   ├── components/
│   │   └── Auth/
│   │       ├── AuthProvider.js  # Auth context provider
│   │       ├── useAuth.js      # Auth hook
│   │       └── withAuth.js     # Auth HOC for route protection
│   └── css/
│       └── auth.css     # Auth-specific styles
├── docusaurus.config.js # Docusaurus configuration with auth settings
└── package.json         # Frontend dependencies including Better Auth client
```

**Structure Decision**: Web application structure with separate backend (FastAPI) and frontend (Docusaurus) components. Better Auth service runs as a Node.js service with JWT plugin enabled for backend token verification. Neon Postgres serves as the shared database for both Better Auth user data and custom onboarding data.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [None] | [No violations found] | [All requirements align with constitution] |
