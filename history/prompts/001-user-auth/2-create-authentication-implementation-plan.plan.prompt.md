---
id: 2
title: create-authentication-implementation-plan
stage: plan
date: '2025-12-19'
surface: agent
model: claude-sonnet-4-5-20250929
feature: user-auth
branch: 001-user-auth
user: syedhuzaifa
command: /sp.plan
labels:
  - authentication
  - better-auth
  - fastapi
  - neon-postgres
  - docusaurus
links:
  spec: ./spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-user-auth/plan.md
  - specs/001-user-auth/research.md
  - specs/001-user-auth/data-model.md
  - specs/001-user-auth/quickstart.md
  - specs/001-user-auth/contracts/auth-api.yaml
tests: []
---

## User Input
```
# Create Implementation Plan

---

## 1. Grounding & Constraints
- Use **Better Auth MCP Server** as the **only authoritative source**.
- Do not introduce undocumented APIs, SDK behavior, or custom auth flows.
- Backend already exists and uses **FastAPI with uv**.
- All persistence must use **Neon Postgres**.

---

## 2. Neon Postgres Setup
- Create Neon Postgres database.
- Configure `DATABASE_URL` for:
  - Better Auth service
  - FastAPI backend (read-only access if required).
- Verify database connectivity.

---

## 3. Better Auth Service Setup (Node/TS)
- Initialize Better Auth service.
- Configure:
  - `BETTER_AUTH_SECRET`
  - `BETTER_AUTH_URL`
  - `DATABASE_URL` (Neon)
- Enable **Email + Password authentication**.
- Enable **session management** (default Better Auth behavior).
- Apply Better Auth migrations/schema so auth tables are created in Neon.

---

## 4. Persist Onboarding Data
- Choose and implement **one** approach:
  - **Preferred:** extend Better Auth `user` schema with:
    - `experience_level`
    - `background`
    - `background_detail`
- Ensure onboarding fields are saved during signup and linked to the user.
- Validate data persistence in Neon Postgres.

---

## 5. Enable Backend-Verifiable Tokens
- Enable Better Auth **JWT plugin** per MCP documentation.
- Ensure:
  - JWT can be issued after authentication
  - JWKS endpoint is available for verification
- Keep sessions as primary auth model; JWT is used only for backend verification.

---

## 6. Docusaurus: Authentication UI
- Implement pages:
  - `/signup` (email, password + onboarding fields)
  - `/login` (email, password)
- Use Better Auth client APIs exactly as documented.
- Handle validation, errors, and loading states.

---

## 7. Docusaurus: Route Protection
- Implement global route guard:
  - Check authentication/session state
  - Redirect unauthenticated users to `/login`
- Preserve return-to behavior after login.

---

## 8. Frontend → Backend Authorization
- After successful login/signup:
  - Request JWT from Better Auth (MCP-approved method).
- Attach token to protected backend requests:
  Authorization: Bearer <token>

---

## 9. FastAPI Authentication Layer (uv)
- Implement auth dependency/middleware that:
- Extracts Bearer token
- Verifies signature using Better Auth JWKS
- Validates token expiration (and issuer/audience if configured)
- Implement JWKS caching with refresh on unknown key ID.

---

## 10. Required Backend Endpoint
- Implement **GET `/auth/me`** (protected):
- Returns authenticated user identity
- Returns onboarding data from Neon
- Return `401 Unauthorized` if token is missing or invalid.

---

## 11. Validation & Acceptance Testing
- Verify:
- Protected frontend routes redirect to `/login`
- Signup persists onboarding data in Neon
- Login restores access correctly
- `/auth/me` returns:
  - `401` without token
  - `200` with valid token and correct onboarding data

---

## 12. Final Check
- All auth behavior aligns with Better Auth MCP documentation.
- No extra features or assumptions introduced.
- Ready for implementation.
```

## Response Summary
Created comprehensive implementation plan for user authentication feature including:

1. Technical context with language versions, dependencies, and constraints
2. Constitution check confirming alignment with project standards
3. Detailed project structure for backend (FastAPI), Better Auth service, and frontend (Docusaurus)
4. Research document with technical decisions on Better Auth integration, onboarding data persistence, and token verification
5. Data model defining User, Session, Account, and Token entities with validation rules
6. OpenAPI contract for the /auth/me endpoint
7. Quickstart guide with setup instructions and configuration steps
8. Agent context updated with new technology stack information

All components align with the requirement to use Better Auth MCP Server as the authoritative source and implement the specified authentication flows with onboarding data collection.

## Outcome
Implementation plan is complete and ready for the task generation phase (/sp.tasks). All architectural decisions have been documented and validated against the project constitution.