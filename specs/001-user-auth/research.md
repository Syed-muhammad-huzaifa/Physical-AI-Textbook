# Research: User Authentication Implementation

## Better Auth Configuration and MCP Integration

### Decision: Use Better Auth with JWT Plugin
Better Auth will be configured with the JWT plugin enabled to provide backend-verifiable tokens. This aligns with the requirement to use Better Auth MCP Server as the authoritative source. The backend is already initialized with uv and FastAPI.

### Rationale:
- Better Auth MCP Server is the required authoritative source
- JWT plugin enables backend token verification
- Maintains session-based authentication for frontend while providing token-based verification for backend
- Supports the required `/auth/me` endpoint functionality
- Backend already uses uv and FastAPI as required

### Alternatives considered:
- Custom JWT implementation: Would violate MCP requirement to use Better Auth as authoritative source
- Separate auth service: Would add unnecessary complexity and violate MCP requirement

## Onboarding Data Persistence Strategy

### Decision: Extend Better Auth User Schema
Onboarding data (experience_level, background, background_detail) will be added as extended fields to the Better Auth user schema rather than using a separate profile table.

### Rationale:
- Simpler data model with fewer joins required
- Better Auth supports user schema extension
- Direct access to onboarding data through Better Auth APIs
- Aligns with the preferred approach mentioned in the feature requirements

### Alternatives considered:
- Separate profile table linked by user_id: Would require additional queries and joins
- Store in application database only: Would not leverage Better Auth's user management

## Frontend Authentication Flow

### Decision: Use Better Auth Client SDK
The Docusaurus frontend will use Better Auth's official client SDK for authentication flows, ensuring compliance with MCP documentation.

### Rationale:
- Direct integration with Better Auth service
- Handles session management automatically
- Provides documented APIs for login, signup, and session checking
- Ensures compliance with MCP requirements

### Alternatives considered:
- Custom authentication client: Would risk non-compliance with MCP requirements
- Third-party auth libraries: Would not integrate properly with Better Auth

## Backend Token Verification

### Decision: JWT Verification with JWKS
The FastAPI backend will verify tokens using Better Auth's JWKS endpoint, implementing proper caching to optimize performance.

### Rationale:
- Standard approach for JWT verification
- JWKS allows for key rotation without service interruption
- Caching reduces repeated HTTP requests to JWKS endpoint
- Aligns with Better Auth's JWT plugin capabilities

### Implementation approach:
- Extract Bearer token from Authorization header
- Verify signature using Better Auth's JWKS
- Validate token expiration and other claims
- Implement JWKS caching with refresh on unknown key ID

## Docusaurus Route Protection

### Decision: Higher-Order Component (HOC) with Redirect
Route protection will be implemented using a Higher-Order Component that checks authentication status and redirects unauthenticated users to login.

### Rationale:
- Standard pattern in React/Docusaurus applications
- Can preserve return-to behavior after login
- Centralized protection logic
- Works well with Docusaurus's page-based routing

### Implementation approach:
- Create `withAuth` HOC that checks session status
- Redirect to `/login` if not authenticated
- Preserve original URL for redirect after login
- Include loading state during authentication check

## Neon Postgres Database Setup

### Decision: Single Database for Both Systems
A single Neon Postgres database will be used for both Better Auth's internal tables and custom onboarding data.

### Rationale:
- Simplifies infrastructure and management
- Better Auth handles its own schema and migrations
- Onboarding data can be stored in extended user fields
- Reduces data consistency issues between systems

### Implementation approach:
- Better Auth applies its own migrations for auth tables
- Custom onboarding fields added to Better Auth user schema
- FastAPI connects with read access to user/onboarding data