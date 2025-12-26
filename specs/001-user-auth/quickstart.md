# Quickstart: User Authentication Setup

## Prerequisites

- Node.js 18+ (for Better Auth service)
- Python 3.11+ (for FastAPI backend)
- uv package manager (for Python dependency management)
- Neon Postgres database instance

## Environment Setup

### 1. Neon Postgres Database
```bash
# Create a Neon Postgres database and get the connection string
# Set the DATABASE_URL environment variable
export DATABASE_URL="postgresql://username:password@ep-xxxxxx.us-east-1.aws.neon.tech/dbname"
```

### 2. Better Auth Configuration
```bash
# Set Better Auth environment variables
export BETTER_AUTH_SECRET="your-super-secret-jwt-signing-key"
export BETTER_AUTH_URL="http://localhost:8080"
export DATABASE_URL="your-neon-postgres-connection-string"
```

## Installation

### 1. Backend Setup
The backend is already initialized with uv and FastAPI. Install additional authentication dependencies:
```bash
cd backend
source .venv/bin/activate  # Activate existing virtual environment
uv pip install python-jose[cryptography] passlib[bcrypt] psycopg2-binary python-multipart better-auth-jwt-plugin
```

### 2. Better Auth Service
```bash
# Install Better Auth with JWT plugin
npm install better-auth better-auth-jwt-plugin
```

### 3. Frontend (Docusaurus) Setup
```bash
cd docs
npm install better-auth better-auth-react
```

## Configuration

### 1. Better Auth Service Configuration
Create `.better-auth/config.ts`:
```typescript
import { betterAuth } from "better-auth";
import { jwtPlugin } from "better-auth-jwt-plugin";

export const auth = betterAuth({
  database: {
    url: process.env.DATABASE_URL!,
    type: "postgres"
  },
  secret: process.env.BETTER_AUTH_SECRET!,
  plugins: [
    jwtPlugin({
      secret: process.env.BETTER_AUTH_SECRET!,
    })
  ],
  // Extend user schema with onboarding fields
  user: {
    fields: {
      experience_level: 'experience_level',
      background: 'background',
      background_detail: 'background_detail',
    }
  }
});
```

### 2. FastAPI Backend Configuration
Create `backend/src/config/settings.py`:
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str
    better_auth_secret: str
    better_auth_url: str
    jwks_url: str = f"{better_auth_url}/api/auth/jwks"

    class Config:
        env_file = ".env"

settings = Settings()
```

## Running the Services

### 1. Start Better Auth Service
```bash
npm run dev  # Runs Better Auth on port 8080
```

### 2. Start FastAPI Backend
```bash
cd backend
source .venv/bin/activate
uvicorn src.main:app --reload --port 8000
```

### 3. Start Docusaurus Frontend
```bash
cd docs
npm run start
```

## Testing the Setup

### 1. Verify Better Auth Service
- Visit `http://localhost:8080/api/auth/health` - should return health status
- Verify JWKS endpoint at `http://localhost:8080/api/auth/jwks`

### 2. Test Authentication Flow
- Visit `http://localhost:3000/signup` to test registration
- Use the `/auth/me` endpoint to verify token-based authentication

### 3. Test Protected Endpoints
```bash
# Get a JWT token from Better Auth after login
# Then test the /auth/me endpoint:
curl -H "Authorization: Bearer <token>" http://localhost:8000/auth/me
```

## Next Steps

1. Implement signup form with onboarding fields
2. Create login form with email/password
3. Implement route protection in Docusaurus
4. Connect frontend to backend with proper authorization headers
5. Test the complete authentication flow