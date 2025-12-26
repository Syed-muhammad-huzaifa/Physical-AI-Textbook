# Data Model: User Authentication

## User Entity

### Core User Fields (Managed by Better Auth)
- **id**: string (primary key, UUID) - Unique identifier for the user
- **email**: string (unique, indexed) - User's email address for authentication
- **email_verified**: boolean - Whether the email has been verified
- **name**: string (optional) - User's display name
- **image**: string (optional) - Profile image URL
- **created_at**: timestamp - Account creation time
- **updated_at**: timestamp - Last update time
- **hashed_password**: string - Bcrypt-hashed password

### Extended Onboarding Fields (Custom)
- **experience_level**: string (enum: "beginner", "intermediate", "advanced") - User's experience level
- **background**: string (enum: "software", "hardware", "other") - User's background
- **background_detail**: string (optional) - Additional details about user's background

### Relationships
- **sessions**: One-to-Many (user has many active sessions)
- **accounts**: One-to-Many (user has many OAuth accounts - if extended later)

## Session Entity (Managed by Better Auth)

### Session Fields
- **id**: string (primary key, UUID) - Unique session identifier
- **user_id**: string (foreign key) - Reference to user
- **expires_at**: timestamp - Session expiration time
- **created_at**: timestamp - Session creation time
- **updated_at**: timestamp - Last update time

## Account Entity (Managed by Better Auth)

### Account Fields
- **id**: string (primary key, UUID) - Unique account identifier
- **user_id**: string (foreign key) - Reference to user
- **provider**: string - Authentication provider (e.g., "credentials" for email/password)
- **provider_account_id**: string - Provider's account identifier
- **created_at**: timestamp - Account creation time
- **updated_at**: timestamp - Last update time

## Token Entity (Managed by Better Auth)

### Token Fields
- **id**: string (primary key, UUID) - Unique token identifier
- **user_id**: string (foreign key) - Reference to user
- **type**: string (enum: "refresh", "access", "magiclink", etc.) - Token type
- **expires_at**: timestamp - Token expiration time
- **created_at**: timestamp - Token creation time

## Validation Rules

### User Validation
- Email must be a valid email format
- Experience level must be one of: "beginner", "intermediate", "advanced"
- Background must be one of: "software", "hardware", "other"
- Background detail, if provided, must be less than 500 characters
- Email must be unique across all users

### Session Validation
- Session must not be expired (expires_at > current time)
- Session must be linked to an existing user

## State Transitions

### User States
- **Active**: User has completed registration and email verification
- **Pending**: User has registered but not yet verified email
- **Suspended**: User account has been suspended (not implemented in this feature)

### Session States
- **Active**: Session is valid and not expired
- **Expired**: Session has passed its expiration time
- **Revoked**: Session has been manually invalidated (logout)