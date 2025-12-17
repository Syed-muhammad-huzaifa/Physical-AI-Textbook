---
name: update-humanoid-robotics-constitution
model: claude-sonnet-4-5-20250929
description: Update the project constitution for the Humanoid & Robotics AI Book Platform
---

## Context

This PHR documents the creation and update of the project constitution for the Humanoid & Robotics AI Book Platform.

## Task

Update the project constitution at .specify/memory/constitution.md with the complete requirements for the Humanoid & Robotics AI Book Platform.

# Project Constitution — Humanoid & Robotics AI Book Platform

## Project Scope
This Constitution defines the immutable standards for a unified AI-driven robotics textbook with an embedded RAG chatbot and a professional frontend experience.

All features, UI components, AI behavior, and integrations MUST comply with this Constitution.

---

## Core Principles

1. Grounded & Deterministic AI
- All AI responses MUST be grounded strictly in retrieved book content
- Hallucinated or fabricated answers are forbidden

2. Textbook-First Experience
- The platform MUST feel like a technical textbook, not a marketing site
- Readability and clarity ALWAYS take priority over visuals

3. Feature Isolation
- Each feature (chat, personalization, translation, UI sections) MUST be independently testable

---

## Platform & Stack Standards

### Book Platform
- The book MUST be built with Docusaurus
- The book MUST be deployed to GitHub Pages
- All content MUST be version-controlled

### Backend & AI
- Backend framework: FastAPI
- RAG orchestration: OpenAI Agents SDK / ChatKit SDK
- Vector DB: Qdrant Cloud (Free Tier)
- Relational DB: Neon Serverless Postgres
- A single embedding model MUST be used consistently

---

## RAG Chatbot Rules

- Answers MUST come ONLY from retrieved book content
- If no relevant content exists, the system MUST say it does not know
- Source chapter or section MUST be referenced
- Selected-text-only answering MUST be supported
- Selected-text mode MUST NOT use global context
- External internet access is forbidden

---

## UX / UI DESIGN CONSTITUTION (FRONTEND LAW)

### Design Identity
- Theme name: Robotics Lab
- Tone: technical, calm, futuristic, readable
- Dark-mode first

### Typography
- Headings: Space Grotesk (600–700)
- Body: Inter (400–600)
- Body size: 16–18px
- Line height: 1.6–1.8
- Max paragraph width: 70–78ch

### Color Tokens
- Background: #070A12
- Surface: #0B1220
- Border: rgba(255,255,255,0.08)
- Text: #EAF0FF
- Muted: rgba(234,240,255,0.72)
- Accent: #22D3EE
- Optional secondary accent: #8B5CF6

### Layout Rules
- 8px spacing system
- Section spacing: 48–72px
- Cards: radius 16px, padding 18–22px
- Grid gaps: 12–16px
- Navbar: blur + thin border

### Motion Rules (Minimal & Purposeful)
- Section entry: single-time fade + slide-up (6–12px max)
- Card hover: lift 2–4px with very soft glow or border emphasis
- Buttons/CTAs: subtle press feedback only
- Motion MUST never cause layout shift
- Animations MUST respect prefers-reduced-motion

Forbidden:
- Particles, parallax, or decorative motion
- Constant or looping animations
- Bouncy or spring-based effects
- Flashy neon glows

### Accessibility
- Mobile-first, no overflow
- Accessible contrast
- No layout shifts
- Text must remain readable on all devices

---

## Authentication & Personalization (Bonus)

- Authentication MUST use better-auth
- Signup MUST collect software & hardware background
- Personalized content MUST adapt tone and depth only
- Meaning of original content MUST NOT change
- Urdu translation MUST preserve technical accuracy

---

## Quality Gates

The project is successful ONLY if:
- Book is live on GitHub Pages
- RAG chatbot works with grounded answers
- Selected-text mode works correctly
- UI matches Robotics Lab standards
- Mobile UX is flawless
- No hallucinations observed

---

## Final Rule
If design beauty conflicts with readability:
**Readability ALWAYS wins.**

## Solution

Updated the constitution file with the complete Humanoid & Robotics AI Book Platform constitution, replacing the template placeholders with specific requirements for the project. The constitution now includes:

1. Core principles for grounded AI, textbook-first experience, and feature isolation
2. Platform & Stack Standards specifying Docusaurus, FastAPI, Qdrant Cloud, and Neon Postgres
3. RAG Chatbot Rules ensuring responses are grounded in book content
4. Comprehensive UX/UI Design Constitution with Robotics Lab theme
5. Authentication & Personalization guidelines
6. Quality gates for project success
7. Governance section with version information

The constitution is now properly configured for the humanoid and robotics textbook project.

## Outcome

- ✅ Impact: High - established foundational constitution for the project
- 🧪 Tests: Not applicable for constitution update
- 📁 Files: Updated .specify/memory/constitution.md with complete project constitution
- 🔁 Next steps: Can proceed with feature specifications based on this constitution
- 🧠 Reflection: Constitution provides clear guidelines for development decisions