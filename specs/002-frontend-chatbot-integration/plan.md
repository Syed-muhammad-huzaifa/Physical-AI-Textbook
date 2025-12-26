# Implementation Plan: Frontend Chatbot Integration for Docusaurus

**Branch**: `002-frontend-chatbot-integration` | **Date**: 2025-12-20 | **Spec**: [specs/002-frontend-chatbot-integration/spec.md](specs/002-frontend-chatbot-integration/spec.md)

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Integrate the RAG chatbot into the Docusaurus frontend with two interaction modes: RAG mode for full book context queries and selection mode for contextual responses to selected text. The implementation will include a persistent chat interface, text selection detection, and proper API communication with the backend `/chat` endpoint.

## Technical Context

**Language/Version**: TypeScript, JavaScript, React/JSX for Docusaurus integration
**Primary Dependencies**: Docusaurus v3.x, React, TypeScript, axios/fetch for API calls
**Storage**: Browser localStorage for conversation history, DOM API for text selection
**Testing**: Jest, React Testing Library, Cypress for E2E testing
**Target Platform**: Web browsers (Chrome, Firefox, Safari, Edge)
**Project Type**: Docusaurus documentation website enhancement
**Performance Goals**: <5 seconds response time, non-blocking UI interactions, minimal bundle size impact
**Constraints**: Must not interfere with existing Docusaurus functionality, responsive design required, accessibility compliance
**Scale/Scope**: Single website integration with persistent chat across all pages

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Grounded & Deterministic AI Compliance
- ✅ All AI responses will be grounded in retrieved book content from existing RAG system
- ✅ System will respond "Not enough information" when context is insufficient
- ✅ No hallucinated or fabricated answers will be generated

### RAG Chatbot Rules Compliance
- ✅ Answers will come ONLY from retrieved book content via existing rag.service.retrieve()
- ✅ System will say it does not know when no relevant content exists
- ✅ Source chapter/section will be referenced via citations (doc_id, chunk_id)
- ✅ Selected-text-only answering will be supported (no global context in selection mode)
- ✅ Selected-text mode will NOT use global context (retrieval tool disabled)
- ✅ External internet access will be forbidden (using existing rag.service only)

### Platform & Stack Standards Compliance
- ✅ Backend framework: FastAPI (already established in project)
- ✅ RAG orchestration: OpenAI Agents SDK (as required by spec)
- ✅ Vector DB: Using existing Qdrant Cloud via rag.service.retrieve() (no direct access)

### Frontend/UX DESIGN CONSTITUTION (FRONTEND LAW) Compliance
- ✅ Theme: Robotics Lab (dark-mode first, technical, calm, futuristic)
- ✅ Typography: Space Grotesk for headings, Inter for body
- ✅ Color tokens: Background #070A12, Surface #0B1220, Text #EAF0FF
- ✅ Layout: 8px spacing system, cards with 16-22px padding
- ✅ Motion: Minimal fade/slide animations, respects prefers-reduced-motion
- ✅ Accessibility: Mobile-first, proper contrast ratios, keyboard navigation

## Project Structure

### Documentation (this feature)
```text
specs/[###-feature]/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (Docusaurus structure)
```text
book-source/
├── src/
│   ├── components/
│   │   ├── ChatBot/           # New chatbot component directory
│   │   │   ├── ChatBot.tsx    # Main chatbot UI component
│   │   │   ├── ChatWindow.tsx # Chat window with messages
│   │   │   ├── MessageBubble.tsx # Individual message display
│   │   │   ├── TextInput.tsx  # Input field with send button
│   │   │   ├── SelectionHandler.tsx # Text selection detection
│   │   │   └── api.ts         # API communication utilities
│   │   └── FloatingButton.tsx # Floating chat button
│   ├── css/
│   │   └── chatbot.css        # Chatbot-specific styles
│   └── theme/
│       └── Root.tsx           # Wrap app with chatbot context
└── docusaurus.config.js       # Update to include chatbot components
```

**Structure Decision**: Docusaurus component-based approach with floating chat interface that integrates seamlessly with existing documentation pages.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |