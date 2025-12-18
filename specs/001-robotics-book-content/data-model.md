# Data Model: Robotics & Physical AI Book Content

## Overview
This document defines the data model for the 15-chapter Humanoid Robotics book. Since this is a documentation project, the "data model" represents the content structure and relationships.

## Book Chapter Entity

**Name**: Book Chapter
- **title** (string): The chapter title for display and navigation
- **sidebar_position** (number): Position in the sidebar navigation
- **description** (string): Brief description for metadata
- **tags** (array of strings): Content tags for categorization
- **word_count** (number): Estimated word count (2000-4000)
- **learning_objectives** (array of strings): 4-6 learning objectives
- **content_sections** (array of objects): Main content sections
- **code_examples** (array of objects): Minimum 3 code examples
- **diagrams** (array of objects): Minimum 2 Mermaid diagrams
- **troubleshooting_content** (string): Troubleshooting section
- **cross_references** (array of strings): Links to related chapters
- **further_reading** (array of strings): Additional resources

### Content Section Sub-entity
- **section_type** (string): "introduction", "theory", "practical", "summary"
- **title** (string): Section title
- **content** (string): Section content
- **admonitions** (array of objects): Docusaurus admonitions (:::tip, :::warning, :::danger)

### Code Example Sub-entity
- **language** (string): "python", "bash", "yaml", "xml", etc.
- **code** (string): The actual code content
- **file_path** (string): Suggested file path for the example
- **dependencies** (array of strings): Package dependencies
- **expected_output** (string): Expected output when running the code
- **error_handling** (string): Error handling and logging implementation

### Diagram Sub-entity
- **type** (string): "mermaid"
- **diagram_code** (string): Mermaid diagram code
- **caption** (string): Diagram caption
- **purpose** (string): What the diagram illustrates

## Module Entity

**Name**: Module
- **name** (string): Module name (e.g., "Module 1 - ROS2 Fundamentals")
- **chapters** (array of Book Chapter references): Chapters in this module
- **difficulty_level** (string): "beginner", "intermediate", "advanced", "expert"
- **prerequisites** (array of strings): Knowledge required before this module
- **learning_outcomes** (array of strings): What learners will achieve

## Relationships

### Chapter Relationships
- Each **Book Chapter** belongs to exactly one **Module**
- Each **Module** contains multiple **Book Chapters**
- **Book Chapters** can reference other **Book Chapters** through **cross_references**

### Content Relationships
- Each **Book Chapter** contains multiple **Content Sections**
- Each **Book Chapter** contains multiple **Code Examples**
- Each **Book Chapter** contains multiple **Diagrams**
- Each **Content Section** may contain multiple **Admonitions**

## Validation Rules

### Book Chapter Validation
- **word_count** must be between 2000 and 4000
- **learning_objectives** must contain 4-6 items
- **code_examples** must contain at least 3 items
- **diagrams** must contain at least 2 items
- **sidebar_position** must be unique within the module
- **content** must follow Flesch-Kincaid grade 8-10 reading level
- **content** must use active voice and direct address ("you will...")

### Module Validation
- **difficulty_level** must be one of: "beginner", "intermediate", "advanced", "expert"
- **chapters** must be ordered by increasing difficulty
- **prerequisites** must reference completed modules or basic requirements

## State Transitions

Since this is a content project, the "state" represents the completion status of each chapter:

### Chapter State Model
- **draft**: Initial state, basic structure created
- **content_added**: Main content written
- **code_examples_added**: Code examples integrated and tested
- **diagrams_added**: Diagrams created and embedded
- **troubleshooting_added**: Troubleshooting section added
- **reviewed**: Chapter reviewed against constitution checklist
- **built**: Chapter successfully builds with Docusaurus
- **complete**: All quality gates passed, ready for publication

### Module State Model
- **planning**: Module structure defined
- **in_progress**: Chapters being created
- **content_complete**: All chapters have content
- **review_complete**: All chapters reviewed
- **built**: Module successfully builds
- **complete**: All chapters in module complete