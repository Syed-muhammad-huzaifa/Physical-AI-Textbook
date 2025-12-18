# Content API Contract: Humanoid Robotics Book

## Overview
This contract defines the structure and format for the 15-chapter Humanoid Robotics book content. Since this is a documentation project, the "API" represents the content structure contract between different components.

## Chapter Content Structure Contract

### MDX Frontmatter Schema
```yaml
title: string (required) - Chapter title
sidebar_position: number (required) - Position in navigation
description: string (required) - Chapter description
tags: array[string] (required) - Content tags
```

### Chapter Content Schema
```yaml
learning_objectives: array[string] (4-6 items, required)
introduction: string (required)
theory_section: string (required)
practical_examples: array[CodeExample] (minimum 3, required)
troubleshooting_section: string (required)
summary: string (required)
further_reading: array[string] (required)
diagrams: array[Diagram] (minimum 2, required)
cross_references: array[string] (optional)
```

### Code Example Schema
```yaml
language: string (required) - One of: python, bash, yaml, xml
code: string (required) - The actual code content
file_path: string (optional) - Suggested file location
dependencies: array[string] (optional) - Package dependencies
expected_output: string (optional) - What the code should output
comments: string (required) - Explanation of the code
```

### Diagram Schema
```yaml
type: string (required) - Always "mermaid"
code: string (required) - Mermaid diagram code
caption: string (required) - Diagram description
```

## Content Generation API

### Agent Interface Contracts

#### @book-content-writer Contract
- **Input**: Chapter topic, learning objectives, target word count (2000-4000)
- **Output**: Complete MDX chapter with basic structure
- **Guarantees**:
  - Proper MDX frontmatter
  - Learning objectives section
  - Introduction, theory, and summary sections
  - Flesch-Kincaid grade 8-10 readability
  - Active voice and direct address ("you will...")

#### @code-generator Contract
- **Input**: Chapter content, target language, ROS2 context
- **Output**: Minimum 3 working code examples with proper error handling
- **Guarantees**:
  - Code runs on Ubuntu 22.04/24.04 with ROS2 Jazzy/Humble
  - Proper Python snake_case naming conventions
  - File paths in comments
  - Package dependencies documented
  - Expected output included

#### @diagram-generator Contract
- **Input**: Chapter content, visualization needs
- **Output**: Minimum 2 Mermaid diagrams with captions
- **Guarantees**:
  - Valid Mermaid syntax
  - Diagrams enhance understanding of content
  - Proper captions explaining the diagram

#### @troubleshooting-writer Contract
- **Input**: Chapter content, code examples
- **Output**: Troubleshooting section with common issues
- **Guarantees**:
  - Common errors and solutions
  - Debugging tips
  - Troubleshooting commands

## Validation Contract

### Content Validation Rules
- **Word Count**: 2000-4000 words per chapter
- **Learning Objectives**: 4-6 clear, measurable objectives
- **Code Examples**: Minimum 3, all tested and working
- **Diagrams**: Minimum 2 Mermaid diagrams with captions
- **Cross-References**: Accurate links to related chapters
- **Reading Level**: Flesch-Kincaid grade 8-10
- **Writing Style**: Active voice, direct address ("you will...")

### Build Validation Contract
- **Docusaurus Build**: All chapters must build successfully
- **Link Validation**: All internal links must be valid
- **Content Structure**: Must follow MDX schema
- **Frontmatter**: Must include all required fields

## Module Dependencies Contract

### Module Prerequisites
- **Module 1** (ROS2 Fundamentals): No prerequisites beyond Python knowledge
- **Module 2** (Digital Twin): Requires Module 1 completion
- **Module 3** (Isaac Platform): Requires Module 1-2 completion
- **Module 4** (VLA): Requires Module 1-3 completion

### Cross-Module References
- Modules may reference earlier modules but not later ones
- Common concepts should be explained in earlier modules
- Advanced concepts build on earlier foundation knowledge

## Quality Gates Contract

Each chapter must pass all quality gates before being considered complete:

- [ ] Content meets word count requirements (2000-4000 words)
- [ ] Includes 4-6 learning objectives
- [ ] Contains minimum 3 working code examples
- [ ] Includes minimum 2 Mermaid diagrams
- [ ] Has troubleshooting section
- [ ] Proper MDX frontmatter included
- [ ] Docusaurus build succeeds
- [ ] Meets constitution quality standards