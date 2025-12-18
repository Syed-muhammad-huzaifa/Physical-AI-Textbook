# Quickstart Guide: Robotics & Physical AI Book Content

## Overview
This quickstart guide provides the essential information needed to begin creating the 15-chapter Humanoid Robotics book using Claude Code agents.

## Prerequisites

### System Requirements
- Ubuntu 22.04/24.04 (primary target platform)
- ROS2 Jazzy/Humble installed
- Node.js 18+ with npm/yarn
- Git for version control
- Claude Code environment properly configured

### Development Environment Setup
```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install Docusaurus dependencies
npm install

# Verify ROS2 installation
source /opt/ros/jazzy/setup.bash  # or humble
ros2 --version
```

## Agent Configuration

### Available Claude Code Agents
- `@book-content-writer` - Generate complete chapter structure and content
- `@code-generator` - Add code examples (ROS2 nodes, launch files, configs)
- `@diagram-generator` - Create Mermaid diagrams
- `@troubleshooting-writer` - Add debugging sections

## Creating a Chapter

### Step 1: Create the MDX File
Create the chapter file in the appropriate module directory with proper MDX frontmatter:

```md
---
title: "Chapter Title"
sidebar_position: 1
description: "Brief description of the chapter content"
tags: [tag1, tag2, tag3]
---

import { Diagram1, Diagram2 } from './diagrams';

# Chapter Title

## Learning Objectives

- Objective 1
- Objective 2
- Objective 3
- Objective 4
- Objective 5
- Objective 6

## Introduction

Your introduction content here...

## Theory

Theoretical concepts and explanations...

## Practical Examples

Code examples and practical applications...

## Troubleshooting

Common issues and solutions...

## Summary

Chapter summary...

## Further Reading

Additional resources and references...
```

### Step 2: Generate Content with Agents
1. Use `@book-content-writer` to create the basic chapter structure and content
2. Use `@code-generator` to add 3+ working code examples
3. Use `@diagram-generator` to create 2+ Mermaid diagrams
4. Use `@troubleshooting-writer` to add troubleshooting section

### Step 3: Validate Quality Gates
- [ ] Word count: 2000-4000 words
- [ ] 4-6 learning objectives
- [ ] 3+ working code examples
- [ ] 2+ Mermaid diagrams with captions
- [ ] Troubleshooting section
- [ ] Proper MDX frontmatter
- [ ] Docusaurus build succeeds: `npm run build`
- [ ] Constitution checklist passed

## Module Creation Order

Follow this sequential order to maintain proper learning progression:

1. **Module 1 - ROS2 Fundamentals** (Week 1)
   - INTRO.md
   - ROS2-ARCHITECTURE.md
   - COMMUNICATION-MODEL.md
   - PYTHON-CONTROL-AGENTS.md
   - HUMANOID-URDF-DESIGN.md

2. **Module 2 - Digital Twin** (Week 2)
   - GAZEBO-ENVIRONMENT.md
   - PHYSICS-AND-COLLISIONS.md
   - SENSOR-SIMULATION.md
   - UNITY-VISUALIZATION.md

3. **Module 3 - Isaac Platform** (Week 3)
   - ISAAC-SIM-PLATFORM.md
   - SYNTHETIC-DATA-PERCEPTION.md
   - VSLAM-AND-LOCALIZATION.md
   - NAVIGATION-AND-SIM2REAL.md

4. **Module 4 - Vision-Language-Action** (Week 4)
   - VISION-LANGUAGE-ACTION.md
   - VOICE-TO-INTENT.md
   - LLM-TASK-PLANNING.md
   - AUTONOMOUS-HUMANOID-CAPSTONE.md

## Content Standards

### Writing Style
- Use active voice and direct address ("you will...")
- Maintain Flesch-Kincaid grade 8-10 reading level
- Include Docusaurus admonitions: `:::tip`, `:::warning`, `:::danger`

### Code Examples
- Use Python snake_case naming conventions
- Include file paths in comments
- Document package dependencies
- Include proper error handling and logging
- Show expected output

### Diagrams
- Use Mermaid diagrams for scalability
- Include descriptive captions
- Ensure diagrams enhance understanding

## Testing and Validation

### Build Testing
```bash
# Test Docusaurus build
npm run build

# Test local development
npm run start
```

### Quality Validation
- Verify all code examples work in Ubuntu 22.04/24.04 with ROS2 Jazzy/Humble
- Ensure cross-references between chapters are accurate
- Confirm content progresses logically from beginner to expert
- Validate all diagrams render correctly