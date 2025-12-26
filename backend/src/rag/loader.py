"""
Markdown loader for the RAG (Retrieval Augmented Generation) system.

This module provides functionality to load markdown files from a directory.
"""
import os
from pathlib import Path
from typing import List, Tuple
from .config import rag_config


def load_markdown_files(docs_dir: str = None) -> List[Tuple[str, str, str]]:
    """
    Load all markdown files from the specified directory recursively,
    excluding ignored directories.

    Args:
        docs_dir: Directory to load markdown files from (defaults to config value)

    Returns:
        List of tuples containing (doc_id, source, text) where:
        - doc_id: relative path from docs_dir
        - source: same as doc_id
        - text: content of the markdown file
    """
    if docs_dir is None:
        docs_dir = rag_config.DOCS_DIR

    docs_dir_path = Path(docs_dir)
    if not docs_dir_path.exists():
        raise FileNotFoundError(f"Directory {docs_dir} does not exist")

    markdown_files = []
    ignored_dirs = {'.git', 'node_modules', 'build', '.docusaurus', '__pycache__'}

    for file_path in docs_dir_path.rglob('*.md'):
        # Check if any parent directory is in ignored list
        if any(parent.name in ignored_dirs for parent in file_path.parents):
            continue

        # Calculate doc_id as relative path from docs_dir
        doc_id = str(file_path.relative_to(docs_dir_path))

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # source is the same as doc_id
        source = doc_id

        markdown_files.append((doc_id, source, content))

    return markdown_files


def is_valid_doc_id(doc_id: str) -> bool:
    """
    Validate if a document ID is valid based on requirements.

    Args:
        doc_id: The document ID to validate

    Returns:
        True if valid, False otherwise
    """
    # Must end with ".md" extension
    if not doc_id.lower().endswith('.md'):
        return False

    # Must not contain path traversal sequences (../)
    if '../' in doc_id or '..\\' in doc_id:
        return False

    # Additional validation can be added here

    return True