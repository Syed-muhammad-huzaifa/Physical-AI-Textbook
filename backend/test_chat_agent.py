#!/usr/bin/env python3
"""
Test script for the chat agent functionality.
This script tests both RAG and selection modes.
"""
import asyncio
import os
from src.api.chat_agent import process_rag_mode_with_agent, process_selection_mode_with_agent, format_agent_input

async def test_rag_mode():
    """Test RAG mode with a query about ROS."""
    print("Testing RAG mode...")

    query = "What is ROS?"
    agent_input = format_agent_input(query, "rag")

    try:
        result = await process_rag_mode_with_agent(
            query=query,
            agent_input=agent_input,
            top_k=3
        )
        print(f"RAG Mode Result:")
        print(f"  Answer: {result.answer}")
        print(f"  Citations: {len(result.citations)}")
        print(f"  Mode: {result.mode}")
        print()
        return result
    except Exception as e:
        print(f"Error in RAG mode: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_selection_mode():
    """Test selection mode with provided text."""
    print("Testing Selection mode...")

    selected_text = "ROS (Robot Operating System) is not an operating system but rather a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms."
    query = "Summarize this text"
    agent_input = format_agent_input(query, "selection", selected_text)

    try:
        result = await process_selection_mode_with_agent(agent_input=agent_input)
        print(f"Selection Mode Result:")
        print(f"  Answer: {result.answer}")
        print(f"  Citations: {len(result.citations)}")
        print(f"  Mode: {result.mode}")
        print()
        return result
    except Exception as e:
        print(f"Error in Selection mode: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Run tests for both modes."""
    print("Starting chat agent tests...\n")

    # Test RAG mode
    rag_result = await test_rag_mode()

    # Test Selection mode
    selection_result = await test_selection_mode()

    print("Tests completed!")

    if rag_result:
        print(f"RAG mode test: {'PASSED' if rag_result.answer else 'FAILED'}")
    else:
        print("RAG mode test: FAILED")

    if selection_result:
        print(f"Selection mode test: {'PASSED' if selection_result.answer else 'FAILED'}")
    else:
        print("Selection mode test: FAILED")

if __name__ == "__main__":
    asyncio.run(main())