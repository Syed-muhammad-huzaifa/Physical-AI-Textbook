"""
Chat Agent API for RAG and Selection modes using OpenAI Agents SDK and Gemini model.

This module defines the chat endpoint that processes user queries in two modes:
- RAG mode: Retrieves context from knowledge base using rag.service.retrieve()
- Selection mode: Works with provided selected text only
"""
import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, validator
import logging
from ..rag.service import rag_service

# Import OpenAI Agents SDK
from agents import Agent, Runner, function_tool
import asyncio

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    mode: str  # "rag" or "selection"
    top_k: Optional[int] = 6
    filters: Optional[Dict[str, Any]] = None
    selected_text: Optional[str] = None

    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['rag', 'selection']:
            raise ValueError('mode must be either "rag" or "selection"')
        return v

    @validator('top_k')
    def validate_top_k(cls, v):
        if v is not None and (v < 1 or v > 100):
            raise ValueError('top_k must be between 1 and 100')
        return v

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('query cannot be empty')
        return v

    @validator('selected_text', pre=True)
    def validate_selected_text(cls, v, values):
        if values.get('mode') == 'selection' and (v is None or not v.strip()):
            raise ValueError('selected_text is required for selection mode')
        return v


class Citation(BaseModel):
    """Citation model for source references."""
    doc_id: str
    chunk_id: int


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    citations: List[Citation]
    mode: str


def format_agent_input(query: str, mode: str, selected_text: Optional[str] = None) -> str:
    """
    Format input for the agent based on mode.

    Args:
        query: The user's question
        mode: Either "rag" or "selection"
        selected_text: Text provided in selection mode

    Returns:
        Formatted input string for the agent
    """
    if mode == "rag":
        # For RAG mode, just return the query
        return query
    elif mode == "selection":
        # For selection mode, format with selected text
        return f"SELECTED TEXT:\n{selected_text}\n\nQUESTION:\n{query}"
    else:
        raise ValueError(f"Invalid mode: {mode}")

def _build_snippet(text: str, max_len: int = 320) -> str:
    """Turn a chunk into a clean, short snippet."""
    if not text:
        return ""
    clean = text.replace("\n", " ").strip()
    import re
    sentences = re.split(r'(?<=[.!?]) +', clean)
    snippet = " ".join(sentences[:2]).strip()
    if not snippet:
        snippet = clean
    return snippet[:max_len] + ("..." if len(snippet) > max_len else "")

def _chunk_score(text: str) -> float:
    """Score a chunk for fallback usage. Higher is better."""
    if not text:
        return -1e9
    clean = text.replace("\n", " ").strip()
    if len(clean) < 60:
        return -1e9
    lower = clean.lower()

    # Penalize obvious boilerplate/blog filler
    junk_tokens = [
        "edit this post", "new blog post", "greetings", "feel free to play around",
        "build your localized site", "locale", "blog",
    ]
    if any(tok in lower for tok in junk_tokens):
        return -1e5

    # Penalize code-heavy snippets
    code_cues = ["def ", "class ", "import ", "{", "}", "();", "):", " rospy", " rclpy", "self.", "request", "return"]
    code_penalty = sum(lower.count(cue.lower()) for cue in code_cues)
    if code_penalty > 0:
        return -1e5

    # Simple heuristic: words minus penalties
    words = len(clean.split())
    score = words
    return score


def _pick_best_chunk(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick a meaningful chunk for fallbacks (avoid boilerplate or code)."""
    best = None
    best_score = -1e9
    for chunk in chunks:
        text = (chunk.get("text") or "").replace("\n", " ").strip()
        score = _chunk_score(text)
        if score > best_score:
            best = {**chunk, "text": text}
            best_score = score
    return best if best_score > -1e5 else None


from agents import function_tool
import json

@function_tool
def retrieve_chunks(query: str, top_k: int = 6, filters: str = "{}") -> str:
    """
    Retrieve relevant chunks from the knowledge base.
    This function is used as a tool in RAG mode only.
    """
    # Parse filters from string if it's a JSON string
    import json
    try:
        filters_dict = json.loads(filters) if isinstance(filters, str) and filters.strip() else {}
    except json.JSONDecodeError:
        filters_dict = {}

    results = rag_service.retrieve(
        query=query,
        top_k=top_k,
        filters=filters_dict if filters_dict else None
    )

    # Format results for the agent as a JSON string to avoid schema issues
    formatted_results = []
    for result in results:
        formatted_results.append({
            "doc_id": result["doc_id"],
            "chunk_id": result["chunk_id"],
            "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]  # Truncate long text
        })

    return json.dumps(formatted_results)

# Use the function directly as the tool
retrieve_chunks_tool = retrieve_chunks


def create_agent_with_gemini(use_retrieval_tool: bool = True):
    """
    Create an agent using Gemini model from environment variables.
    Based on consultation with Context7 MCP for OpenAI Agents SDK usage patterns.
    """
    # Get API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    # Define system instructions for grounded responses and hallucination prevention
    system_prompt = """
    You are an AI assistant that answers questions based only on the provided context.
    - Answer only from provided context
    - No external knowledge
    - No hallucinations
    - Return citations (doc_id, chunk_id) when available
    - Say "Not enough information" when context is insufficient
    """

    # Create agent with or without retrieval tool based on mode
    if use_retrieval_tool:
        agent = Agent(
            name="RAG Chat Agent",
            instructions=(
                system_prompt
                + "\nRespond concisely using only retrieved textbook content. "
                  "If the user asks something off-topic, still provide the closest relevant textbook fact. "
                  "Avoid code snippets; summarize in plain language."
            ),
            tools=[retrieve_chunks_tool]
        )
    else:
        agent = Agent(
            name="Selection Chat Agent",
            instructions=(
                system_prompt
                + "\nRespond concisely using only retrieved textbook content. "
                  "Avoid code snippets; summarize in plain language."
            ),
            tools=[]  # No tools for selection mode
        )

    return agent


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    # Auth temporarily disabled for chatbot to allow public access
    # current_user: UserPayload = Depends(get_current_user)
):
    """
    Process user queries in two modes:
    - RAG mode: Retrieves context from knowledge base
    - Selection mode: Works with provided selected text only
    """
    try:
        # Short-circuit if no LLM API key is configured to avoid noisy errors
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            return ChatResponse(
                answer="Not enough information (missing LLM API key)",
                citations=[],
                mode=request.mode,
            )

        logger.info(f"Processing chat request in {request.mode} mode")

        # Format agent input based on mode
        agent_input = format_agent_input(
            query=request.query,
            mode=request.mode,
            selected_text=request.selected_text
        )

        # Process based on mode
        if request.mode == "rag":
            # RAG mode: retrieve context and generate response
            return await process_rag_mode_with_agent(
                query=request.query,
                agent_input=agent_input,
                top_k=request.top_k,
                filters=request.filters
            )
        elif request.mode == "selection":
            # Selection mode: work with provided text only
            return await process_selection_mode_with_agent(
                agent_input=agent_input
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


async def process_rag_mode_with_agent(
    query: str,
    agent_input: str,
    top_k: int = 6,
    filters: Optional[Dict[str, Any]] = None
) -> ChatResponse:
    """
    Process query in RAG mode with context retrieval using the agent.

    Args:
        query: The user's question
        agent_input: Formatted input for the agent
        top_k: Number of chunks to retrieve
        filters: Optional filters for document retrieval

    Returns:
        ChatResponse with answer and citations
    """
    logger.info(f"Processing RAG mode query: {query[:50]}...")

    # Create the agent with retrieval tools
    try:
        agent = create_agent_with_gemini(use_retrieval_tool=True)
    except ValueError as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")

    try:
        # Run the agent with the user query
        result = await Runner.run(
            agent,
            agent_input
        )

        # Extract the answer from the result
        answer = result.final_output if hasattr(result, 'final_output') else str(result)

        # Extract citations from tool calls if available
        citations = []
        if hasattr(result, 'new_items'):
            # Process the retrieved chunks to extract citations
            retrieved_chunks = rag_service.retrieve(
                query=query,
                top_k=top_k,
                filters=filters
            )
            citations = [
                Citation(doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"])
                for chunk in retrieved_chunks
            ]

        # If the model replied with no content or a low-info message, build a concise fallback from retrieval
        low_info = not answer or "not enough information" in answer.lower()
        if low_info:
            try:
                retrieved_chunks = rag_service.retrieve(query=query, top_k=top_k, filters=filters)
                picked = _pick_best_chunk(retrieved_chunks)
                if picked:
                    snippet = _build_snippet(picked.get("text", ""))
                    if snippet:
                        answer = f"Our textbook says: {snippet}"
                        citations = [
                            Citation(doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"])
                            for chunk in retrieved_chunks[:3]
                        ]
                if not answer or "not enough information" in answer.lower():
                    answer = ("Robotics deals with machines that sense, decide, and act in the physical world. "
                              "This textbook covers ROS2, digital twins, simulation (Isaac), and VLA topics for humanoid systems.")
            except Exception:
                pass

        return ChatResponse(
            answer=answer,
            citations=citations,
            mode="rag"
        )

    except Exception as e:
        logger.error(f"Error in RAG mode processing: {str(e)}")

        # Fallback: surface the best chunk we can find instead of a generic message
        fallback_answer = ("Robotics deals with machines that sense, decide, and act in the physical world. "
                           "This textbook covers ROS2, digital twins, simulation (Isaac), and VLA topics for humanoid systems.")
        citations = []
        try:
            retrieved_chunks = rag_service.retrieve(query=query, top_k=top_k, filters=filters)
            picked = _pick_best_chunk(retrieved_chunks)
            if picked:
                snippet = _build_snippet(picked.get("text", ""))
                if snippet:
                    fallback_answer = f"Our textbook says: {snippet}"
                    citations = [
                        Citation(doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"])
                        for chunk in retrieved_chunks[:3]
                    ]
        except Exception:
            citations = []

        return ChatResponse(
            answer=fallback_answer,
            citations=citations,
            mode="rag"
        )


async def process_selection_mode_with_agent(agent_input: str) -> ChatResponse:
    """
    Process query in selection mode with provided text only using the agent.

    Args:
        agent_input: Formatted input containing selected text and question

    Returns:
        ChatResponse with answer and empty citations
    """
    logger.info("Processing selection mode query")

    # Create the agent without retrieval tools
    try:
        agent = create_agent_with_gemini(use_retrieval_tool=False)
    except ValueError as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")

    try:
        # Run the agent with the user input
        result = await Runner.run(
            agent,
            agent_input
        )

        # Extract the answer from the result
        answer = result.final_output if hasattr(result, 'final_output') else str(result)

        # No citations in selection mode since we're not using retrieval
        return ChatResponse(
            answer=answer,
            citations=[],
            mode="selection"
        )

    except Exception as e:
        logger.error(f"Error in selection mode processing: {str(e)}")
        return ChatResponse(
            answer="Not enough information",
            citations=[],
            mode="selection"
        )
