from fastapi import APIRouter
from .rag import router as rag_router
from .chat_agent import router as chat_agent_router

# Main API router that includes all sub-routers
api_router = APIRouter()

# Include RAG routes
api_router.include_router(rag_router, prefix="/rag")

# Include chat agent routes
api_router.include_router(chat_agent_router, prefix="")