from dotenv import load_dotenv

# Load environment variables before importing app modules so config picks them up
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.router import api_router
import os

def create_app():
    app = FastAPI(title="Humanoid Robotics Textbook API", version="1.0.0")

    # Add CORS middleware - following MCP server guidance
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",    # Default Docusaurus dev server
            "http://localhost:3001",    # Alternative Docusaurus port
            "http://localhost:8080",    # Alternative Docusaurus port
            "http://localhost:8000",    # For local development
            "http://localhost:8081",    # For local development
            "https://your-docusaurus-site.example.com"  # Production domain
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Allow credentials to be passed with cross-origin requests
        # This is essential for cookies to work properly
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {"message": "Humanoid Robotics Textbook API"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "localhost"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )
