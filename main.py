"""
Healthcare Knowledge Navigator — Application Entry Point.
Initializes all components and starts the FastAPI server.
"""
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("healthcare-rag")

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          🏥  Healthcare Knowledge Navigator  🏥              ║
║          Medical RAG + SHAP Explainable AI                   ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_api_key():
    """Check if Gemini API key is configured."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key == "your_gemini_api_key_here":
        print("\n⚠️  GEMINI_API_KEY not configured!")
        print("   To enable AI-powered answers:")
        print("   1. Get a free API key at https://aistudio.google.com/")
        print("   2. Create a .env file in the project root:")
        print("      GEMINI_API_KEY=your_key_here")
        print("\n   The app will still work for document retrieval without the key.\n")
        return False
    return True


def main():
    """Initialize components and start the server."""
    print_banner()

    # Check API key
    gemini_ok = check_api_key()

    # Initialize ingestion pipeline
    logger.info("Initializing document ingestion pipeline...")
    from backend.ingestion import DocumentIngestionPipeline
    pipeline = DocumentIngestionPipeline()

    # Ingest knowledge base on first run
    logger.info("Checking knowledge base...")
    result = pipeline.ingest_directory()
    if result.get("skipped"):
        logger.info(f"Knowledge base already indexed ({result['total_chunks']} chunks)")
    else:
        logger.info(
            f"Indexed {result['files_processed']} files → "
            f"{result['total_chunks']} chunks"
        )

    # Initialize RAG engine
    logger.info("Initializing RAG engine...")
    from backend.rag_engine import RAGEngine
    rag = RAGEngine(pipeline)

    # Initialize XAI engine
    logger.info("Initializing SHAP XAI engine...")
    from backend.xai_engine import XAIEngine
    xai = XAIEngine(pipeline)

    # Wire up to the API module
    import backend.api as api_module
    api_module.ingestion_pipeline = pipeline
    api_module.rag_engine = rag
    api_module.xai_engine = xai

    # Create FastAPI app
    app = api_module.create_app()

    # Print status
    docs = pipeline.get_document_list()
    print(f"\n✅ System ready!")
    print(f"   📚 Documents indexed: {len(docs)}")
    print(f"   📄 Total chunks: {pipeline.get_total_chunks()}")
    print(f"   🧠 Gemini API: {'✅ Connected' if gemini_ok else '❌ Not configured'}")
    print(f"   🔍 SHAP XAI: ✅ Enabled")
    print(f"\n🌐 Open http://localhost:8000 in your browser\n")

    # Start server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
