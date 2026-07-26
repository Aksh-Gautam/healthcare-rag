"""
FastAPI application — REST API for the Healthcare Knowledge Navigator.
"""
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.ingestion import DocumentIngestionPipeline
from backend.rag_engine import RAGEngine
from backend.xai_engine import XAIEngine
from backend.models import (
    QueryRequest,
    QueryResponse,
    DocumentListResponse,
    DocumentInfo,
    UploadResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

# Module-level references (initialized by main.py)
ingestion_pipeline: DocumentIngestionPipeline = None  # type: ignore
rag_engine: RAGEngine = None  # type: ignore
xai_engine: XAIEngine = None  # type: ignore


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Healthcare Knowledge Navigator",
        description="Medical RAG assistant with SHAP-based explainable AI",
        version="1.0.0",
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- API Routes ----

    @app.post("/api/query", response_model=QueryResponse)
    async def query_endpoint(request: QueryRequest):
        """Submit a medical query and get a RAG response with SHAP explanation."""
        if rag_engine is None:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")

        # Run RAG pipeline
        response = await rag_engine.query(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter,
            enable_shap=request.enable_shap,
        )

        # Compute SHAP explanation if enabled and chunks were retrieved
        if request.enable_shap and response.retrieved_chunks and xai_engine:
            try:
                shap_explanation = xai_engine.compute_shap_explanation(
                    query=request.query,
                    chunks=response.retrieved_chunks,
                    rag_engine=rag_engine,
                    nsamples=32,
                )
                response.shap_explanation = shap_explanation
            except Exception as e:
                logger.error(f"SHAP computation error: {e}", exc_info=True)
                # Response still returned without SHAP

        return response

    @app.get("/api/documents", response_model=DocumentListResponse)
    async def list_documents():
        """List all indexed documents."""
        if ingestion_pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        docs = ingestion_pipeline.get_document_list()
        total_chunks = ingestion_pipeline.get_total_chunks()

        return DocumentListResponse(
            documents=[DocumentInfo(**d) for d in docs],
            total_documents=len(docs),
            total_chunks=total_chunks,
        )

    @app.post("/api/documents/upload", response_model=UploadResponse)
    async def upload_document(file: UploadFile = File(...)):
        """Upload a new medical document for indexing."""
        if ingestion_pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Read file content
        content = await file.read()
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be a UTF-8 text file (.txt)",
            )

        # Save to knowledge base directory
        kb_dir = Path(__file__).parent.parent / "knowledge_base"
        kb_dir.mkdir(exist_ok=True)
        filepath = kb_dir / file.filename
        filepath.write_text(text_content, encoding="utf-8")

        # Ingest the document
        chunks_created = ingestion_pipeline.ingest_uploaded_content(
            filename=file.filename,
            content=text_content,
        )

        return UploadResponse(
            success=True,
            message=f"Document '{file.filename}' indexed successfully with {chunks_created} chunks.",
            filename=file.filename,
            chunks_created=chunks_created,
        )

    @app.get("/api/categories")
    async def get_categories():
        """Get available document categories."""
        if ingestion_pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        categories = ingestion_pipeline.get_categories()
        return {"categories": categories}

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        docs_indexed = 0
        chunks_indexed = 0
        if ingestion_pipeline:
            docs = ingestion_pipeline.get_document_list()
            docs_indexed = len(docs)
            chunks_indexed = ingestion_pipeline.get_total_chunks()

        return HealthResponse(
            status="healthy",
            documents_indexed=docs_indexed,
            chunks_indexed=chunks_indexed,
            gemini_configured=rag_engine.is_configured() if rag_engine else False,
        )

    # ---- Static Files (Frontend) ----
    frontend_dir = Path(__file__).parent.parent / "frontend"

    @app.get("/")
    async def serve_index():
        """Serve the frontend index.html."""
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "Frontend not found. Place files in /frontend directory."}

    # Mount static files AFTER explicit routes
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    return app
