"""
Pydantic models for request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request model for medical queries."""
    query: str = Field(..., min_length=3, max_length=2000, description="The medical question to answer")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of documents to retrieve")
    category_filter: Optional[str] = Field(default=None, description="Filter by document category")
    enable_shap: bool = Field(default=True, description="Whether to compute SHAP explanations")


class DocumentChunk(BaseModel):
    """A retrieved document chunk with metadata."""
    chunk_id: str
    content: str
    source: str
    title: str
    category: str
    evidence_level: str
    specialty: str
    relevance_score: float
    date: str


class ShapExplanation(BaseModel):
    """SHAP explanation for document contributions."""
    chunk_labels: list[str] = Field(description="Labels for each retrieved chunk")
    shap_values: list[float] = Field(description="SHAP value for each chunk's contribution")
    base_value: float = Field(description="Base value (expected score without any chunks)")
    explanation_text: str = Field(description="Human-readable explanation of the SHAP analysis")


class ConfidenceBreakdown(BaseModel):
    """Confidence scoring breakdown."""
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall confidence 0-1")
    retrieval_relevance: float = Field(ge=0.0, le=1.0, description="How relevant the retrieved docs are")
    source_agreement: float = Field(ge=0.0, le=1.0, description="How much sources agree")
    evidence_quality: float = Field(ge=0.0, le=1.0, description="Quality of evidence cited")
    coverage: float = Field(ge=0.0, le=1.0, description="How well sources cover the query")


class Citation(BaseModel):
    """A citation reference."""
    index: int
    source: str
    title: str
    excerpt: str
    relevance: float


class QueryResponse(BaseModel):
    """Full response to a medical query."""
    query: str
    answer: str
    confidence: ConfidenceBreakdown
    citations: list[Citation]
    retrieved_chunks: list[DocumentChunk]
    shap_explanation: Optional[ShapExplanation] = None
    disclaimer: str


class DocumentInfo(BaseModel):
    """Info about an indexed document."""
    filename: str
    title: str
    source: str
    category: str
    evidence_level: str
    specialty: str
    date: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response listing all indexed documents."""
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class UploadResponse(BaseModel):
    """Response after uploading a document."""
    success: bool
    message: str
    filename: str
    chunks_created: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    documents_indexed: int
    chunks_indexed: int
    gemini_configured: bool
