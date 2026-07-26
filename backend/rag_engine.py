"""
RAG Engine — Core retrieval-augmented generation pipeline.
Uses ChromaDB for retrieval and Google Gemini for generation.
"""
import json
import logging
import os
from typing import Optional

from google import genai

from backend.ingestion import DocumentIngestionPipeline
from backend.models import (
    Citation,
    ConfidenceBreakdown,
    DocumentChunk,
    QueryResponse,
)
from backend.prompts import (
    CONFIDENCE_ASSESSMENT_PROMPT,
    DISCLAIMER,
    QUERY_TEMPLATE,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class RAGEngine:
    """Core RAG pipeline: retrieve → generate → cite → score."""

    def __init__(self, ingestion_pipeline: DocumentIngestionPipeline):
        self.pipeline = ingestion_pipeline
        self.gemini_client = None
        self.model_name = "gemini-2.0-flash"
        self._init_gemini()

    def _init_gemini(self):
        """Initialize Google Gemini client."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key and api_key != "your_gemini_api_key_here":
            try:
                self.gemini_client = genai.Client(api_key=api_key)
                logger.info("Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        else:
            logger.warning(
                "GEMINI_API_KEY not configured. Set it in .env file. "
                "Get a free key at https://aistudio.google.com/"
            )

    def is_configured(self) -> bool:
        """Check if the Gemini API is configured."""
        return self.gemini_client is not None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Retrieve relevant document chunks for a query."""
        raw_results = self.pipeline.search(
            query=query,
            top_k=top_k,
            category_filter=category_filter,
        )

        chunks = []
        for r in raw_results:
            meta = r["metadata"]
            chunks.append(DocumentChunk(
                chunk_id=r["chunk_id"],
                content=r["content"],
                source=meta.get("source", "Unknown"),
                title=meta.get("title", "Unknown"),
                category=meta.get("category", "General"),
                evidence_level=meta.get("evidence_level", "Not specified"),
                specialty=meta.get("specialty", "General"),
                relevance_score=round(r["relevance_score"], 4),
                date=meta.get("date", "Unknown"),
            ))

        return chunks

    def _build_context(self, chunks: list[DocumentChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] — {chunk.title} "
                f"(Source: {chunk.source}, Evidence Level: {chunk.evidence_level}, "
                f"Category: {chunk.category})\n"
                f"{chunk.content}\n"
            )
        return "\n---\n".join(context_parts)

    def generate(self, query: str, chunks: list[DocumentChunk]) -> str:
        """Generate an answer using Gemini given retrieved context."""
        if not self.gemini_client:
            return (
                "⚠️ Gemini API not configured. Please set your GEMINI_API_KEY in the .env file. "
                "Get a free key at https://aistudio.google.com/\n\n"
                "Based on the retrieved documents, here are the relevant sources:\n" +
                "\n".join(
                    f"- [{c.title}] ({c.source}): relevance {c.relevance_score:.0%}"
                    for c in chunks
                )
            )

        context = self._build_context(chunks)
        prompt = QUERY_TEMPLATE.format(context=context, query=query)

        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.3,
                    "max_output_tokens": 2048,
                },
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"Error generating response: {str(e)}"

    def generate_with_subset(
        self,
        query: str,
        all_chunks: list[DocumentChunk],
        mask: list[bool],
    ) -> str:
        """Generate answer with a subset of chunks (used by SHAP)."""
        subset = [c for c, m in zip(all_chunks, mask) if m]
        if not subset:
            return ""
        return self.generate(query, subset)

    def _assess_confidence(
        self,
        query: str,
        answer: str,
        chunks: list[DocumentChunk],
    ) -> ConfidenceBreakdown:
        """Assess confidence using Gemini or heuristic fallback."""
        # Heuristic-based confidence scoring (works without Gemini too)
        if not chunks:
            return ConfidenceBreakdown(
                overall_score=0.0,
                retrieval_relevance=0.0,
                source_agreement=0.0,
                evidence_quality=0.0,
                coverage=0.0,
            )

        # Retrieval relevance: average similarity of top chunks
        relevance_scores = [c.relevance_score for c in chunks]
        retrieval_relevance = sum(relevance_scores) / len(relevance_scores)

        # Source agreement: low variance = high agreement
        if len(relevance_scores) > 1:
            mean_score = sum(relevance_scores) / len(relevance_scores)
            variance = sum((s - mean_score) ** 2 for s in relevance_scores) / len(relevance_scores)
            source_agreement = max(0.0, 1.0 - variance * 4)
        else:
            source_agreement = 0.7

        # Evidence quality: based on evidence levels
        evidence_map = {"Level A": 1.0, "Level B": 0.8, "Level C": 0.6, "Not specified": 0.4}
        evidence_scores = [evidence_map.get(c.evidence_level, 0.5) for c in chunks]
        evidence_quality = sum(evidence_scores) / len(evidence_scores)

        # Coverage: how many unique sources contribute
        unique_titles = set(c.title for c in chunks)
        coverage = min(1.0, len(unique_titles) / 3.0)  # 3+ unique sources = full coverage

        # If Gemini is available, try to get a more nuanced assessment
        if self.gemini_client:
            try:
                sources_text = "\n".join(
                    f"- {c.title} ({c.source}, {c.evidence_level})"
                    for c in chunks
                )
                prompt = CONFIDENCE_ASSESSMENT_PROMPT.format(
                    query=query,
                    answer=answer,
                    sources=sources_text,
                )
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"temperature": 0.1, "max_output_tokens": 256},
                )

                text = response.text.strip()
                # Try to extract JSON
                if "{" in text:
                    json_str = text[text.index("{"):text.rindex("}") + 1]
                    scores = json.loads(json_str)
                    retrieval_relevance = float(scores.get("retrieval_relevance", retrieval_relevance))
                    source_agreement = float(scores.get("source_agreement", source_agreement))
                    evidence_quality = float(scores.get("evidence_quality", evidence_quality))
                    coverage = float(scores.get("coverage", coverage))
            except Exception as e:
                logger.warning(f"Gemini confidence assessment failed, using heuristics: {e}")

        # Clamp values
        retrieval_relevance = max(0.0, min(1.0, retrieval_relevance))
        source_agreement = max(0.0, min(1.0, source_agreement))
        evidence_quality = max(0.0, min(1.0, evidence_quality))
        coverage = max(0.0, min(1.0, coverage))

        overall = (
            retrieval_relevance * 0.35 +
            source_agreement * 0.2 +
            evidence_quality * 0.25 +
            coverage * 0.2
        )

        return ConfidenceBreakdown(
            overall_score=round(overall, 3),
            retrieval_relevance=round(retrieval_relevance, 3),
            source_agreement=round(source_agreement, 3),
            evidence_quality=round(evidence_quality, 3),
            coverage=round(coverage, 3),
        )

    def _extract_citations(self, answer: str, chunks: list[DocumentChunk]) -> list[Citation]:
        """Extract citation references from the generated answer."""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            marker = f"[Source {i}]"
            if marker in answer:
                # Find the sentence containing the citation
                sentences = answer.split(".")
                excerpt = ""
                for sentence in sentences:
                    if marker in sentence:
                        excerpt = sentence.strip().replace(marker, "").strip()
                        break

                citations.append(Citation(
                    index=i,
                    source=chunk.source,
                    title=chunk.title,
                    excerpt=excerpt[:200] if excerpt else chunk.content[:200],
                    relevance=chunk.relevance_score,
                ))

        # If no citations found in text, create from all chunks
        if not citations:
            for i, chunk in enumerate(chunks, 1):
                citations.append(Citation(
                    index=i,
                    source=chunk.source,
                    title=chunk.title,
                    excerpt=chunk.content[:200],
                    relevance=chunk.relevance_score,
                ))

        return citations

    async def query(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        enable_shap: bool = True,
    ) -> QueryResponse:
        """Execute the full RAG pipeline."""
        # Step 1: Retrieve
        chunks = self.retrieve(query, top_k, category_filter)

        if not chunks:
            return QueryResponse(
                query=query,
                answer="No relevant documents found in the knowledge base for this query. "
                       "Please try rephrasing your question or upload relevant documents.",
                confidence=ConfidenceBreakdown(
                    overall_score=0.0,
                    retrieval_relevance=0.0,
                    source_agreement=0.0,
                    evidence_quality=0.0,
                    coverage=0.0,
                ),
                citations=[],
                retrieved_chunks=chunks,
                shap_explanation=None,
                disclaimer=DISCLAIMER,
            )

        # Step 2: Generate
        answer = self.generate(query, chunks)

        # Step 3: Extract citations
        citations = self._extract_citations(answer, chunks)

        # Step 4: Assess confidence
        confidence = self._assess_confidence(query, answer, chunks)

        # Step 5: SHAP explanation (computed separately in the API layer)
        return QueryResponse(
            query=query,
            answer=answer,
            confidence=confidence,
            citations=citations,
            retrieved_chunks=chunks,
            shap_explanation=None,  # Populated by XAI engine
            disclaimer=DISCLAIMER,
        )
