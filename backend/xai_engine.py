"""
SHAP-based Explainable AI Engine for the Healthcare RAG pipeline.
Computes document-level Shapley values showing each chunk's contribution
to the final answer quality.
"""
import logging
import numpy as np
from typing import Optional

import shap
from sklearn.metrics.pairwise import cosine_similarity

from backend.ingestion import DocumentIngestionPipeline
from backend.models import DocumentChunk, ShapExplanation

logger = logging.getLogger(__name__)


class XAIEngine:
    """
    SHAP explainer for RAG document attribution.

    Approach:
    1. After retrieval, we have N candidate document chunks.
    2. We define a scoring function that takes a binary mask vector (length N)
       indicating which chunks are included in context.
    3. For each combination, we compute the answer quality using embedding-based
       cosine similarity between the generated response and the query.
    4. SHAP KernelExplainer computes Shapley values showing each chunk's
       contribution to the answer quality.
    """

    def __init__(self, ingestion_pipeline: DocumentIngestionPipeline):
        self.pipeline = ingestion_pipeline
        # Cache to avoid redundant Gemini calls
        self._response_cache: dict[str, str] = {}

    def _get_embedding_score(
        self,
        query: str,
        response_text: str,
    ) -> float:
        """
        Score a response by computing cosine similarity between the
        response embedding and the query embedding.
        """
        if not response_text or not response_text.strip():
            return 0.0

        embeddings = self.pipeline.get_embeddings_for_texts([query, response_text])
        query_emb = np.array(embeddings[0]).reshape(1, -1)
        response_emb = np.array(embeddings[1]).reshape(1, -1)

        similarity = cosine_similarity(query_emb, response_emb)[0][0]
        return float(max(0.0, similarity))

    def _scoring_function(
        self,
        masks: np.ndarray,
        query: str,
        chunks: list[DocumentChunk],
        rag_engine,
    ) -> np.ndarray:
        """
        Scoring function for SHAP KernelExplainer.
        Takes a matrix of binary masks (each row is a mask) and returns
        an array of scores.
        """
        scores = []

        for mask_row in masks:
            mask = mask_row.astype(bool).tolist()

            # Create a cache key from the mask
            cache_key = str(mask)

            if cache_key in self._response_cache:
                response = self._response_cache[cache_key]
            else:
                # Generate response with this subset of chunks
                active_chunks = [c for c, m in zip(chunks, mask) if m]
                if not active_chunks:
                    response = ""
                else:
                    try:
                        response = rag_engine.generate(query, active_chunks)
                    except Exception as e:
                        logger.warning(f"Generation failed for mask {mask}: {e}")
                        response = ""
                self._response_cache[cache_key] = response

            # Score the response
            score = self._get_embedding_score(query, response)
            scores.append(score)

        return np.array(scores)

    def compute_shap_explanation(
        self,
        query: str,
        chunks: list[DocumentChunk],
        rag_engine,
        nsamples: int = 32,
    ) -> Optional[ShapExplanation]:
        """
        Compute SHAP values for document-level attribution.

        Args:
            query: The user's medical question
            chunks: Retrieved document chunks
            rag_engine: The RAG engine for generating responses
            nsamples: Number of SHAP samples (more = more accurate but slower)

        Returns:
            ShapExplanation with values for each chunk
        """
        if not chunks or len(chunks) < 2:
            if chunks:
                return ShapExplanation(
                    chunk_labels=[self._make_label(c) for c in chunks],
                    shap_values=[1.0],
                    base_value=0.0,
                    explanation_text="Only one document was retrieved, so it accounts for 100% of the answer.",
                )
            return None

        # Clear response cache for each new query
        self._response_cache = {}

        n_features = len(chunks)

        # Background: all zeros (no documents included)
        background = np.zeros((1, n_features))

        # Instance to explain: all ones (all documents included)
        instance = np.ones((1, n_features))

        # Create scoring function wrapper
        def score_fn(masks):
            return self._scoring_function(masks, query, chunks, rag_engine)

        try:
            # Create SHAP KernelExplainer
            explainer = shap.KernelExplainer(score_fn, background)

            # Compute SHAP values
            # Limit nsamples to keep computation tractable
            actual_nsamples = min(nsamples, 2 ** n_features)
            shap_values = explainer.shap_values(
                instance,
                nsamples=actual_nsamples,
                silent=True,
            )

            # shap_values shape: (1, n_features)
            values = shap_values[0].tolist() if hasattr(shap_values[0], 'tolist') else list(shap_values[0])
            base_value = float(explainer.expected_value)

            # Create human-readable labels
            labels = [self._make_label(c) for c in chunks]

            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                labels, values, base_value, chunks
            )

            return ShapExplanation(
                chunk_labels=labels,
                shap_values=[round(v, 4) for v in values],
                base_value=round(base_value, 4),
                explanation_text=explanation_text,
            )

        except Exception as e:
            logger.error(f"SHAP computation failed: {e}", exc_info=True)
            # Fallback: use relevance scores as proxy for SHAP values
            return self._fallback_explanation(chunks)

    def _make_label(self, chunk: DocumentChunk) -> str:
        """Create a short label for a document chunk."""
        title = chunk.title
        if len(title) > 40:
            title = title[:37] + "..."
        return f"{title} [{chunk.category}]"

    def _generate_explanation_text(
        self,
        labels: list[str],
        values: list[float],
        base_value: float,
        chunks: list[DocumentChunk],
    ) -> str:
        """Generate a human-readable explanation of the SHAP analysis."""
        # Sort by absolute SHAP value
        sorted_pairs = sorted(
            zip(labels, values, chunks),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        lines = ["**Document Contribution Analysis (SHAP)**\n"]
        lines.append(
            f"Base answer quality score (without any documents): {base_value:.3f}\n"
        )

        for label, value, chunk in sorted_pairs:
            direction = "↑ improves" if value > 0 else "↓ reduces"
            abs_val = abs(value)
            strength = "strongly" if abs_val > 0.1 else "moderately" if abs_val > 0.05 else "slightly"

            lines.append(
                f"• **{label}** ({chunk.evidence_level}): "
                f"{direction} answer quality by {abs_val:.3f} ({strength})"
            )

        # Top contributor summary
        if sorted_pairs:
            top_label, top_value, top_chunk = sorted_pairs[0]
            lines.append(
                f"\n**Key Insight:** The most influential source is "
                f"*{top_chunk.title}* from {top_chunk.source}, "
                f"which {'positively' if top_value > 0 else 'negatively'} "
                f"contributes {abs(top_value):.3f} to the answer quality."
            )

        return "\n".join(lines)

    def _fallback_explanation(self, chunks: list[DocumentChunk]) -> ShapExplanation:
        """
        Fallback explanation using relevance scores when SHAP fails.
        Normalized relevance scores serve as proxy attribution values.
        """
        labels = [self._make_label(c) for c in chunks]
        relevances = [c.relevance_score for c in chunks]

        # Normalize to sum to 1
        total = sum(relevances) if sum(relevances) > 0 else 1.0
        normalized = [r / total for r in relevances]

        return ShapExplanation(
            chunk_labels=labels,
            shap_values=[round(v, 4) for v in normalized],
            base_value=0.0,
            explanation_text=(
                "**Document Contribution Analysis (Relevance-based Fallback)**\n\n"
                "SHAP computation was not available. Contributions are estimated "
                "based on retrieval relevance scores.\n\n" +
                "\n".join(
                    f"• **{label}**: contribution weight {value:.3f}"
                    for label, value in zip(labels, normalized)
                )
            ),
        )
