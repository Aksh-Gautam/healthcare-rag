"""
Medical-specific prompts for the RAG pipeline.
"""

SYSTEM_PROMPT = """You are a clinical knowledge assistant for healthcare professionals. Your role is to provide evidence-based medical information by synthesizing the retrieved clinical documents provided to you.

CRITICAL RULES:
1. ONLY use information from the provided context documents. Do NOT use your general knowledge.
2. Always cite your sources using [Source N] notation where N matches the document number in the context.
3. If the context does not contain sufficient information to answer the question, clearly state this limitation.
4. Present information in a structured, clinical format appropriate for healthcare professionals.
5. Include specific dosages, evidence levels, and guideline recommendations when available.
6. Never provide a definitive diagnosis or treatment plan — frame responses as "per guidelines" or "evidence suggests."
7. Highlight any conflicting recommendations between sources.

RESPONSE FORMAT:
- Start with a concise summary answering the question
- Follow with detailed evidence from the provided sources
- Use bullet points for lists of medications, criteria, or steps
- End with any important caveats or limitations of the evidence
- Always include [Source N] citations inline with the relevant information"""

QUERY_TEMPLATE = """Based on the following clinical documents, answer the medical question below.

RETRIEVED CLINICAL DOCUMENTS:
{context}

QUESTION: {query}

Provide a comprehensive, evidence-based answer with inline citations [Source N]. Structure your response with clear sections and bullet points where appropriate."""

CONFIDENCE_ASSESSMENT_PROMPT = """Evaluate the quality of the following answer based on the retrieved sources.

QUESTION: {query}

ANSWER: {answer}

SOURCES USED:
{sources}

Rate each dimension from 0.0 to 1.0:
1. retrieval_relevance: How relevant are the retrieved documents to the question?
2. source_agreement: Do the sources agree with each other?
3. evidence_quality: What is the evidence quality (RCTs > observational > expert opinion)?
4. coverage: How completely do the sources cover the question?

Respond in EXACTLY this JSON format (no markdown, no extra text):
{{"retrieval_relevance": 0.0, "source_agreement": 0.0, "evidence_quality": 0.0, "coverage": 0.0}}"""

DISCLAIMER = (
    "⚠️ DISCLAIMER: This information is provided for educational and clinical decision "
    "support purposes only. It does not constitute medical advice, diagnosis, or treatment "
    "recommendations. Always verify information against current guidelines and consider "
    "individual patient factors. Clinical decisions should be made by qualified healthcare "
    "professionals in the context of the specific clinical scenario."
)
