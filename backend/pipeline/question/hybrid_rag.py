import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.pipeline.question.vector_retriever import (
    retrieve_vector_context,
    format_vector_context,
)

from backend.pipeline.question.graph_retriever import (
    retrieve_graph_context,
    format_graph_context,
)

from backend.pipeline.question.context_merger import (
    merge_contexts,
)

from backend.guardrails.input_guard import (
    validate_input,
)

from backend.guardrails.retrieval_guard import (
    validate_retrieval,
)

from backend.guardrails.output_guard import (
    validate_output,
)


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)


def get_llm():

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from backend/.env"
        )

    return ChatOpenAI(
        model="gpt-5.6-luna"
    )


def hybrid_rag_answer(
    question: str,
    document_id: str,
    vector_k: int = 4,
    graph_limit: int = 30
):
    """
    Hybrid RAG for one uploaded document.
    """

    if not document_id:
        raise ValueError(
            "document_id is required."
        )


    # ==================================
    # 1. INPUT GUARDRAIL
    # ==================================

    input_result = validate_input(
        question
    )

    if not input_result.allowed:

        return (
            "Request blocked: "
            f"{input_result.message}"
        )

    print("\n[1] Input guardrail passed.")


    # ==================================
    # 2. FAISS RETRIEVAL
    # ==================================

    print(
        f"[2] Searching FAISS for "
        f"document: {document_id}"
    )

    vector_documents = retrieve_vector_context(
        question=question,
        document_id=document_id,
        k=vector_k
    )

    vector_context = format_vector_context(
        vector_documents
    )


    # ==================================
    # 3. NEO4J RETRIEVAL
    # ==================================

    print(
        f"[3] Searching Neo4j for "
        f"document: {document_id}"
    )

    graph_records = retrieve_graph_context(
        question=question,
        document_id=document_id,
        limit=graph_limit
    )

    graph_context = format_graph_context(
        graph_records
    )


    # ==================================
    # 4. RETRIEVAL GUARDRAIL
    # ==================================

    retrieval_result = validate_retrieval(
        vector_documents,
        graph_records
    )

    if not retrieval_result.allowed:

        return (
            "I could not find enough relevant evidence "
            "in the uploaded document to answer "
            "that question."
        )

    print("[4] Retrieval guardrail passed.")

    print(
        f"    FAISS evidence: "
        f"{retrieval_result.vector_ok}"
    )

    print(
        f"    Neo4j evidence: "
        f"{retrieval_result.graph_ok}"
    )


    # ==================================
    # 5. MERGE CONTEXT
    # ==================================

    hybrid_context = merge_contexts(
        vector_context,
        graph_context
    )

    print("[5] Hybrid context created.")


    # ==================================
    # 6. GENERATE FINAL ANSWER
    # ==================================

    llm = get_llm()

    prompt = f"""
You are the answer-generation component of a Hybrid RAG system.

The retrieved information belongs only to the currently
selected uploaded PDF.

The context contains:

1. VECTOR CONTEXT
   Retrieved from the selected PDF using FAISS.

2. KNOWLEDGE GRAPH CONTEXT
   Retrieved from the selected PDF's Neo4j graph.

Rules:

- Answer only from the supplied retrieved context.
- Do not use outside knowledge.
- Do not invent information.
- Use vector context for detailed document information.
- Use graph context for relationships between entities.
- Combine both forms of evidence when useful.
- If the context is insufficient, clearly state that.
- Return ONE final answer.
- Do not return separate FAISS and Neo4j answers.

DOCUMENT ID:
{document_id}

USER QUESTION:
{question}

HYBRID CONTEXT:
{hybrid_context}

FINAL ANSWER:
"""

    print("[6] Generating final answer...")

    response = llm.invoke(
        prompt
    )

    answer = response.content


    # ==================================
    # 7. OUTPUT GUARDRAIL
    # ==================================

    print("[7] Checking answer grounding...")

    output_result = validate_output(
        question=question,
        answer=answer,
        context=hybrid_context
    )

    if not output_result.allowed:

        print(
            "Output guardrail rejected answer:"
        )

        print(
            output_result.reason
        )

        return (
            "I found relevant information, but I could "
            "not produce an answer sufficiently grounded "
            "in the uploaded document."
        )

    print("[8] Output guardrail passed.")

    return answer
