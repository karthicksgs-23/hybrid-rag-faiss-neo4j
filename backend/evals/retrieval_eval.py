from typing import List

from backend.pipeline.question.vector_retriever import (
    retrieve_vector_context,
    format_vector_context,
)

from backend.pipeline.question.graph_retriever import (
    retrieve_graph_context,
    format_graph_context,
)


def concept_coverage(
    text: str,
    expected_concepts: List[str]
):
    """
    Measure how many expected concepts appear
    in the retrieved text.

    Returns a score between 0 and 1.
    """

    if not expected_concepts:
        return 1.0

    normalized_text = text.lower()

    matched = []

    missing = []

    for concept in expected_concepts:

        if concept.lower() in normalized_text:

            matched.append(
                concept
            )

        else:

            missing.append(
                concept
            )

    score = (
        len(matched)
        / len(expected_concepts)
    )

    return {
        "score": round(score, 3),
        "matched": matched,
        "missing": missing
    }


def evaluate_retrieval(
    document_id: str,
    question: str,
    expected_concepts: List[str],
    vector_k: int = 4,
    graph_limit: int = 30
):
    """
    Evaluate FAISS, Neo4j, and combined
    Hybrid RAG retrieval.
    """

    # ==================================
    # 1. FAISS
    # ==================================

    vector_documents = retrieve_vector_context(
        question=question,
        document_id=document_id,
        k=vector_k
    )

    vector_context = format_vector_context(
        vector_documents
    )


    # ==================================
    # 2. NEO4J
    # ==================================

    graph_records = retrieve_graph_context(
        question=question,
        document_id=document_id,
        limit=graph_limit
    )

    graph_context = format_graph_context(
        graph_records
    )


    # ==================================
    # 3. COMBINED HYBRID CONTEXT
    # ==================================

    hybrid_context = (
        vector_context
        + "\n"
        + graph_context
    )


    # ==================================
    # 4. SCORES
    # ==================================

    vector_score = concept_coverage(
        vector_context,
        expected_concepts
    )

    graph_score = concept_coverage(
        graph_context,
        expected_concepts
    )

    hybrid_score = concept_coverage(
        hybrid_context,
        expected_concepts
    )


    return {
        "question": question,

        "expected_concepts":
            expected_concepts,

        "faiss": vector_score,

        "neo4j": graph_score,

        "hybrid": hybrid_score
    }


def display_retrieval_result(
    result
):

    print("\n================================")
    print("RETRIEVAL EVALUATION")
    print("================================")

    print(
        "\nQuestion:",
        result["question"]
    )

    print(
        "\nExpected concepts:",
        result["expected_concepts"]
    )


    print("\nFAISS")

    print(
        "Score:",
        result["faiss"]["score"]
    )

    print(
        "Matched:",
        result["faiss"]["matched"]
    )

    print(
        "Missing:",
        result["faiss"]["missing"]
    )


    print("\nNEO4J")

    print(
        "Score:",
        result["neo4j"]["score"]
    )

    print(
        "Matched:",
        result["neo4j"]["matched"]
    )

    print(
        "Missing:",
        result["neo4j"]["missing"]
    )


    print("\nHYBRID")

    print(
        "Score:",
        result["hybrid"]["score"]
    )

    print(
        "Matched:",
        result["hybrid"]["matched"]
    )

    print(
        "Missing:",
        result["hybrid"]["missing"]
    )
