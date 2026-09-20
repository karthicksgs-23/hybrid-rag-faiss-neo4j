from backend.evals.retrieval_eval import (
    evaluate_retrieval,
)

from backend.evals.generation_eval import (
    evaluate_generation,
)

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

from backend.pipeline.question.hybrid_rag import (
    hybrid_rag_answer,
)


def evaluate_rag(
    document_id: str,
    question: str,
    expected_answer: str,
    expected_concepts: list[str]
):
    """
    Run complete Hybrid RAG evaluation.

    Measures:
    - FAISS retrieval
    - Neo4j retrieval
    - Hybrid retrieval
    - Answer correctness
    - Answer relevance
    - Answer faithfulness
    - Overall generation quality
    """

    print("\n================================")
    print("HYBRID RAG EVALUATION STARTED")
    print("================================")


    # ==================================
    # 1. RETRIEVAL EVALUATION
    # ==================================

    print("\n[1] Evaluating retrieval...")

    retrieval_result = evaluate_retrieval(
        document_id=document_id,
        question=question,
        expected_concepts=expected_concepts
    )


    # ==================================
    # 2. BUILD RETRIEVED CONTEXT
    # ==================================

    print("\n[2] Retrieving evaluation context...")

    vector_documents = retrieve_vector_context(
        question=question,
        document_id=document_id,
        k=4
    )

    vector_context = format_vector_context(
        vector_documents
    )

    graph_records = retrieve_graph_context(
        question=question,
        document_id=document_id,
        limit=30
    )

    graph_context = format_graph_context(
        graph_records
    )

    hybrid_context = merge_contexts(
        vector_context,
        graph_context
    )


    # ==================================
    # 3. GENERATE HYBRID RAG ANSWER
    # ==================================

    print("\n[3] Generating Hybrid RAG answer...")

    generated_answer = hybrid_rag_answer(
        question=question,
        document_id=document_id
    )


    # ==================================
    # 4. GENERATION EVALUATION
    # ==================================

    print("\n[4] Evaluating generated answer...")

    generation_result = evaluate_generation(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        retrieved_context=hybrid_context
    )


    # ==================================
    # 5. FINAL SCORE
    # ==================================

    hybrid_retrieval_score = (
        retrieval_result[
            "hybrid"
        ][
            "score"
        ]
    )

    generation_score = (
        generation_result.overall_score
    )

    final_score = round(
        (
            hybrid_retrieval_score
            +
            generation_score
        )
        / 2,
        3
    )


    return {
        "question": question,

        "expected_answer":
            expected_answer,

        "generated_answer":
            generated_answer,

        "retrieval": {
            "faiss_score":
                retrieval_result[
                    "faiss"
                ][
                    "score"
                ],

            "neo4j_score":
                retrieval_result[
                    "neo4j"
                ][
                    "score"
                ],

            "hybrid_score":
                hybrid_retrieval_score
        },

        "generation": {
            "correctness":
                generation_result.correctness,

            "relevance":
                generation_result.relevance,

            "faithfulness":
                generation_result.faithfulness,

            "overall":
                generation_result.overall_score,

            "reason":
                generation_result.reason
        },

        "final_score":
            final_score
    }


def display_rag_evaluation(
    result
):

    print("\n================================")
    print("COMPLETE HYBRID RAG EVALUATION")
    print("================================")

    print(
        "\nQuestion:"
    )

    print(
        result["question"]
    )


    print(
        "\nExpected Answer:"
    )

    print(
        result["expected_answer"]
    )


    print(
        "\nGenerated Answer:"
    )

    print(
        result["generated_answer"]
    )


    print(
        "\n--- RETRIEVAL ---"
    )

    print(
        "FAISS:",
        result["retrieval"][
            "faiss_score"
        ]
    )

    print(
        "Neo4j:",
        result["retrieval"][
            "neo4j_score"
        ]
    )

    print(
        "Hybrid:",
        result["retrieval"][
            "hybrid_score"
        ]
    )


    print(
        "\n--- GENERATION ---"
    )

    print(
        "Correctness:",
        result["generation"][
            "correctness"
        ]
    )

    print(
        "Relevance:",
        result["generation"][
            "relevance"
        ]
    )

    print(
        "Faithfulness:",
        result["generation"][
            "faithfulness"
        ]
    )

    print(
        "Generation Overall:",
        result["generation"][
            "overall"
        ]
    )


    print(
        "\nFINAL RAG SCORE:",
        result["final_score"]
    )

    print(
        "\nEvaluation reason:"
    )

    print(
        result["generation"][
            "reason"
        ]
    )
