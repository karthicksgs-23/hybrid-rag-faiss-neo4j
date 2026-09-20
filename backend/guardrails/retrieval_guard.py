from dataclasses import dataclass


@dataclass
class RetrievalGuardResult:
    allowed: bool
    message: str = ""
    vector_ok: bool = False
    graph_ok: bool = False


MIN_VECTOR_DOCUMENTS = 1
MIN_GRAPH_RECORDS = 1


def check_vector_results(vector_documents):
    """
    Check whether FAISS returned usable documents.
    """

    if not vector_documents:
        return False

    usable_documents = [
        doc
        for doc in vector_documents
        if getattr(doc, "page_content", "").strip()
    ]

    return len(usable_documents) >= MIN_VECTOR_DOCUMENTS


def check_graph_results(graph_records):
    """
    Check whether Neo4j returned usable graph records.
    """

    if not graph_records:
        return False

    usable_records = []

    for record in graph_records:

        if not isinstance(record, dict):
            continue

        entity = record.get("entity")
        connected = record.get("connected_entity")

        if entity or connected:
            usable_records.append(record)

    return len(usable_records) >= MIN_GRAPH_RECORDS


def validate_retrieval(
    vector_documents,
    graph_records
):
    """
    Validate Hybrid RAG retrieval.

    For hybrid retrieval, we allow the request to continue
    when at least one retrieval source contains usable evidence.

    If both FAISS and Neo4j fail to return useful evidence,
    the answer should not be generated from the LLM.
    """

    vector_ok = check_vector_results(
        vector_documents
    )

    graph_ok = check_graph_results(
        graph_records
    )

    if not vector_ok and not graph_ok:

        return RetrievalGuardResult(
            allowed=False,
            message=(
                "No relevant evidence was found "
                "in the vector database or knowledge graph."
            ),
            vector_ok=False,
            graph_ok=False
        )

    return RetrievalGuardResult(
        allowed=True,
        message="Retrieval passed guardrails.",
        vector_ok=vector_ok,
        graph_ok=graph_ok
    )


if __name__ == "__main__":

    class FakeDocument:
        def __init__(self, text):
            self.page_content = text


    print("\nTEST 1: Vector evidence available")

    result = validate_retrieval(
        vector_documents=[
            FakeDocument(
                "Redis is used for caching and sessions."
            )
        ],
        graph_records=[]
    )

    print(result)


    print("\nTEST 2: Graph evidence available")

    result = validate_retrieval(
        vector_documents=[],
        graph_records=[
            {
                "entity": "Redis",
                "relationship": "USED_FOR",
                "connected_entity": "Caching"
            }
        ]
    )

    print(result)


    print("\nTEST 3: No evidence")

    result = validate_retrieval(
        vector_documents=[],
        graph_records=[]
    )

    print(result)
