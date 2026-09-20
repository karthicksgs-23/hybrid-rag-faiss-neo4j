from pathlib import Path
from uuid import UUID

from backend.pipeline.ingestion.embeddings import get_embedding_model
from backend.pipeline.ingestion.faiss_store import load_faiss_store


BASE_DIR = Path(__file__).resolve().parents[2]
FAISS_BASE_DIR = BASE_DIR / "database" / "faiss"


def get_document_faiss_path(document_id: str) -> Path:
    """
    Return the FAISS directory for one uploaded document.
    """

    try:
        UUID(document_id)
    except ValueError:
        raise ValueError(
            "Invalid document_id."
        )

    faiss_path = (
        FAISS_BASE_DIR / document_id
    )

    if not faiss_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found for document: {document_id}"
        )

    return faiss_path


def retrieve_vector_context(
    question: str,
    document_id: str,
    k: int = 4
):
    """
    Retrieve relevant chunks only from the FAISS index
    belonging to the selected uploaded document.
    """

    embeddings = get_embedding_model()

    faiss_path = get_document_faiss_path(
        document_id
    )

    vector_store = load_faiss_store(
        embeddings,
        save_path=str(faiss_path)
    )

    documents = vector_store.similarity_search(
        question,
        k=k
    )

    return documents


def format_vector_context(documents):
    """
    Convert retrieved FAISS documents into LLM context.
    """

    if not documents:
        return "No relevant vector context found."

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        page = document.metadata.get(
            "page",
            "unknown"
        )

        context_parts.append(
            f"""
--- Vector Result {index} ---
Page: {page}

{document.page_content}
"""
        )

    return "\n".join(
        context_parts
    )
