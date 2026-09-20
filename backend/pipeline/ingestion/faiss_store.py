from pathlib import Path

from langchain_community.vectorstores import FAISS


DEFAULT_FAISS_PATH = "backend/database/faiss_index"


def create_faiss_store(chunks, embeddings, save_path=DEFAULT_FAISS_PATH):
    """
    Create a FAISS vector database from document chunks
    and save it locally.
    """

    if not chunks:
        raise ValueError("No document chunks were provided.")

    print(f"Creating FAISS index from {len(chunks)} chunks...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(save_path).mkdir(parents=True, exist_ok=True)

    vector_store.save_local(save_path)

    print("FAISS vector database created successfully.")
    print(f"FAISS database saved to: {save_path}")

    return vector_store


def load_faiss_store(embeddings, save_path=DEFAULT_FAISS_PATH):
    """
    Load an existing FAISS vector database.
    """

    if not Path(save_path).exists():
        raise FileNotFoundError(
            f"FAISS database not found at: {save_path}"
        )

    vector_store = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"FAISS database loaded from: {save_path}")

    return vector_store
