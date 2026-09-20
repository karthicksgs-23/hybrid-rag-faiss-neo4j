import uuid
from pathlib import Path

from backend.pipeline.ingestion.pdf_loader import load_pdf
from backend.pipeline.ingestion.text_splitter import split_documents
from backend.pipeline.ingestion.embeddings import get_embedding_model
from backend.pipeline.ingestion.faiss_store import create_faiss_store
from backend.pipeline.ingestion.entity_extractor import extract_entities
from backend.pipeline.ingestion.relationship_extractor import extract_relationships
from backend.pipeline.ingestion.neo4j_store import store_graph


BASE_DIR = Path(__file__).resolve().parents[2]

UPLOAD_DIR = BASE_DIR / "uploads"
FAISS_DIR = BASE_DIR / "database" / "faiss"


def create_document_id() -> str:
    """
    Generate a unique ID for every uploaded PDF.
    """
    return str(uuid.uuid4())


def get_document_paths(
    document_id: str,
    filename: str
):
    """
    Create and return document-specific storage paths.
    """

    upload_folder = (
        UPLOAD_DIR / document_id
    )

    faiss_folder = (
        FAISS_DIR / document_id
    )

    upload_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_path = (
        upload_folder / filename
    )

    return {
        "pdf_path": pdf_path,
        "faiss_path": faiss_folder
    }


def ingest_document(
    pdf_path: str,
    document_id: str,
    faiss_path: str
):
    """
    Complete ingestion pipeline for one uploaded PDF.

    PDF
      -> Load
      -> Split
      -> FAISS
      -> Entity extraction
      -> Relationship extraction
      -> Neo4j
    """

    print("\n====================================")
    print("DOCUMENT INGESTION STARTED")
    print("====================================")

    print(
        f"Document ID: {document_id}"
    )

    print(
        f"PDF: {pdf_path}"
    )


    # ----------------------------------
    # 1. Load PDF
    # ----------------------------------

    documents = load_pdf(
        str(pdf_path)
    )


    # ----------------------------------
    # 2. Split document
    # ----------------------------------

    chunks = split_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )


    # Add document metadata to every chunk
    for chunk in chunks:

        chunk.metadata[
            "document_id"
        ] = document_id

        chunk.metadata[
            "filename"
        ] = Path(pdf_path).name


    # ----------------------------------
    # 3. CREATE FAISS VECTOR DATABASE
    # ----------------------------------

    print("\nCreating FAISS database...")

    embeddings = get_embedding_model()

    create_faiss_store(
        chunks=chunks,
        embeddings=embeddings,
        save_path=str(faiss_path)
    )

    print(
        "FAISS ingestion complete."
    )


    # ----------------------------------
    # 4. CREATE KNOWLEDGE GRAPH
    # ----------------------------------

    print(
        "\nCreating Neo4j Knowledge Graph..."
    )

    total_entities = 0
    total_relationships = 0


    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\nProcessing graph chunk "
            f"{index}/{len(chunks)}"
        )

        text = chunk.page_content

        try:

            # Entity extraction
            entities = extract_entities(
                text
            )

            print(
                f"Entities: "
                f"{len(entities.entities)}"
            )


            # Relationship extraction
            relationships = extract_relationships(
                text,
                entities
            )

            print(
                f"Relationships: "
                f"{len(relationships.relationships)}"
            )


            # Store only under this document_id
            store_graph(
                entity_list=entities,
                relationship_list=relationships,
                document_id=document_id
            )


            total_entities += len(
                entities.entities
            )

            total_relationships += len(
                relationships.relationships
            )


        except Exception as error:

            print(
                f"Graph extraction failed "
                f"for chunk {index}: {error}"
            )


    print("\n====================================")
    print("DOCUMENT INGESTION COMPLETE")
    print("====================================")

    print(
        f"Document ID: {document_id}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Extracted entities: "
        f"{total_entities}"
    )

    print(
        f"Extracted relationships: "
        f"{total_relationships}"
    )


    return {
        "document_id": document_id,
        "filename": Path(pdf_path).name,
        "chunks": len(chunks),
        "entities": total_entities,
        "relationships": total_relationships,
        "faiss_path": str(faiss_path)
    }
