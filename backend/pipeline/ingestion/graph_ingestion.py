import sys

from backend.pipeline.ingestion.pdf_loader import load_pdf
from backend.pipeline.ingestion.text_splitter import split_documents
from backend.pipeline.ingestion.entity_extractor import extract_entities
from backend.pipeline.ingestion.relationship_extractor import extract_relationships
from backend.pipeline.ingestion.neo4j_store import store_graph


def ingest_pdf_to_knowledge_graph(pdf_path: str):
    """
    Full PDF -> Knowledge Graph ingestion pipeline.
    """

    print("\n===================================")
    print("KNOWLEDGE GRAPH INGESTION STARTED")
    print("===================================\n")

    # 1. Load PDF
    documents = load_pdf(pdf_path)

    # 2. Split PDF
    chunks = split_documents(documents)

    print(f"\nProcessing {len(chunks)} chunks...\n")

    total_entities = 0
    total_relationships = 0

    # 3. Process every chunk
    for index, chunk in enumerate(chunks):

        print(
            f"\n---------- Chunk {index + 1}/{len(chunks)} ----------"
        )

        try:
            text = chunk.page_content

            # 4. Extract entities
            entities = extract_entities(text)

            print(
                f"Entities extracted: "
                f"{len(entities.entities)}"
            )

            # 5. Extract relationships
            relationships = extract_relationships(
                text,
                entities
            )

            print(
                f"Relationships extracted: "
                f"{len(relationships.relationships)}"
            )

            # 6. Store in Neo4j
            store_graph(
                entities,
                relationships
            )

            total_entities += len(
                entities.entities
            )

            total_relationships += len(
                relationships.relationships
            )

        except Exception as error:
            print(
                f"Error processing chunk "
                f"{index + 1}: {error}"
            )

    print("\n===================================")
    print("KNOWLEDGE GRAPH INGESTION COMPLETE")
    print("===================================")

    print(
        f"Total extracted entities: "
        f"{total_entities}"
    )

    print(
        f"Total extracted relationships: "
        f"{total_relationships}"
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python graph_ingestion.py "file.pdf"'
        )
        sys.exit(1)

    pdf_file = sys.argv[1]

    ingest_pdf_to_knowledge_graph(
        pdf_file
    )
