import re

from backend.database.neo4j_connection import get_neo4j_driver


def clean_relationship_type(relationship: str) -> str:
    """
    Convert relationship names into safe Neo4j relationship types.
    """

    relationship = relationship.upper().strip()

    relationship = re.sub(
        r"[^A-Z0-9_]+",
        "_",
        relationship
    )

    relationship = relationship.strip("_")

    if not relationship:
        relationship = "RELATED_TO"

    return relationship


def store_entities(
    session,
    entity_list,
    document_id: str
):
    """
    Store entities belonging to one uploaded document.
    """

    for entity in entity_list.entities:

        session.run(
            """
            MERGE (
                e:Entity {
                    document_id: $document_id,
                    name: $name
                }
            )

            SET e.type = $type,
                e.description = $description
            """,
            document_id=document_id,
            name=entity.name,
            type=entity.type,
            description=entity.description
        )

    print(
        f"Stored {len(entity_list.entities)} "
        f"entities for document {document_id}."
    )


def store_relationships(
    session,
    relationship_list,
    document_id: str
):
    """
    Store relationships only between entities
    belonging to the same uploaded document.
    """

    for relation in relationship_list.relationships:

        relationship_type = clean_relationship_type(
            relation.relationship
        )

        query = f"""
        MERGE (
            source:Entity {{
                document_id: $document_id,
                name: $source
            }}
        )

        MERGE (
            target:Entity {{
                document_id: $document_id,
                name: $target
            }}
        )

        MERGE (
            source
        )-[r:{relationship_type}]->(
            target
        )

        SET r.description = $description,
            r.document_id = $document_id
        """

        session.run(
            query,
            document_id=document_id,
            source=relation.source,
            target=relation.target,
            description=relation.description
        )

    print(
        f"Stored {len(relationship_list.relationships)} "
        f"relationships for document {document_id}."
    )


def store_graph(
    entity_list,
    relationship_list,
    document_id: str
):
    """
    Store one document's extracted Knowledge Graph
    in Neo4j Aura.
    """

    if not document_id:
        raise ValueError(
            "document_id is required for graph storage."
        )

    driver = get_neo4j_driver()

    try:

        with driver.session() as session:

            store_entities(
                session,
                entity_list,
                document_id
            )

            store_relationships(
                session,
                relationship_list,
                document_id
            )

        print(
            "Knowledge graph stored successfully "
            f"for document: {document_id}"
        )

    finally:

        driver.close()
