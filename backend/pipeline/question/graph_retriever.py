import re

from backend.database.neo4j_connection import get_neo4j_driver
from backend.pipeline.ingestion.entity_extractor import extract_entities


STOP_WORDS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "for",
    "of",
    "to",
    "in",
    "on",
    "with",
    "and",
    "or",
    "does",
    "do",
    "did",
    "used",
    "use"
}


def extract_search_terms(question: str):
    """
    Build Neo4j search terms from both:
    1. LLM-extracted entities
    2. Keywords from the original question
    """

    # ----------------------------------
    # 1. Try entity extraction
    # ----------------------------------

    extracted = extract_entities(
        question
    )

    entity_names = [
        entity.name.strip()
        for entity in extracted.entities
        if entity.name.strip()
    ]


    # ----------------------------------
    # 2. Keyword fallback
    # ----------------------------------

    words = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_\-\.]+",
        question.lower()
    )

    keywords = [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 2
    ]


    # ----------------------------------
    # 3. Combine + remove duplicates
    # ----------------------------------

    search_terms = []

    for term in entity_names + keywords:

        normalized = term.strip()

        if (
            normalized
            and normalized.lower()
            not in [
                existing.lower()
                for existing in search_terms
            ]
        ):
            search_terms.append(
                normalized
            )

    print(
        f"Question entities: {entity_names}"
    )

    print(
        f"Neo4j search terms: {search_terms}"
    )

    return search_terms


def retrieve_graph_context(
    question: str,
    document_id: str,
    limit: int = 30
):
    """
    Retrieve graph context only from the selected
    uploaded document.

    Searches entity name, type, and description.
    """

    if not document_id:

        raise ValueError(
            "document_id is required "
            "for graph retrieval."
        )


    # ----------------------------------
    # 1. Build search terms
    # ----------------------------------

    search_terms = extract_search_terms(
        question
    )

    if not search_terms:
        return []


    # ----------------------------------
    # 2. Neo4j query
    # ----------------------------------

    driver = get_neo4j_driver()

    query = """
    MATCH (entity:Entity)

    WHERE
        entity.document_id = $document_id

    WITH
        entity,
        [
            term IN $search_terms
            WHERE
                toLower(
                    coalesce(entity.name, '')
                ) CONTAINS toLower(term)

                OR

                toLower(
                    coalesce(entity.type, '')
                ) CONTAINS toLower(term)

                OR

                toLower(
                    coalesce(entity.description, '')
                ) CONTAINS toLower(term)
        ] AS matched_terms

    WHERE size(matched_terms) > 0

    OPTIONAL MATCH
        (entity)-[relationship]-(connected:Entity)

    WHERE
        connected IS NULL
        OR connected.document_id = $document_id

    RETURN DISTINCT
        entity.name AS entity,
        entity.type AS entity_type,
        entity.description AS entity_description,

        type(relationship) AS relationship,

        connected.name AS connected_entity,
        connected.type AS connected_type,
        connected.description AS connected_description,

        size(matched_terms) AS match_score

    ORDER BY match_score DESC

    LIMIT $limit
    """

    try:

        with driver.session() as session:

            result = session.run(
                query,
                search_terms=search_terms,
                document_id=document_id,
                limit=limit
            )

            records = [
                record.data()
                for record in result
            ]

        return records

    finally:

        driver.close()


def format_graph_context(records):
    """
    Convert Neo4j records into text for Hybrid RAG.
    """

    if not records:

        return (
            "No relevant knowledge graph "
            "context found."
        )

    context_lines = []

    for record in records:

        entity = record.get(
            "entity"
        )

        entity_type = record.get(
            "entity_type"
        )

        description = record.get(
            "entity_description"
        )

        relationship = record.get(
            "relationship"
        )

        connected = record.get(
            "connected_entity"
        )

        connected_type = record.get(
            "connected_type"
        )

        if entity:

            context_lines.append(
                f"Entity: {entity} "
                f"(Type: {entity_type})"
            )

        if description:

            context_lines.append(
                f"Description: {description}"
            )

        if relationship and connected:

            context_lines.append(
                f"{entity} "
                f"--[{relationship}]-- "
                f"{connected} "
                f"(Type: {connected_type})"
            )

        context_lines.append("")

    return "\n".join(
        context_lines
    )
