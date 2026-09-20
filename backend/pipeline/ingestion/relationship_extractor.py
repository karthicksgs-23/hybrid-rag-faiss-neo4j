import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from backend.pipeline.ingestion.entity_extractor import EntityList


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)


class Relationship(BaseModel):
    source: str = Field(
        description="Name of the source entity"
    )

    relationship: str = Field(
        description="Short relationship name such as USES, STORES, SUPPORTS, PROVIDES, MANAGES, or CONNECTS_TO"
    )

    target: str = Field(
        description="Name of the target entity"
    )

    description: str = Field(
        default="",
        description="Short factual explanation of the relationship"
    )


class RelationshipList(BaseModel):
    relationships: List[Relationship]


def get_relationship_extraction_model():
    """
    Create an LLM that returns structured relationships.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from backend/.env"
        )

    llm = ChatOpenAI(
        model="gpt-5.6-luna"
    )

    return llm.with_structured_output(RelationshipList)


def extract_relationships(
    text: str,
    entity_list: EntityList
) -> RelationshipList:
    """
    Extract relationships between entities using the source text.
    """

    structured_llm = get_relationship_extraction_model()

    entities_text = "\n".join(
        [
            f"- {entity.name} ({entity.type})"
            for entity in entity_list.entities
        ]
    )

    prompt = f"""
Extract knowledge-graph relationships from the SOURCE TEXT.

AVAILABLE ENTITIES:
{entities_text}

Rules:
1. Use only relationships explicitly supported by the SOURCE TEXT.
2. Do not invent facts.
3. source and target should preferably be entities from AVAILABLE ENTITIES.
4. Use concise relationship names.
5. Write relationship names in UPPERCASE_WITH_UNDERSCORES.
6. Examples include:
   USES
   STORES
   SUPPORTS
   PROVIDES
   MANAGES
   INTEGRATES_WITH
   CONNECTS_TO
   POWERED_BY
   DEPENDS_ON
7. Do not create duplicate relationships.
8. Keep descriptions short and factual.

SOURCE TEXT:
{text}
"""

    result = structured_llm.invoke(prompt)

    return result


def display_relationships(
    relationship_list: RelationshipList
):
    print(
        f"Total relationships: "
        f"{len(relationship_list.relationships)}"
    )

    for relation in relationship_list.relationships:
        print(
            f"- {relation.source} "
            f"--[{relation.relationship}]--> "
            f"{relation.target}"
        )

        if relation.description:
            print(
                f"  {relation.description}"
            )
