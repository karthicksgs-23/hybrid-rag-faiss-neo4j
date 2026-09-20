import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)


class Entity(BaseModel):
    name: str = Field(
        description="Canonical name of the entity"
    )

    type: str = Field(
        description="Entity type such as Technology, Service, Database, Feature, Component, Infrastructure, or Storage"
    )

    description: str = Field(
        description="Short description based only on the supplied source text"
    )


class EntityList(BaseModel):
    entities: List[Entity]


def get_entity_extraction_model():
    """
    Create an LLM that returns structured EntityList output.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from backend/.env"
        )

    llm = ChatOpenAI(
        model="gpt-5.6-luna"
    )

    structured_llm = llm.with_structured_output(EntityList)

    return structured_llm


def extract_entities(text: str) -> EntityList:
    """
    Extract entities from one PDF text chunk.
    """

    structured_llm = get_entity_extraction_model()

    prompt = f"""
Extract important knowledge-graph entities from the source text below.

Rules:
1. Use only information explicitly present in the source text.
2. Do not invent entities.
3. Extract meaningful architecture concepts such as:
   - technologies
   - databases
   - services
   - infrastructure
   - storage systems
   - application components
   - important features
4. Use a canonical concise entity name.
5. Avoid duplicates.
6. Keep descriptions short and factual.

SOURCE TEXT:
{text}
"""

    result = structured_llm.invoke(prompt)

    return result


def display_entities(entity_list: EntityList):
    print(f"Total entities: {len(entity_list.entities)}")

    for entity in entity_list.entities:
        print(
            f"- {entity.name} | "
            f"{entity.type} | "
            f"{entity.description}"
        )
