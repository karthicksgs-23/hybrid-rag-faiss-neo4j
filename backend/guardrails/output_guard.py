import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


class OutputGuardResult(BaseModel):

    allowed: bool = Field(
        description=(
            "True only when the answer is supported "
            "by the retrieved context."
        )
    )

    grounded: bool = Field(
        description=(
            "Whether the important claims in the answer "
            "are grounded in the supplied context."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation for the decision."
        )
    )


def get_output_guard_model():
    """
    Create the LLM used to judge answer grounding.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from backend/.env"
        )

    llm = ChatOpenAI(
        model="gpt-5.6-luna"
    )

    return llm.with_structured_output(
        OutputGuardResult
    )


def validate_output(
    question: str,
    answer: str,
    context: str
) -> OutputGuardResult:
    """
    Check whether the final RAG answer is grounded
    in the retrieved Hybrid RAG context.
    """

    if not answer or not answer.strip():

        return OutputGuardResult(
            allowed=False,
            grounded=False,
            reason="The generated answer is empty."
        )

    if not context or not context.strip():

        return OutputGuardResult(
            allowed=False,
            grounded=False,
            reason="No retrieval context was available."
        )

    judge = get_output_guard_model()

    prompt = f"""
You are evaluating the output of a Hybrid RAG system.

Your job is to determine whether the GENERATED ANSWER
is supported by the RETRIEVED CONTEXT.

Rules:

1. Judge only against the supplied RETRIEVED CONTEXT.
2. Do not use outside knowledge.
3. Important factual claims in the answer must be supported
   by the context.
4. Minor wording differences are acceptable.
5. Reasonable summarization is acceptable.
6. If the answer introduces unsupported facts, mark it
   as not grounded.
7. If the context does not contain enough evidence for
   the answer, mark it as not grounded.
8. Do not judge writing style.
9. Do not answer the user's question yourself.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""

    result = judge.invoke(prompt)

    return result


if __name__ == "__main__":

    question = (
        "What is Redis used for?"
    )

    context = """
VECTOR CONTEXT:
Redis is used for caching and sessions.

KNOWLEDGE GRAPH CONTEXT:
Redis --[USED_FOR]--> Caching
Redis --[USED_FOR]--> Sessions
"""

    print("\nTEST 1: Grounded answer")

    answer = (
        "Redis is used for caching and sessions."
    )

    result = validate_output(
        question,
        answer,
        context
    )

    print(result)


    print("\nTEST 2: Unsupported answer")

    answer = (
        "Redis is used for caching, sessions, "
        "and video transcoding."
    )

    result = validate_output(
        question,
        answer,
        context
    )

    print(result)
