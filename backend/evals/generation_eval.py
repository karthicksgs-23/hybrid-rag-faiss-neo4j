import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


class GenerationEvalResult(BaseModel):

    correctness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well the generated answer matches "
            "the expected answer."
        )
    )

    relevance: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How directly the generated answer "
            "answers the user's question."
        )
    )

    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well the generated answer is supported "
            "by the retrieved context."
        )
    )

    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Overall generation quality score."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation of the evaluation."
        )
    )


def get_evaluator():

    if not os.getenv("OPENAI_API_KEY"):

        raise ValueError(
            "OPENAI_API_KEY is missing from backend/.env"
        )

    llm = ChatOpenAI(
        model="gpt-5.6-luna"
    )

    return llm.with_structured_output(
        GenerationEvalResult
    )


def evaluate_generation(
    question: str,
    expected_answer: str,
    generated_answer: str,
    retrieved_context: str
):
    """
    Evaluate the final Hybrid RAG answer.
    """

    evaluator = get_evaluator()

    prompt = f"""
You are evaluating the final answer produced by
a Hybrid RAG system.

Evaluate using ONLY the information supplied below.

Give scores between 0.0 and 1.0.

CORRECTNESS:
Compare the generated answer with the expected answer.

RELEVANCE:
Determine whether the generated answer directly answers
the user's question.

FAITHFULNESS:
Determine whether the factual claims in the generated
answer are supported by the retrieved context.

OVERALL SCORE:
Provide an overall assessment based on correctness,
relevance, and faithfulness.

Do not answer the user's question yourself.
Do not use outside knowledge.

USER QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

RETRIEVED CONTEXT:
{retrieved_context}

GENERATED ANSWER:
{generated_answer}
"""

    return evaluator.invoke(
        prompt
    )


def display_generation_result(
    result: GenerationEvalResult
):

    print("\n================================")
    print("GENERATION EVALUATION")
    print("================================")

    print(
        f"Correctness:  {result.correctness}"
    )

    print(
        f"Relevance:    {result.relevance}"
    )

    print(
        f"Faithfulness: {result.faithfulness}"
    )

    print(
        f"Overall:      {result.overall_score}"
    )

    print(
        f"Reason:       {result.reason}"
    )


if __name__ == "__main__":

    question = (
        "What database is used for relational data?"
    )

    expected_answer = (
        "PostgreSQL is used for relational data."
    )

    generated_answer = (
        "PostgreSQL is used for relational data "
        "such as users, playlists, and subscriptions."
    )

    retrieved_context = """
PostgreSQL is used for relational data
including users, playlists, subscriptions,
and related application data.
"""

    result = evaluate_generation(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        retrieved_context=retrieved_context
    )

    display_generation_result(
        result
    )
