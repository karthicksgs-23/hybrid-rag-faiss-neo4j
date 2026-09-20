from typing import List, Optional
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """
    One Hybrid RAG evaluation test case.
    """

    question: str = Field(
        description="Question to ask the uploaded document."
    )

    expected_answer: str = Field(
        description="Reference answer expected from the document."
    )

    expected_concepts: List[str] = Field(
        default_factory=list,
        description=(
            "Important concepts expected in retrieval "
            "or in the final answer."
        )
    )

    notes: Optional[str] = None


class EvalDataset(BaseModel):
    """
    Collection of evaluation cases for one document.
    """

    document_id: str

    cases: List[EvalCase]


def display_dataset(dataset: EvalDataset):

    print(
        f"Document ID: {dataset.document_id}"
    )

    print(
        f"Evaluation cases: {len(dataset.cases)}"
    )

    for index, case in enumerate(
        dataset.cases,
        start=1
    ):

        print(
            f"\nCase {index}"
        )

        print(
            f"Question: {case.question}"
        )

        print(
            f"Expected answer: "
            f"{case.expected_answer}"
        )

        print(
            f"Expected concepts: "
            f"{case.expected_concepts}"
        )


if __name__ == "__main__":

    sample_dataset = EvalDataset(
        document_id="example-document-id",
        cases=[
            EvalCase(
                question=(
                    "What database is used "
                    "for relational data?"
                ),
                expected_answer="PostgreSQL",
                expected_concepts=[
                    "PostgreSQL",
                    "relational data"
                ]
            ),

            EvalCase(
                question=(
                    "What is Redis used for?"
                ),
                expected_answer=(
                    "Redis is used for caching "
                    "and sessions."
                ),
                expected_concepts=[
                    "Redis",
                    "caching",
                    "sessions"
                ]
            )
        ]
    )

    display_dataset(
        sample_dataset
    )
