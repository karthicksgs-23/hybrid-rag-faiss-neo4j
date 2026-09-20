from backend.evals.rag_evaluator import evaluate_rag
from pathlib import Path
import shutil

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File
)
from pydantic import BaseModel

from backend.pipeline.question.hybrid_rag import (
    hybrid_rag_answer
)

from backend.pipeline.ingestion.document_ingestion import (
    create_document_id,
    get_document_paths,
    ingest_document
)


app = FastAPI(
    title="Hybrid RAG API",
    description="Hybrid RAG using FAISS + Neo4j",
    version="1.0.0"
)


# =================================================
# RESPONSE / REQUEST MODELS
# =================================================

class QuestionRequest(BaseModel):
    document_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str

class EvalRequest(BaseModel):
    document_id: str
    question: str
    expected_answer: str
    expected_concepts: list[str]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    entities: int
    relationships: int
    message: str


# =================================================
# ROOT
# =================================================

@app.get("/")
def root():

    return {
        "message": "Hybrid RAG API is running"
    }


# =================================================
# HEALTH
# =================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =================================================
# PDF UPLOAD
# =================================================

@app.post(
    "/upload",
    response_model=UploadResponse
)
def upload_pdf(
    file: UploadFile = File(...)
):

    try:

        # -----------------------------------------
        # 1. Validate filename
        # -----------------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file has no filename."
            )


        # -----------------------------------------
        # 2. Allow PDF only
        # -----------------------------------------

        safe_filename = Path(
            file.filename
        ).name

        if not safe_filename.lower().endswith(
            ".pdf"
        ):

            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported."
            )


        # -----------------------------------------
        # 3. Generate unique document ID
        # -----------------------------------------

        document_id = create_document_id()


        # -----------------------------------------
        # 4. Create document-specific paths
        # -----------------------------------------

        paths = get_document_paths(
            document_id=document_id,
            filename=safe_filename
        )

        pdf_path = paths[
            "pdf_path"
        ]

        faiss_path = paths[
            "faiss_path"
        ]


        # -----------------------------------------
        # 5. Save uploaded PDF
        # -----------------------------------------

        with open(
            pdf_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        print(
            f"\nUploaded PDF saved to: "
            f"{pdf_path}"
        )


        # -----------------------------------------
        # 6. Run complete ingestion
        # -----------------------------------------

        result = ingest_document(
            pdf_path=str(pdf_path),
            document_id=document_id,
            faiss_path=str(faiss_path)
        )


        # -----------------------------------------
        # 7. Return document information
        # -----------------------------------------

        return UploadResponse(
            document_id=document_id,
            filename=result[
                "filename"
            ],
            chunks=result[
                "chunks"
            ],
            entities=result[
                "entities"
            ],
            relationships=result[
                "relationships"
            ],
            message=(
                "PDF uploaded and indexed "
                "successfully."
            )
        )


    except HTTPException:
        raise


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =================================================
# ASK
# =================================================

@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask_question(
    request: QuestionRequest
):

    try:

        if not request.question.strip():

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

       
        answer = hybrid_rag_answer(
            question=request.question,
            document_id=request.document_id
        )

        return QuestionResponse(
            answer=answer
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
    
@app.post("/evaluate")
def evaluate_hybrid_rag(request: EvalRequest):

    try:

        result = evaluate_rag(
            document_id=request.document_id,
            question=request.question,
            expected_answer=request.expected_answer,
            expected_concepts=request.expected_concepts
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )