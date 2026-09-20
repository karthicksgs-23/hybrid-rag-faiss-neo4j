import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="📄",
    layout="wide"
)


st.title("📄 Hybrid RAG")
st.caption(
    "Upload a PDF and ask questions using "
    "FAISS + Neo4j Knowledge Graph"
)


# =========================================================
# SESSION STATE
# =========================================================

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "filename" not in st.session_state:
    st.session_state.filename = None


# =========================================================
# BACKEND HEALTH CHECK
# =========================================================

def check_backend():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


# =========================================================
# UPLOAD PDF
# =========================================================

def upload_pdf(uploaded_file):

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf"
        )
    }

    response = requests.post(
        f"{API_URL}/upload",
        files=files,
        timeout=600
    )

    if response.status_code != 200:

        raise Exception(
            response.text
        )

    return response.json()


# =========================================================
# ASK HYBRID RAG
# =========================================================

def ask_question(
    document_id,
    question
):

    payload = {
        "document_id": document_id,
        "question": question
    }

    response = requests.post(
        f"{API_URL}/ask",
        json=payload,
        timeout=180
    )

    if response.status_code != 200:

        raise Exception(
            response.text
        )

    return response.json()["answer"]

def ask_question(
    document_id,
    question
):

    payload = {
        "document_id": document_id,
        "question": question
    }

    response = requests.post(
        f"{API_URL}/ask",
        json=payload,
        timeout=180
    )

    if response.status_code != 200:
        raise Exception(
            response.text
        )

    return response.json()["answer"]


# ADD THE NEW FUNCTION HERE
def evaluate_hybrid_rag(
    document_id,
    question,
    expected_answer,
    expected_concepts
):
    payload = {
        "document_id": document_id,
        "question": question,
        "expected_answer": expected_answer,
        "expected_concepts": expected_concepts
    }

    response = requests.post(
        f"{API_URL}/evaluate",
        json=payload,
        timeout=600
    )

    if response.status_code != 200:
        raise Exception(
            response.text
        )

    return response.json()


# =========================================================
# BACKEND STATUS
# =========================================================

if check_backend():

    st.success(
        "FastAPI backend connected"
    )

else:

    st.error(
        "FastAPI backend is not running."
    )

    st.code(
        "uvicorn backend.main:app --reload"
    )

    st.stop()


# =========================================================
# PDF UPLOAD SECTION
# =========================================================

st.subheader("1. Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    if st.button(
        "Process PDF",
        type="primary"
    ):

        try:

            with st.spinner(
                "Processing PDF..."
                "\n\nCreating FAISS vector database "
                "and Neo4j knowledge graph."
            ):

                result = upload_pdf(
                    uploaded_file
                )

                st.session_state.document_id = (
                    result["document_id"]
                )

                st.session_state.filename = (
                    result["filename"]
                )


            st.success(
                "PDF processed successfully."
            )


            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Chunks",
                    result["chunks"]
                )

            with col2:
                st.metric(
                    "Entities",
                    result["entities"]
                )

            with col3:
                st.metric(
                    "Relationships",
                    result["relationships"]
                )


        except Exception as error:

            st.error(
                f"Upload failed: {error}"
            )


# =========================================================
# ACTIVE DOCUMENT
# =========================================================

if st.session_state.document_id:

    st.divider()

    st.success(
        f"Active document: "
        f"{st.session_state.filename}"
    )

    with st.expander(
        "Document information"
    ):

        st.write(
            "Document ID:"
        )

        st.code(
            st.session_state.document_id
        )


# =========================================================
# QUESTION SECTION
# =========================================================

st.divider()

st.subheader(
    "2. Ask your PDF"
)


if not st.session_state.document_id:

    st.info(
        "Upload and process a PDF before "
        "asking questions."
    )

else:

    question = st.text_area(
        "Question",
        placeholder=(
            "Ask a question based on "
            "the uploaded PDF..."
        ),
        height=120
    )


    if st.button(
        "Ask Hybrid RAG",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Searching FAISS and Neo4j..."
                ):

                    answer = ask_question(
                        document_id=(
                            st.session_state.document_id
                        ),
                        question=question
                    )


                st.subheader(
                    "Answer"
                )

                st.write(
                    answer
                )


            except Exception as error:

                st.error(
                    f"Question failed: {error}"
                )

# =========================================================
# EVALUATION SECTION
# =========================================================

if st.session_state.document_id:

    st.divider()

    st.subheader("3. Evaluate Hybrid RAG")

    st.caption(
        "Evaluate retrieval and answer quality "
        "for the currently uploaded PDF."
    )

    eval_question = st.text_area(
        "Evaluation Question",
        placeholder=(
            "Example: What database is used "
            "for relational data?"
        ),
        key="eval_question"
    )

    expected_answer = st.text_area(
        "Expected Answer",
        placeholder=(
            "Example: PostgreSQL is used "
            "for relational data."
        ),
        key="expected_answer"
    )

    expected_concepts_text = st.text_input(
        "Expected Concepts",
        placeholder=(
            "Example: PostgreSQL, relational data"
        )
    )

    if st.button(
        "Run Evaluation",
        use_container_width=True
    ):

        if not eval_question.strip():

            st.warning(
                "Please enter an evaluation question."
            )

        elif not expected_answer.strip():

            st.warning(
                "Please enter the expected answer."
            )

        else:

            expected_concepts = [
                concept.strip()
                for concept in expected_concepts_text.split(",")
                if concept.strip()
            ]

            try:

                with st.spinner(
                    "Running Hybrid RAG evaluation..."
                ):

                    result = evaluate_hybrid_rag(
                        document_id=(
                            st.session_state.document_id
                        ),
                        question=eval_question,
                        expected_answer=expected_answer,
                        expected_concepts=expected_concepts
                    )


                st.success(
                    "Evaluation completed."
                )


                # -----------------------------------------
                # Retrieval scores
                # -----------------------------------------

                st.subheader(
                    "Retrieval Evaluation"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "FAISS",
                        result[
                            "retrieval"
                        ][
                            "faiss_score"
                        ]
                    )

                with col2:

                    st.metric(
                        "Neo4j",
                        result[
                            "retrieval"
                        ][
                            "neo4j_score"
                        ]
                    )

                with col3:

                    st.metric(
                        "Hybrid",
                        result[
                            "retrieval"
                        ][
                            "hybrid_score"
                        ]
                    )


                # -----------------------------------------
                # Generation scores
                # -----------------------------------------

                st.subheader(
                    "Generation Evaluation"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Correctness",
                        result[
                            "generation"
                        ][
                            "correctness"
                        ]
                    )

                with col2:

                    st.metric(
                        "Relevance",
                        result[
                            "generation"
                        ][
                            "relevance"
                        ]
                    )

                with col3:

                    st.metric(
                        "Faithfulness",
                        result[
                            "generation"
                        ][
                            "faithfulness"
                        ]
                    )

                with col4:

                    st.metric(
                        "Generation Overall",
                        result[
                            "generation"
                        ][
                            "overall"
                        ]
                    )


                # -----------------------------------------
                # Final score
                # -----------------------------------------

                st.subheader(
                    "Final Hybrid RAG Score"
                )

                st.metric(
                    "Final RAG Score",
                    result[
                        "final_score"
                    ]
                )


                # -----------------------------------------
                # Generated answer
                # -----------------------------------------

                with st.expander(
                    "Generated Answer"
                ):

                    st.write(
                        result[
                            "generated_answer"
                        ]
                    )


                # -----------------------------------------
                # Evaluation reason
                # -----------------------------------------

                with st.expander(
                    "Evaluation Reason"
                ):

                    st.write(
                        result[
                            "generation"
                        ][
                            "reason"
                        ]
                    )


            except Exception as error:

                st.error(
                    f"Evaluation failed: {error}"
                )
# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Hybrid RAG = FAISS semantic retrieval "
    "+ Neo4j knowledge graph + guardrails"
)
