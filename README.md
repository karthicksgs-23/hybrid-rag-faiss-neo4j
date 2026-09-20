\# Hybrid RAG with FAISS and Neo4j



A Hybrid Retrieval-Augmented Generation (RAG) application that combines \*\*vector search using FAISS\*\* with \*\*knowledge graph retrieval using Neo4j\*\* to answer questions from uploaded PDF documents.



The application includes:



\- PDF ingestion and chunking

\- Sentence-transformer embeddings

\- FAISS vector database

\- Neo4j knowledge graph

\- LLM-based entity extraction

\- LLM-based relationship extraction

\- Hybrid vector + graph retrieval

\- Input, retrieval, and output guardrails

\- RAG evaluation pipeline

\- FastAPI backend

\- Streamlit frontend

\- Document-specific data isolation



\---



\## Architecture



```text

&#x20;                        ┌─────────────────────┐

&#x20;                        │     Streamlit UI    │

&#x20;                        │                     │

&#x20;                        │  Upload PDF / Ask   │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │       FastAPI       │

&#x20;                        │       Backend       │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                        PDF Ingestion Pipeline

&#x20;                                   │

&#x20;                  ┌────────────────┴────────────────┐

&#x20;                  │                                 │

&#x20;                  ▼                                 ▼

&#x20;       ┌────────────────────┐            ┌────────────────────┐

&#x20;       │ Vector Pipeline    │            │ Knowledge Graph    │

&#x20;       │                    │            │ Pipeline            │

&#x20;       │ PDF Loader         │            │ Entity Extraction  │

&#x20;       │ Text Splitter      │            │ Relationship       │

&#x20;       │ Embeddings         │            │ Extraction         │

&#x20;       │ FAISS Storage      │            │ Neo4j Storage      │

&#x20;       └──────────┬─────────┘            └──────────┬─────────┘

&#x20;                  │                                 │

&#x20;                  └────────────────┬────────────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │   Hybrid Retrieval  │

&#x20;                        │                     │

&#x20;                        │ FAISS + Neo4j       │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │   Context Merger    │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │    LLM Generation   │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │     Guardrails      │

&#x20;                        │                     │

&#x20;                        │ Input Guard         │

&#x20;                        │ Retrieval Guard     │

&#x20;                        │ Output Guard        │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │    Final Answer     │

&#x20;                        └─────────────────────┘

```



\---



\## Technologies Used



| Component | Technology |

|---|---|

| Frontend | Streamlit |

| Backend API | FastAPI |

| PDF Loading | LangChain PyPDFLoader |

| Text Splitting | RecursiveCharacterTextSplitter |

| Embeddings | Sentence Transformers |

| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |

| Vector Database | FAISS |

| Knowledge Graph | Neo4j Aura |

| LLM Integration | LangChain OpenAI |

| Entity Extraction | Structured LLM Output |

| Relationship Extraction | Structured LLM Output |

| Validation | Pydantic |

| Evaluation | Custom Hybrid RAG Evaluation Pipeline |



\---



\## Project Structure



```text

hybrid-rag-faiss-neo4j/

│

├── frontend/

│   └── streamlit\_app.py

│

├── backend/

│   │

│   ├── main.py

│   ├── requirements.txt

│   │

│   ├── config/

│   │   ├── \_\_init\_\_.py

│   │   └── settings.py

│   │

│   ├── database/

│   │   └── neo4j\_connection.py

│   │

│   ├── guardrails/

│   │   ├── \_\_init\_\_.py

│   │   ├── input\_guard.py

│   │   ├── retrieval\_guard.py

│   │   └── output\_guard.py

│   │

│   ├── evals/

│   │   ├── \_\_init\_\_.py

│   │   ├── eval\_dataset.py

│   │   ├── retrieval\_eval.py

│   │   ├── generation\_eval.py

│   │   └── rag\_evaluator.py

│   │

│   └── pipeline/

│       │

│       ├── ingestion/

│       │   ├── \_\_init\_\_.py

│       │   ├── pdf\_loader.py

│       │   ├── text\_splitter.py

│       │   ├── embeddings.py

│       │   ├── faiss\_store.py

│       │   ├── entity\_extractor.py

│       │   ├── relationship\_extractor.py

│       │   ├── neo4j\_store.py

│       │   ├── graph\_ingestion.py

│       │   └── document\_ingestion.py

│       │

│       └── question/

│           ├── \_\_init\_\_.py

│           ├── vector\_retriever.py

│           ├── graph\_retriever.py

│           ├── context\_merger.py

│           ├── hybrid\_rag.py

│           └── rag\_chain.py

│

├── .gitignore

└── README.md

```



Generated files such as uploaded PDFs, FAISS indexes, environment variables, and cache files are excluded from Git.



\---



\## How the System Works



\### 1. PDF Upload



The user uploads a PDF through the Streamlit frontend.



The PDF is sent to the FastAPI backend through the `/upload` endpoint.



A unique `document\_id` is generated for each uploaded document.



\---



\### 2. PDF Loading



The PDF is loaded using LangChain's `PyPDFLoader`.



```text

PDF

&#x20;↓

Pages

&#x20;↓

LangChain Documents

```



\---



\### 3. Text Chunking



The document is split into smaller overlapping chunks using:



```text

RecursiveCharacterTextSplitter

```



Current configuration:



```text

chunk\_size = 1000

chunk\_overlap = 200

```



\---



\### 4. Embedding Generation



Each text chunk is converted into a vector representation using:



```text

sentence-transformers/all-MiniLM-L6-v2

```



The model produces \*\*384-dimensional embeddings\*\*.



\---



\### 5. FAISS Vector Storage



The embeddings and document chunks are stored in FAISS.



Each uploaded document receives its own vector index:



```text

backend/database/faiss/<document\_id>/

```



This provides document-specific vector retrieval.



\---



\### 6. Entity Extraction



An OpenAI-powered structured extraction pipeline identifies entities from each document chunk.



Examples of possible entities include:



```text

PostgreSQL

Redis

FastAPI

Neo4j

AWS

User Service

Authentication Service

```



Each extracted entity may include:



```text

name

type

description

```



\---



\### 7. Relationship Extraction



Relationships between extracted entities are identified using structured LLM output.



Example:



```text

User Service

&#x20;   |

&#x20;   | USES

&#x20;   ▼

PostgreSQL

```



The resulting graph is stored in Neo4j.



\---



\## Neo4j Knowledge Graph



Each entity is stored with its associated `document\_id`.



Example:



```text

(:Entity {

&#x20;   document\_id: "...",

&#x20;   name: "PostgreSQL",

&#x20;   type: "Database",

&#x20;   description: "Relational database"

})

```



Relationships are also associated with the same document.



This prevents information from different uploaded PDFs from being mixed during retrieval.



\---



\## Hybrid Retrieval



When a user asks a question, two retrieval pipelines run.



\### FAISS Retrieval



The question is embedded and compared against document chunks stored in FAISS.



```text

Question

&#x20;  ↓

Embedding

&#x20;  ↓

FAISS Similarity Search

&#x20;  ↓

Relevant Text Chunks

```



\### Neo4j Retrieval



Entities and important terms are extracted from the question.



Neo4j searches entity:



```text

name

type

description

```



and retrieves relevant entities and relationships.



The graph retriever also contains keyword fallback logic so retrieval can still work when the LLM does not identify a clear entity from the question.



\---



\## Hybrid Context



Results from FAISS and Neo4j are combined into one context.



```text

FAISS Context

&#x20;     +

Neo4j Context

&#x20;     ↓

Merged Hybrid Context

&#x20;     ↓

LLM

&#x20;     ↓

Final Answer

```



This allows the application to use both:



```text

Semantic similarity from FAISS

\+

Structured relationships from Neo4j

```



\---



\## Guardrails



The project contains three guardrail layers.



\### Input Guard



Checks user questions before retrieval.



It handles cases such as:



```text

Empty questions

Questions exceeding the maximum length

Potential prompt injection patterns

```



\### Retrieval Guard



Checks whether useful information was retrieved from FAISS or Neo4j.



If no supporting evidence is found, the application avoids generating an unsupported response.



\### Output Guard



The generated response is checked against the retrieved context.



The output guard evaluates whether the answer is grounded in the supplied document evidence.



\---



\## RAG Evaluation



The project contains an evaluation pipeline for both retrieval and generation.



\### Retrieval Evaluation



Measures whether expected concepts were found in:



```text

FAISS retrieval

Neo4j retrieval

Hybrid retrieval

```



\### Generation Evaluation



The generated answer is evaluated for:



```text

Correctness

Relevance

Faithfulness

Overall generation quality

```



\### Final RAG Score



The final evaluation combines hybrid retrieval performance with generation performance.



Conceptually:



```text

Final Score =

(Hybrid Retrieval Score + Generation Overall Score) / 2

```



\---



\## API Endpoints



\### Health Check



```http

GET /health

```



Used to confirm that the backend is running.



\---



\### Upload Document



```http

POST /upload

```



Uploads and processes a PDF.



Example response:



```json

{

&#x20; "document\_id": "example-document-id",

&#x20; "filename": "example.pdf",

&#x20; "chunks": 16,

&#x20; "entities": 100,

&#x20; "relationships": 50,

&#x20; "message": "Document processed successfully"

}

```



\---



\### Ask Question



```http

POST /ask

```



Example request:



```json

{

&#x20; "document\_id": "example-document-id",

&#x20; "question": "What database is used for relational data?"

}

```



The backend retrieves information from both FAISS and Neo4j and produces one grounded answer.



\---



\### Evaluate RAG



```http

POST /evaluate

```



Example request:



```json

{

&#x20; "document\_id": "example-document-id",

&#x20; "question": "What database is used for relational data?",

&#x20; "expected\_answer": "PostgreSQL is used for relational data.",

&#x20; "expected\_concepts": \[

&#x20;   "PostgreSQL",

&#x20;   "relational data"

&#x20; ]

}

```



The response contains retrieval metrics, generation metrics, and the final RAG evaluation score.



\---



\## Installation



Clone the repository:



```bash

git clone https://github.com/karthicksgs-23/hybrid-rag-faiss-neo4j.git

```



Move into the project directory:



```bash

cd hybrid-rag-faiss-neo4j

```



Create a Python virtual environment:



```bash

python -m venv venv

```



\### Windows PowerShell



Activate it with:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

pip install -r backend/requirements.txt

```



\---



\## Environment Variables



Create:



```text

backend/.env

```



Add the following variables:



```env

NEO4J\_URI=neo4j+s://YOUR\_NEO4J\_AURA\_INSTANCE

NEO4J\_USERNAME=neo4j

NEO4J\_PASSWORD=YOUR\_NEO4J\_PASSWORD



OPENAI\_API\_KEY=YOUR\_OPENAI\_API\_KEY

```



Never commit the `.env` file to GitHub.



The repository `.gitignore` is configured to exclude it.



\---



\## Neo4j Aura Setup



Create a Neo4j Aura database and obtain:



```text

NEO4J\_URI

NEO4J\_USERNAME

NEO4J\_PASSWORD

```



Add these values to:



```text

backend/.env

```



The application will create entities and relationships automatically during PDF ingestion.



\---



\## Running the Application



Two terminals are required.



\### Terminal 1 — FastAPI Backend



Activate the environment and run:



```powershell

uvicorn backend.main:app --reload

```



FastAPI will normally run at:



```text

http://127.0.0.1:8000

```



Swagger documentation is available at:



```text

http://127.0.0.1:8000/docs

```



\---



\### Terminal 2 — Streamlit Frontend



Activate the same virtual environment and run:



```powershell

streamlit run .\\frontend\\streamlit\_app.py

```



Streamlit will normally open at:



```text

http://localhost:8501

```



\---



\## Application Workflow



```text

1\. Start FastAPI

&#x20;       ↓

2\. Start Streamlit

&#x20;       ↓

3\. Upload a PDF

&#x20;       ↓

4\. Generate document ID

&#x20;       ↓

5\. Load and split PDF

&#x20;       ↓

6\. Generate embeddings

&#x20;       ↓

7\. Store vectors in FAISS

&#x20;       ↓

8\. Extract entities

&#x20;       ↓

9\. Extract relationships

&#x20;       ↓

10\. Store graph in Neo4j

&#x20;       ↓

11\. Ask a question

&#x20;       ↓

12\. Retrieve FAISS context

&#x20;       ↓

13\. Retrieve Neo4j context

&#x20;       ↓

14\. Merge contexts

&#x20;       ↓

15\. Generate answer

&#x20;       ↓

16\. Apply output grounding guard

&#x20;       ↓

17\. Return final answer

```



\---



\## Document Isolation



Every uploaded document receives a UUID.



Example:



```text

4935d182-b01d-485f-a476-703a5e49a8fa

```



FAISS storage:



```text

backend/database/faiss/<document\_id>/

```



Neo4j entities:



```text

document\_id = <document\_id>

```



Questions therefore retrieve information only from the selected document.



\---



\## Security



Sensitive information is not committed to GitHub.



The following are excluded using `.gitignore`:



```text

.env files

Uploaded PDFs

FAISS indexes

Python cache files

Virtual environments

IDE configuration

```



API keys and database passwords must always be supplied through environment variables.



\---



\## Example



Question:



```text

What database is used for relational data?

```



Hybrid retrieval can collect:



```text

FAISS:

Relevant text describing the application's database architecture.



Neo4j:

PostgreSQL → Database → relational data

```



Final response:



```text

PostgreSQL is used for relational data.

```



\---



\## Current Features



\- Hybrid FAISS + Neo4j RAG

\- PDF ingestion

\- Per-document UUID isolation

\- Semantic vector retrieval

\- Knowledge graph retrieval

\- Entity extraction

\- Relationship extraction

\- Keyword fallback graph search

\- Context merging

\- Prompt-injection input guard

\- Retrieval validation

\- Grounded-output validation

\- Retrieval evaluation

\- Generation evaluation

\- Final composite RAG score

\- FastAPI API

\- Streamlit user interface



\---



\## Future Improvements



Possible extensions include:



\- Persistent document registry

\- Multi-document querying

\- User authentication

\- Conversation history

\- Graph visualization

\- Reranking

\- Better graph entity resolution

\- Automated evaluation datasets

\- Docker deployment

\- Cloud deployment

\- Observability and tracing

\- Async ingestion

\- Background processing for large PDFs



\---



\## Repository



GitHub:



```text

https://github.com/karthicksgs-23/hybrid-rag-faiss-neo4j

```



\---



\## Author



\*\*Karthick\*\*



Hybrid RAG project combining semantic vector search and knowledge graph retrieval.

