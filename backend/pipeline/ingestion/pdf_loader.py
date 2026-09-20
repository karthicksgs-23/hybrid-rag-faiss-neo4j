from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Load a PDF and return LangChain Document objects.
    Each PDF page becomes one Document.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file must be a PDF.")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    print(f"PDF loaded successfully: {path.name}")
    print(f"Number of pages loaded: {len(documents)}")

    return documents


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print('python pdf_loader.py "path_to_pdf.pdf"')
        sys.exit(1)

    pdf_file = sys.argv[1]

    docs = load_pdf(pdf_file)

    if docs:
        print("\n--- First page preview ---")
        print(docs[0].page_content[:1000])
