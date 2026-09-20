from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split PDF documents into smaller chunks for embedding.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks
