from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Load and return the Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print(f"Embedding model loaded: {MODEL_NAME}")

    return embeddings


if __name__ == "__main__":
    model = get_embedding_model()

    test_text = "Spotify uses microservices for its backend architecture."

    vector = model.embed_query(test_text)

    print(f"Vector dimensions: {len(vector)}")
    print("First 10 values:")
    print(vector[:10])
