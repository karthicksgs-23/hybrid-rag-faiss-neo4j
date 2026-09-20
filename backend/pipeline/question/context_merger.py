def merge_contexts(vector_context: str, graph_context: str) -> str:
    """
    Merge FAISS vector context and Neo4j graph context
    into one context block for the final LLM.
    """

    merged_context = f"""
==============================
VECTOR CONTEXT
==============================

{vector_context}


==============================
KNOWLEDGE GRAPH CONTEXT
==============================

{graph_context}
"""

    return merged_context.strip()


if __name__ == "__main__":

    sample_vector_context = """
PostgreSQL is used for relational data.
Redis is used for caching and sessions.
"""

    sample_graph_context = """
PostgreSQL --[STORES]-- Relational Data
Redis --[USED_FOR]-- Caching
Redis --[USED_FOR]-- Sessions
"""

    merged = merge_contexts(
        sample_vector_context,
        sample_graph_context
    )

    print(merged)
