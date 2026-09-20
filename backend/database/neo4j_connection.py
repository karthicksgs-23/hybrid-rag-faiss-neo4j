import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_neo4j_driver():
    if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
        raise ValueError(
            "Neo4j credentials are missing. Check backend/.env"
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    return driver


def test_connection():
    driver = get_neo4j_driver()

    try:
        driver.verify_connectivity()

        with driver.session() as session:
            result = session.run(
                "RETURN 'Neo4j connection successful' AS message"
            )
            record = result.single()

            print(record["message"])

    finally:
        driver.close()


if __name__ == "__main__":
    test_connection()
