"""Aplica um arquivo .cypher (um comando por ';') no Neo4j. Uso: python scripts/apply_schema.py schema.cypher"""
import os, sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
path = sys.argv[1] if len(sys.argv) > 1 else "schema.cypher"
text = "\n".join(l for l in open(path, encoding="utf-8") if not l.lstrip().startswith("//"))
commands = [c.strip() for c in text.split(";") if c.strip()]

driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
with driver:
    driver.verify_connectivity()
    for c in commands:
        driver.execute_query(c)
        print("ok:", c.splitlines()[0][:90])
