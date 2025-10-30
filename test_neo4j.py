#!/usr/bin/env python3
"""
Quick Neo4j connection test
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing Neo4j Connection...")
print("=" * 50)

uri = os.getenv('KG_NEO4J_URI', 'bolt://localhost:7687')
username = os.getenv('KG_NEO4J_USERNAME', 'neo4j')
password = os.getenv('KG_NEO4J_PASSWORD')

print(f"URI: {uri}")
print(f"Username: {username}")
print(f"Password: {'*' * len(password) if password else 'NOT SET'}")
print()

if not password or password == 'your_neo4j_password_here':
    print("❌ ERROR: Password not set in .env file!")
    print("Please edit /home/user/NON_API_UNIFIED_RAG/.env")
    print("and set KG_NEO4J_PASSWORD to your actual Neo4j password")
    exit(1)

try:
    from neo4j import GraphDatabase

    print("Attempting connection...")
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        test = result.single()["test"]

        if test == 1:
            print("✅ SUCCESS! Neo4j connection works!")

            # Try to get entity count
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"✅ Found {count} entities in the database")
        else:
            print("❌ Unexpected response from Neo4j")

    driver.close()

except Exception as e:
    print(f"❌ ERROR: {e}")
    print()
    print("Common issues:")
    print("1. Wrong password - check your .env file")
    print("2. Neo4j not running - start your Neo4j server")
    print("3. Wrong port - verify Neo4j is on port 7687 (or update KG_NEO4J_URI)")
    exit(1)
