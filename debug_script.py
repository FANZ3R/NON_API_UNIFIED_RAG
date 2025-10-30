"""
Debug script to test components individually
Updated to use local Llama 3 model via Ollama instead of OpenRouter API
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError as e:
        print(f"❌ Streamlit missing: {e}")
    
    try:
        import requests
        print("✅ Requests available")
    except ImportError as e:
        print(f"❌ Requests missing: {e}")
    
    try:
        from neo4j import GraphDatabase
        print("✅ Neo4j available")
    except ImportError as e:
        print(f"❌ Neo4j missing: {e}")
    
    try:
        from qdrant_client import QdrantClient
        print("✅ Qdrant client available")
    except ImportError as e:
        print(f"❌ Qdrant client missing: {e}")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers available")
    except ImportError as e:
        print(f"❌ SentenceTransformers missing: {e}")

def test_environment():
    """Test environment variables"""
    print("\n🔍 Testing environment variables...")

    required_vars = [
        'QDRANT_URL',
        'DEFAULT_COLLECTION_NAME',
        'LOCAL_EMBEDDING_MODEL',
        'KG_NEO4J_URI',
        'KG_NEO4J_USERNAME',
        'KG_NEO4J_PASSWORD'
    ]

    optional_vars = [
        'LOCAL_LLM_URL',
        'LOCAL_LLM_MODEL'
    ]

    for var in required_vars:
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var:
                print(f"✅ {var}: {'*' * 10}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")

    print("\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            default = 'http://localhost:11434' if 'URL' in var else 'llama3'
            print(f"ℹ️  {var}: Not set (will use default: {default})")

def test_vector_connection():
    """Test Qdrant connection - UPDATED VARIABLE NAMES"""
    print("\n🔍 Testing Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        
        url = os.getenv('QDRANT_URL', 'http://localhost:6333')  # Updated variable name
        client = QdrantClient(url=url)
        
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        print(f"✅ Connected to Qdrant at {url}")
        print(f"   Available collections: {collection_names}")
        
        target_collection = os.getenv('DEFAULT_COLLECTION_NAME', 'test_business_data')  # Updated variable name
        if target_collection in collection_names:
            print(f"✅ Target collection '{target_collection}' found")
            return True
        else:
            print(f"❌ Target collection '{target_collection}' not found")
            return False
            
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def test_kg_connection():
    """Test Neo4j connection"""
    print("\n🔍 Testing Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('KG_NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('KG_NEO4J_USERNAME', 'neo4j')
        password = os.getenv('KG_NEO4J_PASSWORD')
        
        if not password:
            print("❌ KG_NEO4J_PASSWORD not set")
            return False
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) as count LIMIT 1")
            count = result.single()["count"]
        
        driver.close()
        
        print(f"✅ Connected to Neo4j at {uri}")
        print(f"   Found {count:,} entities")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def test_llm_setup():
    """Test local LLM setup (Ollama)"""
    print("\n🔍 Testing local LLM setup (Ollama)...")

    try:
        import requests

        base_url = os.getenv('LOCAL_LLM_URL', 'http://localhost:11434')
        model_name = os.getenv('LOCAL_LLM_MODEL', 'llama3')

        # Test connection to Ollama
        response = requests.get(f"{base_url}/api/tags", timeout=5)

        if response.status_code == 200:
            print(f"✅ Connected to Ollama at {base_url}")

            models_data = response.json()
            models = models_data.get('models', [])
            model_names = [m.get('name', '') for m in models]

            print(f"   Available models: {', '.join(model_names)}")

            # Check if the configured model is available
            if model_name in model_names or any(model_name in name for name in model_names):
                print(f"✅ Model '{model_name}' is available")

                # Test a simple generation
                print(f"   Testing generation with {model_name}...")
                test_response = requests.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "Say 'test successful' and nothing else.",
                        "stream": False,
                        "options": {"num_predict": 10}
                    },
                    timeout=30
                )

                if test_response.status_code == 200:
                    result = test_response.json()
                    print(f"✅ Test generation successful: {result.get('response', '')[:50]}...")
                    return True
                else:
                    print(f"❌ Test generation failed: {test_response.status_code}")
                    return False
            else:
                print(f"❌ Model '{model_name}' not found in available models")
                print(f"   Run: ollama pull {model_name}")
                return False
        else:
            print(f"❌ Failed to connect to Ollama: HTTP {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Ollama at {base_url}")
        print("   Make sure Ollama is running:")
        print("   - Install: https://ollama.ai/download")
        print("   - Start: The Ollama app should be running")
        print("   - Verify: Run 'ollama list' in terminal")
        return False
    except Exception as e:
        print(f"❌ LLM setup test failed: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers specifically"""
    print("\n🔍 Testing SentenceTransformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        print(f"Loading model: {model_name}")
        
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded successfully")
        print(f"   Embedding dimension: {embedding_dim}")
        
        # Test a simple embedding
        test_text = "This is a test."
        embedding = model.encode(test_text)
        
        print(f"✅ Test embedding created: {len(embedding)} dimensions")
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformers test failed: {e}")
        return False

def main():
    print("🚀 UNIFIED CHATBOT DEBUGGING (Local LLM)")
    print("=" * 50)

    test_imports()
    test_environment()

    vector_ok = test_vector_connection()
    kg_ok = test_kg_connection()
    llm_ok = test_llm_setup()
    st_ok = test_sentence_transformers()

    print("\n📊 SUMMARY")
    print("=" * 20)
    print(f"Vector System: {'✅' if vector_ok else '❌'}")
    print(f"Knowledge Graph: {'✅' if kg_ok else '❌'}")
    print(f"Local LLM (Ollama): {'✅' if llm_ok else '❌'}")
    print(f"SentenceTransformers: {'✅' if st_ok else '❌'}")

    if vector_ok and kg_ok and llm_ok and st_ok:
        print("\n🎉 All systems ready! You can run:")
        print("streamlit run unified_chatbot.py")
    else:
        print("\n🔧 Fix the issues above before running the unified chatbot")

        if not vector_ok:
            print("\nVector system fixes:")
            print("- Make sure Qdrant is running: docker-compose up -d")
            print("- Check DEFAULT_COLLECTION_NAME matches your actual collection")

        if not kg_ok:
            print("\nKnowledge graph fixes:")
            print("- Make sure Neo4j is running")
            print("- Check KG_NEO4J_PASSWORD is correct")
            print("- Verify your data was imported successfully")

        if not llm_ok:
            print("\nLocal LLM fixes:")
            print("- Install Ollama: https://ollama.ai/download")
            print("- Pull Llama 3: ollama pull llama3")
            print("- Check if Ollama is running: ollama list")

        if not st_ok:
            print("\nSentenceTransformers fixes:")
            print("- Install: pip install sentence-transformers")
            print("- Install: pip install qdrant-client")

if __name__ == "__main__":
    main()