# Unified RAG Chatbot with Local LLM

A Retrieval-Augmented Generation (RAG) chatbot that combines **vector search** (Qdrant) and **knowledge graph** (Neo4j) capabilities with a **local Llama 3 model** running via Ollama.

## Features

- **Dual Search System**: Combines semantic vector search and knowledge graph relationships
- **Local LLM**: Uses Llama 3 running locally via Ollama (no API costs!)
- **Real-time Context**: Retrieves relevant information from both vector and graph databases
- **Streamlit UI**: Clean, interactive web interface

## Architecture

```
User Query
    ↓
[Vector Search] ← Qdrant + Sentence Transformers
    +
[Knowledge Graph Search] ← Neo4j
    ↓
[Context Assembly]
    ↓
[Local Llama 3] ← Ollama
    ↓
Response
```

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. Qdrant (Vector Database)
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Neo4j (Knowledge Graph)
```bash
docker pull neo4j:latest
docker run -p 7474:7474 -p 7687:7687 neo4j:latest
```

### 4. Ollama (Local LLM)
Download and install from: https://ollama.ai/download

Then pull Llama 3:
```bash
ollama pull llama3
```

Verify installation:
```bash
ollama list
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd NON_API_UNIFIED_RAG
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Verify setup**
```bash
python debug_script.py
```

## Configuration

Edit `.env` file with your settings:

```bash
# Vector Database
QDRANT_URL=http://localhost:6333
DEFAULT_COLLECTION_NAME=test_business_data
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Knowledge Graph
KG_NEO4J_URI=bolt://localhost:7687
KG_NEO4J_USERNAME=neo4j
KG_NEO4J_PASSWORD=your_password

# Local LLM (Ollama)
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama3
```

### Available Llama Models

You can use different Llama models based on your hardware:

- `llama3` or `llama3:8b` - Llama 3 8B (4.7GB, faster)
- `llama3:70b` - Llama 3 70B (40GB, better quality)
- `llama3.1` - Llama 3.1 with extended context window
- `llama3.2` - Latest Llama 3.2 model

Pull a model with:
```bash
ollama pull llama3:8b
```

## Usage

### Run the Streamlit App

```bash
streamlit run unified_chatbot.py
```

The app will open in your browser at `http://localhost:8501`

### Run Debug Tests

```bash
python debug_script.py
```

This will test:
- All Python dependencies
- Environment variables
- Qdrant connection
- Neo4j connection
- Ollama/LLM connection
- Sentence Transformers

## File Structure

```
NON_API_UNIFIED_RAG/
├── unified_chatbot.py          # Main Streamlit application
├── debug_script.py             # System diagnostics and testing
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── README.md                   # This file
└── ...
```

## How It Works

### 1. Query Processing
User submits a question through the Streamlit interface.

### 2. Parallel Search
- **Vector Search**: Query is embedded using Sentence Transformers and searched in Qdrant
- **Knowledge Graph Search**: Query is matched against entities and relationships in Neo4j

### 3. Context Assembly
Results from both searches are formatted into a structured context:
- Top 5 vector search results with scores
- Top 5 knowledge graph relationships with confidence

### 4. LLM Generation
The context and query are sent to the local Llama 3 model via Ollama API:
- Temperature: 0.3 (balanced creativity/accuracy)
- Max tokens: 1000
- Streaming: Disabled for simplicity

### 5. Response Display
The generated response is displayed along with detailed search results.

## Prompt Template

The system uses this prompt structure (kept from original):

```
You are an AI assistant with access to both content similarity search and knowledge graph data.

User Question: {query}

Available Information:
{context_text}

Instructions:
1. Use content similarity results for understanding concepts
2. Use knowledge graph relationships for factual connections
3. Provide a clear, informative response combining both sources when available

Provide a clear response:
```

## Troubleshooting

### Ollama Connection Issues

**Error**: "Cannot connect to Ollama"

**Solutions**:
1. Make sure Ollama is running (check system tray/menu bar)
2. Verify with: `curl http://localhost:11434/api/tags`
3. Check if model is installed: `ollama list`
4. Pull the model if missing: `ollama pull llama3`

### Model Not Found

**Error**: "Model 'llama3' not found"

**Solution**:
```bash
ollama pull llama3
```

### Slow Response Times

**Solutions**:
1. Use a smaller model: `llama3:8b` instead of `llama3:70b`
2. Reduce `num_predict` in the code (default: 1000)
3. Ensure sufficient RAM/VRAM available

### Qdrant Connection Failed

**Solutions**:
1. Start Qdrant: `docker start qdrant`
2. Check if running: `docker ps | grep qdrant`
3. Verify URL: `curl http://localhost:6333/collections`

### Neo4j Connection Failed

**Solutions**:
1. Start Neo4j: `docker start neo4j`
2. Check if running: `docker ps | grep neo4j`
3. Verify credentials in `.env` file
4. Access Neo4j Browser: `http://localhost:7474`

## Performance Tips

### Memory Management
- **8GB RAM**: Use `llama3:8b` model
- **16GB+ RAM**: Use `llama3:70b` for better quality
- **GPU**: Ollama automatically uses GPU if available (CUDA/Metal)

### Response Speed
- Reduce vector/KG search limits in sidebar
- Use smaller embedding models
- Cache frequently asked queries

## Migration from OpenRouter

This project was migrated from using OpenRouter API to local Llama 3:

**Changes made**:
- ✅ Removed OpenAI client dependency
- ✅ Replaced API calls with Ollama HTTP requests
- ✅ Updated environment variables (no API key needed)
- ✅ Kept all prompts and context generation identical
- ✅ Updated debug scripts and documentation

**Benefits**:
- ❌ No API costs
- ❌ No rate limits
- ❌ Data stays local/private
- ✅ Full control over model

## License

[Your License Here]

## Support

For issues or questions:
- Check the debug output: `python debug_script.py`
- Review logs in the Streamlit app
- Consult Ollama docs: https://ollama.ai/docs

## Acknowledgments

- **Ollama** for local LLM infrastructure
- **Llama 3** by Meta AI
- **Qdrant** for vector search
- **Neo4j** for knowledge graphs
- **Streamlit** for the UI framework
