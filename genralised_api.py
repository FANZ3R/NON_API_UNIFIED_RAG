"""
unified_search_service.py

Framework-agnostic unified search & RAG orchestration module.
Drop-in usage: Import UnifiedSearchService and adapters into your FastAPI app,
or run the example at bottom which wires it into FastAPI.

Replace the adapter stubs (VectorSearcherAdapter, KnowledgeGraphAdapter) with
your real clients (Qdrant, Neo4j, etc.).
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Protocol, Dict, Any
from abc import abstractmethod
from pydantic import BaseModel, Field
import time
import logging
import asyncio

# Logging
logger = logging.getLogger("unified_search")
logging.basicConfig(level=logging.INFO)


# ----------------------------
# Domain / DTO models (Pydantic)
# ----------------------------
class VectorResult(BaseModel):
    score: float
    field_name: Optional[str] = None
    header: Optional[str] = None
    content: Optional[str] = None
    record_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class KGResult(BaseModel):
    confidence: float
    relationship_text: str
    strategy: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class UnifiedResponse(BaseModel):
    prompt: str
    response_text: str
    vector_results: List[VectorResult] = []
    kg_results: List[KGResult] = []
    search_time_seconds: float


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    vector_limit: int = 5
    kg_limit: int = 8
    include_detailed: bool = True


class StatusInfo(BaseModel):
    vector_connected: bool
    kg_connected: bool
    vector_points: Optional[int] = None
    vector_collection: Optional[str] = None
    llm_key_set: bool = False


# ----------------------------
# Interfaces / Protocols
# ----------------------------
class IVectorSearcher(Protocol):
    """Vector searcher interface - implement this for Qdrant, Pinecone, etc."""
    connected: bool

    @abstractmethod
    async def search(self, query: str, limit: int) -> List[VectorResult]:
        ...

    @abstractmethod
    async def test(self) -> List[VectorResult]:
        ...

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        ...


class IKnowledgeGraphSearcher(Protocol):
    """Knowledge graph searcher interface - implement this for Neo4j, etc."""
    connected: bool

    @abstractmethod
    async def search(self, query: str, limit: int) -> List[KGResult]:
        ...

    @abstractmethod
    async def test(self) -> List[KGResult]:
        ...

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        ...


# ----------------------------
# Example Adapter stubs (replace with real implementations)
# ----------------------------
class VectorSearcherAdapter:
    """
    Replace methods with actual Qdrant / semantic-search code.
    Keep semantics: connected bool, async search returning List[VectorResult].
    """
    def __init__(self, collection_name: str = "default"):
        self.connected = True
        self.total_points = 0
        self.collection_name = collection_name

    async def search(self, query: str, limit: int) -> List[VectorResult]:
        # Replace with real embedding + vector DB lookup
        await asyncio.sleep(0)  # keep it non-blocking
        # dummy result
        return [
            VectorResult(
                score=0.93 - i * 0.01,
                field_name="content",
                header=f"Doc {i}",
                content=f"Simulated snippet for query '{query}' (result {i})",
                record_id=str(i),
            ) for i in range(min(limit, 3))
        ]

    async def test(self) -> List[VectorResult]:
        return await self.search("test query", limit=1)

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "total_points": self.total_points,
            "collection_name": self.collection_name,
        }


class KnowledgeGraphAdapter:
    """
    Replace with Neo4j driver usage. Return KGResult objects.
    """
    def __init__(self):
        self.connected = True

    async def search(self, query: str, limit: int) -> List[KGResult]:
        await asyncio.sleep(0)
        return [
            KGResult(
                confidence=0.8 - i * 0.05,
                relationship_text=f"Entity A -[{i}]-> Entity B for '{query}'",
                strategy="path-finding",
            ) for i in range(min(limit, 2))
        ]

    async def test(self) -> List[KGResult]:
        return await self.search("test", limit=1)

    def get_status(self) -> Dict[str, Any]:
        return {"connected": self.connected}


# ----------------------------
# Orchestrator / Service (Business Logic)
# ----------------------------
class UnifiedSearchService:
    """
    Orchestrates vector + KG searches and generates unified responses.
    Single responsibility: business logic (no HTTP, no UI).
    """
    def __init__(self, vector_searcher: IVectorSearcher, kg_searcher: IKnowledgeGraphSearcher, llm_key_present: bool = False):
        self.vector_searcher = vector_searcher
        self.kg_searcher = kg_searcher
        self.llm_key_present = llm_key_present

    async def test_connections(self) -> Dict[str, Any]:
        out = {}
        try:
            out['vector'] = await self.vector_searcher.test() if self.vector_searcher.connected else []
        except Exception as e:
            out['vector_error'] = str(e)
            logger.exception("Vector test failed")

        try:
            out['kg'] = await self.kg_searcher.test() if self.kg_searcher.connected else []
        except Exception as e:
            out['kg_error'] = str(e)
            logger.exception("KG test failed")

        return out

    async def search(self, req: SearchRequest) -> UnifiedResponse:
        """Main orchestration method. Non-blocking and safe for async frameworks."""
        start = time.perf_counter()
        vector_results: List[VectorResult] = []
        kg_results: List[KGResult] = []

        # Run both searches concurrently where possible
        tasks = []
        if getattr(self.vector_searcher, "connected", False):
            tasks.append(asyncio.create_task(self.vector_searcher.search(req.prompt, req.vector_limit)))
        else:
            logger.warning("Vector searcher not connected.")

        if getattr(self.kg_searcher, "connected", False):
            tasks.append(asyncio.create_task(self.kg_searcher.search(req.prompt, req.kg_limit)))
        else:
            logger.warning("KG searcher not connected.")

        # gather with error isolation
        results = []
        if tasks:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            # map back results by type heuristic (VectorResult vs KGResult)
            for item in gathered:
                if isinstance(item, Exception):
                    logger.exception("One search task failed", exc_info=item)
                    continue
                if len(item) == 0:
                    continue
                # simple type check on first element
                first = item[0]
                if isinstance(first, VectorResult):
                    vector_results = item
                elif isinstance(first, KGResult):
                    kg_results = item
                else:
                    # unknown - try to inspect shape
                    if hasattr(first, "score"):
                        vector_results = item
                    elif hasattr(first, "confidence"):
                        kg_results = item

        search_time = time.perf_counter() - start

        # Generate unified response text (RAG orchestration)
        response_text = self._generate_unified_response_text(req.prompt, vector_results, kg_results)

        return UnifiedResponse(
            prompt=req.prompt,
            response_text=response_text,
            vector_results=vector_results if req.include_detailed else [],
            kg_results=kg_results if req.include_detailed else [],
            search_time_seconds=search_time
        )

    def _generate_unified_response_text(self, prompt: str, vector_results: List[VectorResult], kg_results: List[KGResult]) -> str:
        """
        Lightweight RAG composition. For production, replace with LLM call (LLM call should be extracted behind an interface).
        Keep it deterministic and testable.
        """
        parts = []
        parts.append(f"Answer (composed) for: {prompt}")

        if vector_results:
            parts.append("\nVector evidence:")
            for r in vector_results[:5]:
                parts.append(f"- [{r.score:.3f}] {r.header or r.field_name}: { (r.content or '')[:200] }")

        if kg_results:
            parts.append("\nKG evidence:")
            for r in kg_results[:5]:
                parts.append(f"- [{r.confidence:.2f}] {r.relationship_text} (strategy={r.strategy})")

        # If no sources, fallback
        if not vector_results and not kg_results:
            parts.append("\nNo evidence found in vector store or KG. Fallback: generic answer.")
            parts.append("You may want to check your data ingestion pipeline or increase search limits.")
        else:
            parts.append("\nSynthesis: Combine the above evidence to answer the user's prompt.")

        # NOTE: In prod, here we would call an LLM with a prompt template that includes the evidence,
        # and return the LLM-generated answer.

        return "\n".join(parts)

    def get_status(self) -> StatusInfo:
        vs = self.vector_searcher.get_status() if self.vector_searcher else {}
        ks = self.kg_searcher.get_status() if self.kg_searcher else {}
        return StatusInfo(
            vector_connected=bool(vs.get("connected", False)),
            kg_connected=bool(ks.get("connected", False)),
            vector_points=vs.get("total_points"),
            vector_collection=vs.get("collection_name"),
            llm_key_set=bool(self.llm_key_present)
        )


# ----------------------------
# Example FastAPI wiring (OpenAPI-ready)
# ----------------------------
# This section is optional — remove if you embed into an existing app.
try:
    from fastapi import FastAPI, Depends, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Unified RAG Search API", version="0.1.0")

    # Create adapters (in real apps prefer DI container)
    _vector_adapter = VectorSearcherAdapter(collection_name="test_business_data")
    _kg_adapter = KnowledgeGraphAdapter()
    _service = UnifiedSearchService(_vector_adapter, _kg_adapter, llm_key_present=False)

    # CORS & middleware example
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/search", response_model=UnifiedResponse)
    async def search_endpoint(req: SearchRequest):
        # input validation handled by Pydantic
        try:
            return await _service.search(req)
        except Exception as ex:
            logger.exception("Search failed")
            raise HTTPException(status_code=500, detail=str(ex))

    @app.get("/status", response_model=StatusInfo)
    async def status_endpoint():
        return _service.get_status()

    @app.post("/test-connections")
    async def test_connections():
        out = await _service.test_connections()
        return JSONResponse(content={"result_counts": {k: len(v) if isinstance(v, list) else "error" for k, v in out.items()}, "raw": out})

    # Minimal chat-like endpoint saving session is left to the caller
    # For stateless servers: clients should manage session/message history.
    # If you want server-side sessions, store messages in Redis / DB keyed by session id.

except Exception:
    # If FastAPI not available during import, ignore - file still imports fine.
    pass


# ----------------------------
# Design notes & trade-offs
# ----------------------------
"""
Design choices:
- Protocols (typing.Protocol) used to express dependencies (IVectorSearcher / IKnowledgeGraphSearcher).
  This follows Dependency Inversion: high-level UnifiedSearchService depends on abstractions.
- UnifiedSearchService is pure business logic: no HTTP, no UI, easy to unit-test.
- Adapters are responsible for integration details (Qdrant, Neo4j); keep them single-responsibility.
- Pydantic models for request/response ensure fast validation and OpenAPI generation.
- Async design: searches are run concurrently with asyncio.gather to reduce latency.

Trade-offs:
- Simplicity vs completeness: this example uses simplistic stub adapters and a naive RAG compose method.
  For production you'd:
    - Add robust error handling, retries, timeouts.
    - Add observability (metrics, tracing).
    - Replace _generate_unified_response_text with an LLM-call behind an ILLM interface.
    - Add rate limiting and auth at API gateway or FastAPI layer.
- Session state: this service is stateless. If you need persistent session-based chat, store conversation state
  in Redis or a DB and pass history into the LLM / reranker.
- Testing: keep adapters small so you can inject test doubles (mocks/fakes) easily.

Security:
- Never expose raw internal error traces in production.
- Sanitize any logs that might contain PII or confidential data.
- Protect the LLM API key with vaults / environment variables.

Scalability:
- For very high throughput, separate responsibilities into microservices:
    - ingestion pipeline (embeddings -> vector DB)
    - search service (this module)
    - LLM service (wraps calls to local Ollama/Llama 3)
    - KG service (Neo4j / query microservice)
- Use batching for embedding and search calls where possible to reduce overhead.

"""

# If run directly, launch a development Uvicorn server (optional).
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("genralised_api:app", host="0.0.0.0", port=8000, reload=True)
    except Exception:
        logger.info("Uvicorn not available or failed to launch; module loaded successfully.")
