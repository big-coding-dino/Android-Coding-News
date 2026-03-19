import json
import os
from typing import Optional

import chromadb
import trafilatura
from fastapi import FastAPI, HTTPException, Query
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel

from config import DATA_FILE, EMBED_MODEL, STORAGE_DIR, TOP_K

app = FastAPI(title="Android Weekly RAG API")

DATA_FILE_PATH = DATA_FILE
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

_chroma = chromadb.PersistentClient(path=STORAGE_DIR)
_collection = _chroma.get_or_create_collection("android_weekly")
_vector_store = ChromaVectorStore(chroma_collection=_collection)
_storage_context = StorageContext.from_defaults(vector_store=_vector_store)
_index = VectorStoreIndex.from_vector_store(_vector_store, storage_context=_storage_context)


def load_articles():
    with open(DATA_FILE_PATH) as f:
        data = json.load(f)
        return data.get("articles", data)


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: dict
    tool_calls: list


@app.get("/health")
def health_check():
    articles = load_articles()
    return {"status": "ok", "articles_indexed": len(articles)}


@app.get("/articles")
def get_articles(
    issue: Optional[int] = Query(None),
    category: Optional[str] = Query(None),
    domain_type: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
):
    articles = load_articles()
    filtered = articles

    if issue is not None:
        filtered = [a for a in filtered if a.get("issue") == issue]
    if category:
        filtered = [a for a in filtered if a.get("category") == category]
    if domain_type:
        filtered = [a for a in filtered if a.get("domain_type") == domain_type]
    if q:
        q_lower = q.lower()
        filtered = [
            a for a in filtered
            if q_lower in a.get("title", "").lower() or q_lower in a.get("description", "").lower()
        ]

    return {"total": len(filtered), "articles": filtered}


@app.get("/issues")
def get_issues():
    articles = load_articles()
    issues_map = {}
    for a in articles:
        issue_num = a.get("issue")
        if issue_num not in issues_map:
            issues_map[issue_num] = {"issue": issue_num, "date": a.get("date"), "article_count": 0}
        issues_map[issue_num]["article_count"] += 1

    issues = sorted(issues_map.values(), key=lambda x: x["issue"], reverse=True)
    return {"issues": issues}


@app.get("/search")
def search(
    q: str = Query(..., description="Natural language query"),
    top_k: int = Query(TOP_K, description="Number of results"),
):
    nodes = _index.as_retriever(similarity_top_k=top_k).retrieve(q)
    seen, results = set(), []
    for n in nodes:
        url = n.metadata.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        results.append({
            "score": n.score,
            "title": n.metadata.get("title"),
            "description": n.metadata.get("description"),
            "url": url,
            "issue": n.metadata.get("issue"),
            "date": n.metadata.get("date"),
            "category": n.metadata.get("category"),
            "has_full_content": n.metadata.get("has_full_content"),
        })

    return {"query": q, "results": results}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Agentic RAG chat endpoint.
    
    Pass prior conversation history for multi-turn chats:
    {
        "message": "what about Hilt?",
        "history": [
            {"role": "user", "content": "how does Koin differ from Hilt?"},
            {"role": "assistant", "content": "Koin is..."}
        ]
    }
    """
    from query import ask_agentic

    try:
        result = ask_agentic(
            question=request.message,
            history=request.history,
            verbose=False
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            tool_calls=result["tool_calls"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
