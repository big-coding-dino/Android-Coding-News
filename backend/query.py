"""
Agentic RAG query engine for Android Weekly.

The LLM always starts by reading the newsletter index, then decides whether
to do additional web research to fill gaps. Designed to grow into a chat.

Run directly:
    python query.py "your question here"

Import and call:
    from query import ask_agentic
    result = ask_agentic("how does Koin differ from Hilt?")
"""
import json
import logging
import sys

import chromadb
import trafilatura
from duckduckgo_search import DDGS
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import OpenAI

from config import (
    EMBED_MODEL,
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    STORAGE_DIR,
    TOP_K,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons — loaded once, reused across calls
# ---------------------------------------------------------------------------

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = None

_chroma = chromadb.PersistentClient(path=STORAGE_DIR)
_collection = _chroma.get_or_create_collection("android_weekly")
_vector_store = ChromaVectorStore(chroma_collection=_collection)
_storage_context = StorageContext.from_defaults(vector_store=_vector_store)
_index = VectorStoreIndex.from_vector_store(_vector_store, storage_context=_storage_context)
_llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_search_articles(query: str, top_k: int = TOP_K) -> str:
    """Search the Android Weekly vector index. Returns formatted article chunks."""
    nodes = _index.as_retriever(similarity_top_k=top_k).retrieve(query)
    seen, results = set(), []
    for n in nodes:
        url = n.metadata.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        results.append(n)

    if not results:
        return "No relevant articles found in the newsletter index."

    return "\n\n---\n\n".join(
        f"[NEWSLETTER] {n.metadata.get('title')} | Issue {n.metadata.get('issue')} | {n.metadata.get('date')}\n"
        f"URL: {n.metadata.get('url')}\n"
        f"{n.text[:800]}"
        for n in results
    )


def _tool_web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. Returns titles, snippets, and URLs."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"[WEB] {r['title']}\nURL: {r['href']}\n{r['body']}"
            for r in results
        )
    except Exception as e:
        return f"Web search failed: {e}"


def _tool_fetch_url(url: str) -> str:
    """Fetch and extract readable content from a URL using trafilatura."""
    try:
        html = trafilatura.fetch_url(url)
        if not html:
            return f"Could not fetch content from {url}"
        text = trafilatura.extract(html)
        if not text:
            return f"Could not extract readable content from {url}"
        # Cap at 3000 chars to keep context manageable
        return text[:3000] + ("..." if len(text) > 3000 else "")
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


# ---------------------------------------------------------------------------
# Tool registry — maps LLM tool names to implementations
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_articles",
            "description": (
                "Search the Android Weekly newsletter archive for articles on a topic. "
                "Always call this first before doing any web research. "
                "Returns article excerpts with issue numbers and dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or question to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for additional context, recent developments, or topics "
                "not covered in the newsletter archive. Use after searching articles first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch and read the full content of a specific URL — e.g. an article "
                "found via web_search or a newsletter article you want to read in full."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"],
            },
        },
    },
]

_TOOL_FN_MAP = {
    "search_articles": _tool_search_articles,
    "web_search": _tool_web_search,
    "fetch_url": _tool_fetch_url,
}

SYSTEM_PROMPT = """You are an expert Android development assistant powered by the Android Weekly newsletter archive.

Your research process:
1. ALWAYS start by calling search_articles to ground your answer in the newsletter.
2. If the articles raise questions or you need more depth, call web_search.
3. If a web result looks highly relevant, call fetch_url to read the full article.
4. Once you have enough context, write a comprehensive, well-structured answer.

Cite your sources clearly — distinguish between newsletter articles (issue number + date) and web sources (URL).
Be specific. Prefer depth over breadth."""


# ---------------------------------------------------------------------------
# Main agentic function
# ---------------------------------------------------------------------------

def ask_agentic(question: str, history: list = None, verbose: bool = True) -> dict:
    """
    Agentic RAG + web research. The LLM drives the research loop.

    Args:
        question: The user's question.
        history:  Prior conversation turns as [{"role": ..., "content": ...}].
                  Pass this for chat continuity across turns.
        verbose:  Print tool calls as they happen.

    Returns:
        {
            "answer": str,
            "sources": {"newsletter": [...], "web": [...]},
            "tool_calls": [{"tool": str, "args": dict}],
        }
    """
    # Build message list — system + prior history + current question
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    tool_call_log = []
    newsletter_sources = {}
    web_sources = []

    # Agentic loop
    while True:
        resp = _llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2000,
        )

        choice = resp.choices[0]
        msg = choice.message

        # Append assistant turn to history
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ] or None,
        })

        # No tool calls → final answer
        if not msg.tool_calls:
            break

        # Execute each tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            tool_call_log.append({"tool": name, "args": args})

            if verbose:
                arg_str = list(args.values())[0][:60]
                print(f"  [{name}] {arg_str}")

            result = _TOOL_FN_MAP[name](**args)

            # Track sources
            if name == "search_articles":
                for line in result.split("\n"):
                    if line.startswith("URL: "):
                        url = line[5:].strip()
                        newsletter_sources[url] = url
            elif name in ("web_search", "fetch_url"):
                for line in result.split("\n"):
                    if line.startswith("URL: "):
                        url = line[5:].strip()
                        if url not in web_sources:
                            web_sources.append(url)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return {
        "answer": msg.content,
        "sources": {
            "newsletter": list(newsletter_sources.keys()),
            "web": web_sources,
        },
        "tool_calls": tool_call_log,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the latest Android development trends?"

    print(f"\nQuestion: {q}")
    print("─" * 60)
    print("Research steps:")

    result = ask_agentic(q, verbose=True)

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")
    if result["sources"]["newsletter"]:
        print("Newsletter:")
        for url in result["sources"]["newsletter"]:
            print(f"  {url}")
    if result["sources"]["web"]:
        print("Web:")
        for url in result["sources"]["web"]:
            print(f"  {url}")

    print(f"\n[{len(result['tool_calls'])} tool calls made]")
