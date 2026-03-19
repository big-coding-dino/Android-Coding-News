import json
import logging
import time

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import DATA_FILE, EMBED_MODEL, SKIP_DOMAIN_TYPES, STORAGE_DIR
from fetchers import fetch_content

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NETWORK_STRATEGIES = {"trafilatura", "github_api", "youtube_transcript"}


def main():
    # LlamaIndex globals — set before any index operations
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None  # not needed during ingestion

    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=STORAGE_DIR)
    chroma_collection = chroma_client.get_or_create_collection("android_weekly")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build set of already-ingested URLs
    existing = chroma_collection.get(include=["metadatas"])
    ingested_urls = {m["url"] for m in existing["metadatas"] if m and "url" in m}
    logger.info(f"Already ingested: {len(ingested_urls)} articles")

    # Load and filter articles
    with open(DATA_FILE) as f:
        data = json.load(f)
    articles = data["articles"]

    articles = [a for a in articles if a.get("domain_type") not in SKIP_DOMAIN_TYPES]
    new_articles = [a for a in articles if a["url"] not in ingested_urls]
    logger.info(f"Skipped (domain): {len(data['articles']) - len(articles)} | New to ingest: {len(new_articles)}")

    if not new_articles:
        logger.info("Nothing new to ingest.")
        return

    # Fetch and build documents
    docs = []
    for i, article in enumerate(new_articles):
        url = article["url"]
        strategy = article.get("fetch_strategy")
        logger.info(f"[{i+1}/{len(new_articles)}] {strategy} — {article['title'][:60]}")

        content = fetch_content(article)
        has_full_content = content is not None

        if not has_full_content:
            content = article.get("description", "")
            if not content:
                logger.warning(f"  No content or description, skipping: {url}")
                continue

        doc = Document(
            text=content,
            id_=url,
            metadata={
                "url": url,
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "issue": article.get("issue", 0),
                "date": article.get("date", ""),
                "category": article.get("category", ""),
                "domain_type": article.get("domain_type", ""),
                "has_full_content": has_full_content,
            },
        )
        docs.append(doc)

        if strategy in NETWORK_STRATEGIES:
            time.sleep(0.5)

    logger.info(f"Built {len(docs)} documents. Inserting into index...")

    if chroma_collection.count() == 0:
        VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        for doc in docs:
            index.insert(doc)

    logger.info(f"Done. Total indexed: {chroma_collection.count()}")


if __name__ == "__main__":
    main()
