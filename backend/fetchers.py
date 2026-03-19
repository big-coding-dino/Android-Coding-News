import logging
import urllib.parse

import requests
import trafilatura

logger = logging.getLogger(__name__)


def fetch_content(article: dict) -> str | None:
    strategy = article.get("fetch_strategy")
    url = article["url"]

    try:
        if strategy == "trafilatura":
            return _fetch_trafilatura(url)
        elif strategy == "github_api":
            return _fetch_github(url)
        elif strategy == "youtube_transcript":
            return _fetch_youtube(url)
        else:
            return None  # metadata_only or unknown
    except Exception as e:
        logger.warning(f"fetch_content failed for {url}: {e}")
        return None


def _fetch_trafilatura(url: str) -> str | None:
    html = trafilatura.fetch_url(url)
    if not html:
        return None
    return trafilatura.extract(html) or None


def _fetch_github(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    resp = requests.get(
        api_url,
        headers={"Accept": "application/vnd.github.raw+json"},
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.text
    logger.warning(f"GitHub API returned {resp.status_code} for {url}")
    return None


def _fetch_youtube(url: str) -> str | None:
    # Deferred — YouTube transcript fetching not yet implemented
    return None
