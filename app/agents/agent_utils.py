import httpx
from app.core.config import settings
from typing import Optional

def get_proxy_map() -> Optional[dict]:
    """Returns a dictionary suitable for httpx's proxies argument, or None."""
    proxy_url = settings.PROXY_URL
    if proxy_url:
        return {"http://": proxy_url, "https://": proxy_url}
    return None

async def get_httpx_client() -> httpx.AsyncClient:
    """Creates an async httpx client, potentially configured with proxies."""
    proxies = get_proxy_map()
    # Adjust transport/limits as needed for high concurrency
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    timeout = httpx.Timeout(30.0, connect=10.0) # 30s total, 10s connect

    # Add headers to mimic browser if needed for scraping
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36", # Example User-Agent
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    client = httpx.AsyncClient(
        proxies=proxies,
        limits=limits,
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
        http2=True # Enable HTTP/2 if supported by server/proxy
    )
    return client

# Add other common utilities here, e.g., logging setup, specific data parsing functions.