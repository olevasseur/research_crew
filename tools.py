import requests
from bs4 import BeautifulSoup

def fetch_url_text(url: str) -> str:
    headers = {"User-Agent": "research_crew/0.1"}
    r = requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ")
    text = " ".join(text.split())
    return text[:12000]
