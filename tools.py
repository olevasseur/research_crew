import requests
from bs4 import BeautifulSoup

def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20)
        html = r.text
        soup = BeautifulSoup(html, "lxml")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(" ")
        text = " ".join(text.split())

        return text[:8000]
    except Exception as e:
        return f"ERROR: {e}"
