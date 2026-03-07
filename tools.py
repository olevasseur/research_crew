import requests
from bs4 import BeautifulSoup
from crewai.tools import tool


@tool
def fetch_webpage(url: str) -> str:
    """
    Fetch the visible text from a webpage URL.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        # compress whitespace
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        return "\n".join(lines[:5000])  # prevent giant pages

    except Exception as e:
        return f"Error fetching page: {str(e)}"
