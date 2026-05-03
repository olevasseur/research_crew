"""API-level test for /books/{id}/examples ranked example browsing data.

pytest is not installed in this venv, so this file is written to be directly
runnable:  python -m tests.test_book_examples_endpoint
"""

from fastapi.testclient import TestClient

from rag_api import app


BOOK_ID = "digital-minimalism"


def _get_examples(client):
    resp = client.get(f"/books/{BOOK_ID}/examples")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_examples_endpoint_returns_ranked_examples():
    client = TestClient(app)
    payload = _get_examples(client)
    examples = payload["examples"]
    assert examples, "expected at least one ranked example"
    scores = [ex["rank_score"] for ex in examples]
    assert scores == sorted(scores, reverse=True), "expected examples sorted by rank_score desc"


def test_example_shape_includes_navigation_and_associations():
    client = TestClient(app)
    payload = _get_examples(client)
    sample = payload["examples"][0]
    for field in (
        "text",
        "section",
        "window",
        "rank_score",
        "associated_ideas",
        "source",
        "source_fields",
    ):
        assert field in sample, f"missing {field}"

    assert isinstance(sample["text"], str) and sample["text"]
    assert isinstance(sample["rank_score"], (int, float))
    assert isinstance(sample["associated_ideas"], list)


def test_associated_ideas_have_link_metadata_when_present():
    client = TestClient(app)
    payload = _get_examples(client)
    with_links = [ex for ex in payload["examples"] if ex["associated_ideas"]]
    assert with_links, "expected at least one example associated with ideas"
    idea = with_links[0]["associated_ideas"][0]
    for field in ("key", "text", "type", "section", "window", "rank_score", "pinned", "match_score"):
        assert field in idea, f"missing associated idea field {field}"


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
