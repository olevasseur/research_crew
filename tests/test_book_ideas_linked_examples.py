"""API-level test for /books/{id}/ideas linked_examples enrichment.

pytest is not installed in this venv, so this file is written to be directly
runnable:  python -m tests.test_book_ideas_linked_examples
"""

from fastapi.testclient import TestClient

from rag_api import app


BOOK_ID = "digital-minimalism"


def _get_ideas(client):
    resp = client.get(f"/books/{BOOK_ID}/ideas")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_ideas_endpoint_includes_linked_examples():
    client = TestClient(app)
    payload = _get_ideas(client)
    ideas = payload["ideas"]
    assert ideas, "expected at least one idea for digital-minimalism"

    non_examples = [i for i in ideas if i.get("type") != "example"]
    assert non_examples, "expected non-example ideas"
    for i in non_examples:
        assert "linked_examples" in i, f"missing linked_examples on {i.get('type')}"
        assert isinstance(i["linked_examples"], list)

    with_links = [i for i in non_examples if i["linked_examples"]]
    assert with_links, "expected at least one non-example idea to have linked examples"


def test_linked_example_shape():
    client = TestClient(app)
    payload = _get_ideas(client)
    target = next(
        i for i in payload["ideas"]
        if i.get("type") != "example" and i.get("linked_examples")
    )
    for ex in target["linked_examples"]:
        assert set(ex.keys()).issubset(
            {"text", "section", "window", "type", "source", "note_key", "match_score"}
        )
        assert ex.get("type") == "example"
        assert isinstance(ex.get("text"), str) and ex["text"]
        assert isinstance(ex.get("match_score"), (int, float))


def test_example_ideas_do_not_carry_linked_examples():
    client = TestClient(app)
    payload = _get_ideas(client)
    for i in payload["ideas"]:
        if i.get("type") == "example":
            assert "linked_examples" not in i, "examples should not have linked_examples"


def test_existing_fields_preserved():
    """Ranking / curation / notes / source fields must still be present."""
    client = TestClient(app)
    payload = _get_ideas(client)
    sample = payload["ideas"][0]
    for field in ("type", "text", "section", "source", "note_key", "rank_score"):
        assert field in sample, f"missing {field} on idea"
    assert "pinned" in sample and "hidden" in sample


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
