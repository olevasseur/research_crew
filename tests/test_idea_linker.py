from rag.idea_linker import idea_key, link_ideas_to_examples, rank_examples_by_idea_links


def _idea(text, section="Chapter 1", window=0, type_="key_idea"):
    return {"type": type_, "text": text, "section": section, "window": window}


def test_empty_inputs_return_empty_dict():
    assert link_ideas_to_examples([], []) == {}


def test_idea_key_is_stable():
    idea = _idea("Solitude matters for clear thinking")
    assert idea_key(idea) == idea_key(dict(idea))


def test_returns_entry_per_idea_even_without_matches():
    ideas = [_idea("Completely unrelated topic about gardening tulips", section="Chapter 1", window=0)]
    examples = [
        _idea("Thoreau walked alone at Walden Pond", section="Chapter 9", window=20, type_="example")
    ]
    out = link_ideas_to_examples(ideas, examples)
    assert len(out) == 1
    key = next(iter(out))
    assert out[key] == []


def test_token_overlap_matches():
    ideas = [_idea("Solitude and silence help clear thinking")]
    examples = [
        _idea("Thoreau sought solitude and silence at Walden", section="Chapter 2", window=4, type_="example"),
        _idea("The company released a new smartphone model", section="Chapter 9", window=20, type_="example"),
    ]
    out = link_ideas_to_examples(ideas, examples)
    key = idea_key(ideas[0])
    matched = out[key]
    assert len(matched) == 1
    assert "Thoreau" in matched[0]["text"]
    assert matched[0]["match_score"] > 0


def test_same_section_boosts_score():
    idea = _idea("Attention is a finite resource", section="Chapter 3", window=5)
    ex_same_section = _idea(
        "Attention fragments when interruptions arrive",
        section="Chapter 3",
        window=5,
        type_="example",
    )
    ex_other_section = _idea(
        "Attention fragments when interruptions arrive",
        section="Chapter 7",
        window=12,
        type_="example",
    )
    out = link_ideas_to_examples([idea], [ex_same_section, ex_other_section], top_k=2)
    matched = out[idea_key(idea)]
    assert len(matched) >= 1
    assert matched[0]["section"] == "Chapter 3"
    assert matched[0]["match_score"] > matched[-1]["match_score"] or len(matched) == 1


def test_top_k_respected_and_sorted_desc():
    idea = _idea("Deep work requires solitude and focus")
    examples = [
        _idea("Deep work and solitude make focus possible", type_="example"),
        _idea("Solitude is a core practice for focus", type_="example", section="Chapter 2", window=3),
        _idea("Deep work deserves dedicated blocks", type_="example"),
        _idea("Focus is the scarce resource", type_="example"),
    ]
    out = link_ideas_to_examples([idea], examples, top_k=2)
    matched = out[idea_key(idea)]
    assert len(matched) == 2
    assert matched[0]["match_score"] >= matched[1]["match_score"]


def test_duplicate_example_texts_suppressed():
    idea = _idea("Solitude matters")
    examples = [
        _idea("Thoreau on solitude", type_="example"),
        _idea("Thoreau on solitude", type_="example"),
    ]
    out = link_ideas_to_examples([idea], examples, top_k=5)
    matched = out[idea_key(idea)]
    assert len(matched) == 1


def test_rank_examples_prefers_links_to_high_ranked_ideas():
    ideas = [
        {
            **_idea("Solitude helps people focus during demanding creative work"),
            "rank_score": 0.95,
        },
        {
            **_idea("Solitude helps people focus during demanding creative work"),
            "rank_score": 0.10,
        },
    ]
    examples = [
        _idea("Solitude helps focus during creative work at the cabin", type_="example"),
        _idea("Solitude helps focus during creative work in the office", type_="example"),
    ]

    ranked = rank_examples_by_idea_links([ideas[0]], [examples[0]], top_k=1)
    low_ranked = rank_examples_by_idea_links([ideas[1]], [examples[1]], top_k=1)

    assert ranked[0]["example_score"] > low_ranked[0]["example_score"]
    assert ranked[0]["example_score_parts"]["rank_boost"] > low_ranked[0]["example_score_parts"]["rank_boost"]


def test_rank_examples_prefers_pinned_idea_links():
    ideas = [
        {
            **_idea("Attention fragments when interruptions arrive", section="Chapter 1"),
            "rank_score": 0.30,
            "pinned": True,
        },
        {
            **_idea("Deep work improves when people protect focus", section="Chapter 2"),
            "rank_score": 0.95,
            "pinned": False,
        },
    ]
    examples = [
        _idea(
            "Attention fragments when interruptions arrive in a busy office",
            section="Chapter 1",
            type_="example",
        ),
        _idea(
            "Deep work improves when people protect focus in morning blocks",
            section="Chapter 2",
            type_="example",
        ),
    ]

    ranked = rank_examples_by_idea_links(ideas, examples, top_k=2)

    assert ranked[0]["text"].startswith("Attention fragments")
    assert ranked[0]["example_score_parts"]["pinned_boost"] > 0
    assert ranked[0]["associated_ideas"][0]["pinned"] is True


def test_rank_examples_includes_inspectable_fields_without_mutating_inputs():
    idea = {**_idea("Named examples make abstract claims easier to remember"), "rank_score": 0.6}
    example = _idea(
        "Cal Newport describes named examples in 2019 to make claims easier to remember",
        type_="example",
    )

    ranked = rank_examples_by_idea_links([idea], [example])

    assert "example_score" in ranked[0]
    assert set(ranked[0]["example_score_parts"]) == {
        "link_score",
        "rank_boost",
        "pinned_boost",
        "association_bonus",
        "specificity",
    }
    assert ranked[0]["associated_ideas"][0]["text"] == idea["text"]
    assert "example_score" not in example


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
