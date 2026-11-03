from src.rag import HybridKnowledgeBase, _chunks, _clean


def test_clean_removes_pdf_wrap_noise_and_preserves_table_rows():
    raw = "interi-\nors\n\n| Package | Rate |\n| Premium | ₹1,950 |\n\n"
    cleaned = _clean(raw)
    assert "interiors" in cleaned
    assert "| Package | Rate |" in cleaned
    assert "| Premium | ₹1,950 |" in cleaned


def test_chunker_starts_new_chunks_at_headings_and_keeps_table_block():
    table = "| Package | Rate |\n| Premium | ₹1,950 |"
    chunks = _chunks("# Rate card\n" + ("Details. " * 120) + "\n\n" + table, 180, 30)
    assert any(chunk.startswith("# Rate card") for chunk in chunks)
    assert any(table in chunk for chunk in chunks)


def test_chunker_hard_splits_oversized_unbroken_text():
    chunks = _chunks("x" * 1200, 300, 40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert all(set(chunk) == {"x"} for chunk in chunks)


def test_chunker_keeps_limit_when_overlap_meets_near_maximum_sentence():
    text = "# Heading\n" + ("a" * 80) + ". " + ("b" * 98) + "."
    chunks = _chunks(text, 100, 20)

    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunker_splits_large_markdown_tables_and_repeats_header():
    newline = chr(10)
    table = "| Package | Rate |" + newline + "|---|---|" + newline + "".join(
        f"| Package {index} | rate-{index * 1000} |" + newline for index in range(30)
    )
    chunks = _chunks(table, 240, 40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert all("| Package | Rate |" in chunk for chunk in chunks)
    assert all("|---|---|" in chunk for chunk in chunks)
    combined = "\n".join(chunks)
    assert all(f"Package {index}" in combined for index in range(30))


def test_chunker_rejects_invalid_size_and_overlap():
    import pytest

    with pytest.raises(ValueError):
        _chunks("text", 0, 0)
    with pytest.raises(ValueError):
        _chunks("text", 100, 100)
    with pytest.raises(ValueError):
        HybridKnowledgeBase(chunk_size=0)
    with pytest.raises(ValueError):
        HybridKnowledgeBase(chunk_size=100, overlap=100)


def test_search_returns_empty_for_blank_query_empty_index_or_nonpositive_k():
    class EmptyCollection:
        def count(self):
            return 0

    class EncoderMustNotRun:
        def encode(self, *_args, **_kwargs):
            raise AssertionError("encoder should not run for empty searches")

    kb = object.__new__(HybridKnowledgeBase)
    kb.collection = EmptyCollection()
    kb.encoder = EncoderMustNotRun()

    assert kb.search("anything") == []
    assert kb.search(" ") == []
    assert kb.search("anything", k=0) == []


def test_lexical_terms_include_currency_and_ignore_single_characters():
    assert HybridKnowledgeBase._terms("₹1,950 per sq ft") == {"₹1", "950", "per", "sq", "ft"}


def test_reranker_prefers_current_authority_for_policy_lookups():
    query = "premium interior package rate per square foot"
    assert HybridKnowledgeBase._authority_boost(query, "04_pricing_rate_card.pdf") > 0
    assert HybridKnowledgeBase._authority_boost(query, "14_case_study_3bhk.pdf") == 0
    historical_query = "historical case study price for a 3BHK"
    assert HybridKnowledgeBase._authority_boost(historical_query, "04_pricing_rate_card.pdf") == 0
