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


def test_lexical_terms_include_currency_and_ignore_single_characters():
    assert HybridKnowledgeBase._terms("₹1,950 per sq ft") == {"₹1", "950", "per", "sq", "ft"}


def test_reranker_prefers_current_authority_for_policy_lookups():
    query = "premium interior package rate per square foot"
    assert HybridKnowledgeBase._authority_boost(query, "04_pricing_rate_card.pdf") > 0
    assert HybridKnowledgeBase._authority_boost(query, "14_case_study_3bhk.pdf") == 0
    historical_query = "historical case study price for a 3BHK"
    assert HybridKnowledgeBase._authority_boost(historical_query, "04_pricing_rate_card.pdf") == 0
