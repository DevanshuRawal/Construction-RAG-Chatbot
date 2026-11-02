from src.presentation import format_answer


def test_pdf_citation_is_rendered_as_a_readable_source_label():
    answer = (
        "Workmanship is covered for 12 months. "
        "[08_warranty_and_after_sales.pdf]"
    )

    rendered = format_answer(answer)

    assert "[08_warranty_and_after_sales.pdf]" not in rendered
    assert "**Source:** Warranty & After-sales" in rendered
    assert "Workmanship is covered for 12 months." in rendered


def test_duplicate_citations_are_shown_once():
    answer = "See policy [06_project_process.pdf]. Also [06_project_process.pdf]."

    rendered = format_answer(answer)

    assert rendered.count("Project Process") == 1
    assert "[06_project_process.pdf]" not in rendered


def test_hindi_answer_gets_hindi_source_label():
    rendered = format_answer("मॉड्यूलर किचन की वारंटी है। [08_warranty_and_after_sales.pdf]")

    assert "[08_warranty_and_after_sales.pdf]" not in rendered
    assert "**स्रोत:** वारंटी और बिक्री-पश्चात सेवा" in rendered
