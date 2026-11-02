"""Human-friendly presentation helpers for grounded answers."""
import re


_DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097f]")
_SOURCE_PATTERN = re.compile(r"\[([^\]]+\.(?:pdf|docx|md|txt))\]", re.IGNORECASE)
_SOURCE_LABELS = {
    "08_warranty_and_after_sales.pdf": "Warranty & After-sales",
    "06_project_process.pdf": "Project Process",
    "16_scope_and_decision_register.md": "Scope & Decision Register",
}
_HINDI_SOURCE_LABELS = {
    "08_warranty_and_after_sales.pdf": "वारंटी और बिक्री-पश्चात सेवा",
    "06_project_process.pdf": "परियोजना प्रक्रिया",
    "16_scope_and_decision_register.md": "नीति और कार्य-सीमा",
}


def _source_label(filename: str) -> str:
    if filename in _SOURCE_LABELS:
        return _SOURCE_LABELS[filename]
    name = re.sub(r"\.(?:pdf|docx|md|txt)$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"^\d+_", "", name).replace("_", " ").replace("-", " ")
    return name.title()


def format_answer(answer: str) -> str:
    """Replace raw document filename citations with readable source labels."""
    filenames = _SOURCE_PATTERN.findall(answer)
    if not filenames:
        return answer

    is_hindi = bool(_DEVANAGARI_PATTERN.search(answer))
    labels = list(dict.fromkeys(
        _HINDI_SOURCE_LABELS.get(filename, _source_label(filename)) if is_hindi else _source_label(filename)
        for filename in filenames
    ))
    body = _SOURCE_PATTERN.sub("", answer)
    body = re.sub(r"[ \t]+([,.!?])", r"\1", body)
    body = re.sub(r"[ \t]{2,}", " ", body).strip()
    heading = "**स्रोत:**" if is_hindi else ("**Source:**" if len(labels) == 1 else "**Sources:**")
    return f"{body}\n\n{heading} " + "; ".join(labels)
