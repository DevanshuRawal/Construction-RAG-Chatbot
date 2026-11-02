import json

from src.assistant import ConstructionAssistant
from src.rag import HybridKnowledgeBase


class Matrix(list):
    def tolist(self):
        return list(self)


class FakeEncoder:
    """Deterministic vectors keep this integration test offline and reproducible."""
    def encode(self, texts, normalize_embeddings=True):
        return Matrix([[1.0, 0.0, 0.0] for _ in texts])


def test_markdown_ingest_reconciles_updated_and_deleted_sources(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "04_pricing_rate_card.md").write_text(
        "# Interior rates\nPremium interior package is ₹1,950 per sq. ft.\n\n"
        "## Package table\n| Package | Rate |\n| Premium | ₹1,950 |\n", encoding="utf-8"
    )
    kb = HybridKnowledgeBase(str(data), str(tmp_path / "chroma"), encoder=FakeEncoder())

    assert kb.ingest() == 2
    first = kb.search("Premium interior rate ₹1,950", k=2)
    assert first[0]["source"] == "04_pricing_rate_card.md"
    assert "₹1,950" in first[0]["text"]
    assert kb.ingest() == 0
    assert kb.collection.count() == 2

    (data / "04_pricing_rate_card.md").write_text(
        "# Interior rates\nPremium interior package is ₹9,000 per sq. ft.\n\n"
        "## Package table\n| Package | Rate |\n| Premium | ₹9,000 |\n", encoding="utf-8"
    )
    assert kb.ingest() == 2
    assert kb.collection.count() == 2
    indexed = kb.collection.get(include=["documents"])["documents"]
    assert all("₹1,950" not in document for document in indexed)
    updated = kb.search("Premium interior rate ₹9,000", k=2)
    assert updated[0]["source"] == "04_pricing_rate_card.md"
    assert "₹9,000" in updated[0]["text"]

    (data / "04_pricing_rate_card.md").unlink()
    assert kb.ingest() == 0
    assert kb.collection.count() == 0


def test_same_filename_in_nested_folders_keeps_distinct_sources(tmp_path):
    data = tmp_path / "data"
    (data / "east").mkdir(parents=True)
    (data / "west").mkdir()
    (data / "east" / "policy.md").write_text("East region policy.", encoding="utf-8")
    (data / "west" / "policy.md").write_text("West region policy.", encoding="utf-8")
    kb = HybridKnowledgeBase(str(data), str(tmp_path / "chroma"), encoder=FakeEncoder())

    assert kb.ingest() == 2
    metadata = kb.collection.get(include=["metadatas"])["metadatas"]
    assert {item["source"] for item in metadata} == {"east/policy.md", "west/policy.md"}


def test_buying_intent_captures_all_fields_only_after_consent(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    from cryptography.fernet import Fernet
    encryption_key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("LEADS_ENCRYPTION_KEY", encryption_key)
    leads_path = tmp_path / "leads.jsonl"
    bot = ConstructionAssistant(kb=None, leads_file=str(leads_path))
    lead = {}
    turns = [
        "I want to start next month in Pune, budget around 15 lakhs, full interiors, 1,200 sq ft",
        "Asha Rao",
        "asha@example.com",
        "Yes",
    ]

    answers = []
    for turn in turns:
        answer, lead = bot.respond(turn, [], lead)
        answers.append(answer)

    assert leads_path.exists()
    encrypted = json.loads(leads_path.read_text(encoding="utf-8").splitlines()[0])
    assert "asha@example.com" not in leads_path.read_text(encoding="utf-8")
    from cryptography.fernet import Fernet
    saved = json.loads(Fernet(encryption_key.encode()).decrypt(encrypted["payload"]).decode())
    assert saved["name"] == "Asha Rao"
    assert saved["phone_or_email"] == "asha@example.com"
    assert saved["city"] == "Pune"
    assert saved["project_type"] == "Turnkey interiors"
    assert saved["area_sqft"] == 1200
    assert saved["budget"] == "15 lakhs"
    assert saved["contact_consent"] is True
    assert any("180 days" in answer for answer in answers)


def test_declining_contact_consent_discards_unsaved_lead(tmp_path):
    leads_path = tmp_path / "leads.jsonl"
    bot = ConstructionAssistant(kb=None, leads_file=str(leads_path))
    lead = {
        "name": "Asha Rao",
        "phone_or_email": "asha@example.com",
        "city": "Pune",
        "project_type": "Turnkey interiors",
        "area_sqft": 1200,
        "budget": "15 lakhs",
    }

    _, lead = bot.respond("I want a quote", [], lead)
    assert lead["_awaiting"] == "contact_consent"
    answer, lead = bot.respond("No, please don't contact me", [], lead)

    assert "won't share or save" in answer.lower()
    assert lead == {}
    assert not leads_path.exists()


def test_follow_up_timeline_question_retrieves_with_previous_topic_context():
    class QueryRecordingKnowledgeBase:
        def __init__(self):
            self.queries = []

        def search(self, query):
            self.queries.append(query)
            return [{
                "text": "Modular kitchen installation usually takes 25-35 working days after design freeze. Client-caused delays may extend the schedule.",
                "source": "07_timelines_and_milestones.pdf",
                "category": "timelines",
                "score": 0.2,
            }]

    kb = QueryRecordingKnowledgeBase()
    bot = ConstructionAssistant(kb=kb)
    previous_turns = [{"role": "user", "content": "How long does modular kitchen installation take?"}]

    bot.respond("How long will it take? And what if there's a delay?", previous_turns, {})

    assert "modular kitchen installation" in kb.queries[0].lower()
    assert "what if there's a delay" in kb.queries[0].lower()
