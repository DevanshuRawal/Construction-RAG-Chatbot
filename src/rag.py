"""Document ingestion and hybrid retrieval for Aangan Habitat."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path



def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Preserve table rows and headings; collapse PDF's noisy single line breaks.
    return re.sub(r"(?<!\n)\n(?!\n)(?!\s*(?:\||[-*] |\d+\.|#{1,6} ))", " ", text).strip()


def _read(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        pages = []
        for page in PdfReader(str(path)).pages:
            pages.append(page.extract_text(extraction_mode="layout") or page.extract_text() or "")
        return "\n\n".join(pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            parts.append("\n".join(" | ".join(c.text.strip() for c in row.cells) for row in table.rows))
        return "\n".join(parts)
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    return ""


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    # Split at headings first. Keep markdown/PDF table blocks intact where practical.
    sections = re.split(r"(?=^#{1,6} .+$)", text, flags=re.M)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= size:
            chunks.append(section)
            continue
        paragraphs = re.split(r"\n\s*\n", section)
        current = ""
        for para in paragraphs:
            # Oversized prose is split by sentence; table blocks remain an atomic unit.
            pieces = [para] if para.lstrip().startswith("|") else re.split(r"(?<=[.!?])\s+", para)
            for piece in pieces:
                if len(current) + len(piece) + 1 > size and current:
                    chunks.append(current.strip())
                    current = current[-overlap:] + " " if overlap else ""
                current += piece + "\n"
        if current.strip():
            chunks.append(current.strip())
    return [c for c in chunks if c]


class HybridKnowledgeBase:
    def __init__(self, data_dir: str = "data", db_dir: str = "chroma_db", chunk_size: int = 900, overlap: int = 140, encoder=None):
        import chromadb
        self.data_dir = Path(data_dir)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection("aangan_documents", metadata={"hnsw:space": "cosine"})
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = encoder
        self.chunk_size, self.overlap = chunk_size, overlap

    def ingest(self, rebuild: bool = False) -> int:
        if rebuild:
            self.client.delete_collection("aangan_documents")
            self.collection = self.client.get_or_create_collection("aangan_documents", metadata={"hnsw:space": "cosine"})
        existing = set()
        if self.collection.count():
            existing = set(self.collection.get(include=["metadatas"])["ids"])
        ids, docs, metas, desired = [], [], [], set()
        for path in sorted(self.data_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".pdf", ".docx", ".md", ".txt"}:
                continue
            source = path.relative_to(self.data_dir).as_posix()
            content = _clean(_read(path))
            for index, chunk in enumerate(_chunks(content, self.chunk_size, self.overlap)):
                digest = hashlib.sha1(f"{source}:{index}:{chunk}".encode()).hexdigest()
                desired.add(digest)
                if digest in existing:
                    continue
                ids.append(digest); docs.append(chunk)
                metas.append({"source": source, "chunk": index, "category": path.stem.split("_")[1] if "_" in path.stem else "general"})
        obsolete = sorted(existing - desired)
        for start in range(0, len(obsolete), 500):
            self.collection.delete(ids=obsolete[start:start + 500])
        for start in range(0, len(ids), 64):
            batch_docs = docs[start:start + 64]
            vectors = self.encoder.encode(batch_docs, normalize_embeddings=True).tolist()
            self.collection.add(ids=ids[start:start + 64], documents=batch_docs, metadatas=metas[start:start + 64], embeddings=vectors)
        return len(ids)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[\w₹]+", text) if len(t) > 1}

    @staticmethod
    def _authority_boost(query: str, source: str) -> float:
        query = query.lower()
        source = source.lower().replace("\\", "/")
        historical = any(term in query for term in ("case study", "past project", "historical", "previous project"))
        if not historical and any(term in query for term in ("rate", "pricing", "price", "cost", "per sq", "square foot")):
            if source.endswith("04_pricing_rate_card.pdf"):
                return 0.15
        if any(term in query for term in ("warranty", "after sales", "after-sales", "repair")):
            if source.endswith("08_warranty_and_after_sales.pdf"):
                return 0.15
        if any(term in query for term in ("payment", "advance", "milestone", "cancellation", "refund")):
            if source.endswith("09_payment_and_cancellation.pdf"):
                return 0.15
        if any(term in query for term in ("timeline", "how long", "delay", "milestone", "duration")):
            if source.endswith("07_timelines_and_milestones.pdf"):
                return 0.15
        if any(term in query for term in ("discount", "service area", "outside india", "coverage", "policy")):
            if source.endswith("16_scope_and_decision_register.md"):
                return 0.15
        return 0.0

    def search(self, query: str, k: int = 6) -> list[dict]:
        query_vec = self.encoder.encode([query], normalize_embeddings=True).tolist()[0]
        n = min(max(k * 4, 12), max(self.collection.count(), 1))
        result = self.collection.query(query_embeddings=[query_vec], n_results=n, include=["documents", "metadatas", "distances"])
        qterms = self._terms(query)
        candidates = []
        for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            terms = self._terms(doc)
            lexical = len(qterms & terms) / max(len(qterms), 1)
            semantic = 1 - float(distance) / 2
            score = 0.72 * semantic + 0.28 * lexical + self._authority_boost(query, meta["source"])
            candidates.append({"text": doc, "source": meta["source"], "category": meta["category"], "score": score})
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:k]

