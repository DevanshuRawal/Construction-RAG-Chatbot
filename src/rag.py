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


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Split an oversized text span at word boundaries, with a hard fallback."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _table_chunks(table: str, size: int) -> list[str]:
    """Split markdown tables by rows and repeat their header in each chunk."""
    lines = [line.rstrip() for line in table.splitlines() if line.strip()]
    if not lines:
        return []
    has_header = len(lines) > 1 and re.match(r"^\s*\|?\s*:?-{3,}", lines[1])
    header = lines[:2] if has_header else lines[:1]
    rows = lines[len(header):]
    if len("\n".join(lines)) <= size:
        return ["\n".join(lines)]
    if len("\n".join(header)) >= size:
        # A header larger than the target cannot be repeated without breaking the cap.
        return _hard_split("\n".join(lines), size, 0)

    result = []
    current = "\n".join(header)
    for row in rows:
        candidate = current + "\n" + row
        if len(candidate) <= size:
            current = candidate
            continue
        if current != "\n".join(header):
            result.append(current)
        current = "\n".join(header)
        if len(current) + len(row) + 1 <= size:
            current += "\n" + row
        else:
            # A single unusually wide table row cannot be kept atomic.
            row_size = max(1, size - len(current) - 1)
            for piece in _hard_split(row, row_size, 0):
                if len(current) + len(piece) + 1 > size:
                    result.append(current)
                    current = "\n".join(header)
                current += "\n" + piece
    if current.strip() and current not in result:
        result.append(current)
    return result


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk overlap must be non-negative and smaller than chunk size")

    # Split at headings first. Keep table rows together where possible.
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
            lines = [line for line in para.splitlines() if line.strip()]
            is_table = bool(lines) and all(line.lstrip().startswith("|") for line in lines)
            if is_table:
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(_table_chunks(para, size))
                continue

            for piece in re.split(r"(?<=[.!?])\s+", para):
                if not piece.strip():
                    continue
                if len(piece) > size:
                    if current.strip():
                        chunks.append(current.strip())
                        current = ""
                    chunks.extend(_hard_split(piece, size, overlap))
                    continue
                if len(current) + len(piece) + 1 > size and current:
                    previous = current.strip()
                    chunks.append(previous)
                    current = previous[-overlap:] + " " if overlap else ""
                    if len(current) + len(piece) + 1 > size:
                        keep = max(0, size - len(piece) - 1)
                        current = current[-keep:] if overlap and keep else ""
                current += piece + "\n"
        if current.strip():
            chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


class HybridKnowledgeBase:
    def __init__(self, data_dir: str = "data", db_dir: str = "chroma_db", chunk_size: int = 900, overlap: int = 140, encoder=None):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
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
        if k <= 0 or not query or not query.strip() or self.collection.count() == 0:
            return []
        query_vec = self.encoder.encode([query], normalize_embeddings=True).tolist()[0]
        n = min(max(k * 4, 12), self.collection.count())
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

