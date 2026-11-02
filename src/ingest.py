"""Build or incrementally update the local Chroma index: python -m src.ingest [--rebuild]."""
import argparse
import os
from dotenv import load_dotenv
from src.rag import HybridKnowledgeBase

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="replace the vector collection from current source files")
    args = parser.parse_args()
    kb = HybridKnowledgeBase(os.getenv("DATA_DIR", "data"), os.getenv("CHROMA_DIR", "chroma_db"), int(os.getenv("CHUNK_SIZE", "900")), int(os.getenv("CHUNK_OVERLAP", "140")))
    count = kb.ingest(rebuild=args.rebuild)
    print(f"Indexed {count} new chunks; collection contains {kb.collection.count()} chunks.")

if __name__ == "__main__": main()
