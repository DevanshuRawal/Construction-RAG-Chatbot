"""Small retrieval hit-rate check: python -m src.evaluate (expected sources are curated)."""
from src.rag import HybridKnowledgeBase

CASES = [
    ("modular kitchen warranty hardware workmanship", "08_warranty_and_after_sales.pdf"),
    ("premium interior package rate per square foot", "04_pricing_rate_card.pdf"),
    ("payment milestones advance balance cancellation", "09_payment_and_cancellation.pdf"),
    ("Pune Mumbai Nashik service area outside India Dubai", "16_scope_and_decision_register.md"),
    ("1,200 carpet area premium estimate 1950", "16_scope_and_decision_register.md"),
    ("project timeline 2BHK design freeze delay", "07_timelines_and_milestones.pdf"),
]

def main():
    kb = HybridKnowledgeBase()
    hits = 0
    for query, expected in CASES:
        found = [item["source"] for item in kb.search(query, k=6)]
        hit = expected in found; hits += hit
        print(f"{'HIT ' if hit else 'MISS'} {expected}: {query}")
    print(f"Source hit rate: {hits}/{len(CASES)} = {hits/len(CASES):.0%}")

if __name__ == "__main__": main()
