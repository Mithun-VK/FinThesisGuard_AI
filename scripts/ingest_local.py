#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingestion.ingest_pipeline import pipeline
from backend.retrieval.pinecone_client import pinecone_client   # ← ADD THIS

async def main():
    # ✅ INITIALIZE PINECONE FIRST — must be called before any upsert
    await pinecone_client.initialize_index()
    print("✅ Pinecone initialized")

    print("=== LOCAL SEBI PDFs (26 files) ===")
    sebi_results = await pipeline.ingest_directory(
        "data/raw/sebi/",
        source_type="sebi",
    )
    
    print("=== Individual PDFs ===")
    await pipeline.ingest_file("data/raw/rbi_policy.pdf", source_type="rbi")
    await pipeline.ingest_file("data/raw/hdfc_earnings.pdf", source_type="annual_report")
    await pipeline.ingest_file("data/raw/sebi_nri_tax.pdf", source_type="sebi")
    
    print("=== News JSONs (17 files) ===")
    news_results = await pipeline.ingest_directory(
        "data/raw/news/",
        source_type="news",
    )
    
    stats = pipeline.get_ingestion_stats()
    print("\n=== FINAL STATS ===")
    print(f"SEBI PDFs:  {sum(1 for r in sebi_results if r.success)}/{len(sebi_results)}")
    print(f"News:       {sum(1 for r in news_results if r.success)}/{len(news_results)}")
    print(f"TOTAL VECTORS: {stats['total_vectors']}")

if __name__ == "__main__":
    asyncio.run(main())
