# scripts/ingest_local.py  ← run THIS first
import asyncio
from .ingest_pipeline import pipeline

async def main():
    # Ingest all existing local PDFs (no web crawl needed)
    r = await pipeline.ingest_directory("data/raw/sebi/", source_type="sebi")
    ok = sum(1 for x in r if x.success)
    vecs = sum(x.vectors_upserted for x in r if x.success)
    print(f"\n✅ {ok}/{len(r)} SEBI PDFs → {vecs} vectors upserted")

    r2 = await pipeline.ingest_file("data/raw/rbi_policy.pdf", source_type="rbi")
    r3 = await pipeline.ingest_file("data/raw/hdfc_earnings.pdf", source_type="annual_report")
    r4 = await pipeline.ingest_file("data/raw/sebi_nri_tax.pdf", source_type="sebi")
    r5 = await pipeline.ingest_directory("data/raw/news/", source_type="news")

    print(pipeline.get_ingestion_stats())

asyncio.run(main())
