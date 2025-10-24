#!/usr/bin/env python3
"""
Financial Reports (BCTC) - Simple Usage Example
==============================================

This script demonstrates how to fetch, store, and search
financial reports using the implemented system.
"""

import asyncio
import os
import sys

# Add core_services to path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

from core_services.data_ingestion.database_storage import DatabaseStorage  # noqa: E402
from core_services.data_ingestion.qdrant_storage import QdrantFinancialReportStorage  # noqa: E402
from core_services.data_ingestion.source_adapters import VNStockAdapter  # noqa: E402


async def fetch_and_store_bctc(
    symbol: str,
    year: int,
    quarter: int = None
):
    """
    Fetch and store financial report for a stock

    Args:
        symbol: Stock symbol (e.g., "VNM", "FPT")
        year: Fiscal year (e.g., 2024)
        quarter: Optional quarter (1-4)
    """
    print(f"\n{'='*60}")
    print(f"Fetching Financial Report: {symbol} - {year}Q{quarter or 'Y'}")
    print(f"{'='*60}\n")

    # Configuration
    DB_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/stockaids"
    )
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Initialize adapter
    adapter = VNStockAdapter()

    # Initialize storage with both PostgreSQL and Qdrant
    storage = DatabaseStorage(
        database_url=DB_URL,
        qdrant_url=QDRANT_URL
    )

    try:
        # Fetch from VNStock
        print("📡 Fetching from VNStock...")
        async with adapter:
            result = await adapter.fetch_financial_report(
                symbol=symbol,
                year=year,
                quarter=quarter
            )

        if not result.success:
            print(f"❌ Failed: {result.error}")
            return False

        print(f"✅ Fetched successfully ({result.response_time_ms:.0f}ms)")

        # Display key metrics
        data = result.data
        print("\n📊 Key Metrics:")
        print(f"   Revenue:     {data.get('revenue'):,.0f} VND")
        print(f"   Net Profit:  {data.get('net_profit'):,.0f} VND")
        print(f"   EPS:         {data.get('eps')} VND")
        print(f"   ROE:         {data.get('roe')}%")
        print(f"   Total Assets: {data.get('total_assets'):,.0f} VND")

        # Store in databases
        print("\n💾 Storing in databases...")
        with storage.get_session() as session:
            success = storage._store_financial_report(data, session)
            if success:
                session.commit()
                print("✅ Stored in PostgreSQL + Qdrant")
            else:
                print("❌ Storage failed")
                return False

        print("\n✨ Complete!\n")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def search_financial_reports(query: str, symbol: str = None):
    """
    Search for financial reports using natural language

    Args:
        query: Vietnamese query (e.g., "doanh thu VNM quý 2")
        symbol: Optional stock symbol filter
    """
    print(f"\n{'='*60}")
    print(f"Searching: {query}")
    print(f"{'='*60}\n")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

    try:
        qdrant = QdrantFinancialReportStorage(qdrant_url=QDRANT_URL)

        results = qdrant.search_similar_reports(
            query=query,
            symbol=symbol,
            limit=5
        )

        if not results:
            print("No results found")
            return

        print(f"Found {len(results)} results:\n")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['symbol']} - {res['year']}Q{res.get('quarter', 'Y')}")
            print(f"   Score: {res['score']:.4f}")
            print(f"   Type: {res['chunk_type']}")
            if res.get('roe'):
                print(f"   ROE: {res['roe']}%")
            preview = res['text'][:120].replace('\n', ' ')
            print(f"   Text: {preview}...\n")

    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Fetch report:  python run_bctc.py fetch VNM 2024 2")
        print("  Search:        python run_bctc.py search \"doanh thu VNM\"")
        print("  Run E2E test:  python run_bctc.py test")
        return

    command = sys.argv[1]

    if command == "fetch":
        if len(sys.argv) < 4:
            print("Usage: python run_bctc.py fetch SYMBOL YEAR [QUARTER]")
            return

        symbol = sys.argv[2]
        year = int(sys.argv[3])
        quarter = int(sys.argv[4]) if len(sys.argv) > 4 else None

        await fetch_and_store_bctc(symbol, year, quarter)

    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python run_bctc.py search \"query\" [SYMBOL]")
            return

        query = sys.argv[2]
        symbol = sys.argv[3] if len(sys.argv) > 3 else None

        await search_financial_reports(query, symbol)

    elif command == "test":
        # Run E2E test
        from core_services.tests.test_financial_reports_e2e import test_e2e_financial_report

        success = await test_e2e_financial_report()
        sys.exit(0 if success else 1)

    else:
        print(f"Unknown command: {command}")
        print("Available commands: fetch, search, test")


if __name__ == "__main__":
    asyncio.run(main())
