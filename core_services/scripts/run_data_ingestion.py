#!/usr/bin/env python3
"""
Data Ingestion Runner Script
============================

Comprehensive script to run data ingestion from Vietnamese stock sources
and load crawled data into PostgreSQL + Qdrant databases.

Supports:
- Stock prices (current and historical)
- Market indices
- Financial reports (BCTC - Báo cáo tài chính)

Usage:
    # Run full pipeline
    python run_data_ingestion.py

    # Test single stock
    python run_data_ingestion.py --test-symbol VNM

    # Fetch historical data for a date range
    python run_data_ingestion.py --symbol VNM --start-date 2020-01-01 --end-date 2025-01-01

    # Fetch historical data for multiple symbols
    python run_data_ingestion.py --symbols VNM,FPT,VCB --start-date 2023-01-01 --end-date 2024-01-01

    # Fetch financial report (BCTC)
    python run_data_ingestion.py --bctc --symbol VNM --year 2024 --quarter 2

    # Fetch financial reports for multiple symbols
    python run_data_ingestion.py --bctc --symbols VNM,FPT,VCB --year 2024 --quarter 2

    # Development mode (reduced load)
    python run_data_ingestion.py --dev

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (required)
    REDIS_URL: Redis connection string (default: redis://localhost:6379)
    QDRANT_URL: Qdrant connection string (default: http://localhost:6333)
"""

import asyncio
import os
import sys
from datetime import datetime

from sqlalchemy import create_engine, text

from core_services.data_ingestion.database_storage import DatabaseStorage
from core_services.data_ingestion.pipeline_runner import PipelineManager
from core_services.data_ingestion.source_adapters import VNStockAdapter
from core_services.data_ingestion.task_manager import Priority, TaskTemplates
from core_services.utils.logger_utils import logger


async def ingest_financial_reports(
    symbols: list,
    year: int,
    quarter: int = None
):
    """
    Ingest financial reports (BCTC) for given symbols.

    Args:
        symbols: List of stock symbols (e.g., ["VNM", "FPT", "VCB"])
        year: Fiscal year (e.g., 2024)
        quarter: Optional quarter (1-4), omit for annual reports
    """
    period = f"{year}Q{quarter}" if quarter else f"{year} Annual"
    logger.info(f"Starting financial reports ingestion for {len(symbols)} symbols...")
    logger.info(f"Period: {period}")

    # Check environment
    database_url = os.getenv("DATABASE_URL")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        logger.info("Example: export DATABASE_URL='postgresql://user:pass@localhost:5432/stockaids'")
        return

    # Initialize adapter and storage
    adapter = VNStockAdapter()
    storage = DatabaseStorage(
        database_url=database_url,
        qdrant_url=qdrant_url
    )

    success_count = 0
    failed_count = 0

    try:
        async with adapter:
            for symbol in symbols:
                try:
                    logger.info(f"📡 Fetching {symbol} - {period}...")

                    # Fetch financial report
                    result = await adapter.fetch_financial_report(
                        symbol=symbol.upper(),
                        year=year,
                        quarter=quarter
                    )

                    if not result.success:
                        logger.error(f"❌ Failed to fetch {symbol}: {result.error}")
                        failed_count += 1
                        continue

                    logger.info(f"✅ Fetched {symbol} in {result.response_time_ms:.0f}ms")

                    # Display key metrics (handle None values safely)
                    data = result.data
                    revenue = data.get("revenue") or 0
                    net_profit = data.get("net_profit") or 0
                    eps = data.get("eps") or 0
                    roe = data.get("roe") or 0

                    logger.info(f"   Revenue: {revenue:,.0f} VND")
                    logger.info(f"   Net Profit: {net_profit:,.0f} VND")
                    logger.info(f"   EPS: {eps} VND")
                    logger.info(f"   ROE: {roe}%")

                    # Store in databases
                    logger.info(f"💾 Storing {symbol} in PostgreSQL + Qdrant...")
                    with storage.get_session() as session:
                        if storage._store_financial_report(data, session):
                            session.commit()
                            logger.info(f"✅ Stored {symbol} successfully")
                            success_count += 1
                        else:
                            logger.error(f"❌ Failed to store {symbol}")
                            failed_count += 1

                except Exception as e:
                    logger.error(f"❌ Error processing {symbol}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full traceback for debugging
                    failed_count += 1
                    continue

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Financial Reports Ingestion Summary")
        logger.info("=" * 60)
        logger.info(f"Total: {len(symbols)} symbols")
        logger.info(f"Success: {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info("=" * 60 + "\n")

        # Provide query examples
        if success_count > 0:
            logger.info("Check PostgreSQL for financial data:")
            for symbol in symbols[:3]:  # Show first 3 examples
                logger.info("  SELECT * FROM financial_statements fs")
                logger.info("  JOIN stocks s ON fs.stock_id = s.id")
                logger.info(f"  WHERE s.symbol = '{symbol.upper()}' AND fiscal_year = {year}")
                quarter_clause = f"AND fiscal_quarter = {quarter}" if quarter else ""
                logger.info(f"  {quarter_clause};")

    except Exception as e:
        logger.error(f"Financial reports ingestion failed: {e}")
        import traceback
        traceback.print_exc()


async def test_single_stock(symbol: str):
    """
    Test data ingestion for a single stock.

    Args:
        symbol: Stock symbol to test (e.g., "VNM")
    """
    logger.info(f"Testing data ingestion for {symbol}...")

    # Check environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        logger.info("Example: export DATABASE_URL='postgresql://user:pass@localhost:5432/stockaids'")
        return

    # Create pipeline manager
    manager = PipelineManager()
    try:
        # Initialize pipeline
        await manager.orchestrator.initialize()

        # Start pipeline with workers
        logger.info("Starting pipeline workers...")
        await manager.orchestrator.start()

        # Wait a moment for workers to start
        await asyncio.sleep(2)

        # Submit task
        task = TaskTemplates.create_stock_price_task(symbol.upper(), Priority.HIGH)
        task_id = await manager.orchestrator.submit_adhoc_task(task)

        logger.info(f"Submitted task {task_id} for {symbol}")

        # Wait for task to be processed
        logger.info("Waiting for task to complete...")
        await asyncio.sleep(15)

        # Get status
        status = await manager.orchestrator.get_status()
        logger.info(f"Pipeline metrics: {status.get('metrics', {})}")

        logger.info(f"✅ Data ingestion test completed for {symbol}")
        logger.info("Check PostgreSQL for crawled data:")
        logger.info(f"  SELECT * FROM stocks WHERE symbol = '{symbol.upper()}';")
        logger.info(f"  SELECT * FROM stock_prices WHERE stock_id = (SELECT id FROM stocks WHERE symbol = "
                    f"'{symbol.upper()}') ORDER BY date DESC LIMIT 5;")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Stopping pipeline...")
        await manager.orchestrator.stop()


async def ingest_historical_data(symbols: list, start_date: str, end_date: str):
    """
    Ingest historical data for given symbols within a date range.

    Args:
        symbols: List of stock symbols (e.g., ["VNM", "FPT", "VCB"])
        start_date: Start date in YYYY-MM-DD format (e.g., "2020-01-01")
        end_date: End date in YYYY-MM-DD format (e.g., "2025-01-01")
    """
    logger.info(f"Starting historical data ingestion for {len(symbols)} symbols...")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Check environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        logger.info("Example: export DATABASE_URL='postgresql://user:pass@localhost:5432/stockaids'")
        return

    # Validate date format
    try:
        from datetime import datetime as dt
        dt.strptime(start_date, "%Y-%m-%d")
        dt.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        logger.info("Date format should be YYYY-MM-DD (e.g., 2020-01-01)")
        return

    # Create pipeline manager
    manager = PipelineManager()
    try:
        # Initialize pipeline
        await manager.orchestrator.initialize()

        # Start pipeline workers ONLY (without scheduled tasks to avoid queue flooding)
        logger.info("Starting pipeline workers (without scheduled tasks)...")
        await manager.orchestrator.start_workers_only()
        await asyncio.sleep(2)

        # Submit historical data tasks for each symbol
        task_ids = []
        for symbol in symbols:
            task = TaskTemplates.create_historical_data_task(
                symbol.upper(),
                start_date,
                end_date,
                Priority.HIGH
            )
            task_id = await manager.orchestrator.submit_adhoc_task(task)
            task_ids.append(task_id)
            logger.info(f"Submitted historical data task {task_id} for {symbol.upper()}")

        # Wait for all tasks to be processed
        logger.info(f"Waiting for {len(task_ids)} tasks to complete...")
        wait_time = 30 + (len(symbols) * 10)  # Base time + 10s per symbol
        logger.info(f"Estimated wait time: {wait_time} seconds")
        await asyncio.sleep(wait_time)

        # Get status
        status = await manager.orchestrator.get_status()
        logger.info(f"Pipeline metrics: {status.get('metrics', {})}")

        logger.info(f"✅ Historical data ingestion completed for {len(symbols)} symbols")
        logger.info("Check PostgreSQL for crawled data:")
        for symbol in symbols:
            logger.info(f"  SELECT COUNT(*) FROM stock_prices WHERE stock_id = "
                        f"(SELECT id FROM stocks WHERE symbol = '{symbol.upper()}') "
                        f"AND date BETWEEN '{start_date}' AND '{end_date}';")

    except Exception as e:
        logger.error(f"Historical data ingestion failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Stopping pipeline...")
        await manager.orchestrator.stop()


async def run_full_pipeline(dev_mode: bool = False):
    """
    Run full data ingestion pipeline.

    Args:
        dev_mode: Run in development mode with reduced load
    """
    logger.info("Starting Vietnamese Stock Market Data Ingestion Pipeline...")

    # Check environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        logger.info("Example: export DATABASE_URL='postgresql://user:pass@localhost:5432/stockaids'")
        return

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    logger.info(f"Database: {database_url.split('@')[1] if '@' in database_url else 'configured'}")
    logger.info(f"Redis: {redis_url}")

    # Create pipeline manager
    manager = PipelineManager()

    try:
        if dev_mode:
            logger.info("Running in development mode (reduced load)...")
            await manager.start_development_mode()
        else:
            logger.info("Running in production mode...")
            await manager.run()

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        await manager.orchestrator.stop()


async def test_database_connection():
    """Test database connection before running pipeline."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return False

    logger.info("Testing database connection...")
    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        # Test connection
        conn.execute(text("SELECT 1")).fetchone()
        logger.info("✅ Database connection successful")

        # Check if required tables exist
        tables_query = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('stocks', 'indices', 'stock_prices', 'index_history', 'financial_statements')
        """)
        tables = conn.execute(tables_query).fetchall()
        table_names = [row[0] for row in tables]

        logger.info(f"Found tables: {', '.join(table_names)}")

        if "stocks" not in table_names:
            logger.warning("⚠️  'stocks' table not found. Run migrations first:")
            logger.warning("  cd stockaids-backend && python scripts/init_schema.py")
            return False

        if "financial_statements" in table_names:
            logger.info("✅ Financial statements table available (BCTC support enabled)")
        else:
            logger.info("ℹ️  Financial statements table not found (BCTC features disabled)")

    engine.dispose()
    return True


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vietnamese Stock Market Data Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Existing stock price arguments
    parser.add_argument(
        "--test-symbol",
        help="Test ingestion for a single stock symbol (e.g., VNM, FPT)"
    )
    parser.add_argument(
        "--symbol",
        help="Single stock symbol for data ingestion"
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of stock symbols (e.g., VNM,FPT,VCB)"
    )
    parser.add_argument(
        "--start-date",
        help="Start date for historical data in YYYY-MM-DD format (e.g., 2020-01-01)"
    )
    parser.add_argument(
        "--end-date",
        help="End date for historical data in YYYY-MM-DD format (e.g., 2025-01-01)"
    )

    # Financial reports (BCTC) arguments
    parser.add_argument(
        "--bctc",
        action="store_true",
        help="Fetch financial reports (Báo cáo tài chính) instead of stock prices"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Fiscal year for financial reports (e.g., 2024)"
    )
    parser.add_argument(
        "--quarter",
        type=int,
        choices=[1, 2, 3, 4],
        help="Fiscal quarter (1-4), omit for annual reports"
    )

    # General arguments
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with reduced load"
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Only test database connection and exit"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Vietnamese Stock Market Data Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Check database connection first
    if args.check_db:
        db_ok = await test_database_connection()
        sys.exit(0 if db_ok else 1)

    db_ok = await test_database_connection()
    if not db_ok:
        logger.error("Database connection failed. Exiting.")
        sys.exit(1)

    # Handle BCTC (financial reports) ingestion
    if args.bctc:
        if not args.year:
            logger.error("--year is required when using --bctc")
            parser.print_help()
            sys.exit(1)

        if not (args.symbol or args.symbols):
            logger.error("--symbol or --symbols is required when using --bctc")
            parser.print_help()
            sys.exit(1)

        # Parse symbols
        if args.symbol:
            symbols = [args.symbol]
        else:
            symbols = [s.strip() for s in args.symbols.split(",")]

        await ingest_financial_reports(
            symbols=symbols,
            year=args.year,
            quarter=args.quarter
        )
        return

    # Validate historical data arguments
    if (args.symbol or args.symbols) and (not args.start_date or not args.end_date):
        logger.error("--start-date and --end-date are required when using --symbol or --symbols")
        parser.print_help()
        sys.exit(1)

    if (args.start_date or args.end_date) and not (args.symbol or args.symbols):
        logger.error("--symbol or --symbols is required when using --start-date and --end-date")
        parser.print_help()
        sys.exit(1)

    # Run based on arguments
    if args.test_symbol:
        await test_single_stock(args.test_symbol)
    elif args.symbol or args.symbols:
        # Historical data ingestion
        if args.symbol:
            symbols = [args.symbol]
        else:
            symbols = [s.strip() for s in args.symbols.split(",")]
        await ingest_historical_data(symbols, args.start_date, args.end_date)
    else:
        await run_full_pipeline(dev_mode=args.dev)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nPipeline stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
