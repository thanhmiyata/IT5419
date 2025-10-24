"""
Database Storage Integration for Data Pipeline
==============================================

Minimal integration layer that connects the data ingestion pipeline
to PostgreSQL and Qdrant databases using the enhanced schema system.

This module provides a storage handler that can be injected into the
existing CrawlingPipeline without major architecture changes.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from dateutil.parser import parse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from core_services.utils.common import (DB_DEFAULT_BATCH_SIZE, DB_DEFAULT_TIMEOUT, DB_TABLE_INDEX_HISTORY,
                                        DB_TABLE_INDICES, DB_TABLE_NEWS_ARTICLES, DB_TABLE_STOCK_PRICES,
                                        DB_TABLE_STOCKS, DataType)
from core_services.utils.logger_utils import logger

if TYPE_CHECKING:
    from core_services.data_ingestion.pipeline_architecture import CrawlResult


class DatabaseStorage:
    """
    Storage handler for persisting crawled data to PostgreSQL and Qdrant.
    Designed to integrate with existing CrawlingPipeline architecture.
    """

    def __init__(
        self,
        database_url: str,
        qdrant_url: Optional[str] = None,
        batch_size: int = DB_DEFAULT_BATCH_SIZE,
        timeout: int = DB_DEFAULT_TIMEOUT
    ):
        """
        Initialize database storage.

        Args:
            database_url: PostgreSQL connection string
            qdrant_url: Optional Qdrant URL for vector storage
            batch_size: Batch size for bulk operations
            timeout: Database operation timeout in seconds
        """
        self.database_url = database_url
        self.batch_size = batch_size
        self.timeout = timeout
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args={'connect_timeout': timeout}
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        # Initialize Qdrant storage if URL provided
        self.qdrant_storage = None
        if qdrant_url:
            try:
                from core_services.data_ingestion.qdrant_storage import QdrantFinancialReportStorage
                self.qdrant_storage = QdrantFinancialReportStorage(
                    qdrant_url
                )
                logger.info("Qdrant storage initialized for embeddings")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Qdrant storage: {e}"
                )

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    def store_crawl_result(self, result: "CrawlResult", session: Optional[Session] = None) -> bool:
        """
        Store crawl result to database based on data type.

        Args:
            result: CrawlResult from pipeline
            session: Optional SQLAlchemy session (creates new if None)

        Returns:
            True if storage successful, False otherwise
        """
        if not result.success or not result.data:
            logger.warning(f"Skipping storage for failed result: {result.error}")
            return False

        # Use provided session or create new one
        close_session = False
        if session is None:
            session = self.get_session()
            close_session = True

        try:
            # Route to appropriate storage method based on data type
            if result.data_type == DataType.STOCK_PRICE:
                success = self._store_stock_price(result.data, session)
            elif result.data_type == DataType.INDEX_DATA:
                success = self._store_index_data(result.data, session)
            elif result.data_type == DataType.HISTORICAL_DATA:
                # Historical data is a list of stock prices
                success = self._store_historical_data(result.data, session)
            elif result.data_type == DataType.NEWS_ARTICLE:
                success = self._store_news_article(result.data, session)
            elif result.data_type == DataType.FINANCIAL_REPORT:
                success = self._store_financial_report(result.data, session)
            else:
                logger.warning(f"Unsupported data type for storage: {result.data_type}")
                return False

            if success:
                session.commit()
                logger.info(f"Stored {result.data_type.value} from {result.source.value}")

            return success

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store crawl result: {e}")
            return False
        finally:
            if close_session:
                session.close()

    def _store_stock_price(self, data: Dict[str, Any], session: Session) -> bool:
        """
        Store stock price data.

        Args:
            data: Stock price data from crawler
            session: Database session

        Returns:
            True if successful
        """
        try:
            # Get or create stock
            stock_id = self._get_or_create_stock(data['symbol'], session)

            if not stock_id:
                logger.error(f"Failed to get stock_id for {data['symbol']}")
                return False

            # Prepare price data for stock_prices table
            # vnstock historical data uses 'time' instead of 'date'
            # Try 'time' first (from historical data), then 'timestamp', then 'date', then fallback to now
            time_value = data.get("time") or data.get("timestamp") or data.get("date")
            if time_value:
                # Handle pandas Timestamp objects
                if hasattr(time_value, "to_pydatetime"):
                    # Pandas Timestamp - convert to Python datetime
                    dt = time_value.to_pydatetime()
                    price_timestamp = dt  # Store as datetime object for PostgreSQL
                    price_date = dt.date()  # Extract date part
                elif hasattr(time_value, "strftime"):
                    # Already a datetime object
                    price_timestamp = time_value
                    price_date = time_value.date() if hasattr(time_value, "date") else time_value
                else:
                    dt = parse(str(time_value))
                    price_timestamp = dt
                    price_date = dt.date()
            else:
                # Fallback to current time
                now = datetime.now()
                price_timestamp = now
                price_date = now.date()

            # Use raw SQL for UPSERT (ON CONFLICT) to handle duplicates
            table_name = DB_TABLE_STOCK_PRICES
            upsert_query = text(f"""
                INSERT INTO {table_name} (
                    stock_id, timestamp, date, open, high, low, close, volume,
                    change_value, change_percent, source_id, created_at
                )
                VALUES (
                    :stock_id, :timestamp, :date, :open, :high, :low, :close, :volume,
                    :change, :change_percent, :source_id, CURRENT_TIMESTAMP
                )
                ON CONFLICT (stock_id, timestamp)
                DO UPDATE SET
                    close = EXCLUDED.close,
                    high = CASE
                        WHEN EXCLUDED.high > {table_name}.high
                        THEN EXCLUDED.high
                        ELSE {table_name}.high
                    END,
                    low = CASE
                        WHEN EXCLUDED.low < {table_name}.low
                        THEN EXCLUDED.low
                        ELSE {table_name}.low
                    END,
                    volume = EXCLUDED.volume,
                    change_value = EXCLUDED.change_value,
                    change_percent = EXCLUDED.change_percent
            """)

            session.execute(upsert_query, {
                'stock_id': stock_id,
                'timestamp': price_timestamp,
                'date': price_date,
                'open': data.get('open'),
                'high': data.get('high'),
                'low': data.get('low'),
                'close': data.get('price', data.get('close')),
                'volume': data.get('volume'),
                'change': data.get('change', 0),
                'change_percent': data.get('change_percent', 0),
                'source_id': data.get('source', 'vnstock')
            })

            return True

        except Exception as e:
            logger.error(f"Failed to store stock price data: {e}")
            return False

    def _store_historical_data(self, data_list: list, session: Session) -> bool:
        """
        Store historical stock price data (list of price records).

        Args:
            data_list: List of stock price data dictionaries
            session: Database session

        Returns:
            True if successful
        """
        if not data_list:
            logger.warning("Empty historical data list")
            return False

        try:
            success_count = 0
            for data in data_list:
                # Store each record using the existing stock price method
                if self._store_stock_price(data, session):
                    success_count += 1

            logger.info(f"Stored {success_count}/{len(data_list)} historical price records")
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to store historical data: {e}")
            return False

    def _store_index_data(self, data: Dict[str, Any], session: Session) -> bool:
        """
        Store market index data.

        Args:
            data: Index data from crawler
            session: Database session

        Returns:
            True if successful
        """
        try:
            # Get or create index
            index_id = self._get_or_create_index(data['name'], session)

            if not index_id:
                logger.error(f"Failed to get index_id for {data['name']}")
                return False

            index_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))

            # UPSERT index history
            table_name = DB_TABLE_INDEX_HISTORY
            upsert_query = text(f"""
                INSERT INTO {table_name} (
                    index_id, date, close, volume
                )
                VALUES (
                    :index_id, :date, :value, :volume
                )
                ON CONFLICT (index_id, date)
                DO UPDATE SET
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """)

            session.execute(upsert_query, {
                'index_id': index_id,
                'date': index_date,
                'value': data.get('value'),
                'volume': data.get('volume'),
                'change': data.get('change'),
                'change_percent': data.get('change_percent')
            })

            return True

        except Exception as e:
            logger.error(f"Failed to store index data: {e}")
            return False

    def _store_news_article(self, data: Dict[str, Any], session: Session) -> bool:
        """
        Store news article data.

        Args:
            data: News article data from crawler
            session: Database session

        Returns:
            True if successful
        """
        try:
            # Insert news article
            table_name = DB_TABLE_NEWS_ARTICLES
            insert_query = text(f"""
                INSERT INTO {table_name} (
                    title, content, summary, source_url,
                    published_at, source_name, category,
                    created_at
                )
                VALUES (
                    :title, :content, :summary, :url,
                    :published_at, :source, :category,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (source_url)
                DO NOTHING
            """)

            session.execute(insert_query, {
                'title': data.get('title', ''),
                'content': data.get('description', ''),
                'summary': data.get('description', '')[:500],
                'url': data.get('link', ''),
                'published_at': data.get('published_at', datetime.now().isoformat()),
                'source': data.get('source', 'unknown'),
                'category': data.get('category', 'general')
            })

            return True

        except Exception as e:
            logger.error(f"Failed to store news article: {e}")
            return False

    def _store_financial_report(
        self, data: Dict[str, Any], session: Session
    ) -> bool:
        """
        Store financial report data to PostgreSQL.

        Args:
            data: Financial report data from crawler
            session: Database session

        Returns:
            True if successful
        """
        try:
            # Get or create stock
            stock_id = self._get_or_create_stock(data['symbol'], session)

            if not stock_id:
                logger.error(f"Failed to get stock_id for {data['symbol']}")
                return False

            # UPSERT financial statement
            table_name = "financial_statements"
            upsert_query = text(f"""
                INSERT INTO {table_name} (
                    stock_id, fiscal_year, fiscal_quarter, period_type,
                    report_date,
                    revenue, cost_of_goods_sold, gross_profit,
                    operating_expenses, operating_profit,
                    ebit, ebitda, interest_expense,
                    profit_before_tax, tax_expense, net_profit,
                    net_profit_to_shareholders,
                    total_assets, current_assets, fixed_assets,
                    total_liabilities, current_liabilities,
                    long_term_liabilities, shareholders_equity,
                    operating_cash_flow, investing_cash_flow,
                    financing_cash_flow, net_cash_flow, free_cash_flow,
                    eps, roe, roa, debt_to_equity, current_ratio, quick_ratio,
                    data_source, created_at
                )
                VALUES (
                    :stock_id, :fiscal_year, :fiscal_quarter, :period_type,
                    :report_date,
                    :revenue, :cost_of_goods_sold, :gross_profit,
                    :operating_expenses, :operating_profit,
                    :ebit, :ebitda, :interest_expense,
                    :profit_before_tax, :tax_expense, :net_profit,
                    :net_profit_to_shareholders,
                    :total_assets, :current_assets, :fixed_assets,
                    :total_liabilities, :current_liabilities,
                    :long_term_liabilities, :shareholders_equity,
                    :operating_cash_flow, :investing_cash_flow,
                    :financing_cash_flow, :net_cash_flow, :free_cash_flow,
                    :eps, :roe, :roa, :debt_to_equity, :current_ratio,
                    :quick_ratio,
                    :data_source, CURRENT_TIMESTAMP
                )
                ON CONFLICT (stock_id, fiscal_year, fiscal_quarter, period_type)
                DO UPDATE SET
                    revenue = EXCLUDED.revenue,
                    gross_profit = EXCLUDED.gross_profit,
                    operating_profit = EXCLUDED.operating_profit,
                    ebit = EXCLUDED.ebit,
                    ebitda = EXCLUDED.ebitda,
                    net_profit = EXCLUDED.net_profit,
                    total_assets = EXCLUDED.total_assets,
                    current_assets = EXCLUDED.current_assets,
                    shareholders_equity = EXCLUDED.shareholders_equity,
                    operating_cash_flow = EXCLUDED.operating_cash_flow,
                    eps = EXCLUDED.eps,
                    roe = EXCLUDED.roe,
                    roa = EXCLUDED.roa,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    data_source = EXCLUDED.data_source
            """)

            session.execute(upsert_query, {
                'stock_id': stock_id,
                'fiscal_year': data['fiscal_year'],
                'fiscal_quarter': data.get('fiscal_quarter'),
                'period_type': data['period_type'],
                'report_date': data['report_date'],
                'revenue': data.get('revenue'),
                'cost_of_goods_sold': data.get('cost_of_goods_sold'),
                'gross_profit': data.get('gross_profit'),
                'operating_expenses': data.get('operating_expenses'),
                'operating_profit': data.get('operating_profit'),
                'ebit': data.get('ebit'),
                'ebitda': data.get('ebitda'),
                'interest_expense': data.get('interest_expense'),
                'profit_before_tax': data.get('profit_before_tax'),
                'tax_expense': data.get('tax_expense'),
                'net_profit': data.get('net_profit'),
                'net_profit_to_shareholders': data.get(
                    'net_profit_to_shareholders'
                ),
                'total_assets': data.get('total_assets'),
                'current_assets': data.get('current_assets'),
                'fixed_assets': data.get('fixed_assets'),
                'total_liabilities': data.get('total_liabilities'),
                'current_liabilities': data.get('current_liabilities'),
                'long_term_liabilities': data.get('long_term_liabilities'),
                'shareholders_equity': data.get('shareholders_equity'),
                'operating_cash_flow': data.get('operating_cash_flow'),
                'investing_cash_flow': data.get('investing_cash_flow'),
                'financing_cash_flow': data.get('financing_cash_flow'),
                'net_cash_flow': data.get('net_cash_flow'),
                'free_cash_flow': data.get('free_cash_flow'),
                'eps': data.get('eps'),
                'roe': data.get('roe'),
                'roa': data.get('roa'),
                'debt_to_equity': data.get('debt_to_equity'),
                'current_ratio': data.get('current_ratio'),
                'quick_ratio': data.get('quick_ratio'),
                'data_source': data.get('data_source', 'vnstock')
            })

            logger.info(
                f"Stored financial report for {data['symbol']} "
                f"{data['fiscal_year']}Q{data.get('fiscal_quarter', 'Y')} "
                f"in PostgreSQL"
            )

            # Also store in Qdrant if available
            if self.qdrant_storage:
                try:
                    self.qdrant_storage.store_financial_report(data)
                    logger.info(
                        f"Stored embeddings for {data['symbol']} "
                        f"{data['fiscal_year']}Q{data.get('fiscal_quarter', 'Y')} "
                        f"in Qdrant"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to store in Qdrant (non-critical): {e}"
                    )

            return True

        except Exception as e:
            logger.error(f"Failed to store financial report: {e}")
            return False

    def _get_or_create_stock(self, symbol: str, session: Session) -> Optional[int]:
        """
        Get stock_id or create new stock entry.

        Args:
            symbol: Stock symbol (e.g., 'VNM')
            session: Database session

        Returns:
            stock_id or None if failed
        """
        try:
            # Try to get existing stock
            table_name = DB_TABLE_STOCKS
            result = session.execute(
                text(f"SELECT id FROM {table_name} WHERE symbol = :symbol"),
                {'symbol': symbol.upper()}
            ).fetchone()

            if result:
                return result[0]

            # Create new stock
            insert_result = session.execute(
                text(f"""
                    INSERT INTO {table_name} (symbol, name, created_at)
                    VALUES (:symbol, :name, CURRENT_TIMESTAMP)
                    RETURNING id
                """),
                {
                    'symbol': symbol.upper(),
                    'name': symbol.upper()  # Will be updated later with full name
                }
            )
            return insert_result.fetchone()[0]

        except Exception as e:
            logger.error(f"Failed to get/create stock {symbol}: {e}")
            return None

    def _get_or_create_index(self, name: str, session: Session) -> Optional[int]:
        """
        Get index_id or create new index entry.

        Args:
            name: Index name (e.g., 'VN-Index')
            session: Database session

        Returns:
            index_id or None if failed
        """
        try:
            # Try to get existing index
            table_name = DB_TABLE_INDICES
            result = session.execute(
                text(f"SELECT id FROM {table_name} WHERE name = :name"),
                {'name': name}
            ).fetchone()

            if result:
                return result[0]

            # Create new index
            insert_result = session.execute(
                text(f"""
                    INSERT INTO {table_name} (name, created_at)
                    VALUES (:name, CURRENT_TIMESTAMP)
                    RETURNING id
                """),
                {'name': name}
            )
            return insert_result.fetchone()[0]

        except Exception as e:
            logger.error(f"Failed to get/create index {name}: {e}")
            return None

    def cleanup(self):
        """Cleanup database connections"""
        if self.engine:
            self.engine.dispose()


# Factory function for easy integration
def create_database_storage(database_url: str) -> DatabaseStorage:
    """
    Create DatabaseStorage instance.

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        DatabaseStorage instance
    """
    return DatabaseStorage(database_url)
