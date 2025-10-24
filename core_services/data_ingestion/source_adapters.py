"""
Enhanced Source Adapters for Vietnamese Stock Market Data Sources
================================================================

Production-ready adapters for all free Vietnamese stock data sources
with comprehensive error handling, rate limiting, and data validation.

"""

import asyncio
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from vnstock import Vnstock

from core_services.data_ingestion.field_mappings import SourceFieldMapperFactory
from core_services.data_ingestion.source_schema import SourceSchemaManager, get_schema_manager
from core_services.utils.logger_utils import logger, suppress_logger

# Suppress vnstock's verbose INFO logs at module level
suppress_logger("vnstock.common.data.data_explorer")


@dataclass
class CrawlResult:
    """Standardized crawl result"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    source: Optional[str] = None
    timestamp: datetime = None
    response_time_ms: float = 0
    cached: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseSourceAdapter(ABC):
    """Base class for all source adapters"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        self.schema_manager = schema_manager or get_schema_manager()
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0.0
        self.request_count = 0
        self.error_count = 0

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def initialize(self):
        """Initialize the adapter"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def respect_rate_limit(self, requests_per_second: float):
        """Ensure rate limiting compliance"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / requests_per_second

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()

    @abstractmethod
    async def fetch_stock_price(self, symbol: str) -> CrawlResult:
        """Fetch current stock price"""

    @abstractmethod
    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> CrawlResult:
        """Fetch historical price data"""

    async def fetch_market_index(self) -> CrawlResult:
        """Fetch market index data (default implementation)"""
        return CrawlResult(success=False, error="Market index not supported by this source")

    async def fetch_news(self, limit: int = 10) -> CrawlResult:
        """Fetch news articles (default implementation)"""
        return CrawlResult(success=False, error="News not supported by this source")


class VNStockAdapter(BaseSourceAdapter):
    """Enhanced VNStock library adapter"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        super().__init__(schema_manager)
        self.source_id = "vnstock"
        self.schema = self.schema_manager.get_source(self.source_id)
        self.vnstock = Vnstock()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def fetch_stock_price(self, symbol: str) -> CrawlResult:
        """Fetch real-time stock price using VNStock"""
        start_time = time.time()
        await self.respect_rate_limit(self.schema.rate_limits.requests_per_second)

        # Get current price data - let vnstock exceptions propagate naturally
        price_data = await asyncio.to_thread(
            self.vnstock.stock_historical_data,
            symbol=symbol.upper(),
            start_date=(datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            resolution="1D"
        )

        if price_data.empty:
            return CrawlResult(
                success=False,
                error=f"No data available for {symbol}",
                source=self.source_id,
                response_time_ms=(time.time() - start_time) * 1000
            )

        latest = price_data.iloc[-1]
        previous = price_data.iloc[-2] if len(price_data) > 1 else latest

        result_data = {
            "symbol": symbol.upper(),
            "price": float(latest["close"]),
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "volume": int(latest["volume"]),
            "change": float(latest["close"] - previous["close"]),
            "change_percent": float((latest["close"] - previous["close"]) / previous["close"] * 100),
            "date": latest.name.strftime("%Y-%m-%d") if hasattr(latest.name, "strftime") else str(latest.name),
            "timestamp": datetime.now().isoformat(),
            "source": self.source_id,
            "market": "vietnam"
        }

        self.request_count += 1
        response_time = (time.time() - start_time) * 1000

        return CrawlResult(
            success=True,
            data=result_data,
            source=self.source_id,
            response_time_ms=response_time
        )

    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> CrawlResult:
        """Fetch historical data using VNStock"""
        start_time = time.time()
        await self.respect_rate_limit(self.schema.rate_limits.requests_per_second)

        # Fetch historical data - let vnstock exceptions propagate naturally
        df = await asyncio.to_thread(
            self.vnstock.stock_historical_data,
            symbol=symbol.upper(),
            start_date=start_date,
            end_date=end_date,
            resolution="1D"
        )

        if df.empty:
            return CrawlResult(
                success=False,
                error=f"No historical data for {symbol}",
                source=self.source_id,
                response_time_ms=(time.time() - start_time) * 1000
            )

        # Convert to list of dictionaries
        historical_data = []
        for idx, row in df.iterrows():
            historical_data.append({
                "symbol": symbol.upper(),
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "source": self.source_id
            })

        self.request_count += 1
        response_time = (time.time() - start_time) * 1000

        return CrawlResult(
            success=True,
            data={"symbol": symbol, "data": historical_data, "count": len(historical_data)},
            source=self.source_id,
            response_time_ms=response_time
        )

    async def fetch_market_index(self) -> CrawlResult:
        """Fetch VN-Index data"""
        start_time = time.time()
        await self.respect_rate_limit(self.schema.rate_limits.requests_per_second)

        # Get VN-Index data - let vnstock exceptions propagate naturally
        index_data = await asyncio.to_thread(
            self.vnstock.stock_historical_data,
            symbol="VNINDEX",
            start_date=(datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            resolution="1D"
        )

        if index_data.empty:
            return CrawlResult(
                success=False,
                error="No VN-Index data available",
                source=self.source_id,
                response_time_ms=(time.time() - start_time) * 1000
            )

        latest = index_data.iloc[-1]
        previous = index_data.iloc[-2] if len(index_data) > 1 else latest

        result_data = {
            "name": "VN-Index",
            "value": float(latest["close"]),
            "change": float(latest["close"] - previous["close"]),
            "change_percent": float((latest["close"] - previous["close"]) / previous["close"] * 100),
            "volume": int(latest["volume"]),
            "date": latest.name.strftime("%Y-%m-%d") if hasattr(latest.name, "strftime") else str(latest.name),
            "timestamp": datetime.now().isoformat(),
            "source": self.source_id
        }

        self.request_count += 1
        response_time = (time.time() - start_time) * 1000

        return CrawlResult(
            success=True,
            data=result_data,
            source=self.source_id,
            response_time_ms=response_time
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def fetch_financial_report(
        self,
        symbol: str,
        year: int,
        quarter: Optional[int] = None,
        report_type: str = "annual"
    ) -> CrawlResult:
        """
        Fetch financial report using VNStock library

        VNStock 3.x provides:
        - stock.finance.income_statement(period='quarter', lang='vi')
        - stock.finance.balance_sheet(period='quarter', lang='vi')
        - stock.finance.cash_flow(period='quarter', lang='vi')
        - stock.finance.ratio(period='quarter', lang='vi')
        """
        start_time = time.time()
        await self.respect_rate_limit(self.schema.rate_limits.requests_per_second)

        try:
            # Get stock object
            stock = self.vnstock.stock(symbol=symbol.upper(), source='VCI')

            # Determine period type
            period = 'quarter' if quarter else 'year'

            # Fetch all financial statements
            income_stmt = await asyncio.to_thread(
                stock.finance.income_statement, period=period, lang='vi'
            )
            balance_sheet = await asyncio.to_thread(
                stock.finance.balance_sheet, period=period, lang='vi'
            )
            cash_flow = await asyncio.to_thread(
                stock.finance.cash_flow, period=period, lang='vi'
            )
            ratios = await asyncio.to_thread(
                stock.finance.ratio, period=period, lang='vi'
            )

            # Check if we got data
            if income_stmt.empty and balance_sheet.empty:
                return CrawlResult(
                    success=False,
                    error=f"No financial data available for {symbol}",
                    source=self.source_id,
                    response_time_ms=(time.time() - start_time) * 1000
                )

            # Get the most recent row (or filter by year/quarter)
            # VNStock returns data with most recent first
            income_row = income_stmt.iloc[0] if not income_stmt.empty else None
            balance_row = (
                balance_sheet.iloc[0] if not balance_sheet.empty else None
            )
            cash_row = cash_flow.iloc[0] if not cash_flow.empty else None
            ratio_row = ratios.iloc[0] if not ratios.empty else None

            # Initialize field mappers
            income_mapper = SourceFieldMapperFactory.get_mapper(
                "vnstock_income"
            )
            balance_mapper = SourceFieldMapperFactory.get_mapper(
                "vnstock_balance"
            )
            cashflow_mapper = SourceFieldMapperFactory.get_mapper(
                "vnstock_cashflow"
            )
            ratio_mapper = SourceFieldMapperFactory.get_mapper(
                "vnstock_ratio"
            )

            # Prepare report date
            if quarter:
                report_date = f"{year}-{quarter*3:02d}-01"
            else:
                report_date = f"{year}-12-31"

            # Extract data using field mappers
            result_data = {
                "symbol": symbol.upper(),
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "period_type": "Q" if quarter else "Y",
                "report_date": report_date,

                # Income Statement
                "revenue": income_mapper.extract_numeric(
                    income_row, "revenue"
                ),
                "cost_of_goods_sold": income_mapper.extract_numeric(
                    income_row, "cost_of_goods_sold"
                ),
                "gross_profit": income_mapper.extract_numeric(
                    income_row, "gross_profit"
                ),
                "operating_expenses": income_mapper.extract_numeric(
                    income_row, "selling_expenses"
                ),
                "operating_profit": income_mapper.extract_numeric(
                    income_row, "operating_profit"
                ),
                "ebit": ratio_mapper.extract_numeric(
                    ratio_row, "ebit"
                ),
                "ebitda": ratio_mapper.extract_numeric(
                    ratio_row, "ebitda"
                ),
                "interest_expense": income_mapper.extract_numeric(
                    income_row, "interest_expense"
                ),
                "profit_before_tax": income_mapper.extract_numeric(
                    income_row, "profit_before_tax"
                ),
                "tax_expense": income_mapper.extract_numeric(
                    income_row, "current_tax_expense"
                ),
                "net_profit": income_mapper.extract_numeric(
                    income_row, "net_profit"
                ),
                "net_profit_to_shareholders": income_mapper.extract_numeric(
                    income_row, "net_profit_to_shareholders"
                ),

                # Balance Sheet
                "total_assets": balance_mapper.extract_numeric(
                    balance_row, "total_assets"
                ),
                "current_assets": balance_mapper.extract_numeric(
                    balance_row, "current_assets"
                ),
                "fixed_assets": balance_mapper.extract_numeric(
                    balance_row, "fixed_assets"
                ),
                "total_liabilities": balance_mapper.extract_numeric(
                    balance_row, "total_liabilities"
                ),
                "current_liabilities": balance_mapper.extract_numeric(
                    balance_row, "current_liabilities"
                ),
                "long_term_liabilities": balance_mapper.extract_numeric(
                    balance_row, "long_term_liabilities"
                ),
                "shareholders_equity": balance_mapper.extract_numeric(
                    balance_row, "shareholders_equity"
                ),

                # Cash Flow
                "operating_cash_flow": cashflow_mapper.extract_numeric(
                    cash_row, "operating_cash_flow"
                ),
                "investing_cash_flow": cashflow_mapper.extract_numeric(
                    cash_row, "investing_cash_flow"
                ),
                "financing_cash_flow": cashflow_mapper.extract_numeric(
                    cash_row, "financing_cash_flow"
                ),
                "net_cash_flow": cashflow_mapper.extract_numeric(
                    cash_row, "net_cash_flow"
                ),

                # Key Ratios
                "eps": ratio_mapper.extract_numeric(ratio_row, "eps"),
                "roe": ratio_mapper.extract_numeric(ratio_row, "roe"),
                "roa": ratio_mapper.extract_numeric(ratio_row, "roa"),
                "debt_to_equity": ratio_mapper.extract_numeric(
                    ratio_row, "debt_to_equity"
                ),
                "current_ratio": ratio_mapper.extract_numeric(
                    ratio_row, "current_ratio"
                ),
                "quick_ratio": ratio_mapper.extract_numeric(
                    ratio_row, "quick_ratio"
                ),

                # Metadata
                "data_source": self.source_id,
                "timestamp": datetime.now().isoformat()
            }

            # Validate that we have meaningful data before returning success
            # Check if at least one core financial metric has a valid value
            core_metrics = [
                result_data.get("revenue"),
                result_data.get("net_profit"),
                result_data.get("total_assets"),
                result_data.get("shareholders_equity")
            ]

            has_valid_data = any(
                metric is not None and metric > 0
                for metric in core_metrics
            )

            if not has_valid_data:
                logger.warning(
                    f"No valid financial data found for {symbol} "
                    f"{year}Q{quarter if quarter else 'Y'}. "
                    f"All core metrics are None or zero. "
                    f"Data will NOT be stored in databases."
                )
                return CrawlResult(
                    success=False,
                    error=(
                        f"No valid financial data available for {symbol}. "
                        f"All core metrics (revenue, net_profit, total_assets, "
                        f"shareholders_equity) are None or zero."
                    ),
                    source=self.source_id,
                    response_time_ms=(time.time() - start_time) * 1000
                )

            self.request_count += 1
            response_time = (time.time() - start_time) * 1000

            logger.info(
                f"Fetched financial report for {symbol} "
                f"{year}Q{quarter if quarter else 'Y'}"
            )

            return CrawlResult(
                success=True,
                data=result_data,
                source=self.source_id,
                response_time_ms=response_time
            )

        except Exception as e:
            logger.error(f"Failed to fetch financial report for {symbol}: {e}")
            return CrawlResult(
                success=False,
                error=str(e),
                source=self.source_id,
                response_time_ms=(time.time() - start_time) * 1000
            )


class CafeFAdapter(BaseSourceAdapter):
    """Enhanced CafeF.vn web scraping adapter"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        super().__init__(schema_manager)
        self.source_id = "cafef"
        self.schema = self.schema_manager.get_source(self.source_id)
        self.base_url = self.schema.base_url

    async def initialize(self):
        """Initialize with proper headers"""
        if self.session is None:
            headers = self.schema.headers.copy()
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def fetch_stock_price(self, symbol: str) -> CrawlResult:
        """Scrape stock price from CafeF"""
        start_time = time.time()
        await self.respect_rate_limit(self.schema.rate_limits.requests_per_second)

        url = f"{self.base_url}/{symbol.upper()}.chn"

        async with self.session.get(url) as response:
            if response.status != 200:
                return CrawlResult(
                    success=False,
                    error=f"HTTP {response.status}",
                    source=self.source_id,
                    response_time_ms=(time.time() - start_time) * 1000
                )

            html = await response.text()

        soup = BeautifulSoup(html, "html.parser")

        # Extract price data using multiple selectors
        price = self._extract_price(soup, symbol)
        change = self._extract_change(soup)
        volume = self._extract_volume(soup)

        if price is None:
            return CrawlResult(
                success=False,
                error=f"Could not extract price for {symbol}",
                source=self.source_id,
                response_time_ms=(time.time() - start_time) * 1000
            )

        result_data = {
            "symbol": symbol.upper(),
            "price": price,
            "change": change or 0.0,
            "change_percent": (change / (price - change) * 100) if change and (price - change) != 0 else 0.0,
            "volume": volume or 0,
            "timestamp": datetime.now().isoformat(),
            "source": self.source_id,
            "market": "vietnam"
        }

        self.request_count += 1
        response_time = (time.time() - start_time) * 1000

        return CrawlResult(
            success=True,
            data=result_data,
            source=self.source_id,
            response_time_ms=response_time
        )

    def _extract_price(self, soup: BeautifulSoup, symbol: str) -> Optional[float]:
        """Extract price using multiple selectors"""
        selectors = [
            {"id": "ctl00_ContentPlaceHolder1_LabelMaCP"},
            {"class": "price"},
            {"class": "stockprice"},
            {"class": "current-price"},
            {"id": f"price_{symbol}"},
        ]

        for selector in selectors:
            element = soup.find("span", selector)
            if element:
                try:
                    price_text = element.get_text().strip().replace(",", "").replace(" ", "")
                    # Remove currency symbols and extra characters
                    price_text = re.sub(r"[^\d.]", "", price_text)
                    return float(price_text)
                except (ValueError, AttributeError):
                    continue

        return None

    def _extract_change(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract price change"""
        selectors = [
            {"id": "ctl00_ContentPlaceHolder1_LabelChange"},
            {"class": "change"},
            {"class": "price-change"},
        ]

        for selector in selectors:
            element = soup.find("span", selector)
            if element:
                try:
                    change_text = element.get_text().strip().replace(",", "").replace(" ", "")
                    change_text = re.sub(r"[^\d.-]", "", change_text)
                    return float(change_text)
                except (ValueError, AttributeError):
                    continue

        return None

    def _extract_volume(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract trading volume"""
        selectors = [
            {"id": "ctl00_ContentPlaceHolder1_LabelVolume"},
            {"class": "volume"},
            {"class": "trading-volume"},
        ]

        for selector in selectors:
            element = soup.find("span", selector)
            if element:
                try:
                    volume_text = element.get_text().strip().replace(",", "").replace(" ", "")
                    volume_text = re.sub(r"[^\d]", "", volume_text)
                    return int(volume_text)
                except (ValueError, AttributeError):
                    continue

        return None

    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> CrawlResult:
        """CafeF doesn"t provide easy historical data access"""
        return CrawlResult(
            success=False,
            error="Historical data not available from CafeF source",
            source=self.source_id
        )

    async def fetch_market_index(self) -> CrawlResult:
        """Scrape VN-Index from CafeF homepage"""
        start_time = time.time()
        await self.respect_rate_limit(self.schema.rate_limits.requests_per_second)

        async with self.session.get(self.base_url) as response:
            if response.status != 200:
                return CrawlResult(
                    success=False,
                    error=f"HTTP {response.status}",
                    source=self.source_id,
                    response_time_ms=(time.time() - start_time) * 1000
                )

            html = await response.text()

        soup = BeautifulSoup(html, "html.parser")

        # Try multiple selectors for VN-Index
        index_value = self._extract_index_value(soup)

        if index_value is None:
            return CrawlResult(
                success=False,
                error="Could not extract VN-Index data",
                source=self.source_id,
                response_time_ms=(time.time() - start_time) * 1000
            )

        result_data = {
            "name": "VN-Index",
            "value": index_value,
            "change": 0.0,  # Would need more parsing
            "change_percent": 0.0,
            "timestamp": datetime.now().isoformat(),
            "source": self.source_id
        }

        self.request_count += 1
        response_time = (time.time() - start_time) * 1000

        return CrawlResult(
            success=True,
            data=result_data,
            source=self.source_id,
            response_time_ms=response_time
        )

    def _extract_index_value(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract VN-Index value using multiple selectors"""
        selectors = [
            {"class": "indexvalue"},
            {"class": "vnindex"},
            {"id": "vnindex_value"},
            {"class": "index-point"},
        ]

        for selector in selectors:
            element = soup.find("div", selector) or soup.find("span", selector)
            if element:
                try:
                    value_text = element.get_text().strip().replace(",", "").replace(" ", "")
                    value_text = re.sub(r"[^\d.]", "", value_text)
                    return float(value_text)
                except (ValueError, AttributeError):
                    continue

        return None


class VnExpressNewsAdapter(BaseSourceAdapter):
    """VnExpress financial news adapter"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        super().__init__(schema_manager)
        self.source_id = "vnexpress"
        self.rss_url = "https://vnexpress.net/rss/kinh-doanh.rss"

    async def fetch_stock_price(self, symbol: str) -> CrawlResult:
        """News source doesn't provide price data"""
        return CrawlResult(
            success=False,
            error="News source doesn't provide price data",
            source=self.source_id
        )

    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> CrawlResult:
        """News source doesn't provide historical data"""
        return CrawlResult(
            success=False,
            error="News source doesn't provide historical data",
            source=self.source_id
        )

    async def fetch_news(self, limit: int = 10) -> CrawlResult:
        """Fetch latest financial news"""
        start_time = time.time()

        # Use asyncio.to_thread for feedparser - let it propagate exceptions naturally
        feed = await asyncio.to_thread(feedparser.parse, self.rss_url)

        if not feed.entries:
            return CrawlResult(
                success=False,
                error="No news articles found",
                source=self.source_id,
                response_time_ms=(time.time() - start_time) * 1000
            )

        articles = []
        for entry in feed.entries[:limit]:
            article = {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "description": entry.get("description", ""),
                "published_at": entry.get("published", ""),
                "source": self.source_id,
                "category": "financial_news",
                "timestamp": datetime.now().isoformat()
            }
            articles.append(article)

        self.request_count += 1
        response_time = (time.time() - start_time) * 1000

        return CrawlResult(
            success=True,
            data={"articles": articles, "count": len(articles)},
            source=self.source_id,
            response_time_ms=response_time
        )


class InvestingComAdapter(BaseSourceAdapter):
    """Investing.com Vietnam section adapter"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        super().__init__(schema_manager)
        self.source_id = "investing_com"
        self.schema = self.schema_manager.get_source(self.source_id)
        self.base_url = self.schema.base_url

    async def initialize(self):
        """Initialize with proper headers"""
        if self.session is None:
            headers = self.schema.headers.copy()
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)

    async def fetch_stock_price(self, symbol: str) -> CrawlResult:
        """Limited stock data from Investing.com Vietnam section"""
        return CrawlResult(
            success=False,
            error="Individual stock prices not easily accessible",
            source=self.source_id
        )

    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> CrawlResult:
        """Historical data requires complex parsing"""
        return CrawlResult(
            success=False,
            error="Historical data requires premium access",
            source=self.source_id
        )

    async def fetch_market_index(self) -> CrawlResult:
        """Fetch VN-Index from Investing.com"""
        start_time = time.time()
        await self.respect_rate_limit(self.schema.rate_limits.requests_per_second)

        url = f"{self.base_url}/indices/vietnam-ho-chi-minh-stock-index"

        async with self.session.get(url) as response:
            if response.status != 200:
                return CrawlResult(
                    success=False,
                    error=f"HTTP {response.status}",
                    source=self.source_id,
                    response_time_ms=(time.time() - start_time) * 1000
                )

            html = await response.text()

        soup = BeautifulSoup(html, "html.parser")

        # Extract index data
        index_value = self._extract_investing_index(soup)

        if index_value is None:
            return CrawlResult(
                success=False,
                error="Could not extract VN-Index from Investing.com",
                source=self.source_id,
                response_time_ms=(time.time() - start_time) * 1000
            )

        result_data = {
            "name": "VN-Index",
            "value": index_value,
            "change": 0.0,
            "change_percent": 0.0,
            "timestamp": datetime.now().isoformat(),
            "source": self.source_id
        }

        self.request_count += 1
        response_time = (time.time() - start_time) * 1000

        return CrawlResult(
            success=True,
            data=result_data,
            source=self.source_id,
            response_time_ms=response_time
        )

    def _extract_investing_index(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract index value from Investing.com"""
        selectors = [
            {"data-test": "instrument-price-last"},
            {"class": "text-2xl"},
            {"class": "last-price-value"},
        ]

        for selector in selectors:
            element = soup.find("span", selector)
            if element:
                try:
                    value_text = element.get_text().strip().replace(",", "")
                    return float(value_text)
                except (ValueError, AttributeError):
                    continue

        return None


class SourceAdapterFactory:
    """Factory for creating source adapters"""

    @staticmethod
    def create_adapter(source_id: str, schema_manager: SourceSchemaManager = None) -> BaseSourceAdapter:
        """Create adapter for given source"""
        adapters = {
            "vnstock": VNStockAdapter,
            "cafef": CafeFAdapter,
            "vnexpress": VnExpressNewsAdapter,
            "investing_com": InvestingComAdapter,
        }

        adapter_class = adapters.get(source_id)
        if not adapter_class:
            raise ValueError(f"Unknown source ID: {source_id}")

        return adapter_class(schema_manager)

    @staticmethod
    def create_all_adapters(schema_manager: SourceSchemaManager = None) -> Dict[str, BaseSourceAdapter]:
        """Create all available adapters"""
        sm = schema_manager or get_schema_manager()
        adapters = {}

        for source_id in sm.schemas.keys():
            try:
                adapters[source_id] = SourceAdapterFactory.create_adapter(source_id, sm)
            except ValueError:
                logger.warning(f"No adapter available for source: {source_id}")

        return adapters


# Usage example and testing
async def test_adapters():
    """Test all adapters"""
    print("🧪 Testing Vietnamese Stock Market Source Adapters")
    print("=" * 60)

    adapters = SourceAdapterFactory.create_all_adapters()

    for source_id, adapter in adapters.items():
        print(f"\n📡 Testing {source_id}...")

        async with adapter:
            # Test stock price
            if hasattr(adapter, "fetch_stock_price"):
                result = await adapter.fetch_stock_price("VNM")
                status = "OK" if result.success else "ERROR"
                print(f"  Stock Price: {status} {result.response_time_ms:.0f}ms")
                if result.success and result.data:
                    print(f"    VNM Price: {result.data.get('price', 'N/A')}")

            # Test market index
            if hasattr(adapter, "fetch_market_index"):
                result = await adapter.fetch_market_index()
                status = "OK" if result.success else "ERROR"
                print(f"  Market Index: {status} {result.response_time_ms:.0f}ms")
                if result.success and result.data:
                    print(f"    VN-Index: {result.data.get('value', 'N/A')}")

            # Test news (if supported)
            if hasattr(adapter, "fetch_news"):
                result = await adapter.fetch_news(limit=5)
                status = "OK" if result.success else "ERROR"
                print(f"  News: {status} {result.response_time_ms:.0f}ms")
                if result.success and result.data:
                    print(f"    Articles: {result.data.get('count', 0)}")


if __name__ == "__main__":
    asyncio.run(test_adapters())
