"""
Pipeline Architecture
====================

Provides core pipeline components and execution framework.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import redis
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential
from vnstock import Vnstock

from core_services.data_ingestion.data_normalization import DataNormalizer
from core_services.data_ingestion.database_storage import create_database_storage
from core_services.data_ingestion.source_schema import SourceSchemaManager, get_schema_manager
from core_services.utils.common import (REDIS_KEY_PREFIX_STOCK, REDIS_KEY_PREFIX_TASK, REDIS_QUEUE_HIGH_PRIORITY,
                                        REDIS_QUEUE_LOW_PRIORITY, REDIS_QUEUE_MEDIUM_PRIORITY, REDIS_TTL_STOCK_PRICE,
                                        DataCategory, DataSourceType, DataType, Priority)
from core_services.utils.logger_utils import logger

if TYPE_CHECKING:
    pass


@dataclass
class CrawlTask:
    """Individual crawl task definition"""
    id: str
    source: DataSourceType
    data_type: DataType
    symbol: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None


@dataclass
class CrawlResult:
    """Result of a crawl operation"""
    task_id: str
    source: DataSourceType
    data_type: DataType
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0


class DataSourceInterface(ABC):
    """Abstract interface for all data sources"""

    @abstractmethod
    async def fetch_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Fetch stock price data"""

    @abstractmethod
    async def fetch_index_data(self) -> Dict[str, Any]:
        """Fetch market index data"""

    @abstractmethod
    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch historical price data"""

    @abstractmethod
    def get_rate_limit(self) -> int:
        """Return rate limit in requests per second"""

    @abstractmethod
    def get_reliability_score(self) -> float:
        """Return reliability score (0.0 - 1.0)"""


class VNStockSource(DataSourceInterface):
    """VNSTOCK library data source implementation with schema integration"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        self.schema_manager = schema_manager or get_schema_manager()
        self.schema = self.schema_manager.get_source("vnstock")
        self.name = self.schema.name if self.schema else "VNSTOCK"

        # Load configuration from schema
        if self.schema:
            self.rate_limit = self.schema.rate_limits.requests_per_second
            self.reliability_score = self.schema.success_rate
            self.capabilities = self.schema.capabilities
        else:
            # Fallback values
            self.rate_limit = 5
            self.reliability_score = 0.95
            self.capabilities = {}

    async def fetch_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Fetch current stock price using VNSTOCK"""
        try:
            # Import vnstock dynamically to avoid dependency issues
            from datetime import timedelta

            from vnstock import Vnstock

            # Get latest available data (last 30 days to ensure we get data)
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            quote = stock.quote.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

            if quote is not None and not quote.empty:
                latest = quote.iloc[-1]

                return {
                    "symbol": symbol,
                    "price": float(latest['close']),
                    "open": float(latest['open']),
                    "high": float(latest['high']),
                    "low": float(latest['low']),
                    "close": float(latest['close']),
                    "volume": int(latest['volume']),
                    "timestamp": (
                        str(quote.iloc[-1]['time'])
                        if 'time' in quote.columns
                        else datetime.now().isoformat()
                    ),
                    "source": "vnstock",
                    "date": (
                        str(quote.iloc[-1]['time']).split()[0]
                        if "time" in quote.columns
                        else datetime.now().strftime("%Y-%m-%d")
                    )
                }
            else:
                logger.warning(f"No data available for {symbol}")
                raise ValueError(f"No data available for {symbol}")

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"VNStock fetch failed for {symbol}: {e}")
            raise ValueError(f"VNStock API error for {symbol}: {str(e)}")

    async def fetch_index_data(self) -> Dict[str, Any]:
        """Fetch VN-Index data"""
        try:
            from datetime import timedelta

            from vnstock import Vnstock

            # Get index data (last 30 days to ensure we get data)
            stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            quote = stock.quote.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

            if quote is not None and not quote.empty:
                latest = quote.iloc[-1]

                return {
                    "name": "VN-Index",
                    "value": float(latest['close']),
                    "change": 0,  # Would need previous day data to calculate
                    "change_percent": 0,
                    "volume": int(latest.get("volume", 0)) if "volume" in latest else None,
                    "timestamp": (
                        str(quote.iloc[-1]["time"])
                        if "time" in quote.columns
                        else datetime.now().isoformat()
                    ),
                    "date": (
                        str(quote.iloc[-1]["time"]).split()[0]
                        if "time" in quote.columns
                        else datetime.now().strftime("%Y-%m-%d")
                    ),
                    "source": "vnstock"
                }
            else:
                logger.warning("No VN-Index data available")
                raise ValueError("No VN-Index data available")

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"VNStock index fetch failed: {e}")
            raise ValueError(f"VNStock API error for VNINDEX: {str(e)}")

    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch historical data"""
        try:
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            stock = Vnstock().stock(symbol=symbol, source="VCI")
            df = stock.quote.history(start=start_date, end=end_date)

            if df is not None and not df.empty:
                records = df.to_dict("records")
                # Add symbol to each record since it's not in the DataFrame
                for record in records:
                    record["symbol"] = symbol
                logger.info(f"Successfully fetched {len(records)} historical records for {symbol}")
                return records
            else:
                logger.warning(f"No historical data available for {symbol}")
                return []
        except Exception as e:
            logger.error(f"VNStock historical fetch failed for {symbol}: {e}")
            raise

    def get_rate_limit(self) -> int:
        return self.rate_limit

    def get_reliability_score(self) -> float:
        return self.reliability_score


class CafeFSource(DataSourceInterface):
    """CafeF.vn web scraping data source with schema integration"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        self.schema_manager = schema_manager or get_schema_manager()
        self.schema = self.schema_manager.get_source("cafef")
        self.name = self.schema.name if self.schema else "CAFEF"

        # Load configuration from schema
        if self.schema:
            self.rate_limit = self.schema.rate_limits.requests_per_second
            self.reliability_score = self.schema.success_rate
            self.base_url = self.schema.base_url
            self.capabilities = self.schema.capabilities
            self.headers = self.schema.headers
        else:
            # Fallback values
            self.rate_limit = 2
            self.reliability_score = 0.85
            self.base_url = "https://cafef.vn"
            self.capabilities = {}
            self.headers = {}

    async def fetch_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Scrape stock price from CafeF"""
        try:
            import aiohttp
            from bs4 import BeautifulSoup

            url = f"{self.base_url}/{symbol}.chn"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        # Extract price data (implement specific parsing logic)
                        price_element = soup.find("span", {"class": "price"})
                        price = float(price_element.text.strip()) if price_element else 0

                        return {
                            "symbol": symbol,
                            "price": price,
                            "timestamp": datetime.now().isoformat(),
                            "source": "cafef"
                        }
        except Exception as e:
            logger.error(f"CafeF fetch failed for {symbol}: {e}")
            raise

    async def fetch_index_data(self) -> Dict[str, Any]:
        """Scrape index data from CafeF homepage"""
        # Implementation for index scraping

    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch historical data from CafeF"""
        # Implementation for historical data

    def get_rate_limit(self) -> int:
        return self.rate_limit

    def get_reliability_score(self) -> float:
        return self.reliability_score


class TaskScheduler:
    """Manages task scheduling and priority queues"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.priority_queues = {
            Priority.HIGH: REDIS_QUEUE_HIGH_PRIORITY,
            Priority.MEDIUM: REDIS_QUEUE_MEDIUM_PRIORITY,
            Priority.LOW: REDIS_QUEUE_LOW_PRIORITY
        }

    async def schedule_task(self, task: CrawlTask) -> bool:
        """Add task to appropriate priority queue"""
        try:
            queue_name = self.priority_queues[task.priority]
            task_data = {
                "id": task.id,
                "source": task.source.value,
                "data_type": task.data_type.value,
                "symbol": task.symbol,
                "params": json.dumps(task.params),
                "created_at": task.created_at.isoformat(),
                "retry_count": task.retry_count
            }

            self.redis.lpush(queue_name, json.dumps(task_data))
            logger.info(f"Task {task.id} scheduled in {queue_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to schedule task {task.id}: {e}")
            return False

    async def get_next_task(self) -> Optional[CrawlTask]:
        """Get next task from highest priority queue"""
        for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            queue_name = self.priority_queues[priority]
            task_id = self.redis.rpop(queue_name)

            if task_id:
                # Decode task_id if it's bytes
                if isinstance(task_id, bytes):
                    task_id = task_id.decode()

                # Fetch task details from hash
                task_key = f"{REDIS_KEY_PREFIX_TASK}{task_id}"
                task_data = self.redis.hgetall(task_key)

                if not task_data:
                    logger.warning(f"Task {task_id} not found in Redis")
                    continue

                # Decode bytes to strings if necessary
                task_dict = {}
                for key, value in task_data.items():
                    k = key.decode() if isinstance(key, bytes) else key
                    v = value.decode() if isinstance(value, bytes) else value
                    task_dict[k] = v

                # Parse task dict into CrawlTask
                return CrawlTask(
                    id=task_dict["id"],
                    source=DataSourceType(task_dict["source"]),
                    data_type=DataType(task_dict["data_type"]),
                    symbol=task_dict.get("symbol"),
                    params=json.loads(task_dict["params"]) if task_dict.get("params") else {},
                    retry_count=int(task_dict.get("retry_count", 0)),
                    created_at=datetime.fromisoformat(task_dict["created_at"]),
                    priority=Priority(int(task_dict.get("priority", Priority.MEDIUM.value)))
                )

        return None


class DataValidator:
    """Validates crawled data for quality and consistency"""

    @staticmethod
    def validate_stock_price(data: Dict[str, Any]) -> bool:
        """Validate stock price data"""
        try:
            required_fields = ["symbol", "price", "timestamp"]

            # Check required fields
            if not all(field in data for field in required_fields):
                return False

            # Validate price is positive
            price = float(data["price"])
            if price <= 0:
                return False

            # Validate timestamp format
            datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

            # Additional business logic validations
            # - Price within reasonable range for Vietnamese stocks
            if price > 1000000:  # 1M VND per share seems unreasonable
                return False

            return True

        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Validation failed: {e}")
            return False

    @staticmethod
    def validate_index_data(data: Dict[str, Any]) -> bool:
        """Validate index data"""
        try:
            required_fields = ["name", "value", "timestamp"]
            return all(field in data for field in required_fields)
        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False


class DataMerger:
    """Merges data from multiple sources with schema-based conflict resolution"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        self.schema_manager = schema_manager or get_schema_manager()

        # Calculate source weights based on schema data
        self.source_weights = self._calculate_source_weights()

    def _calculate_source_weights(self) -> Dict[str, float]:
        """Calculate source weights based on schema reliability and quality"""
        weights = {}
        total_score = 0

        # Calculate scores for each source
        source_scores = {}
        for source_id, schema in self.schema_manager.schemas.items():
            # Score based on reliability, success rate, and response time
            reliability_score = {
                "very_high": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            }.get(schema.reliability.value, 0.5)

            performance_score = (
                schema.success_rate * 0.5
                + reliability_score * 0.3
                + max(0, 1 - schema.average_response_time_ms / 5000) * 0.2
            )

            source_scores[source_id] = performance_score
            total_score += performance_score

        # Normalize to weights
        if total_score > 0:
            for source_id, score in source_scores.items():
                weights[source_id] = score / total_score

        return weights

    def merge_stock_data(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge stock data from multiple sources using schema-based weights"""
        if not data_list:
            return {}

        if len(data_list) == 1:
            return data_list[0]

        # Weighted average for numerical fields
        merged = {}
        total_weight = 0

        for data in data_list:
            source_id = data.get("source", "unknown")

            # Get weight from schema-based calculation
            weight = self.source_weights.get(source_id, 0.1)

            # Additional weight adjustment based on data quality for this specific data type
            if source_id in self.schema_manager.schemas:
                schema = self.schema_manager.schemas[source_id]
                if DataCategory.REAL_TIME_PRICES in schema.capabilities:
                    capability = schema.capabilities[DataCategory.REAL_TIME_PRICES]
                    quality_multiplier = capability.data_quality_score
                    weight *= quality_multiplier

            total_weight += weight

            for key, value in data.items():
                if key in ["price", "volume", "change", "change_percent"]:
                    if key not in merged:
                        merged[key] = 0
                    try:
                        merged[key] += float(value) * weight
                    except (ValueError, TypeError):
                        pass
                elif key not in merged:
                    merged[key] = value

        # Normalize weighted values
        if total_weight > 0:
            for key in ["price", "volume", "change", "change_percent"]:
                if key in merged:
                    merged[key] = round(merged[key] / total_weight, 2)

        merged["sources"] = [data.get("source") for data in data_list]
        merged["merge_timestamp"] = datetime.now().isoformat()
        merged["merge_method"] = "schema_weighted_average"
        merged["total_weight"] = total_weight

        return merged


class CrawlingPipeline:
    """Main pipeline orchestrator"""

    def __init__(self,
                 data_sources: Dict[DataSourceType, DataSourceInterface],
                 scheduler: TaskScheduler,
                 validator: DataValidator,
                 merger: DataMerger,
                 db_session: Session,
                 redis_client: redis.Redis,
                 normalizer: Optional[DataNormalizer] = None):
        self.sources = data_sources
        self.scheduler = scheduler
        self.validator = validator
        self.merger = merger
        self.normalizer = normalizer or DataNormalizer()
        self.db = db_session
        self.redis = redis_client
        self.is_running = False
        self.worker_tasks = []

    async def start(self, num_workers: int = 4):
        """Start the crawling pipeline"""
        self.is_running = True

        # Start worker tasks
        for i in range(num_workers):
            worker_task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker_task)

        logger.info(f"Pipeline started with {num_workers} workers")

    async def stop(self):
        """Stop the crawling pipeline"""
        self.is_running = False

        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        logger.info("Pipeline stopped")

    async def _worker(self, worker_id: str):
        """Individual worker process"""
        logger.info(f"Worker {worker_id} started")

        while self.is_running:
            try:
                # Get next task
                task = await self.scheduler.get_next_task()

                if not task:
                    await asyncio.sleep(1)  # No tasks available
                    continue

                # Process task
                result = await self._process_task(task)

                # Handle result
                await self._handle_result(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(5)  # Brief pause on error

        logger.info(f"Worker {worker_id} stopped")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _process_task(self, task: CrawlTask) -> CrawlResult:
        """Process individual crawl task"""
        start_time = datetime.now()
        logger.info(f"Processing task: {task.data_type.value} for symbol: {task.symbol}, params: {task.params}")
        try:
            source = self.sources.get(task.source)
            if not source:
                raise ValueError(f"Unknown source: {task.source}")

            # Execute based on data type
            if task.data_type == DataType.STOCK_PRICE:
                data = await source.fetch_stock_price(task.symbol)
            elif task.data_type == DataType.INDEX_DATA:
                data = await source.fetch_index_data()
            elif task.data_type == DataType.HISTORICAL_DATA:
                # Extract start_date and end_date from task params
                start_date = task.params.get("start_date")
                end_date = task.params.get("end_date")
                logger.info(f"Extracted dates from params - start: {start_date}, end: {end_date}")
                if not start_date or not end_date:
                    raise ValueError("Historical data requires start_date and end_date in params")
                data = await source.fetch_historical_data(task.symbol, start_date, end_date)
            else:
                raise ValueError(f"Unsupported data type: {task.data_type}")

            # Validate data
            is_valid = False
            if task.data_type == DataType.STOCK_PRICE:
                is_valid = self.validator.validate_stock_price(data)
            elif task.data_type == DataType.INDEX_DATA:
                is_valid = self.validator.validate_index_data(data)
            elif task.data_type == DataType.HISTORICAL_DATA:
                # For historical data (list of records), validate it's a non-empty list
                is_valid = isinstance(data, list) and len(data) > 0
                if is_valid:
                    logger.info(f"Historical data fetched: {len(data)} records for {task.symbol}")
                else:
                    logger.warning(f"Historical data validation failed: got {type(data)} with "
                                   f"{len(data) if isinstance(data, list) else 'N/A'} records")

            if not is_valid:
                raise ValueError("Data validation failed")

            processing_time = (datetime.now() - start_time).total_seconds()
            return CrawlResult(
                task_id=task.id,
                source=task.source,
                data_type=task.data_type,
                success=True,
                data=data,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()

            return CrawlResult(
                task_id=task.id,
                source=task.source,
                data_type=task.data_type,
                success=False,
                error=str(e),
                processing_time=processing_time
            )

    async def _handle_result(self, result: CrawlResult):
        """Handle crawl result with normalization before storage"""
        try:
            if result.success:
                # Normalize data before caching and storage
                normalized_data = self._normalize_result_data(result)

                # Determine symbol for caching
                if isinstance(normalized_data, list):
                    # Historical data - use symbol from task or first record
                    symbol = normalized_data[0].get("symbol", "unknown") if normalized_data else "unknown"
                else:
                    # Single record - get symbol directly
                    symbol = normalized_data.get("symbol", "index")

                # Cache in Redis for fast access (serialize with date handling)
                cache_key = f"{REDIS_KEY_PREFIX_STOCK}{symbol}:latest"
                # Convert any datetime/Timestamp objects to ISO format strings for JSON serialization
                cache_data = json.dumps(normalized_data, default=str)
                self.redis.setex(cache_key, REDIS_TTL_STOCK_PRICE, cache_data)

                # Store in database if storage handler is configured
                if hasattr(self, 'db_storage') and self.db_storage:
                    # Update result with normalized data before storage
                    normalized_result = CrawlResult(
                        task_id=result.task_id,
                        source=result.source,
                        data_type=result.data_type,
                        success=result.success,
                        data=normalized_data,
                        error=result.error,
                        timestamp=result.timestamp,
                        processing_time=result.processing_time
                    )
                    await asyncio.to_thread(
                        self.db_storage.store_crawl_result,
                        normalized_result,
                        self.db
                    )

                logger.info(f"Task {result.task_id} completed successfully (normalized)")
            else:
                logger.error(f"Task {result.task_id} failed: {result.error}")

                # TODO: Implement retry logic for failed tasks

        except Exception as e:
            logger.error(f"Failed to handle result for task {result.task_id}: {e}")

    def _normalize_result_data(self, result: CrawlResult) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Normalize crawl result data based on data type.

        Args:
            result: CrawlResult with raw data

        Returns:
            Normalized data (single dict or list of dicts)
        """
        try:
            if result.data_type == DataType.STOCK_PRICE:
                # Single stock price record
                return self.normalizer.normalize_stock_price(result.data)

            elif result.data_type == DataType.INDEX_DATA:
                # Market index record
                return self.normalizer.normalize_index_data(result.data)

            elif result.data_type == DataType.HISTORICAL_DATA:
                # List of historical records
                if isinstance(result.data, list):
                    return self.normalizer.normalize_historical_data(result.data)
                else:
                    logger.warning(f"Historical data is not a list: {type(result.data)}")
                    return result.data

            else:
                # Unknown type - return as is
                logger.warning(f"Unknown data type for normalization: {result.data_type}")
                return result.data

        except Exception as e:
            logger.error(f"Normalization failed for {result.data_type}: {e}")
            # Return original data if normalization fails
            return result.data


# Pipeline Factory
class PipelineFactory:
    """Factory for creating pipeline instances with schema integration"""

    @staticmethod
    def create_pipeline(redis_url: str = "redis://localhost:6379",
                        db_session: Session = None,
                        database_url: str = None,
                        schema_manager: SourceSchemaManager = None,
                        task_manager=None) -> CrawlingPipeline:
        """Create a fully configured pipeline instance with schema-based sources"""

        # Initialize schema manager
        if not schema_manager:
            schema_manager = get_schema_manager()

        # Initialize Redis client
        redis_client = redis.from_url(redis_url)

        # Initialize data sources based on schema
        sources = {}

        # Add VNSTOCK if available in schema
        if schema_manager.get_source("vnstock"):
            sources[DataSourceType.VNSTOCK] = VNStockSource(schema_manager)

        # Add CafeF if available in schema
        if schema_manager.get_source("cafef"):
            sources[DataSourceType.CAFEF] = CafeFSource(schema_manager)

        # Initialize components with schema integration
        # Use provided task_manager or fall back to TaskScheduler (for backwards compatibility)
        if task_manager:
            scheduler = task_manager
        else:
            scheduler = TaskScheduler(redis_client)

        validator = DataValidator()
        merger = DataMerger(schema_manager)
        normalizer = DataNormalizer()  # Initialize data normalizer

        pipeline = CrawlingPipeline(
            data_sources=sources,
            scheduler=scheduler,
            validator=validator,
            merger=merger,
            db_session=db_session,
            redis_client=redis_client,
            normalizer=normalizer
        )

        # Store schema manager reference
        pipeline.schema_manager = schema_manager

        # Initialize database storage if URL provided
        if database_url:
            try:
                pipeline.db_storage = create_database_storage(database_url)
                logger.info("Database storage initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize database storage: {e}")

        return pipeline


if __name__ == "__main__":
    # Example usage
    from data_ingestion.config import PipelineConfig

    config = PipelineConfig()
    pipeline = PipelineFactory.create_pipeline(config.redis_url)

    # This would be run by your main application
    # asyncio.run(pipeline.start(config.num_workers))
