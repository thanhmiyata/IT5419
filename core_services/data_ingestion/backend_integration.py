"""
Backend Integration Module
===========================

Provides integration with the backend API for data delivery.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from config import ConfigManager
from data_ingestion.pipeline_runner import PipelineOrchestrator
from data_ingestion.task_manager import Priority, TaskManager, TaskStatus, TaskTemplates
from fastapi import APIRouter, Depends, HTTPException

from core_services.utils.logger_utils import logger


class StockDataService:
    """
    High-level service for integrating stock data pipeline with FastAPI backend.
    Provides easy-to-use methods for common stock data operations.
    """

    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 config_path: Optional[str] = None,
                 auto_start: bool = True):
        """
        Initialize the stock data service.

        Args:
            redis_url: Redis connection URL
            config_path: Path to configuration file (optional)
            auto_start: Whether to auto-start the pipeline
        """
        self.redis_client = redis.from_url(redis_url)
        self.orchestrator: Optional[PipelineOrchestrator] = None
        self.task_manager: Optional[TaskManager] = None
        self._initialized = False

        # Load configuration
        if config_path:
            self.config = ConfigManager.load_from_file(config_path)
        else:
            self.config = ConfigManager.load_from_env()

        # Setup pipeline
        self.orchestrator = PipelineOrchestrator(self.config)

        if auto_start:
            asyncio.create_task(self._initialize_async())

    async def _initialize_async(self):
        """Initialize pipeline asynchronously"""
        if not self._initialized:
            try:
                await self.orchestrator.initialize()
                self.task_manager = self.orchestrator.task_manager
                self._initialized = True
                logger.info("StockDataService initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize StockDataService: {e}")

    async def ensure_initialized(self):
        """Ensure service is initialized"""
        if not self._initialized:
            await self._initialize_async()

    # High-level data retrieval methods

    async def get_stock_price(self,
                              symbol: str,
                              use_cache: bool = True,
                              max_age_seconds: int = 300) -> Optional[Dict[str, Any]]:
        """
        Get current stock price with caching.

        Args:
            symbol: Stock symbol (e.g., 'VNM', 'FPT')
            use_cache: Whether to use cached data
            max_age_seconds: Maximum age of cached data in seconds

        Returns:
            Dict with stock price data or None if failed
        """
        await self.ensure_initialized()

        # Check cache first if enabled
        if use_cache:
            cache_key = f"stock:{symbol}:latest"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                data = json.loads(cached_data.decode())

                # Check if data is still fresh
                if 'timestamp' in data:
                    cached_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    age = (datetime.now(cached_time.tzinfo) - cached_time).total_seconds()

                    if age <= max_age_seconds:
                        logger.info(f"Returning cached data for {symbol} (age: {age:.1f}s)")
                        return data

        # Cache miss or stale data - fetch fresh data
        logger.info(f"Fetching fresh data for {symbol}")
        task = TaskTemplates.create_stock_price_task(symbol, Priority.HIGH)
        task_id = await self.task_manager.submit_immediate_task(task)

        # Wait for completion with timeout
        for _ in range(30):  # 30 second timeout
            await asyncio.sleep(1)
            status = await self.task_manager.get_task_status(task_id)

            if status == TaskStatus.COMPLETED:
                # Try to get from cache again
                cache_key = f"stock:{symbol}:latest"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data.decode())
                break
            elif status == TaskStatus.FAILED:
                logger.error(f"Failed to fetch data for {symbol}")
                break

        return None

    async def get_multiple_stocks(self,
                                  symbols: List[str],
                                  use_cache: bool = True,
                                  max_age_seconds: int = 300) -> Dict[str, Any]:
        """
        Get data for multiple stocks in parallel.

        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data
            max_age_seconds: Maximum age of cached data

        Returns:
            Dict mapping symbols to their data
        """
        await self.ensure_initialized()

        # Submit tasks for all symbols
        tasks = []
        for symbol in symbols:
            task = self.get_stock_price(symbol, use_cache, max_age_seconds)
            tasks.append((symbol, task))

        # Wait for all tasks to complete
        results = {}
        for symbol, task in tasks:
            try:
                data = await task
                results[symbol] = data
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                results[symbol] = None

        return results

    async def get_market_index(self,
                               use_cache: bool = True,
                               max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Get VN-Index data.

        Args:
            use_cache: Whether to use cached data
            max_age_seconds: Maximum age of cached data

        Returns:
            Dict with index data or None if failed
        """
        await self.ensure_initialized()

        # Check cache
        if use_cache:
            cache_key = "stock:index:latest"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                data = json.loads(cached_data.decode())
                if 'timestamp' in data:
                    cached_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    age = (datetime.now(cached_time.tzinfo) - cached_time).total_seconds()

                    if age <= max_age_seconds:
                        return data

        # Fetch fresh data
        task = TaskTemplates.create_index_data_task(Priority.HIGH)
        task_id = await self.task_manager.submit_immediate_task(task)

        # Wait for completion
        for _ in range(30):
            await asyncio.sleep(1)
            status = await self.task_manager.get_task_status(task_id)

            if status == TaskStatus.COMPLETED:
                cache_key = "stock:index:latest"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data.decode())
                break
            elif status == TaskStatus.FAILED:
                break

        return None

    async def get_top_stocks(self,
                             category: str = "volume",
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top stocks by category.

        Args:
            category: 'volume', 'gainers', or 'losers'
            limit: Number of stocks to return

        Returns:
            List of top stocks
        """
        await self.ensure_initialized()

        # This would integrate with your existing database models
        # For now, return mock data from popular Vietnamese stocks
        mock_data = {
            "volume": [
                {"symbol": "HPG", "volume": 15200000, "price": 28500, "change_percent": 2.15},
                {"symbol": "VCB", "volume": 8500000, "price": 88000, "change_percent": -0.85},
                {"symbol": "VIC", "volume": 6200000, "price": 62500, "change_percent": 1.30},
                {"symbol": "VHM", "volume": 5800000, "price": 48200, "change_percent": -1.45},
                {"symbol": "MSN", "volume": 5200000, "price": 125000, "change_percent": 0.95},
            ],
            "gainers": [
                {"symbol": "VHM", "price": 48200, "change_value": 1200, "change_percent": 2.55},
                {"symbol": "MSN", "price": 125000, "change_value": 2800, "change_percent": 2.29},
                {"symbol": "HPG", "price": 28500, "change_value": 600, "change_percent": 2.15},
                {"symbol": "VIC", "price": 62500, "change_value": 800, "change_percent": 1.30},
                {"symbol": "CTG", "price": 35200, "change_value": 450, "change_percent": 1.30},
            ],
            "losers": [
                {"symbol": "VCB", "price": 88000, "change_value": -750, "change_percent": -0.85},
                {"symbol": "BID", "price": 42500, "change_value": -600, "change_percent": -1.39},
                {"symbol": "GAS", "price": 78500, "change_value": -1200, "change_percent": -1.51},
                {"symbol": "VNM", "price": 92000, "change_value": -1800, "change_percent": -1.92},
                {"symbol": "FPT", "price": 88500, "change_value": -2100, "change_percent": -2.32},
            ]
        }

        return mock_data.get(category, [])[:limit]

    # Pipeline management methods

    async def start_realtime_monitoring(self, symbols: List[str]) -> bool:
        """
        Start real-time monitoring for specific stocks.

        Args:
            symbols: List of stock symbols to monitor

        Returns:
            True if successfully started
        """
        await self.ensure_initialized()

        try:
            for symbol in symbols:
                await self.task_manager.schedule_interval_task(
                    name=f"Realtime-{symbol}",
                    task_template=TaskTemplates.create_stock_price_task(symbol, Priority.HIGH),
                    interval_seconds=30  # Every 30 seconds
                )

            logger.info(f"Started real-time monitoring for {len(symbols)} stocks")
            return True

        except Exception as e:
            logger.error(f"Failed to start real-time monitoring: {e}")
            return False

    async def stop_realtime_monitoring(self, symbols: List[str] = None) -> bool:
        """
        Stop real-time monitoring for specific stocks or all stocks.

        Args:
            symbols: List of symbols to stop monitoring (None for all)

        Returns:
            True if successfully stopped
        """
        await self.ensure_initialized()

        try:
            # Get all scheduled tasks
            scheduled_tasks = self.task_manager.scheduled_tasks

            for task_id, task in scheduled_tasks.items():
                # Check if this is a realtime monitoring task
                if task.name.startswith("Realtime-"):
                    if symbols is None:
                        # Stop all realtime tasks
                        await self.task_manager.cancel_scheduled_task(task_id)
                    else:
                        # Check if this task is for one of the specified symbols
                        for symbol in symbols:
                            if f"Realtime-{symbol}" == task.name:
                                await self.task_manager.cancel_scheduled_task(task_id)
                                break

            logger.info("Stopped real-time monitoring")
            return True

        except Exception as e:
            logger.error(f"Failed to stop real-time monitoring: {e}")
            return False

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        await self.ensure_initialized()

        status = await self.orchestrator.get_status()

        # Add service-specific information
        status['service'] = {
            'initialized': self._initialized,
            'redis_connected': bool(self.redis_client),
            'config_environment': self.config.environment
        }

        return status

    async def get_task_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent task execution history."""
        await self.ensure_initialized()
        return await self.task_manager.get_task_history(limit)

    # Cleanup methods

    async def shutdown(self):
        """Shutdown the service gracefully."""
        if self.orchestrator:
            await self.orchestrator.stop()

        if self.redis_client:
            self.redis_client.close()

        logger.info("StockDataService shutdown complete")


# Convenience functions for FastAPI integration

async def get_stock_service() -> StockDataService:
    """Dependency injection for FastAPI"""
    if not hasattr(get_stock_service, '_service'):
        get_stock_service._service = StockDataService()

    return get_stock_service._service


# FastAPI endpoint handlers
async def _handle_get_stock_price(
    symbol: str,
    use_cache: bool,
    max_age: int,
    service: StockDataService
):
    """Handle individual stock price request"""
    data = await service.get_stock_price(
        symbol=symbol.upper(),
        use_cache=use_cache,
        max_age_seconds=max_age
    )

    if data is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found or unavailable")

    return data


async def _handle_get_multiple_prices(
    symbols,
    use_cache: bool,
    max_age: int,
    service: StockDataService
):
    """Handle multiple stock prices request"""
    symbols_upper = [s.upper() for s in symbols]
    return await service.get_multiple_stocks(
        symbols=symbols_upper,
        use_cache=use_cache,
        max_age_seconds=max_age
    )


async def _handle_get_market_index(
    use_cache: bool,
    max_age: int,
    service: StockDataService
):
    """Handle market index request"""
    data = await service.get_market_index(
        use_cache=use_cache,
        max_age_seconds=max_age
    )

    if data is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Index data not available")

    return data


async def _handle_get_top_stocks(
    category: str,
    limit: int,
    service: StockDataService
):
    """Handle top stocks request"""
    if category not in ['volume', 'gainers', 'losers']:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Category must be volume, gainers, or losers")

    return await service.get_top_stocks(category=category, limit=limit)


async def _handle_start_monitoring(
    symbols,
    service: StockDataService
):
    """Handle start monitoring request"""
    success = await service.start_realtime_monitoring([s.upper() for s in symbols])
    return {"success": success, "message": f"Started monitoring {len(symbols)} stocks"}


async def _handle_stop_monitoring(
    symbols,
    service: StockDataService
):
    """Handle stop monitoring request"""
    symbols_upper = [s.upper() for s in symbols] if symbols else None
    success = await service.stop_realtime_monitoring(symbols_upper)
    return {"success": success, "message": "Stopped monitoring"}


def _create_error_handler(operation_name: str):
    """Create standardized error handler"""
    def handle_error(e: Exception):
        from fastapi import HTTPException
        logger.error(f"Error in {operation_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    return handle_error


# Individual endpoint functions
async def _stock_price_endpoint(
    symbol: str,
    use_cache: bool = True,
    max_age: int = 300,
    service: StockDataService = None
):
    """Get current stock price"""
    try:
        return await _handle_get_stock_price(symbol, use_cache, max_age, service)
    except HTTPException:
        raise
    except Exception as e:
        _create_error_handler("get_stock_price")(e)


async def _multiple_prices_endpoint(
    symbols: List[str],
    use_cache: bool = True,
    max_age: int = 300,
    service: StockDataService = None
):
    """Get prices for multiple stocks"""
    try:
        return await _handle_get_multiple_prices(symbols, use_cache, max_age, service)
    except Exception as e:
        _create_error_handler("get_multiple_prices")(e)


async def _market_index_endpoint(
    use_cache: bool = True,
    max_age: int = 60,
    service: StockDataService = None
):
    """Get VN-Index data"""
    try:
        return await _handle_get_market_index(use_cache, max_age, service)
    except HTTPException:
        raise
    except Exception as e:
        _create_error_handler("get_market_index")(e)


async def _top_stocks_endpoint(
    category: str,
    limit: int = 10,
    service: StockDataService = None
):
    """Get top stocks by category (volume/gainers/losers)"""
    try:
        return await _handle_get_top_stocks(category, limit, service)
    except HTTPException:
        raise
    except Exception as e:
        _create_error_handler("get_top_stocks")(e)


async def _pipeline_status_endpoint(service: StockDataService = None):
    """Get pipeline status"""
    try:
        return await service.get_pipeline_status()
    except Exception as e:
        _create_error_handler("get_pipeline_status")(e)


async def _start_monitoring_endpoint(
    symbols: List[str],
    service: StockDataService = None
):
    """Start real-time monitoring for stocks"""
    try:
        return await _handle_start_monitoring(symbols, service)
    except Exception as e:
        _create_error_handler("start_monitoring")(e)


async def _stop_monitoring_endpoint(
    symbols: Optional[List[str]] = None,
    service: StockDataService = None
):
    """Stop real-time monitoring"""
    try:
        return await _handle_stop_monitoring(symbols, service)
    except Exception as e:
        _create_error_handler("stop_monitoring")(e)


def _register_stock_endpoints(router):
    """Register all stock endpoints to the router"""

    @router.get("/price/{symbol}")
    async def get_stock_price_endpoint(
        symbol: str,
        use_cache: bool = True,
        max_age: int = 300,
        service: StockDataService = Depends(get_stock_service)
    ):
        return await _stock_price_endpoint(symbol, use_cache, max_age, service)

    @router.post("/prices")
    async def get_multiple_prices_endpoint(
        symbols: List[str],
        use_cache: bool = True,
        max_age: int = 300,
        service: StockDataService = Depends(get_stock_service)
    ):
        return await _multiple_prices_endpoint(symbols, use_cache, max_age, service)

    @router.get("/index")
    async def get_market_index_endpoint(
        use_cache: bool = True,
        max_age: int = 60,
        service: StockDataService = Depends(get_stock_service)
    ):
        return await _market_index_endpoint(use_cache, max_age, service)

    @router.get("/top/{category}")
    async def get_top_stocks_endpoint(
        category: str,
        limit: int = 10,
        service: StockDataService = Depends(get_stock_service)
    ):
        return await _top_stocks_endpoint(category, limit, service)

    @router.get("/pipeline/status")
    async def get_pipeline_status_endpoint(
        service: StockDataService = Depends(get_stock_service)
    ):
        return await _pipeline_status_endpoint(service)

    @router.post("/monitoring/start")
    async def start_monitoring_endpoint(
        symbols: List[str],
        service: StockDataService = Depends(get_stock_service)
    ):
        return await _start_monitoring_endpoint(symbols, service)

    @router.post("/monitoring/stop")
    async def stop_monitoring_endpoint(
        symbols: Optional[List[str]] = None,
        service: StockDataService = Depends(get_stock_service)
    ):
        return await _stop_monitoring_endpoint(symbols, service)


# FastAPI router integration example
def create_stock_data_router():
    """
    Create FastAPI router with stock data endpoints.
    Use this in your main FastAPI app.
    """
    router = APIRouter(prefix="/api/v1/stocks", tags=["stock-data"])
    _register_stock_endpoints(router)
    return router


# Example usage in main FastAPI app
if __name__ == "__main__":
    # Example of how to integrate with existing FastAPI app
    from fastapi import FastAPI

    app = FastAPI(title="Vietnamese Stock Data API")

    # Add stock data router
    stock_router = create_stock_data_router()
    app.include_router(stock_router)

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        # Initialize stock service
        service = await get_stock_service()
        await service.ensure_initialized()

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        service = await get_stock_service()
        await service.shutdown()

    # Run the API server
    # uvicorn.run(app, host="0.0.0.0", port=8000)
