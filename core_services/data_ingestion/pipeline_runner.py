"""
Pipeline Runner
===============

Provides the main pipeline execution and coordination.
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional

import redis

from core_services.data_ingestion.config import ConfigManager, ConfigPresets, MainConfig
from core_services.data_ingestion.pipeline_architecture import (CrawlingPipeline, CrawlTask, DataSourceType, DataType,
                                                                PipelineFactory)
from core_services.data_ingestion.task_manager import Priority, TaskManager, TaskTemplates
from core_services.utils.logger_utils import logger


class PipelineOrchestrator:
    """Main pipeline orchestrator that manages all components"""

    def __init__(self, config: MainConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.task_manager: Optional[TaskManager] = None
        self.pipeline: Optional[CrawlingPipeline] = None
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")

        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.config.redis.url)
            await self._test_redis_connection()

            # Initialize task manager
            self.task_manager = TaskManager(self.redis_client)

            # Initialize main pipeline with database storage
            # Get database URL from environment or config
            import os
            database_url = os.getenv(
                'DATABASE_URL',
                'postgresql://user:password@localhost:5432/stockaids'
            )

            self.pipeline = PipelineFactory.create_pipeline(
                redis_url=self.config.redis.url,
                database_url=database_url,
                db_session=None,  # Session managed by DatabaseStorage
                task_manager=self.task_manager  # Pass the task_manager
            )

            # Setup scheduled tasks
            await self._setup_scheduled_tasks()

            logger.info("Pipeline initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    async def _test_redis_connection(self):
        """Test Redis connectivity"""
        try:
            await asyncio.to_thread(self.redis_client.ping)
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    async def _setup_scheduled_tasks(self):
        """Setup recurring scheduled tasks"""
        logger.info("Setting up scheduled tasks...")

        scheduler_config = self.config.scheduler

        # Schedule realtime stock updates (high frequency)
        for symbol in scheduler_config.realtime_stocks:
            await self.task_manager.schedule_interval_task(
                name=f"Realtime-{symbol}",
                task_template=TaskTemplates.create_stock_price_task(symbol, Priority.HIGH),
                interval_seconds=scheduler_config.realtime_interval
            )

        # Schedule regular stock updates (medium frequency)
        for symbol in scheduler_config.regular_stocks:
            await self.task_manager.schedule_interval_task(
                name=f"Regular-{symbol}",
                task_template=TaskTemplates.create_stock_price_task(symbol, Priority.MEDIUM),
                interval_seconds=scheduler_config.regular_interval
            )

        # Schedule index updates
        await self.task_manager.schedule_interval_task(
            name="VN-Index-Update",
            task_template=TaskTemplates.create_index_data_task(Priority.HIGH),
            interval_seconds=scheduler_config.index_update_interval
        )

        # Schedule news crawling
        await self.task_manager.schedule_interval_task(
            name="News-Crawling",
            task_template=TaskTemplates.create_news_crawl_task(Priority.LOW),
            interval_seconds=scheduler_config.news_crawl_interval
        )

        # Market hours aware scheduling (Vietnam time)
        await self._schedule_market_hours_tasks()

        logger.info(f"Scheduled tasks for {len(scheduler_config.realtime_stocks)} realtime stocks")
        logger.info(f"Scheduled tasks for {len(scheduler_config.regular_stocks)} regular stocks")

    async def _schedule_market_hours_tasks(self):
        """Schedule tasks that should only run during market hours"""
        # Market opening tasks - more frequent during trading hours
        await self.task_manager.schedule_recurring_task(
            name="Market-Open-Boost",
            task_template=TaskTemplates.create_index_data_task(Priority.HIGH),
            # Weekdays only
            cron_expression=(f"{self.config.scheduler.market_open_hour}-"
                             f"{self.config.scheduler.market_close_hour} * * 1-5 *")
        )

        # Pre-market preparation
        await self.task_manager.schedule_recurring_task(
            name="Pre-Market-Setup",
            task_template=CrawlTask(
                id="",
                source=DataSourceType.VNSTOCK,
                data_type=DataType.INDEX_DATA,
                priority=Priority.MEDIUM
            ),
            cron_expression=f"{self.config.scheduler.market_open_hour-1} * * 1-5 *"  # 1 hour before market
        )

        # Post-market summary
        await self.task_manager.schedule_recurring_task(
            name="Post-Market-Summary",
            task_template=TaskTemplates.create_index_data_task(Priority.MEDIUM),
            cron_expression=f"{self.config.scheduler.market_close_hour+1} * * 1-5 *"  # 1 hour after market
        )

    async def start_workers_only(self):
        """Start only the pipeline workers without scheduled tasks (for adhoc task execution)"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return

        logger.info("Starting pipeline workers only (no scheduled tasks)...")

        try:
            # Start main pipeline workers ONLY
            await self.pipeline.start(num_workers=self.config.pipeline.num_workers)

            self.is_running = True
            logger.info("Pipeline workers started successfully")

        except Exception as e:
            logger.error(f"Failed to start pipeline workers: {e}")
            await self.stop()
            raise

    async def start(self):
        """Start the pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return

        logger.info("Starting Vietnamese Stock Market Data Pipeline...")

        try:
            # Start main pipeline workers
            await self.pipeline.start(num_workers=self.config.pipeline.num_workers)

            # Start task scheduler
            scheduler_task = asyncio.create_task(self.task_manager.run_scheduler())
            self.background_tasks.append(scheduler_task)

            # Start monitoring tasks
            if self.config.monitoring.enabled:
                monitoring_task = asyncio.create_task(self._run_monitoring())
                self.background_tasks.append(monitoring_task)

            # Start health check server
            health_task = asyncio.create_task(self._run_health_check_server())
            self.background_tasks.append(health_task)

            # Start metrics collection
            if self.config.monitoring.prometheus_enabled:
                metrics_task = asyncio.create_task(self._collect_metrics())
                self.background_tasks.append(metrics_task)

            self.is_running = True
            logger.info("Pipeline started successfully")

            # Submit initial high-priority tasks for immediate data
            await self._submit_initial_tasks()

        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            await self.stop()
            raise

    async def _submit_initial_tasks(self):
        """Submit initial high-priority tasks for immediate data availability"""
        logger.info("Submitting initial high-priority tasks...")

        # Get VN-Index immediately
        index_task = TaskTemplates.create_index_data_task(Priority.HIGH)
        await self.task_manager.submit_immediate_task(index_task)

        # Get top 10 most important stocks immediately
        priority_stocks = self.config.scheduler.realtime_stocks[:10]
        for symbol in priority_stocks:
            stock_task = TaskTemplates.create_stock_price_task(symbol, Priority.HIGH)
            await self.task_manager.submit_immediate_task(stock_task)

        logger.info(f"Submitted {1 + len(priority_stocks)} initial tasks")

    async def stop(self):
        """Stop the pipeline gracefully"""
        if not self.is_running:
            return

        logger.info("Stopping pipeline...")
        self.is_running = False

        try:
            # Stop main pipeline
            if self.pipeline:
                await self.pipeline.stop()

            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()

            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)

            # Close Redis connection
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)

            logger.info("Pipeline stopped successfully")

        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}")

    async def _run_monitoring(self):
        """Background monitoring task"""
        logger.info("Starting monitoring task...")

        while self.is_running:
            try:
                # Get pipeline metrics
                if self.task_manager:
                    metrics = await self.task_manager.get_task_metrics()

                    # Log key metrics
                    logger.info(f"Pipeline Status - Running: {metrics['currently_running']}, "
                                f"Success Rate: {metrics['success_rate']:.2%}, "
                                f"Hourly Tasks: {metrics['hourly_tasks']}")

                    # Check for alerts
                    await self._check_alerts(metrics)

                await asyncio.sleep(self.config.monitoring.metrics_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _check_alerts(self, metrics: Dict):
        """Check metrics against alert thresholds"""
        alerts = []

        # Error rate alert
        if metrics['success_rate'] < (1 - self.config.monitoring.error_rate_threshold):
            alerts.append(f"High error rate: {(1-metrics['success_rate']):.2%}")

        # Queue size alert (placeholder - would need to implement queue size tracking)
        # if queue_size > self.config.monitoring.queue_size_threshold:
        #     alerts.append(f"High queue size: {queue_size}")

        # No recent tasks alert
        if metrics['hourly_tasks'] == 0:
            alerts.append("No tasks completed in the last hour")

        # Send alerts if any
        if alerts:
            await self._send_alerts(alerts)

    async def _send_alerts(self, alerts: List[str]):
        """Send alerts via configured channels"""
        alert_message = "Pipeline Alerts:\n" + "\n".join(f"- {alert}" for alert in alerts)

        logger.warning(alert_message)

        # Send to Slack if configured
        if self.config.monitoring.slack_webhook_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.config.monitoring.slack_webhook_url,
                        json={"text": alert_message}
                    )
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

    async def _run_health_check_server(self):
        """Simple HTTP health check server"""
        try:
            from aiohttp import web

            async def health_check(request):
                """Health check endpoint"""
                status = {
                    "status": "healthy" if self.is_running else "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    "pipeline_running": self.is_running,
                    "redis_connected": bool(self.redis_client),
                    "workers_active": len(self.pipeline.worker_tasks) if self.pipeline else 0
                }

                if self.task_manager:
                    metrics = await self.task_manager.get_task_metrics()
                    status.update({
                        "tasks_running": metrics["currently_running"],
                        "success_rate": metrics["success_rate"],
                        "hourly_tasks": metrics["hourly_tasks"]
                    })

                return web.json_response(status)

            app = web.Application()
            app.router.add_get(self.config.monitoring.health_check_path, health_check)

            runner = web.AppRunner(app)
            await runner.setup()

            site = web.TCPSite(
                runner,
                "0.0.0.0",
                self.config.monitoring.health_check_port
            )
            await site.start()

            logger.info(f"Health check server started on port {self.config.monitoring.health_check_port}")

            # Keep server running
            while self.is_running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Health check server error: {e}")

    async def _collect_metrics(self):
        """Collect and export Prometheus metrics"""
        try:
            # This would integrate with prometheus_client
            # For now, just log metrics periodically
            while self.is_running:
                if self.task_manager:
                    metrics = await self.task_manager.get_task_metrics()

                    # Log metrics in Prometheus format (simplified)
                    logger.debug(f"pipeline_tasks_total {metrics['total_tasks']}")
                    logger.debug(f"pipeline_tasks_success_rate {metrics['success_rate']}")
                    logger.debug(f"pipeline_tasks_running {metrics['currently_running']}")

                await asyncio.sleep(self.config.pipeline.metrics_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")

    async def get_status(self) -> Dict:
        """Get current pipeline status"""
        status = {
            "running": self.is_running,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "environment": self.config.environment,
                "workers": self.config.pipeline.num_workers,
                "sources_enabled": sum(1 for source in [
                    self.config.crawler.vnstock.enabled,
                    self.config.crawler.cafef.enabled,
                    self.config.crawler.vietfin.enabled,
                    self.config.crawler.hose.enabled
                ] if source)
            }
        }

        if self.task_manager:
            task_metrics = await self.task_manager.get_task_metrics()
            status["metrics"] = task_metrics

        return status

    async def submit_adhoc_task(self, task: CrawlTask) -> str:
        """Submit an ad-hoc task for immediate execution"""
        if not self.task_manager:
            raise RuntimeError("Task manager not initialized")

        return await self.task_manager.submit_immediate_task(task)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


class PipelineManager:
    """High-level pipeline management interface"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config = ConfigManager.load_from_file(config_path)
        else:
            self.config = ConfigManager.load_from_env()

        self.orchestrator = PipelineOrchestrator(self.config)

    async def run(self):
        """Run the pipeline with proper initialization and cleanup"""
        try:
            # Setup signal handlers
            self.orchestrator._setup_signal_handlers()

            # Initialize components
            await self.orchestrator.initialize()

            # Start pipeline
            await self.orchestrator.start()

            logger.info("Pipeline is running. Press Ctrl+C to stop.")

            # Keep running until signal
            while self.orchestrator.is_running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            await self.orchestrator.stop()

    async def start_development_mode(self):
        """Start pipeline in development mode with reduced load"""
        # Use development preset
        self.config = ConfigPresets.development()
        self.orchestrator = PipelineOrchestrator(self.config)
        await self.run()

    async def run_single_crawl(self, symbol: str) -> Dict:
        """Run a single crawl operation for testing"""
        await self.orchestrator.initialize()

        task = TaskTemplates.create_stock_price_task(symbol, Priority.HIGH)
        task_id = await self.orchestrator.submit_adhoc_task(task)

        # Wait for task completion (simplified)
        await asyncio.sleep(5)

        status = await self.orchestrator.get_status()
        await self.orchestrator.stop()

        return {"task_id": task_id, "status": status}


# CLI Interface
async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Vietnamese Stock Market Data Pipeline")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    parser.add_argument("--test-crawl", help="Test crawl for specific symbol")
    parser.add_argument("--environment", choices=["development", "testing", "production"],
                        help="Environment preset")

    args = parser.parse_args()

    # Create pipeline manager
    if args.environment:
        if args.environment == "development":
            config = ConfigPresets.development()
        elif args.environment == "testing":
            config = ConfigPresets.testing()
        else:
            config = ConfigPresets.production()
        manager = PipelineManager()
        manager.config = config
        manager.orchestrator = PipelineOrchestrator(config)
    else:
        manager = PipelineManager(args.config)

    # Run based on arguments
    if args.test_crawl:
        result = await manager.run_single_crawl(args.test_crawl)
        print(f"Test crawl result: {result}")
    elif args.dev:
        await manager.start_development_mode()
    else:
        await manager.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
