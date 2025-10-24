"""
Comprehensive Monitoring and Alerting System
===========================================

Advanced monitoring system for Vietnamese stock market data pipeline
with real-time metrics, health checks, alerts, and dashboards.

"""

import asyncio
import json
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import psutil
import redis.asyncio as redis

from core_services.utils.logger_utils import logger


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    help_text: Optional[str] = None

    def to_prometheus(self) -> str:
        """Convert to Prometheus format"""
        labels_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(label_pairs) + "}"

        return f"{self.name}{labels_str} {self.value} {int(self.timestamp.timestamp() * 1000)}"


@dataclass
class Alert:
    """Alert definition and state"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    metric_name: str
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    fired_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    fire_count: int = 0

    @property
    def is_firing(self) -> bool:
        """Check if alert is currently firing"""
        return self.fired_at is not None and self.resolved_at is None

    @property
    def duration(self) -> Optional[timedelta]:
        """Get alert duration if firing"""
        if self.fired_at:
            end_time = self.resolved_at or datetime.now()
            return end_time - self.fired_at
        return None


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable[[], bool]
    description: str
    timeout_seconds: int = 10
    critical: bool = False
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None
    check_count: int = 0
    failure_count: int = 0


class MetricsCollector:
    """Collects and stores metrics"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_flush = datetime.now()

    async def record_metric(self, metric: Metric):
        """Record a single metric"""
        # Store in buffer for fast access
        key = f"{metric.name}:{json.dumps(metric.labels, sort_keys=True)}"
        self.metrics_buffer[key].append(metric)

        # Store in Redis for persistence
        await self._store_metric_redis(metric)

    async def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record counter metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels or {}
        )
        await self.record_metric(metric)

    async def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record gauge metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        )
        await self.record_metric(metric)

    async def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {}
        )
        await self.record_metric(metric)

    async def get_metric_values(self, name: str,
                                labels: Dict[str, str] = None,
                                duration: timedelta = timedelta(hours=1)) -> List[Metric]:
        """Get metric values for specified duration"""
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"

        if key in self.metrics_buffer:
            # Get from buffer if available
            cutoff_time = datetime.now() - duration
            return [m for m in self.metrics_buffer[key] if m.timestamp >= cutoff_time]

        # Fallback to Redis
        return await self._get_metrics_from_redis(name, labels, duration)

    async def get_metric_summary(self, name: str,
                                 labels: Dict[str, str] = None,
                                 duration: timedelta = timedelta(hours=1)) -> Dict[str, float]:
        """Get metric summary statistics"""
        values = await self.get_metric_values(name, labels, duration)

        if not values:
            return {}

        metric_values = [m.value for m in values]

        return {
            "count": len(metric_values),
            "sum": sum(metric_values),
            "min": min(metric_values),
            "max": max(metric_values),
            "mean": statistics.mean(metric_values),
            "median": statistics.median(metric_values),
            "latest": metric_values[-1] if metric_values else 0,
            "rate_per_second": len(metric_values) / duration.total_seconds()
        }

    async def _store_metric_redis(self, metric: Metric):
        """Store metric in Redis with TTL"""
        key = f"metrics:{metric.name}"

        metric_data = {
            "value": metric.value,
            "type": metric.metric_type.value,
            "labels": json.dumps(metric.labels),
            "timestamp": metric.timestamp.isoformat()
        }

        # Store as hash with timestamp as field
        field = str(int(metric.timestamp.timestamp() * 1000))
        await self.redis.hset(key, field, json.dumps(metric_data))

        # Set TTL (keep metrics for 7 days)
        await self.redis.expire(key, 604800)

    async def _get_metrics_from_redis(self, name: str,
                                      labels: Dict[str, str] = None,
                                      duration: timedelta = timedelta(hours=1)) -> List[Metric]:
        """Retrieve metrics from Redis"""
        key = f"metrics:{name}"
        cutoff_timestamp = int((datetime.now() - duration).timestamp() * 1000)

        # Get all fields and filter by timestamp
        all_data = await self.redis.hgetall(key)
        metrics = []

        for field_name, data in all_data.items():
            timestamp_ms = int(field_name.decode())
            if timestamp_ms >= cutoff_timestamp:
                try:
                    metric_data = json.loads(data.decode())
                    metric_labels = json.loads(metric_data["labels"])

                    # Filter by labels if specified
                    if labels and not all(metric_labels.get(k) == v for k, v in labels.items()):
                        continue

                    metric = Metric(
                        name=name,
                        value=metric_data["value"],
                        metric_type=MetricType(metric_data["type"]),
                        labels=metric_labels,
                        timestamp=datetime.fromisoformat(metric_data["timestamp"])
                    )
                    metrics.append(metric)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return sorted(metrics, key=lambda m: m.timestamp)


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.notification_channels: List["NotificationChannel"] = []
        self.alert_history: deque = deque(maxlen=1000)

    def add_alert(self, alert: Alert):
        """Add alert rule"""
        self.alerts[alert.id] = alert
        logger.info(f"Added alert: {alert.name}")

    def remove_alert(self, alert_id: str):
        """Remove alert rule"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Removed alert: {alert_id}")

    def add_notification_channel(self, channel: "NotificationChannel"):
        """Add notification channel"""
        self.notification_channels.append(channel)
        logger.info(f"Added notification channel: {channel.name}")

    async def check_alerts(self):
        """Check all alert conditions"""
        for alert in self.alerts.values():
            if not alert.enabled:
                continue

            try:
                should_fire = await self._evaluate_alert_condition(alert)

                if should_fire and not alert.is_firing:
                    await self._fire_alert(alert)
                elif not should_fire and alert.is_firing:
                    await self._resolve_alert(alert)

            except Exception as e:
                logger.error(f"Error checking alert {alert.id}: {e}")

    async def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate if alert condition is met"""
        # Get recent metric values
        recent_values = await self.metrics.get_metric_values(
            alert.metric_name,
            alert.labels,
            duration=timedelta(minutes=5)
        )

        if not recent_values:
            return False

        # Use latest value for evaluation
        latest_value = recent_values[-1].value

        # Simple threshold-based evaluation (can be extended for complex conditions)
        if ">" in alert.condition:
            return latest_value > alert.threshold
        elif "<" in alert.condition:
            return latest_value < alert.threshold
        elif "==" in alert.condition:
            return abs(latest_value - alert.threshold) < 0.001

        return False

    async def _fire_alert(self, alert: Alert):
        """Fire an alert"""
        alert.fired_at = datetime.now()
        alert.resolved_at = None
        alert.fire_count += 1

        # Store in history
        self.alert_history.append({
            "alert_id": alert.id,
            "action": "fired",
            "timestamp": alert.fired_at.isoformat(),
            "severity": alert.severity.value
        })

        # Send notifications
        await self._send_notifications(alert, "fired")

        logger.warning(f"Alert FIRED: {alert.name} - {alert.description}")

    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert"""
        alert.resolved_at = datetime.now()

        # Store in history
        self.alert_history.append({
            "alert_id": alert.id,
            "action": "resolved",
            "timestamp": alert.resolved_at.isoformat(),
            "duration_seconds": alert.duration.total_seconds() if alert.duration else 0
        })

        # Send notifications
        await self._send_notifications(alert, "resolved")

        logger.info(f"Alert RESOLVED: {alert.name} - Duration: {alert.duration}")

    async def _send_notifications(self, alert: Alert, action: str):
        """Send notifications to all channels"""
        for channel in self.notification_channels:
            try:
                await channel.send_alert_notification(alert, action)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.name}: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get currently firing alerts"""
        return [alert for alert in self.alerts.values() if alert.is_firing]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary"""
        active_alerts = self.get_active_alerts()

        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1

        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "severity_breakdown": dict(severity_counts),
            "notification_channels": len(self.notification_channels),
            "alert_history_size": len(self.alert_history)
        }


class NotificationChannel(ABC):
    """Abstract base for notification channels"""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    async def send_alert_notification(self, alert: Alert, action: str):
        """Send alert notification"""


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""

    def __init__(self, webhook_url: str, channel: str = None):
        super().__init__("Slack")
        self.webhook_url = webhook_url
        self.channel = channel

    async def send_alert_notification(self, alert: Alert, action: str):
        """Send alert to Slack"""
        if not self.enabled:
            return

        color = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }.get(alert.severity, "warning")

        title = f"Alert {action.upper()}: {alert.name}"

        if action == "fired":
            text = f"🚨 {alert.description}\nThreshold: {alert.threshold}\nSeverity: {alert.severity.value}"
        else:
            duration = alert.duration.total_seconds() if alert.duration else 0
            text = f"✅ Alert resolved after {duration:.0f} seconds"

        payload = {
            "attachments": [{
                "color": color,
                "title": title,
                "text": text,
                "timestamp": int(datetime.now().timestamp())
            }]
        }

        if self.channel:
            payload["channel"] = self.channel

        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to send Slack notification: {response.status}")


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, recipients: List[str]):
        super().__init__("Email")
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients

    async def send_alert_notification(self, alert: Alert, action: str):
        """Send alert via email"""
        if not self.enabled:
            return

        # Implementation would use aiosmtplib or similar
        # Simplified for demo
        logger.info(f"Would send email notification for alert {alert.name} to {self.recipients}")


class HealthChecker:
    """Health check manager"""

    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_status = HealthStatus.UNKNOWN

    def add_health_check(self, health_check: HealthCheck):
        """Add health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        critical_failed = False
        any_failed = False

        for name, check in self.health_checks.items():
            try:
                start_time = time.time()

                # Run check with timeout
                is_healthy = await asyncio.wait_for(
                    asyncio.to_thread(check.check_function),
                    timeout=check.timeout_seconds
                )

                check.last_check = datetime.now()
                check.check_count += 1
                check.last_error = None

                if is_healthy:
                    check.last_status = HealthStatus.HEALTHY
                else:
                    check.last_status = HealthStatus.UNHEALTHY
                    check.failure_count += 1
                    any_failed = True

                    if check.critical:
                        critical_failed = True

                results[name] = {
                    "status": check.last_status.value,
                    "duration_ms": (time.time() - start_time) * 1000,
                    "description": check.description,
                    "critical": check.critical,
                    "check_count": check.check_count,
                    "failure_count": check.failure_count
                }

            except asyncio.TimeoutError:
                check.last_status = HealthStatus.UNHEALTHY
                check.last_error = "Timeout"
                check.failure_count += 1
                any_failed = True

                if check.critical:
                    critical_failed = True

                results[name] = {
                    "status": "unhealthy",
                    "error": "Timeout",
                    "critical": check.critical
                }

            except Exception as e:
                check.last_status = HealthStatus.UNHEALTHY
                check.last_error = str(e)
                check.failure_count += 1
                any_failed = True

                if check.critical:
                    critical_failed = True

                results[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "critical": check.critical
                }

        # Determine overall status
        if critical_failed:
            self.overall_status = HealthStatus.UNHEALTHY
        elif any_failed:
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY

        return {
            "overall_status": self.overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": results
        }


class PipelineMonitor:
    """Main monitoring system for the pipeline"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.metrics = MetricsCollector(redis_client)
        self.alerts = AlertManager(self.metrics)
        self.health = HealthChecker()
        self.is_running = False

        # System metrics tracking
        self.start_time = datetime.now()
        self.last_system_check = datetime.now()

        # Setup default health checks and alerts
        self._setup_default_monitoring()

    def _setup_default_monitoring(self):
        """Setup default monitoring configuration"""

        # Redis connectivity health check
        def check_redis():
            try:
                asyncio.create_task(self.redis.ping())
                return True
            except BaseException:
                return False

        self.health.add_health_check(HealthCheck(
            name="redis_connectivity",
            check_function=check_redis,
            description="Redis connection health",
            critical=True
        ))

        # System resource health checks
        def check_memory():
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if > 90% memory usage

        def check_cpu():
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 80  # Alert if > 80% CPU usage

        def check_disk():
            disk = psutil.disk_usage("/")
            return disk.percent < 90  # Alert if > 90% disk usage

        self.health.add_health_check(HealthCheck(
            name="memory_usage",
            check_function=check_memory,
            description="System memory usage",
            critical=False
        ))

        self.health.add_health_check(HealthCheck(
            name="cpu_usage",
            check_function=check_cpu,
            description="System CPU usage",
            critical=False
        ))

        self.health.add_health_check(HealthCheck(
            name="disk_usage",
            check_function=check_disk,
            description="System disk usage",
            critical=False
        ))

        # Default alerts
        self.alerts.add_alert(Alert(
            id="high_error_rate",
            name="High Error Rate",
            description="Pipeline error rate is above threshold",
            severity=AlertSeverity.WARNING,
            condition="error_rate > threshold",
            threshold=0.1,  # 10% error rate
            metric_name="pipeline_error_rate"
        ))

        self.alerts.add_alert(Alert(
            id="queue_backlog",
            name="Task Queue Backlog",
            description="Task queue size is growing",
            severity=AlertSeverity.WARNING,
            condition="queue_size > threshold",
            threshold=1000,
            metric_name="task_queue_size"
        ))

        self.alerts.add_alert(Alert(
            id="source_failure",
            name="Data Source Failure",
            description="Data source success rate below threshold",
            severity=AlertSeverity.ERROR,
            condition="success_rate < threshold",
            threshold=0.8,  # 80% success rate
            metric_name="source_success_rate"
        ))

    async def start_monitoring(self):
        """Start monitoring background tasks"""
        self.is_running = True

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._run_health_checks()),
            asyncio.create_task(self._check_alerts()),
            asyncio.create_task(self._cleanup_old_metrics())
        ]

        logger.info("Pipeline monitoring started")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Monitoring stopped")

    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False

    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self.is_running:
            try:
                # System metrics
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                disk = psutil.disk_usage("/")

                await self.metrics.record_gauge("system_memory_percent", memory.percent)
                await self.metrics.record_gauge("system_cpu_percent", cpu_percent)
                await self.metrics.record_gauge("system_disk_percent", disk.percent)

                # Pipeline uptime
                uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                await self.metrics.record_gauge("pipeline_uptime_seconds", uptime_seconds)

                # Task metrics (would be updated by pipeline components)
                await self.metrics.record_gauge("active_tasks", 0)  # Placeholder
                await self.metrics.record_gauge("completed_tasks_total", 0)  # Placeholder

                await asyncio.sleep(30)  # Collect every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(30)

    async def _run_health_checks(self):
        """Run health checks periodically"""
        while self.is_running:
            try:
                health_results = await self.health.run_health_checks()

                # Record health check metrics
                for check_name, result in health_results["checks"].items():
                    status_value = 1 if result["status"] == "healthy" else 0
                    await self.metrics.record_gauge(
                        "health_check_status",
                        status_value,
                        {"check": check_name}
                    )

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error running health checks: {e}")
                await asyncio.sleep(60)

    async def _check_alerts(self):
        """Check alerts periodically"""
        while self.is_running:
            try:
                await self.alerts.check_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(30)

    async def _cleanup_old_metrics(self):
        """Cleanup old metrics from memory"""
        while self.is_running:
            try:
                # Clean up metrics buffer periodically
                cutoff_time = datetime.now() - timedelta(hours=2)

                for key, metrics_deque in self.metrics.metrics_buffer.items():
                    # Remove old metrics from deque
                    while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
                        metrics_deque.popleft()

                await asyncio.sleep(3600)  # Cleanup every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error cleaning up metrics: {e}")
                await asyncio.sleep(3600)

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""

        # Get health status
        health_data = await self.health.run_health_checks()

        # Get alert summary
        alert_summary = self.alerts.get_alert_summary()

        # Get key metrics
        metrics_summary = {}
        key_metrics = [
            "system_memory_percent",
            "system_cpu_percent",
            "pipeline_uptime_seconds",
            "completed_tasks_total"
        ]

        for metric_name in key_metrics:
            summary = await self.metrics.get_metric_summary(metric_name)
            if summary:
                metrics_summary[metric_name] = summary["latest"]

        # Get active alerts
        active_alerts = [
            {
                "id": alert.id,
                "name": alert.name,
                "severity": alert.severity.value,
                "duration_seconds": alert.duration.total_seconds() if alert.duration else 0
            }
            for alert in self.alerts.get_active_alerts()
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "health": health_data,
            "alerts": {
                "summary": alert_summary,
                "active": active_alerts
            },
            "metrics": metrics_summary,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }

    # Methods for pipeline components to record metrics
    async def record_task_started(self, task_type: str, source: str):
        """Record task start"""
        await self.metrics.record_counter(
            "tasks_started_total",
            labels={"type": task_type, "source": source}
        )

    async def record_task_completed(self, task_type: str, source: str, duration_seconds: float):
        """Record task completion"""
        await self.metrics.record_counter(
            "tasks_completed_total",
            labels={"type": task_type, "source": source}
        )
        await self.metrics.record_histogram(
            "task_duration_seconds",
            duration_seconds,
            labels={"type": task_type, "source": source}
        )

    async def record_task_failed(self, task_type: str, source: str, error_type: str):
        """Record task failure"""
        await self.metrics.record_counter(
            "tasks_failed_total",
            labels={"type": task_type, "source": source, "error": error_type}
        )

    async def record_data_quality_score(self, source: str, symbol: str, score: float):
        """Record data quality score"""
        await self.metrics.record_gauge(
            "data_quality_score",
            score,
            labels={"source": source, "symbol": symbol}
        )

    async def record_source_response_time(self, source: str, response_time_ms: float):
        """Record source response time"""
        await self.metrics.record_histogram(
            "source_response_time_ms",
            response_time_ms,
            labels={"source": source}
        )


# Factory function
def create_pipeline_monitor(redis_url: str = "redis://localhost:6379") -> PipelineMonitor:
    """Create pipeline monitor with Redis connection"""
    redis_client = redis.from_url(redis_url)
    return PipelineMonitor(redis_client)


# Usage example
async def demo_monitoring():
    """Demonstrate monitoring system"""
    print("🔍 Vietnamese Stock Pipeline Monitoring Demo")
    print("=" * 50)

    monitor = create_pipeline_monitor()

    # Add Slack notifications (example)
    # slack_channel = SlackNotificationChannel("https://hooks.slack.com/services/...")
    # monitor.alerts.add_notification_channel(slack_channel)

    # Start monitoring for demo
    monitoring_task = asyncio.create_task(monitor.start_monitoring())

    # Simulate some metrics
    await monitor.record_task_started("stock_price", "vnstock")
    await asyncio.sleep(1)
    await monitor.record_task_completed("stock_price", "vnstock", 0.5)
    await monitor.record_data_quality_score("vnstock", "VNM", 0.95)

    # Get dashboard data
    dashboard = await monitor.get_dashboard_data()
    print(f"Overall Health: {dashboard['health']['overall_status']}")
    print(f"Active Alerts: {len(dashboard['alerts']['active'])}")
    print(f"Uptime: {dashboard['uptime_seconds']:.0f} seconds")

    # Stop monitoring
    await monitor.stop_monitoring()
    monitoring_task.cancel()

    print("✅ Monitoring demo completed")


if __name__ == "__main__":
    asyncio.run(demo_monitoring())
