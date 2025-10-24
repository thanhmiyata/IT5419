"""
Task Management System for Vietnamese Stock Market Data Pipeline
=============================================================

Handles task scheduling, execution monitoring, and resource management
for the data crawling pipeline.

"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import redis
from croniter import croniter

from core_services.data_ingestion.pipeline_architecture import CrawlTask
from core_services.utils.common import (REDIS_KEY_PREFIX_TASK, DataSourceType, DataType, Priority, ScheduleType,
                                        TaskStatus)
from core_services.utils.logger_utils import logger


@dataclass
class TaskExecutionContext:
    """Context information for task execution"""
    task_id: str
    worker_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    execution_duration: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    id: str
    name: str
    task_template: CrawlTask
    schedule_type: ScheduleType
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskManager:
    """Centralized task management system"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.running_tasks: Dict[str, TaskExecutionContext] = {}
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.task_history_key = "task_history"
        self.scheduled_tasks_key = "scheduled_tasks"
        self._load_scheduled_tasks()

    async def submit_immediate_task(self, task: CrawlTask) -> str:
        """Submit task for immediate execution"""
        task_id = task.id or str(uuid.uuid4())
        task.id = task_id

        # Add to high priority queue
        task.priority = Priority.HIGH

        # Store task details
        task_data = self._serialize_task(task)
        self.redis.hset(f"{REDIS_KEY_PREFIX_TASK}{task_id}", mapping=task_data)
        logger.info(f"Stored task {task_id} with data_type={task.data_type.value}, params={task.params}")

        # Add to execution queue
        queue_key = f"crawl_queue:{task.priority.name.lower()}"
        self.redis.lpush(queue_key, task_id)

        queue_length = self.redis.llen(queue_key)
        logger.info(f"Immediate task {task_id} submitted to {queue_key} (queue length: {queue_length})")
        return task_id

    async def get_next_task(self) -> Optional[CrawlTask]:
        """Get next task from highest priority queue"""
        for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            queue_key = f"crawl_queue:{priority.name.lower()}"
            task_id = self.redis.rpop(queue_key)

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

    async def schedule_recurring_task(self,
                                      name: str,
                                      task_template: CrawlTask,
                                      cron_expression: str,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      max_runs: Optional[int] = None) -> str:
        """Schedule recurring task with cron expression"""

        scheduled_task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            task_template=task_template,
            schedule_type=ScheduleType.CRON,
            cron_expression=cron_expression,
            start_time=start_time or datetime.now(),
            end_time=end_time,
            max_runs=max_runs
        )

        # Calculate next run time
        scheduled_task.next_run = self._calculate_next_run(scheduled_task)

        # Store scheduled task
        self.scheduled_tasks[scheduled_task.id] = scheduled_task
        await self._save_scheduled_task(scheduled_task)

        logger.info(f"Recurring task {name} scheduled with cron: {cron_expression}")
        return scheduled_task.id

    async def schedule_interval_task(self,
                                     name: str,
                                     task_template: CrawlTask,
                                     interval_seconds: int,
                                     start_time: Optional[datetime] = None,
                                     max_runs: Optional[int] = None) -> str:
        """Schedule task to run at fixed intervals"""

        scheduled_task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            task_template=task_template,
            schedule_type=ScheduleType.RECURRING,
            interval_seconds=interval_seconds,
            start_time=start_time or datetime.now(),
            max_runs=max_runs
        )

        scheduled_task.next_run = self._calculate_next_run(scheduled_task)

        self.scheduled_tasks[scheduled_task.id] = scheduled_task
        await self._save_scheduled_task(scheduled_task)

        logger.info(f"Interval task {name} scheduled every {interval_seconds} seconds")
        return scheduled_task.id

    async def cancel_scheduled_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].enabled = False
            await self._save_scheduled_task(self.scheduled_tasks[task_id])
            logger.info(f"Scheduled task {task_id} cancelled")
            return True
        return False

    async def start_task_execution(self, task_id: str, worker_id: str) -> TaskExecutionContext:
        """Start tracking task execution"""
        context = TaskExecutionContext(
            task_id=task_id,
            worker_id=worker_id,
            start_time=datetime.now(),
            status=TaskStatus.RUNNING
        )

        self.running_tasks[task_id] = context

        # Update task status in Redis
        self.redis.hset(f"task:{task_id}", "status", TaskStatus.RUNNING.value)
        self.redis.hset(f"task:{task_id}", "worker_id", worker_id)
        self.redis.hset(f"task:{task_id}", "start_time", context.start_time.isoformat())

        return context

    async def complete_task_execution(self,
                                      task_id: str,
                                      success: bool,
                                      error_message: Optional[str] = None) -> bool:
        """Complete task execution and update status"""

        if task_id not in self.running_tasks:
            logger.warning(f"Task {task_id} not found in running tasks")
            return False

        context = self.running_tasks[task_id]
        context.end_time = datetime.now()
        context.execution_duration = (context.end_time - context.start_time).total_seconds()
        context.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        context.error_message = error_message

        # Update in Redis
        self.redis.hset(f"task:{task_id}", "status", context.status.value)
        self.redis.hset(f"task:{task_id}", "end_time", context.end_time.isoformat())
        self.redis.hset(f"task:{task_id}", "execution_duration", context.execution_duration)

        if error_message:
            self.redis.hset(f"task:{task_id}", "error_message", error_message)

        # Store in history
        await self._store_task_history(context)

        # Remove from running tasks
        del self.running_tasks[task_id]

        logger.info(f"Task {task_id} completed with status: {context.status.value}")
        return True

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current task status"""
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].status

        # Check Redis
        status_str = self.redis.hget(f"task:{task_id}", "status")
        if status_str:
            return TaskStatus(status_str.decode())

        return None

    async def get_running_tasks(self) -> List[TaskExecutionContext]:
        """Get list of currently running tasks"""
        return list(self.running_tasks.values())

    async def get_task_history(self, limit: int = 100) -> List[Dict]:
        """Get task execution history"""
        history_data = self.redis.lrange(self.task_history_key, 0, limit - 1)
        return [json.loads(data.decode()) for data in history_data]

    async def get_task_metrics(self) -> Dict[str, any]:
        """Get task execution metrics"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)

        # Get recent history
        recent_history = await self.get_task_history(1000)

        # Calculate metrics
        total_tasks = len(recent_history)
        successful_tasks = sum(1 for h in recent_history if h["status"] == TaskStatus.COMPLETED.value)
        failed_tasks = sum(1 for h in recent_history if h["status"] == TaskStatus.FAILED.value)

        hourly_tasks = sum(1 for h in recent_history
                           if datetime.fromisoformat(h["end_time"]) > last_hour)
        daily_tasks = sum(1 for h in recent_history
                          if datetime.fromisoformat(h["end_time"]) > last_day)

        avg_duration = sum(h["execution_duration"] for h in recent_history) / max(total_tasks, 1)

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / max(total_tasks, 1),
            "hourly_tasks": hourly_tasks,
            "daily_tasks": daily_tasks,
            "average_duration": avg_duration,
            "currently_running": len(self.running_tasks),
            "scheduled_tasks": len([t for t in self.scheduled_tasks.values() if t.enabled])
        }

    async def run_scheduler(self):
        """Main scheduler loop - run this as background task"""
        logger.info("Task scheduler started")

        while True:
            try:
                await self._process_scheduled_tasks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _process_scheduled_tasks(self):
        """Process scheduled tasks that are due"""
        now = datetime.now()

        for scheduled_task in list(self.scheduled_tasks.values()):
            if not scheduled_task.enabled:
                continue

            if not scheduled_task.next_run or scheduled_task.next_run > now:
                continue

            # Check if task should still run
            if scheduled_task.end_time and now > scheduled_task.end_time:
                scheduled_task.enabled = False
                await self._save_scheduled_task(scheduled_task)
                continue

            if scheduled_task.max_runs and scheduled_task.run_count >= scheduled_task.max_runs:
                scheduled_task.enabled = False
                await self._save_scheduled_task(scheduled_task)
                continue

            # Create new task instance
            task = CrawlTask(
                id=str(uuid.uuid4()),
                source=scheduled_task.task_template.source,
                data_type=scheduled_task.task_template.data_type,
                symbol=scheduled_task.task_template.symbol,
                params=scheduled_task.task_template.params.copy(),
                priority=scheduled_task.task_template.priority
            )

            # Submit task
            await self.submit_immediate_task(task)

            # Update scheduled task
            scheduled_task.last_run = now
            scheduled_task.run_count += 1
            scheduled_task.next_run = self._calculate_next_run(scheduled_task)

            await self._save_scheduled_task(scheduled_task)

            logger.info(f"Scheduled task {scheduled_task.name} executed")

    def _calculate_next_run(self, scheduled_task: ScheduledTask) -> datetime:
        """Calculate next run time for scheduled task"""
        if scheduled_task.schedule_type == ScheduleType.CRON:
            cron = croniter(scheduled_task.cron_expression, scheduled_task.last_run or datetime.now())
            return cron.get_next(datetime)

        elif scheduled_task.schedule_type == ScheduleType.RECURRING:
            base_time = scheduled_task.last_run or scheduled_task.start_time or datetime.now()
            return base_time + timedelta(seconds=scheduled_task.interval_seconds)

        return datetime.now()

    def _serialize_task(self, task: CrawlTask) -> Dict[str, str]:
        """Serialize task for Redis storage"""
        return {
            "id": task.id,
            "source": task.source.value,
            "data_type": task.data_type.value,
            "symbol": task.symbol or "",
            "params": json.dumps(task.params),
            "priority": task.priority.value,
            "retry_count": str(task.retry_count),
            "max_retries": str(task.max_retries),
            "created_at": task.created_at.isoformat(),
            "status": TaskStatus.PENDING.value
        }

    async def _save_scheduled_task(self, scheduled_task: ScheduledTask):
        """Save scheduled task to Redis"""
        task_data = {
            "id": scheduled_task.id,
            "name": scheduled_task.name,
            "schedule_type": scheduled_task.schedule_type.value,
            "enabled": str(scheduled_task.enabled),
            "run_count": str(scheduled_task.run_count),
            "created_at": scheduled_task.created_at.isoformat()
        }

        if scheduled_task.cron_expression:
            task_data["cron_expression"] = scheduled_task.cron_expression

        if scheduled_task.interval_seconds:
            task_data["interval_seconds"] = str(scheduled_task.interval_seconds)

        if scheduled_task.next_run:
            task_data["next_run"] = scheduled_task.next_run.isoformat()

        if scheduled_task.last_run:
            task_data["last_run"] = scheduled_task.last_run.isoformat()

        # Serialize task template
        task_data["task_template"] = json.dumps({
            "source": scheduled_task.task_template.source.value,
            "data_type": scheduled_task.task_template.data_type.value,
            "symbol": scheduled_task.task_template.symbol,
            "params": scheduled_task.task_template.params,
            "priority": scheduled_task.task_template.priority.value
        })

        self.redis.hset(f"scheduled_task:{scheduled_task.id}", mapping=task_data)
        self.redis.sadd(self.scheduled_tasks_key, scheduled_task.id)

    def _load_scheduled_tasks(self):
        """Load scheduled tasks from Redis"""
        try:
            task_ids = self.redis.smembers(self.scheduled_tasks_key)

            for task_id in task_ids:
                task_data_raw = self.redis.hgetall(f"scheduled_task:{task_id.decode()}")

                if task_data_raw:
                    # Convert bytes keys to strings
                    task_data = {}
                    for key, value in task_data_raw.items():
                        k = key.decode() if isinstance(key, bytes) else key
                        v = value.decode() if isinstance(value, bytes) else value
                        task_data[k] = v

                    # Deserialize task template
                    template_data = json.loads(task_data["task_template"])
                    task_template = CrawlTask(
                        id="",  # Will be generated for each execution
                        source=DataSourceType(template_data["source"]),
                        data_type=DataType(template_data["data_type"]),
                        symbol=template_data["symbol"],
                        params=template_data["params"],
                        priority=Priority(template_data["priority"])
                    )

                    # Create scheduled task
                    scheduled_task = ScheduledTask(
                        id=task_data["id"],
                        name=task_data["name"],
                        task_template=task_template,
                        schedule_type=ScheduleType(task_data["schedule_type"]),
                        enabled=task_data["enabled"] == "True",
                        run_count=int(task_data["run_count"]),
                        created_at=datetime.fromisoformat(task_data["created_at"])
                    )

                    # Load optional fields
                    if "cron_expression" in task_data:
                        scheduled_task.cron_expression = task_data["cron_expression"]

                    if "interval_seconds" in task_data:
                        scheduled_task.interval_seconds = int(task_data["interval_seconds"])

                    if "next_run" in task_data:
                        scheduled_task.next_run = datetime.fromisoformat(task_data["next_run"])

                    if "last_run" in task_data:
                        scheduled_task.last_run = datetime.fromisoformat(task_data["last_run"])

                    self.scheduled_tasks[scheduled_task.id] = scheduled_task

            logger.info(f"Loaded {len(self.scheduled_tasks)} scheduled tasks")

        except Exception as e:
            logger.error(f"Failed to load scheduled tasks: {e}")
            import traceback
            traceback.print_exc()

    async def _store_task_history(self, context: TaskExecutionContext):
        """Store task execution context in history"""
        history_entry = {
            "task_id": context.task_id,
            "worker_id": context.worker_id,
            "status": context.status.value,
            "start_time": context.start_time.isoformat(),
            "end_time": context.end_time.isoformat() if context.end_time else None,
            "execution_duration": context.execution_duration,
            "retry_count": context.retry_count,
            "error_message": context.error_message
        }

        # Add to history list (keep only recent entries)
        self.redis.lpush(self.task_history_key, json.dumps(history_entry))
        self.redis.ltrim(self.task_history_key, 0, 9999)  # Keep last 10k entries


# Predefined task templates for common operations
class TaskTemplates:
    """Common task templates for Vietnamese stock market"""

    @staticmethod
    def create_stock_price_task(symbol: str, priority: Priority = Priority.MEDIUM) -> CrawlTask:
        """Create task for fetching stock price"""
        return CrawlTask(
            id=str(uuid.uuid4()),
            source=DataSourceType.VNSTOCK,
            data_type=DataType.STOCK_PRICE,
            symbol=symbol,
            priority=priority
        )

    @staticmethod
    def create_historical_data_task(symbol: str, start_date: str, end_date: str,
                                    priority: Priority = Priority.MEDIUM) -> CrawlTask:
        """Create task for fetching historical data within a date range"""
        return CrawlTask(
            id=str(uuid.uuid4()),
            source=DataSourceType.VNSTOCK,
            data_type=DataType.HISTORICAL_DATA,
            symbol=symbol,
            params={"start_date": start_date, "end_date": end_date},
            priority=priority
        )

    @staticmethod
    def create_index_data_task(priority: Priority = Priority.HIGH) -> CrawlTask:
        """Create task for fetching index data"""
        return CrawlTask(
            id=str(uuid.uuid4()),
            source=DataSourceType.VNSTOCK,
            data_type=DataType.INDEX_DATA,
            priority=priority
        )

    @staticmethod
    def create_news_crawl_task(priority: Priority = Priority.LOW) -> CrawlTask:
        """Create task for crawling news"""
        return CrawlTask(
            id=str(uuid.uuid4()),
            source=DataSourceType.NEWS,
            data_type=DataType.NEWS_ARTICLE,
            priority=priority
        )


if __name__ == "__main__":
    # Example usage
    redis_client = redis.from_url("redis://localhost:6379")
    task_manager = TaskManager(redis_client)

    # Schedule recurring tasks
    asyncio.run(task_manager.schedule_recurring_task(
        name="VNIndex Update",
        task_template=TaskTemplates.create_index_data_task(),
        cron_expression="*/5 * * * *"  # Every 5 minutes
    ))

    # Start scheduler
    # asyncio.run(task_manager.run_scheduler())
