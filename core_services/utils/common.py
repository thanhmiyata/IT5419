"""
Common Constants and Enumerations for Data Ingestion Module
===========================================================

Centralized definitions for all constants, enumerations, and shared types
used across the Vietnamese stock market data ingestion pipeline.

This module serves as single source of truth for all configuration values,
preventing duplication and ensuring consistency across the codebase.
"""

from enum import Enum

# ============================================================================
# Data Source and Type Enumerations
# ============================================================================


class DataSourceType(Enum):
    """Enumeration of supported data sources"""
    VNSTOCK = "vnstock"
    CAFEF = "cafef"
    VIETFIN = "vietfin"
    HOSE = "hose"
    HNX = "hnx"
    NEWS = "news"
    INVESTING_COM = "investing_com"
    VNEXPRESS = "vnexpress"
    VNDIRECT = "vndirect"
    HOSE_OFFICIAL = "hose_official"
    # Source types (for schema)
    WEB_SCRAPING = "web_scraping"
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    FTP = "ftp"
    LIBRARY = "library"


class DataType(Enum):
    """Types of financial data"""
    STOCK_PRICE = "stock_price"
    INDEX_DATA = "index_data"
    VOLUME_DATA = "volume_data"
    NEWS_ARTICLE = "news_article"
    FINANCIAL_REPORT = "financial_report"
    TECHNICAL_INDICATOR = "technical_indicator"
    HISTORICAL_DATA = "historical_data"


class DataCategory(Enum):
    """Categories of data capabilities"""
    REAL_TIME_PRICES = "real_time_prices"
    HISTORICAL_PRICES = "historical_prices"
    COMPANY_FUNDAMENTALS = "company_fundamentals"
    FINANCIAL_STATEMENTS = "financial_statements"
    TECHNICAL_ANALYSIS = "technical_analysis"
    MARKET_NEWS = "market_news"
    TRADING_SIGNALS = "trading_signals"
    MARKET_DEPTH = "market_depth"
    # Extended categories
    MARKET_INDICES = "market_indices"
    VOLUME_DATA = "volume_data"
    TOP_MOVERS = "top_movers"
    COMPANY_INFO = "company_info"
    FINANCIAL_REPORTS = "financial_reports"
    TECHNICAL_INDICATORS = "technical_indicators"
    NEWS_SENTIMENT = "news_sentiment"
    ECONOMIC_CALENDAR = "economic_calendar"
    CHARTS_DATA = "charts_data"
    INSIDER_TRADING = "insider_trading"
    FOREIGN_FLOWS = "foreign_flows"


class ReliabilityLevel(Enum):
    """Source reliability levels"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Task Management Enumerations
# ============================================================================

class Priority(Enum):
    """Task priority levels"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ScheduleType(Enum):
    """Task scheduling types"""
    ONE_TIME = "one_time"
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    INTERVAL = "interval"
    RECURRING = "recurring"
    CRON = "cron"
    MARKET_HOURS = "market_hours"


# ============================================================================
# Data Quality Enumerations
# ============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class QualityDimension(Enum):
    """Data quality assessment dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


# ============================================================================
# Monitoring and Alerting Enumerations
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ============================================================================
# Redis Configuration
# ============================================================================

# Queue names
REDIS_QUEUE_HIGH_PRIORITY = "crawl_queue:high"
REDIS_QUEUE_MEDIUM_PRIORITY = "crawl_queue:medium"
REDIS_QUEUE_LOW_PRIORITY = "crawl_queue:low"

# Redis key prefixes
REDIS_KEY_PREFIX_TASK = "task:"
REDIS_KEY_PREFIX_STOCK = "stock:"
REDIS_KEY_PREFIX_INDEX = "index:"
REDIS_KEY_PREFIX_LOCK = "lock:"
REDIS_KEY_PREFIX_METRICS = "metrics:"

# Redis TTL values (seconds)
REDIS_TTL_STOCK_PRICE = 300  # 5 minutes
REDIS_TTL_INDEX_DATA = 60    # 1 minute
REDIS_TTL_NEWS = 1800        # 30 minutes
REDIS_TTL_LOCK = 30          # 30 seconds


# ============================================================================
# Database Configuration
# ============================================================================

# Table names
DB_TABLE_STOCKS = "stocks"
DB_TABLE_STOCK_PRICES = "stock_prices"
DB_TABLE_INDICES = "indices"
DB_TABLE_INDEX_HISTORY = "index_history"
DB_TABLE_NEWS_ARTICLES = "news_articles"
DB_TABLE_COMPANIES = "companies"
DB_TABLE_FINANCIAL_STATEMENTS = "financial_statements"

# Default values
DB_DEFAULT_BATCH_SIZE = 100
DB_DEFAULT_TIMEOUT = 30  # seconds


# ============================================================================
# Crawling Configuration
# ============================================================================

# Rate limiting (requests per second)
RATE_LIMIT_VNSTOCK = 5
RATE_LIMIT_CAFEF = 2
RATE_LIMIT_INVESTING_COM = 1
RATE_LIMIT_NEWS = 1
RATE_LIMIT_DEFAULT = 2

# Timeouts (seconds)
HTTP_TIMEOUT_DEFAULT = 30
HTTP_TIMEOUT_SCRAPING = 45

# Retry configuration
MAX_RETRIES_DEFAULT = 3
RETRY_BACKOFF_MULTIPLIER = 1
RETRY_BACKOFF_MIN = 2  # seconds
RETRY_BACKOFF_MAX = 10  # seconds

# User agents
USER_AGENT_DEFAULT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
USER_AGENT_MOBILE = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"


# ============================================================================
# Scheduling Configuration
# ============================================================================

# Intervals (seconds)
SCHEDULE_INTERVAL_REALTIME = 30     # 30 seconds for high-priority stocks
SCHEDULE_INTERVAL_REGULAR = 300     # 5 minutes for regular stocks
SCHEDULE_INTERVAL_INDEX = 60        # 1 minute for market index
SCHEDULE_INTERVAL_NEWS = 900        # 15 minutes for news
SCHEDULE_INTERVAL_BATCH = 3600      # 1 hour for batch operations

# Market hours (Vietnam time, 24-hour format)
MARKET_OPEN_HOUR = 9    # 9 AM
MARKET_CLOSE_HOUR = 15  # 3 PM

# Monitoring intervals
MONITORING_METRICS_INTERVAL = 60    # 1 minute
MONITORING_HEALTH_CHECK_INTERVAL = 30  # 30 seconds


# ============================================================================
# Pipeline Configuration
# ============================================================================

# Worker configuration
PIPELINE_DEFAULT_WORKERS = 4
PIPELINE_MAX_WORKERS = 16
PIPELINE_MIN_WORKERS = 1

# Queue configuration
PIPELINE_QUEUE_MAX_SIZE = 1000
PIPELINE_TASK_TIMEOUT = 30  # seconds

# Health check
HEALTH_CHECK_PORT = 8081
HEALTH_CHECK_PATH = "/health"


# ============================================================================
# Data Validation Configuration
# ============================================================================

# Stock price validation
STOCK_PRICE_MIN = 0.1           # Minimum valid stock price (VND)
STOCK_PRICE_MAX = 10000000      # Maximum valid stock price (10M VND)
STOCK_VOLUME_MIN = 0            # Minimum trading volume
STOCK_CHANGE_PERCENT_MAX = 100  # Maximum daily change (100%)

# Index validation
INDEX_VALUE_MIN = 0
INDEX_VALUE_MAX = 100000

# Data freshness
DATA_FRESHNESS_MAX_AGE = 86400  # 24 hours (seconds)


# ============================================================================
# Vietnamese Stock Market Specific
# ============================================================================

# Major Vietnamese stock symbols
VN_MAJOR_STOCKS = [
    "VNM", "FPT", "HPG", "VCB", "VIC",
    "VHM", "MSN", "MWG", "SAB", "GAS",
    "BID", "CTG", "VPB", "TCB", "MBB"
]

# Market indices
VN_INDICES = [
    "VNINDEX",  # Ho Chi Minh Stock Exchange
    "HNX",      # Hanoi Stock Exchange
    "UPCOM"     # Unlisted Public Company Market
]

# Stock exchanges
VN_EXCHANGES = ["HOSE", "HNX", "UPCOM"]

# Trading sessions
VN_MORNING_SESSION_START = "09:00"
VN_MORNING_SESSION_END = "11:30"
VN_AFTERNOON_SESSION_START = "13:00"
VN_AFTERNOON_SESSION_END = "15:00"


# ============================================================================
# Monitoring Thresholds
# ============================================================================

# Alert thresholds
ALERT_THRESHOLD_ERROR_RATE = 0.05      # 5% error rate triggers alert
ALERT_THRESHOLD_QUEUE_SIZE = 500       # Queue size threshold
ALERT_THRESHOLD_RESPONSE_TIME = 5000   # 5 seconds response time
ALERT_THRESHOLD_NO_TASKS_HOURS = 1     # Alert if no tasks for 1 hour

# Performance thresholds
PERF_THRESHOLD_SUCCESS_RATE = 0.95     # 95% minimum success rate
PERF_THRESHOLD_TASKS_PER_HOUR = 60     # Minimum tasks per hour


# ============================================================================
# Logging Configuration
# ============================================================================

# Log levels
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"

# Log format
LOG_FORMAT_DEFAULT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_DETAILED = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# Log file rotation
LOG_MAX_BYTES = 10485760    # 10MB
LOG_BACKUP_COUNT = 5


# ============================================================================
# Schema Versions
# ============================================================================

SCHEMA_VERSION_CURRENT = "1.0"
SCHEMA_VERSION_COMPATIBILITY = ["1.0"]


# ============================================================================
# API Endpoints (if backend integration)
# ============================================================================

API_ENDPOINT_INGEST = "/api/v1/ingest"
API_ENDPOINT_STOCK_PRICE = "/api/v1/stocks/price"
API_ENDPOINT_INDEX = "/api/v1/stocks/index"
API_ENDPOINT_NEWS = "/api/v1/news"

# API defaults
API_DEFAULT_PAGE_SIZE = 50
API_MAX_PAGE_SIZE = 500


# ============================================================================
# File Paths (relative to data_ingestion directory)
# ============================================================================

PATH_SCHEMAS = "schemas"
PATH_LOGS = "logs"
PATH_CACHE = "cache"
PATH_CONFIG = "config"

# Schema files
SCHEMA_FILE_VIETNAMESE_SOURCES = "vietnamese_sources.json"
SCHEMA_FILE_DEVELOPMENT_SOURCES = "development_sources.json"


# ============================================================================
# Error Messages
# ============================================================================

ERROR_MSG_DATABASE_CONNECTION = "Failed to connect to database"
ERROR_MSG_REDIS_CONNECTION = "Failed to connect to Redis"
ERROR_MSG_INVALID_SYMBOL = "Invalid stock symbol"
ERROR_MSG_NO_DATA = "No data available"
ERROR_MSG_RATE_LIMIT = "Rate limit exceeded"
ERROR_MSG_TIMEOUT = "Request timeout"
ERROR_MSG_VALIDATION_FAILED = "Data validation failed"


# ============================================================================
# Success Messages
# ============================================================================

SUCCESS_MSG_DATA_STORED = "Data stored successfully"
SUCCESS_MSG_TASK_COMPLETED = "Task completed successfully"
SUCCESS_MSG_PIPELINE_STARTED = "Pipeline started successfully"
SUCCESS_MSG_PIPELINE_STOPPED = "Pipeline stopped successfully"


# ============================================================================
# Metric Names (for Prometheus/monitoring)
# ============================================================================

METRIC_TASKS_TOTAL = "pipeline_tasks_total"
METRIC_TASKS_SUCCESS = "pipeline_tasks_success"
METRIC_TASKS_FAILED = "pipeline_tasks_failed"
METRIC_TASKS_RUNNING = "pipeline_tasks_running"
METRIC_RESPONSE_TIME = "pipeline_response_time_seconds"
METRIC_QUEUE_SIZE = "pipeline_queue_size"
METRIC_SUCCESS_RATE = "pipeline_success_rate"


# ============================================================================
# Feature Flags
# ============================================================================

FEATURE_ENABLE_CACHING = True
FEATURE_ENABLE_RETRY = True
FEATURE_ENABLE_METRICS = True
FEATURE_ENABLE_ALERTS = True
FEATURE_ENABLE_HEALTH_CHECK = True


# ============================================================================
# Environment Names
# ============================================================================

ENV_DEVELOPMENT = "development"
ENV_TESTING = "testing"
ENV_STAGING = "staging"
ENV_PRODUCTION = "production"
