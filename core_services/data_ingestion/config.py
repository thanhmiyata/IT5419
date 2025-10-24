"""
Configuration Management for Vietnamese Stock Market Data Pipeline
================================================================

Centralized configuration management with environment variable support
and validation.

"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5

    @property
    def url(self) -> str:
        """Generate Redis URL"""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "stockaids"
    username: str = "postgres"
    password: str = "password"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    ssl_require: bool = False

    @property
    def url(self) -> str:
        """Generate database URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class CrawlerSourceConfig:
    """Configuration for individual data sources"""
    enabled: bool = True
    rate_limit: int = 5  # requests per second
    timeout: int = 30    # seconds
    retry_count: int = 3
    retry_delay: int = 5
    priority_weight: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlerConfig:
    """Main crawler configuration"""
    # Source configurations
    vnstock: CrawlerSourceConfig = field(default_factory=lambda: CrawlerSourceConfig(
        enabled=True, rate_limit=5, timeout=30, priority_weight=0.4
    ))

    cafef: CrawlerSourceConfig = field(default_factory=lambda: CrawlerSourceConfig(
        enabled=True, rate_limit=2, timeout=30, priority_weight=0.2,
        headers={
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/91.0.4472.124 Safari/537.36')
        }
    ))

    vietfin: CrawlerSourceConfig = field(default_factory=lambda: CrawlerSourceConfig(
        enabled=True, rate_limit=3, timeout=30, priority_weight=0.1
    ))

    hose: CrawlerSourceConfig = field(default_factory=lambda: CrawlerSourceConfig(
        enabled=True, rate_limit=1, timeout=60, priority_weight=0.3
    ))

    news: CrawlerSourceConfig = field(default_factory=lambda: CrawlerSourceConfig(
        enabled=True, rate_limit=1, timeout=45, priority_weight=0.1
    ))


@dataclass
class SchedulerConfig:
    """Task scheduler configuration"""
    # Real-time tasks (high frequency)
    realtime_stocks: List[str] = field(default_factory=lambda: [
        "VNM", "VIC", "VCB", "BID", "CTG", "FPT", "GAS", "HPG", "MSN", "VHM",
        "BVH", "GVR", "HDB", "MWG", "PLX", "POW", "SAB", "STB", "TCB", "VJC"
    ])
    realtime_interval: int = 30  # seconds

    # Regular update tasks
    regular_stocks: List[str] = field(default_factory=lambda: [
        "ACB", "ANV", "BCM", "BIC", "CMG", "DCM", "DHG", "DIG", "DXG", "EIB",
        "EVF", "FLC", "GMD", "HAG", "HNG", "HSG", "ITA", "KBC", "KDC", "KDH",
        "LPB", "MBB", "NLG", "NVL", "OCB", "PDR", "PNJ", "PVD", "REE", "ROS",
        "SHB", "SSI", "TPB", "VCI", "VGC", "VIB", "VND", "VPB", "VRE", "VSH"
    ])
    regular_interval: int = 300  # 5 minutes

    # Index updates
    index_update_interval: int = 60  # 1 minute

    # News crawling
    news_crawl_interval: int = 600  # 10 minutes

    # Historical data backfill
    historical_batch_size: int = 50
    historical_interval: int = 3600  # 1 hour

    # Market hours (Vietnam time)
    market_open_hour: int = 9
    market_close_hour: int = 15
    weekend_enabled: bool = False


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    # Worker configuration
    num_workers: int = 4
    worker_max_tasks: int = 100
    worker_timeout: int = 300

    # Queue configuration
    high_priority_queue_size: int = 1000
    medium_priority_queue_size: int = 5000
    low_priority_queue_size: int = 10000

    # Data processing
    batch_size: int = 100
    batch_timeout: int = 30
    data_retention_days: int = 365

    # Validation
    enable_data_validation: bool = True
    enable_cross_source_validation: bool = True
    validation_threshold: float = 0.05  # 5% price difference threshold

    # Caching
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_max_entries: int = 10000

    # Monitoring
    metrics_enabled: bool = True
    metrics_interval: int = 60
    health_check_interval: int = 30

    # Error handling
    max_retry_attempts: int = 3
    retry_exponential_base: int = 2
    dead_letter_queue_enabled: bool = True

    # Rate limiting
    global_rate_limit: int = 100  # requests per second across all sources
    rate_limit_window: int = 60   # seconds


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    structured_logging: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enabled: bool = True

    # Health check endpoints
    health_check_port: int = 8080
    health_check_path: str = "/health"

    # Metrics
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    metrics_interval: int = 60  # Seconds between metrics collection

    # Alerting thresholds
    error_rate_threshold: float = 0.1  # 10%
    response_time_threshold: float = 30.0  # 30 seconds
    queue_size_threshold: int = 1000
    memory_usage_threshold: float = 0.8  # 80%
    cpu_usage_threshold: float = 0.8    # 80%

    # Notification settings
    email_alerts_enabled: bool = False
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1

    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Rate limiting
    api_rate_limit: int = 1000  # requests per minute
    api_rate_limit_window: int = 60


@dataclass
class MainConfig:
    """Main application configuration"""
    # Environment
    environment: str = "development"
    debug: bool = True

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def __post_init__(self):
        """Post-initialization validation"""
        # Adjust settings based on environment
        if self.environment == "production":
            self.debug = False
            self.logging.level = "WARNING"
            self.api.debug = False
            self.api.reload = False


class ConfigManager:
    """Configuration management with environment variable support"""

    @staticmethod
    def load_from_env() -> MainConfig:
        """Load configuration from environment variables"""
        config = MainConfig()

        # Environment
        config.environment = os.getenv("ENVIRONMENT", "development")
        config.debug = os.getenv("DEBUG", "true").lower() == "true"

        # Database
        config.database.host = os.getenv("DB_HOST", "localhost")
        config.database.port = int(os.getenv("DB_PORT", "5432"))
        config.database.database = os.getenv("DB_NAME", "stockaids")
        config.database.username = os.getenv("DB_USER", "postgres")
        config.database.password = os.getenv("DB_PASSWORD", "password")

        # Redis
        config.redis.host = os.getenv("REDIS_HOST", "localhost")
        config.redis.port = int(os.getenv("REDIS_PORT", "6379"))
        config.redis.db = int(os.getenv("REDIS_DB", "0"))
        config.redis.password = os.getenv("REDIS_PASSWORD")

        # Pipeline
        config.pipeline.num_workers = int(os.getenv("PIPELINE_WORKERS", "4"))
        config.pipeline.batch_size = int(os.getenv("PIPELINE_BATCH_SIZE", "100"))

        # API
        config.api.host = os.getenv("API_HOST", "0.0.0.0")
        config.api.port = int(os.getenv("API_PORT", "8000"))

        # Monitoring
        config.monitoring.prometheus_port = int(os.getenv("PROMETHEUS_PORT", "9090"))
        config.monitoring.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        return config

    @staticmethod
    def load_from_file(config_path: str) -> MainConfig:
        """Load configuration from YAML file"""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f)

        return ConfigManager._dict_to_config(data)

    @staticmethod
    def save_to_file(config: MainConfig, config_path: str):
        """Save configuration to YAML file"""
        path = Path(config_path)
        data = ConfigManager._config_to_dict(config)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                json.dump(data, f, indent=2, default=str)
            else:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> MainConfig:
        """Convert dictionary to configuration object"""
        # This would be a complex recursive function to handle nested dataclasses
        # For now, return default config with some key overrides
        config = MainConfig()

        if 'environment' in data:
            config.environment = data['environment']

        if 'database' in data:
            db_data = data['database']
            if 'host' in db_data:
                config.database.host = db_data['host']
            if 'port' in db_data:
                config.database.port = db_data['port']
            # ... handle other database fields

        # Similar handling for other config sections
        return config

    @staticmethod
    def _config_to_dict(config: MainConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        # This would convert the dataclass to a dictionary
        # For now, simplified implementation
        return {
            'environment': config.environment,
            'debug': config.debug,
            'database': {
                'host': config.database.host,
                'port': config.database.port,
                'database': config.database.database,
                'username': config.database.username,
                'pool_size': config.database.pool_size
            },
            'redis': {
                'host': config.redis.host,
                'port': config.redis.port,
                'db': config.redis.db
            },
            'pipeline': {
                'num_workers': config.pipeline.num_workers,
                'batch_size': config.pipeline.batch_size
            }
        }

    @staticmethod
    def validate_config(config: MainConfig) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate database config
        if not config.database.host:
            errors.append("Database host is required")

        if config.database.port <= 0 or config.database.port > 65535:
            errors.append("Database port must be between 1 and 65535")

        # Validate Redis config
        if not config.redis.host:
            errors.append("Redis host is required")

        if config.redis.port <= 0 or config.redis.port > 65535:
            errors.append("Redis port must be between 1 and 65535")

        # Validate pipeline config
        if config.pipeline.num_workers <= 0:
            errors.append("Number of workers must be positive")

        if config.pipeline.batch_size <= 0:
            errors.append("Batch size must be positive")

        # Validate scheduler config
        if not config.scheduler.realtime_stocks:
            errors.append("At least one realtime stock must be configured")

        # Validate monitoring config
        if config.monitoring.error_rate_threshold < 0 or config.monitoring.error_rate_threshold > 1:
            errors.append("Error rate threshold must be between 0 and 1")

        return errors


# Configuration presets for different environments
class ConfigPresets:
    """Predefined configuration presets"""

    @staticmethod
    def development() -> MainConfig:
        """Development environment preset"""
        config = MainConfig()
        config.environment = "development"
        config.debug = True
        config.logging.level = "DEBUG"
        config.logging.console_output = True
        config.pipeline.num_workers = 2
        config.crawler.vnstock.rate_limit = 3
        config.crawler.cafef.rate_limit = 1
        return config

    @staticmethod
    def testing() -> MainConfig:
        """Testing environment preset"""
        config = MainConfig()
        config.environment = "testing"
        config.debug = True
        config.database.database = "stockaids_test"
        config.redis.db = 1
        config.pipeline.num_workers = 1
        config.scheduler.realtime_interval = 60
        return config

    @staticmethod
    def production() -> MainConfig:
        """Production environment preset"""
        config = MainConfig()
        config.environment = "production"
        config.debug = False
        config.logging.level = "INFO"
        config.logging.file_path = "/var/log/stockaids/pipeline.log"
        config.pipeline.num_workers = 8
        config.monitoring.enabled = True
        config.api.workers = 4
        return config


if __name__ == "__main__":
    # Example usage

    # Load from environment
    config = ConfigManager.load_from_env()

    # Validate configuration
    errors = ConfigManager.validate_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")

    # Save example config
    ConfigManager.save_to_file(config, "config.yaml")

    # Use preset
    prod_config = ConfigPresets.production()
    print(f"Production config environment: {prod_config.environment}")
