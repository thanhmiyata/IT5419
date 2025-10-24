"""
Data Source Schema Definition for Vietnamese Stock Market Pipeline
================================================================

Defines comprehensive schemas for all data sources including capabilities,
rate limits, data types, reliability scores, and configuration parameters.

"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

from core_services.utils.logger_utils import logger


class DataSourceType(Enum):
    """Types of data sources"""
    WEB_SCRAPING = "web_scraping"
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    FTP = "ftp"
    LIBRARY = "library"


class ReliabilityLevel(Enum):
    """Reliability levels for data sources"""
    VERY_HIGH = "very_high"  # 99%+ uptime, official sources
    HIGH = "high"           # 95%+ uptime, stable sources
    MEDIUM = "medium"       # 85%+ uptime, community sources
    LOW = "low"            # <85% uptime, experimental


class DataCategory(Enum):
    """Categories of financial data"""
    REAL_TIME_PRICES = "real_time_prices"
    HISTORICAL_PRICES = "historical_prices"
    MARKET_INDICES = "market_indices"
    VOLUME_DATA = "volume_data"
    TOP_MOVERS = "top_movers"
    COMPANY_INFO = "company_info"
    FINANCIAL_REPORTS = "financial_reports"
    TECHNICAL_INDICATORS = "technical_indicators"
    NEWS_SENTIMENT = "news_sentiment"
    ECONOMIC_CALENDAR = "economic_calendar"
    CHARTS_DATA = "charts_data"
    TRADING_SIGNALS = "trading_signals"
    MARKET_DEPTH = "market_depth"
    INSIDER_TRADING = "insider_trading"
    FOREIGN_FLOWS = "foreign_flows"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    burst_limit: int = 10
    backoff_factor: float = 1.5
    max_retry_delay: int = 300  # seconds


@dataclass
class DataTypeCapability:
    """Capability definition for a specific data type"""
    supported: bool = True
    update_frequency_seconds: int = 300  # 5 minutes default
    historical_depth_days: Optional[int] = None
    real_time: bool = False
    batch_support: bool = False
    max_symbols_per_request: int = 1
    data_quality_score: float = 0.8  # 0.0-1.0
    cost_per_request: float = 0.0  # 0 for free sources
    notes: Optional[str] = None


@dataclass
class SourceSchema:
    """Complete schema definition for a data source"""

    # Basic Information
    source_id: str
    name: str
    description: str
    source_type: DataSourceType
    reliability: ReliabilityLevel

    # Connection Details
    base_url: str
    api_version: Optional[str] = None
    auth_required: bool = False
    api_key_required: bool = False

    # Rate Limiting
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Data Capabilities
    capabilities: Dict[DataCategory, DataTypeCapability] = field(default_factory=dict)

    # Geographic & Market Coverage
    supported_exchanges: List[str] = field(default_factory=lambda: ["HOSE", "HNX", "UPCOM"])
    supported_currencies: List[str] = field(default_factory=lambda: ["VND"])
    timezone: str = "Asia/Ho_Chi_Minh"

    # Technical Details
    request_format: str = "json"  # json, xml, html, csv
    response_format: str = "json"
    encoding: str = "utf-8"
    user_agent_required: bool = True
    headers: Dict[str, str] = field(default_factory=dict)

    # Operational Details
    maintenance_windows: List[str] = field(default_factory=list)  # Cron expressions
    expected_downtime: List[str] = field(default_factory=list)
    contact_info: Optional[str] = None
    documentation_url: Optional[str] = None

    # Performance Metrics
    average_response_time_ms: float = 1000
    success_rate: float = 0.95
    data_freshness_seconds: int = 60

    # Integration Details
    parser_class: Optional[str] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    transformation_rules: Dict[str, Any] = field(default_factory=dict)

    # Monitoring
    health_check_url: Optional[str] = None
    status_page_url: Optional[str] = None


# Vietnamese Stock Market Data Sources Configuration
VIETNAMESE_SOURCES_SCHEMA = {

    # VNSTOCK Library - Primary Source
    "vnstock": SourceSchema(
        source_id="vnstock",
        name="VNSTOCK Python Library",
        description="Comprehensive Vietnamese stock market data library with official API access",
        source_type=DataSourceType.LIBRARY,
        reliability=ReliabilityLevel.VERY_HIGH,
        base_url="https://api.vnstock.io/v1",
        rate_limits=RateLimitConfig(
            requests_per_second=5.0,
            requests_per_minute=300,
            requests_per_hour=18000,
            burst_limit=20
        ),
        capabilities={
            DataCategory.REAL_TIME_PRICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=30,
                real_time=True,
                batch_support=True,
                max_symbols_per_request=50,
                data_quality_score=0.95
            ),
            DataCategory.HISTORICAL_PRICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=3600,  # 1 hour
                historical_depth_days=3650,    # 10 years
                batch_support=True,
                max_symbols_per_request=20,
                data_quality_score=0.98
            ),
            DataCategory.MARKET_INDICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=60,   # 1 minute
                real_time=True,
                data_quality_score=0.99
            ),
            DataCategory.COMPANY_INFO: DataTypeCapability(
                supported=True,
                update_frequency_seconds=86400,  # Daily
                data_quality_score=0.95
            ),
            DataCategory.FINANCIAL_REPORTS: DataTypeCapability(
                supported=True,
                update_frequency_seconds=86400,  # Daily
                historical_depth_days=1825,     # 5 years
                data_quality_score=0.92
            )
        },
        supported_exchanges=["HOSE", "HNX", "UPCOM"],
        average_response_time_ms=800,
        success_rate=0.98,
        data_freshness_seconds=30,
        documentation_url="https://vnstock.site/docs/"
    ),

    # CafeF.vn - Secondary Source
    "cafef": SourceSchema(
        source_id="cafef",
        name="CafeF Vietnam Financial Portal",
        description="Leading Vietnamese financial news and data portal with comprehensive market coverage",
        source_type=DataSourceType.WEB_SCRAPING,
        reliability=ReliabilityLevel.HIGH,
        base_url="https://cafef.vn",
        rate_limits=RateLimitConfig(
            requests_per_second=2.0,
            requests_per_minute=120,
            requests_per_hour=7200,
            burst_limit=5
        ),
        capabilities={
            DataCategory.REAL_TIME_PRICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,  # 5 minutes
                real_time=False,
                data_quality_score=0.88
            ),
            DataCategory.MARKET_INDICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,
                data_quality_score=0.90
            ),
            DataCategory.TOP_MOVERS: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,
                batch_support=True,
                max_symbols_per_request=20,
                data_quality_score=0.85
            ),
            DataCategory.NEWS_SENTIMENT: DataTypeCapability(
                supported=True,
                update_frequency_seconds=600,  # 10 minutes
                data_quality_score=0.75
            ),
            DataCategory.CHARTS_DATA: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,
                data_quality_score=0.80
            )
        },
        headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/91.0.4472.124 Safari/537.36")
        },
        average_response_time_ms=1500,
        success_rate=0.92,
        data_freshness_seconds=300
    ),

    # Investing.com Vietnam
    "investing_com": SourceSchema(
        source_id="investing_com",
        name="Investing.com Vietnam Section",
        description="International financial platform with Vietnamese market coverage",
        source_type=DataSourceType.WEB_SCRAPING,
        reliability=ReliabilityLevel.HIGH,
        base_url="https://www.investing.com",
        rate_limits=RateLimitConfig(
            requests_per_second=1.0,
            requests_per_minute=60,
            requests_per_hour=3600,
            burst_limit=3
        ),
        capabilities={
            DataCategory.MARKET_INDICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=60,   # 1 minute
                real_time=True,
                data_quality_score=0.93
            ),
            DataCategory.HISTORICAL_PRICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=3600,
                historical_depth_days=7300,   # 20 years
                data_quality_score=0.90
            ),
            DataCategory.ECONOMIC_CALENDAR: DataTypeCapability(
                supported=True,
                update_frequency_seconds=3600,
                data_quality_score=0.88
            ),
            DataCategory.TECHNICAL_INDICATORS: DataTypeCapability(
                supported=True,
                update_frequency_seconds=900,  # 15 minutes
                data_quality_score=0.85
            )
        },
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
        average_response_time_ms=2000,
        success_rate=0.90,
        data_freshness_seconds=60
    ),

    # Vietstock Finance
    "vietstock": SourceSchema(
        source_id="vietstock",
        name="Vietstock Finance Portal",
        description="Vietnamese stock analysis platform with technical indicators",
        source_type=DataSourceType.WEB_SCRAPING,
        reliability=ReliabilityLevel.MEDIUM,
        base_url="https://finance.vietstock.vn",
        rate_limits=RateLimitConfig(
            requests_per_second=1.0,
            requests_per_minute=20,
            requests_per_hour=1200,
            burst_limit=2
        ),
        capabilities={
            DataCategory.REAL_TIME_PRICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,
                data_quality_score=0.85
            ),
            DataCategory.TECHNICAL_INDICATORS: DataTypeCapability(
                supported=True,
                update_frequency_seconds=900,
                data_quality_score=0.88
            ),
            DataCategory.FINANCIAL_REPORTS: DataTypeCapability(
                supported=True,
                update_frequency_seconds=86400,
                data_quality_score=0.82
            ),
            DataCategory.COMPANY_INFO: DataTypeCapability(
                supported=True,
                update_frequency_seconds=86400,
                data_quality_score=0.80
            )
        },
        average_response_time_ms=2500,
        success_rate=0.85,
        data_freshness_seconds=300
    ),

    # Cophieu68
    "cophieu68": SourceSchema(
        source_id="cophieu68",
        name="Cophieu68 Stock Screening Platform",
        description="Vietnamese stock screening and technical analysis platform",
        source_type=DataSourceType.WEB_SCRAPING,
        reliability=ReliabilityLevel.MEDIUM,
        base_url="https://www.cophieu68.vn",
        rate_limits=RateLimitConfig(
            requests_per_second=0.5,
            requests_per_minute=15,
            requests_per_hour=900,
            burst_limit=2
        ),
        capabilities={
            DataCategory.TECHNICAL_INDICATORS: DataTypeCapability(
                supported=True,
                update_frequency_seconds=600,  # 10 minutes
                data_quality_score=0.80
            ),
            DataCategory.TRADING_SIGNALS: DataTypeCapability(
                supported=True,
                update_frequency_seconds=1800,  # 30 minutes
                data_quality_score=0.75
            ),
            DataCategory.CHARTS_DATA: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,
                data_quality_score=0.78
            )
        },
        average_response_time_ms=3000,
        success_rate=0.80,
        data_freshness_seconds=600
    ),

    # FireAnt (Alternative)
    "fireant": SourceSchema(
        source_id="fireant",
        name="FireAnt Financial Data",
        description="Alternative Vietnamese financial data provider",
        source_type=DataSourceType.REST_API,
        reliability=ReliabilityLevel.MEDIUM,
        base_url="https://api.fireant.vn/v1",
        api_version="v1",
        rate_limits=RateLimitConfig(
            requests_per_second=3.0,
            requests_per_minute=180,
            requests_per_hour=10800,
            burst_limit=10
        ),
        capabilities={
            DataCategory.REAL_TIME_PRICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=60,
                real_time=True,
                data_quality_score=0.83
            ),
            DataCategory.MARKET_INDICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=60,
                data_quality_score=0.85
            ),
            DataCategory.HISTORICAL_PRICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=3600,
                historical_depth_days=1095,  # 3 years
                data_quality_score=0.82
            )
        },
        average_response_time_ms=1200,
        success_rate=0.88,
        data_freshness_seconds=60
    ),

    # HOSE Official (Limited)
    "hose_official": SourceSchema(
        source_id="hose_official",
        name="Ho Chi Minh Stock Exchange Official",
        description="Official data from HOSE exchange (limited public access)",
        source_type=DataSourceType.WEB_SCRAPING,
        reliability=ReliabilityLevel.VERY_HIGH,
        base_url="https://www.hsx.vn",
        rate_limits=RateLimitConfig(
            requests_per_second=0.5,
            requests_per_minute=10,
            requests_per_hour=600,
            burst_limit=1
        ),
        capabilities={
            DataCategory.MARKET_INDICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,
                data_quality_score=1.0  # Official source
            ),
            DataCategory.TOP_MOVERS: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,
                data_quality_score=1.0
            ),
            DataCategory.VOLUME_DATA: DataTypeCapability(
                supported=True,
                update_frequency_seconds=300,
                data_quality_score=1.0
            )
        },
        supported_exchanges=["HOSE"],
        average_response_time_ms=5000,
        success_rate=0.95,
        data_freshness_seconds=300,
        maintenance_windows=["0 18 * * 0"]  # Sunday 6 PM maintenance
    ),

    # VnDirect (Brokerage Data)
    "vndirect": SourceSchema(
        source_id="vndirect",
        name="VnDirect Securities",
        description="Major Vietnamese brokerage with public data access",
        source_type=DataSourceType.WEB_SCRAPING,
        reliability=ReliabilityLevel.HIGH,
        base_url="https://dchart.vndirect.com.vn",
        rate_limits=RateLimitConfig(
            requests_per_second=2.0,
            requests_per_minute=60,
            requests_per_hour=3600,
            burst_limit=5
        ),
        capabilities={
            DataCategory.REAL_TIME_PRICES: DataTypeCapability(
                supported=True,
                update_frequency_seconds=30,
                real_time=True,
                data_quality_score=0.92
            ),
            DataCategory.CHARTS_DATA: DataTypeCapability(
                supported=True,
                update_frequency_seconds=60,
                data_quality_score=0.90
            ),
            DataCategory.MARKET_DEPTH: DataTypeCapability(
                supported=True,
                update_frequency_seconds=30,
                real_time=True,
                data_quality_score=0.88
            )
        },
        average_response_time_ms=800,
        success_rate=0.93,
        data_freshness_seconds=30
    )
}


class JSONSchemaLoader:
    """Loads and validates source schemas from JSON files"""

    def __init__(self, schema_dir: str = None):
        if schema_dir is None:
            # Default to schemas directory relative to this file
            current_dir = Path(__file__).parent
            self.schema_dir = current_dir / "schemas"
        else:
            self.schema_dir = Path(schema_dir)

        # Load JSON schema template for validation
        self.json_schema = self._load_json_schema_template()

    def _load_json_schema_template(self) -> Optional[dict]:
        """Load JSON schema template for validation"""
        schema_template_path = self.schema_dir / "schema_template.json"
        if not schema_template_path.exists():
            logger.warning(f"Schema template not found at {schema_template_path}")
            return None

        try:
            with open(schema_template_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema template: {e}")
            return None

    def validate_json_schema(self, schema_data: dict) -> List[str]:
        """Validate JSON schema against template"""
        errors: List[str] = []
        try:
            jsonschema.validate(schema_data, self.json_schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema template error: {e.message}")

        return errors

    def load_schema_file(self, filename: str) -> Dict[str, SourceSchema]:
        """Load source schemas from JSON file"""
        schema_file = self.schema_dir / filename

        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON schema file {filename}: {e}")

        # Validate JSON schema
        validation_errors = self.validate_json_schema(schema_data)
        if validation_errors:
            logger.warning(f"Schema validation errors in {filename}: {validation_errors}")

        # Convert JSON to SourceSchema objects
        return self._json_to_source_schemas(schema_data)

    def _json_to_source_schemas(self, schema_data: dict) -> Dict[str, SourceSchema]:
        """Convert JSON schema data to SourceSchema objects"""
        schemas = {}

        for source_id, source_data in schema_data.get("sources", {}).items():
            try:
                # Convert string enums to enum objects
                source_type = DataSourceType(source_data["source_type"])
                reliability = ReliabilityLevel(source_data["reliability"])

                # Convert rate limits
                rate_limits = RateLimitConfig(**source_data["rate_limits"])

                # Convert capabilities
                capabilities = {}
                for cap_name, cap_data in source_data.get("capabilities", {}).items():
                    try:
                        data_category = DataCategory(cap_name)
                        capability = DataTypeCapability(**cap_data)
                        capabilities[data_category] = capability
                    except ValueError as e:
                        logger.warning(f"Unknown data category '{cap_name}' in source {source_id}: {e}")

                # Create SourceSchema object
                schema = SourceSchema(
                    source_id=source_data["source_id"],
                    name=source_data["name"],
                    description=source_data["description"],
                    source_type=source_type,
                    reliability=reliability,
                    base_url=source_data["base_url"],
                    api_version=source_data.get("api_version"),
                    auth_required=source_data.get("auth_required", False),
                    api_key_required=source_data.get("api_key_required", False),
                    rate_limits=rate_limits,
                    capabilities=capabilities,
                    supported_exchanges=source_data.get("supported_exchanges", []),
                    supported_currencies=source_data.get("supported_currencies", ["VND"]),
                    timezone=source_data.get("timezone", "Asia/Ho_Chi_Minh"),
                    request_format=source_data.get("request_format", "json"),
                    response_format=source_data.get("response_format", "json"),
                    encoding=source_data.get("encoding", "utf-8"),
                    user_agent_required=source_data.get("user_agent_required", False),
                    headers=source_data.get("headers", {}),
                    maintenance_windows=source_data.get("maintenance_windows", []),
                    expected_downtime=source_data.get("expected_downtime", []),
                    contact_info=source_data.get("contact_info"),
                    documentation_url=source_data.get("documentation_url"),
                    average_response_time_ms=source_data.get("average_response_time_ms", 1000),
                    success_rate=source_data.get("success_rate", 0.95),
                    data_freshness_seconds=source_data.get("data_freshness_seconds", 60),
                    parser_class=source_data.get("parser_class"),
                    validation_rules=source_data.get("validation_rules", {}),
                    transformation_rules=source_data.get("transformation_rules", {}),
                    health_check_url=source_data.get("health_check_url"),
                    status_page_url=source_data.get("status_page_url")
                )

                schemas[source_id] = schema

            except Exception as e:
                logger.error(f"Failed to create schema for source {source_id}: {e}")

        logger.info(f"Loaded {len(schemas)} source schemas from JSON")
        return schemas

    def save_schemas_to_json(self, schemas: Dict[str, SourceSchema], filename: str,
                             environment: str = "production") -> bool:
        """Save SourceSchema objects to JSON file"""
        schema_data = {
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat() + "Z",
            "description": f"Vietnamese Stock Market Data Sources Schema - {environment.title()} Environment",
            "environment": environment,
            "sources": {}
        }

        for source_id, schema in schemas.items():
            # Convert SourceSchema to JSON-serializable dict
            source_data = {
                "source_id": schema.source_id,
                "name": schema.name,
                "description": schema.description,
                "source_type": schema.source_type.value,
                "reliability": schema.reliability.value,
                "base_url": schema.base_url,
                "api_version": schema.api_version,
                "auth_required": schema.auth_required,
                "api_key_required": schema.api_key_required,
                "rate_limits": {
                    "requests_per_second": schema.rate_limits.requests_per_second,
                    "requests_per_minute": schema.rate_limits.requests_per_minute,
                    "requests_per_hour": schema.rate_limits.requests_per_hour,
                    "burst_limit": schema.rate_limits.burst_limit,
                    "backoff_factor": schema.rate_limits.backoff_factor,
                    "max_retry_delay": schema.rate_limits.max_retry_delay
                },
                "capabilities": {},
                "supported_exchanges": schema.supported_exchanges,
                "supported_currencies": schema.supported_currencies,
                "timezone": schema.timezone,
                "request_format": schema.request_format,
                "response_format": schema.response_format,
                "encoding": schema.encoding,
                "user_agent_required": schema.user_agent_required,
                "headers": schema.headers,
                "maintenance_windows": schema.maintenance_windows,
                "expected_downtime": schema.expected_downtime,
                "contact_info": schema.contact_info,
                "documentation_url": schema.documentation_url,
                "average_response_time_ms": schema.average_response_time_ms,
                "success_rate": schema.success_rate,
                "data_freshness_seconds": schema.data_freshness_seconds,
                "parser_class": schema.parser_class,
                "validation_rules": schema.validation_rules,
                "transformation_rules": schema.transformation_rules,
                "health_check_url": schema.health_check_url,
                "status_page_url": schema.status_page_url
            }

            # Convert capabilities
            for data_category, capability in schema.capabilities.items():
                source_data["capabilities"][data_category.value] = {
                    "supported": capability.supported,
                    "update_frequency_seconds": capability.update_frequency_seconds,
                    "historical_depth_days": capability.historical_depth_days,
                    "real_time": capability.real_time,
                    "batch_support": capability.batch_support,
                    "max_symbols_per_request": capability.max_symbols_per_request,
                    "data_quality_score": capability.data_quality_score,
                    "cost_per_request": capability.cost_per_request,
                    "notes": capability.notes
                }

            schema_data["sources"][source_id] = source_data

        # Save to file
        try:
            schema_file = self.schema_dir / filename
            with open(schema_file, "w", encoding="utf-8") as f:
                json.dump(schema_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved schema to {schema_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save schema to {filename}: {e}")
            return False

    def list_available_schemas(self) -> List[str]:
        """List available schema files"""
        if not self.schema_dir.exists():
            return []

        json_files = []
        for file in self.schema_dir.glob("*.json"):
            if file.name != "schema_template.json":  # Exclude template
                json_files.append(file.name)

        return sorted(json_files)


class SourceSchemaManager:
    """Manager for data source schemas and configurations with JSON support"""

    def __init__(self, schemas: Dict[str, SourceSchema] = None,
                 schema_file: str = None, environment: str = None):
        """
        Initialize SourceSchemaManager

        Args:
            schemas: Pre-loaded schemas dict (for backward compatibility)
            schema_file: JSON schema file to load (e.g., "vietnamese_sources.json")
            environment: Environment to auto-select schema file (dev/staging/production)
        """
        self.json_loader = JSONSchemaLoader()

        # Priority: 1) Provided schemas, 2) Specified file, 3) Environment-based, 4) Default
        if schemas:
            self.schemas = schemas
            logger.info("Using provided schemas")
        elif schema_file:
            self.schemas = self.load_from_file(schema_file)
            logger.info(f"Loaded schemas from file: {schema_file}")
        elif environment:
            self.schemas = self.load_for_environment(environment)
            logger.info(f"Loaded schemas for environment: {environment}")
        else:
            # Default: try to load production schemas, fallback to hardcoded
            try:
                self.schemas = self.load_from_file("vietnamese_sources.json")
                logger.info("Loaded default production schemas from JSON")
            except Exception as e:
                logger.warning(f"Failed to load JSON schemas, using hardcoded fallback: {e}")
                self.schemas = VIETNAMESE_SOURCES_SCHEMA

    def load_from_file(self, filename: str) -> Dict[str, SourceSchema]:
        """Load schemas from JSON file"""
        return self.json_loader.load_schema_file(filename)

    def load_for_environment(self, environment: str) -> Dict[str, SourceSchema]:
        """Load schemas for specific environment"""
        env_files = {
            "development": "development_sources.json",
            "dev": "development_sources.json",
            "testing": "testing_sources.json",
            "test": "testing_sources.json",
            "staging": "staging_sources.json",
            "stage": "staging_sources.json",
            "production": "vietnamese_sources.json",
            "prod": "vietnamese_sources.json"
        }

        filename = env_files.get(environment.lower(), "vietnamese_sources.json")

        try:
            return self.load_from_file(filename)
        except FileNotFoundError:
            logger.warning(f"Environment-specific schema {filename} not found, using production")
            return self.load_from_file("vietnamese_sources.json")

    def save_to_file(self, filename: str, environment: str = "production") -> bool:
        """Save current schemas to JSON file"""
        return self.json_loader.save_schemas_to_json(self.schemas, filename, environment)

    def reload_schemas(self, schema_file: str = None) -> bool:
        """Hot-reload schemas from file"""
        try:
            if schema_file:
                self.schemas = self.load_from_file(schema_file)
            else:
                # Reload from the same file type (try to detect environment)
                available_files = self.json_loader.list_available_schemas()
                if "vietnamese_sources.json" in available_files:
                    self.schemas = self.load_from_file("vietnamese_sources.json")
                elif available_files:
                    self.schemas = self.load_from_file(available_files[0])
                else:
                    logger.error("No schema files available for reload")
                    return False

            logger.info("Schemas reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload schemas: {e}")
            return False

    def list_available_schema_files(self) -> List[str]:
        """List available JSON schema files"""
        return self.json_loader.list_available_schemas()

    def get_schema_metadata(self) -> Dict[str, Any]:
        """Get metadata about current schema set"""
        return {
            "total_sources": len(self.schemas),
            "source_types": list(set(s.source_type.value for s in self.schemas.values())),
            "reliability_levels": list(set(s.reliability.value for s in self.schemas.values())),
            "supported_exchanges": list(set(
                exchange for s in self.schemas.values()
                for exchange in s.supported_exchanges
            )),
            "data_categories": list(set(
                cat.value for s in self.schemas.values()
                for cat in s.capabilities.keys()
            )),
            "source_ids": list(self.schemas.keys())
        }

    def get_source(self, source_id: str) -> Optional[SourceSchema]:
        """Get source schema by ID"""
        return self.schemas.get(source_id)

    def get_sources_by_capability(self, capability: DataCategory) -> List[SourceSchema]:
        """Get all sources that support a specific data capability"""
        return [
            schema for schema in self.schemas.values()
            if capability in schema.capabilities and schema.capabilities[capability].supported
        ]

    def get_sources_by_reliability(self, min_reliability: ReliabilityLevel) -> List[SourceSchema]:
        """Get sources with minimum reliability level"""
        reliability_order = {
            ReliabilityLevel.VERY_HIGH: 4,
            ReliabilityLevel.HIGH: 3,
            ReliabilityLevel.MEDIUM: 2,
            ReliabilityLevel.LOW: 1
        }
        min_score = reliability_order[min_reliability]

        return [
            schema for schema in self.schemas.values()
            if reliability_order[schema.reliability] >= min_score
        ]

    def get_best_source_for_data(self, data_category: DataCategory) -> Optional[SourceSchema]:
        """Get the best source for a specific data category"""
        sources = self.get_sources_by_capability(data_category)

        if not sources:
            return None

        # Score sources based on quality, reliability, and performance
        def score_source(schema: SourceSchema) -> float:
            capability = schema.capabilities[data_category]
            reliability_score = {
                ReliabilityLevel.VERY_HIGH: 1.0,
                ReliabilityLevel.HIGH: 0.8,
                ReliabilityLevel.MEDIUM: 0.6,
                ReliabilityLevel.LOW: 0.4
            }[schema.reliability]

            return (
                capability.data_quality_score * 0.4
                + reliability_score * 0.3
                + schema.success_rate * 0.2
                + (1.0 - min(schema.average_response_time_ms / 5000, 1.0)) * 0.1
            )

        return max(sources, key=score_source)

    def get_source_priorities(self, data_category: DataCategory) -> List[SourceSchema]:
        """Get sources ordered by priority for a data category"""
        sources = self.get_sources_by_capability(data_category)

        def score_source(schema: SourceSchema) -> float:
            capability = schema.capabilities[data_category]
            reliability_score = {
                ReliabilityLevel.VERY_HIGH: 1.0,
                ReliabilityLevel.HIGH: 0.8,
                ReliabilityLevel.MEDIUM: 0.6,
                ReliabilityLevel.LOW: 0.4
            }[schema.reliability]

            return (
                capability.data_quality_score * 0.4
                + reliability_score * 0.3
                + schema.success_rate * 0.2
                + (1.0 - min(schema.average_response_time_ms / 5000, 1.0)) * 0.1
            )

        return sorted(sources, key=score_source, reverse=True)

    def validate_source_schema(self, schema: SourceSchema) -> List[str]:
        """Validate a source schema and return list of errors"""
        errors = []

        # Basic validation
        if not schema.source_id:
            errors.append("source_id is required")

        if not schema.name:
            errors.append("name is required")

        if not schema.base_url:
            errors.append("base_url is required")

        # Rate limit validation
        if schema.rate_limits.requests_per_second <= 0:
            errors.append("requests_per_second must be positive")

        # Capability validation
        for category, capability in schema.capabilities.items():
            if capability.data_quality_score < 0 or capability.data_quality_score > 1:
                errors.append(f"{category.value}: data_quality_score must be between 0 and 1")

            if capability.update_frequency_seconds <= 0:
                errors.append(f"{category.value}: update_frequency_seconds must be positive")

        return errors

    def export_schema_json(self, source_id: str = None) -> str:
        """Export schema(s) to JSON format"""
        if source_id:
            schema = self.get_source(source_id)
            if not schema:
                raise ValueError(f"Source {source_id} not found")
            return json.dumps(schema, default=str, indent=2)
        else:
            return json.dumps(self.schemas, default=str, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about configured sources"""
        total_sources = len(self.schemas)
        by_type = {}
        by_reliability = {}
        by_capability = {}

        for schema in self.schemas.values():
            # Count by type
            type_name = schema.source_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Count by reliability
            rel_name = schema.reliability.value
            by_reliability[rel_name] = by_reliability.get(rel_name, 0) + 1

            # Count by capability
            for capability in schema.capabilities.keys():
                cap_name = capability.value
                by_capability[cap_name] = by_capability.get(cap_name, 0) + 1

        avg_response_time = sum(s.average_response_time_ms for s in self.schemas.values()) / total_sources
        avg_success_rate = sum(s.success_rate for s in self.schemas.values()) / total_sources

        return {
            "total_sources": total_sources,
            "by_type": by_type,
            "by_reliability": by_reliability,
            "by_capability": by_capability,
            "average_response_time_ms": avg_response_time,
            "average_success_rate": avg_success_rate,
            "free_sources": total_sources  # All Vietnamese sources are free
        }


# Convenience functions
def get_schema_manager(environment: str = None, schema_file: str = None) -> SourceSchemaManager:
    """
    Get schema manager with Vietnamese sources

    Args:
        environment: Environment to load ("development", "production", etc.)
        schema_file: Specific JSON schema file to load

    Returns:
        SourceSchemaManager instance
    """
    return SourceSchemaManager(environment=environment, schema_file=schema_file)


def get_real_time_sources(environment: str = None) -> List[SourceSchema]:
    """Get sources that support real-time data"""
    manager = get_schema_manager(environment=environment)
    return manager.get_sources_by_capability(DataCategory.REAL_TIME_PRICES)


def get_free_sources(environment: str = None) -> Dict[str, SourceSchema]:
    """Get all free data sources (all Vietnamese sources are free)"""
    manager = get_schema_manager(environment=environment)
    return manager.schemas


def create_development_schema_manager() -> SourceSchemaManager:
    """Create schema manager with development-specific settings"""
    return SourceSchemaManager(environment="development")


def create_production_schema_manager() -> SourceSchemaManager:
    """Create schema manager with production settings"""
    return SourceSchemaManager(environment="production")


# Export for use in other modules
__all__ = [
    "SourceSchema", "DataSourceType", "ReliabilityLevel", "DataCategory",
    "DataTypeCapability", "RateLimitConfig", "SourceSchemaManager",
    "JSONSchemaLoader", "VIETNAMESE_SOURCES_SCHEMA",
    "get_schema_manager", "get_real_time_sources", "get_free_sources",
    "create_development_schema_manager", "create_production_schema_manager"
]


if __name__ == "__main__":
    # Example usage and testing with JSON-based schema system
    print("📊 JSON-Based Vietnamese Stock Market Data Sources")
    print("=" * 60)

    try:
        # Try loading production schemas from JSON
        manager = get_schema_manager(environment="production")
        print("✅ Loaded production schemas from JSON")
    except Exception as e:
        print(f"⚠️  Failed to load JSON schemas: {e}")
        print("Using fallback schemas...")
        manager = SourceSchemaManager(schemas=VIETNAMESE_SOURCES_SCHEMA)

    # Schema metadata
    metadata = manager.get_schema_metadata()
    print("\n📈 Schema Metadata:")
    print(f"  • Total Sources: {metadata['total_sources']}")
    print(f"  • Source Types: {', '.join(metadata['source_types'])}")
    print(f"  • Reliability Levels: {', '.join(metadata['reliability_levels'])}")

    # List available JSON schema files
    available_files = manager.list_available_schema_files()
    print("\n📁 Available Schema Files:")
    for file in available_files:
        print(f"  • {file}")

    # Get statistics
    stats = manager.get_statistics()
    print("\n📊 Performance Statistics:")
    print(f"  • Average Response Time: {stats['average_response_time_ms']:.0f}ms")
    print(f"  • Average Success Rate: {stats['average_success_rate']:.2%}")

    print("\n🎯 Data Categories Supported:")
    for capability, count in stats['by_capability'].items():
        print(f"  • {capability.replace('_', ' ').title()}: {count} sources")

    print("\n⭐ Best Sources by Data Type:")
    for category in [DataCategory.REAL_TIME_PRICES, DataCategory.MARKET_INDICES,
                     DataCategory.HISTORICAL_PRICES, DataCategory.NEWS_SENTIMENT]:
        best = manager.get_best_source_for_data(category)
        if best:
            print(f"  • {category.value}: {best.name}")

    print("\n🔄 Real-time Sources:")
    try:
        rt_sources = get_real_time_sources()
        for source in rt_sources:
            if DataCategory.REAL_TIME_PRICES in source.capabilities:
                capability = source.capabilities[DataCategory.REAL_TIME_PRICES]
                print(f"  • {source.name}: {capability.update_frequency_seconds}s refresh")
    except Exception as e:
        print(f"  ⚠️  Error getting real-time sources: {e}")

    # Test environment switching
    print("\n🧪 Testing Environment Switching:")
    try:
        dev_manager = get_schema_manager(environment="development")
        print(f"  • Development: {len(dev_manager.schemas)} sources loaded")
    except Exception as e:
        print(f"  ⚠️  Development environment not available: {e}")

    print("\n✨ JSON Schema System Ready!")
    print("Use get_schema_manager(environment='dev') for development settings")
