"""
Advanced Data Quality and Validation Framework
==============================================

Comprehensive data quality management for Vietnamese stock market data
including validation, cleansing, anomaly detection, and quality scoring.

"""

import asyncio
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from data_ingestion.source_schema import DataCategory, SourceSchemaManager, get_schema_manager

from core_services.utils.logger_utils import logger


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


@dataclass
class ValidationIssue:
    """Represents a data validation issue"""
    dimension: QualityDimension
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    rule: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    source: str
    data_category: DataCategory
    symbol: Optional[str]
    timestamp: datetime
    overall_score: float  # 0.0 - 1.0
    dimension_scores: Dict[QualityDimension, float]
    issues: List[ValidationIssue]
    record_count: int
    passed_rules: int
    total_rules: int

    @property
    def pass_rate(self) -> float:
        """Calculate rule pass rate"""
        return self.passed_rules / max(self.total_rules, 1)

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get critical issues only"""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues"""
        return len(self.critical_issues) > 0


class BaseValidator(ABC):
    """Base class for all validators"""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Validate data and return issues"""


class StockPriceValidator(BaseValidator):
    """Validator for stock price data"""

    def __init__(self):
        super().__init__("StockPriceValidator")

        # Vietnamese stock market specific ranges
        self.min_price = 100      # 100 VND minimum (penny stocks)
        self.max_price = 1000000  # 1M VND maximum (very expensive stocks)
        self.max_daily_change = 0.07  # 7% daily limit (normal stocks)
        self.max_penny_change = 0.30  # 30% for penny stocks
        self.min_volume = 0
        self.max_volume = 100000000  # 100M shares reasonable max

        # Valid Vietnamese stock symbols pattern
        self.symbol_pattern = re.compile(r"^[A-Z]{3,4}$")

    def _validate_required_fields(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate required fields presence"""
        issues = []
        required_fields = ["symbol", "price", "timestamp", "source"]
        for field in required_fields:
            if field not in data or data[field] is None:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Missing required field: {field}",
                    field=field,
                    rule="required_fields"
                ))
        return issues

    def _validate_symbol(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate symbol format"""
        issues = []
        symbol = str(data["symbol"]).upper()
        if not self.symbol_pattern.match(symbol):
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid symbol format: {symbol}",
                field="symbol",
                value=symbol,
                expected="3-4 uppercase letters",
                rule="symbol_format"
            ))
        return issues

    def _validate_price(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate price value and range"""
        issues = []
        try:
            price = float(data["price"])

            if price <= 0:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.VALIDITY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Price must be positive: {price}",
                    field="price",
                    value=price,
                    rule="positive_price"
                ))
            elif price < self.min_price:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.ACCURACY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Price below typical range: {price} VND",
                    field="price",
                    value=price,
                    expected=f">= {self.min_price}",
                    rule="price_range"
                ))
            elif price > self.max_price:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.ACCURACY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Price above reasonable range: {price} VND",
                    field="price",
                    value=price,
                    expected=f"<= {self.max_price}",
                    rule="price_range"
                ))

        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.CRITICAL,
                message=f"Invalid price format: {data.get('price')}",
                field="price",
                value=data.get("price"),
                rule="price_format"
            ))
        return issues

    def _validate_change_percent(self, data: Dict[str, Any], price: float) -> List[ValidationIssue]:
        """Validate change percentage"""
        issues = []
        if "change_percent" not in data:
            return issues

        try:
            change_pct = float(data["change_percent"])
            max_change = self.max_penny_change if price < 10000 else self.max_daily_change

            if abs(change_pct) > max_change * 100:  # Convert to percentage
                severity = ValidationSeverity.WARNING if abs(change_pct) < 20 else ValidationSeverity.ERROR
                issues.append(ValidationIssue(
                    dimension=QualityDimension.ACCURACY,
                    severity=severity,
                    message=f"Change percentage exceeds daily limit: {change_pct:.2f}%",
                    field="change_percent",
                    value=change_pct,
                    expected=f"<= {max_change * 100:.1f}%",
                    rule="daily_limit"
                ))
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.WARNING,
                message=f"Invalid change_percent format: {data.get('change_percent')}",
                field="change_percent",
                value=data.get("change_percent"),
                rule="change_format"
            ))
        return issues

    def _validate_volume(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate volume data"""
        issues = []
        if "volume" not in data:
            return issues

        try:
            volume = int(data["volume"])

            if volume < 0:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Volume cannot be negative: {volume}",
                    field="volume",
                    value=volume,
                    rule="positive_volume"
                ))
            elif volume > self.max_volume:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.ACCURACY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Volume unusually high: {volume:,}",
                    field="volume",
                    value=volume,
                    expected=f"<= {self.max_volume:,}",
                    rule="volume_range"
                ))
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.WARNING,
                message=f"Invalid volume format: {data.get('volume')}",
                field="volume",
                value=data.get("volume"),
                rule="volume_format"
            ))
        return issues

    def _validate_timestamp(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate timestamp format and recency"""
        issues = []
        try:
            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            now = datetime.now()

            # Check if timestamp is too far in the future
            if timestamp > now + timedelta(minutes=5):
                issues.append(ValidationIssue(
                    dimension=QualityDimension.TIMELINESS,
                    severity=ValidationSeverity.ERROR,
                    message=f"Timestamp too far in future: {timestamp}",
                    field="timestamp",
                    value=timestamp,
                    rule="future_timestamp"
                ))

            # Check if timestamp is too old
            if timestamp < now - timedelta(days=1):
                issues.append(ValidationIssue(
                    dimension=QualityDimension.TIMELINESS,
                    severity=ValidationSeverity.WARNING,
                    message=f"Timestamp older than 1 day: {timestamp}",
                    field="timestamp",
                    value=timestamp,
                    rule="stale_timestamp"
                ))

        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid timestamp format: {data.get('timestamp')}",
                field="timestamp",
                value=data.get("timestamp"),
                rule="timestamp_format"
            ))
        return issues

    def _validate_ohlc_consistency(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate OHLC data consistency"""
        issues = []
        if not all(field in data for field in ["open", "high", "low", "close"]):
            return issues

        try:
            o, h, l, c = float(data["open"]), float(data["high"]), float(data["low"]), float(data["close"])

            if not (l <= o <= h and l <= c <= h):
                issues.append(ValidationIssue(
                    dimension=QualityDimension.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"OHLC values inconsistent: O={o}, H={h}, L={l}, C={c}",
                    field="ohlc",
                    rule="ohlc_consistency"
                ))

            if h == l and h != 0:  # Suspicious if high == low (except for no trading)
                issues.append(ValidationIssue(
                    dimension=QualityDimension.ACCURACY,
                    severity=ValidationSeverity.WARNING,
                    message=f"High equals low (no price movement): {h}",
                    field="ohlc",
                    rule="price_movement"
                ))

        except (ValueError, TypeError) as e:
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid OHLC format: {e}",
                field="ohlc",
                rule="ohlc_format"
            ))
        return issues

    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Validate stock price data"""
        issues = []

        # Validate required fields first
        issues.extend(self._validate_required_fields(data))
        if issues:  # Don't continue if critical fields missing
            return issues

        # Run all validations
        issues.extend(self._validate_symbol(data))
        price_issues = self._validate_price(data)
        issues.extend(price_issues)

        # Get price value for change percent validation if price is valid
        price = None
        if not price_issues:
            try:
                price = float(data["price"])
            except (ValueError, TypeError):
                price = 0

        if price is not None:
            issues.extend(self._validate_change_percent(data, price))

        issues.extend(self._validate_volume(data))
        issues.extend(self._validate_timestamp(data))
        issues.extend(self._validate_ohlc_consistency(data))

        return issues


class IndexDataValidator(BaseValidator):
    """Validator for market index data"""

    def __init__(self):
        super().__init__("IndexDataValidator")

        # VN-Index specific ranges
        self.min_index = 200   # Historical low range
        self.max_index = 2000  # Reasonable high range
        self.max_daily_change = 0.10  # 10% daily change is extreme but possible

    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Validate index data"""
        issues = []

        # Required fields
        required_fields = ["name", "value", "timestamp"]
        for field in required_fields:
            if field not in data or data[field] is None:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Missing required field: {field}",
                    field=field,
                    rule="required_fields"
                ))

        if issues:
            return issues

        # Index value validation
        try:
            value = float(data["value"])

            if value <= 0:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.VALIDITY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Index value must be positive: {value}",
                    field="value",
                    value=value,
                    rule="positive_value"
                ))
            elif "VN-Index" in data["name"] and (value < self.min_index or value > self.max_index):
                severity = ValidationSeverity.WARNING if self.min_index * \
                    0.8 <= value <= self.max_index * 1.2 else ValidationSeverity.ERROR
                issues.append(ValidationIssue(
                    dimension=QualityDimension.ACCURACY,
                    severity=severity,
                    message=f"VN-Index value outside expected range: {value}",
                    field="value",
                    value=value,
                    expected=f"{self.min_index}-{self.max_index}",
                    rule="index_range"
                ))

        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.CRITICAL,
                message=f"Invalid index value format: {data.get('value')}",
                field="value",
                value=data.get("value"),
                rule="value_format"
            ))

        # Change validation
        if "change_percent" in data:
            try:
                change_pct = float(data["change_percent"])

                if abs(change_pct) > self.max_daily_change * 100:
                    issues.append(ValidationIssue(
                        dimension=QualityDimension.ACCURACY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Large index change: {change_pct:.2f}%",
                        field="change_percent",
                        value=change_pct,
                        rule="large_change"
                    ))
            except (ValueError, TypeError):
                pass  # Optional field

        return issues


class CrossSourceValidator(BaseValidator):
    """Validator for cross-source data consistency"""

    def __init__(self, tolerance: float = 0.05):
        super().__init__("CrossSourceValidator")
        self.tolerance = tolerance  # 5% tolerance by default

    async def validate_multiple_sources(self, data_list: List[Dict[str, Any]],
                                        context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Validate consistency across multiple data sources"""
        issues = []

        if len(data_list) < 2:
            return issues

        data_list[0].get("symbol", "unknown")

        # Price consistency check
        prices = []
        sources = []

        for data in data_list:
            if "price" in data and "source" in data:
                try:
                    price = float(data["price"])
                    prices.append(price)
                    sources.append(data["source"])
                except (ValueError, TypeError):
                    continue

        if len(prices) >= 2:
            price_array = np.array(prices)
            mean_price = np.mean(price_array)

            for i, (price, source) in enumerate(zip(prices, sources)):
                deviation = abs(price - mean_price) / mean_price

                if deviation > self.tolerance:
                    severity = (ValidationSeverity.WARNING if deviation < self.tolerance * 2
                                else ValidationSeverity.ERROR)
                    issues.append(ValidationIssue(
                        dimension=QualityDimension.CONSISTENCY,
                        severity=severity,
                        message=f"Price from {source} deviates {deviation:.2%} from cross-source average",
                        field="price",
                        value=price,
                        expected=f"~{mean_price:.2f}",
                        rule="cross_source_consistency"
                    ))

        # Timestamp consistency check
        timestamps = []
        for data in data_list:
            if "timestamp" in data:
                try:
                    ts = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    continue

        if len(timestamps) >= 2:
            # Check if timestamps are reasonably close (within 10 minutes)
            min_ts = min(timestamps)
            max_ts = max(timestamps)

            if (max_ts - min_ts).total_seconds() > 600:  # 10 minutes
                issues.append(ValidationIssue(
                    dimension=QualityDimension.TIMELINESS,
                    severity=ValidationSeverity.WARNING,
                    message=f"Large timestamp spread across sources: {(max_ts - min_ts).total_seconds():.0f} seconds",
                    field="timestamp",
                    rule="timestamp_spread"
                ))

        return issues

    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Single source validation (not applicable for cross-source)"""
        return []


class AnomalyDetector:
    """Detect anomalies in time series data"""

    def __init__(self, window_size: int = 20, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.historical_data: Dict[str, List[float]] = {}

    def update_history(self, symbol: str, value: float):
        """Update historical data for a symbol"""
        if symbol not in self.historical_data:
            self.historical_data[symbol] = []

        self.historical_data[symbol].append(value)

        # Keep only recent window
        if len(self.historical_data[symbol]) > self.window_size * 2:
            self.historical_data[symbol] = self.historical_data[symbol][-self.window_size:]

    def detect_price_anomaly(self, symbol: str, current_price: float) -> Optional[ValidationIssue]:
        """Detect if current price is anomalous"""
        if symbol not in self.historical_data or len(self.historical_data[symbol]) < 10:
            return None  # Need enough history

        history = self.historical_data[symbol]
        mean_price = statistics.mean(history)
        std_price = statistics.stdev(history) if len(history) > 1 else 0

        if std_price == 0:
            return None  # No variation

        z_score = abs(current_price - mean_price) / std_price

        if z_score > self.z_threshold:
            severity = ValidationSeverity.WARNING if z_score < self.z_threshold * 1.5 else ValidationSeverity.ERROR
            return ValidationIssue(
                dimension=QualityDimension.ACCURACY,
                severity=severity,
                message=f"Price anomaly detected: {current_price} (Z-score: {z_score:.2f})",
                field="price",
                value=current_price,
                expected=f"~{mean_price:.2f} ± {std_price * 2:.2f}",
                rule="price_anomaly"
            )

        return None

    def detect_volume_anomaly(self, symbol: str, current_volume: int) -> Optional[ValidationIssue]:
        """Detect volume anomalies"""
        volume_key = f"{symbol}_volume"

        if volume_key not in self.historical_data or len(self.historical_data[volume_key]) < 5:
            return None

        history = self.historical_data[volume_key]
        median_volume = statistics.median(history)

        # Volume spike detection (more than 5x median)
        if current_volume > median_volume * 5 and median_volume > 0:
            return ValidationIssue(
                dimension=QualityDimension.ACCURACY,
                severity=ValidationSeverity.WARNING,
                message=f"Volume spike detected: {current_volume:,} vs median {median_volume:,}",
                field="volume",
                value=current_volume,
                expected=f"~{median_volume:,}",
                rule="volume_spike"
            )

        return None


class DataQualityManager:
    """Main data quality management system"""

    def __init__(self, schema_manager: SourceSchemaManager = None):
        self.schema_manager = schema_manager or get_schema_manager()
        self.validators: Dict[DataCategory, List[BaseValidator]] = {}
        self.cross_source_validator = CrossSourceValidator()
        self.anomaly_detector = AnomalyDetector()

        # Initialize validators for each data category
        self._setup_validators()

    def _setup_validators(self):
        """Setup validators for different data categories"""
        # Stock price validators
        stock_validators = [StockPriceValidator()]
        self.validators[DataCategory.REAL_TIME_PRICES] = stock_validators
        self.validators[DataCategory.HISTORICAL_PRICES] = stock_validators

        # Index validators
        index_validators = [IndexDataValidator()]
        self.validators[DataCategory.MARKET_INDICES] = index_validators

        # Add more validators as needed for other categories

    async def validate_data(self, data: Dict[str, Any],
                            data_category: DataCategory,
                            source: str,
                            symbol: Optional[str] = None) -> QualityReport:
        """Comprehensive data validation"""

        start_time = datetime.now()
        all_issues = []

        # Run category-specific validators
        validators = self.validators.get(data_category, [])
        total_rules = 0
        passed_rules = 0

        for validator in validators:
            if not validator.enabled:
                continue

            try:
                issues = await validator.validate(data)
                all_issues.extend(issues)

                # Count rules (simplified - each validator is one rule set)
                total_rules += 1
                if not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues):
                    passed_rules += 1

            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {e}")
                all_issues.append(ValidationIssue(
                    dimension=QualityDimension.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validator error: {e}",
                    rule=validator.name
                ))

        # Anomaly detection for price data
        if data_category in [DataCategory.REAL_TIME_PRICES, DataCategory.HISTORICAL_PRICES] and symbol:
            if "price" in data:
                try:
                    price = float(data["price"])
                    anomaly = self.anomaly_detector.detect_price_anomaly(symbol, price)
                    if anomaly:
                        all_issues.append(anomaly)
                    else:
                        passed_rules += 1
                    total_rules += 1

                    # Update history for future anomaly detection
                    self.anomaly_detector.update_history(symbol, price)
                except (ValueError, TypeError):
                    pass

            if "volume" in data:
                try:
                    volume = int(data["volume"])
                    volume_anomaly = self.anomaly_detector.detect_volume_anomaly(symbol, volume)
                    if volume_anomaly:
                        all_issues.append(volume_anomaly)
                    else:
                        passed_rules += 1
                    total_rules += 1

                    # Update volume history
                    self.anomaly_detector.update_history(f"{symbol}_volume", float(volume))
                except (ValueError, TypeError):
                    pass

        # Calculate quality scores
        dimension_scores = self._calculate_dimension_scores(all_issues)
        overall_score = self._calculate_overall_score(all_issues, passed_rules, total_rules)

        return QualityReport(
            source=source,
            data_category=data_category,
            symbol=symbol,
            timestamp=start_time,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=all_issues,
            record_count=1,  # Single record validation
            passed_rules=passed_rules,
            total_rules=total_rules
        )

    async def validate_cross_source_data(self, data_list: List[Dict[str, Any]],
                                         data_category: DataCategory) -> List[ValidationIssue]:
        """Validate data consistency across multiple sources"""
        return await self.cross_source_validator.validate_multiple_sources(data_list)

    def _calculate_dimension_scores(self, issues: List[ValidationIssue]) -> Dict[QualityDimension, float]:
        """Calculate quality scores by dimension"""
        dimension_scores = {}

        # Group issues by dimension
        issues_by_dimension = {}
        for issue in issues:
            if issue.dimension not in issues_by_dimension:
                issues_by_dimension[issue.dimension] = []
            issues_by_dimension[issue.dimension].append(issue)

        # Calculate score for each dimension
        for dimension in QualityDimension:
            dimension_issues = issues_by_dimension.get(dimension, [])

            if not dimension_issues:
                dimension_scores[dimension] = 1.0
            else:
                # Weight by severity
                penalty = 0.0
                for issue in dimension_issues:
                    if issue.severity == ValidationSeverity.CRITICAL:
                        penalty += 0.5
                    elif issue.severity == ValidationSeverity.ERROR:
                        penalty += 0.3
                    elif issue.severity == ValidationSeverity.WARNING:
                        penalty += 0.1
                    else:  # INFO
                        penalty += 0.05

                dimension_scores[dimension] = max(0.0, 1.0 - penalty)

        return dimension_scores

    def _calculate_overall_score(self, issues: List[ValidationIssue],
                                 passed_rules: int, total_rules: int) -> float:
        """Calculate overall quality score"""
        if total_rules == 0:
            return 1.0

        # Base score from rule pass rate
        rule_score = passed_rules / total_rules

        # Penalty for critical and error issues
        critical_penalty = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]) * 0.3
        error_penalty = len([i for i in issues if i.severity == ValidationSeverity.ERROR]) * 0.15
        warning_penalty = len([i for i in issues if i.severity == ValidationSeverity.WARNING]) * 0.05

        total_penalty = critical_penalty + error_penalty + warning_penalty

        final_score = max(0.0, rule_score - total_penalty)
        return min(1.0, final_score)

    def get_quality_summary(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Generate quality summary from multiple reports"""
        if not reports:
            return {}

        total_score = sum(r.overall_score for r in reports) / len(reports)
        total_issues = sum(len(r.issues) for r in reports)
        critical_issues = sum(len(r.critical_issues) for r in reports)

        # Dimension averages
        avg_dimension_scores = {}
        for dimension in QualityDimension:
            scores = [r.dimension_scores.get(dimension, 1.0) for r in reports]
            avg_dimension_scores[dimension] = sum(scores) / len(scores)

        # Source quality breakdown
        source_quality = {}
        for report in reports:
            if report.source not in source_quality:
                source_quality[report.source] = []
            source_quality[report.source].append(report.overall_score)

        # Average by source
        for source in source_quality:
            scores = source_quality[source]
            source_quality[source] = {
                "average_score": sum(scores) / len(scores),
                "report_count": len(scores),
                "min_score": min(scores),
                "max_score": max(scores)
            }

        return {
            "overall_quality_score": total_score,
            "total_reports": len(reports),
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "dimension_scores": avg_dimension_scores,
            "source_quality": source_quality,
            "timestamp": datetime.now().isoformat()
        }


# Usage example
async def test_data_quality():
    """Test data quality framework"""
    print("🔍 Testing Data Quality Framework")
    print("=" * 50)

    quality_manager = DataQualityManager()

    # Test valid stock data
    valid_data = {
        "symbol": "VNM",
        "price": 75000,
        "open": 74500,
        "high": 75500,
        "low": 74000,
        "close": 75000,
        "volume": 1250000,
        "change": 500,
        "change_percent": 0.67,
        "timestamp": datetime.now().isoformat(),
        "source": "vnstock"
    }

    report = await quality_manager.validate_data(
        valid_data,
        DataCategory.REAL_TIME_PRICES,
        "vnstock",
        "VNM"
    )

    print(f"Valid data quality score: {report.overall_score:.2f}")
    print(f"Issues found: {len(report.issues)}")

    # Test invalid stock data
    invalid_data = {
        "symbol": "INVALID123",
        "price": -1000,  # Invalid negative price
        "change_percent": 50,  # Exceeds daily limit
        "volume": -500,  # Invalid negative volume
        "timestamp": "invalid_timestamp",
        "source": "test"
    }

    report2 = await quality_manager.validate_data(
        invalid_data,
        DataCategory.REAL_TIME_PRICES,
        "test",
        "INVALID123"
    )

    print(f"\nInvalid data quality score: {report2.overall_score:.2f}")
    print(f"Issues found: {len(report2.issues)}")
    for issue in report2.issues:
        print(f"  - {issue.severity.value.upper()}: {issue.message}")


if __name__ == "__main__":
    asyncio.run(test_data_quality())
