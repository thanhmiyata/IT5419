"""
Data Normalization and Standardization Module
==============================================

Normalizes and standardizes data from different sources before database insertion.
Handles field name mapping, timestamp standardization, numeric precision, and unit conversion.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Union

from core_services.utils.logger_utils import logger


@dataclass
class NormalizationConfig:
    """Configuration for data normalization"""
    price_decimal_places: int = 2
    volume_decimal_places: int = 0
    percentage_decimal_places: int = 2
    timestamp_format: str = "iso8601"  # iso8601, unix, datetime
    currency: str = "VND"
    volume_unit: str = "shares"  # shares, thousands, millions


class DataNormalizer:
    """
    Normalizes and standardizes data from multiple sources.

    Features:
    - Field name standardization (source-specific names → unified schema)
    - Timestamp normalization (all formats → ISO 8601)
    - Numeric precision control (configurable decimal places)
    - Unit conversion support (volumes, currencies)
    - Data type enforcement
    """

    # Standard field name mappings (unified_name: [source_variations])
    FIELD_MAPPINGS = {
        # Price fields
        "symbol": ["symbol", "ticker", "code", "stock_code"],
        "price": ["price", "close", "close_price", "last_price", "current_price"],
        "open": ["open", "open_price", "opening_price"],
        "high": ["high", "high_price", "highest_price", "day_high"],
        "low": ["low", "low_price", "lowest_price", "day_low"],
        "close": ["close", "close_price", "closing_price"],
        "volume": ["volume", "trading_volume", "vol", "quantity"],

        # Change fields
        "change": ["change", "price_change", "change_amount", "net_change"],
        "change_percent": ["change_percent", "change_pct", "percent_change", "pct_change", "%_change"],

        # Timestamp fields
        "timestamp": ["timestamp", "time", "datetime", "created_at", "updated_at"],
        "date": ["date", "trading_date", "trade_date", "day"],

        # Index fields
        "name": ["name", "index_name", "title"],
        "value": ["value", "index_value", "point", "points"],

        # Metadata
        "source": ["source", "data_source", "provider"],
        "market": ["market", "exchange", "market_name"],
    }

    # Financial metrics that should be validated as positive
    POSITIVE_METRICS = {
        "revenue", "total_assets", "current_assets", "shareholders_equity",
        "total_liabilities", "gross_profit", "ebitda", "market_cap"
    }

    # Financial metrics that can be negative
    NULLABLE_METRICS = {
        "net_profit", "operating_profit", "ebit", "net_cash_flow",
        "operating_cash_flow", "financing_cash_flow", "investing_cash_flow"
    }

    # Ratio metrics (typically 0-1 or percentages)
    RATIO_METRICS = {
        "roe", "roa", "roic", "current_ratio", "quick_ratio",
        "debt_to_equity", "gross_margin", "operating_margin", "net_margin"
    }

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()

    def normalize_stock_price(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize stock price data.

        Args:
            data: Raw stock price data from any source

        Returns:
            Normalized stock price data with standardized fields
        """
        normalized = {}

        # Map field names to standard schema
        for standard_field, variations in self.FIELD_MAPPINGS.items():
            value = self._extract_field(data, variations)
            if value is not None:
                normalized[standard_field] = value

        # Standardize specific field types
        if "symbol" in normalized:
            normalized["symbol"] = self._normalize_symbol(normalized["symbol"])

        # Numeric fields
        for field in ["price", "open", "high", "low", "close"]:
            if field in normalized:
                normalized[field] = self._normalize_price(normalized[field])

        if "volume" in normalized:
            normalized["volume"] = self._normalize_volume(normalized["volume"])

        for field in ["change"]:
            if field in normalized:
                normalized[field] = self._normalize_price(normalized[field])

        if "change_percent" in normalized:
            normalized["change_percent"] = self._normalize_percentage(normalized["change_percent"])

        # Timestamp fields
        if "timestamp" in normalized:
            normalized["timestamp"] = self._normalize_timestamp(normalized["timestamp"])

        if "date" in normalized:
            normalized["date"] = self._normalize_date(normalized["date"])

        # Add metadata if missing
        if "currency" not in normalized:
            normalized["currency"] = self.config.currency

        if "volume_unit" not in normalized:
            normalized["volume_unit"] = self.config.volume_unit

        # Add normalization metadata
        normalized["_normalized"] = True
        normalized["_normalized_at"] = datetime.now().isoformat()

        return normalized

    def normalize_historical_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize a list of historical price records.

        Args:
            records: List of historical price records

        Returns:
            List of normalized records
        """
        normalized_records = []

        for record in records:
            try:
                normalized = self.normalize_stock_price(record)
                normalized_records.append(normalized)
            except Exception as e:
                logger.warning(f"Failed to normalize record: {e}, record: {record}")
                # Skip invalid records
                continue

        logger.info(f"Normalized {len(normalized_records)}/{len(records)} historical records")
        return normalized_records

    def normalize_index_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize market index data.

        Args:
            data: Raw index data

        Returns:
            Normalized index data
        """
        normalized = {}

        # Map field names
        for standard_field, variations in self.FIELD_MAPPINGS.items():
            value = self._extract_field(data, variations)
            if value is not None:
                normalized[standard_field] = value

        # Normalize numeric fields
        if "value" in normalized:
            normalized["value"] = self._normalize_price(normalized["value"])

        if "change" in normalized:
            normalized["change"] = self._normalize_price(normalized["change"])

        if "change_percent" in normalized:
            normalized["change_percent"] = self._normalize_percentage(normalized["change_percent"])

        if "volume" in normalized:
            normalized["volume"] = self._normalize_volume(normalized["volume"])

        # Timestamp
        if "timestamp" in normalized:
            normalized["timestamp"] = self._normalize_timestamp(normalized["timestamp"])

        if "date" in normalized:
            normalized["date"] = self._normalize_date(normalized["date"])

        # Add metadata
        normalized["_normalized"] = True
        normalized["_normalized_at"] = datetime.now().isoformat()

        return normalized

    def normalize_financial_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize financial report (BCTC) data.

        Args:
            data: Raw financial report data from VNStock or other sources

        Returns:
            Normalized financial report data with validated metrics
        """
        normalized = {}

        # Core identifiers
        if "symbol" in data:
            normalized["symbol"] = self._normalize_symbol(data["symbol"])

        # Fiscal period
        for field in ["fiscal_year", "fiscal_quarter", "period_type", "report_date"]:
            if field in data:
                normalized[field] = data[field]

        # Normalize all financial metrics
        financial_fields = [
            # Income Statement
            "revenue", "cost_of_goods_sold", "gross_profit",
            "operating_expenses", "operating_profit",
            "ebit", "ebitda", "interest_expense",
            "profit_before_tax", "tax_expense", "net_profit",
            "net_profit_to_shareholders",

            # Balance Sheet
            "total_assets", "current_assets", "fixed_assets",
            "total_liabilities", "current_liabilities", "long_term_liabilities",
            "shareholders_equity", "cash_and_equivalents", "inventory",
            "receivables",

            # Cash Flow
            "operating_cash_flow", "investing_cash_flow",
            "financing_cash_flow", "net_cash_flow", "free_cash_flow",

            # Key Ratios
            "eps", "roe", "roa", "roic",
            "pe_ratio", "pb_ratio", "debt_to_equity",
            "current_ratio", "quick_ratio",
            "gross_margin", "operating_margin", "net_margin",
        ]

        for field in financial_fields:
            if field in data:
                value = data[field]
                if value is not None:
                    # Determine if this is a ratio/percentage or absolute value
                    if field in self.RATIO_METRICS:
                        normalized[field] = self._normalize_ratio(value, field)
                    else:
                        normalized[field] = self._normalize_financial_value(value, field)
                else:
                    normalized[field] = None

        # Metadata
        if "data_source" in data:
            normalized["data_source"] = data["data_source"]

        if "timestamp" in data:
            normalized["timestamp"] = self._normalize_timestamp(data["timestamp"])
        else:
            normalized["timestamp"] = datetime.now().isoformat()

        # Validate financial data
        if not self._validate_financial_metrics(normalized):
            logger.warning(
                f"Financial report for {normalized.get('symbol')} "
                f"{normalized.get('fiscal_year')}Q{normalized.get('fiscal_quarter')} "
                f"has suspicious or invalid metrics"
            )

        # Add normalization metadata
        normalized["_normalized"] = True
        normalized["_normalized_at"] = datetime.now().isoformat()
        normalized["_data_type"] = "financial_report"

        return normalized

    def _normalize_financial_value(self, value: Union[float, int, str], field_name: str) -> Optional[float]:
        """
        Normalize financial metric values (revenue, assets, etc.)

        Args:
            value: Raw financial value
            field_name: Name of the metric

        Returns:
            Normalized value or None if invalid
        """
        try:
            if value is None:
                return None

            # Convert to float
            if isinstance(value, str):
                value = value.replace(",", "").replace(" ", "").strip()
                # Remove currency symbols
                for symbol in ["$", "¥", "€", "£", "₫", "VND"]:
                    value = value.replace(symbol, "")

            float_value = float(value)

            # Validate based on metric type
            if field_name in self.POSITIVE_METRICS:
                if float_value < 0:
                    logger.warning(
                        f"Negative value for {field_name}: {float_value}. "
                        f"Expected positive value."
                    )
                    return None
            elif field_name in self.NULLABLE_METRICS:
                # These can be negative (e.g., losses, negative cash flow)
                pass

            # Round to 2 decimal places for storage
            return round(float_value, 2)

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to normalize financial value '{value}' for {field_name}: {e}")
            return None

    def _normalize_ratio(self, value: Union[float, int, str], field_name: str) -> Optional[float]:
        """
        Normalize ratio/percentage values (ROE, ROA, margins, etc.)

        Args:
            value: Raw ratio value
            field_name: Name of the ratio

        Returns:
            Normalized ratio value (0-1 or as percentage)
        """
        try:
            if value is None:
                return None

            # Convert to float
            if isinstance(value, str):
                value = value.replace("%", "").replace(" ", "").strip()

            float_value = float(value)

            # Some ratios might be in percentage form (0-100)
            # while others in decimal form (0-1)
            # Normalize to decimal form for consistency
            if abs(float_value) > 5:  # Likely a percentage
                float_value = float_value / 100

            # Validate reasonable ranges
            if field_name in ["roe", "roa", "roic"]:
                # ROE/ROA/ROIC can be negative but typically -1 to 1 (-100% to 100%)
                if abs(float_value) > 10:
                    logger.warning(
                        f"Suspicious {field_name} value: {float_value}. "
                        f"Expected range: -1 to 1"
                    )

            elif field_name in ["current_ratio", "quick_ratio"]:
                # Liquidity ratios should be positive
                if float_value < 0:
                    logger.warning(f"Negative {field_name}: {float_value}")
                    return None

            # Round to 4 decimal places for ratios
            return round(float_value, 4)

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to normalize ratio '{value}' for {field_name}: {e}")
            return None

    def _validate_financial_metrics(self, data: Dict[str, Any]) -> bool:
        """
        Validate financial metrics for logical consistency

        Args:
            data: Normalized financial data

        Returns:
            True if metrics pass basic validation
        """
        # Check revenue vs profit relationship
        revenue = data.get("revenue")
        net_profit = data.get("net_profit")

        if revenue and net_profit:
            if revenue > 0 and abs(net_profit) > revenue * 2:
                logger.warning(
                    f"Net profit ({net_profit}) is unusually large "
                    f"compared to revenue ({revenue})"
                )
                return False

        # Check assets vs liabilities + equity
        total_assets = data.get("total_assets")
        total_liabilities = data.get("total_liabilities")
        shareholders_equity = data.get("shareholders_equity")

        if all([total_assets, total_liabilities, shareholders_equity]):
            balance = abs(total_assets - (total_liabilities + shareholders_equity))
            tolerance = total_assets * 0.01  # 1% tolerance

            if balance > tolerance:
                logger.warning(
                    f"Balance sheet doesn't balance: "
                    f"Assets={total_assets}, "
                    f"Liabilities={total_liabilities}, "
                    f"Equity={shareholders_equity}, "
                    f"Difference={balance}"
                )
                # Don't fail, just warn (might be rounding issues)

        # Check current assets <= total assets
        current_assets = data.get("current_assets")
        if total_assets and current_assets:
            if current_assets > total_assets * 1.01:  # 1% tolerance
                logger.warning(
                    f"Current assets ({current_assets}) exceed "
                    f"total assets ({total_assets})"
                )
                return False

        return True

    def _extract_field(self, data: Dict[str, Any], field_variations: List[str]) -> Optional[Any]:
        """Extract field value trying multiple variations"""
        for variation in field_variations:
            if variation in data:
                return data[variation]
        return None

    def _normalize_symbol(self, symbol: Union[str, Any]) -> str:
        """Normalize stock symbol to uppercase"""
        return str(symbol).upper().strip()

    def _normalize_price(self, price: Union[float, int, str, Decimal]) -> float:
        """
        Normalize price to float with configured decimal places.

        Args:
            price: Raw price value

        Returns:
            Normalized price as float
        """
        try:
            # Convert to Decimal for precise rounding
            if isinstance(price, str):
                # Remove currency symbols and whitespace
                price = price.replace(",", "").replace(" ", "").strip()
                # Remove common currency symbols
                for symbol in ["$", "¥", "€", "£", "₫", "VND"]:
                    price = price.replace(symbol, "")

            decimal_price = Decimal(str(price))

            # Round to configured decimal places
            quantize_string = "0." + "0" * self.config.price_decimal_places
            rounded = decimal_price.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)

            return float(rounded)

        except (ValueError, TypeError, ArithmeticError) as e:
            logger.warning(f"Failed to normalize price '{price}': {e}")
            raise ValueError(f"Invalid price value: {price}")

    def _normalize_volume(self, volume: Union[int, float, str]) -> int:
        """
        Normalize volume to integer.

        Args:
            volume: Raw volume value

        Returns:
            Normalized volume as integer
        """
        try:
            if isinstance(volume, str):
                # Remove commas and whitespace
                volume = volume.replace(",", "").replace(" ", "").strip()

                # Handle K/M/B suffixes (thousands, millions, billions)
                multiplier = 1
                if volume.endswith("K") or volume.endswith("k"):
                    multiplier = 1000
                    volume = volume[:-1]
                elif volume.endswith("M") or volume.endswith("m"):
                    multiplier = 1000000
                    volume = volume[:-1]
                elif volume.endswith("B") or volume.endswith("b"):
                    multiplier = 1000000000
                    volume = volume[:-1]

                volume = float(volume) * multiplier

            # Convert to integer
            return int(float(volume))

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to normalize volume '{volume}': {e}")
            raise ValueError(f"Invalid volume value: {volume}")

    def _normalize_percentage(self, percentage: Union[float, int, str]) -> float:
        """
        Normalize percentage to float with configured decimal places.

        Args:
            percentage: Raw percentage value

        Returns:
            Normalized percentage as float
        """
        try:
            if isinstance(percentage, str):
                # Remove % symbol and whitespace
                percentage = percentage.replace("%", "").replace(" ", "").strip()

            decimal_pct = Decimal(str(percentage))

            # Round to configured decimal places
            quantize_string = "0." + "0" * self.config.percentage_decimal_places
            rounded = decimal_pct.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)

            return float(rounded)

        except (ValueError, TypeError, ArithmeticError) as e:
            logger.warning(f"Failed to normalize percentage '{percentage}': {e}")
            raise ValueError(f"Invalid percentage value: {percentage}")

    def _normalize_timestamp(self, timestamp: Union[str, datetime, int, float]) -> str:
        """
        Normalize timestamp to ISO 8601 format.

        Args:
            timestamp: Raw timestamp in various formats

        Returns:
            ISO 8601 formatted timestamp string
        """
        try:
            # Already a datetime object
            if isinstance(timestamp, datetime):
                return timestamp.isoformat()

            # Unix timestamp (int or float)
            if isinstance(timestamp, (int, float)):
                # Check if it's milliseconds (> year 2100 in seconds)
                if timestamp > 4102444800:
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp).isoformat()

            # String timestamp
            if isinstance(timestamp, str):
                # Already ISO format
                if "T" in timestamp or timestamp.count("-") >= 2:
                    # Try parsing ISO format
                    try:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        return dt.isoformat()
                    except ValueError:
                        pass

                # Try common formats
                formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y/%m/%d %H:%M:%S",
                    "%d-%m-%Y %H:%M:%S",
                    "%d/%m/%Y %H:%M:%S",
                ]

                for fmt in formats:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue

                # If nothing works, raise error
                raise ValueError(f"Unrecognized timestamp format: {timestamp}")

            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")

        except Exception as e:
            logger.warning(f"Failed to normalize timestamp '{timestamp}': {e}")
            # Fallback to current time
            return datetime.now().isoformat()

    def _normalize_date(self, date: Union[str, datetime]) -> str:
        """
        Normalize date to YYYY-MM-DD format.

        Args:
            date: Raw date value

        Returns:
            Normalized date string (YYYY-MM-DD)
        """
        try:
            # Already a datetime object
            if isinstance(date, datetime):
                return date.strftime("%Y-%m-%d")

            # String date
            if isinstance(date, str):
                # Already in YYYY-MM-DD format
                if len(date) == 10 and date.count("-") == 2:
                    # Validate it's a valid date
                    datetime.strptime(date, "%Y-%m-%d")
                    return date

                # Try common date formats
                formats = [
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                ]

                for fmt in formats:
                    try:
                        dt = datetime.strptime(date, fmt)
                        return dt.strftime("%Y-%m-%d")
                    except ValueError:
                        continue

                raise ValueError(f"Unrecognized date format: {date}")

            raise ValueError(f"Unsupported date type: {type(date)}")

        except Exception as e:
            logger.warning(f"Failed to normalize date '{date}': {e}")
            # Fallback to current date
            return datetime.now().strftime("%Y-%m-%d")

    def validate_normalized_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate that data has been properly normalized.

        Args:
            data: Normalized data to validate

        Returns:
            True if data is properly normalized
        """
        # Check normalization flag
        if not data.get("_normalized"):
            logger.warning("Data missing normalization flag")
            return False

        # Validate required fields based on data type
        if "symbol" in data:
            # Stock price data
            required_fields = ["symbol", "price", "timestamp"]
            if not all(field in data for field in required_fields):
                logger.warning("Missing required fields in normalized stock data")
                return False

            # Validate types
            if not isinstance(data["symbol"], str):
                return False
            if not isinstance(data["price"], (int, float)):
                return False

        elif "name" in data:
            # Index data
            required_fields = ["name", "value", "timestamp"]
            if not all(field in data for field in required_fields):
                logger.warning("Missing required fields in normalized index data")
                return False

        return True


# Convenience functions
def normalize_stock_price(data: Dict[str, Any], config: Optional[NormalizationConfig] = None) -> Dict[str, Any]:
    """Normalize stock price data using default or provided config"""
    normalizer = DataNormalizer(config)
    return normalizer.normalize_stock_price(data)


def normalize_historical_data(records: List[Dict[str, Any]],
                              config: Optional[NormalizationConfig] = None) -> List[Dict[str, Any]]:
    """Normalize historical price records using default or provided config"""
    normalizer = DataNormalizer(config)
    return normalizer.normalize_historical_data(records)


def normalize_index_data(data: Dict[str, Any], config: Optional[NormalizationConfig] = None) -> Dict[str, Any]:
    """Normalize index data using default or provided config"""
    normalizer = DataNormalizer(config)
    return normalizer.normalize_index_data(data)


def normalize_financial_report(data: Dict[str, Any], config: Optional[NormalizationConfig] = None) -> Dict[str, Any]:
    """Normalize financial report (BCTC) data using default or provided config"""
    normalizer = DataNormalizer(config)
    return normalizer.normalize_financial_report(data)


# Example usage and testing
if __name__ == "__main__":
    # Test stock price normalization
    raw_stock_data = {
        "ticker": "VNM",
        "close_price": "75,000.50",
        "opening_price": 74500.123,
        "day_high": 75800.999,
        "day_low": 74000,
        "trading_volume": "1,250,000",
        "percent_change": "0.67%",
        "datetime": "2025-01-15 14:30:00",
        "data_source": "vnstock"
    }

    normalizer = DataNormalizer()
    normalized = normalizer.normalize_stock_price(raw_stock_data)

    print("Original data:")
    print(raw_stock_data)
    print("\nNormalized data:")
    print(normalized)
    print("\nValidation:", normalizer.validate_normalized_data(normalized))
