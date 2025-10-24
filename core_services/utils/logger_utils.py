"""
Centralized Logging Utility for core_services
==============================================

Provides consistent logging configuration across all modules.

Features:
- Singleton logger - just import and use!
- Colored console output
- File logging with daily rotation
- Optional structured logging (JSON)
- Logger suppression for noisy libraries
- Performance and error tracking utilities
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{self.BOLD}"
                f"{record.levelname}{self.RESET}"
            )

        # Add color to logger name
        record.name = f"\033[34m{record.name}{self.RESET}"

        return super().format(record)


class LoggerManager:
    """Manages logger configuration and creation"""

    _instance: Optional["LoggerManager"] = None
    _configured: bool = False

    def __new__(cls) -> "LoggerManager":
        """Singleton pattern to ensure single configuration"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger manager"""
        if not self._configured:
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)
            self.configure_logging()
            LoggerManager._configured = True

    def configure_logging(
        self,
        level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        structured: bool = False,
    ) -> None:
        """
        Configure global logging settings

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Enable file logging
            log_to_console: Enable console logging
            structured: Use structured logging (JSON format)
        """
        # Configure structlog if needed
        if structured:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        root_logger.handlers.clear()

        # Console handler with colors
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))

            if structured:
                console_format = logging.Formatter(
                    "%(message)s"  # structlog handles formatting
                )
            else:
                console_format = ColoredFormatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )

            console_handler.setFormatter(console_format)
            root_logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"data_ingestion_{timestamp}.log"

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file

            if structured:
                file_format = logging.Formatter("%(message)s")
            else:
                file_format = logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | "
                    "%(filename)s:%(lineno)d | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )

            file_handler.setFormatter(file_format)
            root_logger.addHandler(file_handler)

    def get_logger(
        self, name: str, level: Optional[str] = None
    ) -> logging.Logger:
        """
        Get a logger instance

        Args:
            name: Logger name (usually __name__)
            level: Optional log level override for this logger

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        if level:
            logger.setLevel(getattr(logging, level.upper()))

        return logger

    def set_level(self, logger_name: str, level: str) -> None:
        """
        Set log level for specific logger

        Args:
            logger_name: Name of the logger
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))

    def suppress_logger(self, logger_name: str, level: str = "WARNING") -> None:
        """
        Suppress a noisy logger

        Args:
            logger_name: Name of the logger to suppress
            level: Minimum level to show (default: WARNING)
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))


# Global logger manager instance
_manager = LoggerManager()


def get_logger(
    name: str, level: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger instance (convenience function)

    Args:
        name: Logger name (usually __name__)
        level: Optional log level override

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("Failed", exc_info=True)
    """
    return _manager.get_logger(name, level)


def configure_logging(**kwargs: Any) -> None:
    """
    Configure global logging settings (convenience function)

    Args:
        **kwargs: Configuration options passed to LoggerManager.configure_logging()

    Example:
        >>> configure_logging(level="DEBUG", structured=True)
    """
    _manager.configure_logging(**kwargs)


def set_level(logger_name: str, level: str) -> None:
    """
    Set log level for specific logger (convenience function)

    Args:
        logger_name: Name of the logger
        level: Log level

    Example:
        >>> set_level("vnstock", "WARNING")
    """
    _manager.set_level(logger_name, level)


def suppress_logger(logger_name: str, level: str = "WARNING") -> None:
    """
    Suppress a noisy logger (convenience function)

    Args:
        logger_name: Name of the logger to suppress
        level: Minimum level to show

    Example:
        >>> suppress_logger("vnstock.common.data.data_explorer")
    """
    _manager.suppress_logger(logger_name, level)


class LogContext:
    """Context manager for temporary log level changes"""

    def __init__(self, logger_name: str, level: str):
        """
        Initialize log context

        Args:
            logger_name: Name of the logger
            level: Temporary log level
        """
        self.logger = logging.getLogger(logger_name)
        self.level = getattr(logging, level.upper())
        self.original_level: Optional[int] = None

    def __enter__(self) -> logging.Logger:
        """Enter context - set temporary level"""
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - restore original level"""
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)


def with_log_context(logger_name: str, level: str) -> LogContext:
    """
    Create a log context for temporary level changes

    Args:
        logger_name: Name of the logger
        level: Temporary log level

    Returns:
        LogContext instance

    Example:
        >>> with with_log_context("vnstock", "ERROR"):
        ...     # vnstock logs only ERROR and above
        ...     fetch_data()
    """
    return LogContext(logger_name, level)


# Convenience logging functions with context
def log_function_call(logger: logging.Logger, func_name: str, **kwargs: Any) -> None:
    """
    Log function call with parameters

    Args:
        logger: Logger instance
        func_name: Function name
        **kwargs: Function parameters to log
    """
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params})")


def log_performance(
    logger: logging.Logger, operation: str, duration: float, **context: Any
) -> None:
    """
    Log performance metrics

    Args:
        logger: Logger instance
        operation: Operation name
        duration: Duration in seconds
        **context: Additional context
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.info(
        f"Performance: {operation} completed in {duration:.2f}s"
        + (f" ({context_str})" if context else "")
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log error with context and stack trace

    Args:
        logger: Logger instance
        error: Exception instance
        context: Additional context information
    """
    context_str = ""
    if context:
        context_str = " | " + ", ".join(f"{k}={v}" for k, v in context.items())

    logger.error(
        f"{error.__class__.__name__}: {str(error)}{context_str}",
        exc_info=True,
    )


# Initialize default configuration and create singleton logger
configure_logging()

# SINGLETON LOGGER - Just import and use!
logger = logging.getLogger("data_ingestion")

# Export the singleton logger as the main interface
__all__ = ['logger', 'suppress_logger', 'configure_logging', 'set_level']
