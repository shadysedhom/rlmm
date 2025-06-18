#!/usr/bin/env python3
"""Logging module for market-making simulator.

Provides structured logging with different verbosity levels and output formats.
"""
from __future__ import annotations

import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging levels
TRACE = 5  # More detailed than DEBUG
logging.addLevelName(TRACE, "TRACE")

# Global logger instance
logger = logging.getLogger("market_maker")

# Add a level property to the module for easy access
def get_level():
    return logger.level

def set_level(level):
    logger.setLevel(level)

level = property(get_level, set_level)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    console: bool = True,
) -> None:
    """Configure logging with the specified parameters.
    
    Args:
        level: Log level (TRACE, DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (if None, logs to console only)
        json_format: Whether to output logs in JSON format
        console: Whether to output logs to console
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if level.upper() == "TRACE":
        numeric_level = TRACE
    
    # Reset existing handlers
    logger.handlers = []
    logger.setLevel(numeric_level)
    
    # Create formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        # Use a custom formatter that properly handles milliseconds
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        # Override formatTime to include milliseconds
        formatter.formatTime = lambda record, datefmt=None: datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class JsonFormatter(logging.Formatter):
    """Format log records as JSON objects."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Add extra attributes if present
        if hasattr(record, "data") and record.data:
            log_data.update(record.data)
            
        return json.dumps(log_data)


def trace(msg: str, **kwargs: Any) -> None:
    """Log at TRACE level with optional structured data."""
    _log(TRACE, msg, **kwargs)


def debug(msg: str, **kwargs: Any) -> None:
    """Log at DEBUG level with optional structured data."""
    _log(logging.DEBUG, msg, **kwargs)


def info(msg: str, **kwargs: Any) -> None:
    """Log at INFO level with optional structured data."""
    _log(logging.INFO, msg, **kwargs)


def warning(msg: str, **kwargs: Any) -> None:
    """Log at WARNING level with optional structured data."""
    _log(logging.WARNING, msg, **kwargs)


def error(msg: str, **kwargs: Any) -> None:
    """Log at ERROR level with optional structured data."""
    _log(logging.ERROR, msg, **kwargs)


def critical(msg: str, **kwargs: Any) -> None:
    """Log at CRITICAL level with optional structured data."""
    _log(logging.CRITICAL, msg, **kwargs)


def _log(level: int, msg: str, **kwargs: Any) -> None:
    """Internal helper to log with structured data."""
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None
    )
    
    # Add structured data
    if kwargs:
        record.data = kwargs
    
    for handler in logger.handlers:
        if record.levelno >= handler.level:
            handler.handle(record)


class OrderEvent:
    """Constants for order lifecycle events."""
    PLACED = "ORDER_PLACED"
    ACTIVATED = "ORDER_ACTIVATED"
    FILLED = "ORDER_FILLED"
    PARTIAL_FILL = "ORDER_PARTIAL_FILL"
    CANCELED = "ORDER_CANCELED"
    EXPIRED = "ORDER_EXPIRED"
    FORCED_LIQ = "ORDER_FORCED_LIQUIDATION"
    REJECTED = "ORDER_REJECTED"


class PnLEvent:
    """Constants for P&L attribution events."""
    SPREAD_CAPTURE = "PNL_SPREAD_CAPTURE"
    INVENTORY_MOVE = "PNL_INVENTORY_MOVE"
    FEE = "FEE"
    REBATE = "REBATE"
    ADVERSE_SELECTION = "ADVERSE_SELECTION"
    MARK_TO_MARKET = "MARK_TO_MARKET"
    FUNDING_RATE = "FUNDING_RATE" 