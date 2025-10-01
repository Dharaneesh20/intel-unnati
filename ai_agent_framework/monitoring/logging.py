"""
Enhanced logging system for the AI Agent Framework
"""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    component: str
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "component": self.component,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata,
        }


class FrameworkLogger:
    """
    Enhanced logger for the AI Agent Framework
    """
    
    def __init__(self, name: str = "ai_agent_framework"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_default_handler()
    
    def _setup_default_handler(self):
        """Setup default console handler"""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method"""
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            component=kwargs.get('component', 'framework'),
            agent_id=kwargs.get('agent_id'),
            task_id=kwargs.get('task_id'),
            workflow_id=kwargs.get('workflow_id'),
            metadata=kwargs.get('metadata')
        )
        
        # Log to standard logger
        log_method = getattr(self.logger, level.value.lower())
        log_method(message, extra=kwargs)
    
    def configure(
        self,
        level: str = "INFO",
        format_string: Optional[str] = None,
        log_file: Optional[str] = None
    ):
        """Configure the logger"""
        # Set level
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatter
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)


# Global framework logger instance
framework_logger = FrameworkLogger()


def get_logger(name: str = "ai_agent_framework") -> FrameworkLogger:
    """Get a logger instance"""
    return FrameworkLogger(name)


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
):
    """Configure global logging"""
    framework_logger.configure(level, format_string, log_file)
