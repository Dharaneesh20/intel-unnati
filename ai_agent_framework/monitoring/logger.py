"""
Structured logging for the AI Agent Framework
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import sys
import asyncio
from pathlib import Path
import traceback


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """A single log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    correlation_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    user_id: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary"""
        entry = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger": self.logger_name,
        }
        
        # Add optional fields if present
        if self.correlation_id:
            entry["correlation_id"] = self.correlation_id
        if self.agent_id:
            entry["agent_id"] = self.agent_id
        if self.task_id:
            entry["task_id"] = self.task_id
        if self.workflow_id:
            entry["workflow_id"] = self.workflow_id
        if self.user_id:
            entry["user_id"] = self.user_id
        if self.exception:
            entry["exception"] = self.exception
        
        # Add extra fields
        entry.update(self.extra_fields)
        
        return entry
    
    def to_json(self) -> str:
        """Convert log entry to JSON string"""
        return json.dumps(self.to_dict())


class LogHandler:
    """Base class for log handlers"""
    
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
    
    def should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged at this level"""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
        }
        return level_order[level] >= level_order[self.level]
    
    async def handle(self, entry: LogEntry):
        """Handle a log entry - to be implemented by subclasses"""
        raise NotImplementedError


class ConsoleHandler(LogHandler):
    """Console log handler"""
    
    def __init__(self, level: LogLevel = LogLevel.INFO, use_json: bool = False):
        super().__init__(level)
        self.use_json = use_json
    
    async def handle(self, entry: LogEntry):
        """Print log entry to console"""
        if not self.should_log(entry.level):
            return
        
        if self.use_json:
            print(entry.to_json())
        else:
            # Human-readable format
            timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            level = entry.level.value.ljust(8)
            logger = entry.logger_name.ljust(20)
            
            line_parts = [f"[{timestamp}]", f"[{level}]", f"[{logger}]", entry.message]
            
            # Add correlation ID if present
            if entry.correlation_id:
                line_parts.insert(-1, f"[{entry.correlation_id}]")
            
            print(" ".join(line_parts))
            
            # Print exception if present
            if entry.exception:
                print(f"Exception: {entry.exception}")


class FileHandler(LogHandler):
    """File log handler"""
    
    def __init__(
        self, 
        file_path: Union[str, Path], 
        level: LogLevel = LogLevel.INFO,
        use_json: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        super().__init__(level)
        self.file_path = Path(file_path)
        self.use_json = use_json
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File rotation support
        self._current_size = 0
        if self.file_path.exists():
            self._current_size = self.file_path.stat().st_size
    
    async def handle(self, entry: LogEntry):
        """Write log entry to file"""
        if not self.should_log(entry.level):
            return
        
        # Check if file rotation is needed
        await self._check_rotation()
        
        # Write log entry
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                if self.use_json:
                    f.write(entry.to_json() + "\n")
                else:
                    timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    line = f"[{timestamp}] [{entry.level.value}] [{entry.logger_name}] {entry.message}\n"
                    f.write(line)
                    
                    if entry.exception:
                        f.write(f"Exception: {entry.exception}\n")
                
                # Update current size
                self._current_size = f.tell()
                
        except Exception as e:
            # Fall back to console if file write fails
            print(f"Failed to write to log file {self.file_path}: {e}")
            console_handler = ConsoleHandler()
            await console_handler.handle(entry)
    
    async def _check_rotation(self):
        """Check if log file needs rotation"""
        if self._current_size >= self.max_file_size:
            await self._rotate_file()
    
    async def _rotate_file(self):
        """Rotate log files"""
        try:
            # Remove oldest backup
            oldest_backup = self.file_path.with_suffix(f".{self.backup_count}")
            if oldest_backup.exists():
                oldest_backup.unlink()
            
            # Rotate existing backups
            for i in range(self.backup_count - 1, 0, -1):
                old_backup = self.file_path.with_suffix(f".{i}")
                new_backup = self.file_path.with_suffix(f".{i + 1}")
                if old_backup.exists():
                    old_backup.rename(new_backup)
            
            # Move current file to .1
            if self.file_path.exists():
                backup_path = self.file_path.with_suffix(".1")
                self.file_path.rename(backup_path)
            
            # Reset current size
            self._current_size = 0
            
        except Exception as e:
            print(f"Failed to rotate log file: {e}")


class ElasticsearchHandler(LogHandler):
    """Elasticsearch log handler"""
    
    def __init__(
        self, 
        hosts: List[str], 
        index_name: str = "ai-agent-logs",
        level: LogLevel = LogLevel.INFO
    ):
        super().__init__(level)
        self.hosts = hosts
        self.index_name = index_name
        self._client = None
    
    async def _get_client(self):
        """Get Elasticsearch client"""
        if self._client is None:
            try:
                from elasticsearch import AsyncElasticsearch
                self._client = AsyncElasticsearch(hosts=self.hosts)
            except ImportError:
                raise ImportError("elasticsearch package required for ElasticsearchHandler")
        return self._client
    
    async def handle(self, entry: LogEntry):
        """Send log entry to Elasticsearch"""
        if not self.should_log(entry.level):
            return
        
        try:
            client = await self._get_client()
            
            # Create document
            doc = entry.to_dict()
            
            # Index document
            await client.index(
                index=self.index_name,
                body=doc
            )
            
        except Exception as e:
            # Fall back to console if Elasticsearch fails
            print(f"Failed to send log to Elasticsearch: {e}")
            console_handler = ConsoleHandler()
            await console_handler.handle(entry)


class FrameworkLogger:
    """
    Main logger for the AI Agent Framework with structured logging,
    correlation IDs, and multiple output formats.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.handlers: List[LogHandler] = []
        self.correlation_id: Optional[str] = None
        self.agent_id: Optional[str] = None
        self.task_id: Optional[str] = None
        self.workflow_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.extra_fields: Dict[str, Any] = {}
    
    def add_handler(self, handler: LogHandler):
        """Add a log handler"""
        self.handlers.append(handler)
    
    def remove_handler(self, handler: LogHandler):
        """Remove a log handler"""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def bind(self, **kwargs) -> "FrameworkLogger":
        """Create a new logger with bound context"""
        new_logger = FrameworkLogger(self.name)
        new_logger.handlers = self.handlers.copy()
        new_logger.correlation_id = kwargs.get("correlation_id", self.correlation_id)
        new_logger.agent_id = kwargs.get("agent_id", self.agent_id)
        new_logger.task_id = kwargs.get("task_id", self.task_id)
        new_logger.workflow_id = kwargs.get("workflow_id", self.workflow_id)
        new_logger.user_id = kwargs.get("user_id", self.user_id)
        new_logger.extra_fields = {**self.extra_fields, **kwargs}
        return new_logger
    
    async def _log(
        self, 
        level: LogLevel, 
        message: str, 
        exception: Optional[Exception] = None,
        **extra_fields
    ):
        """Internal logging method"""
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            logger_name=self.name,
            correlation_id=self.correlation_id,
            agent_id=self.agent_id,
            task_id=self.task_id,
            workflow_id=self.workflow_id,
            user_id=self.user_id,
            extra_fields={**self.extra_fields, **extra_fields},
            exception=traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ) if exception else None
        )
        
        # Send to all handlers
        for handler in self.handlers:
            try:
                await handler.handle(entry)
            except Exception as e:
                # Avoid infinite recursion - print to stderr
                print(f"Log handler error: {e}", file=sys.stderr)
    
    async def debug(self, message: str, **extra_fields):
        """Log debug message"""
        await self._log(LogLevel.DEBUG, message, **extra_fields)
    
    async def info(self, message: str, **extra_fields):
        """Log info message"""
        await self._log(LogLevel.INFO, message, **extra_fields)
    
    async def warning(self, message: str, **extra_fields):
        """Log warning message"""
        await self._log(LogLevel.WARNING, message, **extra_fields)
    
    async def error(self, message: str, exception: Optional[Exception] = None, **extra_fields):
        """Log error message"""
        await self._log(LogLevel.ERROR, message, exception, **extra_fields)
    
    async def critical(self, message: str, exception: Optional[Exception] = None, **extra_fields):
        """Log critical message"""
        await self._log(LogLevel.CRITICAL, message, exception, **extra_fields)
    
    # Synchronous versions for convenience
    def debug_sync(self, message: str, **extra_fields):
        """Synchronous debug logging"""
        asyncio.create_task(self.debug(message, **extra_fields))
    
    def info_sync(self, message: str, **extra_fields):
        """Synchronous info logging"""
        asyncio.create_task(self.info(message, **extra_fields))
    
    def warning_sync(self, message: str, **extra_fields):
        """Synchronous warning logging"""
        asyncio.create_task(self.warning(message, **extra_fields))
    
    def error_sync(self, message: str, exception: Optional[Exception] = None, **extra_fields):
        """Synchronous error logging"""
        asyncio.create_task(self.error(message, exception, **extra_fields))
    
    def critical_sync(self, message: str, exception: Optional[Exception] = None, **extra_fields):
        """Synchronous critical logging"""
        asyncio.create_task(self.critical(message, exception, **extra_fields))


class LoggerFactory:
    """Factory for creating framework loggers with default configuration"""
    
    _default_handlers: List[LogHandler] = []
    _loggers: Dict[str, FrameworkLogger] = {}
    
    @classmethod
    def configure_default_handlers(
        self,
        console_level: LogLevel = LogLevel.INFO,
        file_path: Optional[Union[str, Path]] = None,
        file_level: LogLevel = LogLevel.DEBUG,
        elasticsearch_hosts: Optional[List[str]] = None,
        elasticsearch_level: LogLevel = LogLevel.INFO
    ):
        """Configure default handlers for all loggers"""
        self._default_handlers.clear()
        
        # Console handler
        self._default_handlers.append(ConsoleHandler(level=console_level))
        
        # File handler
        if file_path:
            self._default_handlers.append(FileHandler(file_path, level=file_level))
        
        # Elasticsearch handler
        if elasticsearch_hosts:
            self._default_handlers.append(
                ElasticsearchHandler(elasticsearch_hosts, level=elasticsearch_level)
            )
        
        # Update existing loggers
        for logger in self._loggers.values():
            logger.handlers = self._default_handlers.copy()
    
    @classmethod
    def get_logger(cls, name: str) -> FrameworkLogger:
        """Get or create a logger with the given name"""
        if name not in cls._loggers:
            logger = FrameworkLogger(name)
            logger.handlers = cls._default_handlers.copy()
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def get_agent_logger(cls, agent_id: str, agent_name: str) -> FrameworkLogger:
        """Get a logger for a specific agent"""
        logger_name = f"agent.{agent_name}"
        logger = cls.get_logger(logger_name)
        return logger.bind(agent_id=agent_id)
    
    @classmethod
    def get_task_logger(cls, task_id: str, task_name: str) -> FrameworkLogger:
        """Get a logger for a specific task"""
        logger_name = f"task.{task_name}"
        logger = cls.get_logger(logger_name)
        return logger.bind(task_id=task_id)
    
    @classmethod
    def get_workflow_logger(cls, workflow_id: str, workflow_name: str) -> FrameworkLogger:
        """Get a logger for a specific workflow"""
        logger_name = f"workflow.{workflow_name}"
        logger = cls.get_logger(logger_name)
        return logger.bind(workflow_id=workflow_id)


# Default setup
def setup_default_logging(
    console_level: LogLevel = LogLevel.INFO,
    log_file: Optional[str] = "ai_agent_framework.log"
):
    """Setup default logging configuration"""
    LoggerFactory.configure_default_handlers(
        console_level=console_level,
        file_path=log_file,
        file_level=LogLevel.DEBUG
    )


# Convenience function
def get_logger(name: str) -> FrameworkLogger:
    """Get a logger instance"""
    return LoggerFactory.get_logger(name)