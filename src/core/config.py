"""
Configuration management for the AI Agent Framework.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="agent_framework")
    username: str = Field(default="agent_user")
    password: str = Field(default="secure_password")
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "agent_framework"),
            username=os.getenv("POSTGRES_USER", "agent_user"),
            password=os.getenv("POSTGRES_PASSWORD", "secure_password")
        )


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    
    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD")
        )


class KafkaConfig(BaseModel):
    """Kafka configuration."""
    bootstrap_servers: str = Field(default="localhost:9092")
    group_id: str = Field(default="agent_framework")
    auto_offset_reset: str = Field(default="latest")
    
    @classmethod
    def from_env(cls) -> "KafkaConfig":
        """Create config from environment variables."""
        return cls(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            group_id=os.getenv("KAFKA_GROUP_ID", "agent_framework"),
            auto_offset_reset=os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")
        )


class OpenVINOConfig(BaseModel):
    """Intel OpenVINO configuration."""
    model_path: str = Field(default="/opt/models")
    device: str = Field(default="CPU")
    num_threads: int = Field(default=4)
    
    @classmethod
    def from_env(cls) -> "OpenVINOConfig":
        """Create config from environment variables."""
        return cls(
            model_path=os.getenv("OPENVINO_MODEL_PATH", "/opt/models"),
            device=os.getenv("OPENVINO_DEVICE", "CPU"),
            num_threads=int(os.getenv("OPENVINO_NUM_THREADS", "4"))
        )


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    metrics_enabled: bool = Field(default=True)
    tracing_enabled: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    prometheus_port: int = Field(default=9090)
    jaeger_endpoint: Optional[str] = Field(default=None)
    
    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Create config from environment variables."""
        return cls(
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            tracing_enabled=os.getenv("TRACING_ENABLED", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT")
        )


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration."""
    engine: str = Field(default="airflow")
    max_concurrent_workflows: int = Field(default=100)
    default_timeout: int = Field(default=300)
    
    
class ExecutorConfig(BaseModel):
    """Executor configuration."""
    default_executor: str = Field(default="thread_pool")
    max_workers: int = Field(default=10)
    max_retries: int = Field(default=3)
    backoff_factor: float = Field(default=2.0)


class MemoryConfig(BaseModel):
    """Memory configuration."""
    provider: str = Field(default="redis")
    ttl: int = Field(default=3600)
    max_memory_mb: int = Field(default=1024)


class FrameworkConfig(BaseModel):
    """Main framework configuration."""
    name: str = Field(default="intel-ai-agent-framework")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    openvino: OpenVINOConfig = Field(default_factory=OpenVINOConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "FrameworkConfig":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    @classmethod
    def from_env(cls) -> "FrameworkConfig":
        """Create configuration from environment variables."""
        return cls(
            name=os.getenv("AGENT_FRAMEWORK_NAME", "intel-ai-agent-framework"),
            version=os.getenv("AGENT_FRAMEWORK_VERSION", "1.0.0"),
            environment=os.getenv("AGENT_FRAMEWORK_ENV", "development"),
            debug=os.getenv("AGENT_FRAMEWORK_DEBUG", "false").lower() == "true",
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            kafka=KafkaConfig.from_env(),
            openvino=OpenVINOConfig.from_env(),
            observability=ObservabilityConfig.from_env()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Global configuration instance
_config: Optional[FrameworkConfig] = None


def get_config() -> FrameworkConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = FrameworkConfig.from_env()
    return _config


def set_config(config: FrameworkConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(config_path: str) -> FrameworkConfig:
    """Load configuration from file and set as global."""
    config = FrameworkConfig.from_file(config_path)
    set_config(config)
    return config
