"""
Metrics collection and monitoring for the AI Agent Framework
"""

from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
from collections import defaultdict, deque
import json
import threading


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricUnit(Enum):
    """Metric units"""
    NONE = ""
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    BYTES = "bytes"
    KILOBYTES = "kilobytes"
    MEGABYTES = "megabytes"
    GIGABYTES = "gigabytes"
    REQUESTS = "requests"
    ERRORS = "errors"
    PERCENT = "percent"


@dataclass
class MetricPoint:
    """A single metric data point"""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
        }


@dataclass
class Metric:
    """Metric definition and data"""
    name: str
    metric_type: MetricType
    description: str
    unit: MetricUnit = MetricUnit.NONE
    labels: Dict[str, str] = field(default_factory=dict)
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_point(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Add a data point to the metric"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels={**self.labels, **(labels or {})}
        )
        self.data_points.append(point)
    
    def get_latest_value(self) -> Optional[Union[int, float]]:
        """Get the latest metric value"""
        if self.data_points:
            return self.data_points[-1].value
        return None
    
    def get_values_in_range(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[MetricPoint]:
        """Get metric values within a time range"""
        return [
            point for point in self.data_points
            if start_time <= point.timestamp <= end_time
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit.value,
            "labels": self.labels,
            "created_at": self.created_at.isoformat(),
            "data_points": [point.to_dict() for point in self.data_points],
            "latest_value": self.get_latest_value(),
        }


class MetricsCollector:
    """
    Central metrics collection system for monitoring agent performance,
    system health, and business metrics.
    """
    
    def __init__(self, export_interval: int = 60):
        self.metrics: Dict[str, Metric] = {}
        self.export_interval = export_interval
        self.exporters: List[Callable[[Dict[str, Metric]], None]] = []
        
        # Built-in system metrics
        self._init_system_metrics()
        
        # Export task
        self._export_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Thread safety
        self._lock = threading.Lock()
    
    def _init_system_metrics(self):
        """Initialize system metrics"""
        self.register_metric(
            "agent_executions_total",
            MetricType.COUNTER,
            "Total number of agent executions",
            MetricUnit.REQUESTS
        )
        
        self.register_metric(
            "agent_execution_duration",
            MetricType.HISTOGRAM,
            "Agent execution duration",
            MetricUnit.SECONDS
        )
        
        self.register_metric(
            "agent_failures_total",
            MetricType.COUNTER,
            "Total number of agent failures",
            MetricUnit.ERRORS
        )
        
        self.register_metric(
            "task_executions_total",
            MetricType.COUNTER,
            "Total number of task executions",
            MetricUnit.REQUESTS
        )
        
        self.register_metric(
            "task_execution_duration",
            MetricType.HISTOGRAM,
            "Task execution duration",
            MetricUnit.SECONDS
        )
        
        self.register_metric(
            "workflow_executions_total",
            MetricType.COUNTER,
            "Total number of workflow executions",
            MetricUnit.REQUESTS
        )
        
        self.register_metric(
            "workflow_execution_duration",
            MetricType.HISTOGRAM,
            "Workflow execution duration",
            MetricUnit.SECONDS
        )
        
        self.register_metric(
            "memory_usage",
            MetricType.GAUGE,
            "Memory usage",
            MetricUnit.MEGABYTES
        )
        
        self.register_metric(
            "cpu_usage",
            MetricType.GAUGE,
            "CPU usage percentage",
            MetricUnit.PERCENT
        )
        
        self.register_metric(
            "active_agents",
            MetricType.GAUGE,
            "Number of active agents",
            MetricUnit.NONE
        )
        
        self.register_metric(
            "queued_tasks",
            MetricType.GAUGE,
            "Number of queued tasks",
            MetricUnit.NONE
        )
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: MetricUnit = MetricUnit.NONE,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Register a new metric"""
        with self._lock:
            if name in self.metrics:
                return self.metrics[name]
            
            metric = Metric(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                labels=labels or {}
            )
            
            self.metrics[name] = metric
            return metric
    
    def increment_counter(
        self, 
        name: str, 
        value: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric"""
        with self._lock:
            if name not in self.metrics:
                raise ValueError(f"Metric {name} not found")
            
            metric = self.metrics[name]
            if metric.metric_type != MetricType.COUNTER:
                raise ValueError(f"Metric {name} is not a counter")
            
            current_value = metric.get_latest_value() or 0
            metric.add_point(current_value + value, labels)
    
    def set_gauge(
        self, 
        name: str, 
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric value"""
        with self._lock:
            if name not in self.metrics:
                raise ValueError(f"Metric {name} not found")
            
            metric = self.metrics[name]
            if metric.metric_type != MetricType.GAUGE:
                raise ValueError(f"Metric {name} is not a gauge")
            
            metric.add_point(value, labels)
    
    def observe_histogram(
        self, 
        name: str, 
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ):
        """Observe a value in a histogram metric"""
        with self._lock:
            if name not in self.metrics:
                raise ValueError(f"Metric {name} not found")
            
            metric = self.metrics[name]
            if metric.metric_type != MetricType.HISTOGRAM:
                raise ValueError(f"Metric {name} is not a histogram")
            
            metric.add_point(value, labels)
    
    def time_operation(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return TimedOperation(self, metric_name, labels)
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name"""
        with self._lock:
            return self.metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all metric names"""
        with self._lock:
            return list(self.metrics.keys())
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all metrics"""
        with self._lock:
            return self.metrics.copy()
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics as dictionary"""
        with self._lock:
            return {
                name: metric.to_dict() 
                for name, metric in self.metrics.items()
            }
    
    def add_exporter(self, exporter: Callable[[Dict[str, Metric]], None]):
        """Add a metrics exporter"""
        self.exporters.append(exporter)
    
    async def start_export_loop(self):
        """Start the metrics export loop"""
        if self._is_running:
            return
        
        self._is_running = True
        self._export_task = asyncio.create_task(self._export_loop())
    
    async def stop_export_loop(self):
        """Stop the metrics export loop"""
        self._is_running = False
        
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
    
    async def _export_loop(self):
        """Background export loop"""
        while self._is_running:
            try:
                # Export to all registered exporters
                for exporter in self.exporters:
                    try:
                        exporter(self.get_all_metrics())
                    except Exception as e:
                        # Log error but continue
                        print(f"Metrics export error: {e}")
                
                await asyncio.sleep(self.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Metrics export loop error: {e}")
                await asyncio.sleep(10)
    
    def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            self.set_gauge("memory_usage", memory_mb)
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
    
    def record_agent_execution(
        self, 
        agent_name: str, 
        duration: float, 
        success: bool,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record agent execution metrics"""
        execution_labels = {"agent_name": agent_name, **(labels or {})}
        
        self.increment_counter("agent_executions_total", 1, execution_labels)
        self.observe_histogram("agent_execution_duration", duration, execution_labels)
        
        if not success:
            self.increment_counter("agent_failures_total", 1, execution_labels)
    
    def record_task_execution(
        self, 
        task_name: str, 
        duration: float, 
        success: bool,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record task execution metrics"""
        execution_labels = {"task_name": task_name, **(labels or {})}
        
        self.increment_counter("task_executions_total", 1, execution_labels)
        self.observe_histogram("task_execution_duration", duration, execution_labels)
    
    def record_workflow_execution(
        self, 
        workflow_name: str, 
        duration: float, 
        success: bool,
        task_count: int,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record workflow execution metrics"""
        execution_labels = {
            "workflow_name": workflow_name, 
            "task_count": str(task_count),
            **(labels or {})
        }
        
        self.increment_counter("workflow_executions_total", 1, execution_labels)
        self.observe_histogram("workflow_execution_duration", duration, execution_labels)
    
    def update_active_agents(self, count: int):
        """Update active agents count"""
        self.set_gauge("active_agents", count)
    
    def update_queued_tasks(self, count: int):
        """Update queued tasks count"""
        self.set_gauge("queued_tasks", count)


class TimedOperation:
    """Context manager for timing operations"""
    
    def __init__(
        self, 
        collector: MetricsCollector, 
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.observe_histogram(self.metric_name, duration, self.labels)


# Prometheus exporter
class PrometheusExporter:
    """Export metrics to Prometheus format"""
    
    def __init__(self, port: int = 8000, endpoint: str = "/metrics"):
        self.port = port
        self.endpoint = endpoint
        self.app = None
    
    def export_metrics(self, metrics: Dict[str, Metric]) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for name, metric in metrics.items():
            # Add help and type comments
            lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {self._get_prometheus_type(metric.metric_type)}")
            
            # Add metric data points
            if metric.data_points:
                latest_point = metric.data_points[-1]
                
                if latest_point.labels:
                    labels_str = ",".join([
                        f'{key}="{value}"' 
                        for key, value in latest_point.labels.items()
                    ])
                    lines.append(f"{name}{{{labels_str}}} {latest_point.value}")
                else:
                    lines.append(f"{name} {latest_point.value}")
        
        return "\n".join(lines)
    
    def _get_prometheus_type(self, metric_type: MetricType) -> str:
        """Convert metric type to Prometheus type"""
        mapping = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge",
            MetricType.HISTOGRAM: "histogram",
            MetricType.SUMMARY: "summary",
        }
        return mapping.get(metric_type, "gauge")
    
    async def start_server(self, collector: MetricsCollector):
        """Start Prometheus metrics server"""
        try:
            from aiohttp import web
            
            async def metrics_handler(request):
                metrics = collector.get_all_metrics()
                prometheus_output = self.export_metrics(metrics)
                return web.Response(text=prometheus_output, content_type="text/plain")
            
            self.app = web.Application()
            self.app.router.add_get(self.endpoint, metrics_handler)
            
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, "0.0.0.0", self.port)
            await site.start()
            
            print(f"Prometheus metrics server started on port {self.port}")
            
        except ImportError:
            print("aiohttp not available, cannot start Prometheus server")


# JSON file exporter
class JSONFileExporter:
    """Export metrics to JSON file"""
    
    def __init__(self, file_path: str = "metrics.json"):
        self.file_path = file_path
    
    def __call__(self, metrics: Dict[str, Metric]):
        """Export metrics to JSON file"""
        try:
            metrics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {name: metric.to_dict() for name, metric in metrics.items()}
            }
            
            with open(self.file_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to export metrics to {self.file_path}: {e}")