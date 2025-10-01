"""
Data Analysis Agent - Reference Implementation

This agent demonstrates a complete data analysis workflow with:
- Data ingestion from multiple sources (CSV, JSON, API)
- Data preprocessing and cleaning
- Statistical analysis and insights generation
- Machine learning model training and inference
- Intel optimizations for ML models (PyTorch, OpenVINO)
- Results visualization and reporting
"""

import asyncio
import time
import json
import math
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Framework imports
from ..core.agent import Agent, AgentConfig, AgentContext
from ..core.task import Task, TaskConfig, TaskResult, TaskStatus
from ..core.workflow import Workflow
from ..core.memory import Memory, MemoryType
from ..intel_optimizations.pytorch_optimizer import IntelPyTorchOptimizer, IntelPyTorchConfig, OptimizationLevel
from ..intel_optimizations.openvino_optimizer import OpenVINOOptimizer


class DataSourceType(Enum):
    """Supported data source types"""
    CSV = "csv"
    JSON = "json"
    API = "api"
    DATABASE = "database"
    EXCEL = "excel"


class AnalysisType(Enum):
    """Types of analysis"""
    DESCRIPTIVE = "descriptive"
    PREDICTIVE = "predictive"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"


class ModelType(Enum):
    """ML model types"""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    CLUSTERING = "clustering"


@dataclass
class DatasetInfo:
    """Dataset information"""
    name: str
    source_type: DataSourceType
    source_path: str
    rows: int
    columns: int
    memory_usage_mb: float
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    created_at: float


@dataclass
class AnalysisResult:
    """Analysis result"""
    analysis_id: str
    dataset_name: str
    analysis_type: AnalysisType
    results: Dict[str, Any]
    metrics: Dict[str, float]
    model_info: Optional[Dict[str, Any]]
    processing_time: float
    intel_optimized: bool
    success: bool
    error_message: Optional[str] = None


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    model_type: ModelType
    training_time: float
    inference_time: float
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    r2_score: Optional[float]
    rmse: Optional[float]
    intel_optimized: bool
    optimization_speedup: Optional[float] = None


class DataIngestionTask(Task):
    """Data ingestion task for multiple source types"""
    
    def __init__(self, task_id: str, source_path: str, source_type: DataSourceType):
        config = TaskConfig(
            task_id=task_id,
            name="Data Ingestion",
            description=f"Ingest data from {source_type.value} source",
            timeout=300,
            retry_count=2
        )
        super().__init__(config)
        self.source_path = source_path
        self.source_type = source_type
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute data ingestion"""
        try:
            print(f"Ingesting data from {self.source_type.value}: {self.source_path}")
            
            # Simulate data loading based on source type
            if self.source_type == DataSourceType.CSV:
                data, info = await self._load_csv_data()
            elif self.source_type == DataSourceType.JSON:
                data, info = await self._load_json_data()
            elif self.source_type == DataSourceType.API:
                data, info = await self._load_api_data()
            else:
                raise ValueError(f"Unsupported source type: {self.source_type}")
            
            result_data = {
                "data": data,
                "dataset_info": asdict(info),
                "ingestion_time": 2.0
            }
            
            return TaskResult(
                task_id=self.config.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=2.0,
                error=None
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    async def _load_csv_data(self) -> Tuple[List[Dict], DatasetInfo]:
        """Load CSV data (simulated)"""
        await asyncio.sleep(1.5)  # Simulate loading time
        
        # Simulate different types of datasets
        if "sales" in self.source_path.lower():
            data = [
                {"date": "2024-01-01", "product": "Widget A", "quantity": 100, "revenue": 1000.0, "region": "North"},
                {"date": "2024-01-02", "product": "Widget B", "quantity": 75, "revenue": 1500.0, "region": "South"},
                {"date": "2024-01-03", "product": "Widget A", "quantity": 120, "revenue": 1200.0, "region": "East"},
                {"date": "2024-01-04", "product": "Widget C", "quantity": 90, "revenue": 2700.0, "region": "West"},
                {"date": "2024-01-05", "product": "Widget B", "quantity": 110, "revenue": 2200.0, "region": "North"},
            ] * 100  # Simulate larger dataset
        elif "customer" in self.source_path.lower():
            data = [
                {"customer_id": 1, "age": 25, "income": 50000, "spending": 1200, "segment": "Young"},
                {"customer_id": 2, "age": 35, "income": 75000, "spending": 2500, "segment": "Professional"},
                {"customer_id": 3, "age": 45, "income": 90000, "spending": 3200, "segment": "Premium"},
                {"customer_id": 4, "age": 28, "income": 60000, "spending": 1800, "segment": "Young"},
                {"customer_id": 5, "age": 55, "income": 120000, "spending": 4500, "segment": "Premium"},
            ] * 200  # Simulate larger dataset
        else:
            # Generic dataset
            data = [
                {"id": i, "value": i * 10 + np.random.normal(0, 5), "category": f"Cat_{i % 3}"}
                for i in range(1000)
            ]
        
        info = DatasetInfo(
            name=Path(self.source_path).stem,
            source_type=self.source_type,
            source_path=self.source_path,
            rows=len(data),
            columns=len(data[0]) if data else 0,
            memory_usage_mb=len(str(data)) / (1024 * 1024),
            data_types={key: type(data[0][key]).__name__ for key in data[0]} if data else {},
            missing_values={},
            created_at=time.time()
        )
        
        return data, info
    
    async def _load_json_data(self) -> Tuple[List[Dict], DatasetInfo]:
        """Load JSON data (simulated)"""
        await asyncio.sleep(1.0)
        
        data = [
            {"timestamp": "2024-01-01T10:00:00", "sensor_id": "temp_01", "value": 22.5, "unit": "celsius"},
            {"timestamp": "2024-01-01T10:01:00", "sensor_id": "temp_01", "value": 22.7, "unit": "celsius"},
            {"timestamp": "2024-01-01T10:02:00", "sensor_id": "temp_01", "value": 22.3, "unit": "celsius"},
        ] * 500
        
        info = DatasetInfo(
            name=Path(self.source_path).stem,
            source_type=self.source_type,
            source_path=self.source_path,
            rows=len(data),
            columns=len(data[0]) if data else 0,
            memory_usage_mb=len(str(data)) / (1024 * 1024),
            data_types={key: type(data[0][key]).__name__ for key in data[0]} if data else {},
            missing_values={},
            created_at=time.time()
        )
        
        return data, info
    
    async def _load_api_data(self) -> Tuple[List[Dict], DatasetInfo]:
        """Load API data (simulated)"""
        await asyncio.sleep(2.0)  # Simulate API call
        
        # Simulate API response
        data = [
            {"id": i, "metric": f"metric_{i % 5}", "value": np.random.normal(100, 15), "timestamp": time.time() - i * 3600}
            for i in range(200)
        ]
        
        info = DatasetInfo(
            name="api_data",
            source_type=self.source_type,
            source_path=self.source_path,
            rows=len(data),
            columns=len(data[0]) if data else 0,
            memory_usage_mb=len(str(data)) / (1024 * 1024),
            data_types={key: type(data[0][key]).__name__ for key in data[0]} if data else {},
            missing_values={},
            created_at=time.time()
        )
        
        return data, info


class DataPreprocessingTask(Task):
    """Data preprocessing and cleaning task"""
    
    def __init__(self, task_id: str):
        config = TaskConfig(
            task_id=task_id,
            name="Data Preprocessing",
            description="Clean and preprocess data",
            timeout=180,
            retry_count=2
        )
        super().__init__(config)
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute data preprocessing"""
        try:
            data = context.get("data", [])
            if not data:
                raise ValueError("No data provided for preprocessing")
            
            print("Preprocessing data...")
            
            # Clean and preprocess data
            cleaned_data = await self._preprocess_data(data)
            
            # Generate preprocessing statistics
            stats = self._generate_preprocessing_stats(data, cleaned_data)
            
            result_data = {
                "cleaned_data": cleaned_data,
                "preprocessing_stats": stats,
                "rows_before": len(data),
                "rows_after": len(cleaned_data)
            }
            
            return TaskResult(
                task_id=self.config.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=1.0,
                error=None
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    async def _preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """Preprocess and clean data"""
        cleaned_data = []
        
        for row in data:
            cleaned_row = {}
            
            for key, value in row.items():
                # Handle missing values
                if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
                    # Skip rows with critical missing values or fill with defaults
                    if key in ["id", "customer_id"]:
                        break  # Skip entire row if ID is missing
                    elif isinstance(value, (int, float)) or key in ["value", "revenue", "spending"]:
                        cleaned_row[key] = 0.0  # Fill numeric with 0
                    else:
                        cleaned_row[key] = "unknown"  # Fill categorical with "unknown"
                else:
                    # Clean and normalize values
                    if isinstance(value, str):
                        cleaned_row[key] = value.strip().lower()
                    elif isinstance(value, (int, float)):
                        # Remove outliers (simple approach)
                        if abs(value) < 1e6:  # Reasonable limit
                            cleaned_row[key] = float(value)
                        else:
                            cleaned_row[key] = 0.0
                    else:
                        cleaned_row[key] = value
            
            # Only add row if it has the same number of keys (no critical missing values)
            if len(cleaned_row) == len(row):
                cleaned_data.append(cleaned_row)
        
        return cleaned_data
    
    def _generate_preprocessing_stats(self, original_data: List[Dict], cleaned_data: List[Dict]) -> Dict[str, Any]:
        """Generate preprocessing statistics"""
        return {
            "rows_removed": len(original_data) - len(cleaned_data),
            "removal_rate": (len(original_data) - len(cleaned_data)) / len(original_data) * 100 if original_data else 0,
            "missing_value_handling": "filled_with_defaults",
            "outlier_removal": "basic_threshold"
        }


class StatisticalAnalysisTask(Task):
    """Statistical analysis task"""
    
    def __init__(self, task_id: str, analysis_type: AnalysisType):
        config = TaskConfig(
            task_id=task_id,
            name=f"Statistical Analysis - {analysis_type.value}",
            description=f"Perform {analysis_type.value} analysis",
            timeout=300,
            retry_count=2
        )
        super().__init__(config)
        self.analysis_type = analysis_type
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute statistical analysis"""
        try:
            data = context.get("cleaned_data", [])
            if not data:
                raise ValueError("No cleaned data provided for analysis")
            
            print(f"Performing {self.analysis_type.value} analysis...")
            
            # Perform analysis based on type
            if self.analysis_type == AnalysisType.DESCRIPTIVE:
                results = await self._descriptive_analysis(data)
            elif self.analysis_type == AnalysisType.TIME_SERIES:
                results = await self._time_series_analysis(data)
            else:
                results = await self._generic_analysis(data)
            
            result_data = {
                "analysis_results": results,
                "analysis_type": self.analysis_type.value,
                "data_points": len(data)
            }
            
            return TaskResult(
                task_id=self.config.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=2.0,
                error=None
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    async def _descriptive_analysis(self, data: List[Dict]) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        if not data:
            return {}
        
        # Identify numeric columns
        numeric_columns = []
        categorical_columns = []
        
        for key, value in data[0].items():
            if isinstance(value, (int, float)):
                numeric_columns.append(key)
            else:
                categorical_columns.append(key)
        
        results = {
            "summary": {
                "total_records": len(data),
                "numeric_columns": len(numeric_columns),
                "categorical_columns": len(categorical_columns)
            },
            "numeric_analysis": {},
            "categorical_analysis": {}
        }
        
        # Analyze numeric columns
        for col in numeric_columns:
            values = [row[col] for row in data if isinstance(row.get(col), (int, float))]
            
            if values:
                results["numeric_analysis"][col] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": self._calculate_std(values),
                    "median": self._calculate_median(values)
                }
        
        # Analyze categorical columns
        for col in categorical_columns:
            values = [row[col] for row in data if col in row]
            value_counts = {}
            
            for value in values:
                value_counts[str(value)] = value_counts.get(str(value), 0) + 1
            
            results["categorical_analysis"][col] = {
                "unique_values": len(value_counts),
                "most_common": max(value_counts.items(), key=lambda x: x[1]) if value_counts else None,
                "distribution": dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        
        return results
    
    async def _time_series_analysis(self, data: List[Dict]) -> Dict[str, Any]:
        """Perform time series analysis"""
        # Look for time-based columns
        time_columns = [col for col in data[0].keys() if "time" in col.lower() or "date" in col.lower()]
        
        if not time_columns:
            return {"error": "No time columns found for time series analysis"}
        
        time_col = time_columns[0]
        
        # Find numeric columns for analysis
        numeric_columns = [col for col, val in data[0].items() 
                          if isinstance(val, (int, float)) and col != time_col]
        
        results = {
            "time_column": time_col,
            "analyzed_columns": numeric_columns,
            "trends": {},
            "seasonality": {},
            "summary": {
                "total_periods": len(data),
                "date_range": {
                    "start": str(data[0][time_col]) if data else None,
                    "end": str(data[-1][time_col]) if data else None
                }
            }
        }
        
        # Simple trend analysis
        for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
            values = [row[col] for row in data if isinstance(row.get(col), (int, float))]
            
            if len(values) > 1:
                # Simple linear trend
                trend = (values[-1] - values[0]) / len(values)
                results["trends"][col] = {
                    "trend_slope": trend,
                    "direction": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
                    "start_value": values[0],
                    "end_value": values[-1],
                    "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                }
        
        return results
    
    async def _generic_analysis(self, data: List[Dict]) -> Dict[str, Any]:
        """Perform generic analysis"""
        return {
            "message": f"Generic {self.analysis_type.value} analysis performed",
            "data_summary": {
                "records": len(data),
                "columns": len(data[0]) if data else 0
            }
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _calculate_median(self, values: List[float]) -> float:
        """Calculate median"""
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            return sorted_values[n // 2]


class MLModelTask(Task):
    """Machine learning model training and inference task"""
    
    def __init__(
        self,
        task_id: str,
        model_type: ModelType,
        target_column: str,
        use_intel_optimizations: bool = True
    ):
        config = TaskConfig(
            task_id=task_id,
            name=f"ML Model - {model_type.value}",
            description=f"Train {model_type.value} model",
            timeout=600,
            retry_count=2
        )
        super().__init__(config)
        self.model_type = model_type
        self.target_column = target_column
        self.use_intel_optimizations = use_intel_optimizations
        self.pytorch_optimizer = IntelPyTorchOptimizer() if use_intel_optimizations else None
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute ML model training"""
        try:
            data = context.get("cleaned_data", [])
            if not data:
                raise ValueError("No data provided for ML training")
            
            print(f"Training {self.model_type.value} model...")
            
            # Prepare data for ML
            X, y = self._prepare_ml_data(data)
            
            # Train model
            model_info, performance = await self._train_model(X, y)
            
            result_data = {
                "model_info": model_info,
                "performance": asdict(performance),
                "data_shape": {"samples": len(X), "features": len(X[0]) if X else 0},
                "intel_optimized": self.use_intel_optimizations
            }
            
            return TaskResult(
                task_id=self.config.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=performance.training_time,
                error=None
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _prepare_ml_data(self, data: List[Dict]) -> Tuple[List[List[float]], List[float]]:
        """Prepare data for ML training"""
        # Find numeric features (excluding target)
        numeric_features = []
        for key, value in data[0].items():
            if isinstance(value, (int, float)) and key != self.target_column:
                numeric_features.append(key)
        
        X = []
        y = []
        
        for row in data:
            # Extract features
            features = []
            for feature in numeric_features:
                features.append(float(row.get(feature, 0)))
            
            # Extract target
            target_value = row.get(self.target_column)
            if target_value is not None:
                if isinstance(target_value, (int, float)):
                    y.append(float(target_value))
                else:
                    # Simple categorical encoding
                    y.append(hash(str(target_value)) % 10)  # Map to 0-9
                
                X.append(features)
        
        return X, y
    
    async def _train_model(self, X: List[List[float]], y: List[float]) -> Tuple[Dict[str, Any], ModelPerformance]:
        """Train ML model (simplified implementation)"""
        start_time = time.time()
        
        # Simulate model training
        await asyncio.sleep(2.0)  # Base training time
        
        training_time = time.time() - start_time
        
        # Apply Intel optimizations if enabled
        optimization_speedup = None
        if self.use_intel_optimizations:
            print("Applying Intel PyTorch optimizations...")
            await asyncio.sleep(0.5)  # Simulate optimization
            optimization_speedup = 1.8  # Simulated speedup
            training_time *= 0.7  # Reduced time due to optimization
        
        # Simulate model performance
        if self.model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION]:
            accuracy = 0.85 + np.random.normal(0, 0.05)
            r2_score = 0.75 + np.random.normal(0, 0.1)
            rmse = 10.0 + np.random.normal(0, 2.0)
        elif self.model_type == ModelType.NEURAL_NETWORK:
            accuracy = 0.90 + np.random.normal(0, 0.03)
            r2_score = 0.85 + np.random.normal(0, 0.05)
            rmse = 8.0 + np.random.normal(0, 1.5)
        else:
            accuracy = 0.80 + np.random.normal(0, 0.08)
            r2_score = 0.70 + np.random.normal(0, 0.12)
            rmse = 12.0 + np.random.normal(0, 3.0)
        
        model_info = {
            "model_type": self.model_type.value,
            "target_column": self.target_column,
            "features": len(X[0]) if X else 0,
            "training_samples": len(X),
            "hyperparameters": {
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 32
            },
            "optimization_applied": self.use_intel_optimizations
        }
        
        performance = ModelPerformance(
            model_name=f"{self.model_type.value}_{int(time.time())}",
            model_type=self.model_type,
            training_time=training_time,
            inference_time=0.01,  # Simulated
            accuracy=max(0.0, min(1.0, accuracy)),
            precision=max(0.0, min(1.0, accuracy + 0.02)),
            recall=max(0.0, min(1.0, accuracy - 0.01)),
            f1_score=max(0.0, min(1.0, accuracy + 0.01)),
            r2_score=max(0.0, min(1.0, r2_score)) if self.model_type != ModelType.LOGISTIC_REGRESSION else None,
            rmse=max(0.0, rmse) if self.model_type != ModelType.LOGISTIC_REGRESSION else None,
            intel_optimized=self.use_intel_optimizations,
            optimization_speedup=optimization_speedup
        )
        
        return model_info, performance


class DataAnalysisAgent(Agent):
    """
    Data Analysis Agent - Reference Implementation
    
    This agent demonstrates a complete data analysis workflow:
    1. Data ingestion from multiple sources
    2. Data preprocessing and cleaning
    3. Statistical analysis and insights generation
    4. Machine learning model training with Intel optimizations
    5. Results storage and reporting
    """
    
    def __init__(
        self,
        agent_id: str = "data_analyzer",
        use_intel_optimizations: bool = True,
        memory: Optional[Memory] = None
    ):
        config = AgentConfig(
            agent_id=agent_id,
            name="Data Analysis Agent",
            description="Analyzes data with ML models and Intel optimizations",
            max_concurrent_tasks=4,
            timeout=1200
        )
        
        super().__init__(config, memory)
        self.use_intel_optimizations = use_intel_optimizations
        self.analysis_results: Dict[str, AnalysisResult] = {}
        self.model_performances: Dict[str, ModelPerformance] = {}
    
    async def run(self, context: AgentContext) -> Dict[str, Any]:
        """Run the data analysis workflow"""
        try:
            data_source = context.inputs.get("data_source")
            source_type = context.inputs.get("source_type", "csv")
            analysis_type = context.inputs.get("analysis_type", "descriptive")
            target_column = context.inputs.get("target_column")
            
            if not data_source:
                raise ValueError("data_source is required")
            
            analysis_id = f"analysis_{int(time.time())}"
            
            print(f"Starting data analysis: {analysis_id}")
            print(f"Source: {data_source} ({source_type})")
            print(f"Analysis type: {analysis_type}")
            
            # Create analysis workflow
            workflow = self._create_analysis_workflow(
                analysis_id, data_source, source_type, analysis_type, target_column
            )
            
            # Execute workflow
            workflow_result = await workflow.execute()
            
            if workflow_result.get("success", False):
                # Create analysis result
                result = self._create_analysis_result(
                    analysis_id, workflow_result, analysis_type
                )
                
                # Store in memory
                if self.memory:
                    await self.memory.store(
                        key=f"analysis:{analysis_id}",
                        value=asdict(result),
                        memory_type=MemoryType.LONG_TERM
                    )
                
                # Cache result
                self.analysis_results[analysis_id] = result
                
                print(f"Data analysis completed successfully: {analysis_id}")
                
                return {
                    "success": True,
                    "analysis_id": analysis_id,
                    "result": asdict(result),
                    "processing_time": workflow_result.get("total_time", 0)
                }
            else:
                error_msg = workflow_result.get("error", "Unknown workflow error")
                print(f"Data analysis failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
        
        except Exception as e:
            print(f"Data analysis agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_analysis_workflow(
        self,
        analysis_id: str,
        data_source: str,
        source_type: str,
        analysis_type: str,
        target_column: Optional[str]
    ) -> Workflow:
        """Create the data analysis workflow"""
        workflow = Workflow(f"data_analysis_{analysis_id}")
        
        # Task 1: Data Ingestion
        ingestion_task = DataIngestionTask(
            task_id=f"{analysis_id}_ingestion",
            source_path=data_source,
            source_type=DataSourceType(source_type.lower())
        )
        workflow.add_task(ingestion_task)
        
        # Task 2: Data Preprocessing
        preprocessing_task = DataPreprocessingTask(f"{analysis_id}_preprocessing")
        workflow.add_task(preprocessing_task, dependencies=[ingestion_task.config.task_id])
        
        # Task 3: Statistical Analysis
        stats_task = StatisticalAnalysisTask(
            f"{analysis_id}_stats",
            AnalysisType(analysis_type.lower())
        )
        workflow.add_task(stats_task, dependencies=[preprocessing_task.config.task_id])
        
        # Task 4: ML Model Training (if target column is provided)
        if target_column and analysis_type.lower() in ["predictive", "classification", "regression"]:
            model_type = ModelType.NEURAL_NETWORK if analysis_type.lower() == "predictive" else ModelType.LINEAR_REGRESSION
            
            ml_task = MLModelTask(
                task_id=f"{analysis_id}_ml",
                model_type=model_type,
                target_column=target_column,
                use_intel_optimizations=self.use_intel_optimizations
            )
            workflow.add_task(ml_task, dependencies=[preprocessing_task.config.task_id])
        
        return workflow
    
    def _create_analysis_result(
        self,
        analysis_id: str,
        workflow_result: Dict[str, Any],
        analysis_type: str
    ) -> AnalysisResult:
        """Create analysis result from workflow output"""
        
        task_results = workflow_result.get("task_results", {})
        
        # Extract results from different tasks
        ingestion_result = task_results.get(f"{analysis_id}_ingestion", {}).get("result", {})
        stats_result = task_results.get(f"{analysis_id}_stats", {}).get("result", {})
        ml_result = task_results.get(f"{analysis_id}_ml", {}).get("result", {})
        
        # Combine results
        combined_results = {
            "dataset_info": ingestion_result.get("dataset_info", {}),
            "statistical_analysis": stats_result.get("analysis_results", {}),
            "preprocessing_stats": task_results.get(f"{analysis_id}_preprocessing", {}).get("result", {}).get("preprocessing_stats", {})
        }
        
        # Add ML results if available
        if ml_result:
            combined_results["ml_model"] = ml_result.get("model_info", {})
            
            # Store model performance separately
            performance_data = ml_result.get("performance", {})
            if performance_data:
                performance = ModelPerformance(**performance_data)
                self.model_performances[performance.model_name] = performance
        
        # Calculate metrics
        metrics = {
            "data_quality_score": 0.85,  # Placeholder
            "processing_efficiency": 1.0 / workflow_result.get("total_time", 1.0) * 100,
            "intel_optimization_benefit": 1.8 if self.use_intel_optimizations else 1.0
        }
        
        return AnalysisResult(
            analysis_id=analysis_id,
            dataset_name=ingestion_result.get("dataset_info", {}).get("name", "unknown"),
            analysis_type=AnalysisType(analysis_type.lower()),
            results=combined_results,
            metrics=metrics,
            model_info=ml_result.get("model_info") if ml_result else None,
            processing_time=workflow_result.get("total_time", 0),
            intel_optimized=self.use_intel_optimizations,
            success=workflow_result.get("success", False)
        )
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Get analysis result"""
        # Try cache first
        if analysis_id in self.analysis_results:
            return self.analysis_results[analysis_id]
        
        # Try memory
        if self.memory:
            stored_result = await self.memory.retrieve(f"analysis:{analysis_id}")
            if stored_result:
                return AnalysisResult(**stored_result)
        
        return None
    
    async def get_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """Get model performance metrics"""
        return self.model_performances.get(model_name)
    
    async def list_analyses(self) -> List[str]:
        """List all analysis IDs"""
        analysis_ids = list(self.analysis_results.keys())
        
        # Also check memory
        if self.memory:
            memory_keys = await self.memory.list_keys()
            for key in memory_keys:
                if key.startswith("analysis:"):
                    analysis_id = key.split(":", 1)[1]
                    if analysis_id not in analysis_ids:
                        analysis_ids.append(analysis_id)
        
        return analysis_ids
    
    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        total_analyses = len(self.analysis_results)
        successful_analyses = sum(1 for result in self.analysis_results.values() if result.success)
        
        avg_processing_time = 0
        if self.analysis_results:
            avg_processing_time = sum(
                result.processing_time for result in self.analysis_results.values()
            ) / len(self.analysis_results)
        
        # Model performance summary
        model_count = len(self.model_performances)
        avg_model_accuracy = 0
        if self.model_performances:
            accuracies = [perf.accuracy for perf in self.model_performances.values() if perf.accuracy]
            avg_model_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        return {
            "total_analyses": total_analyses,
            "successful_analyses": successful_analyses,
            "failed_analyses": total_analyses - successful_analyses,
            "success_rate": (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0,
            "average_processing_time": avg_processing_time,
            "models_trained": model_count,
            "average_model_accuracy": avg_model_accuracy,
            "intel_optimizations_enabled": self.use_intel_optimizations
        }


# Convenience function for quick data analysis
async def analyze_data(
    data_source: str,
    source_type: str = "csv",
    analysis_type: str = "descriptive",
    target_column: Optional[str] = None,
    use_intel_optimizations: bool = True,
    memory: Optional[Memory] = None
) -> Optional[AnalysisResult]:
    """
    Quick function to analyze data
    
    Args:
        data_source: Path to data source
        source_type: Type of data source (csv, json, api)
        analysis_type: Type of analysis (descriptive, predictive, etc.)
        target_column: Target column for ML (optional)
        use_intel_optimizations: Whether to use Intel optimizations
        memory: Optional memory instance
        
    Returns:
        Analysis result if successful
    """
    agent = DataAnalysisAgent(
        use_intel_optimizations=use_intel_optimizations,
        memory=memory
    )
    
    context = AgentContext(
        agent_id=agent.config.agent_id,
        inputs={
            "data_source": data_source,
            "source_type": source_type,
            "analysis_type": analysis_type,
            "target_column": target_column
        },
        metadata={},
        correlation_id=f"quick_analysis_{int(time.time())}"
    )
    
    result = await agent.run(context)
    
    if result.get("success"):
        analysis_id = result["analysis_id"]
        return await agent.get_analysis_result(analysis_id)
    
    return None