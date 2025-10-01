"""
Reference Agents for the AI Agent Framework

This module contains reference implementations demonstrating real workflows:
- DocumentProcessingAgent: OCR, text processing, and Intel OpenVINO optimization
- DataAnalysisAgent: ML training, statistical analysis, and Intel PyTorch optimization
"""

from .document_processor import (
    DocumentProcessingAgent,
    DocumentType,
    ProcessingStage,
    DocumentMetadata,
    ProcessingResult,
    OCRTask,
    TextPreprocessingTask,
    EntityExtractionTask,
    SummarizationTask,
    process_document
)

from .data_analyzer import (
    DataAnalysisAgent,
    DataSourceType,
    AnalysisType,
    ModelType,
    DatasetInfo,
    AnalysisResult,
    ModelPerformance,
    DataIngestionTask,
    DataPreprocessingTask,
    StatisticalAnalysisTask,
    MLModelTask,
    analyze_data
)

__all__ = [
    # Document Processing
    'DocumentProcessingAgent',
    'DocumentType',
    'ProcessingStage',
    'DocumentMetadata',
    'ProcessingResult',
    'OCRTask',
    'TextPreprocessingTask',
    'EntityExtractionTask',
    'SummarizationTask',
    'process_document',
    
    # Data Analysis
    'DataAnalysisAgent',
    'DataSourceType',
    'AnalysisType',
    'ModelType',
    'DatasetInfo',
    'AnalysisResult',
    'ModelPerformance',
    'DataIngestionTask',
    'DataPreprocessingTask',
    'StatisticalAnalysisTask',
    'MLModelTask',
    'analyze_data',
]