"""
Document Processing Agent - Reference Implementation

This agent demonstrates a complete document processing workflow with:
- OCR text extraction from images/PDFs
- Text preprocessing and cleaning
- Information extraction using NLP
- Summary generation
- Intel OpenVINO optimization for OCR models
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

# Framework imports
from ..core.agent import Agent, AgentConfig, AgentContext
from ..core.task import Task, TaskConfig, TaskResult, TaskStatus
from ..core.workflow import Workflow
from ..core.memory import Memory, MemoryType
from ..intel_optimizations.openvino_optimizer import OpenVINOOptimizer, OptimizationConfig, ModelType, DeviceType


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    DOCX = "docx"


class ProcessingStage(Enum):
    """Document processing stages"""
    UPLOAD = "upload"
    OCR = "ocr"
    PREPROCESSING = "preprocessing"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    STORAGE = "storage"


@dataclass
class DocumentMetadata:
    """Document metadata"""
    filename: str
    file_type: DocumentType
    file_size: int
    uploaded_at: float
    processing_stages: List[ProcessingStage]
    confidence_scores: Dict[str, float]
    extracted_entities: Dict[str, List[str]]


@dataclass
class ProcessingResult:
    """Document processing result"""
    document_id: str
    metadata: DocumentMetadata
    raw_text: str
    cleaned_text: str
    summary: str
    key_entities: Dict[str, List[str]]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class OCRTask(Task):
    """OCR text extraction task with Intel OpenVINO optimization"""
    
    def __init__(self, task_id: str, document_path: str, use_optimization: bool = True):
        config = TaskConfig(
            task_id=task_id,
            name="OCR Text Extraction",
            description="Extract text from images/PDFs using OCR",
            timeout=300,
            retry_count=2
        )
        super().__init__(config)
        self.document_path = document_path
        self.use_optimization = use_optimization
        self.optimizer = OpenVINOOptimizer() if use_optimization else None
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute OCR extraction"""
        try:
            print(f"Starting OCR extraction for: {self.document_path}")
            
            # Simulate OCR processing time
            await asyncio.sleep(2)
            
            # In a real implementation, this would use:
            # - EasyOCR or PaddleOCR for text extraction
            # - Intel OpenVINO optimized OCR models
            # - Preprocessing with OpenCV
            
            # Simulate OCR text extraction
            extracted_text = self._simulate_ocr_extraction()
            
            # Apply Intel optimizations if enabled
            if self.use_optimization and self.optimizer:
                print("Applying Intel OpenVINO OCR optimization...")
                # This would optimize the OCR model for faster inference
                # For now, simulate the optimization benefit
                await asyncio.sleep(0.5)  # Reduced processing time due to optimization
            
            result_data = {
                "extracted_text": extracted_text,
                "confidence": 0.92,
                "processing_time": 2.0 if not self.use_optimization else 1.5,
                "optimized": self.use_optimization
            }
            
            return TaskResult(
                task_id=self.config.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=result_data["processing_time"],
                error=None
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _simulate_ocr_extraction(self) -> str:
        """Simulate OCR text extraction"""
        # Simulate different document types
        if "invoice" in self.document_path.lower():
            return """
            INVOICE #INV-2024-001
            Date: January 15, 2024
            
            Bill To:
            ABC Corporation
            123 Business St
            City, State 12345
            
            Item Description    Qty    Price    Total
            Software License    1      $500.00  $500.00
            Consulting Hours    10     $150.00  $1,500.00
            
            Subtotal: $2,000.00
            Tax: $200.00
            Total: $2,200.00
            """
        elif "contract" in self.document_path.lower():
            return """
            SERVICE AGREEMENT
            
            This Service Agreement ("Agreement") is entered into on January 1, 2024,
            between Client Corp ("Client") and Service Provider LLC ("Provider").
            
            1. SCOPE OF SERVICES
            Provider agrees to provide software development services as detailed
            in Exhibit A attached hereto.
            
            2. TERM
            This Agreement shall commence on January 1, 2024, and continue for
            a period of twelve (12) months.
            
            3. COMPENSATION
            Client agrees to pay Provider $10,000 per month for services rendered.
            """
        else:
            return """
            Sample Document Content
            
            This is a sample document used for demonstration purposes.
            It contains various types of information that can be extracted
            and processed by the document processing agent.
            
            Key Information:
            - Date: January 2024
            - Amount: $1,000
            - Reference: REF-2024-001
            """


class TextPreprocessingTask(Task):
    """Text preprocessing and cleaning task"""
    
    def __init__(self, task_id: str):
        config = TaskConfig(
            task_id=task_id,
            name="Text Preprocessing",
            description="Clean and preprocess extracted text",
            timeout=60,
            retry_count=1
        )
        super().__init__(config)
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute text preprocessing"""
        try:
            raw_text = context.get("extracted_text", "")
            
            if not raw_text:
                raise ValueError("No text provided for preprocessing")
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(raw_text)
            
            result_data = {
                "cleaned_text": cleaned_text,
                "word_count": len(cleaned_text.split()),
                "character_count": len(cleaned_text)
            }
            
            return TaskResult(
                task_id=self.config.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=0.1,
                error=None
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        
        # Normalize case (keep original for now)
        text = text.strip()
        
        return text


class EntityExtractionTask(Task):
    """Entity extraction task using NLP"""
    
    def __init__(self, task_id: str):
        config = TaskConfig(
            task_id=task_id,
            name="Entity Extraction",
            description="Extract key entities from text",
            timeout=120,
            retry_count=2
        )
        super().__init__(config)
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute entity extraction"""
        try:
            text = context.get("cleaned_text", "")
            
            if not text:
                raise ValueError("No cleaned text provided for entity extraction")
            
            # Extract entities (simplified rule-based approach)
            entities = self._extract_entities(text)
            
            result_data = {
                "entities": entities,
                "entity_count": sum(len(v) for v in entities.values())
            }
            
            return TaskResult(
                task_id=self.config.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=0.5,
                error=None
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using rule-based approach"""
        import re
        
        entities = {
            "dates": [],
            "amounts": [],
            "companies": [],
            "references": [],
            "emails": [],
            "phone_numbers": []
        }
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dates"].extend(matches)
        
        # Extract monetary amounts
        amount_pattern = r'\$[\d,]+\.?\d*'
        entities["amounts"] = re.findall(amount_pattern, text)
        
        # Extract company names (simplified)
        company_patterns = [
            r'\b[A-Z][a-zA-Z]*\s+(?:Corp|Corporation|LLC|Inc|Company|Ltd)\b',
            r'\b[A-Z][a-zA-Z]*\s+[A-Z][a-zA-Z]*\s+(?:Corp|Corporation|LLC|Inc|Company|Ltd)\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            entities["companies"].extend(matches)
        
        # Extract reference numbers
        ref_patterns = [
            r'\b(?:REF|INV|ID)[-\s]*\d+[-\w]*\b',
            r'\b[A-Z]{2,}-\d{4}-\d+\b'
        ]
        
        for pattern in ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["references"].extend(matches)
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["emails"] = re.findall(email_pattern, text)
        
        # Extract phone numbers
        phone_patterns = [
            r'\b\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-\.\s]?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            entities["phone_numbers"].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities


class SummarizationTask(Task):
    """Text summarization task"""
    
    def __init__(self, task_id: str, max_length: int = 500):
        config = TaskConfig(
            task_id=task_id,
            name="Text Summarization",
            description="Generate summary of document text",
            timeout=180,
            retry_count=2
        )
        super().__init__(config)
        self.max_length = max_length
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute text summarization"""
        try:
            text = context.get("cleaned_text", "")
            entities = context.get("entities", {})
            
            if not text:
                raise ValueError("No text provided for summarization")
            
            # Generate summary
            summary = self._generate_summary(text, entities)
            
            result_data = {
                "summary": summary,
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text) if text else 0
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
    
    def _generate_summary(self, text: str, entities: Dict[str, List[str]]) -> str:
        """Generate extractive summary"""
        # Simple extractive summarization
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Score sentences based on entity presence and position
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position score (earlier sentences get higher score)
            score += max(0, 10 - i)
            
            # Entity presence score
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.lower() in sentence.lower():
                        score += 5
            
            # Length score (prefer medium-length sentences)
            words = len(sentence.split())
            if 10 <= words <= 30:
                score += 3
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select sentences for summary
        summary_sentences = []
        current_length = 0
        
        for sentence, score in scored_sentences:
            if current_length + len(sentence) <= self.max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
            
            if len(summary_sentences) >= 3:  # Limit to 3 key sentences
                break
        
        # Reorder sentences by original position
        original_order = []
        for sentence in sentences:
            if sentence in summary_sentences:
                original_order.append(sentence)
        
        return '. '.join(original_order) + '.' if original_order else "No summary available."


class DocumentProcessingAgent(Agent):
    """
    Document Processing Agent - Reference Implementation
    
    This agent demonstrates a complete workflow for processing documents:
    1. OCR text extraction (with Intel OpenVINO optimization)
    2. Text preprocessing and cleaning
    3. Entity extraction using NLP
    4. Text summarization
    5. Results storage in memory
    """
    
    def __init__(
        self,
        agent_id: str = "document_processor",
        use_intel_optimizations: bool = True,
        memory: Optional[Memory] = None
    ):
        config = AgentConfig(
            agent_id=agent_id,
            name="Document Processing Agent",
            description="Processes documents with OCR, NLP, and summarization",
            max_concurrent_tasks=3,
            timeout=600
        )
        
        super().__init__(config, memory)
        self.use_intel_optimizations = use_intel_optimizations
        self.processed_documents: Dict[str, ProcessingResult] = {}
    
    async def run(self, context: AgentContext) -> Dict[str, Any]:
        """Run the document processing workflow"""
        try:
            document_path = context.inputs.get("document_path")
            if not document_path:
                raise ValueError("document_path is required")
            
            document_id = f"doc_{int(time.time())}"
            
            print(f"Starting document processing for: {document_path}")
            
            # Create processing workflow
            workflow = self._create_processing_workflow(document_path, document_id)
            
            # Execute workflow
            workflow_result = await workflow.execute()
            
            if workflow_result.get("success", False):
                # Create processing result
                result = self._create_processing_result(
                    document_id, document_path, workflow_result
                )
                
                # Store in memory
                if self.memory:
                    await self.memory.store(
                        key=f"document:{document_id}",
                        value=result.__dict__,
                        memory_type=MemoryType.LONG_TERM
                    )
                
                # Cache result
                self.processed_documents[document_id] = result
                
                print(f"Document processing completed successfully: {document_id}")
                
                return {
                    "success": True,
                    "document_id": document_id,
                    "result": result.__dict__,
                    "processing_time": workflow_result.get("total_time", 0)
                }
            else:
                error_msg = workflow_result.get("error", "Unknown workflow error")
                print(f"Document processing failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
        
        except Exception as e:
            print(f"Document processing agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_processing_workflow(self, document_path: str, document_id: str) -> Workflow:
        """Create the document processing workflow"""
        workflow = Workflow(f"document_processing_{document_id}")
        
        # Task 1: OCR Text Extraction
        ocr_task = OCRTask(
            task_id=f"{document_id}_ocr",
            document_path=document_path,
            use_optimization=self.use_intel_optimizations
        )
        workflow.add_task(ocr_task)
        
        # Task 2: Text Preprocessing
        preprocessing_task = TextPreprocessingTask(f"{document_id}_preprocessing")
        workflow.add_task(preprocessing_task, dependencies=[ocr_task.config.task_id])
        
        # Task 3: Entity Extraction
        entity_task = EntityExtractionTask(f"{document_id}_entities")
        workflow.add_task(entity_task, dependencies=[preprocessing_task.config.task_id])
        
        # Task 4: Summarization
        summary_task = SummarizationTask(f"{document_id}_summary")
        workflow.add_task(summary_task, dependencies=[preprocessing_task.config.task_id, entity_task.config.task_id])
        
        return workflow
    
    def _create_processing_result(
        self,
        document_id: str,
        document_path: str,
        workflow_result: Dict[str, Any]
    ) -> ProcessingResult:
        """Create processing result from workflow output"""
        
        # Extract results from workflow
        task_results = workflow_result.get("task_results", {})
        
        # Get OCR result
        ocr_result = task_results.get(f"{document_id}_ocr", {}).get("result", {})
        raw_text = ocr_result.get("extracted_text", "")
        
        # Get preprocessing result
        preprocessing_result = task_results.get(f"{document_id}_preprocessing", {}).get("result", {})
        cleaned_text = preprocessing_result.get("cleaned_text", "")
        
        # Get entity extraction result
        entity_result = task_results.get(f"{document_id}_entities", {}).get("result", {})
        entities = entity_result.get("entities", {})
        
        # Get summarization result
        summary_result = task_results.get(f"{document_id}_summary", {}).get("result", {})
        summary = summary_result.get("summary", "")
        
        # Create metadata
        metadata = DocumentMetadata(
            filename=Path(document_path).name,
            file_type=self._detect_file_type(document_path),
            file_size=0,  # Placeholder
            uploaded_at=time.time(),
            processing_stages=[stage for stage in ProcessingStage],
            confidence_scores={
                "ocr": ocr_result.get("confidence", 0.0),
                "entities": 0.85,  # Placeholder
                "summary": 0.90   # Placeholder
            },
            extracted_entities=entities
        )
        
        return ProcessingResult(
            document_id=document_id,
            metadata=metadata,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            summary=summary,
            key_entities=entities,
            processing_time=workflow_result.get("total_time", 0),
            success=workflow_result.get("success", False)
        )
    
    def _detect_file_type(self, file_path: str) -> DocumentType:
        """Detect document type from file path"""
        suffix = Path(file_path).suffix.lower()
        
        if suffix == '.pdf':
            return DocumentType.PDF
        elif suffix in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return DocumentType.IMAGE
        elif suffix == '.docx':
            return DocumentType.DOCX
        else:
            return DocumentType.TEXT
    
    async def get_processing_result(self, document_id: str) -> Optional[ProcessingResult]:
        """Get processing result for a document"""
        # Try cache first
        if document_id in self.processed_documents:
            return self.processed_documents[document_id]
        
        # Try memory
        if self.memory:
            stored_result = await self.memory.retrieve(f"document:{document_id}")
            if stored_result:
                # Convert back to ProcessingResult
                return ProcessingResult(**stored_result)
        
        return None
    
    async def list_processed_documents(self) -> List[str]:
        """List all processed document IDs"""
        document_ids = list(self.processed_documents.keys())
        
        # Also check memory for additional documents
        if self.memory:
            memory_keys = await self.memory.list_keys()
            for key in memory_keys:
                if key.startswith("document:"):
                    doc_id = key.split(":", 1)[1]
                    if doc_id not in document_ids:
                        document_ids.append(doc_id)
        
        return document_ids
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total_docs = len(self.processed_documents)
        successful_docs = sum(1 for result in self.processed_documents.values() if result.success)
        
        avg_processing_time = 0
        if self.processed_documents:
            avg_processing_time = sum(
                result.processing_time for result in self.processed_documents.values()
            ) / len(self.processed_documents)
        
        return {
            "total_documents": total_docs,
            "successful_documents": successful_docs,
            "failed_documents": total_docs - successful_docs,
            "success_rate": (successful_docs / total_docs * 100) if total_docs > 0 else 0,
            "average_processing_time": avg_processing_time,
            "intel_optimizations_enabled": self.use_intel_optimizations
        }


# Convenience function for quick document processing
async def process_document(
    document_path: str,
    use_intel_optimizations: bool = True,
    memory: Optional[Memory] = None
) -> Optional[ProcessingResult]:
    """
    Quick function to process a single document
    
    Args:
        document_path: Path to the document to process
        use_intel_optimizations: Whether to use Intel optimizations
        memory: Optional memory instance for storing results
        
    Returns:
        Processing result if successful
    """
    agent = DocumentProcessingAgent(
        use_intel_optimizations=use_intel_optimizations,
        memory=memory
    )
    
    context = AgentContext(
        agent_id=agent.config.agent_id,
        inputs={"document_path": document_path},
        metadata={},
        correlation_id=f"quick_process_{int(time.time())}"
    )
    
    result = await agent.run(context)
    
    if result.get("success"):
        document_id = result["document_id"]
        return await agent.get_processing_result(document_id)
    
    return None