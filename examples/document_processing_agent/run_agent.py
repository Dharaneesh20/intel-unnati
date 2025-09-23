"""
Document Processing Agent - Reference Implementation

This agent demonstrates a comprehensive document processing workflow:
1. Document ingestion (PDF, DOCX, images)
2. OCR processing with Intel-optimized models
3. Text extraction and chunking
4. Semantic analysis and categorization
5. Summary generation
6. Report creation
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.workflow import Workflow, Task
from src.core.agents import Agent
from src.tools.document_tools import (
    DocumentIngestionTool,
    OCRTool,
    TextExtractionTool,
    ChunkingTool,
    SemanticAnalysisTool,
    SummaryTool,
    ReportGeneratorTool
)


class DocumentProcessingAgent:
    """Document processing agent for automated document analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the document processing workflow and agent."""
        
        # Create workflow
        workflow = Workflow(
            name="document_processing",
            description="Automated document processing and analysis workflow"
        )
        
        # Task 1: Document Ingestion
        ingestion_task = Task(
            name="document_ingestion",
            tool=DocumentIngestionTool(),
            inputs={
                "input_dir": "{{input_directory}}",
                "supported_formats": ["pdf", "docx", "txt", "jpg", "png"]
            },
            outputs=["documents", "metadata"]
        )
        
        # Task 2: OCR Processing (for images and scanned PDFs)
        ocr_task = Task(
            name="ocr_processing",
            tool=OCRTool(
                engine="intel_optimized",  # Use Intel OpenVINO optimized OCR
                languages=["en"]
            ),
            inputs={
                "documents": "{{document_ingestion.documents}}",
                "metadata": "{{document_ingestion.metadata}}"
            },
            outputs=["ocr_results"],
            dependencies=["document_ingestion"]
        )
        
        # Task 3: Text Extraction
        extraction_task = Task(
            name="text_extraction",
            tool=TextExtractionTool(),
            inputs={
                "documents": "{{document_ingestion.documents}}",
                "ocr_results": "{{ocr_processing.ocr_results}}"
            },
            outputs=["extracted_text"],
            dependencies=["document_ingestion", "ocr_processing"]
        )
        
        # Task 4: Text Chunking
        chunking_task = Task(
            name="text_chunking",
            tool=ChunkingTool(
                chunk_size=512,
                overlap=50,
                strategy="semantic"
            ),
            inputs={
                "text_data": "{{text_extraction.extracted_text}}"
            },
            outputs=["text_chunks"],
            dependencies=["text_extraction"]
        )
        
        # Task 5: Semantic Analysis
        analysis_task = Task(
            name="semantic_analysis",
            tool=SemanticAnalysisTool(
                model="intel_optimized_bert",  # Intel OpenVINO optimized model
                categories=["technical", "legal", "financial", "general"]
            ),
            inputs={
                "text_chunks": "{{text_chunking.text_chunks}}"
            },
            outputs=["analysis_results", "categories", "entities"],
            dependencies=["text_chunking"]
        )
        
        # Task 6: Summary Generation
        summary_task = Task(
            name="summary_generation",
            tool=SummaryTool(
                model="intel_optimized_summarizer",
                max_length=200,
                min_length=50
            ),
            inputs={
                "text_data": "{{text_extraction.extracted_text}}",
                "analysis": "{{semantic_analysis.analysis_results}}"
            },
            outputs=["summary"],
            dependencies=["text_extraction", "semantic_analysis"]
        )
        
        # Task 7: Report Generation
        report_task = Task(
            name="report_generation",
            tool=ReportGeneratorTool(
                template="comprehensive_analysis",
                output_format="html"
            ),
            inputs={
                "documents": "{{document_ingestion.documents}}",
                "metadata": "{{document_ingestion.metadata}}",
                "analysis": "{{semantic_analysis.analysis_results}}",
                "summary": "{{summary_generation.summary}}",
                "categories": "{{semantic_analysis.categories}}",
                "entities": "{{semantic_analysis.entities}}"
            },
            outputs=["report", "report_path"],
            dependencies=["semantic_analysis", "summary_generation"]
        )
        
        # Add tasks to workflow
        workflow.add_task(ingestion_task)
        workflow.add_task(ocr_task)
        workflow.add_task(extraction_task)
        workflow.add_task(chunking_task)
        workflow.add_task(analysis_task)
        workflow.add_task(summary_task)
        workflow.add_task(report_task)
        
        # Set up dependencies using >> operator
        workflow.set_dependencies(
            ingestion_task >> ocr_task >> extraction_task >> 
            chunking_task >> analysis_task >> summary_task >> report_task
        )
        
        # Create agent
        agent = Agent(
            name="DocumentProcessingAgent",
            workflow=workflow,
            description="Automated document processing and analysis agent",
            max_retries=2,
            timeout=1800  # 30 minutes
        )
        
        return agent
    
    async def process_documents(
        self,
        input_directory: str,
        output_directory: str = None
    ) -> Dict[str, Any]:
        """Process documents in the specified directory."""
        
        if not os.path.exists(input_directory):
            raise ValueError(f"Input directory does not exist: {input_directory}")
        
        if output_directory is None:
            output_directory = os.path.join(input_directory, "results")
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Prepare inputs
        inputs = {
            "input_directory": input_directory,
            "output_directory": output_directory
        }
        
        # Execute agent
        print(f"Starting document processing for: {input_directory}")
        result = await self.agent.execute(inputs)
        
        if result.status == "completed":
            print(f"Document processing completed successfully!")
            print(f"Results saved to: {output_directory}")
            
            # Extract key outputs
            workflow_outputs = result.outputs
            return {
                "status": "success",
                "processed_documents": workflow_outputs.get("document_ingestion", {}).get("documents", []),
                "report_path": workflow_outputs.get("report_generation", {}).get("report_path"),
                "summary": workflow_outputs.get("summary_generation", {}).get("summary"),
                "categories": workflow_outputs.get("semantic_analysis", {}).get("categories", []),
                "execution_time": result.execution_time
            }
        else:
            print(f"Document processing failed: {result.error}")
            return {
                "status": "error",
                "error": result.error,
                "execution_time": result.execution_time
            }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow."""
        return self.agent.workflow.to_dict()


async def main():
    """Main function for running the document processing agent."""
    parser = argparse.ArgumentParser(description="Document Processing Agent")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing documents to process"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save results (default: input-dir/results)"
    )
    parser.add_argument(
        "--config",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create and run agent
    agent = DocumentProcessingAgent(config)
    
    try:
        result = await agent.process_documents(
            input_directory=args.input_dir,
            output_directory=args.output_dir
        )
        
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"Processed Documents: {len(result['processed_documents'])}")
            print(f"Categories Found: {', '.join(result['categories'])}")
            print(f"Report Path: {result['report_path']}")
            print(f"Execution Time: {result['execution_time']:.2f}s")
            
            if result['summary']:
                print(f"\nSummary:\n{result['summary']}")
        else:
            print(f"Error: {result['error']}")
        
    except Exception as e:
        print(f"Error running document processing agent: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
