"""
Document processing tools for the AI Agent Framework.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import tempfile
from datetime import datetime

# Import for document processing
try:
    import PyPDF2
    import docx
    from PIL import Image
    import easyocr
    import pytesseract
except ImportError as e:
    print(f"Warning: Some document processing dependencies not available: {e}")


class DocumentIngestionTool:
    """Tool for ingesting various document formats."""
    
    def __init__(self, supported_formats: List[str] = None):
        self.supported_formats = supported_formats or [
            'pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'
        ]
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document ingestion."""
        input_dir = inputs.get("input_dir")
        supported_formats = inputs.get("supported_formats", self.supported_formats)
        
        if not input_dir or not os.path.exists(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        
        documents = []
        metadata = []
        
        # Scan directory for supported documents
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower().lstrip('.')
                
                if file_ext in supported_formats:
                    file_info = {
                        'path': file_path,
                        'name': file,
                        'extension': file_ext,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    }
                    
                    documents.append(file_path)
                    metadata.append(file_info)
        
        return {
            "documents": documents,
            "metadata": metadata,
            "total_documents": len(documents)
        }


class OCRTool:
    """Tool for Optical Character Recognition with Intel optimization."""
    
    def __init__(self, engine: str = "easyocr", languages: List[str] = None):
        self.engine = engine
        self.languages = languages or ["en"]
        self._init_ocr_engine()
        
    def _init_ocr_engine(self):
        """Initialize OCR engine."""
        if self.engine == "easyocr":
            try:
                self.reader = easyocr.Reader(self.languages)
            except Exception:
                self.reader = None
        elif self.engine == "tesseract":
            # Configure Tesseract
            self.reader = "tesseract"
        elif self.engine == "intel_optimized":
            # Placeholder for Intel OpenVINO optimized OCR
            self.reader = "intel_optimized"
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute OCR processing."""
        documents = inputs.get("documents", [])
        metadata = inputs.get("metadata", [])
        
        ocr_results = []
        
        for i, doc_path in enumerate(documents):
            doc_metadata = metadata[i] if i < len(metadata) else {}
            file_ext = doc_metadata.get('extension', '').lower()
            
            # Only process image files and PDFs that might need OCR
            if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                try:
                    text = await self._process_image(doc_path)
                    ocr_results.append({
                        'document': doc_path,
                        'text': text,
                        'confidence': 0.9,  # Placeholder
                        'method': 'ocr'
                    })
                except Exception as e:
                    ocr_results.append({
                        'document': doc_path,
                        'text': '',
                        'error': str(e),
                        'method': 'ocr'
                    })
            else:
                # For non-image files, mark as not processed
                ocr_results.append({
                    'document': doc_path,
                    'text': '',
                    'method': 'not_applicable'
                })
        
        return {
            "ocr_results": ocr_results,
            "processed_count": len([r for r in ocr_results if r.get('text')])
        }
    
    async def _process_image(self, image_path: str) -> str:
        """Process a single image with OCR."""
        if self.engine == "easyocr" and self.reader:
            result = self.reader.readtext(image_path)
            return ' '.join([text[1] for text in result])
        
        elif self.engine == "tesseract":
            try:
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                return text
            except Exception:
                return ""
        
        elif self.engine == "intel_optimized":
            # Placeholder for Intel OpenVINO OCR
            # In a real implementation, this would use Intel's optimized models
            return f"[Intel OCR placeholder for {image_path}]"
        
        return ""


class TextExtractionTool:
    """Tool for extracting text from various document formats."""
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text extraction."""
        documents = inputs.get("documents", [])
        ocr_results = inputs.get("ocr_results", [])
        
        extracted_text = []
        
        for i, doc_path in enumerate(documents):
            file_ext = Path(doc_path).suffix.lower().lstrip('.')
            
            try:
                if file_ext == 'pdf':
                    text = await self._extract_pdf_text(doc_path)
                elif file_ext == 'docx':
                    text = await self._extract_docx_text(doc_path)
                elif file_ext == 'txt':
                    text = await self._extract_txt_text(doc_path)
                elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                    # Use OCR results
                    ocr_result = ocr_results[i] if i < len(ocr_results) else {}
                    text = ocr_result.get('text', '')
                else:
                    text = ""
                
                extracted_text.append({
                    'document': doc_path,
                    'text': text,
                    'length': len(text),
                    'extraction_method': file_ext
                })
                
            except Exception as e:
                extracted_text.append({
                    'document': doc_path,
                    'text': '',
                    'error': str(e),
                    'extraction_method': file_ext
                })
        
        return {
            "extracted_text": extracted_text,
            "total_characters": sum(item['length'] for item in extracted_text)
        }
    
    async def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception:
            return ""
    
    async def _extract_docx_text(self, docx_path: str) -> str:
        """Extract text from DOCX."""
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception:
            return ""
    
    async def _extract_txt_text(self, txt_path: str) -> str:
        """Extract text from TXT."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception:
            return ""


class ChunkingTool:
    """Tool for chunking text into smaller segments."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, strategy: str = "fixed"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text chunking."""
        text_data = inputs.get("text_data", [])
        
        all_chunks = []
        
        for text_item in text_data:
            document_path = text_item.get('document', 'unknown')
            text = text_item.get('text', '')
            
            if not text:
                continue
            
            if self.strategy == "fixed":
                chunks = self._fixed_size_chunking(text)
            elif self.strategy == "semantic":
                chunks = self._semantic_chunking(text)
            else:
                chunks = self._fixed_size_chunking(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'document': document_path,
                    'chunk_id': f"{document_path}_{i}",
                    'text': chunk,
                    'index': i,
                    'length': len(chunk)
                })
        
        return {
            "text_chunks": all_chunks,
            "total_chunks": len(all_chunks)
        }
    
    def _fixed_size_chunking(self, text: str) -> List[str]:
        """Fixed size chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
            
            start = end - self.overlap
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """Semantic chunking based on sentences."""
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class SemanticAnalysisTool:
    """Tool for semantic analysis and categorization."""
    
    def __init__(self, model: str = "bert-base", categories: List[str] = None):
        self.model = model
        self.categories = categories or ["general", "technical", "business"]
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic analysis."""
        text_chunks = inputs.get("text_chunks", [])
        
        analysis_results = []
        all_categories = set()
        all_entities = []
        
        for chunk in text_chunks:
            chunk_text = chunk.get('text', '')
            
            # Placeholder analysis - in real implementation would use ML models
            analysis = {
                'chunk_id': chunk.get('chunk_id'),
                'document': chunk.get('document'),
                'category': self._classify_category(chunk_text),
                'sentiment': self._analyze_sentiment(chunk_text),
                'entities': self._extract_entities(chunk_text),
                'topics': self._extract_topics(chunk_text),
                'confidence': 0.85
            }
            
            analysis_results.append(analysis)
            all_categories.add(analysis['category'])
            all_entities.extend(analysis['entities'])
        
        return {
            "analysis_results": analysis_results,
            "categories": list(all_categories),
            "entities": list(set(all_entities)),
            "total_analyzed": len(analysis_results)
        }
    
    def _classify_category(self, text: str) -> str:
        """Classify text category (placeholder implementation)."""
        # Simple keyword-based classification
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['technical', 'api', 'code', 'algorithm']):
            return 'technical'
        elif any(word in text_lower for word in ['business', 'market', 'revenue', 'profit']):
            return 'business'
        elif any(word in text_lower for word in ['legal', 'contract', 'agreement', 'terms']):
            return 'legal'
        else:
            return 'general'
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment (placeholder implementation)."""
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'success']
        negative_words = ['bad', 'poor', 'negative', 'fail', 'problem']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities (placeholder implementation)."""
        # Simple entity extraction
        import re
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        
        # Extract dates (simple pattern)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        
        # Extract capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        entities = emails + dates + proper_nouns[:5]  # Limit proper nouns
        return entities
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics (placeholder implementation)."""
        # Simple topic extraction based on frequent words
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Only consider longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 3 most frequent words as topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        return [topic[0] for topic in topics]


class SummaryTool:
    """Tool for generating summaries."""
    
    def __init__(self, model: str = "summarizer", max_length: int = 200, min_length: int = 50):
        self.model = model
        self.max_length = max_length
        self.min_length = min_length
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summary generation."""
        text_data = inputs.get("text_data", [])
        analysis = inputs.get("analysis", [])
        
        # Combine all text
        all_text = ""
        for text_item in text_data:
            all_text += text_item.get('text', '') + "\n"
        
        # Generate summary (placeholder implementation)
        summary = self._generate_summary(all_text)
        
        # Generate insights from analysis
        insights = self._generate_insights(analysis)
        
        return {
            "summary": summary,
            "insights": insights,
            "length": len(summary)
        }
    
    def _generate_summary(self, text: str) -> str:
        """Generate text summary (placeholder implementation)."""
        if not text:
            return "No content to summarize."
        
        # Simple extractive summarization - take first few sentences
        sentences = text.split('. ')
        
        # Filter out very short sentences
        sentences = [s for s in sentences if len(s) > 20]
        
        # Take first few sentences up to max length
        summary = ""
        for sentence in sentences[:5]:  # Max 5 sentences
            if len(summary) + len(sentence) < self.max_length:
                summary += sentence + ". "
            else:
                break
        
        if len(summary) < self.min_length and sentences:
            # If too short, add more content
            summary = sentences[0] + ". "
        
        return summary.strip()
    
    def _generate_insights(self, analysis: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from analysis."""
        if not analysis:
            return []
        
        insights = []
        
        # Category distribution
        categories = [item.get('category', 'unknown') for item in analysis]
        category_count = {}
        for cat in categories:
            category_count[cat] = category_count.get(cat, 0) + 1
        
        if category_count:
            most_common = max(category_count.items(), key=lambda x: x[1])
            insights.append(f"Most common category: {most_common[0]} ({most_common[1]} instances)")
        
        # Sentiment analysis
        sentiments = [item.get('sentiment', 'neutral') for item in analysis]
        sentiment_count = {}
        for sent in sentiments:
            sentiment_count[sent] = sentiment_count.get(sent, 0) + 1
        
        if sentiment_count:
            dominant_sentiment = max(sentiment_count.items(), key=lambda x: x[1])
            insights.append(f"Dominant sentiment: {dominant_sentiment[0]}")
        
        return insights


class ReportGeneratorTool:
    """Tool for generating comprehensive reports."""
    
    def __init__(self, template: str = "basic", output_format: str = "html"):
        self.template = template
        self.output_format = output_format
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report generation."""
        documents = inputs.get("documents", [])
        metadata = inputs.get("metadata", [])
        analysis = inputs.get("analysis", [])
        summary = inputs.get("summary", "")
        categories = inputs.get("categories", [])
        entities = inputs.get("entities", [])
        
        # Generate report content
        report_content = self._generate_report(
            documents, metadata, analysis, summary, categories, entities
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"document_analysis_report_{timestamp}.{self.output_format}"
        
        # Use temporary directory for now - in real implementation would use proper output dir
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=f'.{self.output_format}', 
            delete=False
        ) as f:
            f.write(report_content)
            report_path = f.name
        
        return {
            "report": report_content,
            "report_path": report_path,
            "report_filename": report_filename,
            "format": self.output_format
        }
    
    def _generate_report(
        self, 
        documents: List[str], 
        metadata: List[Dict], 
        analysis: List[Dict], 
        summary: str,
        categories: List[str],
        entities: List[str]
    ) -> str:
        """Generate report content."""
        
        if self.output_format == "html":
            return self._generate_html_report(
                documents, metadata, analysis, summary, categories, entities
            )
        else:
            return self._generate_text_report(
                documents, metadata, analysis, summary, categories, entities
            )
    
    def _generate_html_report(
        self, 
        documents: List[str], 
        metadata: List[Dict], 
        analysis: List[Dict], 
        summary: str,
        categories: List[str],
        entities: List[str]
    ) -> str:
        """Generate HTML report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Document Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{summary or 'No summary available.'}</p>
            </div>
            
            <div class="section">
                <h2>Overview</h2>
                <div class="metric">Total Documents Processed: {len(documents)}</div>
                <div class="metric">Categories Found: {', '.join(categories) if categories else 'None'}</div>
                <div class="metric">Total Entities Extracted: {len(entities)}</div>
            </div>
            
            <div class="section">
                <h2>Document List</h2>
                <table>
                    <tr><th>Document</th><th>Size</th><th>Type</th></tr>
        """
        
        for i, doc in enumerate(documents):
            meta = metadata[i] if i < len(metadata) else {}
            size = meta.get('size', 0)
            doc_type = meta.get('extension', 'unknown')
            doc_name = os.path.basename(doc)
            
            html += f"<tr><td>{doc_name}</td><td>{size} bytes</td><td>{doc_type}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Analysis Results</h2>
        """
        
        if analysis:
            # Category distribution
            category_count = {}
            for item in analysis:
                cat = item.get('category', 'unknown')
                category_count[cat] = category_count.get(cat, 0) + 1
            
            html += "<h3>Category Distribution</h3><ul>"
            for cat, count in category_count.items():
                html += f"<li>{cat}: {count} chunks</li>"
            html += "</ul>"
        
        if entities:
            html += f"<h3>Key Entities</h3><p>{', '.join(entities[:20])}</p>"  # Show first 20
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_text_report(
        self, 
        documents: List[str], 
        metadata: List[Dict], 
        analysis: List[Dict], 
        summary: str,
        categories: List[str],
        entities: List[str]
    ) -> str:
        """Generate text report."""
        
        report = f"""
DOCUMENT ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
{summary or 'No summary available.'}

OVERVIEW
- Total Documents Processed: {len(documents)}
- Categories Found: {', '.join(categories) if categories else 'None'}
- Total Entities Extracted: {len(entities)}

DOCUMENT LIST
"""
        
        for i, doc in enumerate(documents):
            meta = metadata[i] if i < len(metadata) else {}
            size = meta.get('size', 0)
            doc_type = meta.get('extension', 'unknown')
            doc_name = os.path.basename(doc)
            
            report += f"- {doc_name} ({doc_type}, {size} bytes)\n"
        
        if entities:
            report += f"\nKEY ENTITIES\n{', '.join(entities[:20])}\n"  # Show first 20
        
        return report
