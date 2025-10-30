# app/utils/pdf_parser.py
"""
PDF Parser Utility for Interview Bot
Extracts text content from PDF resumes
"""
import logging
from typing import Optional
import io

# Try multiple PDF parsing libraries for better compatibility
logger = logging.getLogger(__name__)

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")


class PDFParser:
    """Handles PDF text extraction with fallback methods."""
    
    def __init__(self):
        """Initialize PDF parser with available libraries."""
        if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "No PDF parsing library available. Install with: "
                "pip install PyPDF2 pdfplumber"
            )
        
        self.method_priority = []
        if PDFPLUMBER_AVAILABLE:
            self.method_priority.append('pdfplumber')
        if PYPDF2_AVAILABLE:
            self.method_priority.append('pypdf2')
        
        logger.info(f"PDF parser initialized with methods: {', '.join(self.method_priority)}")
    
    def extract_text_from_pdf(self, pdf_file) -> Optional[str]:
        """
        Extract text from PDF file using available methods.
        
        Args:
            pdf_file: File-like object or bytes from uploaded PDF
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        # Convert to bytes if needed
        if hasattr(pdf_file, 'read'):
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)  # Reset file pointer
        else:
            pdf_bytes = pdf_file
        
        # Try each method in priority order
        for method in self.method_priority:
            try:
                if method == 'pdfplumber':
                    text = self._extract_with_pdfplumber(pdf_bytes)
                elif method == 'pypdf2':
                    text = self._extract_with_pypdf2(pdf_bytes)
                
                if text and len(text.strip()) > 0:
                    logger.info(f"Successfully extracted {len(text)} characters using {method}")
                    return text.strip()
                else:
                    logger.warning(f"{method} returned empty text, trying next method...")
                    
            except Exception as e:
                logger.warning(f"Failed to extract with {method}: {e}")
                continue

        logger.error("All PDF extraction methods failed")
        return None
    
    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> str:
        """Extract text using pdfplumber (generally more accurate)."""
        if not PDFPLUMBER_AVAILABLE:
            return None
        
        import pdfplumber
        
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    logger.debug(f"Extracted text from page {page_num}")
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_bytes: bytes) -> str:
        """Extract text using PyPDF2 (fallback method)."""
        if not PYPDF2_AVAILABLE:
            return None
        
        import PyPDF2
        
        text_parts = []
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
                logger.debug(f"Extracted text from page {page_num + 1}")
        
        return "\n\n".join(text_parts)
    
    def validate_resume_text(self, text: str) -> bool:
        """
        Validates that extracted text looks like a resume.
        
        Args:
            text: Extracted text to validate
            
        Returns:
            True if text appears to be a valid resume
        """
        if not text or len(text.strip()) < 50:
            logger.warning("Extracted text too short to be a valid resume")
            return False
        
        # Check for common resume keywords
        resume_indicators = [
            'experience', 'education', 'skills', 'work', 'employment',
            'university', 'college', 'degree', 'email', 'phone',
            'project', 'achievement', 'responsibility', 'position'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for keyword in resume_indicators if keyword in text_lower)
        
        if matches >= 2:
            logger.info(f"Resume validation passed ({matches} indicators found)")
            return True
        else:
            logger.warning(f"Resume validation uncertain (only {matches} indicators found)")
            return True  # Still allow but log warning


def parse_resume_pdf(pdf_file) -> Optional[str]:
    """
    Convenience function to extract text from PDF resume.
    
    Args:
        pdf_file: Uploaded PDF file
        
    Returns:
        Extracted text or None
    """
    parser = PDFParser()
    text = parser.extract_text_from_pdf(pdf_file)
    
    if text and parser.validate_resume_text(text):
        return text
    
    return None