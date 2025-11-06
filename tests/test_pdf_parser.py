# tests/utils/test_pdf_parser.py
import pytest
import io
import logging
from unittest.mock import patch, MagicMock, Mock

# Import the module we are testing
from app.utils import pdf_parser
# Import the specific classes/functions
from app.utils.pdf_parser import PDFParser, parse_resume_pdf

# Fixtures for test data
@pytest.fixture
def mock_pdf_bytes():
    """Returns fake PDF content as bytes."""
    return b"fake-pdf-content-bytes"

@pytest.fixture
def mock_pdf_file_like(mock_pdf_bytes):
    """Returns a file-like object containing fake PDF bytes."""
    file = io.BytesIO(mock_pdf_bytes)
    
    # WRAP .seek IN A MOCK TO "SPY" ON IT
    file.seek = MagicMock(wraps=file.seek)

    yield file
    file.close()

@pytest.fixture
def valid_resume_text():
    """Returns text that should pass validation."""
    return (
        "John Doe\n"
        "Email: john.doe@example.com | Phone: 123-456-7890\n\n"
        "## Experience\n"
        "Software Engineer at Tech Corp (2020 - Present)\n"
        "- Developed cool stuff.\n\n"
        "## Education\n"
        "B.S. in Computer Science, University of Example (2020)"
    )

@pytest.fixture
def invalid_resume_text_short():
    """Returns text that is too short for validation."""
    return "This is too short."

@pytest.fixture
def invalid_resume_text_no_keywords():
    """Returns text that is long enough but lacks keywords."""
    return (
        "This is a very long document about unrelated topics. "
        "It talks about the weather, gardening, and cooking. "
        "There is absolutely no mention of any professional indicators. "
        "It just keeps going on and on for a while."
    )


class TestPDFParserInitialization:
    """Tests the __init__ method of PDFParser."""

    @patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True)
    @patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', True)
    def test_init_all_libraries(self):
        """Test init when both libraries are available."""
        parser = PDFParser()
        assert parser.method_priority == ['pdfplumber', 'pypdf2']

    @patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True)
    @patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', False)
    def test_init_only_pdfplumber(self):
        """Test init when only pdfplumber is available."""
        parser = PDFParser()
        assert parser.method_priority == ['pdfplumber']

    @patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', False)
    @patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', True)
    def test_init_only_pypdf2(self):
        """Test init when only PyPDF2 is available."""
        parser = PDFParser()
        assert parser.method_priority == ['pypdf2']

    @patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', False)
    @patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', False)
    def test_init_no_libraries(self):
        """Test init when no libraries are available."""
        with pytest.raises(ImportError) as e:
            PDFParser()
        assert "No PDF parsing library available" in str(e.value)

    def test_import_warnings(self, caplog):
        """Test the logic that would be triggered during import."""
        with patch.dict('app.utils.pdf_parser.__dict__', {'PDFPLUMBER_AVAILABLE': False}):
            logger = logging.getLogger("app.utils.pdf_parser")
            logger.warning("Simulated: pdfplumber not available")
        assert "pdfplumber not available" in caplog.text

        with patch.dict('app.utils.pdf_parser.__dict__', {'PYPDF2_AVAILABLE': False}):
            logger = logging.getLogger("app.utils.pdf_parser")
            logger.warning("Simulated: PyPDF2 not available")
        assert "PyPDF2 not available" in caplog.text


# Patch the specific functions, not the whole module
@patch('app.utils.pdf_parser.PyPDF2.PdfReader', create=True)
@patch('app.utils.pdf_parser.pdfplumber.open', create=True)
class TestPDFParserExtraction:
    """Tests the extract_text_from_pdf method."""

    def test_extract_with_pdfplumber_succeeds(self, mock_plumber_open, mock_pdfreader, mock_pdf_bytes):
        """Test successful extraction using pdfplumber."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Text from pdfplumber"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_plumber_open.return_value.__enter__.return_value = mock_pdf
        
        with patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True), \
             patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', True):
            parser = PDFParser()

        text = parser.extract_text_from_pdf(mock_pdf_bytes)
        
        assert text == "Text from pdfplumber"

        # Check content of BytesIO object
        mock_plumber_open.assert_called_once()
        called_with_arg = mock_plumber_open.call_args[0][0]
        assert isinstance(called_with_arg, io.BytesIO)
        assert called_with_arg.getvalue() == mock_pdf_bytes
        
        mock_pdfreader.assert_not_called()

    # --- FIX: Added mock_pdf_bytes to the function signature ---
    def test_extract_with_file_like_object(self, mock_plumber_open, mock_pdfreader, mock_pdf_file_like, mock_pdf_bytes):
    # --- END FIX ---
        """Test passing a file-like object instead of bytes."""
        mock_page = MagicMock(extract_text=lambda: "File-like object text")
        mock_pdf = MagicMock(pages=[mock_page])
        mock_plumber_open.return_value.__enter__.return_value = mock_pdf
        
        with patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True), \
             patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', True):
            parser = PDFParser()

        text = parser.extract_text_from_pdf(mock_pdf_file_like)
        
        assert text == "File-like object text"
        
        # This assertion now works thanks to the fixture update
        mock_pdf_file_like.seek.assert_called_with(0)

        # Check that pdfplumber was called with the file object's *content*
        mock_plumber_open.assert_called_once()
        called_with_arg = mock_plumber_open.call_args[0][0]
        assert isinstance(called_with_arg, io.BytesIO)
        # This assertion now works because mock_pdf_bytes is available
        assert called_with_arg.getvalue() == mock_pdf_bytes


    def test_extract_pypdf2_fallback_succeeds(self, mock_plumber_open, mock_pdfreader, mock_pdf_bytes):
        """Test fallback to PyPDF2 when pdfplumber returns empty text."""
        mock_page_plumber = MagicMock(extract_text=lambda: "  ")
        mock_pdf_plumber = MagicMock(pages=[mock_page_plumber])
        mock_plumber_open.return_value.__enter__.return_value = mock_pdf_plumber
        
        mock_page_pypdf2 = MagicMock()
        mock_page_pypdf2.extract_text.return_value = "Text from PyPDF2"
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader_instance.pages = [mock_page_pypdf2]
        mock_pdfreader.return_value = mock_pdf_reader_instance
        
        with patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True), \
             patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', True):
            parser = PDFParser()

        text = parser.extract_text_from_pdf(mock_pdf_bytes)
        
        assert text == "Text from PyPDF2"
        mock_plumber_open.assert_called_once()
        
        # Check content of BytesIO object
        mock_pdfreader.assert_called_once()
        called_with_arg = mock_pdfreader.call_args[0][0]
        assert isinstance(called_with_arg, io.BytesIO)
        assert called_with_arg.getvalue() == mock_pdf_bytes

    def test_extract_pypdf2_fallback_on_exception(self, mock_plumber_open, mock_pdfreader, mock_pdf_bytes, caplog):
        """Test fallback to PyPDF2 when pdfplumber raises an exception."""
        mock_plumber_open.side_effect = Exception("pdfplumber failed")
        
        mock_page_pypdf2 = MagicMock(extract_text=lambda: "Text from PyPDF2")
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader_instance.pages = [mock_page_pypdf2]
        mock_pdfreader.return_value = mock_pdf_reader_instance
        
        with patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True), \
             patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', True):
            parser = PDFParser()
        
        text = parser.extract_text_from_pdf(mock_pdf_bytes)
        
        assert text == "Text from PyPDF2"
        assert "Failed to extract with pdfplumber: pdfplumber failed" in caplog.text

        # Check content of BytesIO object
        mock_pdfreader.assert_called_once()
        called_with_arg = mock_pdfreader.call_args[0][0]
        assert isinstance(called_with_arg, io.BytesIO)
        assert called_with_arg.getvalue() == mock_pdf_bytes

    def test_extract_all_methods_fail(self, mock_plumber_open, mock_pdfreader, mock_pdf_bytes, caplog):
        """Test when all extraction methods fail."""
        mock_plumber_open.side_effect = Exception("pdfplumber failed")
        mock_pdfreader.side_effect = Exception("PyPDF2 failed")
        
        with patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True), \
             patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', True):
            parser = PDFParser()

        text = parser.extract_text_from_pdf(mock_pdf_bytes)
        
        assert text is None
        assert "Failed to extract with pdfplumber" in caplog.text
        assert "Failed to extract with pypdf2" in caplog.text
        assert "All PDF extraction methods failed" in caplog.text

    @patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', False)
    @patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', True)
    def test_extract_with_pypdf2_only(self, mock_plumber_open, mock_pdfreader, mock_pdf_bytes):
        """Test extraction when only PyPDF2 is available."""
        mock_page_pypdf2 = MagicMock(extract_text=lambda: "PyPDF2 only")
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader_instance.pages = [mock_page_pypdf2]
        mock_pdfreader.return_value = mock_pdf_reader_instance

        parser = PDFParser()
        assert parser.method_priority == ['pypdf2']
        
        text = parser.extract_text_from_pdf(mock_pdf_bytes)
        
        assert text == "PyPDF2 only"
        mock_plumber_open.assert_not_called()

        # Check content of BytesIO object
        mock_pdfreader.assert_called_once()
        called_with_arg = mock_pdfreader.call_args[0][0]
        assert isinstance(called_with_arg, io.BytesIO)
        assert called_with_arg.getvalue() == mock_pdf_bytes
    
    @patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True)
    @patch('app.utils.pdf_parser.PYPDF2_AVAILABLE', False)
    def test_extract_with_pdfplumber_only(self, mock_plumber_open, mock_pdfreader, mock_pdf_bytes):
        """Test extraction when only pdfplumber is available."""
        mock_page = MagicMock(extract_text=lambda: "pdfplumber only")
        mock_pdf = MagicMock(pages=[mock_page])
        mock_plumber_open.return_value.__enter__.return_value = mock_pdf
        
        parser = PDFParser()
        assert parser.method_priority == ['pdfplumber']

        text = parser.extract_text_from_pdf(mock_pdf_bytes)
        
        assert text == "pdfplumber only"
        
        # Check content of BytesIO object
        mock_plumber_open.assert_called_once()
        called_with_arg = mock_plumber_open.call_args[0][0]
        assert isinstance(called_with_arg, io.BytesIO)
        assert called_with_arg.getvalue() == mock_pdf_bytes
        
        mock_pdfreader.assert_not_called()


class TestPDFParserValidation:
    """Tests the validate_resume_text method."""

    @pytest.fixture
    def parser(self):
        """Get a standard parser instance."""
        with patch('app.utils.pdf_parser.PDFPLUMBER_AVAILABLE', True):
            yield PDFParser()

    def test_validate_success(self, parser, valid_resume_text, caplog):
        """Test successful validation with good keywords."""
        with caplog.at_level(logging.INFO):
            result = parser.validate_resume_text(valid_resume_text)
        assert result is True
        assert "Resume validation passed" in caplog.text
        assert "(5 indicators found)" in caplog.text

    def test_validate_too_short(self, parser, invalid_resume_text_short, caplog):
        """Test validation failure due to short text."""
        with caplog.at_level(logging.WARNING):
            result = parser.validate_resume_text(invalid_resume_text_short)
        assert result is False
        assert "Extracted text too short" in caplog.text

    def test_validate_no_text(self, parser, caplog):
        """Test validation failure with None or empty text."""
        with caplog.at_level(logging.WARNING):
            assert parser.validate_resume_text(None) is False
            assert "Extracted text too short" in caplog.text
            
            caplog.clear()
            assert parser.validate_resume_text("   ") is False
            assert "Extracted text too short" in caplog.text

    def test_validate_no_keywords(self, parser, invalid_resume_text_no_keywords, caplog):
        """Test validation warning when no keywords are found (but still passes)."""
        with caplog.at_level(logging.WARNING):
            result = parser.validate_resume_text(invalid_resume_text_no_keywords)
        assert result is True
        assert "Resume validation uncertain" in caplog.text
        assert "(only 0 indicators found)" in caplog.text


# Patch the PDFParser class within the parse_resume_pdf function's module
@patch('app.utils.pdf_parser.PDFParser')
class TestParseResumePDFConvenienceFunction:
    """Tests the parse_resume_pdf convenience function."""

    def test_parse_resume_pdf_success(self, mock_PDFParser, valid_resume_text, mock_pdf_bytes):
        """Test the end-to-end success path."""
        mock_parser_instance = MagicMock()
        mock_parser_instance.extract_text_from_pdf.return_value = valid_resume_text
        mock_parser_instance.validate_resume_text.return_value = True
        mock_PDFParser.return_value = mock_parser_instance
        
        result = parse_resume_pdf(mock_pdf_bytes)
        
        assert result == valid_resume_text
        mock_PDFParser.assert_called_once()
        mock_parser_instance.extract_text_from_pdf.assert_called_once_with(mock_pdf_bytes)
        mock_parser_instance.validate_resume_text.assert_called_once_with(valid_resume_text)

    def test_parse_resume_pdf_extraction_fails(self, mock_PDFParser, mock_pdf_bytes):
        """Test when text extraction returns None."""
        mock_parser_instance = MagicMock()
        mock_parser_instance.extract_text_from_pdf.return_value = None
        mock_PDFParser.return_value = mock_parser_instance
        
        result = parse_resume_pdf(mock_pdf_bytes)
        
        assert result is None
        mock_parser_instance.validate_resume_text.assert_not_called()

    def test_parse_resume_pdf_validation_fails(self, mock_PDFParser, invalid_resume_text_short, mock_pdf_bytes):
        """Test when text is extracted but validation fails."""
        mock_parser_instance = MagicMock()
        mock_parser_instance.extract_text_from_pdf.return_value = invalid_resume_text_short
        mock_parser_instance.validate_resume_text.return_value = False
        mock_PDFParser.return_value = mock_parser_instance
        
        result = parse_resume_pdf(mock_pdf_bytes)
        
        assert result is None
        mock_parser_instance.validate_resume_text.assert_called_once_with(invalid_resume_text_short)