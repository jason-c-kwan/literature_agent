import pytest
import json
from unittest.mock import patch, MagicMock, call
import sys
import io # For capturing stdout
import os

# Add the project root to sys.path to allow for absolute imports from cli
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now the import should work
from cli.parse_pdf import main as parse_pdf_main
from cli.parse_pdf import parse_page_numbers


# Helper to run the main script with arguments and capture stdout
def run_script(args_list):
    # Patch sys.argv and run the main function
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Mock sys.exit to prevent tests from exiting
    with patch.object(sys, 'argv', ['parse_pdf.py'] + args_list), \
         patch('sys.exit') as mock_exit:
        try:
            parse_pdf_main()
        except SystemExit as e: # Argparse calls sys.exit on error or --help
            # Allow SystemExit if it's from argparse (e.g. on --help or error)
            # but we want to check mock_exit.called for actual exits from our code
            pass

    sys.stdout = sys.__stdout__  # Restore stdout
    return captured_output.getvalue(), mock_exit

# --- Tests for parse_page_numbers ---

def test_parse_page_numbers_empty_input():
    assert parse_page_numbers(None, 10) is None
    assert parse_page_numbers("", 10) is None
    assert parse_page_numbers("   ", 10) is None
    assert parse_page_numbers(", ,, ", 10) is None # Should also result in all pages

def test_parse_page_numbers_single_pages():
    assert parse_page_numbers("0", 10) == [0]
    assert parse_page_numbers("0,2,4", 10) == [0, 2, 4]
    assert parse_page_numbers(" 1 , 3 ", 10) == [1, 3]

def test_parse_page_numbers_ranges():
    assert parse_page_numbers("0-2", 10) == [0, 1, 2]
    assert parse_page_numbers("0-0", 10) == [0]
    assert parse_page_numbers("1-3,5-6", 10) == [1, 2, 3, 5, 6]

def test_parse_page_numbers_with_n():
    assert parse_page_numbers("N", 5) == [4] # 0-indexed last page
    assert parse_page_numbers("N-1", 5) == [3] # (5-1)-1 = 3
    assert parse_page_numbers("0-N", 5) == [0, 1, 2, 3, 4]
    assert parse_page_numbers("N,0,N-2", 5) == [0, 2, 4] # sorted: 0, (5-1)-2=2, (5-1)=4

def test_parse_page_numbers_mixed():
    assert parse_page_numbers("0,2-4,N", 10) == [0, 2, 3, 4, 9]

def test_parse_page_numbers_out_of_order_duplicates():
    # For a 5-page doc (indices 0-4, N=4)
    # "3,0-1,N,1" -> 3, 0, 1, 4, 1 -> sorted unique [0, 1, 3, 4]
    assert parse_page_numbers("3,0-1,N,1", 5) == [0, 1, 3, 4]

def test_parse_page_numbers_invalid_format():
    with pytest.raises(ValueError, match="Invalid page range format: '1-'"):
        parse_page_numbers("1-", 10)
    with pytest.raises(ValueError, match="Invalid page number format: 'abc'"):
        parse_page_numbers("abc", 10)
    with pytest.raises(ValueError, match=r"Invalid page range format: '1-N-2'"): 
        parse_page_numbers("1-N-2", 10) 
    # Test for range where start > end after N resolution
    with pytest.raises(ValueError, match=r"Page range '1-N' \(resolved to '1-0'\) is invalid for a document with 1 pages"):
         parse_page_numbers("1-N", 1)

def test_parse_page_numbers_page_out_of_bounds():
    with pytest.raises(ValueError, match=r"Page number '10' \(resolved to '10'\) is out of range"):
        parse_page_numbers("10", 10)
    with pytest.raises(ValueError, match=r"Page range '8-10' \(resolved to '8-10'\) is invalid"): 
        parse_page_numbers("8-10", 10)
    # For input "-1", it's treated as a range "", "1", int("") fails.
    with pytest.raises(ValueError, match=r"Invalid page range format: '-1'\. Must be like 'start-end'\."):
        parse_page_numbers("-1", 10)
    with pytest.raises(ValueError, match="Cannot use 'N' for a 0-page document."):
        parse_page_numbers("N", 0)

# --- Tests for main script functionality ---

@patch('cli.parse_pdf.pymupdf.open')
@patch('cli.parse_pdf.to_markdown')
def test_markdown_output_default(mock_to_markdown, mock_pymupdf_open):
    # Mock pymupdf.open()
    mock_doc = MagicMock()
    mock_doc.page_count = 2
    mock_pymupdf_open.return_value = mock_doc

    # Mock pymupdf4llm.to_markdown()
    mock_to_markdown.return_value = "# Header 1\n\nThis is **bold** and *italic* text.\n## Header 2\nAnother paragraph."
    
    # Run the script
    # Input PDF path is a dummy since pymupdf.open is mocked
    output, _ = run_script(['dummy.pdf'])
    
    # Expected plain text output
    # BeautifulSoup get_text with separator='\\n' and strip=True
    # "# Header 1" -> "Header 1"
    # "This is **bold** and *italic* text." -> "This is bold and italic text."
    # "## Header 2" -> "Header 2"
    # "Another paragraph." -> "Another paragraph."
    expected_output = "Header 1\nThis is bold and italic text.\nHeader 2\nAnother paragraph.\n" # Expected final output with newlines
    
    # The script's final output logic for markdown adds a newline if not present,
    # or if empty, outputs one newline.
    # Here, plain_text will end with "paragraph.", so a \n is added.
    assert output.strip() == expected_output.strip() # Compare stripped versions to handle trailing newlines consistently
    mock_pymupdf_open.assert_called_once_with('dummy.pdf')
    mock_to_markdown.assert_called_once_with(
        mock_doc,
        pages=None, # Default is all pages
        ignore_images=True,
        ignore_graphics=True,
        table_strategy=None,
        page_chunks=False
    )
    mock_doc.close.assert_called_once()


@patch('cli.parse_pdf.pymupdf.open')
@patch('cli.parse_pdf.to_markdown')
def test_json_output(mock_to_markdown, mock_pymupdf_open):
    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_pymupdf_open.return_value = mock_doc
    
    mock_to_markdown.return_value = "# First Header\n\nSome paragraph text.\n\n## Second Header (Sub)\n\nAnother *paragraph* with [a link](http://example.com)."
    
    output, _ = run_script(['dummy.pdf', '--format', 'json'])
    
    expected_json_structure = [
        {"type": "heading", "level": 1, "text": "First Header"},
        {"type": "paragraph", "text": "Some paragraph text."},
        {"type": "heading", "level": 2, "text": "Second Header (Sub)"},
        {"type": "paragraph", "text": "Another paragraph with a link."} # BS should strip link markup
    ]
    
    # The output includes a trailing newline, json.loads should handle it or strip it.
    loaded_output = json.loads(output.strip())
    assert loaded_output == expected_json_structure
    mock_doc.close.assert_called_once()

@patch('cli.parse_pdf.pymupdf.open')
@patch('cli.parse_pdf.to_markdown')
def test_page_selection(mock_to_markdown, mock_pymupdf_open):
    mock_doc = MagicMock()
    mock_doc.page_count = 5
    mock_pymupdf_open.return_value = mock_doc
    mock_to_markdown.return_value = "Content from selected pages" # Dummy content

    run_script(['dummy.pdf', '--pages', '0,2-3,N']) # N for 5 pages is 4
    
    mock_to_markdown.assert_called_once_with(
        mock_doc,
        pages=[0, 2, 3, 4], # Expected parsed and sorted page list
        ignore_images=True,
        ignore_graphics=True,
        table_strategy=None,
        page_chunks=False
    )
    mock_doc.close.assert_called_once()

@patch('cli.parse_pdf.pymupdf.open')
@patch('cli.parse_pdf.to_markdown')
def test_empty_pdf_content_markdown(mock_to_markdown, mock_pymupdf_open):
    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_pymupdf_open.return_value = mock_doc
    mock_to_markdown.return_value = "" # Empty markdown
    
    output, _ = run_script(['dummy.pdf', '--format', 'markdown'])
    assert output == "\n" # Script now adds a newline for empty output

    mock_to_markdown.return_value = "   \n   " # Actual newline for Whitespace markdown
    output, _ = run_script(['dummy.pdf', '--format', 'markdown'])
    # After markdown to html and then get_text(strip=True), "   \n   " becomes ""
    # Then script adds a newline.
    assert output == "\n"

@patch('cli.parse_pdf.pymupdf.open')
@patch('cli.parse_pdf.to_markdown')
def test_empty_pdf_content_json(mock_to_markdown, mock_pymupdf_open):
    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_pymupdf_open.return_value = mock_doc
    mock_to_markdown.return_value = "" # Empty markdown
    
    output, _ = run_script(['dummy.pdf', '--format', 'json'])
    loaded_output = json.loads(output.strip())
    assert loaded_output == []

    mock_to_markdown.return_value = "  \n  " # Actual newline
    output, _ = run_script(['dummy.pdf', '--format', 'json'])
    loaded_output = json.loads(output.strip())
    assert loaded_output == [] # Should be empty as "  \n  ".strip() is empty


@patch('cli.parse_pdf.pymupdf.open')
def test_pdf_open_error(mock_pymupdf_open):
    mock_pymupdf_open.side_effect = Exception("Failed to open PDF")
    
    # Capture stderr
    captured_stderr = io.StringIO()
    sys.stderr = captured_stderr
    
    output, mock_exit = run_script(['nonexistent.pdf'])
    
    sys.stderr = sys.__stderr__ # Restore stderr
    
    assert "Error opening PDF 'nonexistent.pdf': Failed to open PDF" in captured_stderr.getvalue()
    mock_exit.assert_called_once_with(1)


@patch('cli.parse_pdf.pymupdf.open')
@patch('cli.parse_pdf.to_markdown')
def test_markdown_conversion_error(mock_to_markdown, mock_pymupdf_open):
    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_pymupdf_open.return_value = mock_doc
    mock_to_markdown.side_effect = Exception("Markdown conversion failed")

    captured_stderr = io.StringIO()
    sys.stderr = captured_stderr

    output, mock_exit = run_script(['dummy.pdf'])

    sys.stderr = sys.__stderr__
    
    assert "Error during Markdown conversion: Markdown conversion failed" in captured_stderr.getvalue()
    mock_exit.assert_called_once_with(1)
    mock_doc.close.assert_called_once() # Ensure doc is closed even on error

def test_invalid_pages_argument_error():
    # This will be caught by argparse or our validator, leading to sys.exit
    # We need to mock pymupdf.open because it's called before page parsing validation for total_pages
    with patch('cli.parse_pdf.pymupdf.open') as mock_open:
        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_open.return_value = mock_doc

        captured_stderr = io.StringIO()
        sys.stderr = captured_stderr

        output, mock_exit = run_script(['dummy.pdf', '--pages', '1,abc,N'])
        
        sys.stderr = sys.__stderr__
        
        assert "Error in --pages argument: Invalid page number format: 'abc'." in captured_stderr.getvalue()
        mock_exit.assert_called_once_with(1)
        mock_doc.close.assert_called_once()


# Test for 0 page document
@patch('cli.parse_pdf.pymupdf.open')
@patch('cli.parse_pdf.to_markdown')
def test_zero_page_document(mock_to_markdown, mock_pymupdf_open):
    mock_doc = MagicMock()
    mock_doc.page_count = 0
    mock_pymupdf_open.return_value = mock_doc
    mock_to_markdown.return_value = "" # Assume it handles 0 pages by returning empty

    captured_stderr = io.StringIO()
    sys.stderr = captured_stderr

    output, _ = run_script(['zeropage.pdf', '--pages', '0']) # Requesting page 0 of 0-page doc
    
    sys.stderr = sys.__stderr__

    # parse_page_numbers will raise "Cannot use 'N' for page numbers..." if 'N' is used.
    # If specific page like '0' is used for a 0-page doc, it will raise "Page number '0' (resolved to '0') is out of range..."
    # The main script catches this ValueError from parse_page_numbers.
    expected_error_msg_part = "Error in --pages argument: Page number '0' (resolved to '0') is out of range"
    assert expected_error_msg_part in captured_stderr.getvalue()
    
    # Re-run to check exit status properly
    with patch('cli.parse_pdf.pymupdf.open', return_value=mock_doc) as mock_open_again, \
         patch('cli.parse_pdf.to_markdown', return_value="") as mock_to_markdown_again:
        
        captured_stderr_again = io.StringIO()
        sys.stderr = captured_stderr_again
        
        _, mock_exit_again = run_script(['zeropage.pdf', '--pages', '0'])
        
        sys.stderr = sys.__stderr__
        
        assert expected_error_msg_part in captured_stderr_again.getvalue()
        mock_exit_again.assert_called_with(1)
    
    # Test 0-page doc with no --pages (should process "all" 0 pages)
    with patch('cli.parse_pdf.pymupdf.open', return_value=mock_doc) as mock_open_again, \
         patch('cli.parse_pdf.to_markdown', return_value="") as mock_to_markdown_again:
        
        output_all_pages, _ = run_script(['zeropage.pdf'])
        assert output_all_pages == "\n" # Script outputs a newline for empty content
        mock_to_markdown_again.assert_called_with(mock_doc, pages=None, ignore_images=True, ignore_graphics=True, table_strategy=None, page_chunks=False)
