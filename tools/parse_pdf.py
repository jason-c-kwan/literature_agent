import argparse
import argparse
import json
import sys
import pymupdf # For pymupdf.open
from pymupdf4llm.helpers.pymupdf_rag import to_markdown
from bs4 import BeautifulSoup
import re
import markdown # For converting Markdown to HTML
from typing import Optional # Added Optional
import os # Moved import os to the top


def parse_page_numbers(page_str: Optional[str], total_pages: int) -> Optional[list[int]]: # Added Optional to arg and return
    """
    Parses a comma-separated page string (e.g., "0,1,5-7,10-N") into a list of 0-based page indices.
    'N' represents the last page.
    Returns None if page_str is None or empty, indicating all pages.
    Raises ValueError for invalid formats or out-of-range pages.
    """
    if not page_str:
        return None  # All pages

    indices = set()
    # Replace 'N' with the actual last page index (0-based)
    # Ensure N is treated as a number, so N-1 becomes (total_pages-1)-1
    # A simple string replace for N should be done carefully if N can be part of a number like "N1"
    # For "N", it means total_pages - 1. For "N-1", it means (total_pages-1) - 1.
    # It's safer to parse N within the loop.

    parts = page_str.split(',')
    for raw_part in parts:
        part_input_str = raw_part.strip() # Keep original for error messages if needed
        if not part_input_str:
            continue

        part_to_process = part_input_str
        is_single_page_expr = False

        if part_to_process == 'N':
            if total_pages == 0: raise ValueError("Cannot use 'N' for a 0-page document.")
            part_to_process = str(total_pages - 1)
            is_single_page_expr = True
        elif re.fullmatch(r"N-(\d+)", part_to_process):
            if total_pages == 0: raise ValueError("Cannot use 'N-k' for a 0-page document.")
            k = int(part_to_process[2:])
            calculated_page = (total_pages - 1) - k
            part_to_process = str(calculated_page)
            is_single_page_expr = True
        elif 'N' in part_to_process: # N is part of a range or other expression
            if total_pages == 0: raise ValueError("Cannot use 'N' in a range for a 0-page document.")
            # Replace N as a boundary for ranges like '0-N' or 'N-N'
            part_to_process = part_to_process.replace('N', str(total_pages - 1))

        # Now, part_to_process has N resolved if it was N, N-k, or a range boundary
        # is_single_page_expr is True if it was N or N-k

        if not is_single_page_expr and '-' in part_to_process: # Process as a range
            try:
                start_str, end_str = part_to_process.split('-', 1)
                start_idx = int(start_str) # Can raise ValueError
                end_idx = int(end_str)     # Can raise ValueError
                
                if not (0 <= start_idx < total_pages and 0 <= end_idx < total_pages and start_idx <= end_idx):
                    # Use part_input_str for error to show original user input
                    raise ValueError(f"Page range '{part_input_str}' (resolved to '{part_to_process}') is invalid for a document with {total_pages} pages. Ensure 0 <= start <= end < {total_pages}.")
                indices.update(range(start_idx, end_idx + 1))
            except ValueError as e:
                if "Page range" in str(e): raise 
                raise ValueError(f"Invalid page range format: '{part_input_str}'. Must be like 'start-end'.") from e
        else: # Process as a single page (either originally, or after N, N-k evaluation)
            try:
                idx = int(part_to_process)
                if not (0 <= idx < total_pages):
                    raise ValueError(f"Page number '{part_input_str}' (resolved to '{part_to_process}') is out of range for a document with {total_pages} pages (0 to {total_pages-1}).")
                indices.add(idx)
            except ValueError as e:
                if "Page number" in str(e): raise
                raise ValueError(f"Invalid page number format: '{part_input_str}'. Must be an integer.") from e
    
    if not indices: # e.g. if page_str was just " " or "," or resulted in no valid pages after N processing
        return None # All pages if input was effectively empty or led to no specific pages
        
    return sorted(list(indices))


def get_raw_markdown_from_pdf(pdf_path: str, pages_to_process_str: Optional[str] = None) -> str:
    """
    Opens a PDF file, processes specified pages, and returns raw Markdown from pymupdf4llm.
    Args:
        pdf_path: Path to the input PDF file.
        pages_to_process_str: Optional comma-separated page string (e.g., "0,1,5-N").
    Returns:
        Raw Markdown string from pymupdf4llm.
    Raises:
        FileNotFoundError, ValueError, RuntimeError.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input PDF file not found: {pdf_path}")
    doc = None
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Error opening PDF '{pdf_path}': {e}") from e
    
    total_pages = doc.page_count
    selected_page_numbers: Optional[list[int]] = None
    if pages_to_process_str:
        try:
            selected_page_numbers = parse_page_numbers(pages_to_process_str, total_pages)
        except ValueError as e:
            if doc: doc.close()
            raise
            
    try:
        markdown_content_raw = to_markdown(
            doc,
            pages=selected_page_numbers,
            ignore_images=True,
            ignore_graphics=True,
            table_strategy=None, 
            page_chunks=False
        )
    except Exception as e:
        raise RuntimeError(f"Error during Markdown conversion from pymupdf4llm: {e}") from e
    finally:
        if doc:
            doc.close()
    return markdown_content_raw

def clean_markdown_to_text_blocks(markdown_content_raw: str) -> str:
    """
    Cleans raw Markdown content into a string of text blocks.
    Headers are converted to plain text lines. Markdown formatting is stripped.
    """
    text_blocks = []
    current_block_lines = []
    is_header_block = False

    def finalize_block_local():
        nonlocal text_blocks, current_block_lines, is_header_block
        if current_block_lines:
            raw_block_text = "\n".join(current_block_lines)
            html_block = markdown.markdown(raw_block_text)
            soup_block = BeautifulSoup(html_block, "html.parser")
            clean_text = soup_block.get_text(separator=' ', strip=True)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            clean_text = re.sub(r'\s+([.,;:?!])', r'\1', clean_text)
            
            if clean_text:
                text_blocks.append(clean_text)
            current_block_lines = []
            is_header_block = False

    if not markdown_content_raw.strip():
        return "" 
    
    for line in markdown_content_raw.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            finalize_block_local() 
            is_header_block = True
            header_text_content = ""
            for i, char_in_line in enumerate(stripped_line):
                if char_in_line != '#':
                    header_text_content = stripped_line[i:].strip()
                    break
            if header_text_content:
                 current_block_lines.append(header_text_content)
        elif stripped_line: 
            if is_header_block: 
                finalize_block_local()
            current_block_lines.append(stripped_line)
        elif not stripped_line and current_block_lines: 
            finalize_block_local()
    
    finalize_block_local() 
    
    return "\n".join(text_blocks)

def convert_pdf_to_markdown_string(pdf_path: str, pages_to_process_str: Optional[str] = None) -> str:
    """
    Converts a PDF file to a cleaned Markdown string (text blocks).
    This function preserves the original public API.
    """
    raw_markdown = get_raw_markdown_from_pdf(pdf_path, pages_to_process_str)
    cleaned_text = clean_markdown_to_text_blocks(raw_markdown)
    return cleaned_text

def main():
    parser = argparse.ArgumentParser(description="Parse PDF files to Markdown or structured JSON.")
    parser.add_argument("input_pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-separated page numbers/ranges to process (e.g., '0,1,5-7,10-N'). 'N' is the last page. Default is all pages."
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format. 'markdown' outputs plain text. 'json' outputs structured data. Default is 'markdown'."
    )
    args = parser.parse_args()
    
    raw_markdown_content = ""
    try:
        raw_markdown_content = get_raw_markdown_from_pdf(args.input_pdf, args.pages)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e: # From parse_page_numbers
        print(f"Error in --pages argument: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e: # From PDF opening or pymupdf4llm conversion
        print(f"Error processing PDF: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # Catch-all for other unexpected errors
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "markdown":
        cleaned_text_output = clean_markdown_to_text_blocks(raw_markdown_content)
        sys.stdout.write(cleaned_text_output)
        sys.stdout.write('\n')

    elif args.format == "json":
        if not raw_markdown_content.strip():
            sys.stdout.write("[]\n")
        else:
            output_elements_json = []
            current_paragraph_lines_json_cli = []
            
            def finalize_paragraph_json_for_cli():
                nonlocal output_elements_json, current_paragraph_lines_json_cli
                if current_paragraph_lines_json_cli:
                    raw_text = "\n".join(current_paragraph_lines_json_cli)
                    html_paragraph = markdown.markdown(raw_text)
                    soup_paragraph = BeautifulSoup(html_paragraph, "html.parser")
                    text_content = soup_paragraph.get_text(separator=' ', strip=True)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    text_content = re.sub(r'\s+([.,;:?!])', r'\1', text_content)
                    if text_content:
                        output_elements_json.append({"type": "paragraph", "text": text_content})
                    current_paragraph_lines_json_cli = []

            for line in raw_markdown_content.splitlines(): # Use raw_markdown_content here
                stripped_line = line.strip()
                if stripped_line.startswith("#"):
                    finalize_paragraph_json_for_cli()
                    level = 0
                    text_content_json = ""
                    for i, char_in_line in enumerate(stripped_line):
                        if char_in_line == '#':
                            level += 1
                        else:
                            text_content_json = stripped_line[i:].strip()
                            break
                    if not text_content_json and level > 0 and stripped_line == '#' * level:
                         pass 
                    elif text_content_json: 
                        html_header = markdown.markdown(text_content_json)
                        soup_header = BeautifulSoup(html_header, "html.parser")
                        clean_header_text = soup_header.get_text(separator=' ', strip=True)
                        clean_header_text = re.sub(r'\s+', ' ', clean_header_text).strip()
                        clean_header_text = re.sub(r'\s+([.,;:?!])', r'\1', clean_header_text)
                        if clean_header_text:
                            output_elements_json.append({"type": "heading", "level": level, "text": clean_header_text})
                elif stripped_line:
                    current_paragraph_lines_json_cli.append(stripped_line)
                elif not stripped_line and current_paragraph_lines_json_cli: 
                    finalize_paragraph_json_for_cli()
            
            finalize_paragraph_json_for_cli() 
            
            sys.stdout.write(json.dumps(output_elements_json, indent=2))
            sys.stdout.write('\n')

if __name__ == "__main__":
    main()
