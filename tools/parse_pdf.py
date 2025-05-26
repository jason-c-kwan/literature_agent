import argparse
import json
import sys
import pymupdf # For pymupdf.open
from pymupdf4llm.helpers.pymupdf_rag import to_markdown
from bs4 import BeautifulSoup
import re
import markdown # For converting Markdown to HTML

def parse_page_numbers(page_str: str | None, total_pages: int) -> list[int] | None:
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

    try:
        doc = pymupdf.open(args.input_pdf)
    except Exception as e:
        print(f"Error opening PDF '{args.input_pdf}': {e}", file=sys.stderr)
        sys.exit(1)
        return # Should not be reached if sys.exit works

    total_pages = doc.page_count
    
    if total_pages == 0 and args.pages:
        print(f"Warning: PDF '{args.input_pdf}' has 0 pages. --pages argument will be ignored.", file=sys.stderr)
        # Proceeding will likely result in empty output or errors from to_markdown if it expects pages.
        # to_markdown might handle doc with 0 pages gracefully, or might error.
        # If it errors, we might want to exit here. For now, let it try.
        
    selected_page_numbers = None
    if args.pages:
        try:
            selected_page_numbers = parse_page_numbers(args.pages, total_pages)
            if selected_page_numbers is None and args.pages.strip(): # If parsing resulted in None but input wasn't empty (e.g. " , ")
                print(f"Warning: --pages argument '{args.pages}' resulted in no specific pages selected; processing all pages.", file=sys.stderr)
        except ValueError as e:
            print(f"Error in --pages argument: {e}", file=sys.stderr)
            doc.close()
            sys.exit(1)
            return # Should not be reached
    
    # If selected_page_numbers is still None here, it means all pages.
    # to_markdown expects None for all pages, or a list of 0-based indices.
    
    # Ensure selected_page_numbers is None if it's an empty list, so to_markdown gets None for all pages.
    if isinstance(selected_page_numbers, list) and not selected_page_numbers:
        # This case might occur if parse_page_numbers was modified to return [] for "all pages"
        # or if an empty page string somehow yielded an empty list.
        # My current parse_page_numbers returns None for "all pages".
        pass


    try:
        # pymupdf4llm.to_markdown handles pages=None as all pages.
        markdown_content = to_markdown(
            doc,
            pages=selected_page_numbers,
            ignore_images=True,
            ignore_graphics=True,
            table_strategy=None, # Disables table processing
            page_chunks=False # Ensures a single string output
        )
    except Exception as e:
        print(f"Error during Markdown conversion: {e}", file=sys.stderr)
        # doc.close() will be handled by the finally block
        sys.exit(1)
        return # Should not be reached
    finally:
        doc.close()

    # Common logic for parsing markdown_content into text blocks
    text_blocks = []
    current_block_lines = []
    is_header_block = False
    header_level = 0

    def finalize_block_for_text_output():
        nonlocal text_blocks, current_block_lines, is_header_block, header_level
        if current_block_lines:
            raw_block_text = "\n".join(current_block_lines)
            html_block = markdown.markdown(raw_block_text)
            soup_block = BeautifulSoup(html_block, "html.parser")
            # Use space as separator, then clean multiple spaces for cleaner text
            clean_text = soup_block.get_text(separator=' ', strip=True)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            clean_text = re.sub(r'\s+([.,;:?!])', r'\1', clean_text) # Remove space before punctuation
            
            if clean_text:
                if is_header_block: # This was a header block
                    text_blocks.append(clean_text) # Just the text for markdown output
                else: # Paragraph block
                    text_blocks.append(clean_text)
            current_block_lines = []
            is_header_block = False # Reset for next block

    if not markdown_content.strip():
        if args.format == "markdown":
            sys.stdout.write("\n")
        elif args.format == "json":
            sys.stdout.write("[]\n")
    else:
        for line in markdown_content.splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("#"):
                finalize_block_for_text_output() # Finalize previous block
                is_header_block = True # Mark current accumulation as header
                # Extract header level and text
                temp_level = 0
                header_text_content = ""
                for i, char_in_line in enumerate(stripped_line):
                    if char_in_line == '#':
                        temp_level += 1
                    else:
                        header_text_content = stripped_line[i:].strip()
                        break
                if header_text_content: # Only if header has text
                     current_block_lines.append(header_text_content) # Store raw text part of header
                     header_level = temp_level # Store its level for JSON
                # finalize_block_for_text_output will be called by next line or end
            elif stripped_line: # Non-empty, not a header
                if is_header_block: # If previous line was a header, finalize it
                    finalize_block_for_text_output()
                current_block_lines.append(stripped_line)
            elif not stripped_line and current_block_lines: # Empty line might end current block
                finalize_block_for_text_output()
        
        finalize_block_for_text_output() # Finalize any last block

        if args.format == "markdown":
            output_text = "\n".join(text_blocks)
            sys.stdout.write(output_text)
            if output_text: # Add newline if there was text
                 sys.stdout.write('\n')
            else: # If text_blocks was empty or all blocks were empty strings
                 sys.stdout.write('\n')


        elif args.format == "json":
            output_elements = []
            # Re-parse markdown_content for JSON structure, this time with levels
            # This is a bit redundant but cleaner than trying to adapt text_blocks
            current_paragraph_lines_json = []
            
            def finalize_paragraph_json():
                nonlocal output_elements, current_paragraph_lines_json
                if current_paragraph_lines_json:
                    raw_text = "\n".join(current_paragraph_lines_json)
                    html_paragraph = markdown.markdown(raw_text)
                    soup_paragraph = BeautifulSoup(html_paragraph, "html.parser")
                    text_content = soup_paragraph.get_text(separator=' ', strip=True)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    text_content = re.sub(r'\s+([.,;:?!])', r'\1', text_content) # Remove space before punctuation
                    if text_content:
                        output_elements.append({"type": "paragraph", "text": text_content})
                    current_paragraph_lines_json = []

            for line in markdown_content.splitlines():
                stripped_line = line.strip()
                if stripped_line.startswith("#"):
                    finalize_paragraph_json()
                    level = 0
                    text_content = ""
                    for i, char_in_line in enumerate(stripped_line):
                        if char_in_line == '#':
                            level += 1
                        else:
                            text_content = stripped_line[i:].strip()
                            break
                    if text_content:
                        html_header = markdown.markdown(text_content)
                        soup_header = BeautifulSoup(html_header, "html.parser")
                        clean_header_text = soup_header.get_text(separator=' ', strip=True)
                        clean_header_text = re.sub(r'\s+', ' ', clean_header_text).strip()
                        clean_header_text = re.sub(r'\s+([.,;:?!])', r'\1', clean_header_text) # Remove space before punctuation
                        if clean_header_text:
                            output_elements.append({"type": "heading", "level": level, "text": clean_header_text})
                elif stripped_line:
                    current_paragraph_lines_json.append(stripped_line)
                elif not stripped_line and current_paragraph_lines_json:
                    finalize_paragraph_json()
            finalize_paragraph_json()
            
            sys.stdout.write(json.dumps(output_elements, indent=2))
            sys.stdout.write('\n')

if __name__ == "__main__":
    main()
