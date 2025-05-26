import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any, Union, Optional
import os
import argparse
import re # Added for re.sub

# --- New functions for Markdown conversion ---

def _process_xml_children_to_markdown_text(element: ET.Element) -> str:
    """
    Recursively processes children of an XML element to extract text and apply inline Markdown.
    This is a helper for inline elements like <p>, <li>, etc.
    """
    text_parts = []
    if element.text:
        text_parts.append(element.text.strip())
    
    for child in element:
        if child.tag in ['b', 'strong']:
            text_parts.append(f"**{_process_xml_children_to_markdown_text(child)}**")
        elif child.tag in ['i', 'italic', 'em']:
            text_parts.append(f"*{_process_xml_children_to_markdown_text(child)}*")
        elif child.tag in ['sup']:
            text_parts.append(f"^{_process_xml_children_to_markdown_text(child)}^")
        elif child.tag in ['sub']:
            text_parts.append(f"~{_process_xml_children_to_markdown_text(child)}~")
        # Add more inline tags as needed (e.g., xref, links)
        else: # For other tags, just get their text content recursively
            # Ensure that recursive calls also return stripped text to avoid compounding spaces
            processed_child_text = _process_xml_children_to_markdown_text(child)
            if processed_child_text: # Only append if there's actual text
                text_parts.append(processed_child_text)

        if child.tail:
            stripped_tail = child.tail.strip()
            if stripped_tail: # Only append if there's actual tail text
                text_parts.append(stripped_tail)
            
    # Join parts, then replace multiple spaces with single, then strip ends.
    joined_text = " ".join(filter(None, text_parts))
    cleaned_text = re.sub(r'\s+', ' ', joined_text).strip()
    # Specific fix for "text .": if a space precedes a period (and possibly other punctuation), remove it.
    cleaned_text = re.sub(r'\s+\.', '.', cleaned_text)
    return cleaned_text

def _xml_to_markdown_elements(element: ET.Element, depth: int) -> List[Dict[str, Any]]:
    """
    Converts an XML element and its children into a list of structured Markdown elements.
    """
    markdown_elements: List[Dict[str, Any]] = []

    if element.tag == 'sec':
        current_depth = depth + 1
        title_element = element.find('./title')
        if title_element is not None:
            title_text = _process_xml_children_to_markdown_text(title_element)
            if title_text:
                markdown_elements.append({"type": "heading", "level": current_depth, "text_content": title_text})
        
        for child in element:
            if child.tag != 'title': # Avoid re-processing title
                markdown_elements.extend(_xml_to_markdown_elements(child, current_depth))
    
    elif element.tag == 'p':
        p_text = _process_xml_children_to_markdown_text(element)
        if p_text:
            markdown_elements.append({"type": "paragraph", "text_content": p_text})

    elif element.tag == 'list': # JATS uses <list list-type="bullet|ordered">
        list_type = element.get('list-type', 'bullet')
        items = []
        for li_element in element.findall('./list-item'):
            # Try to find a <p> tag within list-item first
            p_child = li_element.find('./p')
            if p_child is not None:
                item_text = _process_xml_children_to_markdown_text(p_child)
            else: # Fallback if no direct <p> child, process the whole list-item for inline content
                item_text = _process_xml_children_to_markdown_text(li_element)
            
            if item_text:
                items.append(item_text)
        if items:
            markdown_elements.append({"type": "list", "ordered": list_type == "ordered", "items": items})

    elif element.tag == 'table-wrap': # JATS table
        # Basic table handling: extract caption, then try to parse table if simple
        caption_element = element.find('./caption/p')
        if caption_element is not None:
            caption_text = _process_xml_children_to_markdown_text(caption_element)
            if caption_text:
                 markdown_elements.append({"type": "paragraph", "text_content": f"Table Caption: {caption_text}"})

        table_element = element.find('./table')
        if table_element is not None:
            headers = []
            rows = []
            thead = table_element.find('./thead/tr')
            if thead is not None:
                for th in thead.findall('./th'):
                    headers.append(_process_xml_children_to_markdown_text(th))
            
            tbody = table_element.find('./tbody')
            if tbody is not None:
                for tr_element in tbody.findall('./tr'):
                    row_cells = []
                    for td_element in tr_element.findall('./td'):
                        row_cells.append(_process_xml_children_to_markdown_text(td_element))
                    if row_cells:
                        rows.append(row_cells)
            
            if headers and rows:
                markdown_elements.append({"type": "table", "headers": headers, "rows": rows})
            elif rows: # Table with no headers
                 markdown_elements.append({"type": "table", "headers": [], "rows": rows})


    # Fallback for direct children of body not wrapped in sec, or other unhandled block tags
    elif element.tag not in ['title', 'caption', 'label', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'list-item', 'fig']: # Avoid processing parts of already handled complex elements
        for child_element in element:
             markdown_elements.extend(_xml_to_markdown_elements(child_element, depth))
             
    return markdown_elements


def _process_json_children_to_markdown_text(data: Any) -> str:
    """
    Recursively processes children of a JSON element to extract text and apply inline Markdown.
    """
    text_parts = []
    if isinstance(data, str):
        return data.strip()
    elif isinstance(data, list):
        return " ".join(_process_json_children_to_markdown_text(item) for item in data).strip()
    elif isinstance(data, dict):
        if '#text' in data:
            text_parts.append(str(data['#text']).strip())
        
        # Handling for common inline tags in EuropePMC JSON
        for key, value in data.items():
            if key == 'italic' or key == 'em':
                text_parts.append(f"*{_process_json_children_to_markdown_text(value)}*")
            elif key == 'bold' or key == 'strong':
                text_parts.append(f"**{_process_json_children_to_markdown_text(value)}**")
            elif key == 'sup':
                text_parts.append(f"^{_process_json_children_to_markdown_text(value)}^")
            elif key == 'sub':
                text_parts.append(f"~{_process_json_children_to_markdown_text(value)}~")
            elif key not in ['#text'] and isinstance(value, (str, list, dict)): 
                # Recursively process other potential text-holding fields, avoiding double-counting #text
                # This is a simplification; real JSON structures can be more complex.
                # We might need to be more specific about which keys to process.
                # 'label' and 'caption' are often handled by the block-level parser for figures/tables.
                # 'xref' text content should be extracted.
                if key not in ['label', 'caption']: # Removed 'xref' from this exclusion list
                    extracted_text = _process_json_children_to_markdown_text(value)
                    if extracted_text and extracted_text not in text_parts: # Avoid duplicates
                         text_parts.append(extracted_text)
        
        return " ".join(filter(None, text_parts)).strip()
    return ""


def _json_to_markdown_elements(data: Any, depth: int) -> List[Dict[str, Any]]:
    """
    Converts a JSON structure (from Europe PMC) into a list of structured Markdown elements.
    """
    markdown_elements: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        if 'title' in data: # Typically a section title
            title_text = _process_json_children_to_markdown_text(data['title'])
            if title_text:
                markdown_elements.append({"type": "heading", "level": depth, "text_content": title_text})

        if 'p' in data: # Paragraph
            p_data = data['p']
            if isinstance(p_data, list):
                for p_item in p_data:
                    p_text = _process_json_children_to_markdown_text(p_item)
                    if p_text:
                        markdown_elements.append({"type": "paragraph", "text_content": p_text})
            else:
                p_text = _process_json_children_to_markdown_text(p_data)
                if p_text:
                    markdown_elements.append({"type": "paragraph", "text_content": p_text})
        
        if 'list' in data: # List
            list_data = data['list']
            if not isinstance(list_data, list): list_data = [list_data] # Ensure it's a list

            for single_list in list_data:
                list_type = single_list.get('@list-type', 'bullet')
                items = []
                list_items_data = single_list.get('list-item', [])
                if not isinstance(list_items_data, list): list_items_data = [list_items_data]

                for item_data in list_items_data:
                    # list-item can contain 'p' or just be text
                    item_p_data = item_data.get('p', item_data) # Fallback to item_data if 'p' not found
                    item_text = _process_json_children_to_markdown_text(item_p_data)
                    if item_text:
                        items.append(item_text)
                if items:
                    markdown_elements.append({"type": "list", "ordered": list_type == "ordered", "items": items})

        if 'table-wrap' in data: # Table
            table_wrap_data = data['table-wrap']
            if not isinstance(table_wrap_data, list): table_wrap_data = [table_wrap_data]

            for table_item in table_wrap_data:
                caption_text = ""
                if 'caption' in table_item and 'p' in table_item['caption']:
                    caption_text = _process_json_children_to_markdown_text(table_item['caption']['p'])
                if caption_text:
                    markdown_elements.append({"type": "paragraph", "text_content": f"Table Caption: {caption_text}"})

                if 'table' in table_item:
                    table_data = table_item['table']
                    headers = []
                    rows = []
                    if 'thead' in table_data and 'tr' in table_data['thead'] and 'th' in table_data['thead']['tr']:
                        th_data = table_data['thead']['tr']['th']
                        if isinstance(th_data, list):
                            for th in th_data: headers.append(_process_json_children_to_markdown_text(th))
                        else:
                            headers.append(_process_json_children_to_markdown_text(th_data))
                    
                    if 'tbody' in table_data and 'tr' in table_data['tbody']:
                        tr_list = table_data['tbody']['tr']
                        if not isinstance(tr_list, list): tr_list = [tr_list]
                        for tr_data in tr_list:
                            row_cells = []
                            td_list = tr_data.get('td', [])
                            if not isinstance(td_list, list): td_list = [td_list]
                            for td_data in td_list:
                                row_cells.append(_process_json_children_to_markdown_text(td_data))
                            if row_cells:
                                rows.append(row_cells)
                    if headers and rows:
                         markdown_elements.append({"type": "table", "headers": headers, "rows": rows})
                    elif rows: # Table with no headers
                         markdown_elements.append({"type": "table", "headers": [], "rows": rows})


        # Recursive call for nested sections ('sec')
        if 'sec' in data:
            sec_data = data['sec']
            if not isinstance(sec_data, list): # If 'sec' is not a list, make it one
                sec_data = [sec_data]
            for section in sec_data:
                markdown_elements.extend(_json_to_markdown_elements(section, depth + 1))
    
    elif isinstance(data, list): # If the top-level data is a list (e.g., list of sections)
        for item in data:
            markdown_elements.extend(_json_to_markdown_elements(item, depth))
    elif isinstance(data, str): # Handle raw string data if passed directly (e.g. abstract content)
        if data.strip(): # If it's a non-empty string, treat as a paragraph
            markdown_elements.append({"type": "paragraph", "text_content": _process_json_children_to_markdown_text(data)}) # Ensure inline processing
            
    return markdown_elements

def _render_elements_to_markdown_string(elements: List[Dict[str, Any]]) -> str:
    """
    Converts a list of structured Markdown elements into a single Markdown string.
    """
    markdown_parts = []
    for element in elements:
        if element["type"] == "heading":
            markdown_parts.append(f"{'#' * element['level']} {element['text_content']}")
        elif element["type"] == "paragraph":
            markdown_parts.append(element['text_content'])
        elif element["type"] == "list":
            list_char = "1." if element.get("ordered", False) else "-"
            for item_text in element["items"]:
                markdown_parts.append(f"{list_char} {item_text}")
        elif element["type"] == "table":
            headers = element.get("headers", [])
            rows = element.get("rows", [])
            if headers:
                markdown_parts.append("| " + " | ".join(headers) + " |")
                markdown_parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                markdown_parts.append("| " + " | ".join(row) + " |")
        # Add more types as needed
    return "\n\n".join(markdown_parts)

def _finalize_elements_for_json_output(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the 'text_content' of elements into final 'text' (Markdown string for that block)
    for the --format json output.
    """
    output_elements = []
    for el in elements:
        new_el = {"type": el["type"]}
        if el["type"] == "heading":
            new_el["level"] = el["level"]
            new_el["text"] = f"{'#' * el['level']} {el['text_content']}"
        elif el["type"] == "paragraph":
            new_el["text"] = el['text_content']
        elif el["type"] == "list":
            list_char = "1." if el.get("ordered", False) else "-"
            new_el["text"] = "\n".join([f"{list_char} {item}" for item in el["items"]])
        elif el["type"] == "table":
            # Simplified table rendering for the 'text' field
            md_table_parts = []
            headers = el.get("headers", [])
            rows = el.get("rows", [])
            if headers:
                md_table_parts.append("| " + " | ".join(headers) + " |")
                md_table_parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row_data in rows:
                md_table_parts.append("| " + " | ".join(row_data) + " |")
            new_el["text"] = "\n".join(md_table_parts)
        else:
            # For unknown types, or types that don't have a direct markdown block representation
            new_el["text"] = el.get('text_content', '') 
        output_elements.append(new_el)
    return output_elements

# --- End of new functions ---

def _extract_text_xml(element: Union[ET.Element, None]) -> str: # Keep for now if used by other things, or mark as deprecated
    """Recursively extracts text from an XML element and its children. (Old version)"""
    if element is None:
        return ""
    
    text_parts = []
    if element.text:
        text_parts.append(element.text.strip())
    
    for child in element:
        text_parts.append(_extract_text_xml(child))
        if child.tail:
            text_parts.append(child.tail.strip())
            
    return " ".join(filter(None, text_parts)).strip()

def _extract_text_json(data: Any) -> str:
    """Recursively extracts text from a JSON structure (dict, list, or string)."""
    if isinstance(data, str):
        return data.strip()
    elif isinstance(data, list):
        return " ".join(_extract_text_json(item) for item in data).strip()
    elif isinstance(data, dict):
        text_parts = []
        if '#text' in data:
            # Prioritize #text if it's a simple string or list of strings
            if isinstance(data['#text'], str):
                text_parts.append(data['#text'].strip())
            elif isinstance(data['#text'], list):
                 text_parts.append(" ".join(str(t).strip() for t in data['#text'] if isinstance(t, str)))

        # Process other known inline tags or common text-holding keys
        # This needs to be adapted based on observed JSON structure for complex paragraphs
        for key, value in data.items():
            if key != '#text': # Avoid double processing if #text was primary
                # Add common inline tags here if they appear as keys
                if key in ['p', 'italic', 'bold', 'sup', 'sub', 'xref', 'title', 'label', 'caption']: # Extend as needed
                    text_parts.append(_extract_text_json(value))
                elif isinstance(value, (dict, list)): # Recursively search in nested structures
                     # Avoid infinite recursion on parent links or very deep structures if any
                    pass # Potentially add more specific handling if needed
        
        # Fallback for cases where text might be directly in values without specific keys
        if not text_parts and '#text' not in data:
            for key, value in data.items():
                if isinstance(value, str) and key not in ['@id', '@rid', '@ref-type', '@xmlns:xlink', '@xlink:href', '@article-type', '@dtd-version', '@xml:lang']: # Avoid attribute-like keys
                    text_parts.append(value.strip())
        
        return " ".join(filter(None, text_parts)).strip()
    return ""

def parse_pmc_xml(xml_string: str) -> List[Dict[str, Any]]:
    """
    Parses JATS XML string to extract structured Markdown elements.
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        # print(f"XML Parsing Error: {e}") # Optional: for debugging
        return [{"type": "error", "text_content": f"XML Parsing Error: {str(e)}"}]

    # New way: Convert to Markdown elements
    markdown_elements: List[Dict[str, Any]] = []

    # Abstract processing
    # Find the abstract element within <front><article-meta>
    abstract_xml_element = root.find(".//front/article-meta/abstract")
    if abstract_xml_element is not None:
        markdown_elements.append({"type": "heading", "level": 1, "text_content": "Abstract"})
        # Process children of the abstract element using _xml_to_markdown_elements
        # The depth is 1 because content inside abstract is under the "Abstract" H1.
        for child_of_abstract in abstract_xml_element:
            markdown_elements.extend(_xml_to_markdown_elements(child_of_abstract, 1))
    
    # Body processing
    # Find the body element
    body_xml_element = root.find(".//body")
    if body_xml_element is not None:
        # Process direct children of the body element
        # Depth starts at 0 for top-level elements in the body (e.g., top-level <sec> will become H1)
        for child_of_body in body_xml_element:
            markdown_elements.extend(_xml_to_markdown_elements(child_of_body, 0))
            
    # Note: Figure and Table captions are expected to be handled by _xml_to_markdown_elements
    # when it encounters <table-wrap> or <fig> elements if those are added.
    # Currently, <fig> caption extraction is not explicitly in _xml_to_markdown_elements.
    # <table-wrap> caption extraction is basic.

    return markdown_elements

def parse_europe_pmc_json(json_data: Union[Dict, List]) -> List[Dict[str, Any]]:
    """
    Parses JATS-derived JSON (from Europe PMC) to extract structured Markdown elements.
    """
    if not isinstance(json_data, dict) or 'article' not in json_data:
        return [{"type": "error", "text_content": "Invalid JSON structure: 'article' key missing"}]

    article = json_data.get('article', {})
    markdown_elements: List[Dict[str, Any]] = []

    # Abstract
    try:
        abstract_node = article.get('front', {}).get('article-meta', {}).get('abstract', {})
        if abstract_node:
            markdown_elements.append({"type": "heading", "level": 1, "text_content": "Abstract"})
            # The abstract_node itself might be a 'p' or contain 'p' or 'sec'
            # We pass the content of abstract to _json_to_markdown_elements
            if 'p' in abstract_node: # Common case: abstract has one or more paragraphs
                 markdown_elements.extend(_json_to_markdown_elements(abstract_node['p'], 1))
            elif 'sec' in abstract_node: # Abstract might have sections
                 markdown_elements.extend(_json_to_markdown_elements(abstract_node['sec'], 1))
            else: # Fallback for simple string abstract or other structures
                 abstract_text_content = _process_json_children_to_markdown_text(abstract_node)
                 if abstract_text_content:
                    markdown_elements.append({"type": "paragraph", "text_content": abstract_text_content})

    except Exception as e:
        print(f"Error parsing abstract from JSON: {e}")
        markdown_elements.append({"type": "error", "text_content": f"Error parsing abstract: {e}"})

    # Body sections
    body_node = article.get('body', {})
    if body_node: # body_node could be a dict (common) or sometimes a list of sections
        # _json_to_markdown_elements expects a dict or list and handles 'sec' key internally for depth
        markdown_elements.extend(_json_to_markdown_elements(body_node, 0)) # Start depth at 0 for body

    return markdown_elements


def main():
    parser = argparse.ArgumentParser(description="Parse XML/JSON articles to Markdown.")
    parser.add_argument("input_file", help="Path to the input XML or JSON file.")
    parser.add_argument(
        "--input-type", 
        choices=["xml", "json"], 
        help="Type of the input file (xml or json). Inferred if not provided."
    )
    parser.add_argument(
        "--format", 
        choices=["markdown", "json"], 
        default="markdown", 
        help="Output format: plain Markdown or JSON array of Markdown blocks."
    )
    args = parser.parse_args()

    input_type = args.input_type
    if not input_type:
        if args.input_file.lower().endswith(".xml"):
            input_type = "xml"
        elif args.input_file.lower().endswith(".json"):
            input_type = "json"
        else:
            print("Error: Cannot infer input type. Please use --input-type.")
            return

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    markdown_elements: List[Dict[str, Any]] = []

    if input_type == "xml":
        try:
            root = ET.fromstring(content)
            # The main parsing logic for XML should start from the root or specific parts like <article>
            # Assuming the root is <article> or contains it.
            article_node = root
            if root.tag != 'article' and root.find('.//article') is not None: # Find article if root is not article
                article_node = root.find('.//article')
            
            if article_node is not None:
                 markdown_elements = parse_pmc_xml(content) # parse_pmc_xml now returns elements
            else:
                print("Error: <article> tag not found in XML.")
                return
        except ET.ParseError as e:
            print(f"XML Parsing Error: {e}")
            return
    elif input_type == "json":
        try:
            json_content = json.loads(content)
            markdown_elements = parse_europe_pmc_json(json_content) # parse_europe_pmc_json now returns elements
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}")
            return
    
    if not markdown_elements:
        print("No content parsed.")
        return

    if args.format == "markdown":
        output_string = _render_elements_to_markdown_string(markdown_elements)
        print(output_string)
    elif args.format == "json":
        # The `text` field in each element should be the Markdown for that block
        json_output_elements = _finalize_elements_for_json_output(markdown_elements)
        print(json.dumps(json_output_elements, indent=2))


if __name__ == '__main__':
    main()
