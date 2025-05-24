import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any, Union
import os

def _extract_text_xml(element: Union[ET.Element, None]) -> str:
    """Recursively extracts text from an XML element and its children."""
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

def parse_pmc_xml(xml_string: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parses JATS XML string to extract plain text and structured metadata.
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        print(f"XML Parsing Error: {e}")
        return "", {"error": "XML Parsing Error", "details": str(e)}

    plain_text_parts = []
    structured_metadata: Dict[str, Any] = {
        "sections": [],
        "figure_captions": [],
        "table_captions": []
    }

    # Abstract
    abstract_element = root.find(".//front/article-meta/abstract")
    if abstract_element is not None:
        # JATS abstract can have <p> or <sec> inside. We'll try to get all text.
        abstract_text = _extract_text_xml(abstract_element)
        if abstract_text:
            plain_text_parts.append(abstract_text)
            # Add abstract to sections for consistency, or a dedicated field
            structured_metadata["sections"].append({
                "title": "Abstract",
                "paragraphs": [abstract_text] # Or split if multiple <p> are distinctly found
            })


    # Body sections
    body_element = root.find(".//body")
    if body_element is not None:
        for sec_element in body_element.findall(".//sec"):
            section_title_element = sec_element.find("./title")
            section_title = _extract_text_xml(section_title_element) if section_title_element is not None else "Untitled Section"
            
            paragraphs_text = []
            for p_element in sec_element.findall("./p"):
                p_text = _extract_text_xml(p_element)
                if p_text:
                    paragraphs_text.append(p_text)
                    plain_text_parts.append(p_text)
            
            if paragraphs_text: # Only add section if it has content
                structured_metadata["sections"].append({
                    "title": section_title,
                    "paragraphs": paragraphs_text
                })

        # Global search for figure and table captions within the entire body
        # Figure captions
        for fig_element in body_element.findall(".//fig"): # Search all fig elements in body
            label_element = fig_element.find("./label")
            label_text = _extract_text_xml(label_element).strip() if label_element is not None else ""
            
            caption_p_element = fig_element.find("./caption/p") # More specific to get main caption text
            caption_text_main = _extract_text_xml(caption_p_element).strip() if caption_p_element is not None else ""
            
            if not caption_text_main: # Fallback if caption/p is not found, try caption itself
                caption_element_fallback = fig_element.find("./caption")
                caption_text_main = _extract_text_xml(caption_element_fallback).strip() if caption_element_fallback is not None else ""

            full_caption = (label_text + " " + caption_text_main).strip()
            if full_caption:
                structured_metadata["figure_captions"].append(full_caption)
        
        # Table captions
        for table_wrap_element in body_element.findall(".//table-wrap"): # Search all table-wrap elements in body
            label_element = table_wrap_element.find("./label")
            label_text = _extract_text_xml(label_element).strip() if label_element is not None else ""

            caption_p_element = table_wrap_element.find("./caption/p")
            caption_text_main = _extract_text_xml(caption_p_element).strip() if caption_p_element is not None else ""

            if not caption_text_main: # Fallback
                caption_element_fallback = table_wrap_element.find("./caption")
                caption_text_main = _extract_text_xml(caption_element_fallback).strip() if caption_element_fallback is not None else ""
            
            full_caption = (label_text + " " + caption_text_main).strip()
            if full_caption:
                structured_metadata["table_captions"].append(full_caption)
    
    # Consolidate plain text
    full_plain_text = "\n\n".join(plain_text_parts)
    
    return full_plain_text, structured_metadata

def parse_europe_pmc_json(json_data: Union[Dict, List]) -> Tuple[str, Dict[str, Any]]:
    """
    Parses JATS-derived JSON (from Europe PMC) to extract plain text and structured metadata.
    """
    if not isinstance(json_data, dict) or 'article' not in json_data:
        return "", {"error": "Invalid JSON structure: 'article' key missing"}

    article = json_data.get('article', {})
    plain_text_parts = []
    structured_metadata: Dict[str, Any] = {
        "sections": [],
        "figure_captions": [],
        "table_captions": []
    }

    # Abstract
    try:
        abstract_node = article.get('front', {}).get('article-meta', {}).get('abstract', {})
        if abstract_node:
            # Abstract can be a simple 'p' string, a dict with 'p', or list of 'p' dicts/strings
            abstract_p = abstract_node.get('p', '')
            abstract_text = _extract_text_json(abstract_p)
            if abstract_text:
                plain_text_parts.append(abstract_text)
                structured_metadata["sections"].append({
                    "title": "Abstract",
                    "paragraphs": [abstract_text]
                })
    except Exception as e:
        print(f"Error parsing abstract from JSON: {e}")


    # Body sections
    body_node = article.get('body', {})
    if isinstance(body_node, dict):
        sections = body_node.get('sec', [])
        if not isinstance(sections, list): # Sometimes 'sec' might not be a list if only one section
            sections = [sections] if sections else []

        for sec_item in sections:
            if not isinstance(sec_item, dict): continue

            section_title = _extract_text_json(sec_item.get('title', 'Untitled Section'))
            
            paragraphs_text = []
            paragraphs_data = sec_item.get('p', [])
            if not isinstance(paragraphs_data, list):
                paragraphs_data = [paragraphs_data] if paragraphs_data else []
            
            for p_data in paragraphs_data:
                p_text = _extract_text_json(p_data)
                if p_text:
                    paragraphs_text.append(p_text)
                    plain_text_parts.append(p_text)
            
            if paragraphs_text:
                structured_metadata["sections"].append({
                    "title": section_title,
                    "paragraphs": paragraphs_text
                })

            # Figure captions
            figs = sec_item.get('fig', [])
            if not isinstance(figs, list): figs = [figs] if figs else []
            for fig_item in figs:
                if isinstance(fig_item, dict):
                    label_text = _extract_text_json(fig_item.get('label', '')).strip()
                    # Caption can be a dict with 'p', or sometimes just a string if simple
                    caption_content = fig_item.get('caption', {})
                    if isinstance(caption_content, dict):
                        caption_p_text = _extract_text_json(caption_content.get('p', '')).strip()
                    elif isinstance(caption_content, str): # Handle cases where caption is just a string
                        caption_p_text = _extract_text_json(caption_content).strip()
                    else:
                        caption_p_text = ""
                    
                    full_caption = (label_text + " " + caption_p_text).strip()
                    if full_caption:
                        structured_metadata["figure_captions"].append(full_caption)
            
            # Table captions
            table_wraps = sec_item.get('table-wrap', [])
            if not isinstance(table_wraps, list): table_wraps = [table_wraps] if table_wraps else []
            for table_item in table_wraps:
                if isinstance(table_item, dict):
                    label_text = _extract_text_json(table_item.get('label', '')).strip()
                    caption_content = table_item.get('caption', {})
                    if isinstance(caption_content, dict):
                        caption_p_text = _extract_text_json(caption_content.get('p', '')).strip()
                    elif isinstance(caption_content, str):
                        caption_p_text = _extract_text_json(caption_content).strip()
                    else:
                        caption_p_text = ""

                    full_caption = (label_text + " " + caption_p_text).strip()
                    if full_caption:
                        structured_metadata["table_captions"].append(full_caption)

    full_plain_text = "\n\n".join(plain_text_parts)
    return full_plain_text, structured_metadata

if __name__ == '__main__':
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths to example files relative to the script's directory
    # The example files are in 'workspace', which is a sibling to 'tools' (where this script is)
    # So, from 'tools', we go up one level ('../') and then into 'workspace/'
    xml_file_path = os.path.join(script_dir, "../workspace/pmc_example.xml")
    json_file_path = os.path.join(script_dir, "../workspace/europe_pmc_example.json")

    # Test XML parsing
    try:
        with open(xml_file_path, "r", encoding="utf-8") as f:
            xml_content = f.read()
        
        print("--- Parsing XML ---")
        plain_text_xml, metadata_xml = parse_pmc_xml(xml_content)
        
        print("\n--- XML Plain Text (first 500 chars) ---")
        print(plain_text_xml[:500] + "...")
        
        print("\n--- XML Structured Metadata (sample) ---")
        print(f"Number of sections: {len(metadata_xml.get('sections', []))}")
        if metadata_xml.get('sections') and len(metadata_xml['sections']) > 0:
            print(f"First section title: {metadata_xml['sections'][0].get('title')}")
            if metadata_xml['sections'][0].get('paragraphs') and len(metadata_xml['sections'][0]['paragraphs']) > 0:
                 print(f"First paragraph of first section (first 100 chars): {metadata_xml['sections'][0]['paragraphs'][0][:100]}...")
            else:
                print(f"First section '{metadata_xml['sections'][0].get('title')}' has no paragraphs.")
        else:
            print("No sections found in XML.")

        print(f"Number of figure captions: {len(metadata_xml.get('figure_captions', []))}")
        if metadata_xml.get('figure_captions'):
            print(f"First figure caption (first 100 chars): {metadata_xml['figure_captions'][0][:100]}...")
        print(f"Number of table captions: {len(metadata_xml.get('table_captions', []))}")
        if metadata_xml.get('table_captions'):
            print(f"First table caption (first 100 chars): {metadata_xml['table_captions'][0][:100]}...")

    except FileNotFoundError:
        print(f"Error: pmc_example.xml not found at '{xml_file_path}'. Skipping XML test.")
    except Exception as e:
        print(f"An error occurred during XML testing: {e}")

    print("\n" + "="*50 + "\n")

    # Test JSON parsing
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_content = json.load(f)

        print("--- Parsing JSON ---")
        plain_text_json, metadata_json = parse_europe_pmc_json(json_content)

        print("\n--- JSON Plain Text (first 500 chars) ---")
        print(plain_text_json[:500] + "...")

        print("\n--- JSON Structured Metadata (sample) ---")
        print(f"Number of sections: {len(metadata_json.get('sections', []))}")
        if metadata_json.get('sections') and len(metadata_json['sections']) > 0:
            print(f"First section title: {metadata_json['sections'][0].get('title')}")
            if metadata_json['sections'][0].get('paragraphs') and len(metadata_json['sections'][0]['paragraphs']) > 0:
                print(f"First paragraph of first section (first 100 chars): {metadata_json['sections'][0]['paragraphs'][0][:100]}...")
            else:
                print(f"First section '{metadata_json['sections'][0].get('title')}' has no paragraphs.")
        else:
            print("No sections found in JSON.")

        print(f"Number of figure captions: {len(metadata_json.get('figure_captions', []))}")
        if metadata_json.get('figure_captions'):
            print(f"First figure caption (first 100 chars): {metadata_json['figure_captions'][0][:100]}...")
        print(f"Number of table captions: {len(metadata_json.get('table_captions', []))}")
        if metadata_json.get('table_captions'):
            print(f"First table caption (first 100 chars): {metadata_json['table_captions'][0][:100]}...")
            
    except FileNotFoundError:
        print(f"Error: europe_pmc_example.json not found at '{json_file_path}'. Skipping JSON test.")
    except Exception as e:
        print(f"An error occurred during JSON testing: {e}")
