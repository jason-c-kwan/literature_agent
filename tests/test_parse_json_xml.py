import unittest
import json
import xml.etree.ElementTree as ET
import os
from typing import Dict, Any

# Adjust import path based on how tests are run.
# If run from root of repo: from tools.parse_json_xml import ...
# If tools and tests are sibling packages: from ..tools.parse_json_xml import ...
# Assuming tests are run from the root directory of the project:
from tools.parse_json_xml import (
    # _extract_text_xml, # Old function, not directly tested here anymore for new functionality
    # _extract_text_json, # Old function
    parse_pmc_xml,
    parse_europe_pmc_json,
    _render_elements_to_markdown_string,
    _finalize_elements_for_json_output,
    _process_xml_children_to_markdown_text, # For inline tests
    _process_json_children_to_markdown_text # For inline tests
)

# Get the directory of the current test script
TEST_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the workspace directory (assuming 'tests' and 'workspace' are siblings of the root, or 'tools' and 'workspace' are siblings)
WORKSPACE_DIR = os.path.join(TEST_SCRIPT_DIR, "../workspace")
PMC_EXAMPLE_XML_PATH = os.path.join(WORKSPACE_DIR, "pmc_example.xml")
EUROPE_PMC_EXAMPLE_JSON_PATH = os.path.join(WORKSPACE_DIR, "europe_pmc_example.json")


class TestInlineMarkdownConversion(unittest.TestCase):
    def test_xml_inline_formatting(self):
        xml_p_bold = ET.fromstring("<p>This is <b>bold</b> text.</p>")
        self.assertEqual(_process_xml_children_to_markdown_text(xml_p_bold), "This is **bold** text.")
        
        xml_p_italic = ET.fromstring("<p>This is <i>italic</i> text.</p>")
        self.assertEqual(_process_xml_children_to_markdown_text(xml_p_italic), "This is *italic* text.")

        xml_p_mixed = ET.fromstring("<p><b>Boldly</b> go where <i>no one</i> has gone <sup>before</sup>.</p>")
        self.assertEqual(_process_xml_children_to_markdown_text(xml_p_mixed), "**Boldly** go where *no one* has gone ^before^.")

    def test_json_inline_formatting(self):
        json_p_bold = {"#text": "This is ", "bold": "bold", "italic": " text."} # Example structure
        # Note: _process_json_children_to_markdown_text processes based on keys like 'bold', 'italic'
        # The order might not be guaranteed if multiple keys are at the same level without a specific order array.
        # For this test, assuming a simple case or that the function handles it.
        # A more robust test would use a structure that implies order if the function supports it.
        # Current _process_json_children_to_markdown_text joins parts, order depends on dict iteration.
        # Let's test parts:
        processed_text = _process_json_children_to_markdown_text(json_p_bold)
        self.assertIn("This is", processed_text)
        self.assertIn("**bold**", processed_text)
        self.assertIn("*text.*", processed_text) # Assuming italic key is processed

        json_simple_italic = {"italic": "emphasized"}
        self.assertEqual(_process_json_children_to_markdown_text(json_simple_italic), "*emphasized*")


class TestParsePmcXmlToMarkdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(PMC_EXAMPLE_XML_PATH):
            raise unittest.SkipTest(f"Example XML file not found: {PMC_EXAMPLE_XML_PATH}")
        with open(PMC_EXAMPLE_XML_PATH, "r", encoding="utf-8") as f:
            cls.full_xml_content = f.read()
        
        # Parse once for all tests in this class
        cls.parsed_elements = parse_pmc_xml(cls.full_xml_content)
        cls.markdown_output = _render_elements_to_markdown_string(cls.parsed_elements)
        cls.json_output_str = json.dumps(_finalize_elements_for_json_output(cls.parsed_elements), indent=2)
        cls.json_output_obj = json.loads(cls.json_output_str)


    def test_full_pmc_example_markdown_generation(self):
        self.assertTrue(len(self.markdown_output) > 500, "Markdown output seems too short.")
        self.assertIn("# Abstract", self.markdown_output) # Abstract should be H1
        # Check for some known text from the example
        self.assertIn("After the first aminoglycoside antibiotic streptomycin", self.markdown_output)
        self.assertIn("*P. aeruginosa*", self.markdown_output) # Changed from ** to * for italic

    def test_heading_levels_xml(self):
        # Check abstract heading
        abstract_heading = next((el for el in self.parsed_elements if el.get("type") == "heading" and el.get("text_content") == "Abstract"), None)
        self.assertIsNotNone(abstract_heading, "Abstract heading not found in parsed elements.")
        self.assertEqual(abstract_heading.get("level"), 1, "Abstract heading level is not 1.")

        # Check for a known section title and its expected level (e.g. Introduction is often level 1 after abstract, or level 2 if abstract is H1)
        # The current logic in parse_pmc_xml starts body sections at depth 0, so a top-level <sec> in body gets level 1.
        # If Abstract is H1, then "1 Introduction" should also be H1.
        intro_heading = next((el for el in self.parsed_elements if el.get("type") == "heading" and "Introduction" in el.get("text_content","")), None)
        self.assertIsNotNone(intro_heading, "Introduction heading not found.")
        # The example XML has "1 Introduction" as a top-level section in body.
        # parse_pmc_xml calls _xml_to_markdown_elements with depth 0 for body children.
        # _xml_to_markdown_elements increments depth for <sec>, so title becomes depth+1 = 1.
        self.assertEqual(intro_heading.get("level"), 1, f"Introduction heading level is not 1, it is {intro_heading.get('level')}")


    def test_list_conversion_xml(self):
        # Test with inline XML containing lists
        xml_with_list = """
        <article>
            <body>
                <list list-type="bullet">
                    <list-item><p>Bullet 1</p></list-item>
                    <list-item><p>Bullet 2 <italic>italic</italic></p></list-item>
                </list>
                <list list-type="ordered">
                    <list-item><p>Ordered 1</p></list-item>
                    <list-item><p>Ordered <b>bold</b></p></list-item>
                </list>
            </body>
        </article>
        """
        parsed_elements = parse_pmc_xml(xml_with_list)
        
        list_element_found = any(el.get("type") == "list" for el in parsed_elements)
        self.assertTrue(list_element_found, "No list elements found in parsed output from inline XML.")
        
        markdown_output = _render_elements_to_markdown_string(parsed_elements)
        # The first item won't have a leading \n if the list is the first element.
        self.assertTrue(markdown_output.startswith("- Bullet 1"), "Markdown output should start with the first bullet item.")
        self.assertIn("\n\n- Bullet 2 *italic*", markdown_output, "Unordered list item 2 with italic not found or incorrect.")
        self.assertIn("\n\n1. Ordered 1", markdown_output, "Ordered list item 1 not found or incorrect.")
        self.assertIn("\n\n1. Ordered **bold**", markdown_output, "Ordered list item 2 with bold not found or incorrect.")

    def test_table_conversion_xml(self):
        # This test will continue to use self.parsed_elements from the main example file.
        # If pmc_example.xml doesn't have tables, this might also fail or need adjustment.
        # For now, focusing on list conversion.
        table_element_found = any(el.get("type") == "table" for el in self.parsed_elements)
        self.assertTrue(table_element_found, "No table elements found in parsed output.")
        self.assertIn("| --- |", self.markdown_output) # Markdown table syntax

    def test_json_output_format_xml(self):
        self.assertIsInstance(self.json_output_obj, list)
        self.assertTrue(len(self.json_output_obj) > 0)
        first_element = self.json_output_obj[0]
        self.assertIn("type", first_element)
        self.assertIn("text", first_element)
        if first_element["type"] == "heading":
            self.assertIn("level", first_element)
        
        # Check if abstract is correctly formatted in JSON
        abstract_json_el = next((el for el in self.json_output_obj if el.get("type") == "heading" and el.get("text") == "# Abstract"), None)
        self.assertIsNotNone(abstract_json_el)
        self.assertEqual(abstract_json_el.get("level"), 1)

    def test_malformed_xml_error_handling(self):
        malformed_xml_str = "<article><body><sec><title>Test</title><p>Text<body></article>" # Unclosed sec, body
        elements = parse_pmc_xml(malformed_xml_str)
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["type"], "error")
        self.assertIn("XML Parsing Error", elements[0]["text_content"])

    def test_simple_nested_sections_xml(self):
        xml_str = """
        <article>
            <body>
                <sec><title>Section 1</title>
                    <p>P1</p>
                    <sec><title>Section 1.1</title><p>P1.1</p></sec>
                </sec>
            </body>
        </article>
        """
        elements = parse_pmc_xml(xml_str)
        # Expected: H1 Section 1, P P1, H2 Section 1.1, P P1.1
        
        h1 = next(el for el in elements if el["type"] == "heading" and el["text_content"] == "Section 1")
        p1 = next(el for el in elements if el["type"] == "paragraph" and el["text_content"] == "P1")
        h2 = next(el for el in elements if el["type"] == "heading" and el["text_content"] == "Section 1.1")
        p1_1 = next(el for el in elements if el["type"] == "paragraph" and el["text_content"] == "P1.1")

        self.assertEqual(h1["level"], 1)
        self.assertEqual(h2["level"], 2)
        
        # Check order roughly
        self.assertTrue(elements.index(h1) < elements.index(p1))
        self.assertTrue(elements.index(p1) < elements.index(h2))
        self.assertTrue(elements.index(h2) < elements.index(p1_1))


class TestParseEuropePmcJsonToMarkdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(EUROPE_PMC_EXAMPLE_JSON_PATH):
            raise unittest.SkipTest(f"Example JSON file not found: {EUROPE_PMC_EXAMPLE_JSON_PATH}")
        with open(EUROPE_PMC_EXAMPLE_JSON_PATH, "r", encoding="utf-8") as f:
            cls.full_json_data = json.load(f)

        cls.parsed_elements = parse_europe_pmc_json(cls.full_json_data)
        cls.markdown_output = _render_elements_to_markdown_string(cls.parsed_elements)
        cls.json_output_str = json.dumps(_finalize_elements_for_json_output(cls.parsed_elements), indent=2)
        cls.json_output_obj = json.loads(cls.json_output_str)

    def test_full_europe_pmc_example_markdown_generation(self):
        self.assertTrue(len(self.markdown_output) > 500, "Markdown output seems too short.")
        self.assertIn("# Abstract", self.markdown_output)
        self.assertIn("After the first aminoglycoside antibiotic streptomycin", self.markdown_output)
        # Example JSON has bold/italic, check for that
        # e.g. "<i>P. aeruginosa</i>" should become "*P. aeruginosa*"
        self.assertIn("*P. aeruginosa*", self.markdown_output) 

    def test_heading_levels_json(self):
        abstract_heading = next((el for el in self.parsed_elements if el.get("type") == "heading" and el.get("text_content") == "Abstract"), None)
        self.assertIsNotNone(abstract_heading)
        self.assertEqual(abstract_heading.get("level"), 1)

        # The example JSON has "1 Introduction" section.
        # parse_europe_pmc_json calls _json_to_markdown_elements with depth 0 for body.
        # _json_to_markdown_elements increments depth for 'sec' elements.
        intro_heading = next((el for el in self.parsed_elements if el.get("type") == "heading" and "Introduction" in el.get("text_content","")), None)
        self.assertIsNotNone(intro_heading, "Introduction heading not found.")
        self.assertEqual(intro_heading.get("level"), 1, f"Introduction heading level is not 1, it is {intro_heading.get('level')}")


    def test_list_conversion_json(self):
        # Test with inline JSON containing lists
        json_with_list = {
            "article": {
                "body": { # parse_europe_pmc_json expects body node
                    "list": [
                        {
                            "@list-type": "bullet",
                            "list-item": [
                                {"p": "JSON Bullet 1"},
                                {"p": {"#text": "JSON Bullet 2 ", "italic": "italic"}}
                            ]
                        },
                        {
                            "@list-type": "ordered",
                            "list-item": [
                                {"p": "JSON Ordered 1"},
                                {"p": {"#text": "JSON Ordered 2 ", "bold": "bold"}}
                            ]
                        }
                    ]
                }
            }
        }
        parsed_elements = parse_europe_pmc_json(json_with_list)

        list_element_found = any(el.get("type") == "list" for el in parsed_elements)
        self.assertTrue(list_element_found, "No list elements found in parsed output from inline JSON.")

        markdown_output = _render_elements_to_markdown_string(parsed_elements)
        # The first item won't have a leading \n if the list is the first element.
        self.assertTrue(markdown_output.startswith("- JSON Bullet 1"), "Markdown output should start with the first JSON bullet item.")
        self.assertIn("\n\n- JSON Bullet 2 *italic*", markdown_output, "JSON Unordered list item 2 with italic not found or incorrect.")
        self.assertIn("\n\n1. JSON Ordered 1", markdown_output, "JSON Ordered list item 1 not found or incorrect.")
        self.assertIn("\n\n1. JSON Ordered 2 **bold**", markdown_output, "JSON Ordered list item 2 with bold not found or incorrect.")


    def test_json_output_format_json(self): # Test the --format json output for JSON input
        # This test will continue to use self.json_output_obj from the main example file.
        self.assertIsInstance(self.json_output_obj, list)
        self.assertTrue(len(self.json_output_obj) > 0)
        first_element = self.json_output_obj[0]
        self.assertIn("type", first_element)
        self.assertIn("text", first_element)
        if first_element["type"] == "heading":
            self.assertIn("level", first_element)

        abstract_json_el = next((el for el in self.json_output_obj if el.get("type") == "heading" and el.get("text") == "# Abstract"), None)
        self.assertIsNotNone(abstract_json_el)
        self.assertEqual(abstract_json_el.get("level"), 1)

    def test_invalid_json_structure_error_handling(self):
        invalid_data: Dict[str, Any] = {"data": "no article key"}
        elements = parse_europe_pmc_json(invalid_data)
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["type"], "error")
        self.assertEqual(elements[0]["text_content"], "Invalid JSON structure: 'article' key missing")

    def test_simple_nested_sections_json(self):
        json_data = {
            "article": {
                "body": {
                    "sec": [
                        {"title": "Section 1", "p": "P1", 
                         "sec": [{"title": "Section 1.1", "p": "P1.1"}]}
                    ]
                }
            }
        }
        elements = parse_europe_pmc_json(json_data)
        # Expected: H1 Section 1, P P1, H2 Section 1.1, P P1.1 (Abstract is added by default if not present)
        # Filter out auto-added Abstract for this specific test if it's there
        elements_no_abstract = [el for el in elements if el.get("text_content") != "Abstract"]

        h1 = next(el for el in elements_no_abstract if el["type"] == "heading" and el["text_content"] == "Section 1")
        p1 = next(el for el in elements_no_abstract if el["type"] == "paragraph" and el["text_content"] == "P1")
        h2 = next(el for el in elements_no_abstract if el["type"] == "heading" and el["text_content"] == "Section 1.1")
        p1_1 = next(el for el in elements_no_abstract if el["type"] == "paragraph" and el["text_content"] == "P1.1")

        self.assertEqual(h1["level"], 1) # Since body starts at depth 0, first sec is 1
        self.assertEqual(h2["level"], 2) # Nested sec is 2
        
        self.assertTrue(elements_no_abstract.index(h1) < elements_no_abstract.index(p1))
        self.assertTrue(elements_no_abstract.index(p1) < elements_no_abstract.index(h2))
        self.assertTrue(elements_no_abstract.index(h2) < elements_no_abstract.index(p1_1))


if __name__ == '__main__':
    unittest.main()
