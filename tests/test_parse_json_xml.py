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
    _extract_text_xml,
    _extract_text_json,
    parse_pmc_xml,
    parse_europe_pmc_json
)

# Get the directory of the current test script
TEST_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the workspace directory (assuming 'tests' and 'workspace' are siblings of the root, or 'tools' and 'workspace' are siblings)
WORKSPACE_DIR = os.path.join(TEST_SCRIPT_DIR, "../workspace")
PMC_EXAMPLE_XML_PATH = os.path.join(WORKSPACE_DIR, "pmc_example.xml")
EUROPE_PMC_EXAMPLE_JSON_PATH = os.path.join(WORKSPACE_DIR, "europe_pmc_example.json")


class TestExtractTextXml(unittest.TestCase):
    def test_simple_text(self):
        element = ET.fromstring("<p>Hello World</p>")
        self.assertEqual(_extract_text_xml(element), "Hello World")

    def test_nested_elements(self):
        element = ET.fromstring("<p>Hello <b>World</b>!</p>")
        self.assertEqual(_extract_text_xml(element), "Hello World !") # Note: space handling by ET

    def test_text_in_tail(self):
        element = ET.fromstring("<p><i>italic</i> followed by text.</p>")
        self.assertEqual(_extract_text_xml(element), "italic followed by text.")
    
    def test_mixed_content(self):
        xml_str = "<p>This is <sub>subscript</sub> and <sup>superscript</sup> text with <xref>cross-ref</xref>.</p>"
        element = ET.fromstring(xml_str)
        self.assertEqual(_extract_text_xml(element), "This is subscript and superscript text with cross-ref .")

    def test_empty_element(self):
        element = ET.fromstring("<p/>")
        self.assertEqual(_extract_text_xml(element), "")

    def test_none_element(self):
        self.assertEqual(_extract_text_xml(None), "")


class TestExtractTextJson(unittest.TestCase):
    def test_simple_string(self):
        self.assertEqual(_extract_text_json("Hello World"), "Hello World")

    def test_list_of_strings(self):
        self.assertEqual(_extract_text_json(["Hello", "World"]), "Hello World")

    def test_dict_with_hash_text(self):
        data = {"#text": "Main text"}
        self.assertEqual(_extract_text_json(data), "Main text")

    def test_dict_with_known_inline_tags(self):
        data = {"p": {"italic": "emphasized"}}
        self.assertEqual(_extract_text_json(data), "emphasized")
    
    def test_complex_paragraph_like_json(self):
        # Simplified from Europe PMC JSON structure for a paragraph
        data = {
            "#text": "Streptomycin was the first discovered aminoglycoside antibiotic (AGA) to be used for tuberculosis treatment in the mid-1940s (",
            "xref": [
                {"#text": "Schatz et al., 2005"},
                {"#text": "Figure 1"}
            ],
            "italic": "P. aeruginosa" 
            # In real JSON, xref might be an object not a list if single, and text might be split more
        }
        # Current _extract_text_json might not perfectly replicate complex ordering without more specific rules
        # For now, checking if all text parts are present
        extracted = _extract_text_json(data)
        self.assertIn("Streptomycin was the first", extracted)
        self.assertIn("Schatz et al., 2005", extracted)
        self.assertIn("Figure 1", extracted)
        self.assertIn("P. aeruginosa", extracted)

    def test_empty_data(self):
        self.assertEqual(_extract_text_json({}), "")
        self.assertEqual(_extract_text_json([]), "")
        self.assertEqual(_extract_text_json(None), "") # Though type hint is Any, practically it's str, list, dict

    def test_json_caption_like_structure(self):
        data = {"p": "Timeline of aminoglycoside antibiotics development."}
        self.assertEqual(_extract_text_json(data), "Timeline of aminoglycoside antibiotics development.")
        
        data_with_label = {"label": "FIGURE 1", "caption": {"p": "Timeline..."}}
        # _extract_text_json is generic, so it will pick up "FIGURE 1" and "Timeline..."
        # The main parsing functions are responsible for selecting the correct fields.
        extracted = _extract_text_json(data_with_label)
        self.assertIn("FIGURE 1", extracted)
        self.assertIn("Timeline...", extracted)


class TestParsePmcXml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(PMC_EXAMPLE_XML_PATH):
            raise unittest.SkipTest(f"Example XML file not found: {PMC_EXAMPLE_XML_PATH}")
        with open(PMC_EXAMPLE_XML_PATH, "r", encoding="utf-8") as f:
            cls.full_xml_content = f.read()

    def test_full_pmc_example_parsing(self):
        plain_text, metadata = parse_pmc_xml(self.full_xml_content)
        self.assertTrue(len(plain_text) > 1000, "Plain text seems too short.")
        # Check for content from the abstract, which is part of plain_text
        self.assertIn("After the first aminoglycoside antibiotic streptomycin", plain_text)
        self.assertIn("aminoglycoside antibiotics", plain_text.lower()) # Common term

        self.assertTrue(len(metadata["sections"]) > 0, "No sections extracted.")
        self.assertEqual(metadata["sections"][0]["title"], "Abstract")
        self.assertTrue(len(metadata["sections"][0]["paragraphs"][0]) > 50)
        
        # Example paper has an introduction section
        intro_section_found = any(s["title"].strip().lower() == "1 introduction" for s in metadata["sections"])
        self.assertTrue(intro_section_found, "Introduction section not found or title mismatch.")

        self.assertTrue(len(metadata["figure_captions"]) > 0, "No figure captions extracted.")
        self.assertIn("Timeline of aminoglycoside antibiotics development.", metadata["figure_captions"][0])

        self.assertTrue(len(metadata["table_captions"]) > 0, "No table captions extracted.")
        self.assertIn("Resistance rates of common gram-positive cocci to gentamicin.", metadata["table_captions"][0])

    def test_malformed_xml(self):
        malformed_xml = "<article><body><sec><title>Test</title><p>Text<body></article>" # Unclosed sec, body
        plain_text, metadata = parse_pmc_xml(malformed_xml)
        self.assertEqual(plain_text, "")
        self.assertIn("error", metadata)
        self.assertIn("XML Parsing Error", metadata["error"])

    def test_minimal_xml(self):
        xml_str = """
        <article>
            <front><article-meta><abstract><p>This is abstract.</p></abstract></article-meta></front>
            <body>
                <sec><title>Section 1</title><p>Paragraph 1.1</p><p>Paragraph 1.2</p></sec>
                <sec><title>Section 2</title><p>Paragraph 2.1</p>
                    <fig id="f1"><label>Fig 1</label><caption><p>Figure caption 1</p></caption></fig>
                </sec>
                <table-wrap id="t1"><label>Table 1</label><caption><p>Table caption 1</p></caption></table-wrap>
            </body>
        </article>
        """
        plain_text, metadata = parse_pmc_xml(xml_str)
        
        expected_text_parts = [
            "This is abstract.",
            "Paragraph 1.1", "Paragraph 1.2",
            "Paragraph 2.1"
        ]
        for part in expected_text_parts:
            self.assertIn(part, plain_text)

        self.assertEqual(len(metadata["sections"]), 3) # Abstract + 2 sections
        self.assertEqual(metadata["sections"][0]["title"], "Abstract")
        self.assertEqual(metadata["sections"][0]["paragraphs"], ["This is abstract."])
        self.assertEqual(metadata["sections"][1]["title"], "Section 1")
        self.assertEqual(metadata["sections"][1]["paragraphs"], ["Paragraph 1.1", "Paragraph 1.2"])
        
        self.assertEqual(len(metadata["figure_captions"]), 1)
        self.assertEqual(metadata["figure_captions"][0], "Fig 1 Figure caption 1") # Label + caption text

        self.assertEqual(len(metadata["table_captions"]), 1)
        self.assertEqual(metadata["table_captions"][0], "Table 1 Table caption 1")


class TestParseEuropePmcJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(EUROPE_PMC_EXAMPLE_JSON_PATH):
            raise unittest.SkipTest(f"Example JSON file not found: {EUROPE_PMC_EXAMPLE_JSON_PATH}")
        with open(EUROPE_PMC_EXAMPLE_JSON_PATH, "r", encoding="utf-8") as f:
            cls.full_json_data = json.load(f)

    def test_full_europe_pmc_example_parsing(self):
        plain_text, metadata = parse_europe_pmc_json(self.full_json_data)
        self.assertTrue(len(plain_text) > 1000, "Plain text seems too short.")
        # Title is not part of abstract/body text directly, so check for abstract content
        self.assertIn("After the first aminoglycoside antibiotic streptomycin", plain_text) 
        self.assertIn("aminoglycoside antibiotics", plain_text.lower())

        self.assertTrue(len(metadata["sections"]) > 0, "No sections extracted.")
        self.assertEqual(metadata["sections"][0]["title"], "Abstract") # First section should be abstract
        self.assertTrue(len(metadata["sections"][0]["paragraphs"][0]) > 50)
        
        intro_section_found = any(s["title"].strip().lower() == "1 introduction" for s in metadata["sections"])
        self.assertTrue(intro_section_found, "Introduction section not found or title mismatch.")

        self.assertTrue(len(metadata["figure_captions"]) > 0, "No figure captions extracted.")
        # JSON structure for caption is often {"label": "FIGURE 1", "p": "Text"}
        # _extract_text_json will combine these.
        self.assertIn("FIGURE 1 Timeline of aminoglycoside antibiotics development.", metadata["figure_captions"][0])

        self.assertTrue(len(metadata["table_captions"]) > 0, "No table captions extracted.")
        self.assertIn("TABLE 1 Resistance rates of common gram-positive cocci to gentamicin.", metadata["table_captions"][0])

    def test_invalid_json_structure(self):
        invalid_data: Dict[str, Any] = {"data": "no article key"}
        plain_text, metadata = parse_europe_pmc_json(invalid_data)
        self.assertEqual(plain_text, "")
        self.assertIn("error", metadata)
        self.assertEqual(metadata["error"], "Invalid JSON structure: 'article' key missing")

    def test_minimal_json(self):
        json_data = {
            "article": {
                "front": {"article-meta": {"abstract": {"p": "This is abstract."}}},
                "body": {
                    "sec": [
                        {"title": "Section 1", "p": ["Paragraph 1.1", "Paragraph 1.2"]},
                        {"title": "Section 2", "p": "Paragraph 2.1", 
                         "fig": [{"label": "Fig 1", "caption": {"p": "Figure caption 1"}}]},
                        {"table-wrap": [{"label": "Table 1", "caption": {"p": "Table caption 1"}}]}
                    ]
                }
            }
        }
        plain_text, metadata = parse_europe_pmc_json(json_data)
        
        expected_text_parts = [
            "This is abstract.",
            "Paragraph 1.1", "Paragraph 1.2",
            "Paragraph 2.1"
        ]
        for part in expected_text_parts:
            self.assertIn(part, plain_text)

        self.assertEqual(len(metadata["sections"]), 3) # Abstract + 2 content sections
        self.assertEqual(metadata["sections"][0]["title"], "Abstract")
        self.assertEqual(metadata["sections"][0]["paragraphs"], ["This is abstract."])
        self.assertEqual(metadata["sections"][1]["title"], "Section 1")
        self.assertEqual(metadata["sections"][1]["paragraphs"], ["Paragraph 1.1", "Paragraph 1.2"])
        
        self.assertEqual(len(metadata["figure_captions"]), 1)
        self.assertIn("Fig 1", metadata["figure_captions"][0])
        self.assertIn("Figure caption 1", metadata["figure_captions"][0])

        self.assertEqual(len(metadata["table_captions"]), 1)
        self.assertIn("Table 1", metadata["table_captions"][0])
        self.assertIn("Table caption 1", metadata["table_captions"][0])


if __name__ == '__main__':
    unittest.main()
