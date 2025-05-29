import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock, ANY # Added ANY

import pytest

# Ensure tools package is discoverable if tests are run from project root
import sys
_project_root = Path(__file__).resolve().parents[1] 
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tools import elink_pubmed
from tools import resolve_pdf_link # For resolve_content and its result types
from tools.retrieve_full_text import retrieve_full_texts_for_dois, parse_pdf_to_markdown

# --- Fixtures ---

@pytest.fixture
def mock_elink_pubmed_llinks_response():
    """Loads the mocked JSON response for elink_pubmed.get_article_links (llinks)."""
    fixture_path = Path(__file__).parent / "fixtures" / "elink_pubmed_38692467_llinks.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)

@pytest.fixture
def temp_pdf_file():
    """Creates a temporary dummy PDF file and returns its path."""
    # Create a very simple, valid PDF file content
    # This is a minimal PDF structure.
    pdf_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as tmp_file:
        tmp_file.write(pdf_content)
        tmp_file_path = tmp_file.name
    
    yield tmp_file_path # Provide the path to the test
    
    os.remove(tmp_file_path) # Cleanup after the test

# --- Test Cases ---

@pytest.mark.asyncio
async def test_retrieval_via_pubmed_linkout_pdf_success(
    mock_elink_pubmed_llinks_response, 
    temp_pdf_file,
    event_loop # pytest-asyncio provides this
):
    """
    Tests successful full-text retrieval via PubMed LinkOut (llinks) when a PDF is found.
    """
    test_pmid = "38692467"
    test_doi = "10.1234/test.doi.38692467"
    expected_pdf_url_from_fixture = "https://www.example.com/fulltext/38692467.pdf"
    mocked_markdown_content = "This is the mocked markdown from the PDF."

    article_input = {
        "doi": test_doi,
        "pmid": test_pmid,
        "title": "Test Article for PubMed LinkOut",
        "relevance_score": 5 
    }
    query_refiner_output_mock = {
        "query": "test query",
        "refined_queries": ["test refined query"],
        "triaged_articles": [article_input]
    }

    # 1. Mock elink_pubmed.get_article_links (the new llinks one)
    #    It should return a list of URLs, including the one we expect to succeed.
    #    The fixture contains multiple URLs; the test logic will try them.
    #    We need to extract the URLs that would be returned by the actual get_article_links
    #    based on the fixture content and its filtering logic.
    
    # Simulate the URL extraction logic of the new get_article_links based on the fixture
    extracted_urls_from_fixture = []
    link_keywords_for_sim = ["Free full text", "Full Text", "PubMed Central", "Europe PMC", "PDF"]
    if mock_elink_pubmed_llinks_response.get("linksets"):
        idurls_data = mock_elink_pubmed_llinks_response["linksets"][0].get("idurls", {}).get(test_pmid, {})
        objurls = idurls_data.get("objurls", [])
        for obj_url_entry in objurls:
            url_str = obj_url_entry.get("url", {}).get("$")
            if not url_str: continue
            link_text = obj_url_entry.get("linktext", "").lower()
            provider_name = obj_url_entry.get("provider", {}).get("name", "").lower()
            if any(keyword.lower() in link_text for keyword in link_keywords_for_sim) or \
               any(keyword.lower() in provider_name for keyword in link_keywords_for_sim):
                if url_str not in extracted_urls_from_fixture:
                    extracted_urls_from_fixture.append(url_str)
    
    # Ensure our target PDF URL is in the list returned by the mock
    assert expected_pdf_url_from_fixture in extracted_urls_from_fixture

    with patch("tools.retrieve_full_text.get_from_cache", AsyncMock(return_value=None)) as mock_get_cache_pubmed, \
         patch("tools.retrieve_full_text.write_to_cache", AsyncMock()) as mock_write_cache_pubmed, \
         patch("tools.elink_pubmed.get_article_links", AsyncMock(return_value=extracted_urls_from_fixture)) as mock_get_llinks, \
         patch("tools.retrieve_full_text.resolve_pdf_link.resolve_content") as mock_resolve_content, \
         patch("tools.retrieve_full_text.parse_pdf_to_markdown", AsyncMock(return_value=mocked_markdown_content)) as mock_parse_pdf, \
         patch("tools.retrieve_full_text.retrieve_unpaywall.get_unpaywall_oa_url", AsyncMock(return_value=None)) as mock_unpaywall, \
         patch("tools.retrieve_full_text.elink_pubmed._get_article_links_by_id_type_xml", AsyncMock(return_value=[])) as mock_elink_prlinks, \
         patch("tools.retrieve_full_text.retrieve_europepmc.fetch_europepmc", AsyncMock(return_value=None)) as mock_europepmc, \
         patch("tools.retrieve_full_text.retrieve_pmc.fetch_pmc_xml", AsyncMock(return_value=None)) as mock_pmc_xml:

        # 2. Configure mock_resolve_content:
        #    - For the expected PDF URL, return a FileResult pointing to temp_pdf_file.
        #    - For other URLs, return Failure or basic HTML.
        def resolve_content_side_effect(url, client, original_doi_for_referer, session_cookies):
            if url == expected_pdf_url_from_fixture:
                # Simulate cookies being set by resolve_content if needed for _update_session_cookies
                return resolve_pdf_link.FileResult(type="file", path=temp_pdf_file)
            else: # For any other URL from LinkOut, simulate failure or non-PDF content
                return resolve_pdf_link.Failure(type="failure", reason="Mocked failure for non-target URL")
        
        mock_resolve_content.side_effect = resolve_content_side_effect
        
        # --- Act ---
        result_dict = await retrieve_full_texts_for_dois(query_refiner_output_mock)

        # --- Assert ---
        assert result_dict is not None
        assert "triaged_articles" in result_dict
        processed_articles = result_dict["triaged_articles"]
        assert len(processed_articles) == 1
        
        article_result = processed_articles[0]
        assert article_result.get("fulltext_retrieval_status") == "success"
        # The string "PubMed_LinkOut_llinks_PDF" is a cache source key, not necessarily in the user message.
        # The status message itself is more important.
        # assert "PubMed_LinkOut_llinks_PDF" in article_result.get("fulltext_retrieval_message", "")
        assert f"Retrieved PDF via PubMed LinkOut (llinks) ({expected_pdf_url_from_fixture})" in article_result.get("fulltext_retrieval_message", "")
        assert article_result.get("fulltext") == mocked_markdown_content

        # Assert that mocks were called as expected
        mock_get_llinks.assert_called_once_with(pmid=test_pmid)
        
        # Check calls to resolve_content: it should be called for URLs from LinkOut until success
        # The exact number of calls depends on the order in extracted_urls_from_fixture
        # and which one is `expected_pdf_url_from_fixture`.
        # We need to ensure it was called with expected_pdf_url_from_fixture.
        
        # Construct a list of actual calls to mock_resolve_content
        resolve_content_called_urls = [call_args[0][0] for call_args in mock_resolve_content.call_args_list]
        assert expected_pdf_url_from_fixture in resolve_content_called_urls
        
        # Ensure parse_pdf_to_markdown was called with the path of our temp_pdf_file
        mock_parse_pdf.assert_called_once_with(temp_pdf_file)

        # Ensure prior methods were "tried" and failed (or returned no useful data)
        mock_europepmc.assert_called_once()
        # PMCID is not in input, so PMC XML won't be called if DOI/PMID conversion to PMCID fails or is not mocked
        # For this test, let's assume no PMCID, so mock_pmc_xml is not called.
        # If PMCID logic were to be tested, mock _convert_to_pmid or provide PMCID.
        # mock_pmc_xml.assert_called_once() # Or not_called if no PMCID
        mock_elink_prlinks.assert_called_once() # Called with DOI or PMID
        mock_unpaywall.assert_called_once_with(test_doi, session=ANY)


        # Check if the PDF file was created (it should be, by resolve_content if not mocked away)
        # The `resolve_content` in `retrieve_full_text.py` saves files to `workspace/downloads`.
        # Our mocked `resolve_content` returns `FileResult(type="file", path=temp_pdf_file)`.
        # The actual saving to `workspace/downloads` is part of the real `resolve_content`.
        # Here, we are testing the logic *after* `resolve_content` provides a path.
        # The `temp_pdf_file` fixture ensures a file exists at that path for `parse_pdf_to_markdown`.
        # If the task implies testing the *debug* feature of "PDF bytes written to disk when --debug is on",
        # that would require a different setup, possibly involving CLI arguments and checking `workspace/downloads`.
        # This test focuses on the successful retrieval path via LinkOut.
        # The `resolve_content` function itself is responsible for file saving.
        # We are mocking `resolve_content` here, so we don't directly test its internal saving.
        # We test that if `resolve_content` *provides* a file path, it's used.
        assert os.path.exists(temp_pdf_file) # Confirms fixture worked and file was available for parsing.
