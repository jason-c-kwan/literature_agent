import asyncio
import json
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from tools import retrieve_full_text # Module to test

# Basic article data for testing
SAMPLE_ARTICLE_DOI_ONLY = {"doi": "10.1234/test.doi", "relevance_score": 5}
SAMPLE_ARTICLE_FULL_IDS = {
    "doi": "10.1234/test.doi.full",
    "pmid": "12345678",
    "pmcid": "9876543", # Numeric part
    "relevance_score": 5
}
SAMPLE_ARTICLE_NO_DOI = {"title": "No DOI article", "relevance_score": 5}


class TestGetFullTextForDoi(unittest.IsolatedAsyncioTestCase):

    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.write_to_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_europepmc.fetch_europepmc', new_callable=AsyncMock)
    @patch('tools.parse_json_xml.parse_europe_pmc_json')
    @patch('tools.parse_json_xml._render_elements_to_markdown_string')
    @patch('tools.retrieve_pmc.fetch_pmc_xml', new_callable=AsyncMock)
    @patch('tools.elink_pubmed.get_article_links', new_callable=AsyncMock)
    @patch('tools.retrieve_unpaywall.get_unpaywall_oa_url', new_callable=AsyncMock)
    async def test_retrieval_europepmc_success(
        self, mock_get_unpaywall, mock_get_elink, mock_fetch_pmc,
        mock_render_elements, mock_parse_epmc_json, mock_fetch_epmc,
        mock_write_cache, mock_get_cache
    ):
        mock_get_cache.return_value = None # Cache miss
        mock_fetch_epmc.return_value = {"article": {"body": {"sec": [{"p": "EuropePMC content"}]}}}
        mock_parse_epmc_json.return_value = [{"type": "paragraph", "text_content": "EuropePMC content"}]
        mock_render_elements.return_value = "Markdown from EuropePMC"

        markdown, status = await retrieve_full_text.get_full_text_for_doi(SAMPLE_ARTICLE_DOI_ONLY)

        self.assertEqual(markdown, "Markdown from EuropePMC")
        self.assertEqual(status, "Retrieved and parsed from Europe PMC (JSON)")
        mock_fetch_epmc.assert_called_once_with(SAMPLE_ARTICLE_DOI_ONLY["doi"])
        mock_write_cache.assert_called_once_with(
            SAMPLE_ARTICLE_DOI_ONLY["doi"],
            {"status": "success", "markdown": "Markdown from EuropePMC", "source": "EuropePMC_JSON"}
        )
        # Ensure other methods were not called
        mock_fetch_pmc.assert_not_called()
        mock_get_elink.assert_not_called()
        mock_get_unpaywall.assert_not_called()

    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.write_to_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_europepmc.fetch_europepmc', new_callable=AsyncMock)
    @patch('tools.retrieve_pmc.fetch_pmc_xml', new_callable=AsyncMock)
    @patch('tools.elink_pubmed.get_article_links', new_callable=AsyncMock)
    @patch('tools.retrieve_unpaywall.get_unpaywall_oa_url', new_callable=AsyncMock)
    # We also need to mock resolve_content, advanced_scraper, and parse_pdf_to_markdown for a true all_fail
    @patch('tools.resolve_pdf_link.resolve_content', new_callable=AsyncMock)
    @patch('tools.advanced_scraper.scrape_with_fallback', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.parse_pdf_to_markdown', new_callable=AsyncMock) # Mocking the helper in the module
    async def test_retrieval_all_sources_fail(
        self, mock_parse_pdf, mock_scrape_fallback, mock_resolve_content,
        mock_get_unpaywall, mock_get_elink, mock_fetch_pmc, mock_fetch_epmc,
        mock_write_cache, mock_get_cache
    ):
        mock_get_cache.return_value = None # Cache miss
        mock_fetch_epmc.return_value = None
        mock_fetch_pmc.return_value = None
        mock_get_elink.return_value = [] # No links from Elink
        mock_get_unpaywall.return_value = None # No URL from Unpaywall

        # Mocks for when Elink or Unpaywall *do* return URLs but subsequent steps fail
        mock_resolve_content.return_value = retrieve_full_text.resolve_pdf_link.Failure(type="failure", reason="Test failure")
        mock_scrape_fallback.return_value = retrieve_full_text.advanced_scraper.Failure(type="failure", reason="Test failure")
        mock_parse_pdf.return_value = None


        markdown, status = await retrieve_full_text.get_full_text_for_doi(SAMPLE_ARTICLE_FULL_IDS)

        self.assertIsNone(markdown)
        self.assertEqual(status, "Full text not found after all attempts")

        mock_fetch_epmc.assert_called_once_with(SAMPLE_ARTICLE_FULL_IDS["doi"])
        mock_fetch_pmc.assert_called_once_with(f"PMC{SAMPLE_ARTICLE_FULL_IDS['pmcid']}", session=unittest.mock.ANY)
        mock_get_elink.assert_called_once_with(identifier=SAMPLE_ARTICLE_FULL_IDS["pmid"], id_type="pmid")
        mock_get_unpaywall.assert_called_once_with(SAMPLE_ARTICLE_FULL_IDS["doi"], session=unittest.mock.ANY)
        mock_write_cache.assert_called_once_with(
            SAMPLE_ARTICLE_FULL_IDS["doi"],
            {"status": "failure", "reason": "All retrieval methods failed"}
        )

    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    async def test_retrieval_cache_hit_success(self, mock_get_cache):
        cached_data = {"status": "success", "markdown": "Cached Markdown", "source": "TestCache"}
        mock_get_cache.return_value = cached_data

        markdown, status = await retrieve_full_text.get_full_text_for_doi(SAMPLE_ARTICLE_DOI_ONLY)

        self.assertEqual(markdown, "Cached Markdown")
        self.assertEqual(status, "Retrieved from cache")
        mock_get_cache.assert_called_once_with(SAMPLE_ARTICLE_DOI_ONLY["doi"])

    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    async def test_retrieval_cache_hit_failure(self, mock_get_cache):
        cached_data = {"status": "failure", "reason": "Previously failed"}
        mock_get_cache.return_value = cached_data

        # Need to mock all subsequent calls so they are not made
        with patch('tools.retrieve_europepmc.fetch_europepmc', new_callable=AsyncMock) as mock_fetch_epmc, \
             patch('tools.retrieve_pmc.fetch_pmc_xml', new_callable=AsyncMock) as mock_fetch_pmc, \
             patch('tools.elink_pubmed.get_article_links', new_callable=AsyncMock) as mock_get_elink, \
             patch('tools.retrieve_unpaywall.get_unpaywall_oa_url', new_callable=AsyncMock) as mock_get_unpaywall:

            markdown, status = await retrieve_full_text.get_full_text_for_doi(SAMPLE_ARTICLE_DOI_ONLY)

            self.assertIsNone(markdown)
            self.assertEqual(status, "Previously failed")
            mock_get_cache.assert_called_once_with(SAMPLE_ARTICLE_DOI_ONLY["doi"])
            mock_fetch_epmc.assert_not_called() # Ensure no further processing after cached failure

    async def test_retrieval_no_doi_provided(self):
        markdown, status = await retrieve_full_text.get_full_text_for_doi(SAMPLE_ARTICLE_NO_DOI)
        self.assertIsNone(markdown)
        self.assertEqual(status, "DOI not provided in article data")

    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.write_to_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_europepmc.fetch_europepmc', new_callable=AsyncMock)
    @patch('tools.retrieve_pmc.fetch_pmc_xml', new_callable=AsyncMock)
    @patch('tools.parse_json_xml.parse_pmc_xml')
    @patch('tools.parse_json_xml._render_elements_to_markdown_string')
    async def test_retrieval_pmc_xml_success(
        self, mock_render_elements, mock_parse_pmc_xml, mock_fetch_pmc,
        mock_fetch_epmc, mock_write_cache, mock_get_cache
    ):
        mock_get_cache.return_value = None
        mock_fetch_epmc.return_value = None # EuropePMC fails or returns nothing
        
        mock_fetch_pmc.return_value = "<article><body><p>PMC XML Content</p></body></article>"
        mock_parse_pmc_xml.return_value = [{"type": "paragraph", "text_content": "PMC XML Content"}]
        mock_render_elements.return_value = "Markdown from PMC XML"

        article_input = SAMPLE_ARTICLE_FULL_IDS.copy()
        markdown, status = await retrieve_full_text.get_full_text_for_doi(article_input)

        self.assertEqual(markdown, "Markdown from PMC XML")
        self.assertTrue(status.startswith("Retrieved and parsed from PMC XML"))
        mock_fetch_pmc.assert_called_once_with(f"PMC{article_input['pmcid']}", session=unittest.mock.ANY)
        mock_write_cache.assert_called_once_with(
            article_input["doi"],
            {"status": "success", "markdown": "Markdown from PMC XML", "source": "PMC_XML"}
        )

    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.write_to_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_europepmc.fetch_europepmc', new_callable=AsyncMock)
    @patch('tools.retrieve_pmc.fetch_pmc_xml', new_callable=AsyncMock)
    @patch('tools.elink_pubmed.get_article_links', new_callable=AsyncMock)
    @patch('tools.resolve_pdf_link.resolve_content', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.parse_pdf_to_markdown', new_callable=AsyncMock)
    async def test_retrieval_elink_pdf_success(
        self, mock_parse_pdf, mock_resolve_content, mock_get_elink,
        mock_fetch_pmc, mock_fetch_epmc, mock_write_cache, mock_get_cache
    ):
        mock_get_cache.return_value = None
        mock_fetch_epmc.return_value = None
        mock_fetch_pmc.return_value = None
        
        elink_pdf_url = "http://example.com/article.pdf"
        mock_get_elink.return_value = [elink_pdf_url]
        mock_resolve_content.return_value = retrieve_full_text.resolve_pdf_link.FileResult(
            type="file", path="/fake/path/to/article.pdf"
        )
        mock_parse_pdf.return_value = "Markdown from Elink PDF"

        article_input = SAMPLE_ARTICLE_FULL_IDS.copy()
        markdown, status = await retrieve_full_text.get_full_text_for_doi(article_input)

        self.assertEqual(markdown, "Markdown from Elink PDF")
        self.assertTrue(status.startswith(f"Retrieved PDF via Elink ({elink_pdf_url})"))
        mock_get_elink.assert_called_once_with(identifier=article_input["pmid"], id_type="pmid")
        mock_resolve_content.assert_called_once_with(elink_pdf_url, client=unittest.mock.ANY)
        mock_parse_pdf.assert_called_once_with("/fake/path/to/article.pdf")
        mock_write_cache.assert_called_once()


    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.write_to_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_europepmc.fetch_europepmc', new_callable=AsyncMock)
    @patch('tools.retrieve_pmc.fetch_pmc_xml', new_callable=AsyncMock)
    @patch('tools.elink_pubmed.get_article_links', new_callable=AsyncMock)
    @patch('tools.resolve_pdf_link.resolve_content', new_callable=AsyncMock)
    async def test_retrieval_elink_html_success(
        self, mock_resolve_content, mock_get_elink,
        mock_fetch_pmc, mock_fetch_epmc, mock_write_cache, mock_get_cache
    ):
        mock_get_cache.return_value = None
        mock_fetch_epmc.return_value = None
        mock_fetch_pmc.return_value = None
        
        elink_html_url = "http://example.com/article.html"
        mock_get_elink.return_value = [elink_html_url]
        mock_resolve_content.return_value = retrieve_full_text.resolve_pdf_link.HTMLResult(
            type="html", text="HTML content from Elink"
        )

        article_input = SAMPLE_ARTICLE_FULL_IDS.copy()
        markdown, status = await retrieve_full_text.get_full_text_for_doi(article_input)

        self.assertEqual(markdown, "HTML content from Elink")
        self.assertTrue(status.startswith(f"Retrieved HTML via Elink ({elink_html_url})"))
        mock_write_cache.assert_called_once()

    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.write_to_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_europepmc.fetch_europepmc', new_callable=AsyncMock)
    @patch('tools.retrieve_pmc.fetch_pmc_xml', new_callable=AsyncMock)
    @patch('tools.elink_pubmed.get_article_links', new_callable=AsyncMock) # Fails
    @patch('tools.retrieve_unpaywall.get_unpaywall_oa_url', new_callable=AsyncMock)
    @patch('tools.resolve_pdf_link.resolve_content', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.parse_pdf_to_markdown', new_callable=AsyncMock)
    async def test_retrieval_unpaywall_pdf_success(
        self, mock_parse_pdf, mock_resolve_content, mock_get_unpaywall,
        mock_get_elink, mock_fetch_pmc, mock_fetch_epmc, mock_write_cache, mock_get_cache
    ):
        mock_get_cache.return_value = None
        mock_fetch_epmc.return_value = None
        mock_fetch_pmc.return_value = None
        mock_get_elink.return_value = [] # Elink returns no links

        unpaywall_pdf_url = "http://unpaywall.example.com/doc.pdf"
        mock_get_unpaywall.return_value = unpaywall_pdf_url
        mock_resolve_content.return_value = retrieve_full_text.resolve_pdf_link.FileResult(
            type="file", path="/fake/path/to/unpaywall.pdf"
        )
        mock_parse_pdf.return_value = "Markdown from Unpaywall PDF"
        
        article_input = SAMPLE_ARTICLE_FULL_IDS.copy()
        markdown, status = await retrieve_full_text.get_full_text_for_doi(article_input)

        self.assertEqual(markdown, "Markdown from Unpaywall PDF")
        self.assertTrue(status.startswith(f"Retrieved PDF via Unpaywall ({unpaywall_pdf_url})"))
        mock_get_unpaywall.assert_called_once_with(article_input["doi"], session=unittest.mock.ANY)
        mock_resolve_content.assert_called_once_with(unpaywall_pdf_url, client=unittest.mock.ANY)
        mock_parse_pdf.assert_called_once_with("/fake/path/to/unpaywall.pdf")
        mock_write_cache.assert_called_once()

    @patch('tools.retrieve_full_text.get_from_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_full_text.write_to_cache', new_callable=AsyncMock)
    @patch('tools.retrieve_europepmc.fetch_europepmc', new_callable=AsyncMock)
    @patch('tools.retrieve_pmc.fetch_pmc_xml', new_callable=AsyncMock)
    @patch('tools.elink_pubmed.get_article_links', new_callable=AsyncMock)
    @patch('tools.retrieve_unpaywall.get_unpaywall_oa_url', new_callable=AsyncMock)
    @patch('tools.resolve_pdf_link.resolve_content', new_callable=AsyncMock)
    @patch('tools.advanced_scraper.scrape_with_fallback', new_callable=AsyncMock) # Mock this
    async def test_retrieval_elink_resolve_fails_then_advanced_scraper_html_success(
        self, mock_scrape_fallback, mock_resolve_content_elink, mock_get_unpaywall,
        mock_get_elink, mock_fetch_pmc, mock_fetch_epmc, mock_write_cache, mock_get_cache
    ):
        mock_get_cache.return_value = None
        mock_fetch_epmc.return_value = None
        mock_fetch_pmc.return_value = None
        
        elink_url = "http://example.com/article_page"
        mock_get_elink.return_value = [elink_url]
        # resolve_content fails for the elink URL
        mock_resolve_content_elink.return_value = retrieve_full_text.resolve_pdf_link.Failure(type="failure", reason="resolve_content failed")
        # advanced_scraper succeeds with HTML
        mock_scrape_fallback.return_value = retrieve_full_text.advanced_scraper.HTMLResult(
            type="html", text="HTML from Advanced Scraper via Elink", url=elink_url
        )
        mock_get_unpaywall.return_value = None # Ensure Unpaywall path isn't taken

        article_input = SAMPLE_ARTICLE_FULL_IDS.copy()
        markdown, status = await retrieve_full_text.get_full_text_for_doi(article_input)

        self.assertEqual(markdown, "HTML from Advanced Scraper via Elink")
        self.assertTrue(status.startswith(f"Retrieved HTML via Elink > Advanced Scraper ({elink_url})"))
        mock_resolve_content_elink.assert_called_once_with(elink_url, client=unittest.mock.ANY)
        mock_scrape_fallback.assert_called_once_with(elink_url)
        mock_write_cache.assert_called_once()


class TestRetrieveFullTextsForDois(unittest.IsolatedAsyncioTestCase):

    @patch('tools.retrieve_full_text.get_full_text_for_doi', new_callable=AsyncMock)
    async def test_filters_by_relevance_score_and_processes_relevant(self, mock_get_full_text_doi):
        mock_get_full_text_doi.side_effect = [
            ("Markdown for DOI1", "Success from source1"),
            ("Markdown for DOI3", "Success from source2")
        ]

        input_data = {
            "triaged_articles": [
                {"doi": "doi1", "relevance_score": 5, "title": "Article 1"},
                {"doi": "doi2", "relevance_score": 3, "title": "Article 2"}, # Should be skipped
                {"doi": "doi3", "relevance_score": 4, "title": "Article 3"},
                {"doi": "doi4", "relevance_score": 5, "title": "Article 4 - No DOI"}, # No DOI, should be skipped by get_full_text_for_doi logic
                {"doi": "doi5", "relevance_score": 2, "title": "Article 5"}  # Should be skipped
            ]
        }
        # Simulate no DOI for article 4 for this specific test path
        input_data_modified = json.loads(json.dumps(input_data)) # Deep copy
        del input_data_modified["triaged_articles"][3]["doi"]


        expected_calls_to_get_full_text = [
            unittest.mock.call(input_data["triaged_articles"][0]), # doi1
            unittest.mock.call(input_data["triaged_articles"][2]), # doi3
            # doi4 (originally) would be called but we removed its DOI for this test path
        ]
        
        # We expect get_full_text_for_doi to be called for doi1 and doi3.
        # For doi4 (no DOI), it will be filtered by the list comprehension initially,
        # or if it passed that, get_full_text_for_doi would return (None, "DOI not provided...")
        # Let's adjust the input to truly test the relevance score filtering first.
        
        input_for_relevance_test = {
            "triaged_articles": [
                {"doi": "doi1", "relevance_score": 5, "title": "Article 1"}, # Process
                {"doi": "doi2", "relevance_score": 3, "title": "Article 2"}, # Skip
                {"doi": "doi3", "relevance_score": 4, "title": "Article 3"}, # Process
                {"doi": "doi4", "relevance_score": 2, "title": "Article 4"}  # Skip
            ]
        }
        
        mock_get_full_text_doi.side_effect = [
            ("Markdown for DOI1", "Success from source1"),
            ("Markdown for DOI3", "Success from source2")
        ]


        # Make a deep copy of the specific article objects for assertion *before* they are modified
        article1_original_for_assert = json.loads(json.dumps(input_for_relevance_test["triaged_articles"][0]))
        article3_original_for_assert = json.loads(json.dumps(input_for_relevance_test["triaged_articles"][2]))
        
        result_data = await retrieve_full_text.retrieve_full_texts_for_dois(input_for_relevance_test)

        self.assertEqual(mock_get_full_text_doi.call_count, 2)
        # The mock is called with the article data *before* fulltext fields are added
        mock_get_full_text_doi.assert_any_call(article1_original_for_assert)
        mock_get_full_text_doi.assert_any_call(article3_original_for_assert)

        # Check results in the output
        self.assertEqual(result_data["triaged_articles"][0]["fulltext"], "Markdown for DOI1")
        self.assertEqual(result_data["triaged_articles"][0]["fulltext_retrieval_status"], "success")
        self.assertEqual(result_data["triaged_articles"][1]["fulltext_retrieval_status"], "skipped_relevance")
        self.assertEqual(result_data["triaged_articles"][2]["fulltext"], "Markdown for DOI3")
        self.assertEqual(result_data["triaged_articles"][2]["fulltext_retrieval_status"], "success")
        self.assertEqual(result_data["triaged_articles"][3]["fulltext_retrieval_status"], "skipped_relevance")


    @patch('tools.retrieve_full_text.get_full_text_for_doi', new_callable=AsyncMock)
    async def test_handles_no_relevant_articles(self, mock_get_full_text_doi):
        input_data = {
            "triaged_articles": [
                {"doi": "doi1", "relevance_score": 1},
                {"doi": "doi2", "relevance_score": 2}
            ]
        }
        result_data = await retrieve_full_text.retrieve_full_texts_for_dois(input_data)
        mock_get_full_text_doi.assert_not_called()
        self.assertEqual(len(result_data["triaged_articles"]), 2) # Original articles should be there
        self.assertEqual(result_data["triaged_articles"][0].get("fulltext_retrieval_status"), "skipped_relevance")

    @patch('tools.retrieve_full_text.get_full_text_for_doi', new_callable=AsyncMock)
    async def test_handles_processing_exception_for_one_doi(self, mock_get_full_text_doi):
        mock_get_full_text_doi.side_effect = [
            ("Markdown for DOI1", "Success"),
            Exception("Test processing error for DOI2"), # Simulate an unhandled error
            ("Markdown for DOI3", "Success")
        ]
        input_data = {
            "triaged_articles": [
                {"doi": "doi1", "relevance_score": 5},
                {"doi": "doi2", "relevance_score": 4},
                {"doi": "doi3", "relevance_score": 5}
            ]
        }
        result_data = await retrieve_full_text.retrieve_full_texts_for_dois(input_data)
        
        self.assertEqual(mock_get_full_text_doi.call_count, 3)
        self.assertEqual(result_data["triaged_articles"][0]["fulltext"], "Markdown for DOI1")
        self.assertEqual(result_data["triaged_articles"][0]["fulltext_retrieval_status"], "success")
        
        self.assertIsNone(result_data["triaged_articles"][1].get("fulltext"))
        self.assertEqual(result_data["triaged_articles"][1]["fulltext_retrieval_status"], "failure")
        self.assertTrue("Error: Test processing error for DOI2" in result_data["triaged_articles"][1]["fulltext_retrieval_message"])

        self.assertEqual(result_data["triaged_articles"][2]["fulltext"], "Markdown for DOI3")
        self.assertEqual(result_data["triaged_articles"][2]["fulltext_retrieval_status"], "success")

if __name__ == '__main__':
    unittest.main()
