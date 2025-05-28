import asyncio
import os
import unittest
from unittest.mock import patch, AsyncMock, MagicMock, call
import httpx # Import for httpx.RequestError used in mocks
from urllib.parse import urlparse
from playwright.async_api import Error as PlaywrightError # For mocking Playwright errors

# Make sure tools directory is in path for imports if running tests directly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.advanced_scraper import (
    scrape_with_fallback,
    # fetch_specific_pdf_url, # Removed as it's not in the module anymore or not used by current tasks
    ScraperResult, # Updated from ResolveResult
    HTMLResult,
    FileResult,
    PDFBytesResult, # Added
    Failure,
    is_pdf_content_valid, # For direct testing if needed
    is_html_potentially_paywalled, # For direct testing if needed
    DOWNLOADS_DIR as ADVANCED_SCRAPER_DOWNLOADS_DIR # Import to manage test downloads
)

# Ensure the test downloads directory exists and is clean for some tests
TEST_DOWNLOADS_DIR = os.path.join("workspace", "test_downloads", "advanced_scraper")

class TestAdvancedScraper(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Ensure a clean state for downloads if tests create files
        if not os.path.exists(TEST_DOWNLOADS_DIR):
            os.makedirs(TEST_DOWNLOADS_DIR)
        # Override the default DOWNLOADS_DIR for testing purposes
        self.patcher = patch('tools.advanced_scraper.DOWNLOADS_DIR', TEST_DOWNLOADS_DIR)
        self.mock_download_dir = self.patcher.start()
        
        # Clean up any files from previous test runs in TEST_DOWNLOADS_DIR
        for item in os.listdir(TEST_DOWNLOADS_DIR):
            item_path = os.path.join(TEST_DOWNLOADS_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

    def tearDown(self):
        self.patcher.stop()
        # Clean up test download files after each test
        for item in os.listdir(TEST_DOWNLOADS_DIR):
            item_path = os.path.join(TEST_DOWNLOADS_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
        # if os.path.exists(TEST_DOWNLOADS_DIR) and not os.listdir(TEST_DOWNLOADS_DIR):
        #     os.rmdir(TEST_DOWNLOADS_DIR) # Remove if empty, handle potential race condition if parallel

    @patch('tools.advanced_scraper.httpx.AsyncClient')
    async def test_scrape_direct_html_success(self, MockAsyncClient):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        # Make content long enough to pass MIN_HTML_CONTENT_LENGTH (200) and MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE (7000) if needed
        long_html_text = "This is full article text, long enough to pass all checks. " * 200 
        mock_response.text = f"<html><head><title>Test</title></head><body><article>{long_html_text}</article></body></html>"
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.url = "http://example.com/article" # httpx.URL object

        # Configure the client instance's request method
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        
        # Mock _make_request_with_retry to simplify direct testing of scrape_with_fallback logic
        with patch('tools.advanced_scraper._make_request_with_retry', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.return_value = mock_response

            url = "http://example.com/article"
            result = await scrape_with_fallback(url, attempt_playwright=False)

            self.assertIsInstance(result, HTMLResult)
            self.assertEqual(result.type, "html")
            self.assertIn("This is full article text", result.text)
            self.assertEqual(result.url, "http://example.com/article")
            
            # Check the call arguments more flexibly for headers
            called_args, called_kwargs = mock_make_request.call_args
            self.assertEqual(called_args[0], mock_client_instance)
            self.assertEqual(called_args[1], "GET")
            self.assertEqual(called_args[2], url)
            self.assertIn("text/html", called_kwargs['headers'].get("Accept", ""))
            self.assertIsNone(called_kwargs.get('passed_cookies'))


    @patch('tools.advanced_scraper.httpx.AsyncClient')
    async def test_scrape_direct_pdf_success_via_head(self, MockAsyncClient):
        # Mock HEAD response
        mock_head_response = AsyncMock()
        mock_head_response.status_code = 200
        mock_head_response.headers = {'content-type': 'application/pdf'}
        mock_head_response.url = "http://example.com/document.pdf"

        # Mock GET response (for downloading the PDF)
        mock_get_response = AsyncMock()
        mock_get_response.status_code = 200
        mock_get_response.headers = {'content-type': 'application/pdf'} # Should match
        mock_get_response.content = b"%PDF-1.4\n%Fake PDF content long enough to pass size check.\n" * 100 # Ensure > MIN_PDF_SIZE_KB
        mock_get_response.url = "http://example.com/document.pdf"

        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        
        # Use a side_effect to return different responses for HEAD and GET
        async def make_request_side_effect(client, method, url_called, **kwargs):
            # Initial HTML GET attempt
            accept_header = kwargs.get("headers", {}).get("Accept", "")
            # Check if it's primarily an HTML request, not a PDF request
            is_html_request = "text/html" in accept_header and "application/pdf" not in accept_header

            if method == "GET" and is_html_request and url_called == url:
                # Simulate it's not HTML or not useful HTML, to proceed to PDF check
                mock_html_fail_response = AsyncMock()
                mock_html_fail_response.status_code = 200
                mock_html_fail_response.headers = {'content-type': 'application/octet-stream'} # Not HTML
                mock_html_fail_response.text = "This is not HTML"
                mock_html_fail_response.url = url_called
                return mock_html_fail_response
            elif method == "HEAD" and url_called == url: # url is the main test url
                return mock_head_response
            elif method == "GET" and url_called == mock_get_response.url: # The actual PDF download
                return mock_get_response
            raise ValueError(f"Unexpected method/url_called in mock: {method} {url_called}")

        with patch('tools.advanced_scraper._make_request_with_retry', AsyncMock(side_effect=make_request_side_effect)) as mock_make_request, \
             patch('tools.advanced_scraper.is_pdf_content_valid', MagicMock(return_value=(True, ""))): # Assume PDF is valid

            url = "http://example.com/document.pdf"
            result = await scrape_with_fallback(url, attempt_playwright=False)

            self.assertIsInstance(result, FileResult)
            self.assertEqual(result.type, "file")
            self.assertTrue(os.path.exists(result.path))
            self.assertTrue(result.path.startswith(TEST_DOWNLOADS_DIR))
            self.assertEqual(result.url, "http://example.com/document.pdf")
            
            # Check calls to _make_request_with_retry
            expected_calls = [
                call(mock_client_instance, "GET", url, headers={"Accept": "text/html,*/*;q=0.8"}), # Initial HTML check
                call(mock_client_instance, "HEAD", url, headers={"Accept": "application/pdf,*/*;q=0.8"}), # PDF HEAD check
                call(mock_client_instance, "GET", "http://example.com/document.pdf"), # PDF GET download
            ]
            # Allow for some flexibility in how the mock client is passed if it's wrapped
            # Call 0: GET HTML (mocked to fail to find HTML)
            self.assertEqual(mock_make_request.call_args_list[0][0][1], "GET") # method
            self.assertEqual(mock_make_request.call_args_list[0][0][2], url) # url_called
            self.assertIn("text/html", mock_make_request.call_args_list[0][1]['headers'].get("Accept", ""))
            self.assertIsNone(mock_make_request.call_args_list[0][1].get('passed_cookies'))

            # Call 1: HEAD PDF
            self.assertEqual(mock_make_request.call_args_list[1][0][1], "HEAD") # method
            self.assertEqual(mock_make_request.call_args_list[1][0][2], url) # url_called
            self.assertIn("application/pdf", mock_make_request.call_args_list[1][1]['headers'].get("Accept", ""))
            self.assertIsNone(mock_make_request.call_args_list[1][1].get('passed_cookies'))

            # Call 2: GET PDF (after HEAD success)
            self.assertEqual(mock_make_request.call_args_list[2][0][1], "GET")
            self.assertEqual(mock_make_request.call_args_list[2][0][2], "http://example.com/document.pdf") # Final PDF URL


    @patch('tools.advanced_scraper.async_playwright')
    async def test_scrape_playwright_html_success(self, MockAsyncPlaywright):
        # Mock Playwright objects
        mock_pw_instance = MockAsyncPlaywright.return_value.__aenter__.return_value
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_pw_instance.chromium.launch.return_value = mock_browser
        mock_browser.new_page.return_value = mock_page
        
        long_js_text = "This is JS rendered full article text, and it needs to be very long to pass the checks. " * 150
        mock_page.url = "http://example.com/js-article"
        mock_page.content = AsyncMock(return_value=f"<html><body><article>{long_js_text}</article></body></html>") # Explicitly make .content an AsyncMock
        
        # Mock for page.goto() to return a response with a status attribute
        mock_nav_response_goto = AsyncMock()
        mock_nav_response_goto.status = 200
        mock_page.goto = AsyncMock(return_value=mock_nav_response_goto)
        
        mock_page.on = MagicMock() # page.on is a sync method
        mock_browser.close = AsyncMock()
        mock_browser.is_connected = MagicMock(side_effect=[True, False]) # To handle close being called once effectively

        # Mock httpx calls to fail, forcing Playwright
        with patch('tools.advanced_scraper._make_request_with_retry', AsyncMock(side_effect=httpx.RequestError("Simulated network error"))) as mock_httpx_request:
            url = "http://example.com/js-article"
            result = await scrape_with_fallback(url, attempt_playwright=True)

            self.assertIsInstance(result, HTMLResult)
            self.assertEqual(result.type, "html")
            self.assertIn("This is JS rendered full article text", result.text)
            self.assertEqual(result.url, "http://example.com/js-article")
            
            mock_pw_instance.chromium.launch.assert_called_once()
            mock_browser.new_page.assert_called_once()
            # The wait_until logic is now more complex due to Cloudflare handling.
            # It first tries 'load', then potentially 'networkidle' after iframe interaction.
            # For this specific test path (HTML success without complex CF interaction), 'load' is primary.
            mock_page.goto.assert_called_once_with(url, timeout=unittest.mock.ANY, wait_until="load")
            mock_page.content.assert_called_once()
            mock_browser.close.assert_called_once()

    @patch('tools.advanced_scraper.async_playwright')
    async def test_scrape_playwright_pdf_response_event_success(self, MockAsyncPlaywright): # Renamed for clarity
        mock_pw_instance = MockAsyncPlaywright.return_value.__aenter__.return_value
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        # mock_download_item is not directly used if PDF is captured by response event

        mock_pw_instance.chromium.launch.return_value = mock_browser
        mock_browser.new_page.return_value = mock_page
        
        mock_page.url = "http://example.com/js-pdf-page-event" # New URL for clarity
        # page.goto will be mocked to simulate a PDF response event
        mock_page.on = MagicMock()

        # Mock browser.close and browser.is_connected interaction
        # mock_browser is an AsyncMock. Its attributes (like .close, .is_connected) are also AsyncMocks by default.
        
        closed_state_for_test = [False]  # Use a list for mutable closure

        async def is_connected_side_effect_for_test():
            return not closed_state_for_test[0]

        async def close_side_effect_for_test(*args, **kwargs):
            closed_state_for_test[0] = True
            return None  # close methods usually return None

        mock_browser.is_connected.side_effect = is_connected_side_effect_for_test
        mock_browser.close.side_effect = close_side_effect_for_test
        # The assertion will be on mock_browser.close itself.

        mock_page.context.cookies = AsyncMock(return_value=[{"name": "testcookie", "value": "testvalue", "domain": "example.com"}])


        # This test focuses on the _maybe_capture_pdf handler being triggered.
        # The handler itself raises _ReturnPDFBytes.
        # So, we need to simulate page.goto leading to a state where _maybe_capture_pdf is called
        # and raises the exception, which is then caught by scrape_with_fallback.

        captured_response_handler_for_test = None
        def mock_page_on_side_effect(event_name, handler_func):
            nonlocal captured_response_handler_for_test
            if event_name == "response":
                captured_response_handler_for_test = handler_func
            # Allow other event handlers (like "download") to be registered too
            pass
        mock_page.on.side_effect = mock_page_on_side_effect
        
        # Mock the Playwright response object that _maybe_capture_pdf would receive
        mock_pw_pdf_response = AsyncMock()
        mock_pw_pdf_response.headers = {'content-type': 'application/pdf'}
        mock_pw_pdf_response.url = "http://example.com/streamed.pdf"
        mock_pw_pdf_response.body = AsyncMock(return_value=b"PDF bytes from response event")

        async def mock_page_goto_side_effect(url, **kwargs):
            # Simulate the "response" event being fired with a PDF response
            if captured_response_handler_for_test:
                # This call should trigger the _ReturnPDFBytes exception within scrape_with_fallback
                await captured_response_handler_for_test(mock_pw_pdf_response)
            # If the handler doesn't raise (e.g., not a PDF response), goto would complete normally.
            # For this test, we assume the handler *will* raise.
            # If it doesn't, the test would proceed to other Playwright logic not being tested here.
            # To ensure the test path is clear, we can make goto raise if the handler didn't.
            # However, the SUT should catch _ReturnPDFBytes.
            
            # If _maybe_capture_pdf raises, this part of goto_side_effect might not be reached
            # in the actual execution flow of scrape_with_fallback.
            # The test relies on scrape_with_fallback's try/except _ReturnPDFBytes.
            mock_nav_response = AsyncMock() # Mock a generic navigation response
            mock_nav_response.status = 200
            return mock_nav_response

        mock_page.goto = AsyncMock(side_effect=mock_page_goto_side_effect)

        # Mock httpx calls to fail, forcing Playwright path
        with patch('tools.advanced_scraper._make_request_with_retry', AsyncMock(side_effect=httpx.RequestError("Simulated network error"))):
            
            url_to_scrape = "http://example.com/js-pdf-page-event"
            result = await scrape_with_fallback(url_to_scrape, attempt_playwright=True)

            self.assertIsInstance(result, PDFBytesResult) # Use direct import
            self.assertEqual(result.type, "pdf_bytes")
            self.assertEqual(result.content, b"PDF bytes from response event")
            self.assertEqual(result.url, "http://example.com/streamed.pdf")
            self.assertIsNotNone(result.cookies)
            self.assertEqual(result.cookies, [{"name": "testcookie", "value": "testvalue", "domain": "example.com"}])

            mock_pw_instance.chromium.launch.assert_called_once()
            mock_page.on.assert_any_call("response", unittest.mock.ANY) # Check 'response' handler was set
            mock_page.goto.assert_called_once_with(url_to_scrape, timeout=unittest.mock.ANY, wait_until="load")
            mock_browser.close.assert_called_once() # Ensure browser is closed even when PDFBytesResult is returned

    @patch('tools.advanced_scraper.async_playwright')
    async def test_scrape_playwright_pdf_link_found_success(self, MockAsyncPlaywright):
        mock_pw_instance = MockAsyncPlaywright.return_value.__aenter__.return_value
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        
        mock_pw_instance.chromium.launch.return_value = mock_browser
        mock_browser.new_page.return_value = mock_page
        
        mock_page.url = "http://example.com/page-with-pdf-link"
        mock_page.goto = AsyncMock()
        mock_page.on = MagicMock() # page.on is a sync method
        mock_browser.close = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html><body>Some content <a href='document.pdf'>Download PDF</a></body></html>") # Explicitly make .content an AsyncMock

        # Mock the link element and its attributes
        mock_link_element = AsyncMock()
        mock_link_element.get_attribute.return_value = "document.pdf" # Relative PDF link
        mock_page.query_selector_all.return_value = [mock_link_element] # Simulate finding one link

        # Mock the httpx GET request for the found PDF link
        mock_pdf_response = AsyncMock()
        mock_pdf_response.status_code = 200
        mock_pdf_response.headers = {'content-type': 'application/pdf'}
        mock_pdf_response.content = b"%PDF-1.4\n%Another Fake PDF.\n" * 100
        mock_pdf_response.url = "http://example.com/document.pdf" # Final URL of the PDF

        # This httpx client is used internally by _handle_pdf_download when fetching the link
        # We need to patch _make_request_with_retry that is called for this specific PDF download
        
        async def make_request_side_effect_for_playwright_link(client, method, url, **kwargs):
            if url == "http://example.com/document.pdf": # The absolute URL of the PDF link
                return mock_pdf_response
            # Fallback for initial page load attempts (which should fail to trigger Playwright)
            raise httpx.RequestError("Simulated network error for initial page load")

        # Mock initial httpx calls to fail, then mock the specific PDF link fetch
        with patch('tools.advanced_scraper._make_request_with_retry', AsyncMock(side_effect=make_request_side_effect_for_playwright_link)), \
             patch('tools.advanced_scraper.is_pdf_content_valid', MagicMock(return_value=(True, ""))):

            url = "http://example.com/page-with-pdf-link"
            result = await scrape_with_fallback(url, attempt_playwright=True)

            self.assertIsInstance(result, FileResult)
            self.assertEqual(result.type, "file")
            self.assertTrue(os.path.exists(result.path))
            self.assertTrue(result.path.startswith(TEST_DOWNLOADS_DIR))
            self.assertTrue(result.path.endswith(".pdf")) # Filename doesn't include source_description
            self.assertEqual(result.url, "http://example.com/document.pdf")

            # The selector is now a combination from PDF_LINK_SELECTORS
            # Check that the call was made with a string that includes the basic PDF link selector
            called_selector_string = mock_page.query_selector_all.call_args[0][0]
            self.assertIn("a[href$='.pdf']", called_selector_string)
            self.assertIn("a[href*='downloadSgArticle']", called_selector_string) # Example of another selector part


    # TODO: Add more tests:
    # - test_scrape_paywalled_content_direct
    # - test_scrape_paywalled_content_playwright
    # - test_is_pdf_content_valid_various_scenarios (directly test this helper)
    # - test_is_html_potentially_paywalled_various_scenarios (directly test this helper)
    # - Tests for proxy usage (mocking PROXY_SETTINGS and checking httpx/playwright calls)
    # - Tests for domain_specific_rules (mocking domains_config)

    @patch('tools.advanced_scraper.async_playwright')
    @patch('tools.advanced_scraper.httpx.AsyncClient')
    async def test_scrape_failure_all_methods(self, MockAsyncClient, MockAsyncPlaywright):
        # Mock httpx to always fail
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.request = AsyncMock(side_effect=httpx.RequestError("Simulated network error for httpx"))
        
        with patch('tools.advanced_scraper._make_request_with_retry', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.side_effect = httpx.RequestError("Simulated network error for _make_request_with_retry")

            # Mock Playwright to also fail or find nothing
            mock_pw_instance = MockAsyncPlaywright.return_value.__aenter__.return_value
            mock_browser = AsyncMock()
            mock_page = AsyncMock()
            mock_pw_instance.chromium.launch.return_value = mock_browser
            mock_browser.new_page.return_value = mock_page
            
            mock_page.url = "http://example.com/nothing-works"
            mock_page.goto = AsyncMock(side_effect=PlaywrightError("Simulated Playwright navigation error"))
            mock_page.on = MagicMock() # page.on is a sync method
            mock_browser.close = AsyncMock()
            # Ensure content calls return empty or raise errors if goto fails
            mock_page.content = AsyncMock(return_value="<html></html>")
            mock_page.query_selector_all.return_value = []


            url = "http://example.com/nothing-works"
            result = await scrape_with_fallback(url, attempt_playwright=True)

            self.assertIsInstance(result, Failure)
            self.assertEqual(result.type, "failure")
            # Expect the specific Playwright navigation error
            self.assertIn("Playwright navigation error: Simulated Playwright navigation error", result.reason)
            
            # Check that httpx was tried
            self.assertTrue(mock_make_request.called)
            # Check that playwright was tried
            mock_pw_instance.chromium.launch.assert_called_once()

# Removed tests for fetch_specific_pdf_url as it's not part of the core tasks / may have been removed.
# - test_fetch_specific_pdf_url_success
# - test_fetch_specific_pdf_url_failure_not_pdf
# - test_fetch_specific_pdf_url_failure_suspicious_content_type_but_invalid_pdf

if __name__ == '__main__':
    unittest.main()
