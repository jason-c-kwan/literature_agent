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
    fetch_specific_pdf_url,
    ResolveResult,
    HTMLResult,
    FileResult,
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
            mock_make_request.assert_called_once_with(
                mock_client_instance, "GET", url, headers={"Accept": "text/html,*/*;q=0.8"}
            )

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
            if method == "GET" and kwargs.get("headers", {}).get("Accept") == "text/html,*/*;q=0.8":
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
            self.assertEqual(mock_make_request.call_args_list[0][0][1], "GET")
            self.assertEqual(mock_make_request.call_args_list[0][0][2], url) # Initial URL
            self.assertEqual(mock_make_request.call_args_list[0][1]['headers'], {"Accept": "text/html,*/*;q=0.8"})
            
            # Call 1: HEAD PDF
            self.assertEqual(mock_make_request.call_args_list[1][0][1], "HEAD")
            self.assertEqual(mock_make_request.call_args_list[1][0][2], url) # Initial URL
            self.assertEqual(mock_make_request.call_args_list[1][1]['headers'], {"Accept": "application/pdf,*/*;q=0.8"})

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
        mock_page.goto = AsyncMock()
        mock_page.on = MagicMock() # page.on is a sync method
        mock_browser.close = AsyncMock()

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
            mock_page.goto.assert_called_once_with(url, timeout=unittest.mock.ANY, wait_until="networkidle")
            mock_page.content.assert_called_once()
            mock_browser.close.assert_called_once()

    @patch('tools.advanced_scraper.async_playwright')
    async def test_scrape_playwright_pdf_download_event_success(self, MockAsyncPlaywright):
        mock_pw_instance = MockAsyncPlaywright.return_value.__aenter__.return_value
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        mock_download_item = AsyncMock()

        mock_pw_instance.chromium.launch.return_value = mock_browser
        mock_browser.new_page.return_value = mock_page
        
        # Set a default return value for page.content() to avoid TypeErrors if download path isn't hit first
        mock_page.content = AsyncMock(return_value="<html><body>Fallback content</body></html>") # Explicitly make .content an AsyncMock
        mock_page.url = "http://example.com/js-pdf-page"
        mock_page.goto = AsyncMock()
        mock_page.on = MagicMock() # page.on is a sync method
        mock_browser.close = AsyncMock()

        # Simulate the download event
        # The 'download' event callback in the SUT will call download.save_as()
        # and download.suggested_filename
        mock_download_item.suggested_filename = "downloaded_document.pdf"
        
        async def mock_save_as_side_effect(path_arg_received_by_mock):
            # Ensure the directory exists
            abs_path_arg = os.path.abspath(path_arg_received_by_mock)
            os.makedirs(os.path.dirname(abs_path_arg), exist_ok=True)
            # Create an empty file at the given path to simulate download
            with open(abs_path_arg, 'wb') as f:
                f.write(b"dummy pdf content for test") # Write some bytes
            return None # save_as usually returns None
        
        mock_download_item.save_as = AsyncMock(side_effect=mock_save_as_side_effect)
        mock_download_item.url = "http://example.com/actual_document.pdf" # URL of the downloaded file

        # This is tricky: the SUT's page.on("download", handle_download) sets up a callback.
        # We need to mock `page.on` to capture the callback, then we can invoke it with our mock_download_item.
        captured_download_handler = None
        def mock_page_on(event_name, handler):
            nonlocal captured_download_handler
            if event_name == "download":
                captured_download_handler = handler
        mock_page.on = MagicMock(side_effect=mock_page_on)

        # Mock httpx calls to fail, forcing Playwright
        with patch('tools.advanced_scraper._make_request_with_retry', AsyncMock(side_effect=httpx.RequestError("Simulated network error"))), \
             patch('tools.advanced_scraper.is_pdf_content_valid', MagicMock(return_value=(True, ""))): # Assume PDF is valid
            
            url = "http://example.com/js-pdf-page"
            
            # We need to run scrape_with_fallback and then manually trigger the download handler
            # as if Playwright emitted the event.
            # The call to page.goto() might trigger the download.
            # For this test, we'll assume goto completes, then we manually invoke the captured handler.

            async def goto_side_effect(*args, **kwargs):
                # After goto is called, simulate the download event by calling the captured handler
                if captured_download_handler:
                    # Directly await the handler to ensure it completes before proceeding
                    await captured_download_handler(mock_download_item)
                # page.goto usually returns a Response object or None on success
                # Let's mock it to return a mock response object to be more realistic
                mock_nav_response = AsyncMock()
                mock_nav_response.ok = True 
                return mock_nav_response

            mock_page.goto.side_effect = goto_side_effect
            
            result = await scrape_with_fallback(url, attempt_playwright=True)
            
            # Diagnostic sleep
            await asyncio.sleep(0.01) 

            self.assertIsInstance(result, FileResult)
            self.assertEqual(result.type, "file")
            # Ensure we check the absolute path, consistent with how mock_save_as_side_effect might create it
            abs_result_path = os.path.abspath(result.path)
            self.assertTrue(os.path.exists(abs_result_path), f"File not found at {abs_result_path}")
            # self.assertTrue(result.path.startswith(TEST_DOWNLOADS_DIR)) # This might be tricky if one is abs and other relative
            self.assertTrue(abs_result_path.startswith(os.path.abspath(TEST_DOWNLOADS_DIR)), "Path does not start with test download dir")
            self.assertTrue(result.path.endswith("_playwright.pdf"))
            # The URL in FileResult should be the one from the download item if available
            self.assertEqual(result.url, "http://example.com/actual_document.pdf") 

            mock_pw_instance.chromium.launch.assert_called_once()
            mock_page.on.assert_called_with("download", unittest.mock.ANY) # Check that 'on' was called for 'download'
            mock_download_item.save_as.assert_called_once() # Check that the download was saved

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

            mock_page.query_selector_all.assert_called_once_with("a[href$='.pdf']")


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


    @patch('tools.advanced_scraper.httpx.AsyncClient')
    async def test_fetch_specific_pdf_url_success(self, MockAsyncClient):
        mock_pdf_response = AsyncMock()
        mock_pdf_response.status_code = 200
        mock_pdf_response.headers = {'content-type': 'application/pdf'}
        mock_pdf_response.content = b"%PDF-1.4\n%Specific PDF success.\n" * 100
        mock_pdf_response.url = "http://example.com/direct.pdf"

        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        
        with patch('tools.advanced_scraper._make_request_with_retry', AsyncMock(return_value=mock_pdf_response)) as mock_make_request, \
             patch('tools.advanced_scraper.is_pdf_content_valid', MagicMock(return_value=(True, ""))):

            url = "http://example.com/direct.pdf"
            result = await fetch_specific_pdf_url(url)

            self.assertIsInstance(result, FileResult)
            self.assertEqual(result.type, "file")
            self.assertTrue(os.path.exists(result.path))
            self.assertTrue(result.path.startswith(TEST_DOWNLOADS_DIR))
            self.assertTrue(result.path.endswith(".pdf")) # Filename doesn't include source_description
            self.assertEqual(result.url, "http://example.com/direct.pdf")
            mock_make_request.assert_called_once_with(
                mock_client_instance, "GET", url, headers={"Accept": "application/pdf,application/octet-stream,*/*;q=0.8"}
            )

    @patch('tools.advanced_scraper.httpx.AsyncClient')
    async def test_fetch_specific_pdf_url_failure_not_pdf(self, MockAsyncClient):
        mock_html_response = AsyncMock()
        mock_html_response.status_code = 200
        mock_html_response.headers = {'content-type': 'text/html'}
        mock_html_response.text = "<html><body>Not a PDF</body></html>"
        mock_html_response.url = "http://example.com/notapdf.html"
        
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value

        with patch('tools.advanced_scraper._make_request_with_retry', AsyncMock(return_value=mock_html_response)) as mock_make_request:
            url = "http://example.com/notapdf.html" # URL doesn't end with .pdf
            result = await fetch_specific_pdf_url(url)

            self.assertIsInstance(result, Failure)
            self.assertEqual(result.type, "failure")
            self.assertIn("URL did not yield PDF content", result.reason)
            self.assertEqual(result.status_code, 200)

    @patch('tools.advanced_scraper.httpx.AsyncClient')
    async def test_fetch_specific_pdf_url_failure_suspicious_content_type_but_invalid_pdf(self, MockAsyncClient):
        mock_octet_response = AsyncMock()
        mock_octet_response.status_code = 200
        # URL ends with .pdf, but content-type is octet-stream
        mock_octet_response.headers = {'content-type': 'application/octet-stream'} 
        mock_octet_response.content = b"This is not a real PDF" * 10 # Small content
        mock_octet_response.url = "http://example.com/suspicious.pdf"
        
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value

        with patch('tools.advanced_scraper._make_request_with_retry', AsyncMock(return_value=mock_octet_response)) as mock_make_request, \
             patch('tools.advanced_scraper.is_pdf_content_valid', MagicMock(return_value=(False, "Too small"))): # PDF validation fails

            url = "http://example.com/suspicious.pdf"
            result = await fetch_specific_pdf_url(url)

            self.assertIsInstance(result, Failure)
            self.assertEqual(result.type, "failure")
            self.assertIn("appears invalid: Too small", result.reason)
            # Check that _handle_pdf_download was called with the correct source description
            # This requires patching _handle_pdf_download or checking logs if more detail is needed.
            # For now, the reason check is primary. The log output already confirms the source_description.

if __name__ == '__main__':
    unittest.main()
