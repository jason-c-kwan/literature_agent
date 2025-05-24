import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import aiohttp

# Assuming the script is in the tools directory, and tests is a sibling
# This might need adjustment based on how Python path is configured for tests
from tools.retrieve_unpaywall import get_unpaywall_oa_url, UNPAYWALL_API_EMAIL

class TestRetrieveUnpaywall(unittest.IsolatedAsyncioTestCase):

    @patch('aiohttp.ClientSession.get')
    async def test_get_url_from_best_oa_location_pdf_url(self, mock_get):
        # Mock the API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content_type = 'application/json'
        mock_response.raise_for_status = MagicMock() # Add this
        expected_url = "http://example.com/pdf"
        mock_response.json = AsyncMock(return_value={
            "best_oa_location": {
                "url_for_pdf": expected_url,
                "url": "http://example.com/general",
                "url_for_landing_page": "http://example.com/landing"
            }
        })
        
        # Configure the mock context manager for session.get
        mock_get.return_value.__aenter__.return_value = mock_response
        
        doi = "10.1234/testdoi1"
        result_url = await get_unpaywall_oa_url(doi, email="test@example.com")
        
        self.assertEqual(result_url, expected_url)
        mock_get.assert_called_once_with(f"https://api.unpaywall.org/v2/{doi}?email=test@example.com")

    @patch('aiohttp.ClientSession.get')
    async def test_get_url_from_best_oa_location_general_url(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content_type = 'application/json'
        mock_response.raise_for_status = MagicMock() # Add this
        expected_url = "http://example.com/general"
        mock_response.json = AsyncMock(return_value={
            "best_oa_location": {
                "url_for_pdf": None,
                "url": expected_url,
                "url_for_landing_page": "http://example.com/landing"
            }
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        doi = "10.1234/testdoi2"
        result_url = await get_unpaywall_oa_url(doi, email="test@example.com")
        self.assertEqual(result_url, expected_url)

    @patch('aiohttp.ClientSession.get')
    async def test_get_url_from_best_oa_location_landing_url(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content_type = 'application/json'
        mock_response.raise_for_status = MagicMock() # Add this
        expected_url = "http://example.com/landing"
        mock_response.json = AsyncMock(return_value={
            "best_oa_location": {
                "url_for_pdf": None,
                "url": None,
                "url_for_landing_page": expected_url
            }
        })
        mock_get.return_value.__aenter__.return_value = mock_response

        doi = "10.1234/testdoi3"
        result_url = await get_unpaywall_oa_url(doi, email="test@example.com")
        self.assertEqual(result_url, expected_url)

    @patch('aiohttp.ClientSession.get')
    async def test_fallback_to_oa_locations_pdf_url(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content_type = 'application/json'
        mock_response.raise_for_status = MagicMock() # Add this
        expected_url = "http://example.com/oa_loc_pdf"
        mock_response.json = AsyncMock(return_value={
            "best_oa_location": None, # or missing, or empty dict
            "oa_locations": [
                {"url_for_pdf": None, "url": "http://example.com/other"},
                {"url_for_pdf": expected_url, "url": "http://example.com/another"}
            ]
        })
        mock_get.return_value.__aenter__.return_value = mock_response

        doi = "10.1234/testdoi4"
        result_url = await get_unpaywall_oa_url(doi, email="test@example.com")
        self.assertEqual(result_url, expected_url)

    @patch('aiohttp.ClientSession.get')
    async def test_no_url_found(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content_type = 'application/json'
        mock_response.raise_for_status = MagicMock() # Add this
        mock_response.json = AsyncMock(return_value={
            "best_oa_location": {"url_for_pdf": None, "url": None, "url_for_landing_page": None},
            "oa_locations": [
                {"url_for_pdf": None, "url": None, "url_for_landing_page": None}
            ]
        })
        mock_get.return_value.__aenter__.return_value = mock_response

        doi = "10.1234/testdoi5"
        result_url = await get_unpaywall_oa_url(doi, email="test@example.com")
        self.assertIsNone(result_url)

    @patch('aiohttp.ClientSession.get')
    async def test_http_error_404(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.content_type = 'application/json' # Or text/plain for some errors
        mock_response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            MagicMock(), (), status=404, message="Not Found"
        ))
        mock_response.json = AsyncMock(return_value={"error": "Not Found"}) # Mock error payload
        
        mock_get.return_value.__aenter__.return_value = mock_response

        doi = "10.1234/testdoi_404"
        result_url = await get_unpaywall_oa_url(doi, email="test@example.com")
        self.assertIsNone(result_url)

    @patch('aiohttp.ClientSession.get')
    async def test_network_error(self, mock_get):
        mock_get.side_effect = aiohttp.ClientConnectorError(MagicMock(), OSError("Network down"))

        doi = "10.1234/testdoi_network_error"
        result_url = await get_unpaywall_oa_url(doi, email="test@example.com")
        self.assertIsNone(result_url)

    @patch('aiohttp.ClientSession.get')
    async def test_non_json_response(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content_type = 'text/html'
        mock_response.raise_for_status = MagicMock() # Add this, as it's called before content_type check
        mock_response.text = AsyncMock(return_value="<html><body>Error</body></html>")
        # The SUT checks content_type before calling .json()
        
        mock_get.return_value.__aenter__.return_value = mock_response

        doi = "10.1234/testdoi_non_json"
        result_url = await get_unpaywall_oa_url(doi, email="test@example.com")
        self.assertIsNone(result_url)

    async def test_no_email_provided(self):
        # Patch UNPAYWALL_API_EMAIL in the module under test to simulate it not being set
        # Or, more directly, call with email=None if the function signature allowed easy override
        # For now, testing the default behavior when the global is used and is empty
        with patch('tools.retrieve_unpaywall.UNPAYWALL_API_EMAIL', ''):
             # Re-import or reload module if UNPAYWALL_API_EMAIL is read at import time
             # For this structure, patching should work if it's read at function call time.
             # A better way is to pass email explicitly for this test.
            result_url = await get_unpaywall_oa_url("10.1234/testdoi_no_email", email="") # Pass empty email
            self.assertIsNone(result_url)
        
        # Test with the default email from the module to ensure it's used if not overridden
        # This requires UNPAYWALL_API_EMAIL to be a non-empty string for the positive case.
        # This part of the test is more about ensuring the default mechanism works.
        # For this, we'd need a successful mock_get again.
        # Let's keep this test focused on the "no email" scenario.

if __name__ == '__main__':
    unittest.main()
