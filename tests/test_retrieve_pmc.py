import asyncio
import unittest
from unittest.mock import patch, AsyncMock, Mock
import os
from dotenv import load_dotenv
from urllib.parse import quote # Not CIMultiDict, this is for RequestInfo
import multidict # For CIMultiDict
from yarl import URL
from aiohttp import ClientResponseError, ClientConnectorError, RequestInfo
# ConnectionKey import removed

# Load environment variables for testing context,
# ensuring that the module-level variables in retrieve_pmc are set.
load_dotenv()

# Temporarily set environment variables for consistent testing,
# then restore them. This is important if tests run in different environments.
ORIGINAL_ENV = os.environ.copy()

# Import the function to test AFTER manipulating env vars if necessary,
# or ensure the module re-reads them if it's designed to.
# For retrieve_pmc, it reads env vars at import time.
# So, we might need to reload it or patch its globals if we change env vars per test.
# A simpler approach for now: set expected env vars before importing the module for testing.
os.environ["API_EMAIL"] = "test@example.com"
os.environ["PUBMED_API_KEY"] = "test_api_key"

from tools.retrieve_pmc import fetch_pmc_xml

# Example XML structures for mocking
MOCK_OA_XML = """<article>
    <front>
        <article-meta>
            <article-id pub-id-type="pmc">PMC12345</article-id>
            <title-group><article-title>Test Article</article-title></title-group>
        </article-meta>
    </front>
    <body><p>This is the article body.</p></body>
</article>"""

MOCK_ERROR_XML_NOT_FOUND = """<eFetchResult>
    <ERROR>UID=PMC_INVALID not found in db</ERROR>
</eFetchResult>"""

MOCK_MALFORMED_XML = """<article><title>Test</title><oops"""

class TestFetchPmcXml(unittest.TestCase):

    def setUp(self):
        # Ensure a fresh event loop for each async test
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(ORIGINAL_ENV)


    @patch('tools.retrieve_pmc.aiohttp.ClientSession')
    @patch('tools.retrieve_pmc.API_EMAIL', "test@example.com") # Ensure test-specific email
    @patch('tools.retrieve_pmc.PUBMED_API_KEY', "test_api_key")   # Ensure test-specific API key
    def test_fetch_successful_oa_article(self, MockClientSession): # Corrected argument order
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=MOCK_OA_XML)
        mock_response.raise_for_status = Mock(return_value=None) # Synchronous, does nothing on success
        # Configure the context manager __aenter__ for the response
        mock_response.__aenter__.return_value = mock_response

        mock_session_instance = MockClientSession.return_value
        # Configure the context manager __aenter__ for the session
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.close = AsyncMock() # Ensure close is an AsyncMock

        pmc_id = "PMC12345"
        # Pass the mocked session to the function
        result = self.loop.run_until_complete(fetch_pmc_xml(pmc_id, session=mock_session_instance)) 
        
        self.assertEqual(result, MOCK_OA_XML)
        mock_session_instance.get.assert_called_once()
        args, kwargs = mock_session_instance.get.call_args
        self.assertIn("params", kwargs)
        self.assertEqual(kwargs["params"]["id"], pmc_id)
        self.assertEqual(kwargs["params"]["email"], "test@example.com")
        self.assertEqual(kwargs["params"]["api_key"], "test_api_key")

    @patch('tools.retrieve_pmc.aiohttp.ClientSession')
    @patch('tools.retrieve_pmc.API_EMAIL', None) # Patch API_EMAIL to be None for this test
    def test_api_email_not_configured(self, MockClientSession):
        # This test assumes API_EMAIL is checked at the start of fetch_pmc_xml
        # and if not present, it returns None without making a request.
        pmc_id = "PMC12345"
        # Need to reload or ensure the patched value is used.
        # The current retrieve_pmc.py reads API_EMAIL at module level.
        # For a robust test, we'd ideally patch 'tools.retrieve_pmc.API_EMAIL'
        # or pass config into the function.
        # Given the current structure, let's patch it directly in the module.
        
        # Re-import or patch the global within the module if it's read at import time.
        # A simple way is to ensure the test setup reflects the condition.
        # The @patch('tools.retrieve_pmc.API_EMAIL', None) should handle this.

        # If fetch_pmc_xml creates its own session when API_EMAIL is None, this test is fine.
        # If it expects a session to be passed, and we want to test the no-API_EMAIL path,
        # we might not need to mock/pass a session here, as it should return early.
        result = self.loop.run_until_complete(fetch_pmc_xml(pmc_id))
        
        self.assertIsNone(result)
        # If API_EMAIL is None, get should not be called, regardless of session.
        # MockClientSession().get.assert_not_called() # Accessing get on the instance
        # Or, more directly, if the session isn't even created by the SUT:
        MockClientSession.assert_not_called() # If API_EMAIL is None, session might not be created by SUT

    @patch('tools.retrieve_pmc.aiohttp.ClientSession')
    def test_fetch_error_xml_not_found(self, MockClientSession):
        mock_response = AsyncMock()
        mock_response.status = 200 # NCBI might return 200 OK with an error in XML body
        mock_response.text = AsyncMock(return_value=MOCK_ERROR_XML_NOT_FOUND)
        mock_response.raise_for_status = Mock(return_value=None) # Synchronous, does nothing on success
        mock_response.__aenter__.return_value = mock_response

        mock_session_instance = MockClientSession.return_value
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.close = AsyncMock()

        pmc_id = "PMC_INVALID"
        result = self.loop.run_until_complete(fetch_pmc_xml(pmc_id, session=mock_session_instance))
        
        self.assertIsNone(result) # Expect None due to <ERROR> tag
        mock_session_instance.get.assert_called_once()

    @patch('tools.retrieve_pmc.aiohttp.ClientSession')
    def test_fetch_malformed_xml(self, MockClientSession):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=MOCK_MALFORMED_XML)
        mock_response.raise_for_status = Mock(return_value=None) # Synchronous, does nothing on success
        mock_response.__aenter__.return_value = mock_response

        mock_session_instance = MockClientSession.return_value
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.close = AsyncMock()

        pmc_id = "PMC_MALFORMED"
        result = self.loop.run_until_complete(fetch_pmc_xml(pmc_id, session=mock_session_instance))
        
        self.assertIsNone(result) # Expect None due to ParseError
        mock_session_instance.get.assert_called_once()

    @patch('tools.retrieve_pmc.aiohttp.ClientSession')
    def test_http_error_404(self, MockClientSession):
        # Mock request_info
        mock_request_info = RequestInfo(
            url=URL("http://example.com/test"),
            method="GET",
            headers=multidict.CIMultiDict(),
            real_url=URL("http://example.com/test")
        )
        
        mock_response = AsyncMock()
        mock_response.status = 404
        # Correctly instantiate ClientResponseError
        error_instance = ClientResponseError(
            request_info=mock_request_info,
            history=(), # Empty tuple for history
            status=404,
            message="Not Found",
            headers=multidict.CIMultiDict()
        )
        # raise_for_status is a synchronous method, so mock with Mock, not AsyncMock
        mock_response.raise_for_status = Mock(side_effect=error_instance) 
        mock_response.text = AsyncMock(return_value="Not Found")
        mock_response.__aenter__.return_value = mock_response
        
        mock_session_instance = MockClientSession.return_value
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.close = AsyncMock()
        
        pmc_id = "PMC_HTTP_ERROR"
        result = self.loop.run_until_complete(fetch_pmc_xml(pmc_id, session=mock_session_instance))
        
        self.assertIsNone(result)
        mock_session_instance.get.assert_called_once()

    @patch('tools.retrieve_pmc.aiohttp.ClientSession')
    def test_network_error(self, MockClientSession):
        mock_session_instance = MockClientSession.return_value
        mock_session_instance.__aenter__.return_value = mock_session_instance

        # Instantiate ClientConnectorError with a generic Mock for the first argument (connection_key or similar)
        # The os_error (second argument) is the more critical part for the exception.
        os_error = OSError("Network down")
        # ClientConnectorError typically expects a connection_key-like object and an OSError.
        # A simple Mock() should suffice for the first argument in a testing context.
        connector_error = ClientConnectorError(Mock(), os_error) 
        mock_session_instance.get.side_effect = connector_error
        mock_session_instance.close = AsyncMock()

        pmc_id = "PMC_NET_ERROR"
        result = self.loop.run_until_complete(fetch_pmc_xml(pmc_id, session=mock_session_instance))
        
        self.assertIsNone(result)
        mock_session_instance.get.assert_called_once()

    @patch('tools.retrieve_pmc.aiohttp.ClientSession')
    @patch('tools.retrieve_pmc.PUBMED_API_KEY', None) # Test case where API key is not set
    def test_fetch_successful_without_api_key(self, MockClientSession):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=MOCK_OA_XML)
        mock_response.raise_for_status = Mock(return_value=None) # Synchronous, does nothing on success
        mock_response.__aenter__.return_value = mock_response

        mock_session_instance = MockClientSession.return_value
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.close = AsyncMock()

        pmc_id = "PMC12345"
        result = self.loop.run_until_complete(fetch_pmc_xml(pmc_id, session=mock_session_instance))
        
        self.assertEqual(result, MOCK_OA_XML)
        mock_session_instance.get.assert_called_once()
        args, kwargs = mock_session_instance.get.call_args
        self.assertNotIn("api_key", kwargs["params"]) # Key should not be in params

if __name__ == '__main__':
    unittest.main()
