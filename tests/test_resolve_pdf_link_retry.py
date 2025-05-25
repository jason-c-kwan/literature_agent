import asyncio
import json
import logging
import os
import shutil
from unittest.mock import patch, MagicMock, ANY

import pytest
import httpx
from respx import MockRouter

from tools.resolve_pdf_link import (
    resolve_content,
    Failure,
    HTMLResult,
    FileResult,
    DOWNLOADS_DIR,
    MIN_PDF_SIZE_KB,
    JsonFormatter # Import for potential direct use or inspection if needed
)

# --- Helper Functions ---
def get_json_logs(caplog_fixture):
    """Parses captured log messages from caplog.records using JsonFormatter."""
    json_logs = []
    formatter = JsonFormatter() # Instantiate our custom formatter
    for record in caplog_fixture.records:
        if record.name == "resolve_pdf_link": # Filter for relevant logger
            try:
                # Format the record using our JsonFormatter
                json_string = formatter.format(record)
                json_logs.append(json.loads(json_string))
            except Exception as e:
                print(f"WARNING: test_resolve_pdf_link_retry.py:get_json_logs: Could not format or parse log record as JSON: {record.msg}, Error: {e}")
    return json_logs

@pytest.fixture(autouse=True)
def ensure_downloads_dir_is_clean():
    """Ensure DOWNLOADS_DIR is clean before and after each test."""
    if os.path.exists(DOWNLOADS_DIR):
        shutil.rmtree(DOWNLOADS_DIR)
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    yield
    if os.path.exists(DOWNLOADS_DIR):
        shutil.rmtree(DOWNLOADS_DIR)

@pytest.fixture
def mock_config_settings(mocker):
    """Fixture to mock configuration values for retry logic and clear global state."""
    # Clear global domain semaphores before each test using this fixture
    from tools.resolve_pdf_link import domain_semaphores
    domain_semaphores.clear()

    mocked_config = {
        "resolve_content_timeout": 1, # Short timeout for tests
        "resolve_max_retries": 2,
        "resolve_retry_backoff_factor": 0.01, # Very short backoff
        "resolve_retry_status_forcelist": [500, 503],
        "resolve_per_domain_concurrency": 1, # Default to 1 for easier concurrency testing
        # Keep other necessary defaults from the module if not overridden here
        "MIN_PDF_SIZE_KB": 10,
        "MIN_HTML_CONTENT_LENGTH": 200,
        "MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE": 7000,
    }
    mocker.patch("tools.resolve_pdf_link.config", new=mocked_config)
    # Also patch the direct usages if they are module-level constants
    mocker.patch("tools.resolve_pdf_link.REQUEST_TIMEOUT", mocked_config["resolve_content_timeout"])
    mocker.patch("tools.resolve_pdf_link.MAX_RETRIES", mocked_config["resolve_max_retries"])
    mocker.patch("tools.resolve_pdf_link.RETRY_BACKOFF_FACTOR", mocked_config["resolve_retry_backoff_factor"])
    mocker.patch("tools.resolve_pdf_link.RETRY_STATUS_FORCELIST", mocked_config["resolve_retry_status_forcelist"])
    mocker.patch("tools.resolve_pdf_link.PER_DOMAIN_CONCURRENCY", mocked_config["resolve_per_domain_concurrency"])
    return mocked_config

@pytest.fixture
def mock_datetime_now(mocker):
    mock_dt_instance = MagicMock()
    mock_dt_instance.strftime.return_value = "20240101_120000"
    mock_datetime_class = mocker.patch("tools.resolve_pdf_link.datetime.datetime")
    mock_datetime_class.now.return_value = mock_dt_instance
    return mock_datetime_class

# --- Test Cases ---

@pytest.mark.asyncio
async def test_successful_retry_after_500(respx_mock: MockRouter, caplog, mock_config_settings, mock_datetime_now):
    test_url = "http://example.com/retry_success.pdf"
    pdf_content = b"%PDF-1.4\n%Retry Success PDF"

    # First HEAD fails, second succeeds
    respx_mock.head(test_url).side_effect = [
        httpx.Response(500),
        httpx.Response(200, headers={"Content-Type": "application/pdf"})
    ]
    # First GET (after successful HEAD) fails, second succeeds
    respx_mock.get(test_url).side_effect = [
        httpx.Response(500),
        httpx.Response(200, content=pdf_content, headers={"Content-Type": "application/pdf"})
    ]
    
    caplog.set_level(logging.INFO, logger="resolve_pdf_link")

    with patch("tools.resolve_pdf_link.os.path.getsize", return_value=(MIN_PDF_SIZE_KB + 1) * 1024), \
         patch("tools.resolve_pdf_link.fitz.open") as mock_fitz_open:
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.__len__.return_value = 2 # 2 pages
        mock_fitz_open.return_value = mock_pdf_doc
        result = await resolve_content(test_url)

    assert isinstance(result, FileResult)
    assert result.type == "file"
    
    logs = get_json_logs(caplog)
    
    head_attempts = [log for log in logs if log.get("event") == "head_attempt" and log.get("url") == test_url]
    assert len(head_attempts) == 2
    assert head_attempts[0]["attempt"] == 1
    assert head_attempts[1]["attempt"] == 2

    get_attempts = [log for log in logs if log.get("event") == "get_attempt" and log.get("url") == test_url]
    assert len(get_attempts) == 2
    assert get_attempts[0]["attempt"] == 1
    assert get_attempts[1]["attempt"] == 2
    
    retry_scheduled_logs = [log for log in logs if log.get("event") == "retry_scheduled_status"]
    assert len(retry_scheduled_logs) == 2 # One for HEAD, one for GET
    assert any(log["message"].startswith("HEAD request") for log in retry_scheduled_logs)
    assert any(log["message"].startswith("GET request") for log in retry_scheduled_logs)

    assert any(log["event"] == "pdf_download_success" for log in logs)
    assert any(log["event"] == "pdf_valid" for log in logs)


@pytest.mark.asyncio
async def test_max_retries_exceeded(respx_mock: MockRouter, caplog, mock_config_settings):
    test_url = "http://example.com/max_retries.html"
    max_retries = mock_config_settings["resolve_max_retries"] # Should be 2

    # All HEAD attempts fail
    respx_mock.head(test_url).mock(return_value=httpx.Response(503))
    # GET should not be called if HEAD fails definitively after retries
    get_route = respx_mock.get(test_url).mock(return_value=httpx.Response(200, html="<html></html>"))


    caplog.set_level(logging.INFO, logger="resolve_pdf_link")
    result = await resolve_content(test_url)

    assert isinstance(result, Failure)
    assert "HEAD request failed" in result.reason or "Max retries" in result.reason # Depending on how error propagates

    logs = get_json_logs(caplog)
    head_attempts = [log for log in logs if log.get("event") == "head_attempt" and log.get("url") == test_url]
    assert len(head_attempts) == max_retries + 1 # Initial attempt + max_retries

    # Check for final failure log for HEAD
    assert any(log["event"] == "request_failure_status" and log["method"] == "HEAD" for log in logs)
    assert not get_route.called # GET should not have been attempted


@pytest.mark.asyncio
async def test_non_retriable_error(respx_mock: MockRouter, caplog, mock_config_settings):
    test_url = "http://example.com/notfound.html"
    respx_mock.head(test_url).respond(status_code=404)
    # GET should not be called if HEAD fails with 404
    get_route = respx_mock.get(test_url).mock(return_value=httpx.Response(200, html="<html></html>"))

    caplog.set_level(logging.INFO, logger="resolve_pdf_link")
    result = await resolve_content(test_url)

    assert isinstance(result, Failure)
    assert "404" in result.reason

    logs = get_json_logs(caplog)
    head_attempts = [log for log in logs if log.get("event") == "head_attempt" and log.get("url") == test_url]
    assert len(head_attempts) == 1 # Only one attempt for non-retriable

    retry_scheduled_logs = [log for log in logs if log.get("event") == "retry_scheduled_status"]
    assert len(retry_scheduled_logs) == 0
    assert not get_route.called


@pytest.mark.asyncio
async def test_rate_limit_enforcement_same_domain(respx_mock: MockRouter, caplog, mock_config_settings, event_loop):
    # Set concurrency to 1 for this test via mock_config_settings
    mock_config_settings["resolve_per_domain_concurrency"] = 1
    # We need to re-patch PER_DOMAIN_CONCURRENCY as it's read at module load.
    with patch("tools.resolve_pdf_link.PER_DOMAIN_CONCURRENCY", 1):
        domain = "example.com"
        url1 = f"http://{domain}/page1.html"
        url2 = f"http://{domain}/page2.html"

        # Mock responses with a slight delay to observe serialization
        html_content_long = "<html><body><p>" + "Sufficiently long page content. " * 50 + "</p></body></html>"
        
        respx_mock.head(url1).respond(status_code=200, headers={"Content-Type": "text/html"})
        respx_mock.get(url1).respond(status_code=200, content=html_content_long, headers={"Content-Type": "text/html"})
        respx_mock.head(url2).respond(status_code=200, headers={"Content-Type": "text/html"})
        respx_mock.get(url2).respond(status_code=200, content=html_content_long, headers={"Content-Type": "text/html"})

        caplog.set_level(logging.INFO, logger="resolve_pdf_link")

        # Mock readability to ensure success
        mock_summary_content = "<div>" + "This is sufficiently long content for the test. " * 20 + "</div>"
        with patch("tools.resolve_pdf_link.Document") as mock_document_class:
            mock_doc_instance = MagicMock()
            mock_doc_instance.summary.return_value = mock_summary_content
            mock_document_class.return_value = mock_doc_instance

            # Use asyncio.gather to run them concurrently
            start_time = event_loop.time()
            results = await asyncio.gather(
                resolve_content(url1),
                resolve_content(url2)
            )
            end_time = event_loop.time()

    assert all(isinstance(r, HTMLResult) for r in results)
    
    logs = get_json_logs(caplog)

    semaphore_acquires = [log for log in logs if log.get("event") == "semaphore_acquired" and log.get("domain") == domain]
    assert len(semaphore_acquires) == 2
    
    # Check that one call waited for the semaphore released by the other.
    # This is tricky to assert directly from logs without timestamps or specific ordering guarantees.
    # However, if PER_DOMAIN_CONCURRENCY is 1, they must have been serialized.
    # A rough check on timing (if delays were added to mock responses) could also work.
    # For now, presence of semaphore logs for the domain is a good sign.
    
    # Check for semaphore creation log
    assert any(log["event"] == "semaphore_create" and log["domain"] == domain and log["concurrency"] == 1 for log in logs)

    # If the mock responses had actual delays, end_time - start_time would be approx 2 * delay_per_call
    # Since our mocks are instant, we rely on the semaphore logic being correct.


@pytest.mark.asyncio
async def test_rate_limit_passthrough_different_domains(respx_mock: MockRouter, caplog, mock_config_settings, event_loop):
    mock_config_settings["resolve_per_domain_concurrency"] = 1 # Still 1, but for different domains
    with patch("tools.resolve_pdf_link.PER_DOMAIN_CONCURRENCY", 1):
        url1 = "http://example1.com/page.html"
        url2 = "http://example2.com/page.html"
        
        # Ensure HTML content is long enough
        html_content_1 = "<html><body><p>" + "Page 1 content. " * 50 + "</p></body></html>"
        html_content_2 = "<html><body><p>" + "Page 2 content. " * 50 + "</p></body></html>"


        respx_mock.head(url1).respond(headers={"Content-Type": "text/html"})
        respx_mock.get(url1).respond(content=html_content_1, headers={"Content-Type": "text/html"})
        respx_mock.head(url2).respond(headers={"Content-Type": "text/html"})
        respx_mock.get(url2).respond(content=html_content_2, headers={"Content-Type": "text/html"})

        caplog.set_level(logging.INFO, logger="resolve_pdf_link")
        
        # Mock readability for both calls to ensure success
        # The summary should produce text longer than MIN_HTML_CONTENT_LENGTH when parsed by BeautifulSoup
        mock_summary_content = "<div>" + "This is sufficiently long content for the test. " * 20 + "</div>"

        with patch("tools.resolve_pdf_link.Document") as mock_document_class:
            mock_doc_instance = MagicMock()
            mock_doc_instance.summary.return_value = mock_summary_content
            mock_document_class.return_value = mock_doc_instance
            
            start_time = event_loop.time()
            results = await asyncio.gather(
                resolve_content(url1),
                resolve_content(url2)
            )
            end_time = event_loop.time()

    assert all(isinstance(r, HTMLResult) for r in results)
    logs = get_json_logs(caplog)

    # Two separate semaphores should be created and acquired
    semaphore_creates = [log for log in logs if log.get("event") == "semaphore_create"]
    assert len(semaphore_creates) == 2
    assert any(log["domain"] == "example1.com" for log in semaphore_creates)
    assert any(log["domain"] == "example2.com" for log in semaphore_creates)

    # If responses had delays, end_time - start_time would be approx 1 * delay_per_call


@pytest.mark.asyncio
async def test_logging_content_and_format(respx_mock: MockRouter, caplog, mock_config_settings):
    test_url = "http://example.com/logging_test.html"
    # Make content long enough to pass MIN_HTML_CONTENT_LENGTH
    html_content = "<html><body><p>" + "Log test content. " * 50 + "</p></body></html>"
    
    respx_mock.head(test_url).respond(headers={"Content-Type": "text/html"})
    respx_mock.get(test_url).respond(content=html_content, headers={"Content-Type": "text/html"})

    caplog.set_level(logging.INFO, logger="resolve_pdf_link")
    
    # Mock readability to ensure predictable output for checking logs
    with patch("tools.resolve_pdf_link.Document") as mock_readability_doc:
        mock_doc_instance = MagicMock()
        mock_doc_instance.summary.return_value = "<p>" + "Log test content. " * 50 + "</p>"
        mock_readability_doc.return_value = mock_doc_instance
        
        result = await resolve_content(test_url)

    assert isinstance(result, HTMLResult)
    
    logs = get_json_logs(caplog)
    assert len(logs) > 0 # Ensure some logs were captured

    # Check a few key log events
    assert any(log["event"] == "resolve_start" and log["url"] == test_url for log in logs)
    assert any(log["event"] == "head_attempt" and log["method"] == "HEAD" for log in logs)
    assert any(log["event"] == "get_attempt" and log["method"] == "GET" for log in logs)
    assert any(log["event"] == "get_html_detected" for log in logs)
    assert any(log["event"] == "html_extract_readability_success" for log in logs) # if readability path taken
    assert any(log["event"] == "html_sufficient_content_success" for log in logs)
    assert any(log["event"] == "resolve_end" and log["url"] == test_url for log in logs)

    # Check structure of a sample log
    sample_log = next((log for log in logs if log["event"] == "resolve_start"), None)
    assert sample_log is not None
    assert "timestamp" in sample_log
    assert sample_log["level"] == "INFO"
    assert sample_log["module"] == "resolve_pdf_link"
    assert "message" in sample_log
    assert sample_log["url"] == test_url


@pytest.mark.asyncio
async def test_timeout_retries(respx_mock: MockRouter, caplog, mock_config_settings, mock_datetime_now):
    test_url = "http://example.com/timeout_test.pdf"
    pdf_content = b"%PDF-1.4\n%Timeout Test PDF"
    max_retries = mock_config_settings["resolve_max_retries"]

    # All HEAD attempts time out, then the last one succeeds (or fails if max_retries reached before success)
    head_effects = [httpx.TimeoutException("Test Timeout", request=None)] * max_retries + \
                   [httpx.Response(200, headers={"Content-Type": "application/pdf"})]
    
    respx_mock.head(test_url).side_effect = head_effects
    respx_mock.get(test_url).respond(content=pdf_content, headers={"Content-Type": "application/pdf"})

    caplog.set_level(logging.INFO, logger="resolve_pdf_link")

    with patch("tools.resolve_pdf_link.os.path.getsize", return_value=(MIN_PDF_SIZE_KB + 1) * 1024), \
         patch("tools.resolve_pdf_link.fitz.open") as mock_fitz_open:
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.__len__.return_value = 1 # 1 page is fine if text is sufficient
        
        # Mock the page and get_text call
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is valid PDF text, long enough to pass checks." * 5
        mock_pdf_doc.load_page.return_value = mock_page
        
        mock_fitz_open.return_value = mock_pdf_doc
        result = await resolve_content(test_url)

    assert isinstance(result, FileResult)

    logs = get_json_logs(caplog)
    head_attempts = [log for log in logs if log.get("event") == "head_attempt" and log.get("url") == test_url]
    assert len(head_attempts) == max_retries + 1

    retry_network_logs = [log for log in logs if log.get("event") == "retry_scheduled_network" and "TimeoutException" in log.get("message", "")]
    assert len(retry_network_logs) == max_retries
    
    assert any(log["event"] == "head_success" for log in logs)


@pytest.mark.asyncio
async def test_configuration_impact_max_retries(respx_mock: MockRouter, caplog, mock_config_settings):
    # Override max_retries for this specific test
    new_max_retries = 1
    mock_config_settings["resolve_max_retries"] = new_max_retries
    # Important: Patch the module-level constant that's actually used by the code
    with patch("tools.resolve_pdf_link.MAX_RETRIES", new_max_retries):
        test_url = "http://example.com/config_retries.html"

        # All attempts fail
        respx_mock.head(test_url).mock(return_value=httpx.Response(503))
        
        caplog.set_level(logging.INFO, logger="resolve_pdf_link")
        result = await resolve_content(test_url)

        assert isinstance(result, Failure)
        
        logs = get_json_logs(caplog)
        head_attempts = [log for log in logs if log.get("event") == "head_attempt" and log.get("url") == test_url]
        # Expected attempts = new_max_retries (1) + initial attempt (1) = 2
        assert len(head_attempts) == new_max_retries + 1

        retry_scheduled_logs = [log for log in logs if log.get("event") == "retry_scheduled_status"]
        assert len(retry_scheduled_logs) == new_max_retries
        
        final_failure_log = next(log for log in logs if log.get("event") == "request_failure_status" and log.get("method") == "HEAD")
        assert final_failure_log["max_retries"] == new_max_retries # Check if the log reflects the config used
