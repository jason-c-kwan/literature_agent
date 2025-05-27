import asyncio
import os
import shutil
from unittest.mock import MagicMock, patch
import pytest
import httpx
from respx import MockRouter
import yaml # For creating mock config content

# Assuming resolve_pdf_link is in tools directory, adjust path if necessary
from tools.resolve_pdf_link import (
    resolve_content,
    Failure,
    HTMLResult,
    FileResult,
    DOWNLOADS_DIR, # from resolve_pdf_link
    MIN_PDF_SIZE_KB, # from resolve_pdf_link
    PDF_UNLOCK_PARAMS_PATH, # from resolve_pdf_link (will be added)
    # load_pdf_unlock_params # Will be added to resolve_pdf_link
)

# Fixture from test_resolve_pdf_link.py (can be shared or duplicated)
@pytest.fixture(autouse=True)
def ensure_downloads_dir_is_clean():
    if os.path.exists(DOWNLOADS_DIR):
        shutil.rmtree(DOWNLOADS_DIR)
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    yield
    if os.path.exists(DOWNLOADS_DIR):
        shutil.rmtree(DOWNLOADS_DIR)

@pytest.fixture
def mock_datetime_now(mocker):
    mock_dt_instance = MagicMock()
    mock_dt_instance.strftime.return_value = "20250101_100000" # Unique timestamp for test
    mock_datetime_class = mocker.patch("tools.resolve_pdf_link.datetime.datetime")
    mock_datetime_class.now.return_value = mock_dt_instance
    return mock_datetime_class

@pytest.fixture
def mock_pdf_unlock_config(mocker):
    # This fixture will mock the load_pdf_unlock_params function
    # in the tools.resolve_pdf_link module.
    mock_config_data = {
        'default': ['param=default_val'],
        'nejm.org': ['download=true'],
        'tandfonline.com': ['needAccess=true'],
        'example.com': ['key1=val1', 'key2=val2']
    }
    # The function to mock is tools.resolve_pdf_link.load_pdf_unlock_params
    return mocker.patch('tools.resolve_pdf_link.load_pdf_unlock_params', return_value=mock_config_data)


@pytest.mark.asyncio
async def test_resolve_content_unlocks_nejm_pdf_with_param(
    respx_mock: MockRouter, 
    mock_pdf_unlock_config, # Use the mocked config loader
    ensure_downloads_dir_is_clean, 
    mock_datetime_now
):
    nejm_base_url = "https://www.nejm.org/doi/pdf/10.1056/NEJMoa2501440"
    nejm_unlocked_url = "https://www.nejm.org/doi/pdf/10.1056/NEJMoa2501440?download=true"
    pdf_content = b"%PDF-1.4\\n%NEJM Test PDF Content"

    # Initial request to base URL gets 403
    # IMPORTANT: respx routes based on exact URL match including query params.
    # So, the route for nejm_base_url (without params) should be distinct from nejm_unlocked_url (with params).
    respx_mock.get(nejm_base_url, headers__contains={"Accept": "application/pdf,*/*"}).respond(status_code=403)
    # Also handle if the first attempt is a broader accept header
    respx_mock.get(nejm_base_url, headers__contains={"Accept": "text/html"}).respond(status_code=403)


    # Request to URL with ?download=true gets 200 and PDF
    respx_mock.get(nejm_unlocked_url, headers__contains={"Accept": "application/pdf,*/*"}).respond(
        status_code=200,
        headers={"Content-Type": "application/pdf"},
        content=pdf_content
    )

    # Mock os.path.getsize and fitz.open for PDF validation
    with patch("tools.resolve_pdf_link.os.path.getsize", return_value=(MIN_PDF_SIZE_KB + 10) * 1024), \
         patch("tools.resolve_pdf_link.fitz.open") as mock_fitz_open:
        
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.__len__.return_value = 5 # Say 5 pages
        mock_fitz_open.return_value = mock_pdf_doc

        result = await resolve_content(nejm_base_url)

    assert isinstance(result, FileResult), f"Expected FileResult, got {type(result).__name__}: {getattr(result, 'reason', '')}"
    assert result.type == "file"
    
    # The filename should be based on the URL that successfully returned the PDF, which is nejm_unlocked_url
    expected_filename_part = "".join(c if c.isalnum() else "_" for c in nejm_unlocked_url.split("://")[-1][:50])
    expected_filename = f"20250101_100000_{expected_filename_part}.pdf"
    assert result.path == os.path.join(DOWNLOADS_DIR, expected_filename)
    
    assert os.path.exists(result.path)
    with open(result.path, "rb") as f:
        assert f.read() == pdf_content
    
    mock_fitz_open.assert_called_once_with(result.path)
    mock_pdf_doc.close.assert_called_once()

    # Verify that the initial 403 URL was called, then the successful one
    # Check calls based on the URL pattern, as headers might vary slightly if not strictly matched in respx_mock
    assert respx_mock.get(url__regex=r"https://www.nejm.org/doi/pdf/10.1056/NEJMoa2501440$").called # Base URL without params
    assert respx_mock.get(url__regex=r"https://www.nejm.org/doi/pdf/10.1056/NEJMoa2501440\?download=true$").called # Unlocked URL
