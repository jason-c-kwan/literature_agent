import pytest
import pytest_asyncio
import httpx
from typing import Dict, Any, Optional

# Import the function to test
from tools.retrieve_europepmc import fetch_europepmc

@pytest_asyncio.fixture
async def mock_httpx_client(mocker):
    """
    Fixture to mock httpx.AsyncClient and its methods.
    It yields the mock client instance.
    """
    mock_client = mocker.MagicMock(spec=httpx.AsyncClient)
    mock_response = mocker.MagicMock(spec=httpx.Response)
    
    # Make `get` an AsyncMock
    mock_client.get = mocker.AsyncMock(return_value=mock_response)
    
    # Allow __aenter__ and __aexit__ to be called
    mock_client.__aenter__.return_value = mock_client 
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    # Patch httpx.AsyncClient to return this mock_client
    mocker.patch('httpx.AsyncClient', return_value=mock_client)
    
    return mock_client, mock_response


@pytest.mark.asyncio
async def test_fetch_success_pmcid_direct_json(mock_httpx_client, mocker):
    """Test successful retrieval via PMCID, full text is direct JSON."""
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi = "10.1234/success_pmcid_json"
    dummy_pmcid = "PMC123JSON"
    
    metadata_payload: Dict[str, Any] = {"doi": dummy_doi, "pmcid": dummy_pmcid}
    api_metadata_response: Dict[str, Any] = {"resultList": {"result": [metadata_payload]}}
    
    full_text_payload: Dict[str, Any] = {"article": {"title": "Full Text Direct JSON"}}

    mock_response_metadata.status_code = 200
    mock_response_metadata.json = mocker.MagicMock(return_value=api_metadata_response)

    mock_response_fulltext = mocker.MagicMock(spec=httpx.Response)
    mock_response_fulltext.status_code = 200
    mock_response_fulltext.json = mocker.MagicMock(return_value=full_text_payload) # Direct JSON
    
    mock_client.get.side_effect = [mock_response_metadata, mock_response_fulltext]
    
    from urllib.parse import quote as url_quote
    encoded_doi = url_quote(dummy_doi)
    expected_metadata_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:\"{encoded_doi}\"&format=json&resulttype=core"
    expected_ft_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{dummy_pmcid}/fullTextXML?format=json"

    result = await fetch_europepmc(dummy_doi)
    
    assert mock_client.get.call_count == 2
    mock_client.get.assert_any_call(expected_metadata_url)
    mock_client.get.assert_any_call(expected_ft_url)
    assert result == full_text_payload

@pytest.mark.asyncio
async def test_fetch_success_pmcid_xml_to_dict(mock_httpx_client, mocker):
    """Test successful retrieval via PMCID, full text is XML parsed to dict."""
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi = "10.1234/success_pmcid_xml"
    dummy_pmcid = "PMC123XML"
    
    metadata_payload: Dict[str, Any] = {"doi": dummy_doi, "pmcid": dummy_pmcid}
    api_metadata_response: Dict[str, Any] = {"resultList": {"result": [metadata_payload]}}
    
    xml_text_content = "<article><title>Full Text from XML</title></article>"
    expected_full_text_dict: Dict[str, Any] = {"article": {"title": "Full Text from XML"}}

    mock_response_metadata.status_code = 200
    mock_response_metadata.json = mocker.MagicMock(return_value=api_metadata_response)

    mock_response_fulltext_xml = mocker.MagicMock(spec=httpx.Response)
    mock_response_fulltext_xml.status_code = 200
    mock_response_fulltext_xml.json = mocker.MagicMock(side_effect=ValueError("Not JSON")) # Simulate JSON parse fail
    mock_response_fulltext_xml.text = xml_text_content
    
    mock_client.get.side_effect = [mock_response_metadata, mock_response_fulltext_xml]
    
    from urllib.parse import quote as url_quote
    encoded_doi = url_quote(dummy_doi)
    expected_metadata_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:\"{encoded_doi}\"&format=json&resulttype=core"
    expected_ft_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{dummy_pmcid}/fullTextXML?format=json"
    
    result = await fetch_europepmc(dummy_doi)
    
    assert mock_client.get.call_count == 2
    mock_client.get.assert_any_call(expected_metadata_url)
    mock_client.get.assert_any_call(expected_ft_url)
    assert result == expected_full_text_dict

@pytest.mark.asyncio
async def test_fetch_success_via_fulltexturllist_direct_json(mock_httpx_client, mocker):
    """Test success via fullTextUrlList (direct JSON URL)."""
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi = "10.1234/ftlist_direct_json"
    direct_json_ft_url = "http://example.com/api/article.json"

    metadata_payload: Dict[str, Any] = { # No PMCID, rely on fullTextUrlList
        "doi": dummy_doi, 
        "fullTextUrlList": {"fullTextUrl": [{"documentStyle": "json", "url": direct_json_ft_url}]}
    }
    api_metadata_response: Dict[str, Any] = {"resultList": {"result": [metadata_payload]}}
    full_text_payload: Dict[str, Any] = {"data": "Direct JSON from list"}

    mock_response_metadata.status_code = 200
    mock_response_metadata.json = mocker.MagicMock(return_value=api_metadata_response)
    
    mock_response_fulltext = mocker.MagicMock(spec=httpx.Response)
    mock_response_fulltext.status_code = 200
    mock_response_fulltext.json = mocker.MagicMock(return_value=full_text_payload)
    
    mock_client.get.side_effect = [mock_response_metadata, mock_response_fulltext]
    result = await fetch_europepmc(dummy_doi)
    assert mock_client.get.call_count == 2
    mock_client.get.assert_any_call(direct_json_ft_url) # Check the second call URL
    assert result == full_text_payload


@pytest.mark.asyncio
async def test_fetch_success_via_fulltexturllist_xml_to_json(mock_httpx_client, mocker):
    """Test success via fullTextUrlList (XML URL converted to JSON)."""
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi = "10.1234/ftlist_xml_to_json"
    xml_ft_url_base = "http://example.com/api/article.xml"
    xml_ft_url_with_format = f"{xml_ft_url_base}?format=json"

    metadata_payload: Dict[str, Any] = { # No PMCID
        "doi": dummy_doi,
        "fullTextUrlList": {"fullTextUrl": [{"documentStyle": "xml", "url": xml_ft_url_base}]}
    }
    api_metadata_response: Dict[str, Any] = {"resultList": {"result": [metadata_payload]}}
    
    xml_text_content = "<doc><content>XML from list</content></doc>"
    expected_full_text_dict: Dict[str, Any] = {"doc": {"content": "XML from list"}}

    mock_response_metadata.status_code = 200
    mock_response_metadata.json = mocker.MagicMock(return_value=api_metadata_response)

    mock_response_fulltext_xml = mocker.MagicMock(spec=httpx.Response)
    mock_response_fulltext_xml.status_code = 200
    mock_response_fulltext_xml.json = mocker.MagicMock(side_effect=ValueError("Not JSON"))
    mock_response_fulltext_xml.text = xml_text_content
    
    mock_client.get.side_effect = [mock_response_metadata, mock_response_fulltext_xml]
    result = await fetch_europepmc(dummy_doi)
    assert mock_client.get.call_count == 2
    mock_client.get.assert_any_call(xml_ft_url_with_format)
    assert result == expected_full_text_dict


@pytest.mark.asyncio
async def test_fetch_no_suitable_fulltext_url(mock_httpx_client, mocker):
    """Test when metadata is found but no PMCID and no suitable fullTextUrlList entry."""
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi = "10.1234/nosuitableurl"
    metadata_payload: Dict[str, Any] = { # No PMCID
        "doi": dummy_doi,
        "fullTextUrlList": { "fullTextUrl": [{"documentStyle": "pdf", "url": "http://example.com/paper.pdf"}]}
    }
    api_metadata_response: Dict[str, Any] = {"resultList": {"result": [metadata_payload]}}
    mock_response_metadata.status_code = 200
    mock_response_metadata.json = mocker.MagicMock(return_value=api_metadata_response)
    mock_client.get.return_value = mock_response_metadata

    result = await fetch_europepmc(dummy_doi)
    assert result is None
    assert mock_client.get.call_count == 1

@pytest.mark.asyncio
async def test_fetch_fulltext_fails_404(mock_httpx_client, mocker):
    """Test when PMCID path for full-text returns 404."""
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi = "10.1234/ft_404"
    dummy_pmcid = "PMC_FT_404"
    metadata_payload: Dict[str, Any] = {"doi": dummy_doi, "pmcid": dummy_pmcid}
    api_metadata_response: Dict[str, Any] = {"resultList": {"result": [metadata_payload]}}
    mock_response_metadata.status_code = 200
    mock_response_metadata.json = mocker.MagicMock(return_value=api_metadata_response)

    mock_response_fulltext_404 = mocker.MagicMock(spec=httpx.Response)
    mock_response_fulltext_404.status_code = 404
    
    mock_client.get.side_effect = [mock_response_metadata, mock_response_fulltext_404]

    result = await fetch_europepmc(dummy_doi)
    assert result is None
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_fetch_metadata_api_404(mock_httpx_client, mocker): # Renamed for clarity
    """
    Test handling of an API 'Not Found' (HTTP 404) response for metadata.
    """
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi_not_found = "10.1234/notfound404"
    mock_response_metadata.status_code = 404
    mock_response_metadata.json = mocker.MagicMock() 
    mock_client.get.return_value = mock_response_metadata # Only one call expected

    from urllib.parse import quote as url_quote
    encoded_doi = url_quote(dummy_doi_not_found)
    expected_metadata_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:\"{encoded_doi}\"&format=json&resulttype=core"

    result = await fetch_europepmc(dummy_doi_not_found)
    
    mock_client.get.assert_called_once_with(expected_metadata_url)
    mock_response_metadata.json.assert_not_called()
    assert result is None

@pytest.mark.asyncio
async def test_fetch_metadata_200_empty_results(mock_httpx_client, mocker):
    """
    Test handling of a 200 OK metadata response but with no matching article.
    """
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi_empty_result = "10.1234/emptyresult"
    api_response_payload_empty: Dict[str, Any] = {"resultList": {"result": []}}
    mock_response_metadata.status_code = 200
    mock_response_metadata.json = mocker.MagicMock(return_value=api_response_payload_empty)
    mock_client.get.return_value = mock_response_metadata # Only one call expected
    
    result = await fetch_europepmc(dummy_doi_empty_result)
    assert result is None
    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_fetch_metadata_error_http_status(mock_httpx_client, mocker): # Renamed for clarity
    """
    Test HTTPStatusError during metadata fetch.
    """
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi_error = "10.1234/meta_error"
    mock_response_metadata.status_code = 500
    http_error = httpx.HTTPStatusError("Server Error", request=mocker.MagicMock(spec=httpx.Request), response=mock_response_metadata)
    mock_response_metadata.raise_for_status = mocker.MagicMock(side_effect=http_error)
    mock_client.get.return_value = mock_response_metadata # Only one call expected

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_europepmc(dummy_doi_error)
    assert mock_client.get.call_count == 1
    mock_response_metadata.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_metadata_request_error(mock_httpx_client, mocker):
    """
    Test httpx.RequestError during metadata fetch.
    """
    mock_client, _ = mock_httpx_client 
    dummy_doi_req_error = "10.1234/meta_req_error"
    request_error = httpx.RequestError("Network Error", request=mocker.MagicMock(spec=httpx.Request))
    mock_client.get.side_effect = request_error 
    
    result = await fetch_europepmc(dummy_doi_req_error)
    assert result is None
    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_fetch_metadata_json_parsing_error(mock_httpx_client, mocker):
    """
    Test errors during metadata JSON parsing.
    """
    mock_client, mock_response_metadata = mock_httpx_client
    dummy_doi_parse_error = "10.1234/meta_parseerror"
    mock_response_metadata.status_code = 200
    mock_response_metadata.json = mocker.MagicMock(return_value={"unexpected_structure": True})
    mock_client.get.return_value = mock_response_metadata

    result = await fetch_europepmc(dummy_doi_parse_error)
    assert result is None
    assert mock_client.get.call_count == 1
