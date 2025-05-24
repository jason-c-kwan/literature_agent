import pytest
import httpx
from unittest.mock import AsyncMock, patch

# Import the functions to be tested
from tools.elink_pubmed import (
    get_article_links,
    _convert_to_pmid,
    get_pubmed_prlinks,
    NCBI_ELINK_URL, 
    NCBI_ESEARCH_URL
)

# --- Mock XML Responses ---

# ESearch: DOI to PMID success
MOCK_ESEARCH_DOI_SUCCESS_XML = """
<eSearchResult>
    <Count>1</Count>
    <RetMax>1</RetMax>
    <RetStart>0</RetStart>
    <IdList>
        <Id>123456</Id>
    </IdList>
</eSearchResult>
"""

# ESearch: PMC to PMID success
MOCK_ESEARCH_PMC_SUCCESS_XML = """
<eSearchResult>
    <Count>1</Count>
    <RetMax>1</RetMax>
    <RetStart>0</RetStart>
    <IdList>
        <Id>789012</Id>
    </IdList>
</eSearchResult>
"""

# ESearch: No result found
MOCK_ESEARCH_NO_RESULT_XML = """
<eSearchResult>
    <Count>0</Count>
    <RetMax>0</RetMax>
    <RetStart>0</RetStart>
    <IdList/>
</eSearchResult>
"""

# ESearch: API Error in XML
MOCK_ESEARCH_API_ERROR_XML = """
<eSearchResult>
    <ERROR>Some ESearch API error occurred</ERROR>
</eSearchResult>
"""

# ELink: prlinks success with multiple links
MOCK_ELINK_PRLINKS_SUCCESS_XML = """
<eLinkResult>
    <LinkSet>
        <DbFrom>pubmed</DbFrom>
        <IdUrlList>
            <IdUrlSet>
                <Id>123456</Id>
                <ObjUrl>
                    <Url>http://example.com/fulltext/123456</Url>
                    <Provider><Name>PublisherA</Name></Provider>
                </ObjUrl>
                <ObjUrl>
                    <Url>http://another.example.com/pdf/123456.pdf</Url>
                    <Provider><Name>AggregatorB</Name></Provider>
                </ObjUrl>
            </IdUrlSet>
        </IdUrlList>
    </LinkSet>
</eLinkResult>
"""

# ELink: prlinks success with LinkSetDb/Link structure
MOCK_ELINK_PRLINKS_LINKSETDB_SUCCESS_XML = """
<eLinkResult>
 <LinkSet>
   <DbFrom>pubmed</DbFrom>
   <IdList><Id>98765</Id></IdList>
   <LinkSetDb>
      <LinkName>pubmed_pubmed_prlinks</LinkName>
      <Link><Url>http://example.com/linksetdb/98765</Url></Link>
   </LinkSetDb>
 </LinkSet>
</eLinkResult>
"""


# ELink: prlinks no links found
MOCK_ELINK_PRLINKS_NO_LINKS_XML = """
<eLinkResult>
    <LinkSet>
        <DbFrom>pubmed</DbFrom>
        <IdUrlList>
            <IdUrlSet>
                <Id>123456</Id>
                <!-- No ObjUrl elements -->
            </IdUrlSet>
        </IdUrlList>
    </LinkSet>
</eLinkResult>
"""

# ELink: API Error in XML
MOCK_ELINK_API_ERROR_XML = """
<eLinkResult>
    <ERROR>Some ELink API error occurred</ERROR>
</eLinkResult>
"""

# --- Helper to create mock responses with request attribute ---
def create_mock_response(status_code: int, text_content: str, url: str) -> httpx.Response:
    request = httpx.Request("GET", url)
    return httpx.Response(status_code, text=text_content, request=request)

# --- Test Cases ---

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_convert_to_pmid_doi_success(mock_execute_request):
    mock_response = create_mock_response(200, MOCK_ESEARCH_DOI_SUCCESS_XML, NCBI_ESEARCH_URL)
    mock_execute_request.return_value = mock_response
    
    pmid = await _convert_to_pmid("10.1234/journal.123", "DOI")
    assert pmid == "123456"
    mock_execute_request.assert_called_once()

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_convert_to_pmid_pmc_success(mock_execute_request):
    mock_response = create_mock_response(200, MOCK_ESEARCH_PMC_SUCCESS_XML, NCBI_ESEARCH_URL)
    mock_execute_request.return_value = mock_response
    
    pmid = await _convert_to_pmid("PMC12345", "PMCID")
    assert pmid == "789012"

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_convert_to_pmid_no_result(mock_execute_request):
    mock_response = create_mock_response(200, MOCK_ESEARCH_NO_RESULT_XML, NCBI_ESEARCH_URL)
    mock_execute_request.return_value = mock_response
    
    pmid = await _convert_to_pmid("10.9999/nonexistent", "DOI")
    assert pmid is None

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_convert_to_pmid_api_error_xml(mock_execute_request):
    mock_response = create_mock_response(200, MOCK_ESEARCH_API_ERROR_XML, NCBI_ESEARCH_URL)
    mock_execute_request.return_value = mock_response
    
    pmid = await _convert_to_pmid("10.1234/journal.123", "DOI")
    assert pmid is None

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_convert_to_pmid_http_error(mock_execute_request):
    error_response = create_mock_response(500, "Internal Server Error", NCBI_ESEARCH_URL)
    mock_execute_request.side_effect = httpx.HTTPStatusError(
        "Server Error", request=error_response.request, response=error_response
    )
    pmid = await _convert_to_pmid("10.1234/journal.123", "DOI")
    assert pmid is None
    assert mock_execute_request.call_count == 3 # Due to retries

@pytest.mark.asyncio
async def test_convert_to_pmid_empty_identifier():
    pmid = await _convert_to_pmid("", "DOI")
    assert pmid is None
    pmid = await _convert_to_pmid("  ", "PMCID")
    assert pmid is None


@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_get_pubmed_prlinks_success(mock_execute_request):
    mock_response = create_mock_response(200, MOCK_ELINK_PRLINKS_SUCCESS_XML, NCBI_ELINK_URL)
    mock_execute_request.return_value = mock_response
    
    links = await get_pubmed_prlinks("123456")
    assert len(links) == 2
    assert "http://example.com/fulltext/123456" in links
    assert "http://another.example.com/pdf/123456.pdf" in links

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_get_pubmed_prlinks_linksetdb_success(mock_execute_request):
    mock_response = create_mock_response(200, MOCK_ELINK_PRLINKS_LINKSETDB_SUCCESS_XML, NCBI_ELINK_URL)
    mock_execute_request.return_value = mock_response
    
    links = await get_pubmed_prlinks("98765")
    assert len(links) == 1
    assert "http://example.com/linksetdb/98765" in links

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_get_pubmed_prlinks_no_links(mock_execute_request):
    mock_response = create_mock_response(200, MOCK_ELINK_PRLINKS_NO_LINKS_XML, NCBI_ELINK_URL)
    mock_execute_request.return_value = mock_response
    
    links = await get_pubmed_prlinks("123456")
    assert len(links) == 0

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_get_pubmed_prlinks_api_error_xml(mock_execute_request):
    mock_response = create_mock_response(200, MOCK_ELINK_API_ERROR_XML, NCBI_ELINK_URL)
    mock_execute_request.return_value = mock_response
    
    links = await get_pubmed_prlinks("123456")
    assert len(links) == 0

@pytest.mark.asyncio
@patch('tools.elink_pubmed._execute_ncbi_utility_request', new_callable=AsyncMock)
async def test_get_pubmed_prlinks_http_error(mock_execute_request):
    error_response = create_mock_response(500, "Internal Server Error", NCBI_ELINK_URL)
    mock_execute_request.side_effect = httpx.HTTPStatusError(
        "Server Error", request=error_response.request, response=error_response
    )
    links = await get_pubmed_prlinks("123456")
    assert len(links) == 0
    assert mock_execute_request.call_count == 3 # Due to retries

@pytest.mark.asyncio
async def test_get_pubmed_prlinks_empty_pmid():
    links = await get_pubmed_prlinks("")
    assert len(links) == 0
    links = await get_pubmed_prlinks("  ")
    assert len(links) == 0


@pytest.mark.asyncio
@patch('tools.elink_pubmed._convert_to_pmid', new_callable=AsyncMock)
@patch('tools.elink_pubmed.get_pubmed_prlinks', new_callable=AsyncMock)
async def test_get_article_links_with_pmid(mock_get_prlinks, mock_convert_pmid):
    mock_get_prlinks.return_value = ["http://example.com/link_pmid"]
    
    links = await get_article_links("111", "pmid")
    assert links == ["http://example.com/link_pmid"]
    mock_convert_pmid.assert_not_called()
    mock_get_prlinks.assert_called_once_with("111")

@pytest.mark.asyncio
@patch('tools.elink_pubmed._convert_to_pmid', new_callable=AsyncMock)
@patch('tools.elink_pubmed.get_pubmed_prlinks', new_callable=AsyncMock)
async def test_get_article_links_with_doi(mock_get_prlinks, mock_convert_pmid):
    mock_convert_pmid.return_value = "222" # Converted PMID
    mock_get_prlinks.return_value = ["http://example.com/link_doi"]
    
    links = await get_article_links("10.123/doi", "doi")
    assert links == ["http://example.com/link_doi"]
    mock_convert_pmid.assert_called_once_with("10.123/doi", "DOI")
    mock_get_prlinks.assert_called_once_with("222")

@pytest.mark.asyncio
@patch('tools.elink_pubmed._convert_to_pmid', new_callable=AsyncMock)
@patch('tools.elink_pubmed.get_pubmed_prlinks', new_callable=AsyncMock)
async def test_get_article_links_with_pmc(mock_get_prlinks, mock_convert_pmid):
    mock_convert_pmid.return_value = "333" # Converted PMID
    mock_get_prlinks.return_value = ["http://example.com/link_pmc"]
    
    links = await get_article_links("PMC123", "pmc")
    assert links == ["http://example.com/link_pmc"]
    mock_convert_pmid.assert_called_once_with("PMC123", "PMCID")
    mock_get_prlinks.assert_called_once_with("333")

@pytest.mark.asyncio
@patch('tools.elink_pubmed._convert_to_pmid', new_callable=AsyncMock)
async def test_get_article_links_doi_conversion_fails(mock_convert_pmid):
    mock_convert_pmid.return_value = None # Conversion fails
    
    links = await get_article_links("10.failed/doi", "doi")
    assert len(links) == 0
    mock_convert_pmid.assert_called_once_with("10.failed/doi", "DOI")

@pytest.mark.asyncio
async def test_get_article_links_invalid_type():
    links = await get_article_links("123", "invalid_type")
    assert len(links) == 0

@pytest.mark.asyncio
async def test_get_article_links_empty_identifier():
    links = await get_article_links("", "pmid")
    assert len(links) == 0
    links = await get_article_links("  ", "doi")
    assert len(links) == 0
