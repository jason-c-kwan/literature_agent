import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
from tools.search import AsyncSearchClient, SearchResult

MOCKED_API_EMAIL = "test@example.com"
MOCKED_PUBMED_KEY = "test_pubmed_key"
MOCKED_S2_KEY = "test_s2_key"

@pytest_asyncio.fixture
async def search_client_with_mocks():
    # Using 4-space indentation consistently
    with patch('tools.search.load_dotenv', MagicMock()), \
         patch('tools.search.API_EMAIL', MOCKED_API_EMAIL), \
         patch('tools.search.PUBMED_API_KEY', MOCKED_PUBMED_KEY), \
         patch('tools.search.SEMANTIC_SCHOLAR_API_KEY', MOCKED_S2_KEY), \
         patch('tools.search.UNPAYWALL_EMAIL', MOCKED_API_EMAIL), \
         patch('tools.search.CROSSREF_MAILTO', MOCKED_API_EMAIL):
        
        with patch('tools.search.Entrez') as MockEntrez, \
             patch('tools.search.EuropePMC', new_callable=MagicMock) as MockEuropePMCClass, \
             patch('tools.search.SemanticScholar') as MockS2Class, \
             patch('tools.search.Works') as MockWorksClass, \
             patch('tools.search.Unpywall') as MockUnpywallGlobal, \
             patch('tools.search._Works_imported', True):

            client = AsyncSearchClient()
            
            mocks = {
                "Entrez": MockEntrez,
                "MockEuropePMCClass": MockEuropePMCClass,
                "EuropePMCInstance": MockEuropePMCClass.return_value,
                "S2Instance": client.s2,
                "WorksInstance": client.cr,
                "UnpywallGlobal": MockUnpywallGlobal,
                "MockS2Class": MockS2Class,
                "MockWorksClass": MockWorksClass,
            }
            yield client, mocks
            await client.close()

def mock_async_run_sync(mocker, search_client_instance):
    async_mock = AsyncMock(side_effect=lambda sync_func, *args, **kwargs: sync_func(*args, **kwargs))
    mocker.patch.object(search_client_instance, '_run_sync_in_thread', new=async_mock)

@pytest.mark.asyncio
async def test_search_pubmed_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    MockEntrez = mocks["Entrez"]
    mock_pmids = ["123", "456"]
    mock_esearch_handle, mock_esummary_handle = MagicMock(), MagicMock()
    mock_read_esummary_list_result = [
        {"Id": "123", "Title": "Title 1", "DOI": "doi1", "ArticleIds": {"doi": "doi1"}, "AuthorList": [], "PubDate": "2023", "Source": "J1"},
        {"Id": "456", "Title": "Title 2", "DOI": "doi2", "ArticleIds": {"doi": "doi2"}, "AuthorList": [], "PubDate": "2024", "Source": "J2"}
    ]
    MockEntrez.esearch.return_value = mock_esearch_handle
    MockEntrez.esummary.return_value = mock_esummary_handle
    def entrez_read_side_effect(handle, *args, **kwargs):
        if handle == mock_esearch_handle: return {"IdList": mock_pmids, "QueryKey": "1", "WebEnv": "abc"}
        if handle == mock_esummary_handle: return mock_read_esummary_list_result
        raise ValueError(f"Unexpected handle for Entrez.read: {handle}")
    MockEntrez.read.side_effect = entrez_read_side_effect
    mock_async_run_sync(mocker, search_client)
    results = await search_client.search_pubmed("test query", max_results=2)
    assert len(results) == 2
    assert results[0]["pmid"] == "123"

@pytest.mark.asyncio
async def test_search_pubmed_no_results(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    MockEntrez = mocks["Entrez"]
    MockEntrez.esearch.return_value = MagicMock()
    MockEntrez.read.return_value = {"IdList": []} 
    mock_async_run_sync(mocker, search_client)
    results = await search_client.search_pubmed("test query", max_results=5)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_search_pubmed_api_error(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    MockEntrez = mocks["Entrez"]
    MockEntrez.esearch.side_effect = Exception("PubMed API Error")
    mock_async_run_sync(mocker, search_client)
    results = await search_client.search_pubmed("test query")
    assert len(results) == 0

@pytest.mark.asyncio
async def test_search_europepmc_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    MockEuropePMCInstance = mocks["EuropePMCInstance"]
    mock_record = MagicMock(doi="10.test/epmc", pmid="epmc123", title="EuropePMC Test", 
                            authorList=[MagicMock(fullName="Author EPMC")], 
                            journalTitle="EPMC Journal", pubYear="2023", 
                            abstractText="Abstract here.", pmcid="PMC12345")
    MockEuropePMCInstance.search.return_value = [mock_record]
    mock_async_run_sync(mocker, search_client)
    results = await search_client.search_europepmc("test query", max_results=1)
    assert len(results) == 1
    assert results[0]["doi"] == "10.test/epmc"

@pytest.mark.asyncio
async def test_search_semanticscholar_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    mock_paper = MagicMock(title="S2 Title", authors=[{'name': 'S2 Author'}], year=2023, 
                           journal={'name': 'S2 Journal'}, 
                           externalIds={'PubMed': 's2pmid123', 'DOI': '10.test/s2', 'PubMedCentral': 's2pmcid123'}, 
                           abstract="S2 abstract.", url="http://s2.example.com")
    mocks["S2Instance"].search_paper.return_value = [mock_paper] 
    mock_async_run_sync(mocker, search_client)
    results = await search_client.search_semanticscholar("test query", max_results=1)
    assert len(results) == 1
    assert results[0]["title"] == "S2 Title"

@pytest.mark.asyncio
async def test_search_crossref_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    mock_item = {"DOI": "10.test/cr", "title": ["CrossRef Title"], "author": [{"given": "CR", "family": "Author"}],
                 "issued": {"date-parts": [[2023]]}, "container-title": ["CR Journal"], "URL": "http://cr.example.com"}
    # Mock the chain: client.cr.query(bibliographic=...).sample(max_results)
    # MockWorksClass is the class itself, client.cr is the instance
    if mocks["WorksInstance"]: # client.cr is the mocked Works instance
        mock_query_result = MagicMock()
        mock_query_result.sample.return_value = [mock_item]
        mocks["WorksInstance"].query.return_value = mock_query_result
    
    mock_async_run_sync(mocker, search_client)
    results = await search_client.search_crossref("test query", max_results=1)
    
    # Check if the Works class mock itself was available (i.e., not None due to import issues)
    # and if the instance mock (client.cr) was created.
    if mocks["MockWorksClass"] is not None and mocks["WorksInstance"] is not None:
        assert len(results) == 1
        assert results[0]["doi"] == "10.test/cr"
    else: # If CrossRef was effectively disabled due to import issues not caught by create=True
        assert len(results) == 0

@pytest.mark.asyncio
async def test_search_unpaywall_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    MockUnpywallGlobal = mocks["UnpywallGlobal"] # This is the mock for the 'Unpywall' module in tools.search
    mock_response_obj = MagicMock(doi="10.test/unpaywall", is_oa=True, 
                                  best_oa_location=MagicMock(url="http://oa.example.com"))
    # Unpywall.doi is a static method or class method, so we mock it on the module mock
    # Unpywall.doi returns a pandas DataFrame
    mock_df_data = {
        "doi": ["10.test/unpaywall"],
        "is_oa": [True],
        "best_oa_location": [{"url": "http://oa.example.com"}]
    }
    MockUnpywallGlobal.doi.return_value = pd.DataFrame(mock_df_data)
    mock_async_run_sync(mocker, search_client)
    results_map = await search_client.search_unpaywall(["10.test/unpaywall"])
    assert len(results_map) == 1
    assert results_map["10.test/unpaywall"]["is_open_access"] is True

@pytest.mark.asyncio
async def test_search_all_deduplication_and_enrichment(search_client_with_mocks, mocker):
    search_client, _ = search_client_with_mocks 
    pubmed_res = [SearchResult(doi="10.1000/common", pmid="pmid1", title="PubMed Common", source_api="PubMed")]
    epmc_res = [SearchResult(doi="10.1000/common", pmid="pmid1_epmc", title="EPMC Common", source_api="EuropePMC")]
    s2_res = [SearchResult(doi="10.1000/common", pmid="pmid1_s2", title="S2 Common Richer", source_api="SemanticScholar")]
    cr_res = [SearchResult(doi="10.3000/cr_unique", title="CR Unique", source_api="CrossRef")]
    mocker.patch.object(search_client, 'search_pubmed', AsyncMock(return_value=pubmed_res))
    mocker.patch.object(search_client, 'search_europepmc', AsyncMock(return_value=epmc_res))
    mocker.patch.object(search_client, 'search_semanticscholar', AsyncMock(return_value=s2_res))
    mocker.patch.object(search_client, 'search_crossref', AsyncMock(return_value=cr_res))
    unpaywall_data = {"10.1000/common": SearchResult(is_open_access=True, open_access_url="http://oa.common.com")}
    mocker.patch.object(search_client, 'search_unpaywall', AsyncMock(return_value=unpaywall_data))
    df = await search_client.search_all("test query", max_results_per_source=5)
    assert len(df) == 2 

@pytest.mark.asyncio
async def test_search_all_empty_results(search_client_with_mocks, mocker):
    search_client, _ = search_client_with_mocks
    mocker.patch.object(search_client, 'search_pubmed', AsyncMock(return_value=[]))
    mocker.patch.object(search_client, 'search_europepmc', AsyncMock(return_value=[]))
    mocker.patch.object(search_client, 'search_semanticscholar', AsyncMock(return_value=[]))
    mocker.patch.object(search_client, 'search_crossref', AsyncMock(return_value=[]))
    mocker.patch.object(search_client, 'search_unpaywall', AsyncMock(return_value={}))
    df = await search_client.search_all("query with no results")
    assert df.empty

@pytest.mark.asyncio
async def test_search_pubmed_handles_429_if_semaphore_bypassed(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    MockEntrez = mocks["Entrez"]
    from urllib.error import HTTPError 
    async def mock_run_sync_error_for_esearch(func, *args, **kwargs):
        if func == MockEntrez.esearch: 
            raise HTTPError("url", 429, "Too Many Requests", {}, MagicMock())
        return func(*args, **kwargs) 
    mocker.patch.object(search_client, '_run_sync_in_thread', AsyncMock(side_effect=mock_run_sync_error_for_esearch))
    results = await search_client.search_pubmed("test query")
    assert len(results) == 0
