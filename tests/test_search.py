import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
from tools.search import AsyncSearchClient, search_literature, SearchResult

# Mock environment variables for tests
MOCKED_API_EMAIL = "test@example.com"
MOCKED_PUBMED_KEY = "test_pubmed_key"
MOCKED_S2_KEY = "test_s2_key"

@pytest_asyncio.fixture
async def search_client_with_mocks():
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
    # This mock ensures that _run_sync_in_thread directly calls the mocked sync function
    # instead of trying to run it in a real thread, which can cause issues with other mocks.
    async_mock = AsyncMock(side_effect=lambda sync_func, *args, **kwargs: sync_func(*args, **kwargs))
    mocker.patch.object(search_client_instance, '_run_sync_in_thread', new=async_mock)

# --- Tests for individual AsyncSearchClient methods (kept for completeness) ---

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
    if mocks["WorksInstance"]:
        mock_query_result = MagicMock()
        mock_query_result.sample.return_value = [mock_item]
        mocks["WorksInstance"].query.return_value = mock_query_result
    
    mock_async_run_sync(mocker, search_client)
    results = await search_client.search_crossref("test query", max_results=1)
    
    if mocks["MockWorksClass"] is not None and mocks["WorksInstance"] is not None:
        assert len(results) == 1
        assert results[0]["doi"] == "10.test/cr"
    else:
        assert len(results) == 0

@pytest.mark.asyncio
async def test_search_unpaywall_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    MockUnpywallGlobal = mocks["UnpywallGlobal"]
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

# --- New tests for search_literature and overall functionality ---

@pytest.fixture
def mock_async_search_literature_data():
    # Canned data for _async_search_literature
    return [
        SearchResult(title="Title A", authors=["Author A"], doi="10.1/A", pmid="1", year=2020, source_api="PubMed"),
        SearchResult(title="Title B", authors=["Author B"], doi="10.1/B", pmid="2", year=2021, source_api="EuropePMC"),
        SearchResult(title="Title C", authors=["Author C"], doi="10.1/C", pmid="3", year=2022, source_api="SemanticScholar"),
        SearchResult(title="Title D", authors=["Author D"], doi="10.1/D", pmid="4", year=2023, source_api="CrossRef"),
        # Duplicates to test deduplication
        SearchResult(title="Title A Duplicate", authors=["Author A"], doi="10.1/A", pmid="10", year=2019, source_api="EuropePMC"),
        SearchResult(title="Title E", authors=["Author E"], doi=None, pmid="5", year=2020, source_api="PubMed"),
        SearchResult(title="Title E Duplicate", authors=["Author E"], doi=None, pmid="5", year=2018, source_api="SemanticScholar"),
        SearchResult(title="Title F", authors=["Author F"], doi=None, pmid="6", year=2021, source_api="CrossRef"),
    ]

@pytest.fixture(autouse=True)
def mock_all_async_search_methods(mocker, mock_async_search_literature_data):
    # This fixture will run for all tests and mock the underlying async search methods
    # to prevent actual network calls for search_literature tests.
    # We need to mock search_all within AsyncSearchClient for _async_search_literature
    # to work with canned data.
    
    # Mock AsyncSearchClient's search_all to return a DataFrame from canned data
    async def mock_search_all(query: str, max_results_per_source: int):
        # Filter and limit data based on max_results_per_source if needed,
        # but for simplicity, we'll just return a subset of the canned data.
        # The deduplication and sorting logic is tested within _async_search_literature.
        return pd.DataFrame(mock_async_search_literature_data[:max_results_per_source * 4]) # Return enough for testing

    mocker.patch('tools.search.AsyncSearchClient.search_all', new_callable=AsyncMock, side_effect=mock_search_all)
    
    # DO NOT mock the individual search methods here, as other tests target them specifically.
    # mocker.patch('tools.search.AsyncSearchClient.search_pubmed', AsyncMock(return_value=[]))
    # mocker.patch('tools.search.AsyncSearchClient.search_europepmc', AsyncMock(return_value=[]))
    # mocker.patch('tools.search.AsyncSearchClient.search_semanticscholar', AsyncMock(return_value=[]))
    # mocker.patch('tools.search.AsyncSearchClient.search_crossref', AsyncMock(return_value=[]))
    # mocker.patch('tools.search.AsyncSearchClient.search_unpaywall', AsyncMock(return_value={}))
    
    # Mock nest_asyncio.apply to prevent it from actually modifying the loop policy
    # Only apply this broadly if it doesn't interfere with lower-level tests.
    # For now, let's assume it's okay or manage it per test suite.
    # mocker.patch('nest_asyncio.apply', MagicMock()) # Moved to be more specific if needed


# --- Tests for the OA URL Fallback Mechanism ---

@pytest.mark.asyncio
async def test_get_oa_url_unpaywall_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    mock_async_run_sync(mocker, search_client)
    
    doi = "10.1234/unpaywall_success"
    mock_df_data = {
        "doi": [doi],
        "is_oa": [True],
        "best_oa_location": [{"url": "http://unpaywall.example.com/pdf"}]
    }
    mocks["UnpywallGlobal"].doi.return_value = pd.DataFrame(mock_df_data)

    result = await search_client._get_open_access_url_with_fallback(doi)

    assert result["is_open_access"] is True
    assert result["open_access_url"] == "http://unpaywall.example.com/pdf"
    assert result["open_access_url_source"] == "Unpaywall"
    mocks["UnpywallGlobal"].doi.assert_called_once_with(dois=[doi])


@pytest.mark.asyncio
async def test_get_oa_url_pmc_fallback_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    mock_async_run_sync(mocker, search_client)
    MockEntrez = mocks["Entrez"]

    doi = "10.1234/pmc_success"
    
    # Unpaywall fails (returns no OA or error)
    mocks["UnpywallGlobal"].doi.return_value = pd.DataFrame({"doi": [doi], "is_oa": [False], "best_oa_location": [None]})
    
    # PMC search success
    mock_esearch_handle_pmc = MagicMock()
    MockEntrez.esearch.return_value = mock_esearch_handle_pmc
    
    mock_elink_handle_pmc = MagicMock()
    MockEntrez.elink.return_value = mock_elink_handle_pmc

    def entrez_read_side_effect_pmc(handle, *args, **kwargs):
        if handle == mock_esearch_handle_pmc:
            # Check db parameter if necessary: kwargs.get('db') == 'pmc'
            return {"IdList": ["PMC123"]}
        if handle == mock_elink_handle_pmc:
            return [{"LinkSetDb": [{"LinkName": "pmc_pmc_ft", "Link": [{"Url": "http://pmc.example.com/article/PMC123/pdf"}]}]}]
        raise ValueError("Unexpected handle for Entrez.read in PMC test")

    MockEntrez.read.side_effect = entrez_read_side_effect_pmc
    
    result = await search_client._get_open_access_url_with_fallback(doi)

    assert result["is_open_access"] is True
    assert result["open_access_url"] == "http://pmc.example.com/article/PMC123/pdf"
    assert result["open_access_url_source"] == "PMC"
    
    mocks["UnpywallGlobal"].doi.assert_called_once_with(dois=[doi])
    MockEntrez.esearch.assert_called_once_with(db="pmc", term=f"{doi}[DOI]", retmax="1")
    MockEntrez.elink.assert_called_once_with(dbfrom="pmc", db="pmc", id="PMC123", cmd="prlinks")


@pytest.mark.asyncio
async def test_get_oa_url_europepmc_fallback_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    mock_async_run_sync(mocker, search_client)
    MockEntrez = mocks["Entrez"]
    MockEuropePMCInstance = mocks["EuropePMCInstance"]

    doi = "10.1234/epmc_success"

    # Unpaywall fails
    mocks["UnpywallGlobal"].doi.return_value = pd.DataFrame({"doi": [doi], "is_oa": [False], "best_oa_location": [None]})
    
    # PMC fails (no ID found)
    mock_esearch_handle_pmc_fail = MagicMock()
    MockEntrez.esearch.return_value = mock_esearch_handle_pmc_fail
    MockEntrez.read.side_effect = lambda handle, *args, **kwargs: {"IdList": []} if handle == mock_esearch_handle_pmc_fail else {}
    
    # EuropePMC success
    epmc_api_response = {
        "pmcid": "PMC789",
        "doi": doi,
        "isOpenAccess": "Y",
        "hasTextMinedTerms": "Y",
        "fullTextUrlList": {
            "fullTextUrl": [
                {"documentStyle": "html", "availabilityCode": "OA", "url": "http://epmc.example.com/html"},
                {"documentStyle": "pdf", "availabilityCode": "OA", "url": "http://epmc.example.com/pdf"}
            ]
        },
        "_error": "" 
    }
    MockEuropePMCInstance.search.return_value = epmc_api_response
    
    result = await search_client._get_open_access_url_with_fallback(doi)

    assert result["is_open_access"] is True
    assert result["open_access_url"] == "http://epmc.example.com/pdf" # Prefers PDF
    assert result["open_access_url_source"] == "EuropePMC"

    mocks["UnpywallGlobal"].doi.assert_called_once_with(dois=[doi])
    MockEntrez.esearch.assert_called_once_with(db="pmc", term=f"{doi}[DOI]", retmax="1")
    MockEuropePMCInstance.search.assert_called_once_with(f"DOI:{doi}")


@pytest.mark.asyncio
async def test_get_oa_url_no_source_success(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    mock_async_run_sync(mocker, search_client)
    MockEntrez = mocks["Entrez"]
    MockEuropePMCInstance = mocks["EuropePMCInstance"]

    doi = "10.1234/all_fail"

    # Unpaywall fails
    mocks["UnpywallGlobal"].doi.return_value = pd.DataFrame({"doi": [doi], "is_oa": [False], "best_oa_location": [None]})
    
    # PMC fails
    mock_esearch_handle_pmc_all_fail = MagicMock()
    MockEntrez.esearch.return_value = mock_esearch_handle_pmc_all_fail
    MockEntrez.read.side_effect = lambda handle, *args, **kwargs: {"IdList": []} if handle == mock_esearch_handle_pmc_all_fail else {}
    
    # EuropePMC fails (no result or error)
    MockEuropePMCInstance.search.return_value = {"_error": "No results found"}
    
    result = await search_client._get_open_access_url_with_fallback(doi)

    assert result["is_open_access"] is False
    assert result["open_access_url"] is None
    assert result["open_access_url_source"] is None


@pytest.mark.asyncio
async def test_search_unpaywall_integrates_fallback(search_client_with_mocks, mocker):
    # This test ensures that the main search_unpaywall method correctly uses the fallback.
    search_client, mocks = search_client_with_mocks
    mock_async_run_sync(mocker, search_client)

    doi_unpaywall = "10.1/unpaywall_only"
    doi_pmc = "10.1/pmc_only"
    doi_epmc = "10.1/epmc_only"
    doi_none = "10.1/none"

    # Mock _get_open_access_url_with_fallback directly for simplicity here
    # rather than mocking all individual API calls again.
    async def mock_fallback_logic(doi_arg):
        if doi_arg == doi_unpaywall:
            return {"doi": doi_unpaywall, "is_open_access": True, "open_access_url": "http://u.pdf", "open_access_url_source": "Unpaywall"}
        elif doi_arg == doi_pmc:
            return {"doi": doi_pmc, "is_open_access": True, "open_access_url": "http://p.pdf", "open_access_url_source": "PMC"}
        elif doi_arg == doi_epmc:
            return {"doi": doi_epmc, "is_open_access": True, "open_access_url": "http://e.pdf", "open_access_url_source": "EuropePMC"}
        elif doi_arg == doi_none:
            return {"doi": doi_none, "is_open_access": False, "open_access_url": None, "open_access_url_source": None}
        return {} # Should not happen with given DOIs

    mocker.patch.object(search_client, '_get_open_access_url_with_fallback', side_effect=mock_fallback_logic)

    dois_to_search = [doi_unpaywall, doi_pmc, doi_epmc, doi_none]
    results_map = await search_client.search_unpaywall(dois_to_search)

    assert len(results_map) == 4
    assert results_map[doi_unpaywall]["open_access_url"] == "http://u.pdf"
    assert results_map[doi_unpaywall]["open_access_url_source"] == "Unpaywall"
    assert results_map[doi_pmc]["open_access_url"] == "http://p.pdf"
    assert results_map[doi_pmc]["open_access_url_source"] == "PMC"
    assert results_map[doi_epmc]["open_access_url"] == "http://e.pdf"
    assert results_map[doi_epmc]["open_access_url_source"] == "EuropePMC"
    assert results_map[doi_none]["is_open_access"] is False
    assert search_client._get_open_access_url_with_fallback.call_count == 4


# --- Tests for search_literature and overall functionality (may need adjustment due to mock_all_async_search_methods) ---
# For these higher-level tests, we might want mock_all_async_search_methods to be active.
# If it was `autouse=False`, we'd add `@pytest.mark.usefixtures("mock_all_async_search_methods")` here.
# For now, assuming it's still autouse=True or we manage its scope.
# If `mock_all_async_search_methods` is autouse=True, it will mock AsyncSearchClient.search_all,
# so the detailed OA fallback logic inside search_all (via search_unpaywall) won't be hit directly
# by these specific `test_search_literature_*` tests unless `search_all`'s mock itself incorporates it.
# The current `mock_search_all` in that fixture just returns canned data.
# To test the full integration including OA fallback in `search_literature`,
# `mock_all_async_search_methods` would need to be more sophisticated or disabled for such a test.

# Let's add a specific test for search_literature that ensures the new OA columns are present.
# This will rely on the `mock_all_async_search_methods` fixture's behavior.
# The fixture's `mock_search_all` returns a DataFrame from `mock_async_search_literature_data`.
# This data does not yet include 'open_access_url_source'.
# We need to update `mock_async_search_literature_data` or how `search_all` is mocked.

# For now, let's focus on the lower-level tests of the fallback.
# The existing `test_search_literature_returns_dataframe_with_expected_columns`
# will need `open_access_url_source` added to `expected_columns`.


def test_search_literature_returns_dataframe_with_expected_columns():
    # This test relies on mock_all_async_search_methods fixture.
    # We need to ensure nest_asyncio.apply is mocked if it's not autouse.
    with patch('nest_asyncio.apply', MagicMock()):
        df = search_literature("test query", max_results_per_source=1)
    
    expected_columns = ['title', 'authors', 'doi', 'pmid', 'pmcid', 'year', 'abstract', 'journal',
                        'url', 'source_api', 'is_open_access', 'open_access_url', 'open_access_url_source']
    assert isinstance(df, pd.DataFrame)
    # Check if all expected columns are present. Some might be all NA if not in mock data.
    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found in DataFrame."


@pytest.mark.usefixtures("mock_all_async_search_methods") # Explicitly use if autouse=False
def test_search_literature_deduplication_and_sorting():
    with patch('nest_asyncio.apply', MagicMock()):
        # The mock_all_async_search_methods fixture provides data with duplicates.
        # We expect 6 unique results after deduplication (A, B, C, D, E, F)
        # and sorted by source_api, then year desc.
        df = search_literature("test query", max_results_per_source=2) # Each source returns 2, total 8 raw
    
    # Expected unique DOIs: 10.1/A, 10.1/B, 10.1/C, 10.1/D
    # Expected unique PMIDs (for those without DOI): 5, 6
    # Total expected unique rows: 6
    assert len(df) == 6 

    # Verify sorting: source_api ascending, year descending
    # PubMed (A, E), EuropePMC (B), SemanticScholar (C), CrossRef (D, F)
    # Within PubMed: A (2020), E (2020) - order might be stable or by original index
    # Let's check the overall sort order
    
    # Create expected sorted order based on the mock_async_search_literature_data
    # and the deduplication/sorting logic in _async_search_literature
    expected_order_dois = ["10.1/A", "10.1/B", "10.1/C", "10.1/D"]
    expected_order_pmids_no_doi = ["5", "6"] # PMID 5 from PubMed, PMID 6 from CrossRef

    # Manually construct the expected DataFrame after deduplication and sorting
    # This is a bit brittle if the mock data changes, but necessary for strict sorting checks.
    # The mock_async_search_literature_data needs to be updated if we want to check new OA columns here.
    # For now, this test focuses on the original deduplication and sorting logic.
    expected_data_for_sort_test = [
        # These are from mock_async_search_literature_data, after deduplication
        SearchResult(title="Title D", authors=["Author D"], doi="10.1/D", pmid="4", year=2023, source_api="CrossRef"),
        SearchResult(title="Title F", authors=["Author F"], doi=None, pmid="6", year=2021, source_api="CrossRef"),
        SearchResult(title="Title B", authors=["Author B"], doi="10.1/B", pmid="2", year=2021, source_api="EuropePMC"),
        SearchResult(title="Title A", authors=["Author A"], doi="10.1/A", pmid="1", year=2020, source_api="PubMed"),
        SearchResult(title="Title E", authors=["Author E"], doi=None, pmid="5", year=2020, source_api="PubMed"),
        SearchResult(title="Title C", authors=["Author C"], doi="10.1/C", pmid="3", year=2022, source_api="SemanticScholar"),
    ]
    # Sort this expected data by source_api (asc), then year (desc)
    expected_df_sorted = pd.DataFrame(expected_data_for_sort_test).sort_values(
        by=['source_api', 'year'], ascending=[True, False]
    ).reset_index(drop=True)

    # Compare relevant columns, ignoring potential differences in 'is_open_access', 'open_access_url' etc.
    # as these are not the focus of this specific sorting/deduplication test using the old mock data.
    df_compare = df[['title', 'doi', 'pmid', 'year', 'source_api']].copy().reset_index(drop=True)
    expected_df_compare = expected_df_sorted[['title', 'doi', 'pmid', 'year', 'source_api']].copy().reset_index(drop=True)
    
    # Fill NA for comparison as string 'None' might differ from np.nan
    df_compare.fillna(value=pd.NA, inplace=True)
    expected_df_compare.fillna(value=pd.NA, inplace=True)

    pd.testing.assert_frame_equal(df_compare, expected_df_compare, check_dtype=False)


@pytest.mark.usefixtures("mock_all_async_search_methods")
def test_search_literature_max_results_per_source_limit():
    with patch('nest_asyncio.apply', MagicMock()):
        # With max_results_per_source=1, each of the 4 sources should return 1 unique result.
        # The mock_async_search_literature_data has 4 unique DOIs and 2 unique PMIDs (no DOI).
        # If max_results_per_source is 1, search_all will receive 4 results (1 from each source).
        # After deduplication, we should still have 4 results.
        df = search_literature("test query", max_results_per_source=1)
    assert len(df) == 4 # 4 unique results from 4 sources, 1 per source

@pytest.mark.slow
@pytest.mark.skip(reason="This test hits live network APIs and is slow. Run manually if needed.")
def test_search_literature_live_network():
    # This test requires actual API keys and network access.
    # It should only be run manually or in specific CI environments.
    # Ensure your .env file is correctly configured for this.
    query = "CRISPR gene editing"
    print(f"\nRunning LIVE network search for: {query}")
    df = search_literature(query, max_results_per_source=2)
    print(f"Live search results (first 5 rows):\n{df.head()}")
    assert not df.empty
    assert len(df.columns) >= 10 # Check for a reasonable number of columns
    assert 'title' in df.columns
    assert 'source_api' in df.columns
    assert 'doi' in df.columns or 'pmid' in df.columns
