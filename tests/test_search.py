import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np # Added numpy import
from typing import Optional, List, Dict, Any # Added Optional, List, Dict, Any
from tools.search import AsyncSearchClient, search_literature, SearchResult, SearchLiteratureParams, SearchOutput, Works # Ensure Works is imported for type hints if needed
import httpx # Ensure httpx is imported for type hints if needed for mocks

# Mock environment variables for tests
MOCKED_API_EMAIL = "test@example.com"
MOCKED_PUBMED_KEY = "test_pubmed_key"
MOCKED_S2_KEY = "test_s2_key"

@pytest_asyncio.fixture
async def mock_publication_type_mappings():
    return {
        "research": {
            "pubmed": "Journal Article[Publication Type]",
            "europepmc": "PUB_TYPE:\"journal-article\"",
            "semanticscholar": "JournalArticle",
            "crossref": "journal-article",
            "openalex": "journal-article" 
        },
        "review": {
            "pubmed": "Review[Publication Type]",
            "europepmc": "PUB_TYPE:\"review\"",
            "semanticscholar": "Review",
            "crossref": "review-article",
            "openalex": "review-article"
        }
    }

@pytest_asyncio.fixture
async def search_client_with_mocks(mocker, mock_publication_type_mappings): # Added mocker & mappings
    with patch('tools.search.load_dotenv', MagicMock()), \
         patch('tools.search.API_EMAIL', MOCKED_API_EMAIL), \
         patch('tools.search.PUBMED_API_KEY', MOCKED_PUBMED_KEY), \
         patch('tools.search.SEMANTIC_SCHOLAR_API_KEY', MOCKED_S2_KEY), \
         patch('tools.search.CROSSREF_MAILTO', MOCKED_API_EMAIL), \
         patch('Bio.Entrez.esearch', new_callable=AsyncMock) as mock_entrez_esearch, \
         patch('Bio.Entrez.efetch', new_callable=AsyncMock) as mock_entrez_efetch, \
         patch('Bio.Entrez.read', new_callable=MagicMock) as mock_entrez_read:
        
        client = AsyncSearchClient(publication_type_mappings=mock_publication_type_mappings)
        
        # Mock the client's internal instances' methods
        if hasattr(client, 'epmc'):
            client.epmc.search = AsyncMock() 
        if hasattr(client, 's2'):
            client.s2.search_paper = AsyncMock() # This is an instance method
        if hasattr(client, 'cr') and client.cr is not None:
            mock_cr_query_obj = MagicMock()
            mock_cr_query_obj.filter = MagicMock(return_value=mock_cr_query_obj) 
            mock_cr_query_obj.__iter__ = MagicMock(return_value=iter([])) 
            mock_cr_query_obj.sample = MagicMock(return_value=[]) 
            client.cr.query = MagicMock(return_value=mock_cr_query_obj) # This is an instance method
        
        if hasattr(client, 'client'): # This is the httpx.AsyncClient
             client.client.get = AsyncMock() 

        mocks_dict = { 
            "Entrez_esearch": mock_entrez_esearch, 
            "Entrez_efetch": mock_entrez_efetch,
            "Entrez_read": mock_entrez_read,
            "EuropePMC_search": client.epmc.search if hasattr(client, 'epmc') else None,
            "S2_search_paper": client.s2.search_paper if hasattr(client, 's2') else None,
            "CrossRef_query": client.cr.query if hasattr(client, 'cr') and client.cr else None,
            "OpenAlex_get": client.client.get if hasattr(client, 'client') else None
        }
        yield client, mocks_dict
        await client.close()

def mock_async_run_sync(mocker, search_client_instance):
    async def side_effect_func(sync_func, *args, **kwargs):
        # If func is already a coroutine function, await it. Otherwise, run it.
        if asyncio.iscoroutinefunction(sync_func):
            return await sync_func(*args, **kwargs)
        else:
            return sync_func(*args, **kwargs) # Assuming it's a plain sync function
    
    mocker.patch.object(search_client_instance, '_run_sync_in_thread', AsyncMock(side_effect=side_effect_func))


# --- Tests for individual AsyncSearchClient methods ---

@pytest.mark.asyncio
async def test_search_pubmed_success_and_filters(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    mock_pmids = ["123"] 
    mock_esearch_handle, mock_efetch_handle = MagicMock(), MagicMock()

    class MockEntrezId: # Helper class for mocking Entrez ID objects
        def __init__(self, text, attributes): self._text = text; self.attributes = attributes
        def __str__(self): return self._text
            
    mock_efetch_xml_result = {
        'PubmedArticle': [{'MedlineCitation': {'PMID': '123', 'Article': {'ArticleTitle': 'Title 1', 
        'Abstract': {'AbstractText': ['Abstract 1']}, 'AuthorList': [{'LastName': 'Author', 'ForeName': 'A'}],
        'Journal': {'Title': 'Journal 1', 'JournalIssue': {'PubDate': {'Year': '2023'}}},
        'ELocationID': [MockEntrezId(text='10.test/doi1', attributes={'EIdType': 'doi'})]}},
        'PubmedData': {'ArticleIdList': [MockEntrezId(text='PMC123', attributes={'IdType': 'pmc'})]}}]}

    mocks["Entrez_esearch"].return_value = mock_esearch_handle
    mocks["Entrez_efetch"].return_value = mock_efetch_handle
    
    def entrez_read_side_effect(handle, *args, **kwargs):
        if handle == mock_esearch_handle: return {"IdList": mock_pmids, "QueryKey": "1", "WebEnv": "abc"}
        elif handle == mock_efetch_handle: return mock_efetch_xml_result
        raise ValueError(f"Unexpected handle for Entrez.read: {handle}")
    mocks["Entrez_read"].side_effect = entrez_read_side_effect
    
    mock_async_run_sync(mocker, search_client)

    results = await search_client.search_pubmed("test query", max_results=1)
    assert len(results) == 1; assert results[0]["pmid"] == "123"
    mocks["Entrez_esearch"].assert_called_with(db="pubmed", term="test query", retmax='1', usehistory='y')
    
    mocks["Entrez_esearch"].reset_mock()
    await search_client.search_pubmed("cancer", max_results=1, publication_types=["research"])
    mocks["Entrez_esearch"].assert_called_with(db="pubmed", term="(cancer) AND (Journal Article[Publication Type])", retmax='1', usehistory='y')

    mocks["Entrez_esearch"].reset_mock()
    await search_client.search_pubmed("covid", max_results=1, publication_types=["review"])
    mocks["Entrez_esearch"].assert_called_with(db="pubmed", term="(covid) AND (Review[Publication Type])", retmax='1', usehistory='y')

    mocks["Entrez_esearch"].reset_mock()
    await search_client.search_pubmed("flu", max_results=1, publication_types=["research", "review"])
    expected_term = "(flu) AND (Journal Article[Publication Type] OR Review[Publication Type])"
    mocks["Entrez_esearch"].assert_called_with(db="pubmed", term=expected_term, retmax='1', usehistory='y')

    mocks["Entrez_esearch"].reset_mock()
    await search_client.search_pubmed("heart", max_results=1, publication_types=[])
    mocks["Entrez_esearch"].assert_called_with(db="pubmed", term="heart", retmax='1', usehistory='y')

    mocks["Entrez_esearch"].reset_mock()
    await search_client.search_pubmed("kidney", max_results=1, publication_types=None)
    mocks["Entrez_esearch"].assert_called_with(db="pubmed", term="kidney", retmax='1', usehistory='y')

@pytest.mark.asyncio
async def test_search_europepmc_success_and_filters(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    europe_pmc_search_mock = mocks["EuropePMC_search"]
    if not europe_pmc_search_mock: pytest.skip("EuropePMC client not mocked")

    mock_record = MagicMock(doi="10.test/epmc", pmid="epmc123", title="EuropePMC Test", 
                            authorList=[MagicMock(fullName="Author EPMC")], 
                            journalTitle="EPMC Journal", pubYear="2023", 
                            abstractText="Abstract here.", pmcid="PMC12345")
    europe_pmc_search_mock.return_value = [mock_record]
    
    mock_async_run_sync(mocker, search_client)

    results = await search_client.search_europepmc("test query", max_results=1)
    assert len(results) == 1; assert results[0]["doi"] == "10.test/epmc"
    europe_pmc_search_mock.assert_called_with("test query")

    europe_pmc_search_mock.reset_mock()
    await search_client.search_europepmc("cancer", max_results=1, publication_types=["research"])
    europe_pmc_search_mock.assert_called_with('(cancer) AND (PUB_TYPE:"journal-article")')

    europe_pmc_search_mock.reset_mock()
    await search_client.search_europepmc("flu", max_results=1, publication_types=["research", "review"])
    expected_query = '(flu) AND (PUB_TYPE:"journal-article" OR PUB_TYPE:"review")'
    europe_pmc_search_mock.assert_called_with(expected_query)

@pytest.mark.asyncio
async def test_search_semanticscholar_success_and_filters(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    s2_search_paper_mock = mocks["S2_search_paper"]
    if not s2_search_paper_mock: pytest.skip("Semantic Scholar client not mocked")

    mock_paper_dict = {'title': "S2 Title", 'authors': [{'name': 'S2 Author'}], 'year': 2023,
                       'journal': {'name': 'S2 Journal'}, 'externalIds': {'DOI': '10.test/s2'},
                       'abstract': "S2 abstract.", 'url': "http://s2.example.com", 'citationCount': 10,
                       'publicationTypes': ['JournalArticle']}
    MockPaper = type('MockPaper', (object,), mock_paper_dict)
    
    s2_search_paper_mock.return_value = iter([MockPaper()])
    mock_async_run_sync(mocker, search_client)
    s2_expected_fields = ['title', 'authors', 'year', 'journal', 'abstract', 'url', 'venue', 
                          'publicationDate', 'externalIds', 'citationCount', 'publicationTypes']

    results = await search_client.search_semanticscholar("test query", max_results=1)
    assert len(results) == 1; assert results[0]["title"] == "S2 Title"
    s2_search_paper_mock.assert_called_with(query="test query", limit=1, fields=s2_expected_fields, publication_types=None)

    s2_search_paper_mock.reset_mock()
    await search_client.search_semanticscholar("cancer", max_results=1, publication_types=["research"])
    s2_search_paper_mock.assert_called_with(query="cancer", limit=1, fields=s2_expected_fields, publication_types=['JournalArticle'])

    s2_search_paper_mock.reset_mock()
    await search_client.search_semanticscholar("flu", max_results=1, publication_types=["research", "review"])
    s2_search_paper_mock.assert_called_with(query="flu", limit=1, fields=s2_expected_fields, publication_types=['JournalArticle', 'Review'])

@pytest.mark.asyncio
async def test_search_crossref_success_and_filters(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    crossref_query_mock = mocks["CrossRef_query"]
    if not crossref_query_mock: pytest.skip("CrossRef client not mocked or not available")

    mock_item = {"DOI": "10.test/cr", "title": ["CrossRef Title"], "author": [{"given": "CR", "family": "Author"}],
                 "issued": {"date-parts": [[2023]]}, "container-title": ["CR Journal"], "URL": "http://cr.example.com",
                 "is-referenced-by-count": 5}
    
    # Setup mock chain for CrossRef
    mock_cr_iterable = MagicMock(); mock_cr_iterable.__iter__.return_value = iter([mock_item])
    mock_filtered_query = MagicMock(); mock_filtered_query.__iter__.return_value = iter([mock_item])
    mock_initial_query = MagicMock()
    mock_initial_query.filter = MagicMock(return_value=mock_filtered_query)
    mock_initial_query.__iter__.return_value = iter([mock_item]) # For calls without .filter()
    crossref_query_mock.return_value = mock_initial_query
    
    mock_async_run_sync(mocker, search_client)
    
    results = await search_client.search_crossref("test query", max_results=1)
    assert len(results) == 1; assert results[0]["doi"] == "10.test/cr"
    crossref_query_mock.assert_called_with(bibliographic="test query")
    assert not mock_initial_query.filter.called 

    mock_initial_query.filter.reset_mock()
    await search_client.search_crossref("cancer", max_results=1, publication_types=["research"])
    crossref_query_mock.assert_called_with(bibliographic="cancer")
    mock_initial_query.filter.assert_called_with(type="journal-article")

    mock_initial_query.filter.reset_mock()
    await search_client.search_crossref("flu", max_results=1, publication_types=["research", "review"])
    crossref_query_mock.assert_called_with(bibliographic="flu")
    mock_initial_query.filter.assert_called_with(type="journal-article,review-article")

@pytest.mark.asyncio
async def test_search_openalex_success_and_filters(search_client_with_mocks, mocker):
    search_client, mocks = search_client_with_mocks
    openalex_get_mock = mocks["OpenAlex_get"]
    if not openalex_get_mock: pytest.skip("OpenAlex client (httpx) not mocked")
    
    mock_openalex_item = {"doi": "https://doi.org/10.test/oa", "title": "OpenAlex Title", 
                          "authorships": [{"author": {"display_name": "OA Author"}}],
                          "publication_year": 2023, "host_venue": {"display_name": "OA Journal"},
                          "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/oapmid123", "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMCABC/"},
                          "cited_by_count": 20, "oa_status": "gold", "type": "journal-article"}
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": [mock_openalex_item]}
    openalex_get_mock.return_value = mock_response

    results = await search_client.search_openalex("test query", max_results=1)
    assert len(results) == 1; assert results[0]["doi"] == "10.test/oa"
    openalex_get_mock.assert_called_with(
        "https://api.openalex.org/works", 
        params={"mailto": MOCKED_API_EMAIL, "per-page": "1", "filter": "default.search:test query"}
    )

    openalex_get_mock.reset_mock()
    await search_client.search_openalex("cancer", max_results=1, publication_types=["research"])
    openalex_get_mock.assert_called_with(
        "https://api.openalex.org/works",
        params={"mailto": MOCKED_API_EMAIL, "per-page": "1", "filter": "default.search:cancer,type:journal-article"}
    )

    openalex_get_mock.reset_mock()
    await search_client.search_openalex("flu", max_results=1, publication_types=["research", "review"])
    openalex_get_mock.assert_called_with(
        "https://api.openalex.org/works",
        params={"mailto": MOCKED_API_EMAIL, "per-page": "1", "filter": "default.search:flu,type:journal-article|review-article"}
    )

# --- Tests for search_literature and overall functionality ---

@pytest.fixture
def mock_async_search_literature_data():
    # Canned data for _async_search_literature
    return [
        SearchResult(title="Title A", authors=["Author A"], doi="10.1/A", pmid="1", year=2020, abstract="Abstract A", source_api="PubMed"),
        SearchResult(title="Title B", authors=["Author B"], doi="10.1/B", pmid="2", year=2021, abstract="Abstract B", source_api="EuropePMC"),
        SearchResult(title="Title C", authors=["Author C"], doi="10.1/C", pmid="3", year=2022, abstract="Abstract C", source_api="SemanticScholar"),
        SearchResult(title="Title D", authors=["Author D"], doi="10.1/D", pmid="4", year=2023, abstract="Abstract D", source_api="CrossRef"),
        # Duplicates to test deduplication
        SearchResult(title="Title A Duplicate", authors=["Author A"], doi="10.1/A", pmid="10", year=2019, abstract="Abstract A Dup", source_api="EuropePMC"),
        SearchResult(title="Title E", authors=["Author E"], doi=None, pmid="5", year=2020, abstract="Abstract E", source_api="PubMed"),
        SearchResult(title="Title E Duplicate", authors=["Author E"], doi=None, pmid="5", year=2018, abstract="Abstract E Dup", source_api="SemanticScholar"),
        SearchResult(title="Title F", authors=["Author F"], doi=None, pmid="6", year=2021, abstract="Abstract F", source_api="CrossRef"),
    ]

@pytest.fixture(autouse=True) # Keep autouse if it's meant to apply to all tests in module
def mock_search_all_for_search_literature_tests(mocker, mock_async_search_literature_data):
    # This fixture specifically mocks AsyncSearchClient.search_all for tests on search_literature
    # It ensures that search_literature gets a consistent, pre-processed DataFrame.
    
    df = pd.DataFrame(mock_async_search_literature_data)
    # Apply the same deduplication and sorting logic as in AsyncSearchClient.search_all
    if 'doi' in df.columns:
        df['doi_norm'] = df['doi'].apply(lambda x: str(x).lower().replace("https://doi.org/", "").strip() if pd.notna(x) and x != '' else pd.NA)
    else: df['doi_norm'] = pd.NA
    if 'pmid' in df.columns:
        df['pmid_str'] = df['pmid'].apply(lambda x: str(x) if pd.notna(x) and x != '' else pd.NA)
    else: df['pmid_str'] = pd.NA

    mask_doi_present = df['doi_norm'].notna()
    df_with_doi_present = df[mask_doi_present].copy()
    df_no_doi_present = df[~mask_doi_present].copy()
    
    df_with_doi_deduped = pd.DataFrame()
    if not df_with_doi_present.empty:
        df_with_doi_present['has_abstract'] = df_with_doi_present['abstract'].notna() & (df_with_doi_present['abstract'] != '')
        df_with_doi_present = df_with_doi_present.sort_values(by=['doi_norm', 'has_abstract', 'year', 'pmid_str'], ascending=[True, False, False, True], na_position='last')
        df_with_doi_deduped = df_with_doi_present.drop_duplicates(subset=['doi_norm'], keep='first')
        df_with_doi_deduped = df_with_doi_deduped.drop(columns=['has_abstract'], errors='ignore')

    df_no_doi_deduped = pd.DataFrame()
    if not df_no_doi_present.empty:
        df_no_doi_present['has_abstract'] = df_no_doi_present['abstract'].notna() & (df_no_doi_present['abstract'] != '')
        df_no_doi_present = df_no_doi_present.sort_values(by=['title', 'has_abstract', 'year', 'pmid_str'], ascending=[True, False, False, True], na_position='last')
        use_pmid_for_no_doi_dedup = 'pmid_str' in df_no_doi_present.columns and df_no_doi_present['pmid_str'].notna().any()
        subset_for_no_doi = ['pmid_str'] if use_pmid_for_no_doi_dedup else ['title', 'year']
        df_no_doi_deduped = df_no_doi_present.drop_duplicates(subset=subset_for_no_doi, keep='first')
        df_no_doi_deduped = df_no_doi_deduped.drop(columns=['has_abstract'], errors='ignore')
    
    final_df_mock = pd.concat([df_with_doi_deduped, df_no_doi_deduped], ignore_index=True)
    cols_to_drop_mock = [col for col in ['doi_norm', 'pmid_str', 'has_abstract'] if col in final_df_mock.columns]
    if cols_to_drop_mock: final_df_mock.drop(columns=cols_to_drop_mock, inplace=True)
    
    expected_df_cols_mock = list(SearchResult.__annotations__.keys())
    for col in expected_df_cols_mock:
        if col not in final_df_mock.columns: final_df_mock[col] = pd.NA
    final_df_mock = final_df_mock[expected_df_cols_mock]
    final_df_mock = final_df_mock.sort_values(by=['year', 'title'], ascending=[False, True], na_position='last').reset_index(drop=True)

    # This is the method that will be the side_effect for the mock
    async def mock_search_all_method(self_client, pubmed_query: str, general_query: str, 
                                     max_results_per_source: int = 20, 
                                     publication_types: Optional[List[str]] = None): # Added self_client and publication_types
        # This mock doesn't use publication_types for filtering, it returns a fixed dataset.
        # This is suitable for testing deduplication and output formatting of search_literature.
        return final_df_mock.copy()

    mocker.patch('tools.search.AsyncSearchClient.search_all', new_callable=AsyncMock, side_effect=mock_search_all_method)
    mocker.patch('nest_asyncio.apply', MagicMock())


def test_search_literature_returns_structured_output_with_expected_fields():
    output: SearchOutput = search_literature(
        pubmed_query="test pubmed", 
        general_query="test general", 
        max_results_per_source=1
        # publication_types and publication_type_mappings will be None by default
    )
    
    assert isinstance(output, dict); assert "data" in output; assert "meta" in output
    assert isinstance(output["data"], list); assert isinstance(output["meta"], dict)
    assert "total_hits" in output["meta"]; assert "query_pubmed" in output["meta"]
    assert "query_general" in output["meta"]; assert "timestamp" in output["meta"]
    assert "publication_types_applied" in output["meta"] # New field
    assert output["meta"]["query_pubmed"] == "test pubmed"
    assert output["meta"]["query_general"] == "test general"
    assert isinstance(output["meta"]["total_hits"], int)
    assert isinstance(output["meta"]["timestamp"], str)
    assert output["meta"]["publication_types_applied"] == [] # Default

    expected_data_fields = list(SearchResult.__annotations__.keys())
    for record in output["data"]:
        assert isinstance(record, dict)
        for field in expected_data_fields:
            assert field in record, f"Expected field '{field}' not found: {record}"
        assert all(key in expected_data_fields for key in record.keys()), f"Unexpected keys: {record.keys()}"

@pytest.mark.usefixtures("mock_search_all_for_search_literature_tests")
def test_search_literature_deduplication_and_sorting(mock_async_search_literature_data):
    output: SearchOutput = search_literature(
        pubmed_query="test pubmed", 
        general_query="test general", 
        max_results_per_source=10
    )
    data_list = output["data"]
    
    # Re-create expected list from mock_async_search_literature_data after deduplication/sorting
    # (This logic is duplicated from mock_search_all_for_search_literature_tests for clarity)
    df_for_test = pd.DataFrame(mock_async_search_literature_data)
    if 'doi' in df_for_test.columns: df_for_test['doi_norm'] = df_for_test['doi'].apply(lambda x: str(x).lower().replace("https://doi.org/", "").strip() if pd.notna(x) and x != '' else pd.NA)
    else: df_for_test['doi_norm'] = pd.NA
    if 'pmid' in df_for_test.columns: df_for_test['pmid_str'] = df_for_test['pmid'].apply(lambda x: str(x) if pd.notna(x) and x != '' else pd.NA)
    else: df_for_test['pmid_str'] = pd.NA
    mask_doi_present_test = df_for_test['doi_norm'].notna()
    df_with_doi_present_test = df_for_test[mask_doi_present_test].copy()
    df_no_doi_present_test = df_for_test[~mask_doi_present_test].copy()
    df_with_doi_deduped_test = pd.DataFrame()
    if not df_with_doi_present_test.empty:
        df_with_doi_present_test['has_abstract'] = df_with_doi_present_test['abstract'].notna() & (df_with_doi_present_test['abstract'] != '')
        df_with_doi_present_test = df_with_doi_present_test.sort_values(by=['doi_norm', 'has_abstract', 'year', 'pmid_str'], ascending=[True, False, False, True], na_position='last')
        df_with_doi_deduped_test = df_with_doi_present_test.drop_duplicates(subset=['doi_norm'], keep='first')
        df_with_doi_deduped_test = df_with_doi_deduped_test.drop(columns=['has_abstract'], errors='ignore')
    df_no_doi_deduped_test = pd.DataFrame()
    if not df_no_doi_present_test.empty:
        df_no_doi_present_test['has_abstract'] = df_no_doi_present_test['abstract'].notna() & (df_no_doi_present_test['abstract'] != '')
        df_no_doi_present_test = df_no_doi_present_test.sort_values(by=['title', 'has_abstract', 'year', 'pmid_str'], ascending=[True, False, False, True], na_position='last')
        use_pmid_for_no_doi_dedup_test = 'pmid_str' in df_no_doi_present_test.columns and df_no_doi_present_test['pmid_str'].notna().any()
        subset_for_no_doi_test = ['pmid_str'] if use_pmid_for_no_doi_dedup_test else ['title', 'year']
        df_no_doi_deduped_test = df_no_doi_present_test.drop_duplicates(subset=subset_for_no_doi_test, keep='first')
        df_no_doi_deduped_test = df_no_doi_deduped_test.drop(columns=['has_abstract'], errors='ignore')
    final_df_expected = pd.concat([df_with_doi_deduped_test, df_no_doi_deduped_test], ignore_index=True)
    cols_to_drop_expected = [col for col in ['doi_norm', 'pmid_str', 'has_abstract'] if col in final_df_expected.columns]
    if cols_to_drop_expected: final_df_expected.drop(columns=cols_to_drop_expected, inplace=True)
    expected_df_cols_final = list(SearchResult.__annotations__.keys())
    for col in expected_df_cols_final:
        if col not in final_df_expected.columns: final_df_expected[col] = pd.NA
    final_df_expected = final_df_expected[expected_df_cols_final]
    final_df_expected = final_df_expected.sort_values(by=['year', 'title'], ascending=[False, True], na_position='last').reset_index(drop=True)
    expected_data_list_full = []
    for _, row in final_df_expected.iterrows():
        record = {}
        for key_col in expected_df_cols_final:
            val = row.get(key_col)
            if pd.isna(val): record[key_col] = None
            elif isinstance(val, list): record[key_col] = list(val)
            elif isinstance(val, (np.integer, np.floating)): record[key_col] = val.item()
            else: record[key_col] = val
        expected_data_list_full.append(record)
    
    assert len(data_list) == len(expected_data_list_full)
    assert data_list == expected_data_list_full

@pytest.mark.usefixtures("mock_search_all_for_search_literature_tests")
def test_search_literature_max_results_per_source_limit():
    # This test uses the mock_search_all_for_search_literature_tests which returns a fixed dataset
    # regardless of max_results_per_source. The number of unique items in that dataset is 6.
    output: SearchOutput = search_literature(
        pubmed_query="test pubmed", 
        general_query="test general", 
        max_results_per_source=1 # This param won't affect the mocked output count here
    )
    assert output["meta"]["total_hits"] == 6 
    assert len(output["data"]) == 6

@pytest.mark.slow
@pytest.mark.skip(reason="This test hits live network APIs and is slow. Run manually if needed.")
def test_search_literature_live_network():
    query = "CRISPR gene editing"
    print(f"\nRunning LIVE network search for: {query}")
    # Pass None for mappings, as live search_literature doesn't inherently load them
    # unless the tool wrapper does.
    results_output = search_literature(
        pubmed_query=f"{query}[Mesh]", 
        general_query=query, 
        max_results_per_source=2,
        publication_types=None, # Or specify types for live test
        publication_type_mappings=None
    )
    print(f"Live search results (first 5 rows of data):\n{pd.DataFrame(results_output['data']).head()}")
    assert results_output['data'] # Check if data is not empty
    df = pd.DataFrame(results_output['data'])
    assert not df.empty
    assert len(df.columns) >= 10 # Check based on SearchResult fields
    assert 'title' in df.columns
    assert 'source_api' in df.columns
    assert 'doi' in df.columns or 'pmid' in df.columns
