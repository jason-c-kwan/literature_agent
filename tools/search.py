import asyncio
import os
import time
from typing import List, Dict, Any, Optional, TypedDict, Type
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from Bio import Entrez
from europe_pmc import EuropePMC
from semanticscholar import SemanticScholar
from crossref.restful import Etiquette
from autogen_core.tools import BaseTool
from autogen_core._component_config import Component
from pydantic import BaseModel, Field

# Attempt to import from crossrefapi, with fallbacks for robustness
_Works_imported = False
Works = None

try:
    from crossref.restful import Works as CrWorks
    Works = CrWorks
    _Works_imported = True
except ImportError as e:
    print(f"Warning: Failed to import Works from crossref.restful ({e}). CrossRef search will be disabled.")
except AttributeError as e:
    print(f"Warning: AttributeError during import of crossref.restful ({e}). CrossRef search may be disabled or unstable.")
    Works = None
    _Works_imported = False

# Load environment variables from .env file
load_dotenv()

API_EMAIL = os.getenv("API_EMAIL")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO")

if CROSSREF_MAILTO == "$API_EMAIL" or CROSSREF_MAILTO is None:
    CROSSREF_MAILTO = API_EMAIL


class SearchResult(TypedDict, total=True):
    title: Optional[str]
    authors: Optional[List[str]]
    doi: Optional[str]
    pmid: Optional[str]
    pmcid: Optional[str]
    year: Optional[int]
    abstract: Optional[str]
    journal: Optional[str]
    url: Optional[str]
    source_api: str
    # Fields for TriageAgent
    sjr_percentile: Optional[float]
    oa_status: Optional[str]
    citation_count: Optional[int]


class AsyncSearchClient:
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.email = API_EMAIL
        self.pubmed_api_key = PUBMED_API_KEY
        self.semantic_scholar_api_key = SEMANTIC_SCHOLAR_API_KEY
        self.crossref_mailto = CROSSREF_MAILTO

        if not self.email:
            raise ValueError("API_EMAIL must be set in the .env file.")

        # Semaphores for rate limiting
        pubmed_rps = 2 if self.pubmed_api_key else 1
        self.pubmed_semaphore = asyncio.Semaphore(pubmed_rps)
        self.europepmc_semaphore = asyncio.Semaphore(5)
        self.semanticscholar_semaphore = asyncio.Semaphore(1)
        self.crossref_semaphore = asyncio.Semaphore(50)
        
        Entrez.email = self.email
        if self.pubmed_api_key:
            Entrez.api_key = self.pubmed_api_key

        self.epmc = EuropePMC()
        self.s2 = SemanticScholar(api_key=self.semantic_scholar_api_key if self.semantic_scholar_api_key else None, timeout=20)
        
        self.cr = None
        if _Works_imported and Works is not None:
            try:
                etiquette_obj = Etiquette(contact_email=self.crossref_mailto)
                self.cr = Works(etiquette=etiquette_obj)
            except Exception as e_init:
                print(f"Warning: Failed to instantiate CrossRef Works client ({e_init}). CrossRef search disabled.")
                self.cr = None
        else:
            print("CrossRef Works class not imported. CrossRef search will be disabled.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.client.aclose()

    async def _run_sync_in_thread(self, func, *args, **kwargs):
        """Helper to run synchronous functions in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def search_pubmed(self, query: str, max_results: int = 20) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        async with self.pubmed_semaphore:
            try:
                handle = await self._run_sync_in_thread(
                    Entrez.esearch,
                    db="pubmed",
                    term=query,
                    retmax=str(max_results),
                    usehistory="y"
                )
                search_results_handle = Entrez.read(handle) # Renamed for clarity
                handle.close()
                ids = search_results_handle["IdList"]

                if not ids:
                    return []

                # Fetch full records to get abstracts
                fetch_handle = await self._run_sync_in_thread(
                    Entrez.efetch,
                    db="pubmed",
                    id=ids,
                    rettype="abstract", # Changed from esummary
                    retmode="xml"      # XML is easier to parse for abstracts
                )
                articles_xml = Entrez.read(fetch_handle)
                fetch_handle.close()

                for article_xml in articles_xml.get('PubmedArticle', []):
                    medline_citation = article_xml.get('MedlineCitation', {})
                    article_info = medline_citation.get('Article', {})
                    
                    pmid = str(medline_citation.get('PMID', ''))
                    title = article_info.get('ArticleTitle', 'N/A')
                    
                    abstract_parts = article_info.get('Abstract', {}).get('AbstractText', [])
                    abstract = ""
                    if isinstance(abstract_parts, list):
                        abstract = "\n".join([part for part in abstract_parts if isinstance(part, str)])
                    elif isinstance(abstract_parts, str): # Sometimes it's just a string
                        abstract = abstract_parts
                    
                    authors_list = article_info.get('AuthorList', [])
                    authors = []
                    if isinstance(authors_list, list):
                        for author_entry in authors_list:
                            if isinstance(author_entry, dict):
                                last_name = author_entry.get('LastName', '')
                                fore_name = author_entry.get('ForeName', '')
                                if last_name and fore_name:
                                    authors.append(f"{fore_name} {last_name}")
                                elif last_name:
                                    authors.append(last_name)

                    journal_info = article_info.get('Journal', {})
                    journal_title = journal_info.get('Title', '')
                    pub_date_info = journal_info.get('JournalIssue', {}).get('PubDate', {})
                    year = pub_date_info.get('Year')
                    if year and isinstance(year, str) and year.isdigit():
                        year = int(year)
                    else: # Fallback if Year is not directly available or not a string
                        medline_date = pub_date_info.get('MedlineDate', '') # e.g., "2023" or "2023 Spring"
                        if isinstance(medline_date, str) and len(medline_date) >= 4 and medline_date[:4].isdigit():
                            year = int(medline_date[:4])
                        else:
                            year = None

                    doi = None
                    pmcid = None
                    article_ids_list = medline_citation.get('Article', {}).get('ELocationID', []) # ELocationID can be a list
                    if not isinstance(article_ids_list, list): # Handle cases where it might not be a list
                        article_ids_list = [article_ids_list] if article_ids_list else []

                    for aid in article_ids_list:
                        if hasattr(aid, 'attributes') and aid.attributes.get('EIdType') == 'doi':
                            doi = str(aid)
                        # PMCID might be elsewhere or not consistently in ELocationID
                    
                    # Attempt to get DOI and PMCID from PubmedData/ArticleIdList as fallback
                    pubmed_data = article_xml.get('PubmedData', {})
                    if pubmed_data:
                        id_list = pubmed_data.get('ArticleIdList', [])
                        for article_id_obj in id_list:
                            if hasattr(article_id_obj, 'attributes'):
                                id_type = article_id_obj.attributes.get('IdType')
                                if id_type == 'doi' and not doi:
                                    doi = str(article_id_obj)
                                elif id_type == 'pmc':
                                    pmcid = str(article_id_obj)


                    result_item: SearchResult = {
                        "title": title,
                        "authors": authors,
                        "doi": doi,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "year": year,
                        "abstract": abstract if abstract else None,
                        "journal": journal_title,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{doi}" if doi else None),
                        "source_api": "PubMed",
                        "sjr_percentile": None, # Not available from this PubMed fetch
                        "oa_status": None,      # Not available from this PubMed fetch
                        "citation_count": None  # Not available from this PubMed fetch
                    }
                    results.append(result_item)
            except Exception as e:
                print(f"PubMed search error: {e}")
        end_time = time.time() # This should be outside the async with block if it measures the whole thing
        print(f"PubMed search took {end_time - start_time:.2f} seconds")
        return results

    async def search_europepmc(self, query: str, max_results: int = 20) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        async with self.europepmc_semaphore:
            try:
                raw_results = await self._run_sync_in_thread(
                    self.epmc.search,
                    query
                )
                
                for item in raw_results:
                    doi = getattr(item, 'doi', None)
                    pmid = getattr(item, 'pmid', None)
                    title = getattr(item, 'title', None)
                    authors = [author.fullName for author in getattr(item, 'authorList', []) if hasattr(author, 'fullName')] if getattr(item, 'authorList', None) else []
                    journal = getattr(item, 'journalTitle', None)
                    year_str = getattr(item, 'pubYear', None)
                    year = int(year_str) if year_str and year_str.isdigit() else None
                    abstract = getattr(item, 'abstractText', None)
                    pmcid = getattr(item, 'pmcid', None)

                    url = None
                    if doi:
                        url = f"https://doi.org/{doi}"
                    elif pmid:
                        url = f"https://europepmc.org/article/MED/{pmid}"
                    elif pmcid:
                        url = f"https://europepmc.org/article/PMC/{pmcid}"

                    result_item: SearchResult = {
                        "title": title,
                        "authors": authors,
                        "doi": doi,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "year": year,
                        "abstract": abstract,
                        "journal": journal,
                        "url": url,
                        "source_api": "EuropePMC",
                        "sjr_percentile": None, # Placeholder
                        "oa_status": None,      # Placeholder
                        "citation_count": None  # Placeholder (though EPMC might have it)
                    }
                    results.append(result_item)
            except Exception as e:
                print(f"EuropePMC search error: {e}")
        end_time = time.time()
        print(f"EuropePMC search took {end_time - start_time:.2f} seconds")
        return results

    async def search_semanticscholar(self, query: str, max_results: int = 20) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        async with self.semanticscholar_semaphore:
            try:
                print(f"Semantic Scholar: Starting search for '{query}' with limit {max_results}...")
                raw_results = await self._run_sync_in_thread(
                    self.s2.search_paper,
                    query,
                    limit=max_results,
                    fields=['title', 'authors', 'year', 'journal', 'abstract', 'url', 'venue', 'publicationDate', 'externalIds', 'citationCount'] # Added citationCount
                )
                print(f"Semantic Scholar: Finished search for '{query}'. Processing results...")
                
                item_count = 0
                for item in raw_results: # raw_results is already a list from s2.search_paper
                    if item_count >= max_results:
                        break
                    item_count += 1
                    
                    # Use attribute access for Semantic Scholar Paper object
                    title = item.title if hasattr(item, 'title') else None
                    authors = [author['name'] for author in item.authors if isinstance(author, dict) and 'name' in author] if hasattr(item, 'authors') and item.authors else []
                    year = item.year if hasattr(item, 'year') else None
                    
                    journal_name = None
                    if hasattr(item, 'journal') and item.journal:
                        if isinstance(item.journal, dict):
                            journal_name = item.journal.get('name')
                        elif isinstance(item.journal, str): # sometimes it might be just a string
                            journal_name = item.journal

                    external_ids = item.externalIds if hasattr(item, 'externalIds') and item.externalIds else {}
                    doi = external_ids.get('DOI')
                    pmid = external_ids.get('PubMed')
                    pmcid = external_ids.get('PubMedCentral')
                    
                    abstract = item.abstract if hasattr(item, 'abstract') else None
                    url = item.url if hasattr(item, 'url') else None
                    citation_count = item.citationCount if hasattr(item, 'citationCount') else None

                    result_item: SearchResult = {
                        "title": title,
                        "authors": authors,
                        "doi": doi,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "year": year,
                        "abstract": abstract,
                        "journal": journal_name,
                        "url": url,
                        "source_api": "SemanticScholar",
                        "sjr_percentile": None, 
                        "oa_status": None,      
                        "citation_count": citation_count
                    }
                    results.append(result_item)
            except Exception as e:
                print(f"Semantic Scholar search error: {e}")
        end_time = time.time()
        print(f"Semantic Scholar search took {end_time - start_time:.2f} seconds")
        return results

    async def search_crossref(self, query: str, max_results: int = 20) -> List[SearchResult]:
        if not self.cr:
            print("CrossRef client not initialized. Skipping CrossRef search.")
            return []
            
        start_time = time.time()
        results: List[SearchResult] = []
        async with self.crossref_semaphore:
            try:
                raw_results_iterable = await self._run_sync_in_thread(
                    self.cr.query(bibliographic=query).sample,
                    max_results
                )
                
                for item in raw_results_iterable:
                    if not isinstance(item, dict): continue

                    doi = item.get("DOI")
                    title_list = item.get("title")
                    title = title_list[0] if title_list else None

                    authors_list = item.get("author", [])
                    authors = []
                    for author_entry in authors_list:
                        name_parts = [author_entry.get('given'), author_entry.get('family')]
                        authors.append(" ".join(filter(None, name_parts)))
                    
                    issued_date_parts_list = item.get("issued", {}).get("date-parts", [[]])
                    year = None
                    if issued_date_parts_list and issued_date_parts_list[0] and issued_date_parts_list[0][0] is not None:
                        try:
                            year = int(issued_date_parts_list[0][0])
                        except (ValueError, TypeError):
                            year = None
                    
                    journal_list = item.get("container-title", [])
                    journal = journal_list[0] if journal_list else None
                    
                    url = item.get("URL")
                    citation_count = item.get('is-referenced-by-count')


                    result_item: SearchResult = {
                        "title": title,
                        "authors": authors,
                        "doi": doi,
                        "year": year,
                        "journal": journal,
                        "url": url,
                        "source_api": "CrossRef",
                        "sjr_percentile": None, 
                        "oa_status": None,      
                        "citation_count": citation_count,
                        "abstract": None # CrossRef sample doesn't usually provide abstracts
                    }
                    results.append(result_item)
            except Exception as e:
                print(f"CrossRef search error: {e}")
        end_time = time.time()
        print(f"CrossRef search took {end_time - start_time:.2f} seconds")
        return results

    async def search_all(self, pubmed_query: str, general_query: str, max_results_per_source: int = 20) -> pd.DataFrame:
        start_time = time.time()
        
        tasks = []
        if pubmed_query:
            tasks.append(self.search_pubmed(pubmed_query, max_results_per_source))
            tasks.append(self.search_europepmc(pubmed_query, max_results_per_source))
        if general_query:
            tasks.append(self.search_semanticscholar(general_query, max_results_per_source))
            tasks.append(self.search_crossref(general_query, max_results_per_source))

        all_source_results = await asyncio.gather(*tasks)
        
        flat_results: List[SearchResult] = []
        for source_result_list in all_source_results:
            if source_result_list: # Ensure it's not None
                flat_results.extend(source_result_list)

        if not flat_results:
            return pd.DataFrame()

        df = pd.DataFrame(flat_results)

        if 'doi' in df.columns:
            # Ensure doi_norm is created safely, handling potential all-NA cases for astype(str)
            df['doi_norm'] = df['doi'].apply(lambda x: str(x).lower().replace("https://doi.org/", "").strip() if pd.notna(x) and x != '' else pd.NA)
        else:
            df['doi_norm'] = pd.NA
        
        if 'pmid' in df.columns:
            df['pmid_str'] = df['pmid'].apply(lambda x: str(x) if pd.notna(x) and x != '' else pd.NA)
        else:
            df['pmid_str'] = pd.NA

        # Create boolean masks for filtering
        # A DOI is considered present if doi_norm is not NA (it won't be an empty string due to above .apply)
        mask_doi_present = df['doi_norm'].notna()
        
        df_with_doi_present = df[mask_doi_present].copy()
        df_no_doi_present = df[~mask_doi_present].copy()
        
        df_with_doi_deduped = pd.DataFrame()
        if not df_with_doi_present.empty:
            # Prioritize records with abstracts when multiple exist for the same DOI
            df_with_doi_present['has_abstract'] = df_with_doi_present['abstract'].notna() & (df_with_doi_present['abstract'] != '')
            df_with_doi_present = df_with_doi_present.sort_values(
                by=['doi_norm', 'has_abstract', 'year', 'pmid_str'], 
                ascending=[True, False, False, True], # True for has_abstract means non-None first
                na_position='last'
            )
            df_with_doi_deduped = df_with_doi_present.drop_duplicates(subset=['doi_norm'], keep='first')
            df_with_doi_deduped = df_with_doi_deduped.drop(columns=['has_abstract'], errors='ignore')


        df_no_doi_deduped = pd.DataFrame()
        if not df_no_doi_present.empty:
            df_no_doi_present['has_abstract'] = df_no_doi_present['abstract'].notna() & (df_no_doi_present['abstract'] != '')
            df_no_doi_present = df_no_doi_present.sort_values(
                by=['title', 'has_abstract', 'year', 'pmid_str'], # Use title for deduplication if no DOI
                ascending=[True, False, False, True], 
                na_position='last'
            )
            # For no-DOI records, be more conservative:
            # If pmid_str column exists and has any non-NA values, use pmid_str for deduplication.
            # Otherwise, use title and year.
            use_pmid_for_no_doi_dedup = False
            if 'pmid_str' in df_no_doi_present.columns and df_no_doi_present['pmid_str'].notna().any():
                use_pmid_for_no_doi_dedup = True
            
            subset_for_no_doi = ['pmid_str'] if use_pmid_for_no_doi_dedup else ['title', 'year']
            df_no_doi_deduped = df_no_doi_present.drop_duplicates(subset=subset_for_no_doi, keep='first')
            df_no_doi_deduped = df_no_doi_deduped.drop(columns=['has_abstract'], errors='ignore')

        
        df_final = pd.concat([df_with_doi_deduped, df_no_doi_deduped], ignore_index=True)

        # Ensure 'doi_norm' and 'pmid_str' are dropped if they exist
        cols_to_drop = [col for col in ['doi_norm', 'pmid_str'] if col in df_final.columns]
        if cols_to_drop:
            df_final.drop(columns=cols_to_drop, inplace=True)
        
        # Ensure all expected columns are present
        expected_df_cols = list(SearchResult.__annotations__.keys())
        for col in expected_df_cols:
            if col not in df_final.columns:
                df_final[col] = pd.NA # Use pandas NA
        
        # Reorder columns to match SearchResult TypedDict
        df_final = df_final[expected_df_cols]
        
        df_final = df_final.sort_values(by=['year', 'title'], ascending=[False, True], na_position='last').reset_index(drop=True)

        end_time = time.time()
        print(f"Overall search_all took {end_time - start_time:.2f} seconds. Found {len(df_final)} unique articles.")
        return df_final


class SearchOutput(TypedDict):
    data: List[Dict[str, Any]]
    meta: Dict[str, Any]

async def _async_search_literature(pubmed_query: str, general_query: str,
                                   max_results_per_source: int = 50) -> SearchOutput:
    """
    Internal helper that instantiates AsyncSearchClient and returns the structured output
    from client.search_all(). Always awaits client.close().
    """
    async with AsyncSearchClient() as client:
        df = await client.search_all(pubmed_query, general_query, max_results_per_source)
    
    data_records = []
    if not df.empty:
        for _, row in df.iterrows():
            record = {}
            for key, key_type in SearchResult.__annotations__.items():
                val = row.get(key)
                
                is_val_scalar_na = False
                if isinstance(val, (np.ndarray, pd.Series)):
                    # If val is an array or Series, it's not a single NA value.
                    is_val_scalar_na = False 
                elif isinstance(val, list): # Check for list before pd.isna
                    # If val is a list, it's not a single NA value.
                    is_val_scalar_na = False
                elif pd.isna(val): # Now val is confirmed not to be ndarray, Series, or list
                    is_val_scalar_na = True
                # else: val is not NA, and not array/series/list. is_val_scalar_na remains False.

                if is_val_scalar_na:
                    record[key] = None
                elif isinstance(val, (list, tuple)):
                    record[key] = list(val)
                elif isinstance(val, np.ndarray): # Handle numpy arrays explicitly
                    record[key] = val.tolist() # Convert numpy array to python list
                elif isinstance(val, pd.Series): # Handle pandas series explicitly
                    record[key] = val.tolist() # Convert pandas series to python list
                elif isinstance(val, (int, float, str, bool)):
                    record[key] = val
                else: 
                    try:
                        # For other types, try to convert to string.
                        record[key] = str(val) if val is not None else None
                    except: 
                        record[key] = val # Fallback
            data_records.append(record)

    meta_info = {
        "total_hits": len(data_records), 
        "query_pubmed": pubmed_query,
        "query_general": general_query,
        "timestamp": pd.Timestamp.now(tz='UTC').isoformat()
    }

    return {"data": data_records, "meta": meta_info}


def search_literature(pubmed_query: str, general_query: str,
                      max_results_per_source: int = 50) -> SearchOutput:
    """
    Search the biomedical literature via PubMed, Europe PMC, Semantic Scholar
    and Crossref.

    Args:
        pubmed_query: Boolean search string for PubMed and Europe PMC (can use MeSH).
        general_query: Boolean search string for Semantic Scholar and Crossref.
        max_results_per_source: Records to pull from each API (default = 50).

    Returns:
        A dictionary with 'data' and 'meta' fields.
        'data' is a list of dictionaries, each with 'doi', 'title', 'abstract'.
        'meta' contains 'total_hits', 'query', 'timestamp'.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        # Ensure the task is awaited if called from an already running loop context
        # This part might need adjustment based on how it's integrated into a larger async app
        future = asyncio.ensure_future(_async_search_literature(pubmed_query, general_query, max_results_per_source))
        return loop.run_until_complete(future) # This might be problematic if loop is already run_until_complete
                                              # Consider returning the future/task for the caller to await if in async context
    else:
        return asyncio.run(_async_search_literature(pubmed_query, general_query, max_results_per_source))

class SearchLiteratureParams(BaseModel):
    pubmed_query: str = Field(..., description="Boolean search string for PubMed and Europe PMC (can use MeSH).")
    general_query: str = Field(..., description="Boolean search string for Semantic Scholar and Crossref.")
    max_results_per_source: int = Field(50, description="Records to pull from each API (default = 50).")

class LiteratureSearchToolInstanceConfig(BaseModel):
    pass

class LiteratureSearchTool(
    BaseTool[SearchLiteratureParams, SearchOutput], 
    Component[LiteratureSearchToolInstanceConfig]
):
    component_config_schema: Type[LiteratureSearchToolInstanceConfig] = LiteratureSearchToolInstanceConfig

    def __init__(self):
        super().__init__(
            args_type=SearchLiteratureParams,
            return_type=SearchOutput, 
            name="search_literature",
            description="High-throughput literature search across PubMed, Europe PMC, Semantic Scholar and Crossref."
        )

    @classmethod
    def _from_config(cls, config: LiteratureSearchToolInstanceConfig) -> "LiteratureSearchTool":
        return cls()

    def _to_config(self) -> LiteratureSearchToolInstanceConfig:
        return LiteratureSearchToolInstanceConfig()

    async def run(self, args: SearchLiteratureParams, cancellation_token: Any) -> SearchOutput:
        # The search_literature function is synchronous, but this tool's run method is async.
        # We need to run the synchronous search_literature in a thread to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            search_literature, 
            args.pubmed_query, 
            args.general_query, 
            args.max_results_per_source
        )

__all__ = ["search_literature", "LiteratureSearchTool", "SearchLiteratureParams"]

if __name__ == "__main__":
    import sys
    import rich 
    import argparse 

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    parser = argparse.ArgumentParser(description="Test the literature search tool.")
    parser.add_argument("--pubmed", type=str, default="metagenomics[Mesh]", help="PubMed/EuropePMC query.")
    parser.add_argument("--general", type=str, default="metagenomics", help="General query for other APIs.")
    parser.add_argument("--max_results", type=int, default=10, help="Max results per source.")
    args = parser.parse_args()

    print(f"Searching PubMed/EuropePMC for: {args.pubmed}")
    print(f"Searching general APIs for: {args.general}")
    
    # search_literature is synchronous, so we call it directly
    search_output_dict = search_literature(args.pubmed, args.general, args.max_results)
    
    if search_output_dict and search_output_dict.get('data'):
        df = pd.DataFrame(search_output_dict['data'])
        rich.print(df.head())
    else:
        rich.print("[bold red]No data returned from search_literature.[/bold red]")
