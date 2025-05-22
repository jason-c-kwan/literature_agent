import asyncio
import os
from typing import List, Dict, Any, Optional, TypedDict

import httpx
import pandas as pd
from dotenv import load_dotenv

from Bio import Entrez
import europe_pmc
from semanticscholar import SemanticScholar
# from crossref.restful import Works # Original attempt
from unpywall import Unpywall

# Attempt to import from crossrefapi, with fallbacks for robustness
_Works_imported = False
_crossref_settings_imported = False
Works = None
crossref_settings = None

try:
    from crossref.restful import Works as CrWorks
    Works = CrWorks # Assign to global Works
    _Works_imported = True
    try:
        import crossref.settings as cr_settings
        crossref_settings = cr_settings
        _crossref_settings_imported = True
    except ImportError:
        print("Warning: Failed to import crossref.settings. CrossRef mailto parameter may not be set.")
except ImportError as e:
    print(f"Warning: Failed to import Works from crossref.restful ({e}). CrossRef search will be disabled.")
except AttributeError as e: # Catching AttributeError during import of crossref.restful
    print(f"Warning: AttributeError during import of crossref.restful ({e}). CrossRef search may be disabled or unstable.")
    # If Works was partially imported but is unusable, this might still be an issue.
    # For now, if an AttributeError occurs here, we assume Works might not be usable.
    Works = None # Ensure Works is None if its module had an issue
    _Works_imported = False

# Load environment variables from .env file
load_dotenv()

API_EMAIL = os.getenv("API_EMAIL")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", API_EMAIL) # Default to API_EMAIL
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", API_EMAIL) # Default to API_EMAIL


class SearchResult(TypedDict, total=False):
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
    is_open_access: Optional[bool]
    open_access_url: Optional[str]
    # Add other fields as necessary from different APIs


class AsyncSearchClient:
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.client = httpx.AsyncClient(timeout=30.0) # Increased timeout
        self.email = API_EMAIL
        self.pubmed_api_key = PUBMED_API_KEY
        self.semantic_scholar_api_key = SEMANTIC_SCHOLAR_API_KEY
        self.unpaywall_email = UNPAYWALL_EMAIL
        self.crossref_mailto = CROSSREF_MAILTO

        if not self.email:
            raise ValueError("API_EMAIL must be set in the .env file.")

        # Semaphores for rate limiting
        # PubMed: 3 RPS (anon), 10 RPS (key)
        pubmed_rps = 10 if self.pubmed_api_key else 3
        self.pubmed_semaphore = asyncio.Semaphore(pubmed_rps)
        # EuropePMC: 10 RPS
        self.europepmc_semaphore = asyncio.Semaphore(10)
        # Semantic Scholar: 1 RPS
        self.semanticscholar_semaphore = asyncio.Semaphore(1) # Strictest limit
        # Crossref: 50 RPS
        self.crossref_semaphore = asyncio.Semaphore(50)
        # Unpaywall: No hard limit, but let's be nice (e.g., 10 concurrent)
        self.unpaywall_semaphore = asyncio.Semaphore(10)
        
        # Initialize Bio.Entrez settings
        Entrez.email = self.email
        if self.pubmed_api_key:
            Entrez.api_key = self.pubmed_api_key

        # Initialize other synchronous clients
        self.s2 = SemanticScholar(api_key=self.semantic_scholar_api_key if self.semantic_scholar_api_key else None, timeout=20)
        
        self.cr = None
        if _Works_imported and Works is not None:
            if _crossref_settings_imported and self.crossref_mailto and crossref_settings:
                crossref_settings.mailto = self.crossref_mailto
            try:
                self.cr = Works()
            except Exception as e_init: # Catch error if Works() instantiation fails
                print(f"Warning: Failed to instantiate CrossRef Works client ({e_init}). CrossRef search disabled.")
                self.cr = None
        else:
            print("CrossRef Works class not imported. CrossRef search will be disabled.")
        
        # Unpywall setup - email is typically set globally or per instance
        # For unpywall, email is often set via Unpywall.mailto
        if self.unpaywall_email:
            Unpywall.mailto = self.unpaywall_email


    async def close(self):
        await self.client.aclose()

    async def _run_sync_in_thread(self, func, *args, **kwargs):
        """Helper to run synchronous functions in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def search_pubmed(self, query: str, max_results: int = 20) -> List[SearchResult]:
        results: List[SearchResult] = []
        async with self.pubmed_semaphore:
            try:
                # Step 1: Search for PMIDs
                handle = await self._run_sync_in_thread(
                    Entrez.esearch,
                    db="pubmed",
                    term=query,
                    retmax=str(max_results),
                    usehistory="y"
                )
                search_results = Entrez.read(handle)
                handle.close()
                ids = search_results["IdList"]

                if not ids:
                    return []

                # Step 2: Fetch details for PMIDs
                handle = await self._run_sync_in_thread(
                    Entrez.efetch,
                    db="pubmed",
                    id=ids,
                    rettype="medline", # Consider "xml" for more structured data if needed
                    retmode="text" # or "xml"
                )
                # For simplicity, we'll assume medline text format and parse basic fields.
                # A more robust parser would be needed for complex XML from efetch.
                # This part is highly dependent on the exact parsing strategy for Medline.
                # For now, let's assume a simplified parsing or focus on esummary.
                
                # Alternative: Use esummary for simpler, structured summaries
                summary_handle = await self._run_sync_in_thread(
                    Entrez.esummary,
                    db="pubmed",
                    id=",".join(ids)
                )
                summaries = Entrez.read(summary_handle)
                summary_handle.close()

                for summary in summaries:
                    # Extracting common fields from summary
                    # The exact field names depend on Bio.Entrez.read's parsing of esummary output
                    title = summary.get("Title", summary.get("ArticleTitle", "N/A"))
                    pmid = summary.get("Id", "")
                    doi = summary.get("DOI", "") # DOI might be in ArticleIds
                    article_ids = summary.get("ArticleIds", {})
                    if not doi and isinstance(article_ids, dict): # Check if ArticleIds is a dict
                        doi = article_ids.get("doi", "")

                    authors = [author['Name'] for author in summary.get("AuthorList", []) if 'Name' in author]
                    pub_date_str = summary.get("PubDate", "")
                    year = None
                    if pub_date_str and isinstance(pub_date_str, str) and pub_date_str.isdigit() and len(pub_date_str) >= 4:
                        year = int(pub_date_str[:4])
                    elif pub_date_str and isinstance(pub_date_str, str): # Handle cases like "2023 Spring"
                        year_match = Entrez._get_date(summary).year # Biopython internal helper
                        if year_match:
                            year = year_match


                    journal = summary.get("Source", "")
                    
                    result_item: SearchResult = {
                        "title": title,
                        "authors": authors,
                        "doi": doi if doi else None,
                        "pmid": pmid,
                        "year": year,
                        "journal": journal,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        "source_api": "PubMed"
                    }
                    results.append(result_item)

            except Exception as e:
                print(f"PubMed search error: {e}") # Basic error logging
        return results

    async def search_europepmc(self, query: str, max_results: int = 20) -> List[SearchResult]:
        results: List[SearchResult] = []
        async with self.europepmc_semaphore:
            try:
                # The europe_pmc library is synchronous
                raw_results = await self._run_sync_in_thread(
                    europe_pmc.query,
                    query,
                    resulttype='CORE', # 'LITE' for less data, 'IDLIST', 'ABSTRACTS'
                    page_size=max_results 
                )
                
                for item in raw_results: # raw_results is an iterator of Record objects
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
                        "source_api": "EuropePMC"
                    }
                    results.append(result_item)
            except Exception as e:
                print(f"EuropePMC search error: {e}")
        return results

    async def search_semanticscholar(self, query: str, max_results: int = 20) -> List[SearchResult]:
        results: List[SearchResult] = []
        async with self.semanticscholar_semaphore:
            try:
                # Semantic Scholar library is synchronous
                # Search for papers
                # The library handles fields internally for search results.
                # For more details like abstract, citations, references, a second call per paper might be needed.
                # For bulk search, we get basic info.
                raw_results = await self._run_sync_in_thread(
                    self.s2.search_paper,
                    query,
                    limit=max_results,
                    fields=['title', 'authors', 'year', 'journal', 'doi', 'pmid', 'abstract', 'url', 'venue', 'publicationDate', 'externalIds']
                )
                
                for item in raw_results: # Item is a Paper object
                    # Access attributes directly from the Paper object
                    title = item.title
                    authors = [author['name'] for author in item.authors] if item.authors else []
                    year = item.year
                    journal_info = item.journal if item.journal else {} # journal is a dict {'name': ..., 'volume': ...} or None
                    journal_name = journal_info.get('name') if isinstance(journal_info, dict) else None
                    
                    doi = item.doi or item.externalIds.get('DOI') if item.externalIds else None
                    pmid = item.externalIds.get('PubMed') if item.externalIds else None
                    pmcid = item.externalIds.get('PubMedCentral') if item.externalIds else None
                    abstract = item.abstract
                    url = item.url

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
                        "source_api": "SemanticScholar"
                    }
                    results.append(result_item)
            except Exception as e:
                print(f"Semantic Scholar search error: {e}")
        return results

    async def search_crossref(self, query: str, max_results: int = 20) -> List[SearchResult]:
        if not self.cr:
            print("CrossRef client not initialized. Skipping CrossRef search.")
            return []
            
        results: List[SearchResult] = []
        async with self.crossref_semaphore:
            try:
                # Crossref API library is synchronous
                # Use the query() method of the Works instance
                raw_results_iterable = await self._run_sync_in_thread(
                    self.cr.query,
                    bibliographic=query, 
                    limit=max_results,
                    # select="DOI,title,author,issued,container-title,URL,abstract" # To limit fields
                )
                
                # The iterable might be a generator, convert to list if needed or iterate directly
                # For simplicity, assuming it yields dicts
                for item in raw_results_iterable: # item is a dict
                    if not isinstance(item, dict): continue # Skip if not a dict

                    doi = item.get("DOI")
                    title_list = item.get("title")
                    title = title_list[0] if title_list else None

                    authors_list = item.get("author", [])
                    authors = []
                    for author_entry in authors_list:
                        name_parts = [author_entry.get('given'), author_entry.get('family')]
                        authors.append(" ".join(filter(None, name_parts)))
                    
                    issued_date_parts = item.get("issued", {}).get("date-parts", [[]])[0]
                    year = int(issued_date_parts[0]) if issued_date_parts and len(issued_date_parts) > 0 else None
                    
                    journal_list = item.get("container-title", [])
                    journal = journal_list[0] if journal_list else None
                    
                    url = item.get("URL")
                    # Crossref API's /works endpoint doesn't always return abstracts directly in search results.
                    # A separate call to /works/{doi} might be needed for abstract.
                    # For now, we'll skip abstract from Crossref search.
                    # abstract = item.get("abstract") # Often not available in list results

                    result_item: SearchResult = {
                        "title": title,
                        "authors": authors,
                        "doi": doi,
                        "year": year,
                        "journal": journal,
                        "url": url,
                        "source_api": "CrossRef"
                        # pmid/pmcid not typically primary fields from Crossref search
                    }
                    results.append(result_item)
            except Exception as e:
                print(f"CrossRef search error: {e}")
        return results

    async def search_unpaywall(self, dois: List[str]) -> Dict[str, SearchResult]:
        # Returns a dict mapping DOI to its Unpaywall info
        oa_info_map: Dict[str, SearchResult] = {}
        if not self.unpaywall_email:
            print("Unpaywall email not set, skipping Unpaywall search.")
            return oa_info_map
        
        # Unpywall.mailto should be set at class initialization
        # The library itself might handle batching or we do it one by one with semaphore
        
        async def fetch_one_doi(doi: str):
            async with self.unpaywall_semaphore:
                try:
                    # Unpywall library is synchronous
                    # Unpywall.doi expects a list of DOIs via the 'dois' keyword argument
                    response_list = await self._run_sync_in_thread(Unpywall.doi, dois=[doi])
                    # It should return a list of results, even for a single DOI
                    if response_list and isinstance(response_list, list) and len(response_list) > 0:
                        response = response_list[0] # Get the first (and only) result
                        if response and getattr(response, 'doi', None): # Check if response is valid
                            is_oa = getattr(response, 'is_oa', False)
                        best_oa_location = getattr(response, 'best_oa_location', None)
                        oa_url = best_oa_location.url if best_oa_location and hasattr(best_oa_location, 'url') else None
                        
                        oa_info_map[doi] = {
                            "is_open_access": is_oa,
                            "open_access_url": oa_url,
                            "doi": doi # ensure doi is part of the result for merging
                        }
                except Exception as e:
                    # Log error for specific DOI, but don't let it stop others
                    print(f"Unpaywall error for DOI {doi}: {e}")
        
        tasks = [fetch_one_doi(doi) for doi in dois if doi] # Filter out None or empty DOIs
        await asyncio.gather(*tasks)
        return oa_info_map

    async def search_all(self, query: str, max_results_per_source: int = 20) -> pd.DataFrame:
        # Gather results from all sources concurrently
        all_source_results = await asyncio.gather(
            self.search_pubmed(query, max_results_per_source),
            self.search_europepmc(query, max_results_per_source),
            self.search_semanticscholar(query, max_results_per_source),
            self.search_crossref(query, max_results_per_source),
            # Unpaywall is called later for enrichment
        )

        flat_results: List[SearchResult] = []
        for source_result_list in all_source_results:
            if source_result_list: # Ensure it's not None
                flat_results.extend(source_result_list)

        if not flat_results:
            return pd.DataFrame()

        df = pd.DataFrame(flat_results)

        # Deduplication strategy:
        # 1. Prioritize records with DOI.
        # 2. Normalize DOIs (e.g., lowercase, remove http://doi.org/)
        if 'doi' in df.columns:
            df['doi_norm'] = df['doi'].str.lower().str.replace("https://doi.org/", "", regex=False).str.strip()
        else:
            df['doi_norm'] = None # Ensure column exists

        if 'pmid' in df.columns:
            df['pmid_norm'] = df['pmid'].str.strip()
        else:
            df['pmid_norm'] = None


        # Sort by a preferred source or completeness before dropping duplicates
        # For now, simple deduplication:
        df_deduped = df.sort_values(by=['doi_norm', 'pmid_norm']).drop_duplicates(subset=['doi_norm'], keep='first')
        # For those without DOI, deduplicate by PMID
        df_no_doi = df_deduped[df_deduped['doi_norm'].isna()]
        df_with_doi = df_deduped[df_deduped['doi_norm'].notna()]
        
        if not df_no_doi.empty:
            df_no_doi = df_no_doi.drop_duplicates(subset=['pmid_norm'], keep='first')
            df = pd.concat([df_with_doi, df_no_doi]).reset_index(drop=True)
        else:
            df = df_with_doi.reset_index(drop=True)


        # Enrich with Unpaywall data
        unique_dois = df[df['doi'].notna()]['doi'].unique().tolist()
        if unique_dois:
            oa_data_map = await self.search_unpaywall(unique_dois)
            if oa_data_map:
                oa_df = pd.DataFrame(list(oa_data_map.values()))
                if not oa_df.empty and 'doi' in oa_df.columns:
                    # Normalize DOI in oa_df for merging
                    oa_df['doi_norm_merge'] = oa_df['doi'].str.lower().str.replace("https://doi.org/", "", regex=False).str.strip()
                    # Merge requires 'doi_norm' in the main df to be consistent
                    df = df.merge(oa_df[['doi_norm_merge', 'is_open_access', 'open_access_url']], 
                                  left_on='doi_norm', right_on='doi_norm_merge', how='left')
                    df.drop(columns=['doi_norm_merge'], inplace=True, errors='ignore')


        df.drop(columns=['doi_norm', 'pmid_norm'], inplace=True, errors='ignore')
        return df


# Example Usage (for testing purposes)
async def main():
    search_client = AsyncSearchClient()
    try:
        query = "COVID-19 vaccine"
        print(f"Searching for: {query}")
        
        # Test individual sources
        # print("\n--- PubMed ---")
        # pubmed_results = await search_client.search_pubmed(query, max_results=5)
        # print(pd.DataFrame(pubmed_results))

        # print("\n--- EuropePMC ---")
        # europepmc_results = await search_client.search_europepmc(query, max_results=5)
        # print(pd.DataFrame(europepmc_results))

        # print("\n--- Semantic Scholar ---")
        # s2_results = await search_client.search_semanticscholar(query, max_results=5)
        # print(pd.DataFrame(s2_results))
        
        # print("\n--- CrossRef ---")
        # cr_results = await search_client.search_crossref(query, max_results=5)
        # print(pd.DataFrame(cr_results))

        # Test combined search
        print("\n--- All Sources Combined & Deduplicated ---")
        combined_df = await search_client.search_all(query, max_results_per_source=10)
        print(combined_df)
        
        if not combined_df.empty and 'doi' in combined_df.columns:
            sample_dois = combined_df[combined_df['doi'].notna()]['doi'].head(3).tolist()
            if sample_dois:
                print(f"\n--- Unpaywall for DOIs: {sample_dois} ---")
                unpaywall_res = await search_client.search_unpaywall(sample_dois)
                print(unpaywall_res)


    finally:
        await search_client.close()

if __name__ == "__main__":
    # To run this example: python -m tools.search
    # Ensure .env file is present in the root directory or where the script is run from.
    if os.name == 'nt': # Fix for ProactorLoop an Windows for asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
