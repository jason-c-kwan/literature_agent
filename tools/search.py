import asyncio
import os
import time # Added for debug timing
from typing import List, Dict, Any, Optional, TypedDict
import httpx
import pandas as pd
from dotenv import load_dotenv
from Bio import Entrez
from europe_pmc import EuropePMC # Import the client class
from semanticscholar import SemanticScholar
from unpywall import Unpywall
from crossref.restful import Etiquette # Added for CrossRef mailto
from autogen_core.tools import BaseTool # Import BaseTool
from autogen_core._component_config import Component # Import Component
from pydantic import BaseModel, Field # For defining schema fields
from typing import Type # For Type hint

# Attempt to import from crossrefapi, with fallbacks for robustness
_Works_imported = False
Works = None

try:
    from crossref.restful import Works as CrWorks
    Works = CrWorks # Assign to global Works
    _Works_imported = True
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
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO")

# If UNPAYWALL_EMAIL or CROSSREF_MAILTO were set to "$API_EMAIL" in the .env file,
# os.getenv will return that literal string. We need to resolve it to the actual API_EMAIL value.
# Also, provide a fallback to API_EMAIL if they are not set at all.
if UNPAYWALL_EMAIL == "$API_EMAIL" or UNPAYWALL_EMAIL is None:
    UNPAYWALL_EMAIL = API_EMAIL
if CROSSREF_MAILTO == "$API_EMAIL" or CROSSREF_MAILTO is None:
    CROSSREF_MAILTO = API_EMAIL


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
    open_access_url_source: Optional[str] # Source of the OA URL (Unpaywall, PMC, EuropePMC)
    oa_status: Optional[str] # e.g., 'gold', 'green', 'hybrid' from Unpaywall
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
        # PubMed: 1 RPS (anon), 2 RPS (key) - Reduced to avoid 429 errors
        pubmed_rps = 2 if self.pubmed_api_key else 1
        self.pubmed_semaphore = asyncio.Semaphore(pubmed_rps)
        # EuropePMC: 5 RPS - Reduced to avoid 429 errors
        self.europepmc_semaphore = asyncio.Semaphore(5)
        # Semantic Scholar: 1 RPS
        self.semanticscholar_semaphore = asyncio.Semaphore(1) # Strictest limit
        # Crossref: 50 RPS
        self.crossref_semaphore = asyncio.Semaphore(50)
        # Unpaywall: No hard limit, but let's be nice (e.g., 10 concurrent)
        self.unpaywall_semaphore = asyncio.Semaphore(10)
        # Semaphore to serialize synchronous Unpywall.doi calls
        self.unpywall_sync_semaphore = asyncio.Semaphore(1)
        
        # Initialize Bio.Entrez settings
        Entrez.email = self.email
        if self.pubmed_api_key:
            Entrez.api_key = self.pubmed_api_key

        # Initialize other synchronous clients
        self.epmc = EuropePMC() # Initialize EuropePMC client
        self.s2 = SemanticScholar(api_key=self.semantic_scholar_api_key if self.semantic_scholar_api_key else None, timeout=20)
        
        self.cr = None
        if _Works_imported and Works is not None:
            try:
                # Pass Etiquette object with mailto for CrossRef
                etiquette_obj = Etiquette(contact_email=self.crossref_mailto)
                self.cr = Works(etiquette=etiquette_obj)
            except Exception as e_init: # Catch error if Works() instantiation fails
                print(f"Warning: Failed to instantiate CrossRef Works client ({e_init}). CrossRef search disabled.")
                self.cr = None
        else:
            print("CrossRef Works class not imported. CrossRef search will be disabled.")
        
        # Unpywall setup - email is typically set globally or per instance
        # For unpywall, email is often set via Unpywall.mailto
        if self.unpaywall_email:
            Unpywall.mailto = self.unpaywall_email

    async def _get_open_access_url_with_fallback(self, doi: str) -> SearchResult:
        """
        Tries to find an open-access URL for a given DOI using Unpaywall,
        then PMC, then Europe PMC as fallbacks.
        It also extracts the specific 'oa_status' from Unpaywall.
        """
        oa_url: Optional[str] = None
        is_oa: Optional[bool] = False # Default to False, set to True if any source provides a URL
        source_of_oa_url: Optional[str] = None
        oa_status_str: Optional[str] = None # To store 'gold', 'green', etc.

        # 1. Try Unpaywall
        try:
            async with self.unpaywall_semaphore, self.unpywall_sync_semaphore:
                # Unpywall.doi expects a list of DOIs
                response_df = await self._run_sync_in_thread(Unpywall.doi, dois=[doi])
            if response_df is not None and not response_df.empty:
                record = response_df.iloc[0]
                oa_status_str = record.get('oa_status') # Get the specific OA status
                if record.get('is_oa'):
                    is_oa = True
                    best_oa_location = record.get('best_oa_location')
                    if best_oa_location and isinstance(best_oa_location, dict):
                        oa_url = best_oa_location.get('url')
                        if oa_url:
                            source_of_oa_url = "Unpaywall" # Prioritize Unpaywall if it gives a URL
        except Exception as e:
            print(f"Unpaywall error for DOI {doi}: {e}")

        # 2. Fallback to PubMed Central (PMC) if no URL from Unpaywall (and Unpaywall was the preferred source)
        if not oa_url and doi:
            try:
                async with self.pubmed_semaphore: # Use existing PubMed semaphore
                    # Search PMC for the DOI
                    handle = await self._run_sync_in_thread(
                        Entrez.esearch,
                        db="pmc",
                        term=f"{doi}[DOI]",
                        retmax="1"
                    )
                    search_results = Entrez.read(handle)
                    handle.close()
                    pmc_ids = search_results.get("IdList", [])

                    if pmc_ids:
                        pmcid = pmc_ids[0]
                        # Attempt to construct a direct link or use elink for more robust link finding
                        # For simplicity, let's try a common pattern first.
                        # A more robust way would be to use Entrez.elink to find 'pmc full text' links.
                        # Example: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/
                        # Or check for full text links via esummary or efetch if available.
                        # For now, we'll assume if a PMCID is found, it's likely OA via PMC's main article page.
                        # The actual PDF link might be on that page.
                        oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
                        is_oa = True # Assume if found in PMC, it's OA
                        source_of_oa_url = "PMC"
                        # To get a direct PDF link from PMC, one might need to parse the article page
                        # or use more specific Entrez features if they provide direct PDF links.
                        # Entrez.elink with cmd="prlinks" might be more accurate here.
                        # Let's try elink to get a more direct link if possible
                        link_handle = await self._run_sync_in_thread(
                            Entrez.elink,
                            dbfrom="pmc",
                            db="pmc", # Link within pmc to find full text providers or use 'pubmed' for related
                            id=pmcid,
                            cmd="prlinks" # This command provides links to full-text providers
                        )
                        link_results = Entrez.read(link_handle)
                        link_handle.close()

                        if link_results and link_results[0].get("LinkSetDb"):
                            for link_set_db in link_results[0]["LinkSetDb"]:
                                if link_set_db.get("LinkName") == "pmc_pmc_ft": # Full text link
                                    if link_set_db.get("Link"):
                                        # Prefer PDF if available
                                        pdf_link_found = False
                                        for link_info in link_set_db["Link"]:
                                            if link_info.get("Url") and 'pdf' in link_info.get("Url", "").lower():
                                                oa_url = link_info["Url"]
                                                pdf_link_found = True
                                                break
                                        if not pdf_link_found and link_set_db["Link"]: # Fallback to first link if no PDF
                                            oa_url = link_set_db["Link"][0].get("Url", oa_url) # Keep previous if no new URL
                                        break # Found full text links
                        # If elink didn't provide a better URL, the constructed PMC article URL is a fallback
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = e.response.headers.get('Retry-After')
                    print(f"PMC fallback error for DOI {doi}: HTTP Error 429: Too Many Requests. Retry-After: {retry_after}. Full response: {e.response.text}")
                else:
                    print(f"PMC fallback HTTP error for DOI {doi}: {e}")
            except Exception as e:
                print(f"PMC fallback general error for DOI {doi}: {e}")

        # 3. Fallback to Europe PMC if still no URL
        if not oa_url and doi:
            try:
                async with self.europepmc_semaphore:
                    # The europe_pmc library's search can take a DOI query
                    # query format for DOI: "DOI:10.1234/journal.xxxx"
                    epmc_result_dict = await self._run_sync_in_thread(
                        self.epmc.search, # self.epmc is EuropePMC() instance
                        f"DOI:{doi}"
                    )
                    
                    # Check if result is valid and has pmcid for PDF link construction
                    if epmc_result_dict and not epmc_result_dict.get('_error'):
                        pmcid = epmc_result_dict.get('pmcid')
                        if pmcid:
                             # Check if it has full text available, EuropePMC often indicates this
                            has_ft = epmc_result_dict.get('hasTextMinedTerms') == 'Y' or \
                                     epmc_result_dict.get('isOpenAccess') == 'Y' or \
                                     epmc_result_dict.get('inEPMC') == 'Y'

                            if has_ft or pmcid: # If PMCID exists, good chance of OA link
                                # Construct the PDF URL as per library's pattern
                                potential_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                                # Verify if this URL is likely valid or if there's a better one in response
                                # The library's `fetch` method adds `pdf_url` if pmcid exists.
                                # We can also check `fullTextUrlList` if present in the response.
                                ft_urls = epmc_result_dict.get('fullTextUrlList', {}).get('fullTextUrl', [])
                                found_ft_url = False
                                if ft_urls:
                                    for ft_url_info in ft_urls:
                                        if ft_url_info.get('documentStyle') == 'pdf' and ft_url_info.get('availabilityCode') == 'OA':
                                            oa_url = ft_url_info.get('url')
                                            found_ft_url = True
                                            break
                                    if not found_ft_url: # Take first OA if no PDF
                                        for ft_url_info in ft_urls:
                                            if ft_url_info.get('availabilityCode') == 'OA':
                                                oa_url = ft_url_info.get('url')
                                                found_ft_url = True
                                                break
                                
                                if not found_ft_url and pmcid: # Fallback to constructed PDF URL if PMCID exists
                                    oa_url = potential_url
                                
                                if oa_url:
                                    is_oa = True
                                    source_of_oa_url = "EuropePMC"
            except Exception as e:
                print(f"EuropePMC fallback error for DOI {doi}: {e}")
        
        return {
            "doi": doi, # Ensure DOI is part of the result for mapping
            "is_open_access": is_oa if oa_url else False, # Only True if a URL was found
            "open_access_url": oa_url,
            "open_access_url_source": source_of_oa_url, # Track where the URL came from
            "oa_status": oa_status_str # Add the specific oa_status
        }

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
                    if pub_date_str and isinstance(pub_date_str, str):
                        # Attempt to extract year. PubDate can be "YYYY", "YYYY Mon", "YYYY Mon DD", etc.
                        # Split by space and check if the first part is a 4-digit year.
                        parts = pub_date_str.split()
                        if parts and parts[0].isdigit() and len(parts[0]) == 4:
                            year = int(parts[0])
                        # Fallback for cases where PubDate might just be "YYYY" and isdigit() would catch it
                        elif pub_date_str.isdigit() and len(pub_date_str) == 4:
                             year = int(pub_date_str)
                        # Add more sophisticated parsing if needed for other formats

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
        end_time = time.time()
        print(f"PubMed search took {end_time - start_time:.2f} seconds")
        return results

    async def search_europepmc(self, query: str, max_results: int = 20) -> List[SearchResult]:
        start_time = time.time() # Added missing start_time initialization
        results: List[SearchResult] = []
        async with self.europepmc_semaphore:
            try:
                # The europe_pmc library is synchronous
                # Use the initialized client and its 'search' method
                # Adjust parameter names: resulttype -> result_type
                raw_results = await self._run_sync_in_thread(
                    self.epmc.search,
                    query
                )
                
                # The library returns a generator of Record objects
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
        end_time = time.time()
        print(f"EuropePMC search took {end_time - start_time:.2f} seconds")
        return results

    async def search_semanticscholar(self, query: str, max_results: int = 20) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        async with self.semanticscholar_semaphore:
            try:
                # Semantic Scholar library is synchronous
                # Search for papers
                # The library handles fields internally for search results.
                # For more details like abstract, citations, references, a second call per paper might be needed.
                # For bulk search, we get basic info.
                print(f"Semantic Scholar: Starting search for '{query}' with limit {max_results}...")
                raw_results = await self._run_sync_in_thread(
                    self.s2.search_paper,
                    query,
                    limit=max_results,
                    fields=['title', 'authors', 'year', 'journal', 'abstract', 'url', 'venue', 'publicationDate', 'externalIds']
                )
                print(f"Semantic Scholar: Finished search for '{query}'. Processing results...")
                
                print("DEBUG: Starting iteration over Semantic Scholar raw_results.")
                item_count = 0
                for item in raw_results: # Item is a Paper object
                    if item_count >= max_results: # Explicitly break if max_results is reached
                        print(f"DEBUG: Reached max_results ({max_results}) for Semantic Scholar. Breaking loop.")
                        break
                    item_count += 1
                    print(f"DEBUG: Processing Semantic Scholar item {item_count}...")
                    try:
                        # Access attributes directly from the Paper object
                        title = item.title
                        authors = [author['name'] for author in item.authors] if item.authors else []
                        year = item.year
                        journal_info = item.journal if item.journal else {} # journal is a dict {'name': ..., 'volume': ...} or None
                        journal_name = journal_info.get('name') if isinstance(journal_info, dict) else None
                        
                        external_ids = item.externalIds if item.externalIds else {}
                        doi = external_ids.get('DOI')
                        pmid = external_ids.get('PubMed')
                        pmcid = external_ids.get('PubMedCentral')
                        
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
                        print(f"DEBUG: Successfully processed Semantic Scholar item {item_count}.")
                    except Exception as item_e:
                        print(f"Semantic Scholar item processing error for item {item_count}: {item_e}")
                        # Continue to next item even if one fails
                print(f"DEBUG: Finished iteration over Semantic Scholar raw_results. Processed {item_count} items.")
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
                # Crossref API library is synchronous
                # Use the query() method of the Works instance
                raw_results_iterable = await self._run_sync_in_thread(
                    self.cr.query(bibliographic=query).sample,
                    max_results
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
        end_time = time.time()
        print(f"CrossRef search took {end_time - start_time:.2f} seconds")
        return results

    async def search_all(self, query: str, max_results_per_source: int = 20) -> pd.DataFrame:
        start_time = time.time()
        # Gather results from all sources concurrently
        all_source_results = await asyncio.gather(
            self.search_pubmed(query, max_results_per_source),
            self.search_europepmc(query, max_results_per_source),
            self.search_semanticscholar(query, max_results_per_source),
            self.search_crossref(query, max_results_per_source),
            # Unpaywall is called later for enrichment
        )
        print("DEBUG: All source searches completed.")

        flat_results: List[SearchResult] = []
        print("DEBUG: Starting to flatten results.")
        for source_result_list in all_source_results:
            if source_result_list: # Ensure it's not None
                flat_results.extend(source_result_list)
        print(f"DEBUG: flat_results created with {len(flat_results)} items.")

        if not flat_results:
            print("DEBUG: No flat_results, returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(flat_results)
        print(f"DEBUG: DataFrame created with {len(df)} rows.")

        # Deduplication strategy:
        # 1. Prioritize records with DOI.
        # 2. Normalize DOIs (e.g., lowercase, remove http://doi.org/)
        if 'doi' in df.columns:
            print("DEBUG: Normalizing DOIs.")
            df['doi_norm'] = df['doi'].str.lower().str.replace("https://doi.org/", "", regex=False).str.strip()
        else:
            df['doi_norm'] = None
        print("DEBUG: DOI normalization complete.")

        # pmid_norm creation removed as deduplication is handled in _async_search_literature
        # if 'pmid' in df.columns:
        #     print("DEBUG: Normalizing PMIDs.")
        #     df['pmid_norm'] = df['pmid'].str.strip()
        # else:
        #     df['pmid_norm'] = None
        # print("DEBUG: PMID normalization complete.")

        # Deduplication logic moved to _async_search_literature.
        # The DataFrame df at this point contains all aggregated results before final deduplication.

        # Enrich with OA data using fallback mechanism
        unique_dois = df[df['doi'].notna()]['doi'].unique().tolist()
        if unique_dois:
            print(f"DEBUG: Calling search_unpaywall (with fallback) for {len(unique_dois)} unique DOIs.")
            oa_data_map = await self.search_unpaywall(unique_dois) # This now uses the fallback
            print(f"DEBUG: search_unpaywall (with fallback) returned {len(oa_data_map)} results.")
            if oa_data_map:
                # DataFrame from oa_data_map will have 'doi', 'is_open_access', 'open_access_url', 'open_access_url_source'
                oa_df = pd.DataFrame(list(oa_data_map.values()))
                if not oa_df.empty and 'doi' in oa_df.columns:
                    print("DEBUG: Merging OA fallback data.")
                    oa_df['doi_norm_merge'] = oa_df['doi'].str.lower().str.replace("https://doi.org/", "", regex=False).str.strip()
                    
                    # Columns to merge from oa_df. Ensure all are present in oa_df.
                    oa_merge_cols = ['doi_norm_merge']
                    if 'is_open_access' in oa_df.columns:
                        oa_merge_cols.append('is_open_access')
                    if 'open_access_url' in oa_df.columns:
                        oa_merge_cols.append('open_access_url')
                    if 'open_access_url_source' in oa_df.columns:
                        oa_merge_cols.append('open_access_url_source')
                    if 'oa_status' in oa_df.columns: # Add oa_status to merge
                        oa_merge_cols.append('oa_status')
                    
                    oa_df_to_merge = oa_df[oa_merge_cols].copy()

                    # Store original columns that might be overwritten, to handle NaNs correctly if needed
                    # However, the fallback mechanism provides the definitive values for these OA fields.
                    # So, we will overwrite.
                    
                    # Suffixes: _orig for original df columns if they clash, _oa for new columns from oa_df
                    df = df.merge(oa_df_to_merge,
                                  left_on='doi_norm', right_on='doi_norm_merge',
                                  how='left',
                                  suffixes=('_orig', '_oa'))
                    
                    df.drop(columns=['doi_norm_merge'], inplace=True, errors='ignore')

                    # Update main df columns with the authoritative data from the OA search
                    if 'is_open_access_oa' in df.columns:
                        # If 'is_open_access_orig' exists, we are replacing it.
                        # If it doesn't exist, 'is_open_access_oa' becomes the new 'is_open_access'.
                        df['is_open_access'] = df['is_open_access_oa']
                        df.drop(columns=['is_open_access_oa'], inplace=True)
                        if 'is_open_access_orig' in df.columns and 'is_open_access_orig' != 'is_open_access':
                             df.drop(columns=['is_open_access_orig'], inplace=True, errors='ignore')


                    if 'open_access_url_oa' in df.columns:
                        df['open_access_url'] = df['open_access_url_oa']
                        df.drop(columns=['open_access_url_oa'], inplace=True)
                        if 'open_access_url_orig' in df.columns and 'open_access_url_orig' != 'open_access_url':
                            df.drop(columns=['open_access_url_orig'], inplace=True, errors='ignore')

                    if 'open_access_url_source_oa' in df.columns:
                        df['open_access_url_source'] = df['open_access_url_source_oa']
                        df.drop(columns=['open_access_url_source_oa'], inplace=True)
                        if 'open_access_url_source_orig' in df.columns and 'open_access_url_source_orig' != 'open_access_url_source':
                             df.drop(columns=['open_access_url_source_orig'], inplace=True, errors='ignore')
                    
                    if 'oa_status_oa' in df.columns: # Handle merging oa_status
                        df['oa_status'] = df['oa_status_oa']
                        df.drop(columns=['oa_status_oa'], inplace=True)
                        if 'oa_status_orig' in df.columns and 'oa_status_orig' != 'oa_status':
                            df.drop(columns=['oa_status_orig'], inplace=True, errors='ignore')

                    print("DEBUG: OA fallback data merge complete.")

        # Drop the main doi_norm column used for merging, pmid_norm is handled in _async_search_literature
        df.drop(columns=['doi_norm'], inplace=True, errors='ignore')
        # The pmid_norm column is dropped in _async_search_literature after deduplication.
        # df.drop(columns=['doi_norm', 'pmid_norm'], inplace=True, errors='ignore') # Original line
        print("DEBUG: Dropped normalization columns.")
        end_time = time.time()
        print(f"Overall search_all took {end_time - start_time:.2f} seconds")
        return df

    async def search_unpaywall(self, dois: List[str]) -> Dict[str, SearchResult]:
        start_time = time.time()
        # Returns a dict mapping DOI to its OA info (URL, is_oa status, source)
        oa_info_map: Dict[str, SearchResult] = {}
        if not self.unpaywall_email: # Unpaywall email check is still relevant for the first step
            print("Unpaywall email not set, Unpaywall part of OA search might be skipped or fail.")
            # We can still proceed with PMC and EuropePMC fallbacks if DOI is available.

        tasks = [self._get_open_access_url_with_fallback(doi) for doi in dois if doi]
        print(f"OA Fallback Search: Gathering {len(tasks)} DOI fetch tasks...")
        
        # Collect results from all concurrent fetches
        all_oa_results = await asyncio.gather(*tasks)
        print(f"DEBUG: All OA Fallback DOI fetch tasks gathered.")

        # Build the oa_info_map from collected results
        for result_item in all_oa_results:
            # result_item is a SearchResult dict from _get_open_access_url_with_fallback
            if result_item and result_item.get('doi'):
                # Ensure we store the complete SearchResult structure
                oa_info_map[result_item['doi']] = result_item
        
        end_time = time.time()
        print(f"OA Fallback search (Unpaywall, PMC, EuropePMC) took {end_time - start_time:.2f} seconds")
        return oa_info_map


async def _async_search_literature(query: str,
                                   max_results_per_source: int = 50) -> pd.DataFrame:
    """
    Internal helper that instantiates AsyncSearchClient and returns the DataFrame
    from client.search_all(). Always awaits client.close().
    """
    async with AsyncSearchClient() as client:
        df = await client.search_all(query, max_results_per_source)

        # Post-processing: drop duplicates, sort, reset index
        if not df.empty:
            # Deduplication: prefer DOI, else PMID
            # Assuming 'doi_norm' and 'pmid_norm' are created by search_all
            # If not, ensure they are created here or in search_all
            if 'doi' in df.columns:
                df['doi_norm'] = df['doi'].str.lower().str.replace("https://doi.org/", "", regex=False).str.strip()
            else:
                df['doi_norm'] = None # Use None, pandas handles it as NaN for str ops

            if 'pmid' in df.columns:
                df['pmid_norm'] = df['pmid'].str.strip()
            else:
                df['pmid_norm'] = None

            # Corrected Deduplication Strategy for _async_search_literature
            df_with_doi_present = df[df['doi_norm'].notna()].copy()
            df_no_doi_present = df[df['doi_norm'].isna()].copy()

            df_with_doi_deduped = pd.DataFrame(columns=df.columns)
            if not df_with_doi_present.empty:
                df_with_doi_present = df_with_doi_present.sort_values(
                    by=['doi_norm', 'year', 'pmid_norm'], 
                    ascending=[True, False, True], 
                    na_position='last'
                )
                df_with_doi_deduped = df_with_doi_present.drop_duplicates(subset=['doi_norm'], keep='first')

            df_no_doi_deduped = pd.DataFrame(columns=df.columns)
            if not df_no_doi_present.empty:
                df_no_doi_present = df_no_doi_present.sort_values(
                    by=['pmid_norm', 'year'], 
                    ascending=[True, False], 
                    na_position='last'
                )
                df_no_doi_deduped = df_no_doi_present.drop_duplicates(subset=['pmid_norm'], keep='first')
            
            df = pd.concat([df_with_doi_deduped, df_no_doi_deduped])

            # Drop the temporary normalization columns
            df.drop(columns=['doi_norm', 'pmid_norm'], inplace=True, errors='ignore')

            # Sort by source_api, then year desc
            df = df.sort_values(by=['source_api', 'year'], ascending=[True, False])
            
            # Ensure all expected columns from SearchResult are present
            # Add 'open_access_url_source' and 'oa_status' to expected columns if it's now part of SearchResult
            # For now, SearchResult TypedDict is not modified, but the merge below adds it.
            # If we formally add it to SearchResult, this list should be updated.
            expected_df_cols = list(SearchResult.__annotations__.keys())
            # Manually add new columns if not in SearchResult TypedDict yet for ordering
            # 'open_access_url_source' is already in SearchResult TypedDict as of previous edits.
            # 'oa_status' was added to SearchResult TypedDict in this change.
            # So, SearchResult.__annotations__.keys() should now include them.

            for col in expected_df_cols:
                if col not in df.columns:
                    df[col] = pd.NA # Use pandas NA for missing values
            
            # Reorder columns to a canonical order (optional, but good for consistency)
            # Ensure only columns that *could* exist are included in reindex
            current_cols_ordered = [col for col in expected_df_cols if col in df.columns]
            # Add any other columns that might have been created (e.g. by merge, though unlikely here)
            other_cols = [col for col in df.columns if col not in current_cols_ordered]
            df = df[current_cols_ordered + other_cols]

            # Reset index
            df = df.reset_index(drop=True)

    return df

def search_literature(query: str,
                      max_results_per_source: int = 50) -> pd.DataFrame:
    """
    Search the biomedical literature via PubMed, Europe PMC, Semantic Scholar
    and Crossref.

    Args:
        query: Free-text Boolean search string.
        max_results_per_source: Records to pull from each API (default = 50).

    Returns:
        pandas.DataFrame with columns:
        ['title','authors','doi','pmid','pmcid','year','abstract','journal',
         'url','source_api','is_open_access','open_access_url'].
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        task = loop.create_task(_async_search_literature(query, max_results_per_source))
        return loop.run_until_complete(task)
    else:
        return asyncio.run(_async_search_literature(query, max_results_per_source))

class SearchLiteratureParams(BaseModel): # For the 'run' method arguments
    query: str = Field(..., description="Free-text Boolean search string.")
    max_results_per_source: int = Field(50, description="Records to pull from each API (default = 50).")

# Config model for ComponentLoader instantiation (matches config: {} in YAML)
class LiteratureSearchToolInstanceConfig(BaseModel):
    pass

class LiteratureSearchTool(
    BaseTool[SearchLiteratureParams, pd.DataFrame],
    Component[LiteratureSearchToolInstanceConfig]
):
    # Required by ComponentSchemaType (via Component)
    component_config_schema: Type[LiteratureSearchToolInstanceConfig] = LiteratureSearchToolInstanceConfig

    # component_type = "tool" is inherited from BaseTool

    def __init__(self):
        # Call BaseTool's __init__
        super().__init__(
            args_type=SearchLiteratureParams,
            return_type=pd.DataFrame,
            name="search_literature",
            description="High-throughput literature search across PubMed, Europe PMC, Semantic Scholar and Crossref."
        )

    # Required by ComponentFromConfig (via Component)
    @classmethod
    def _from_config(cls, config: LiteratureSearchToolInstanceConfig) -> "LiteratureSearchTool":
        # The 'config' parameter is from YAML's 'config: {}'.
        # LiteratureSearchTool's __init__ doesn't take arguments from this config.
        return cls()

    # Required by ComponentToConfig (via Component, effectively overriding placeholder in ComponentBase)
    def _to_config(self) -> LiteratureSearchToolInstanceConfig:
        # Corresponds to the config used in _from_config.
        return LiteratureSearchToolInstanceConfig()

    # run method is inherited from BaseTool and needs to be implemented
    async def run(self, args: SearchLiteratureParams, cancellation_token: Any) -> pd.DataFrame:
        # Call the synchronous search_literature function
        return search_literature(args.query, args.max_results_per_source)

__all__ = ["search_literature", "AsyncSearchClient"]

if __name__ == "__main__":
    import sys, rich
    # To run this example: python -m tools.search "CRISPR base editing"
    # Ensure .env file is present in the root directory or where the script is run from.
    if os.name == 'nt': # Fix for ProactorLoop on Windows for asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Use sys.argv for query, default to "metagenomics" if no args
    query_args = sys.argv[1:]
    query = " AND ".join(query_args) if query_args else "metagenomics"
    
    print(f"Searching for: {query}")
    df = search_literature(query)
    rich.print(df.head()) # pretty debug view
