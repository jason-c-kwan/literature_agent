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
    def __init__(self, publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.email = API_EMAIL
        self.pubmed_api_key = PUBMED_API_KEY
        self.semantic_scholar_api_key = SEMANTIC_SCHOLAR_API_KEY
        self.crossref_mailto = CROSSREF_MAILTO
        self.publication_type_mappings = publication_type_mappings if publication_type_mappings else {}

        if not self.email:
            raise ValueError("API_EMAIL must be set in the .env file.")

        # Semaphores for rate limiting
        pubmed_rps = 2 if self.pubmed_api_key else 1
        self.pubmed_semaphore = asyncio.Semaphore(pubmed_rps)
        self.europepmc_semaphore = asyncio.Semaphore(5)
        self.semanticscholar_semaphore = asyncio.Semaphore(1) # S2 rate limits are strict
        self.crossref_semaphore = asyncio.Semaphore(50) # CrossRef is more permissive
        self.openalex_semaphore = asyncio.Semaphore(10) # OpenAlex polite pool limit
        
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

    async def search_pubmed(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        
        final_query = query
        if publication_types and self.publication_type_mappings:
            filters = []
            for pub_type_key in publication_types: # e.g., "research", "review"
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("pubmed")
                if api_specific_value:
                    filters.append(api_specific_value) # Already includes "[Publication Type]"
            if filters:
                filter_term = " OR ".join(filters)
                final_query = f"({query}) AND ({filter_term})"
        
        async with self.pubmed_semaphore:
            try:
                handle = await self._run_sync_in_thread(
                    Entrez.esearch,
                    db="pubmed",
                    term=final_query,
                    retmax=str(max_results),
                    usehistory="y"
                )
                search_results_handle = Entrez.read(handle) 
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
        print(f"PubMed search took {end_time - start_time:.2f} seconds for query: {final_query}")
        return results

    async def search_europepmc(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []

        final_query = query
        if publication_types and self.publication_type_mappings:
            filters = []
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("europepmc")
                if api_specific_value: # e.g., "PUB_TYPE:\"journal-article\""
                    filters.append(api_specific_value)
            if filters:
                filter_term = " OR ".join(filters)
                final_query = f"({query}) AND ({filter_term})"
        
        async with self.europepmc_semaphore:
            try:
                # EuropePMC library might not support complex queries directly in `query` string for pagination.
                # The library's search method takes a query string. We'll append our filter.
                # The EuropePMC API itself supports pageSize. The library might handle it.
                # For now, assume the library handles max_results or we fetch more and slice.
                # The current library `europe_pmc` seems to fetch all and then we might slice.
                # Let's assume it fetches enough and we can take max_results.
                raw_results = await self._run_sync_in_thread(
                    self.epmc.search, # This method in the library might not directly support 'pageSize' or 'retmax'
                    final_query 
                ) # The library might fetch a default number of results.
                
                # Limit results if library doesn't do it.
                processed_results = 0
                for item in raw_results:
                    if processed_results >= max_results:
                        break
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
                    processed_results += 1
            except Exception as e:
                print(f"EuropePMC search error: {e}")
        end_time = time.time()
        print(f"EuropePMC search took {end_time - start_time:.2f} seconds for query: {final_query}")
        return results

    async def search_semanticscholar(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        
        s2_publication_types = []
        if publication_types and self.publication_type_mappings:
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("semanticscholar")
                if api_specific_value: # e.g., "JournalArticle", "Review"
                    s2_publication_types.append(api_specific_value)
        
        # Fields to retrieve, ensure 'publicationTypes' is included if we need to post-filter (though API supports direct filter)
        s2_fields = ['title', 'authors', 'year', 'journal', 'abstract', 'url', 'venue', 
                     'publicationDate', 'externalIds', 'citationCount', 'publicationTypes']

        async with self.semanticscholar_semaphore:
            try:
                print(f"Semantic Scholar: Starting search for '{query}' with limit {max_results}, types: {s2_publication_types}...")
                
                # The `semanticscholar` library's `search_paper` method supports `publication_types` parameter directly.
                # It expects a list of strings, e.g. ['JournalArticle', 'Review']
                raw_results_iterable = await self._run_sync_in_thread(
                    self.s2.search_paper,
                    query=query,
                    limit=max_results, # The library handles pagination to get up to this limit
                    fields=s2_fields,
                    publication_types=s2_publication_types if s2_publication_types else None # Pass None if empty
                )
                # The result of search_paper is an iterator (SearchResult object from the library)
                # We need to iterate it to get Paper objects.
                
                print(f"Semantic Scholar: Finished search for '{query}'. Processing results...")
                
                # The library's search_paper returns an iterator that handles pagination up to `limit`.
                # So, we just iterate through what it gives us.
                for item in raw_results_iterable: # item is a Paper object
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
                    if len(results) >= max_results: # Ensure we don't exceed max_results if library gives more
                        break 
            except Exception as e:
                print(f"Semantic Scholar search error: {e}")
        end_time = time.time()
        print(f"Semantic Scholar search took {end_time - start_time:.2f} seconds for query: {query}, types: {s2_publication_types}")
        return results

    async def search_crossref(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None) -> List[SearchResult]:
        if not self.cr:
            print("CrossRef client not initialized. Skipping CrossRef search.")
            return []
            
        start_time = time.time()
        results: List[SearchResult] = []
        
        crossref_filters = {}
        if publication_types and self.publication_type_mappings:
            mapped_types = []
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("crossref")
                if api_specific_value: # e.g., "journal-article", "review-article"
                    # Quick fix for "review-article" not being valid for CrossRef
                    if api_specific_value == "review-article":
                        mapped_types.append("journal-article")
                    else:
                        mapped_types.append(api_specific_value)
            if mapped_types:
                # For CrossRef, multiple type filters are usually comma-separated for the 'type' filter key
                # Remove duplicates that might arise from mapping "review-article" to "journal-article"
                crossref_filters['type'] = ",".join(list(set(mapped_types)))

        async with self.crossref_semaphore:
            try:
                # The crossref library's query method takes a `filter` dict.
                # Example: .query(bibliographic=query_str, filter={'type': 'journal-article'})
                # The .sample(N) method then takes N samples from the query results.
                
                query_builder = self.cr.query(bibliographic=query)
                if crossref_filters:
                    query_builder = query_builder.filter(**crossref_filters)
                
                # The .sample() method might not be ideal if we want the *most relevant* N results.
                # .sample() gets a random sample.
                # For more controlled fetching, one might use .sort('relevance').limit(max_results)
                # However, the existing code uses .sample(). Let's stick to it for now but be aware.
                # To get a list of items up to max_results, we can iterate the query_builder
                # or use .sample() if that's the intended behavior.
                # Let's try to iterate and limit.
                
                # raw_results_iterable = await self._run_sync_in_thread(
                #     query_builder.sample, # .sample() might not respect sorting or give top N
                #     max_results
                # )
                # Instead of sample, let's try to iterate with a limit.
                # The crossref library query object is an iterator.
                
                # The library itself handles pagination if we iterate.
                # We need to run the iteration part in a thread.
                def fetch_crossref_items():
                    items = []
                    count = 0
                    # The query_builder (WorksQuery object) is iterable.
                    for item_data in query_builder: 
                        items.append(item_data)
                        count += 1
                        if count >= max_results:
                            break
                    return items

                raw_results_list = await self._run_sync_in_thread(fetch_crossref_items)

                for item in raw_results_list: # Iterate over the collected list
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
        print(f"CrossRef search took {end_time - start_time:.2f} seconds for query: {query}, filter: {crossref_filters}")
        return results

    async def search_openalex(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        base_url = "https://api.openalex.org/works"
        
        params = {"mailto": self.email, "per-page": str(max_results)}
        
        filter_parts = []
        if query: # OpenAlex uses specific fields for search, e.g., title.search, default_field.search
            # For a general query, we can use default_field.search or title.search
            # Let's assume 'default.search' for general text query
            filter_parts.append(f"default.search:{query}")

        if publication_types and self.publication_type_mappings:
            mapped_types = []
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("openalex")
                if api_specific_value: # e.g., "journal-article", "review-article"
                    mapped_types.append(api_specific_value)
            if mapped_types:
                # OpenAlex filters are comma-separated for AND, pipe-separated for OR.
                # If we want to match ANY of the provided types, we use OR.
                types_str = "|".join(mapped_types)
                filter_parts.append(f"type:{types_str}")
        
        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        async with self.openalex_semaphore:
            try:
                response = await self.client.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get("results", []):
                    doi = item.get("doi")
                    if doi and doi.startswith("https://doi.org/"):
                        doi = doi.replace("https://doi.org/", "")
                    
                    title = item.get("title")
                    
                    authors = []
                    for authorship in item.get("authorships", []):
                        author = authorship.get("author", {})
                        authors.append(author.get("display_name", ""))
                    
                    year = item.get("publication_year")
                    
                    abstract = None # OpenAlex abstracts are often inverted indexes
                    if item.get("abstract_inverted_index"):
                        # Reconstruct abstract if possible, or leave as None
                        # This is complex, so for now, leave abstract as None from OpenAlex
                        pass

                    journal = item.get("host_venue", {}).get("display_name")
                    pmid = item.get("ids", {}).get("pmid")
                    if pmid and pmid.startswith("https://pubmed.ncbi.nlm.nih.gov/"):
                        pmid = pmid.replace("https://pubmed.ncbi.nlm.nih.gov/", "")
                    pmcid = item.get("ids", {}).get("pmcid")
                    if pmcid and pmcid.startswith("https://www.ncbi.nlm.nih.gov/pmc/articles/"):
                        pmcid = pmcid.replace("https://www.ncbi.nlm.nih.gov/pmc/articles/", "").rstrip('/')


                    url = item.get("doi") # This is the full DOI URL

                    citation_count = item.get("cited_by_count")

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
                        "source_api": "OpenAlex",
                        "sjr_percentile": None,
                        "oa_status": item.get("oa_status"),
                        "citation_count": citation_count
                    }
                    results.append(result_item)
                    if len(results) >= max_results:
                        break
            except httpx.HTTPStatusError as e:
                print(f"OpenAlex HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                print(f"OpenAlex search error: {e}")
        end_time = time.time()
        print(f"OpenAlex search took {end_time - start_time:.2f} seconds for query: {query}, params: {params}")
        return results


    async def search_all(self, pubmed_query: str, general_query: str, 
                         max_results_per_source: int = 20, 
                         publication_types: Optional[List[str]] = None) -> pd.DataFrame:
        start_time = time.time()
        
        tasks = []
        if pubmed_query:
            tasks.append(self.search_pubmed(pubmed_query, max_results_per_source, publication_types))
            tasks.append(self.search_europepmc(pubmed_query, max_results_per_source, publication_types))
        if general_query:
            tasks.append(self.search_semanticscholar(general_query, max_results_per_source, publication_types))
            tasks.append(self.search_crossref(general_query, max_results_per_source, publication_types))
            tasks.append(self.search_openalex(general_query, max_results_per_source, publication_types))


        all_source_results = await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions
        
        flat_results: List[SearchResult] = []
        for i, source_result_list_or_exc in enumerate(all_source_results):
            if isinstance(source_result_list_or_exc, Exception):
                task_name = tasks[i].__qualname__ if hasattr(tasks[i], '__qualname__') else f"Task {i}"
                print(f"Warning: Task {task_name} failed with {type(source_result_list_or_exc).__name__}: {source_result_list_or_exc}")
                continue # Skip this source if it failed
            if source_result_list_or_exc: # Ensure it's not None
                flat_results.extend(source_result_list_or_exc)

        if not flat_results:
            return pd.DataFrame()

        df = pd.DataFrame(flat_results)
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
        print(f"Overall search_all took {end_time - start_time:.2f} seconds. Found {len(df_final)} unique articles with pub_types: {publication_types}.")
        return df_final


class SearchOutput(TypedDict):
    data: List[Dict[str, Any]]
    meta: Dict[str, Any]

async def _async_search_literature(pubmed_query: str, general_query: str,
                                   max_results_per_source: int = 50,
                                   publication_types: Optional[List[str]] = None,
                                   publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None) -> SearchOutput:
    """
    Internal helper that instantiates AsyncSearchClient and returns the structured output
    from client.search_all(). Always awaits client.close().
    """
    # Pass mappings to AsyncSearchClient constructor
    async with AsyncSearchClient(publication_type_mappings=publication_type_mappings) as client:
        df = await client.search_all(
            pubmed_query, 
            general_query, 
            max_results_per_source,
            publication_types=publication_types
        )
    
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
        "publication_types_applied": publication_types if publication_types else [],
        "timestamp": pd.Timestamp.now(tz='UTC').isoformat()
    }

    return {"data": data_records, "meta": meta_info}


def search_literature(pubmed_query: str, general_query: str,
                      max_results_per_source: int = 50,
                      publication_types: Optional[List[str]] = None,
                      publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None) -> SearchOutput:
    """
    Search the biomedical literature via PubMed, Europe PMC, Semantic Scholar, Crossref, and OpenAlex.

    Args:
        pubmed_query: Boolean search string for PubMed and Europe PMC (can use MeSH).
        general_query: Boolean search string for Semantic Scholar, Crossref, and OpenAlex.
        max_results_per_source: Records to pull from each API.
        publication_types: Optional list of types like "research", "review" to filter by.
        publication_type_mappings: Mappings for publication types to API-specific terms.

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
        future = asyncio.ensure_future(_async_search_literature(
            pubmed_query, general_query, max_results_per_source, 
            publication_types, publication_type_mappings
        ))
        return loop.run_until_complete(future)
    else:
        return asyncio.run(_async_search_literature(
            pubmed_query, general_query, max_results_per_source,
            publication_types, publication_type_mappings
        ))

class SearchLiteratureParams(BaseModel):
    pubmed_query: str = Field(..., description="Boolean search string for PubMed and Europe PMC (can use MeSH).")
    general_query: str = Field(..., description="Boolean search string for Semantic Scholar, Crossref, and OpenAlex.")
    max_results_per_source: int = Field(50, description="Records to pull from each API.")
    publication_types: Optional[List[str]] = Field(None, description="Optional list of publication types (e.g. ['research', 'review']) to filter by.")

class LiteratureSearchToolInstanceConfig(BaseModel):
    # This could hold publication_type_mappings if loaded globally for the tool
    publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None

class LiteratureSearchTool(
    BaseTool[SearchLiteratureParams, SearchOutput], 
    Component[LiteratureSearchToolInstanceConfig]
):
    component_config_schema: Type[LiteratureSearchToolInstanceConfig] = LiteratureSearchToolInstanceConfig
    _publication_type_mappings: Optional[Dict[str, Dict[str, str]]]

    def __init__(self, publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None):
        super().__init__(
            args_type=SearchLiteratureParams,
            return_type=SearchOutput, 
            name="search_literature",
            description="High-throughput literature search across PubMed, Europe PMC, Semantic Scholar, Crossref, and OpenAlex."
        )
        self._publication_type_mappings = publication_type_mappings

    @classmethod
    def _from_config(cls, config: LiteratureSearchToolInstanceConfig) -> "LiteratureSearchTool":
        # If mappings are stored in config, pass them here
        return cls(publication_type_mappings=config.publication_type_mappings)

    def _to_config(self) -> LiteratureSearchToolInstanceConfig:
        return LiteratureSearchToolInstanceConfig(publication_type_mappings=self._publication_type_mappings)

    async def run(self, args: SearchLiteratureParams, cancellation_token: Any) -> SearchOutput:
        loop = asyncio.get_running_loop()
        # Pass the tool's mappings to the underlying search_literature function
        return await loop.run_in_executor(
            None, 
            search_literature, 
            args.pubmed_query, 
            args.general_query, 
            args.max_results_per_source,
            args.publication_types,
            self._publication_type_mappings # Pass stored mappings
        )

__all__ = ["search_literature", "LiteratureSearchTool", "SearchLiteratureParams", "SearchResult", "AsyncSearchClient"]

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
    parser.add_argument("--pub_types", type=str, default=None, help="Comma-separated publication types (e.g., research,review).")
    args = parser.parse_args()

    print(f"Searching PubMed/EuropePMC for: {args.pubmed}")
    print(f"Searching general APIs for: {args.general}")
    if args.pub_types:
        print(f"Filtering by publication types: {args.pub_types}")

    # For testing the script directly, we might need to load mappings from a default config path
    # or pass them explicitly if the script were to be more complex.
    # For now, this test call won't use mappings unless search_literature loads them itself.
    # The plan is for LiteratureSearchTool to manage and pass mappings.
    # This __main__ block is primarily for basic testing of search_literature.
    
    publication_types_list = [s.strip() for s in args.pub_types.split(',')] if args.pub_types else None

    # search_literature is synchronous, so we call it directly
    # This direct call won't have publication_type_mappings unless we load them here.
    # For a simple test, we can pass None for mappings.
    search_output_dict = search_literature(
        args.pubmed, 
        args.general, 
        args.max_results,
        publication_types=publication_types_list,
        publication_type_mappings=None # In a real scenario, these would be loaded
    )
    
    if search_output_dict and search_output_dict.get('data'):
        df = pd.DataFrame(search_output_dict['data'])
        rich.print(df.head())
    else:
        rich.print("[bold red]No data returned from search_literature.[/bold red]")
