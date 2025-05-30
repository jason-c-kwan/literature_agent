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

    async def search_pubmed(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        print(f"DEBUG: search_pubmed - Starting for query: {query}, max_results: {max_results}, pub_types: {publication_types}, start: {start_date}, end: {end_date}")
        start_time = time.time()
        results: List[SearchResult] = []
        
        final_query = query
        # Apply publication type filters
        if publication_types and self.publication_type_mappings:
            filters = []
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("pubmed")
                if api_specific_value:
                    filters.append(api_specific_value)
            if filters:
                filter_term = " OR ".join(filters)
                final_query = f"({final_query}) AND ({filter_term})"

        # Prepare Entrez esearch parameters
        entrez_params = {
            "db": "pubmed",
            "term": final_query,
            "retmax": str(max_results),
            "usehistory": "y"
        }

        # Apply date filters (YYYY/MM/DD format for PubMed)
        if start_date:
            try:
                # Validate and reformat if necessary, assuming YYYY-MM-DD input
                pd_start_date = pd.to_datetime(start_date)
                entrez_params["mindate"] = pd_start_date.strftime('%Y/%m/%d')
            except ValueError:
                print(f"Warning: Invalid start_date format for PubMed: {start_date}. Expected YYYY-MM-DD.")
        if end_date:
            try:
                pd_end_date = pd.to_datetime(end_date)
                entrez_params["maxdate"] = pd_end_date.strftime('%Y/%m/%d')
            except ValueError:
                print(f"Warning: Invalid end_date format for PubMed: {end_date}. Expected YYYY-MM-DD.")

        async with self.pubmed_semaphore:
            try:
                print(f"DEBUG: search_pubmed - Calling Entrez.esearch with term: {entrez_params['term']}")
                handle = await self._run_sync_in_thread(
                    Entrez.esearch,
                    **entrez_params
                )
                search_results_handle = Entrez.read(handle) 
                handle.close()
                ids = search_results_handle["IdList"]
                print(f"DEBUG: search_pubmed - Entrez.esearch completed. Number of IDs: {len(ids)}")

                if not ids:
                    print("DEBUG: search_pubmed - No IDs found, returning early.")
                    return []

                print(f"DEBUG: search_pubmed - Calling Entrez.efetch for {len(ids)} IDs.")
                fetch_handle = await self._run_sync_in_thread(
                    Entrez.efetch,
                    db="pubmed",
                    id=ids,
                    rettype="abstract", 
                    retmode="xml"      
                )
                articles_xml = Entrez.read(fetch_handle)
                fetch_handle.close()
                print(f"DEBUG: search_pubmed - Entrez.efetch completed. articles_xml type: {type(articles_xml)}")
                if isinstance(articles_xml, dict):
                    print(f"DEBUG: search_pubmed - articles_xml keys: {list(articles_xml.keys())}, PubmedArticle count: {len(articles_xml.get('PubmedArticle', []))}")
                elif isinstance(articles_xml, list): # Entrez.read can return a list if multiple records
                    print(f"DEBUG: search_pubmed - articles_xml is a list, count: {len(articles_xml)}")


                print("DEBUG: search_pubmed - Starting loop through fetched articles.")
                article_count = 0
                # Ensure articles_xml is iterable and contains PubmedArticle entries
                articles_to_process = []
                if isinstance(articles_xml, dict) and 'PubmedArticle' in articles_xml:
                    articles_to_process = articles_xml.get('PubmedArticle', [])
                elif isinstance(articles_xml, list): # If efetch returns a list of articles directly
                    articles_to_process = articles_xml

                for article_xml_entry in articles_to_process:
                    article_count += 1
                    if article_count % 5 == 0:
                        print(f"DEBUG: search_pubmed - Processing article {article_count} in loop.")
                    
                    # Defensive coding: ensure article_xml_entry is a dict
                    if not isinstance(article_xml_entry, dict):
                        print(f"DEBUG: search_pubmed - Skipping non-dict item in articles_to_process at count {article_count}")
                        continue

                    medline_citation = article_xml_entry.get('MedlineCitation', {})
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
                    pubmed_data = article_xml_entry.get('PubmedData', {})
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
                        "oa_status": None,      
                        "citation_count": None  
                    }
                    results.append(result_item)
                print(f"DEBUG: search_pubmed - Finished loop. Processed {article_count} articles.")
            except Exception as e:
                print(f"PubMed search error: {e}")
        end_time = time.time() 
        print(f"PubMed search took {end_time - start_time:.2f} seconds for query: {entrez_params['term']}")
        return results

    async def search_europepmc(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []

        final_query_parts = [query]
        
        # Apply publication type filters
        if publication_types and self.publication_type_mappings:
            type_filters = []
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("europepmc")
                if api_specific_value:
                    type_filters.append(api_specific_value)
            if type_filters:
                final_query_parts.append(f"({' OR '.join(type_filters)})")

        # Apply date filters (YYYY-MM-DD format for EuropePMC)
        # Example: creationDate:[2020-01-01 TO 2020-12-31]
        date_filter_str = ""
        if start_date and end_date:
            date_filter_str = f"firstPublicationDate:[{start_date} TO {end_date}]"
        elif start_date:
            date_filter_str = f"firstPublicationDate:[{start_date} TO *]"
        elif end_date:
            date_filter_str = f"firstPublicationDate:[* TO {end_date}]"
        
        if date_filter_str:
            final_query_parts.append(date_filter_str)
            
        final_query = " AND ".join(f"({part})" for part in final_query_parts if part)

        async with self.europepmc_semaphore:
            try:
                raw_results = await self._run_sync_in_thread(
                    self.epmc.search,
                    final_query
                )
                
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

    async def search_semanticscholar(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        
        s2_publication_types = []
        if publication_types and self.publication_type_mappings:
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("semanticscholar")
                if api_specific_value:
                    s2_publication_types.append(api_specific_value)
        
        s2_fields = ['title', 'authors', 'year', 'journal', 'abstract', 'url', 'venue', 
                     'publicationDate', 'externalIds', 'citationCount', 'publicationTypes']
        
        # Semantic Scholar uses 'year' parameter, which can be YYYY or YYYY-YYYY
        year_filter = None
        if start_date and end_date:
            start_year = start_date[:4]
            end_year = end_date[:4]
            if start_year == end_year:
                year_filter = start_year
            else:
                year_filter = f"{start_year}-{end_year}"
        elif start_date:
            year_filter = start_date[:4] # Search from start_year onwards (S2 might not support open-ended start)
                                         # Or search for that specific year if that's the S2 behavior for single year
        elif end_date:
            year_filter = end_date[:4]   # Search up to end_year (S2 might not support open-ended end)

        search_params_s2 = {
            "query": query,
            "limit": max_results,
            "fields": s2_fields,
            "publication_types": s2_publication_types if s2_publication_types else None
        }
        if year_filter:
            search_params_s2["year"] = year_filter

        async with self.semanticscholar_semaphore:
            try:
                print(f"Semantic Scholar: Starting search with params: {search_params_s2}...")
                
                raw_results_iterable = await self._run_sync_in_thread(
                    self.s2.search_paper,
                    **search_params_s2
                )
                
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
        print(f"Semantic Scholar search took {end_time - start_time:.2f} seconds for query: {query}, params: {search_params_s2}")
        return results

    async def search_crossref(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        if not self.cr:
            print("CrossRef client not initialized. Skipping CrossRef search.")
            return []
            
        start_time = time.time()
        results: List[SearchResult] = []
        
        # Prepare filters for CrossRef
        crossref_filters: Dict[str, str] = {} # Ensure it's Dict[str, str] for .filter(**crossref_filters)

        # Publication type filters
        if publication_types and self.publication_type_mappings:
            mapped_types = []
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("crossref")
                if api_specific_value:
                    if api_specific_value == "review-article": # map to journal-article for CrossRef
                        mapped_types.append("journal-article")
                    else:
                        mapped_types.append(api_specific_value)
            if mapped_types:
                crossref_filters['type'] = ",".join(list(set(mapped_types)))
        
        # Date filters (YYYY-MM-DD format for CrossRef)
        if start_date:
            crossref_filters['from-pub-date'] = start_date
        if end_date:
            crossref_filters['until-pub-date'] = end_date

        async with self.crossref_semaphore:
            try:
                query_builder = self.cr.query(bibliographic=query)
                if crossref_filters: # Only apply if filters exist
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
        print(f"CrossRef search took {end_time - start_time:.2f} seconds for query: {query}, filter(s): {crossref_filters}")
        return results

    async def search_openalex(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        start_time = time.time()
        results: List[SearchResult] = []
        base_url = "https://api.openalex.org/works"
        
        # Initialize params for OpenAlex
        openalex_params: Dict[str, str] = {"mailto": self.email, "per-page": str(max_results)}
        
        filter_parts = []
        if query:
            filter_parts.append(f"default.search:{query}")

        # Publication type filters
        if publication_types and self.publication_type_mappings:
            mapped_types = []
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("openalex")
                if api_specific_value:
                    mapped_types.append(api_specific_value)
            if mapped_types:
                types_str = "|".join(mapped_types) # OR logic for types
                filter_parts.append(f"type:{types_str}")
        
        # Date filters (YYYY-MM-DD format for OpenAlex)
        if start_date:
            filter_parts.append(f"from_publication_date:{start_date}")
        if end_date:
            filter_parts.append(f"to_publication_date:{end_date}")
        
        if filter_parts:
            openalex_params["filter"] = ",".join(filter_parts) # Comma-separated for AND logic between filter fields

        async with self.openalex_semaphore:
            try:
                response = await self.client.get(base_url, params=openalex_params)
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
        print(f"OpenAlex search took {end_time - start_time:.2f} seconds for query: {query}, params: {openalex_params}")
        return results


    async def search_all(self, pubmed_query: str, general_query: str, 
                         max_results_per_source: int = 20, 
                         publication_types: Optional[List[str]] = None,
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> pd.DataFrame:
        start_time = time.time()
        
        tasks = []
        if pubmed_query:
            tasks.append(self.search_pubmed(pubmed_query, max_results_per_source, publication_types, start_date, end_date))
            tasks.append(self.search_europepmc(pubmed_query, max_results_per_source, publication_types, start_date, end_date))
        if general_query:
            tasks.append(self.search_semanticscholar(general_query, max_results_per_source, publication_types, start_date, end_date))
            tasks.append(self.search_crossref(general_query, max_results_per_source, publication_types, start_date, end_date))
            tasks.append(self.search_openalex(general_query, max_results_per_source, publication_types, start_date, end_date))

        print("DEBUG: search_all - About to await asyncio.gather for all search tasks...")
        all_source_results = await asyncio.gather(*tasks, return_exceptions=True) 
        print(f"DEBUG: search_all - asyncio.gather completed. Number of results/exceptions: {len(all_source_results)}")
        
        flat_results: List[SearchResult] = []
        for i, source_result_list_or_exc in enumerate(all_source_results):
            if isinstance(source_result_list_or_exc, Exception):
                task_name = tasks[i].__qualname__ if hasattr(tasks[i], '__qualname__') else f"Task {i}" # type: ignore
                print(f"Warning: Task {task_name} failed with {type(source_result_list_or_exc).__name__}: {source_result_list_or_exc}")
                continue 
            if source_result_list_or_exc: 
                flat_results.extend(source_result_list_or_exc)

        if not flat_results:
            print("DEBUG: search_all - flat_results is empty after processing asyncio.gather results. Returning empty DataFrame.")
            return pd.DataFrame()

        print(f"DEBUG: search_all - Initial flat_results length: {len(flat_results)}")
        df = pd.DataFrame(flat_results)
        print(f"DEBUG: search_all - DataFrame created from flat_results. Shape: {df.shape}")
        # print(f"DEBUG: search_all - df.info() before deduplication:\n{df.info(verbose=True, show_counts=True)}") # Can be too verbose

        # Corrected: Removed duplicated extension of flat_results and recreation of df.
        # The first creation of df from flat_results is the correct one.

        if 'doi' in df.columns:
            df['doi_norm'] = df['doi'].apply(lambda x: str(x).lower().replace("https://doi.org/", "").strip() if pd.notna(x) and x != '' else pd.NA)
        else:
            df['doi_norm'] = pd.NA
        print(f"DEBUG: search_all - 'doi_norm' column created.")
        
        if 'pmid' in df.columns:
            df['pmid_str'] = df['pmid'].apply(lambda x: str(x) if pd.notna(x) and x != '' else pd.NA)
        else:
            df['pmid_str'] = pd.NA
        print(f"DEBUG: search_all - 'pmid_str' column created.")

        mask_doi_present = df['doi_norm'].notna()
        
        df_with_doi_present = df[mask_doi_present].copy()
        df_no_doi_present = df[~mask_doi_present].copy()
        print(f"DEBUG: search_all - df_with_doi_present shape: {df_with_doi_present.shape}, df_no_doi_present shape: {df_no_doi_present.shape}")
        
        df_with_doi_deduped = pd.DataFrame()
        if not df_with_doi_present.empty:
            print(f"DEBUG: search_all - Processing df_with_doi_present...")
            df_with_doi_present['has_abstract'] = df_with_doi_present['abstract'].notna() & (df_with_doi_present['abstract'] != '')
            print(f"DEBUG: search_all - Sorting df_with_doi_present by ['doi_norm', 'has_abstract', 'year', 'pmid_str']...")
            df_with_doi_present = df_with_doi_present.sort_values(
                by=['doi_norm', 'has_abstract', 'year', 'pmid_str'], 
                ascending=[True, False, False, True], 
                na_position='last'
            )
            print(f"DEBUG: search_all - Dropping duplicates from df_with_doi_present on 'doi_norm'...")
            df_with_doi_deduped = df_with_doi_present.drop_duplicates(subset=['doi_norm'], keep='first')
            df_with_doi_deduped = df_with_doi_deduped.drop(columns=['has_abstract'], errors='ignore')
            print(f"DEBUG: search_all - df_with_doi_deduped shape: {df_with_doi_deduped.shape}")

        df_no_doi_deduped = pd.DataFrame()
        if not df_no_doi_present.empty:
            print(f"DEBUG: search_all - Processing df_no_doi_present...")
            df_no_doi_present['has_abstract'] = df_no_doi_present['abstract'].notna() & (df_no_doi_present['abstract'] != '')
            print(f"DEBUG: search_all - Sorting df_no_doi_present by ['title', 'has_abstract', 'year', 'pmid_str']...")
            df_no_doi_present = df_no_doi_present.sort_values(
                by=['title', 'has_abstract', 'year', 'pmid_str'], 
                ascending=[True, False, False, True], 
                na_position='last'
            )
            use_pmid_for_no_doi_dedup = 'pmid_str' in df_no_doi_present.columns and df_no_doi_present['pmid_str'].notna().any()
            subset_for_no_doi = ['pmid_str'] if use_pmid_for_no_doi_dedup else ['title', 'year']
            print(f"DEBUG: search_all - Dropping duplicates from df_no_doi_present on {subset_for_no_doi}...")
            df_no_doi_deduped = df_no_doi_present.drop_duplicates(subset=subset_for_no_doi, keep='first')
            df_no_doi_deduped = df_no_doi_deduped.drop(columns=['has_abstract'], errors='ignore')
            print(f"DEBUG: search_all - df_no_doi_deduped shape: {df_no_doi_deduped.shape}")
        
        print(f"DEBUG: search_all - Concatenating deduped DataFrames...")
        df_final = pd.concat([df_with_doi_deduped, df_no_doi_deduped], ignore_index=True)
        print(f"DEBUG: search_all - df_final shape after concat: {df_final.shape}")

        cols_to_drop = [col for col in ['doi_norm', 'pmid_str'] if col in df_final.columns]
        if cols_to_drop:
            df_final.drop(columns=cols_to_drop, inplace=True)
        print(f"DEBUG: search_all - Dropped helper columns. Current df_final shape: {df_final.shape}")
        
        expected_df_cols = list(SearchResult.__annotations__.keys())
        for col in expected_df_cols:
            if col not in df_final.columns:
                df_final[col] = pd.NA 
        
        df_final = df_final[expected_df_cols]
        print(f"DEBUG: search_all - Ensured all expected columns and reordered. Current df_final shape: {df_final.shape}")
        
        print(f"DEBUG: search_all - Sorting final DataFrame by ['year', 'title']...")
        df_final = df_final.sort_values(by=['year', 'title'], ascending=[False, True], na_position='last').reset_index(drop=True)
        print(f"DEBUG: search_all - Final sort complete. Final df_final shape: {df_final.shape}")

        end_time = time.time()
        print(f"Overall search_all took {end_time - start_time:.2f} seconds. Found {len(df_final)} unique articles. Pub_types: {publication_types}, Start: {start_date}, End: {end_date}.")
        return df_final


class SearchOutput(TypedDict):
    data: List[Dict[str, Any]]
    meta: Dict[str, Any]

async def _async_search_literature(pubmed_query: str, general_query: str,
                                   max_results_per_source: int = 50,
                                   publication_types: Optional[List[str]] = None,
                                   publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> SearchOutput:
    """
    Internal helper that instantiates AsyncSearchClient and returns the structured output
    from client.search_all(). Always awaits client.close().
    """
    async with AsyncSearchClient(publication_type_mappings=publication_type_mappings) as client:
        df = await client.search_all(
            pubmed_query, 
            general_query, 
            max_results_per_source,
            publication_types=publication_types,
            start_date=start_date,
            end_date=end_date
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
        "start_date_applied": start_date,
        "end_date_applied": end_date,
        "timestamp": pd.Timestamp.now(tz='UTC').isoformat()
    }

    return {"data": data_records, "meta": meta_info}


def search_literature(pubmed_query: str, general_query: str,
                      max_results_per_source: int = 50,
                      publication_types: Optional[List[str]] = None,
                      publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> SearchOutput:
    """
    Search the biomedical literature via PubMed, Europe PMC, Semantic Scholar, Crossref, and OpenAlex.

    Args:
        pubmed_query: Boolean search string for PubMed and Europe PMC (can use MeSH).
        general_query: Boolean search string for Semantic Scholar, Crossref, and OpenAlex.
        max_results_per_source: Records to pull from each API.
        publication_types: Optional list of types like "research", "review" to filter by.
        publication_type_mappings: Mappings for publication types to API-specific terms.
        start_date: Optional start date for search range (YYYY-MM-DD).
        end_date: Optional end date for search range (YYYY-MM-DD).

    Returns:
        A dictionary with 'data' and 'meta' fields.
        'data' is a list of dictionaries, each with 'doi', 'title', 'abstract'.
        'meta' contains 'total_hits', 'query', 'timestamp', and applied filters.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        future = asyncio.ensure_future(_async_search_literature(
            pubmed_query, general_query, max_results_per_source, 
            publication_types, publication_type_mappings,
            start_date, end_date
        ))
        return loop.run_until_complete(future)
    else:
        return asyncio.run(_async_search_literature(
            pubmed_query, general_query, max_results_per_source,
            publication_types, publication_type_mappings,
            start_date, end_date
        ))

class SearchLiteratureParams(BaseModel):
    pubmed_query: str = Field(..., description="Boolean search string for PubMed and Europe PMC (can use MeSH).")
    general_query: str = Field(..., description="Boolean search string for Semantic Scholar, Crossref, and OpenAlex.")
    max_results_per_source: int = Field(50, description="Records to pull from each API.")
    publication_types: Optional[List[str]] = Field(None, description="Optional list of publication types (e.g. ['research', 'review']) to filter by.")
    start_date: Optional[str] = Field(None, description="Start date for search range (YYYY-MM-DD).")
    end_date: Optional[str] = Field(None, description="End date for search range (YYYY-MM-DD).")

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
        return await loop.run_in_executor(
            None, 
            search_literature, 
            args.pubmed_query, 
            args.general_query, 
            args.max_results_per_source,
            args.publication_types,
            self._publication_type_mappings, # Pass stored mappings
            args.start_date,
            args.end_date
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
