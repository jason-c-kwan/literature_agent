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
from semanticscholar import SemanticScholar, AsyncSemanticScholar
from crossref.restful import Etiquette
from autogen_core.tools import BaseTool
from autogen_core._component_config import Component
from pydantic import BaseModel, Field
# import pdb # Removing pdb import as it might have been added by user locally

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
        
        pubmed_rps = 2 if self.pubmed_api_key else 1
        self.pubmed_semaphore = asyncio.Semaphore(pubmed_rps)
        self.europepmc_semaphore = asyncio.Semaphore(5)
        self.semanticscholar_semaphore = asyncio.Semaphore(1)
        self.crossref_semaphore = asyncio.Semaphore(50)
        self.openalex_semaphore = asyncio.Semaphore(10)
        
        Entrez.email = self.email
        if self.pubmed_api_key:
            Entrez.api_key = self.pubmed_api_key

        self.epmc = EuropePMC()
        self.s2_async = AsyncSemanticScholar(
            timeout=20,
            api_key=self.semantic_scholar_api_key if self.semantic_scholar_api_key else None,
            retry=True
        )
        
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def search_pubmed(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        print(f"DEBUG: search_pubmed - Starting for query: {query}, max_results: {max_results}, pub_types: {publication_types}, start: {start_date}, end: {end_date}")
        start_time = time.time()
        results: List[SearchResult] = []
        final_query = query
        if publication_types and self.publication_type_mappings:
            filters = []
            for pub_type_key in publication_types:
                api_specific_value = self.publication_type_mappings.get(pub_type_key, {}).get("pubmed")
                if api_specific_value:
                    filters.append(api_specific_value)
            if filters:
                final_query = f"({final_query}) AND ({' OR '.join(filters)})"
        
        entrez_params = {
            "db": "pubmed",
            "term": final_query,
            "retmax": str(max_results),
            "usehistory": "y"
        }
        if start_date:
            try:
                entrez_params["mindate"] = pd.to_datetime(start_date).strftime('%Y/%m/%d')
            except ValueError:
                print(f"Warning: Invalid start_date for PubMed: {start_date}.")
        if end_date:
            try:
                entrez_params["maxdate"] = pd.to_datetime(end_date).strftime('%Y/%m/%d')
            except ValueError:
                print(f"Warning: Invalid end_date for PubMed: {end_date}.")

        async with self.pubmed_semaphore:
            try:
                print(f"DEBUG: search_pubmed - Calling Entrez.esearch with term: {entrez_params['term']}")
                handle = await self._run_sync_in_thread(Entrez.esearch, **entrez_params)
                search_results_handle = Entrez.read(handle)
                handle.close()
                ids = search_results_handle["IdList"]
                print(f"DEBUG: search_pubmed - Entrez.esearch completed. Number of IDs: {len(ids)}")
                if not ids:
                    print("DEBUG: search_pubmed - No IDs found, returning early.")
                    return []
                
                print(f"DEBUG: search_pubmed - Calling Entrez.efetch for {len(ids)} IDs.")
                fetch_handle = await self._run_sync_in_thread(Entrez.efetch, db="pubmed", id=ids, rettype="abstract", retmode="xml")
                articles_xml = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                articles_to_process = []
                if isinstance(articles_xml, dict) and 'PubmedArticle' in articles_xml:
                    articles_to_process = articles_xml.get('PubmedArticle', [])
                elif isinstance(articles_xml, list):
                    articles_to_process = articles_xml
                
                for article_xml_entry in articles_to_process:
                    if not isinstance(article_xml_entry, dict):
                        continue
                    
                    medline_citation = article_xml_entry.get('MedlineCitation', {})
                    article_info = medline_citation.get('Article', {})
                    pmid = str(medline_citation.get('PMID', ''))
                    title = article_info.get('ArticleTitle', 'N/A')
                    
                    abstract_parts = article_info.get('Abstract', {}).get('AbstractText', [])
                    abstract = ""
                    if isinstance(abstract_parts, list):
                        abstract = "\n".join([str(part) for part in abstract_parts if isinstance(part, str)])
                    elif isinstance(abstract_parts, str):
                        abstract = abstract_parts
                    
                    authors_list = article_info.get('AuthorList', [])
                    authors = []
                    for author_entry in authors_list:
                        if isinstance(author_entry, dict):
                            authors.append(f"{author_entry.get('ForeName', '')} {author_entry.get('LastName', '')}".strip())

                    journal_title = article_info.get('Journal', {}).get('Title', '')
                    pub_date_info = article_info.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                    year_str = pub_date_info.get('Year')
                    year = None
                    if year_str and isinstance(year_str, str) and year_str.isdigit():
                        year = int(year_str)
                    else:
                        medline_date = pub_date_info.get('MedlineDate', '')
                        if isinstance(medline_date, str) and len(medline_date) >= 4 and medline_date[:4].isdigit():
                            year = int(medline_date[:4])
                    
                    doi = None
                    pmcid = None
                    elocation_ids = article_info.get('ELocationID', [])
                    if not isinstance(elocation_ids, list): 
                        elocation_ids = [elocation_ids] if elocation_ids else []
                    for elid in elocation_ids:
                        if hasattr(elid, 'attributes') and elid.attributes.get('EIdType') == 'doi':
                            doi = str(elid)
                            break 
                            
                    pubmed_data_ids = article_xml_entry.get('PubmedData', {}).get('ArticleIdList', [])
                    for aid_obj in pubmed_data_ids:
                        if hasattr(aid_obj, 'attributes'):
                            id_type = aid_obj.attributes.get('IdType')
                            if id_type == 'doi' and not doi:
                                doi = str(aid_obj)
                            elif id_type == 'pmc':
                                pmcid = str(aid_obj)
                                
                    results.append({
                        "title": title, "authors": authors, "doi": doi, "pmid": pmid, "pmcid": pmcid,
                        "year": year, "abstract": abstract or None, "journal": journal_title,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{doi}" if doi else None),
                        "source_api": "PubMed", "sjr_percentile": None, "oa_status": None, "citation_count": None
                    })
            except Exception as e:
                print(f"PubMed search error: {e}")
        end_time = time.time()
        print(f"PubMed search took {end_time - start_time:.2f} seconds for query: {entrez_params['term']}")
        return results

    async def search_europepmc(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        start_time = time.time(); results: List[SearchResult] = []
        final_query_parts = [query]
        if publication_types and self.publication_type_mappings:
            type_filters = [v for k in publication_types if (v := self.publication_type_mappings.get(k, {}).get("europepmc"))]
            if type_filters: final_query_parts.append(f"({' OR '.join(type_filters)})")
        date_filter_str = ""
        if start_date and end_date: date_filter_str = f"firstPublicationDate:[{start_date} TO {end_date}]"
        elif start_date: date_filter_str = f"firstPublicationDate:[{start_date} TO *]"
        elif end_date: date_filter_str = f"firstPublicationDate:[* TO {end_date}]"
        if date_filter_str: final_query_parts.append(date_filter_str)
        final_query = " AND ".join(f"({part})" for part in final_query_parts if part)
        async with self.europepmc_semaphore:
            try:
                raw_results = await self._run_sync_in_thread(self.epmc.search, final_query)
                for i, item in enumerate(raw_results):
                    if i >= max_results: break
                    doi = getattr(item, 'doi', None); pmid = getattr(item, 'pmid', None); title = getattr(item, 'title', None)
                    authors = [au.fullName for au in getattr(item, 'authorList', []) if hasattr(au, 'fullName')] if getattr(item, 'authorList', None) else []
                    journal = getattr(item, 'journalTitle', None); year_str = getattr(item, 'pubYear', None); year = int(year_str) if year_str and year_str.isdigit() else None
                    abstract = getattr(item, 'abstractText', None); pmcid = getattr(item, 'pmcid', None)
                    url = f"https://doi.org/{doi}" if doi else (f"https://europepmc.org/article/MED/{pmid}" if pmid else (f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else None))
                    results.append({"title": title, "authors": authors, "doi": doi, "pmid": pmid, "pmcid": pmcid, "year": year, "abstract": abstract, "journal": journal, "url": url, "source_api": "EuropePMC", "sjr_percentile": None, "oa_status": None, "citation_count": getattr(item, 'citedByCount', None)})
            except Exception as e: print(f"EuropePMC search error: {e}")
        print(f"EuropePMC search took {time.time() - start_time:.2f} seconds for query: {final_query}")
        return results

    async def search_semanticscholar(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        start_time = time.time(); results: List[SearchResult] = []
        s2_publication_types = [v for k in publication_types or [] if (v := self.publication_type_mappings.get(k, {}).get("semanticscholar"))]
        s2_fields = ['title', 'authors', 'year', 'journal', 'abstract', 'url', 'venue', 'publicationDate', 'externalIds', 'citationCount', 'publicationTypes']
        year_filter = None
        if start_date and end_date: 
            start_y, end_y = start_date[:4], end_date[:4]
            year_filter = start_y if start_y == end_y else f"{start_y}-{end_y}"
        elif start_date: year_filter = start_date[:4]
        elif end_date: year_filter = end_date[:4]
        search_params_s2 = {"query": query, "limit": max_results, "fields": s2_fields, "publication_types": s2_publication_types or None}
        if year_filter: search_params_s2["year"] = year_filter
        
        async with self.semanticscholar_semaphore:
            try:
                print(f"Semantic Scholar (async): Starting search with params: {search_params_s2}...")
                paginated_results_obj = await self.s2_async.search_paper(**search_params_s2)
                print(f"Semantic Scholar (async): PaginatedResults object received. Iterating asynchronously...")
                temp_results = []
                item_count = 0
                async for item in paginated_results_obj: 
                    item_count += 1
                    title = getattr(item, 'title', None)
                    authors_data = getattr(item, 'authors', [])
                    authors = [au['name'] for au in authors_data if isinstance(au, dict) and 'name' in au]
                    year = getattr(item, 'year', None)
                    journal_data = getattr(item, 'journal', None)
                    journal_name = None
                    if isinstance(journal_data, dict):
                        journal_name = journal_data.get('name')
                    elif isinstance(journal_data, str):
                        journal_name = journal_data
                    
                    external_ids = getattr(item, 'externalIds', {})
                    doi = external_ids.get('DOI')
                    pmid = external_ids.get('PubMed')
                    pmcid = external_ids.get('PubMedCentral')
                    abstract = getattr(item, 'abstract', None)
                    url = getattr(item, 'url', None)
                    citation_count = getattr(item, 'citationCount', None)
                    
                    result_item: SearchResult = {
                        "title": title, "authors": authors, "doi": doi, "pmid": pmid, "pmcid": pmcid, 
                        "year": year, "abstract": abstract, "journal": journal_name, "url": url, 
                        "source_api": "SemanticScholar", "sjr_percentile": None, 
                        "oa_status": None, "citation_count": citation_count
                    }
                    temp_results.append(result_item)
                    if len(temp_results) >= max_results:
                        print(f"Semantic Scholar (async): Reached max_results limit of {max_results}.")
                        break 
                results.extend(temp_results)
                print(f"Semantic Scholar (async): Finished iteration. Processed {len(temp_results)} items.")
            except Exception as e: 
                print(f"Semantic Scholar (async) search error: {type(e).__name__} - {e}")
        print(f"Semantic Scholar search took {time.time() - start_time:.2f} seconds for query: {query}, params: {search_params_s2}")
        return results

    async def search_crossref(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        if not self.cr: print("CrossRef client not initialized."); return []
        start_time = time.time(); results: List[SearchResult] = []
        crossref_filters: Dict[str, str] = {}
        if publication_types and self.publication_type_mappings:
            mapped_types = [v if v != "review-article" else "journal-article" for k in publication_types if (v := self.publication_type_mappings.get(k, {}).get("crossref"))]
            if mapped_types: crossref_filters['type'] = ",".join(list(set(mapped_types)))
        if start_date: crossref_filters['from-pub-date'] = start_date
        if end_date: crossref_filters['until-pub-date'] = end_date
        async with self.crossref_semaphore:
            try:
                query_builder = self.cr.query(bibliographic=query)
                if crossref_filters: query_builder = query_builder.filter(**crossref_filters)
                def fetch_items():
                    items = []; count = 0
                    for item_data in query_builder: 
                        items.append(item_data)
                        count +=1
                        if count >= max_results: break
                    return items
                raw_list = await self._run_sync_in_thread(fetch_items)
                for item in raw_list:
                    if not isinstance(item, dict): continue
                    doi = item.get("DOI"); title = item.get("title", [None])[0]
                    authors = [" ".join(filter(None, [au.get('given'), au.get('family')])) for au in item.get("author", [])]
                    issued = item.get("issued", {}).get("date-parts", [[]])
                    year_val = issued[0][0] if issued and issued[0] and issued[0][0] is not None else None
                    year = int(year_val) if year_val is not None else None
                    journal = item.get("container-title", [None])[0]; url = item.get("URL"); citation_count = item.get('is-referenced-by-count')
                    results.append({"title": title, "authors": authors, "doi": doi, "year": year, "journal": journal, "url": url, "source_api": "CrossRef", "sjr_percentile": None, "oa_status": None, "citation_count": citation_count, "abstract": None, "pmid": None, "pmcid": None})
            except Exception as e: print(f"CrossRef search error: {e}")
        print(f"CrossRef search took {time.time() - start_time:.2f} seconds for query: {query}, filter(s): {crossref_filters}")
        return results

    async def search_openalex(self, query: str, max_results: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchResult]:
        start_time = time.time(); results: List[SearchResult] = []
        base_url = "https://api.openalex.org/works"; params: Dict[str, str] = {"mailto": self.email, "per-page": str(max_results)}
        filter_parts = [f"default.search:{query}"] if query else []
        if publication_types and self.publication_type_mappings:
            mapped_types = [v for k in publication_types if (v := self.publication_type_mappings.get(k, {}).get("openalex"))]
            if mapped_types: filter_parts.append(f"type:{'|'.join(mapped_types)}")
        if start_date: filter_parts.append(f"from_publication_date:{start_date}")
        if end_date: filter_parts.append(f"to_publication_date:{end_date}")
        if filter_parts: params["filter"] = ",".join(filter_parts)
        async with self.openalex_semaphore:
            try:
                response = await self.client.get(base_url, params=params); response.raise_for_status(); data = response.json()
                for item in data.get("results", []):
                    if len(results) >= max_results: break
                    doi = item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else None
                    title = item.get("title"); authors = [au.get("author", {}).get("display_name", "") for au in item.get("authorships", [])]
                    year = item.get("publication_year"); journal = item.get("host_venue", {}).get("display_name")
                    pmid = item.get("ids", {}).get("pmid", "").replace("https://pubmed.ncbi.nlm.nih.gov/", "") if item.get("ids", {}).get("pmid") else None
                    pmcid = item.get("ids", {}).get("pmcid", "").replace("https://www.ncbi.nlm.nih.gov/pmc/articles/", "").rstrip('/') if item.get("ids", {}).get("pmcid") else None
                    url = item.get("doi"); citation_count = item.get("cited_by_count")
                    results.append({"title": title, "authors": authors, "doi": doi, "pmid": pmid, "pmcid": pmcid, "year": year, "abstract": None, "journal": journal, "url": url, "source_api": "OpenAlex", "sjr_percentile": None, "oa_status": item.get("oa_status"), "citation_count": citation_count})
            except Exception as e: print(f"OpenAlex search error: {e}")
        print(f"OpenAlex search took {time.time() - start_time:.2f} seconds for query: {query}, params: {params}")
        return results

    async def search_all(self, pubmed_query: str, general_query: str, max_results_per_source: int = 20, publication_types: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        start_time = time.time(); tasks = []
        if pubmed_query:
            tasks.append(self.search_pubmed(pubmed_query, max_results_per_source, publication_types, start_date, end_date))
            tasks.append(self.search_europepmc(pubmed_query, max_results_per_source, publication_types, start_date, end_date))
        if general_query:
            # s2_task_coro = self.search_semanticscholar(general_query, max_results_per_source, publication_types, start_date, end_date)
            # tasks.append(asyncio.wait_for(s2_task_coro, timeout=90.0)) # Temporarily disable S2 search
            tasks.append(self.search_crossref(general_query, max_results_per_source, publication_types, start_date, end_date))
            tasks.append(self.search_openalex(general_query, max_results_per_source, publication_types, start_date, end_date))
        print("DEBUG: search_all - About to await asyncio.gather for all search tasks...")
        all_source_results = await asyncio.gather(*tasks, return_exceptions=True)
        print("DEBUG: search_all - asyncio.gather call has successfully RETURNED.")
        print(f"DEBUG: search_all - asyncio.gather completed. Number of results/exceptions: {len(all_source_results)}")
        flat_results: List[SearchResult] = []
        for i, res_or_exc in enumerate(all_source_results):
            if isinstance(res_or_exc, asyncio.TimeoutError):
                print(f"Warning: A search task (likely Semantic Scholar) failed with asyncio.TimeoutError after 90s.")
                continue
            elif isinstance(res_or_exc, Exception):
                task_obj = tasks[i]; coro_name = "UnknownTask"
                actual_coro = getattr(task_obj, '_coro', None) 
                if not actual_coro and hasattr(task_obj, 'get_coro'): 
                    actual_coro = task_obj.get_coro()
                if not actual_coro and asyncio.iscoroutine(task_obj): 
                    actual_coro = task_obj
                if actual_coro and hasattr(actual_coro, '__qualname__'): coro_name = actual_coro.__qualname__
                else: coro_name = f"Task {i} (type: {type(task_obj).__name__})"
                print(f"Warning: Task {coro_name} failed with {type(res_or_exc).__name__}: {res_or_exc}")
                continue
            if res_or_exc: flat_results.extend(res_or_exc)
        if not flat_results: 
            print("DEBUG: search_all - flat_results is empty after processing asyncio.gather results. Returning empty DataFrame.")
            return pd.DataFrame()
        
        df = pd.DataFrame(flat_results)
        df['doi_norm'] = df['doi'].apply(lambda x: str(x).lower().replace("https://doi.org/", "").strip() if pd.notna(x) and x else pd.NA)
        df['pmid_str'] = df['pmid'].apply(lambda x: str(x) if pd.notna(x) and x else pd.NA)
        
        df_doi = df[df['doi_norm'].notna()].copy()
        df_no_doi = df[df['doi_norm'].isna()].copy()
        
        if not df_doi.empty:
            df_doi['has_abstract'] = df_doi['abstract'].notna() & (df_doi['abstract'] != '')
            df_doi = df_doi.sort_values(by=['doi_norm', 'has_abstract', 'year', 'pmid_str'], ascending=[True, False, False, True], na_position='last')
            df_doi = df_doi.drop_duplicates(subset=['doi_norm'], keep='first').drop(columns=['has_abstract'], errors='ignore')
        
        if not df_no_doi.empty:
            df_no_doi['has_abstract'] = df_no_doi['abstract'].notna() & (df_no_doi['abstract'] != '')
            subset_cols = ['pmid_str'] if 'pmid_str' in df_no_doi.columns and df_no_doi['pmid_str'].notna().any() else ['title', 'year']
            sort_by_cols = subset_cols + (['has_abstract'] if 'has_abstract' in df_no_doi.columns else [])
            ascending_order = [True]*len(subset_cols) + ([False] if 'has_abstract' in df_no_doi.columns else [])
            df_no_doi = df_no_doi.sort_values(by=sort_by_cols, ascending=ascending_order, na_position='last')
            df_no_doi = df_no_doi.drop_duplicates(subset=subset_cols, keep='first').drop(columns=['has_abstract'], errors='ignore')
            
        df_final = pd.concat([df_doi, df_no_doi], ignore_index=True)
        cols_to_drop = [col for col in ['doi_norm', 'pmid_str'] if col in df_final.columns]
        if cols_to_drop:
            df_final.drop(columns=cols_to_drop, inplace=True)
        
        expected_cols = list(SearchResult.__annotations__.keys())
        for col in expected_cols:
            if col not in df_final.columns: 
                df_final[col] = pd.NA
        df_final = df_final[expected_cols]
        df_final = df_final.sort_values(by=['year', 'title'], ascending=[False, True], na_position='last').reset_index(drop=True)
        
        print(f"Overall search_all took {time.time() - start_time:.2f} seconds. Found {len(df_final)} unique articles.")
        return df_final

class SearchOutput(TypedDict):
    data: List[Dict[str, Any]]
    meta: Dict[str, Any]

async def _async_search_literature(pubmed_query: str, general_query: str, max_results_per_source: int = 50, publication_types: Optional[List[str]] = None, publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> SearchOutput:
    async with AsyncSearchClient(publication_type_mappings=publication_type_mappings) as client:
        df = await client.search_all(pubmed_query, general_query, max_results_per_source, publication_types, start_date, end_date)
    data_records = []
    if not df.empty:
        for _, row in df.iterrows():
            record = {}
            for k, v_val in row.to_dict().items():
                # Removed breakpoint() here
                print(f"DEBUG_LITSEARCH: Processing key='{k}', value='{str(v_val)[:100]}', type='{type(v_val)}'")
                if k in SearchResult.__annotations__:
                    # ---- NEW DEBUG BLOCK ----
                    temp_is_na_check_for_v_val = None
                    temp_is_na_type = None
                    try:
                        temp_is_na_check_for_v_val = pd.isna(v_val)
                        temp_is_na_type = type(temp_is_na_check_for_v_val)
                    except Exception as e_isna:
                        print(f"DEBUG_LITSEARCH: EXCEPTION during pd.isna(v_val) for key='{k}': {e_isna}")
                        raise 
                    print(f"DEBUG_LITSEARCH: Result of pd.isna(v_val) for key='{k}': value='{str(temp_is_na_check_for_v_val)[:100]}', type='{temp_is_na_type}'")
                    # ---- END NEW DEBUG BLOCK ----
                    
                    # Evaluate the first part of the 'and' for the original first 'if' condition
                    # temp_is_na_check_for_v_val can be a scalar bool or a numpy.ndarray of bools.
                    eval_temp_is_na = None
                    if isinstance(temp_is_na_check_for_v_val, (bool, np.bool_)):
                        eval_temp_is_na = temp_is_na_check_for_v_val
                    elif hasattr(temp_is_na_check_for_v_val, 'all') and callable(temp_is_na_check_for_v_val.all):
                        # If it's an array and has .all(), use it. This handles the case where pd.isna returns an array.
                        # For "is NA", we usually care if *any* part is NA, or if the item *is* an NA marker.
                        # If pd.isna returns an array like [F,F,T], it means parts of v_val are NA.
                        # The original 'if pd.isna(v_val)' for a scalar means "if v_val is the NA marker".
                        # If v_val is a collection and pd.isna(v_val) is an array,
                        # then `temp_is_na_check_for_v_val.all()` means "are all elements of v_val NA?"
                        # and `temp_is_na_check_for_v_val.any()` means "are any elements of v_val NA?"
                        # Let's use .all() for now, assuming the intent is "is the whole v_val considered NA?"
                        eval_temp_is_na = temp_is_na_check_for_v_val.all()
                    else: 
                        # If temp_is_na_check_for_v_val is not bool and not array with .all() (e.g. some other object)
                        # This case should ideally not be reached if pd.isna behaves as expected.
                        print(f"DEBUG_LITSEARCH: Unexpected type for temp_is_na_check_for_v_val: {type(temp_is_na_check_for_v_val)}, value: {str(temp_is_na_check_for_v_val)[:100]}")
                        eval_temp_is_na = False # Default to not being NA if type is unexpected

                    if eval_temp_is_na and not isinstance(v_val, (list, tuple, np.ndarray, pd.Series)):
                        print(f"DEBUG_LITSEARCH: Branch 1 for key='{k}' (eval_temp_is_na={eval_temp_is_na})")
                        record[k] = None
                    elif isinstance(v_val, (list, tuple, np.ndarray, pd.Series)):
                        print(f"DEBUG_LITSEARCH: Branch 2 for key='{k}'")
                        # Ensure items within lists/arrays are also handled for NA (though SearchResult expects specific types)
                        if isinstance(v_val, (np.ndarray, pd.Series)):
                            temp_list = []
                            try:
                                # Handle 0-dim arrays by converting to scalar first if possible
                                if v_val.ndim == 0:
                                    scalar_item = v_val.item()
                                    temp_list = [scalar_item if not pd.isna(scalar_item) else None]
                                else:
                                    temp_list = [item if not pd.isna(item) else None for item in v_val]
                            except TypeError: # If .item() fails or iteration fails for some reason
                                print(f"DEBUG_LITSEARCH: TypeError during ndarray/Series processing for key='{k}'. Value: {str(v_val)[:100]}")
                                temp_list = list(v_val) # Fallback, might contain NA
                            record[k] = temp_list
                        else: # list or tuple
                            processed_authors = []
                            for author_candidate in v_val: # v_val is the list of authors
                                if isinstance(author_candidate, (list, tuple, np.ndarray, pd.Series)):
                                    # This author_candidate is unexpectedly an array/list itself.
                                    # Flatten it and join into a single string.
                                    sub_parts = []
                                    for sub_element in author_candidate:
                                        if not pd.isna(sub_element):
                                            sub_parts.append(str(sub_element))
                                    name_str = " ".join(sub_parts).strip()
                                    if name_str: # Only add if not empty after processing
                                        processed_authors.append(name_str)
                                elif not pd.isna(author_candidate):
                                    name_str = str(author_candidate).strip()
                                    if name_str: # Only add if not empty
                                        processed_authors.append(name_str)
                                # If author_candidate is NA or becomes an empty string, it's skipped.
                            record[k] = processed_authors
                    elif isinstance(v_val, (int, float, bool)):
                        print(f"DEBUG_LITSEARCH: Branch 3 for key='{k}'")
                        record[k] = v_val
                    else: 
                        print(f"DEBUG_LITSEARCH: Branch 4 (else) for key='{k}'")
                        is_not_na_check = pd.notna(v_val)
                        print(f"DEBUG_LITSEARCH:   key='{k}', is_not_na_check='{str(is_not_na_check)[:100]}', type='{type(is_not_na_check)}'")
                        # Check if is_not_na_check is a scalar boolean and True,
                        # OR if it's an array-like object where all elements are True.
                        # Ensure that .all() is only called if it's a method (callable).
                        condition = (isinstance(is_not_na_check, (bool, np.bool_)) and is_not_na_check) or \
                                    (hasattr(is_not_na_check, 'all') and callable(is_not_na_check.all) and is_not_na_check.all())
                        print(f"DEBUG_LITSEARCH:   key='{k}', condition_for_str_conversion='{condition}'")
                        record[k] = str(v_val) if condition else None
            data_records.append(record)
    meta = {"total_hits": len(data_records), "query_pubmed": pubmed_query, "query_general": general_query, "publication_types_applied": publication_types or [], "start_date_applied": start_date, "end_date_applied": end_date, "timestamp": pd.Timestamp.now(tz='UTC').isoformat()}
    return {"data": data_records, "meta": meta}

def search_literature(pubmed_query: str, general_query: str, max_results_per_source: int = 50, publication_types: Optional[List[str]] = None, publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> SearchOutput:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(asyncio.ensure_future(_async_search_literature(pubmed_query, general_query, max_results_per_source, publication_types, publication_type_mappings, start_date, end_date)))
    else:
        return asyncio.run(_async_search_literature(pubmed_query, general_query, max_results_per_source, publication_types, publication_type_mappings, start_date, end_date))

class SearchLiteratureParams(BaseModel):
    pubmed_query: str = Field(..., description="Boolean search string for PubMed and Europe PMC (can use MeSH).")
    general_query: str = Field(..., description="Boolean search string for Semantic Scholar, Crossref, and OpenAlex.")
    max_results_per_source: int = Field(50, description="Records to pull from each API.")
    publication_types: Optional[List[str]] = Field(None, description="Optional list of publication types (e.g. ['research', 'review']) to filter by.")
    start_date: Optional[str] = Field(None, description="Start date for search range (YYYY-MM-DD).")
    end_date: Optional[str] = Field(None, description="End date for search range (YYYY-MM-DD).")

class LiteratureSearchToolInstanceConfig(BaseModel):
    publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None

class LiteratureSearchTool(BaseTool[SearchLiteratureParams, SearchOutput], Component[LiteratureSearchToolInstanceConfig]):
    component_config_schema: Type[LiteratureSearchToolInstanceConfig] = LiteratureSearchToolInstanceConfig
    _publication_type_mappings: Optional[Dict[str, Dict[str, str]]]
    def __init__(self, publication_type_mappings: Optional[Dict[str, Dict[str, str]]] = None):
        super().__init__(args_type=SearchLiteratureParams, return_type=SearchOutput, name="search_literature", description="High-throughput literature search across PubMed, Europe PMC, Semantic Scholar, Crossref, and OpenAlex.")
        self._publication_type_mappings = publication_type_mappings
    @classmethod
    def _from_config(cls, config: LiteratureSearchToolInstanceConfig) -> "LiteratureSearchTool":
        return cls(publication_type_mappings=config.publication_type_mappings)
    def _to_config(self) -> LiteratureSearchToolInstanceConfig:
        return LiteratureSearchToolInstanceConfig(publication_type_mappings=self._publication_type_mappings)
    async def run(self, args: SearchLiteratureParams, cancellation_token: Any) -> SearchOutput:
        return await asyncio.get_running_loop().run_in_executor(None, search_literature, args.pubmed_query, args.general_query, args.max_results_per_source, args.publication_types, self._publication_type_mappings, args.start_date, args.end_date)

__all__ = ["search_literature", "LiteratureSearchTool", "SearchLiteratureParams", "SearchResult", "AsyncSearchClient"]

if __name__ == "__main__":
    import sys; import rich; import argparse 
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description="Test the literature search tool.")
    parser.add_argument("--pubmed", type=str, default="metagenomics[Mesh]", help="PubMed/EuropePMC query.")
    parser.add_argument("--general", type=str, default="metagenomics", help="General query for other APIs.")
    parser.add_argument("--max_results", type=int, default=10, help="Max results per source.")
    parser.add_argument("--pub_types", type=str, default=None, help="Comma-separated publication types (e.g., research,review).")
    args = parser.parse_args()
    print(f"Searching PubMed/EuropePMC for: {args.pubmed}"); print(f"Searching general APIs for: {args.general}")
    if args.pub_types: print(f"Filtering by publication types: {args.pub_types}")
    publication_types_list = [s.strip() for s in args.pub_types.split(',')] if args.pub_types else None
    search_output_dict = search_literature(args.pubmed, args.general, args.max_results, publication_types=publication_types_list, publication_type_mappings=None)
    if search_output_dict and search_output_dict.get('data'):
        df = pd.DataFrame(search_output_dict['data']); rich.print(df.head())
    else: rich.print("[bold red]No data returned from search_literature.[/bold red]")
