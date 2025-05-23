import asyncio
import os
import time
from typing import List, Dict, Any, Optional, TypedDict, Type
import httpx
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
                search_results = Entrez.read(handle)
                handle.close()
                ids = search_results["IdList"]

                if not ids:
                    return []

                summary_handle = await self._run_sync_in_thread(
                    Entrez.esummary,
                    db="pubmed",
                    id=",".join(ids)
                )
                summaries = Entrez.read(summary_handle)
                summary_handle.close()

                for summary in summaries:
                    title = summary.get("Title", summary.get("ArticleTitle", "N/A"))
                    pmid = summary.get("Id", "")
                    doi = summary.get("DOI", "")
                    article_ids = summary.get("ArticleIds", {})
                    if not doi and isinstance(article_ids, dict):
                        doi = article_ids.get("doi", "")

                    authors = [author['Name'] for author in summary.get("AuthorList", []) if 'Name' in author]
                    pub_date_str = summary.get("PubDate", "")
                    year = None
                    if pub_date_str and isinstance(pub_date_str, str):
                        parts = pub_date_str.split()
                        if parts and parts[0].isdigit() and len(parts[0]) == 4:
                            year = int(parts[0])
                        elif pub_date_str.isdigit() and len(pub_date_str) == 4:
                             year = int(pub_date_str)

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
                print(f"PubMed search error: {e}")
        end_time = time.time()
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
                for item in raw_results:
                    if item_count >= max_results:
                        print(f"DEBUG: Reached max_results ({max_results}) for Semantic Scholar. Breaking loop.")
                        break
                    item_count += 1
                    print(f"DEBUG: Processing Semantic Scholar item {item_count}...")
                    try:
                        title = item.title
                        authors = [author['name'] for author in item.authors] if item.authors else []
                        year = item.year
                        journal_info = item.journal if item.journal else {}
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

                    result_item: SearchResult = {
                        "title": title,
                        "authors": authors,
                        "doi": doi,
                        "year": year,
                        "journal": journal,
                        "url": url,
                        "source_api": "CrossRef"
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
        print("DEBUG: All source searches completed.")

        flat_results: List[SearchResult] = []
        print("DEBUG: Starting to flatten results.")
        for source_result_list in all_source_results:
            if source_result_list:
                flat_results.extend(source_result_list)
        print(f"DEBUG: flat_results created with {len(flat_results)} items.")

        if not flat_results:
            print("DEBUG: No flat_results, returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(flat_results)
        print(f"DEBUG: DataFrame created with {len(df)} rows.")

        if 'doi' in df.columns:
            print("DEBUG: Normalizing DOIs.")
            df['doi_norm'] = df['doi'].str.lower().str.replace("https://doi.org/", "", regex=False).str.strip()
        else:
            df['doi_norm'] = None
        print("DEBUG: DOI normalization complete.")

        df_with_doi_present = df[df['doi_norm'].notna()].copy()
        df_no_doi_present = df[df['doi_norm'].isna()].copy()

        df_with_doi_deduped = pd.DataFrame(columns=df.columns)
        if not df_with_doi_present.empty:
            df_with_doi_present = df_with_doi_present.sort_values(
                by=['doi_norm', 'year', 'pmid'], 
                ascending=[True, False, True], 
                na_position='last'
            )
            df_with_doi_deduped = df_with_doi_present.drop_duplicates(subset=['doi_norm'], keep='first')

        df_no_doi_deduped = pd.DataFrame(columns=df.columns)
        if not df_no_doi_present.empty:
            df_no_doi_present = df_no_doi_present.sort_values(
                by=['pmid', 'year'], 
                ascending=[True, False], 
                na_position='last'
            )
            df_no_doi_deduped = df_no_doi_present.drop_duplicates(subset=['pmid'], keep='first')
        
        df = pd.concat([df_with_doi_deduped, df_no_doi_deduped])

        df.drop(columns=['doi_norm'], inplace=True, errors='ignore')
        if 'pmid' in df.columns: # Only drop pmid_norm if pmid column exists
            df.drop(columns=['pmid'], inplace=True, errors='ignore') # pmid is not a temporary column, but we don't need it for final output

        df = df.sort_values(by=['source_api', 'year'], ascending=[True, False])
        
        expected_df_cols = list(SearchResult.__annotations__.keys())

        for col in expected_df_cols:
            if col not in df.columns:
                df[col] = pd.NA
        
        current_cols_ordered = [col for col in expected_df_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in current_cols_ordered]
        df = df[current_cols_ordered + other_cols]

        df = df.reset_index(drop=True)

        end_time = time.time()
        print(f"Overall search_all took {end_time - start_time:.2f} seconds")
        return df


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
    
    # Prepare the 'data' field
    data_records = []
    for _, row in df.iterrows():
        record = {
            "doi": row.get("doi"),
            "title": row.get("title"),
            "abstract": row.get("abstract")
        }
        data_records.append(record)

    # Prepare the 'meta' field
    meta_info = {
        "total_hits": len(df), # Total hits after deduplication and sorting
        "query": general_query,
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
        task = loop.create_task(_async_search_literature(pubmed_query, general_query, max_results_per_source))
        return loop.run_until_complete(task)
    else:
        return asyncio.run(_async_search_literature(pubmed_query, general_query, max_results_per_source))

class SearchLiteratureParams(BaseModel):
    pubmed_query: str = Field(..., description="Boolean search string for PubMed and Europe PMC (can use MeSH).")
    general_query: str = Field(..., description="Boolean search string for Semantic Scholar and Crossref.")
    max_results_per_source: int = Field(50, description="Records to pull from each API (default = 50).")

class LiteratureSearchToolInstanceConfig(BaseModel):
    pass

class LiteratureSearchTool(
    BaseTool[SearchLiteratureParams, SearchOutput], # Changed return type here
    Component[LiteratureSearchToolInstanceConfig]
):
    component_config_schema: Type[LiteratureSearchToolInstanceConfig] = LiteratureSearchToolInstanceConfig

    def __init__(self):
        super().__init__(
            args_type=SearchLiteratureParams,
            return_type=SearchOutput, # Changed return type here
            name="search_literature",
            description="High-throughput literature search across PubMed, Europe PMC, Semantic Scholar and Crossref."
        )

    @classmethod
    def _from_config(cls, config: LiteratureSearchToolInstanceConfig) -> "LiteratureSearchTool":
        return cls()

    def _to_config(self) -> LiteratureSearchToolInstanceConfig:
        return LiteratureSearchToolInstanceConfig()

    async def run(self, args: SearchLiteratureParams, cancellation_token: Any) -> SearchOutput:
        return search_literature(args.pubmed_query, args.general_query, args.max_results_per_source)

__all__ = ["search_literature", "LiteratureSearchTool", "SearchLiteratureParams"]

if __name__ == "__main__":
    import sys, rich
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Example usage for testing:
    # python -m tools.search --pubmed "CRISPR[Mesh]" --general "CRISPR gene editing"
    parser = argparse.ArgumentParser(description="Test the literature search tool.")
    parser.add_argument("--pubmed", type=str, default="metagenomics[Mesh]", help="PubMed/EuropePMC query.")
    parser.add_argument("--general", type=str, default="metagenomics", help="General query for other APIs.")
    parser.add_argument("--max_results", type=int, default=10, help="Max results per source.")
    args = parser.parse_args()

    print(f"Searching PubMed/EuropePMC for: {args.pubmed}")
    print(f"Searching general APIs for: {args.general}")
    df = search_literature(args.pubmed, args.general, args.max_results)
    rich.print(df.head())
