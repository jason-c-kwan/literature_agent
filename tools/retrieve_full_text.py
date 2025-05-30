import asyncio
import json
import logging
import os
import sys # Added import for sys module
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse # Added for cookie domain parsing
import tempfile # Added for handling PDFBytesResult

# Ensure the project root is in sys.path for sibling imports like 'consts'
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import aiohttp
import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log, retry_if_exception_type

# Import project-specific modules
from tools import retrieve_europepmc
from tools import parse_json_xml
from tools import retrieve_pmc
from tools import elink_pubmed # Imports both _get_article_links_by_id_type_xml and new get_article_links
from tools import resolve_pdf_link # Contains resolve_content
from tools import advanced_scraper
from tools.parse_pdf import convert_pdf_to_markdown_string # Import the refactored function
from tools import retrieve_unpaywall

# --- Configuration ---
# TODO: Load from config/settings.yaml
MAX_CONCURRENT_DOI_PROCESSING = 10 # Overall concurrency for processing DOIs
# Semaphores for different services (example values, tune as needed)
SEMAPHORE_EUROPEPMC = asyncio.Semaphore(3)
SEMAPHORE_NCBI_PMC = asyncio.Semaphore(3) # For retrieve_pmc.py
SEMAPHORE_NCBI_ELINK = asyncio.Semaphore(10) # elink_pubmed.py has its own internal rate limiting, but a semaphore can add overall control
SEMAPHORE_UNPAYWALL = asyncio.Semaphore(10)
SEMAPHORE_GENERAL_SCRAPING = asyncio.Semaphore(5) # For resolve_content and advanced_scraper general calls

CACHE_DIR = "workspace/fulltext_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

# --- Logging Setup ---
# (Similar to advanced_scraper.py or resolve_pdf_link.py - can be standardized later)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    # Simpler formatter to avoid issues if 'event_type' is not in all log records from all modules
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Helper function to read from cache ---
async def get_from_cache(doi: str) -> Optional[Dict[str, Any]]:
    cache_file_path = os.path.join(CACHE_DIR, f"{doi.replace('/', '_')}.json")
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                # TODO: Add TTL check if needed
                logger.info(f"Cache hit for DOI: {doi}", extra={"doi": doi, "event_type": "cache_hit"})
                return cached_data
        except Exception as e:
            logger.error(f"Error reading cache for DOI {doi}: {e}", extra={"doi": doi, "event_type": "cache_read_error"})
    return None

# --- Helper function to write to cache ---
async def write_to_cache(doi: str, data: Dict[str, Any]):
    cache_file_path = os.path.join(CACHE_DIR, f"{doi.replace('/', '_')}.json")
    try:
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cached result for DOI: {doi}", extra={"doi": doi, "event_type": "cache_write"})
    except Exception as e:
        logger.error(f"Error writing cache for DOI {doi}: {e}", extra={"doi": doi, "event_type": "cache_write_error"})


"""
Module for retrieving the full text of academic articles from various sources.

This module orchestrates the process of fetching full-text content for a given
list of articles, primarily identified by their DOIs. It attempts retrieval through
several channels in a fallback sequence:
1. Europe PMC (structured JSON/XML if available)
2. PubMed Central (PMC) XML (if PMCID is available)
3. NCBI ELink 'prlinks' command (XML based, for publisher links via PMID or DOI)
4. Unpaywall API (to find Open Access URLs)
5. NCBI ELink 'llinks' command (JSON based, for PubMed LinkOut URLs via PMID)
For URLs obtained from ELink, Unpaywall, or LinkOut, it uses `resolve_content`
and `advanced_scraper` to download and parse PDFs or extract HTML content.
Parsed content is converted to Markdown. Results are cached to avoid redundant
requests.
"""

# --- Refactored PDF Parser (placeholder, actual refactoring of parse_pdf.py needed) ---
async def parse_pdf_to_markdown(pdf_path: str) -> Optional[str]:
    """
    Placeholder for calling the refactored PDF parsing logic.
    This should invoke the core functionality of tools/parse_pdf.py.
    """
    logger.debug(f"Attempting to parse PDF: {pdf_path}", extra={"pdf_path": pdf_path, "event_type": "pdf_parse_attempt"})
    try:
        # Call the synchronous function convert_pdf_to_markdown_string in a separate thread
        # to avoid blocking the asyncio event loop.
        # The convert_pdf_to_markdown_string function itself handles file opening/closing.
        markdown_text = await asyncio.to_thread(
            convert_pdf_to_markdown_string, 
            pdf_path, 
            None # pages_to_process_str, pass None to process all pages
        )
        
        if markdown_text is not None and markdown_text.strip(): # Check if not None and not just whitespace
            logger.info(f"Successfully parsed PDF: {pdf_path}", extra={"pdf_path": pdf_path, "event_type": "pdf_parse_success"})
            return markdown_text
        elif markdown_text is not None: # It's an empty string (or only whitespace)
             logger.warning(f"PDF parsing for {pdf_path} resulted in empty markdown.", extra={"pdf_path": pdf_path, "event_type": "pdf_parse_empty"})
             return None # Treat empty useful string as None for consistency
        else: # Function returned None (should not happen with current refactor unless error)
            logger.error(f"PDF parsing function returned None for {pdf_path}", extra={"pdf_path": pdf_path, "event_type": "pdf_parse_returned_none"})
            return None
            
    except FileNotFoundError:
        logger.error(f"PDF file not found for parsing: {pdf_path}", extra={"pdf_path": pdf_path, "event_type": "pdf_parse_file_not_found"})
        return None
    except ValueError as ve: # From page number parsing, though we pass None here
        logger.error(f"ValueError during PDF parsing for {pdf_path}: {ve}", exc_info=True, extra={"pdf_path": pdf_path, "event_type": "pdf_parse_value_error"})
        return None
    except RuntimeError as re_pdf: # From pymupdf opening or conversion
        logger.error(f"RuntimeError during PDF parsing for {pdf_path}: {re_pdf}", exc_info=True, extra={"pdf_path": pdf_path, "event_type": "pdf_parse_runtime_error"})
        return None
    except Exception as e:
        logger.error(f"Unexpected exception during PDF parsing for {pdf_path}: {e}", exc_info=True, extra={"pdf_path": pdf_path, "event_type": "pdf_parse_exception"})
        return None

# --- Individual DOI Retrieval Orchestrator ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((aiohttp.ClientError, httpx.RequestError, asyncio.TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def get_full_text_for_doi(
    article_data: Dict[str, Any],
    # TODO: Pass shared client sessions if needed, or manage them internally per call
) -> Tuple[Optional[str], str, List[str]]: # Added List[str] for tried_sources
    """
    Orchestrates the retrieval of full text for a single DOI using various methods.
    
    Args:
        article_data: A dictionary containing article metadata, including 'doi',
                      'pmid', and 'pmcid'.
                      
    Returns:
        A tuple containing:
            - Optional[str]: The Markdown content of the full text, or None if not found.
            - str: A status message describing the outcome of the retrieval attempt.
            - List[str]: A list of sources that were attempted for this article.
    """
    doi = article_data.get("doi")
    pmid = article_data.get("pmid")
    pmcid_numeric = article_data.get("pmcid") # Numeric part, e.g., "8905495"
    pmcid_full = f"PMC{pmcid_numeric}" if pmcid_numeric else None

    tried_sources: List[str] = [] # To track attempted sources

    if not doi:
        return None, "DOI not provided in article data", tried_sources

    logger.debug(f"Starting full text retrieval for DOI: {doi}", extra={"doi": doi, "event_type": "doi_retrieval_start"})

    # --- Cookie Management (Task 2c) ---
    session_cookies_by_domain: Dict[str, List[Dict]] = {}
    # Helper to update session_cookies_by_domain from a result object
    def _update_session_cookies(result_obj: Any, current_doi: str): # Ensure Any is imported or use a more specific type
        nonlocal session_cookies_by_domain
        if hasattr(result_obj, 'cookies') and result_obj.cookies and hasattr(result_obj, 'url') and result_obj.url:
            # Ensure result_obj.url is not None before parsing
            if result_obj.url is None:
                logger.warning(f"Result object for DOI {current_doi} has cookies but no URL, cannot determine domain.", extra={"doi": current_doi, "event_type": "cookie_update_no_url"})
                return

            domain = urlparse(result_obj.url).netloc
            if domain:
                # Update cookies: new ones take precedence for the same name, otherwise merge
                existing_domain_cookies = {c['name']: c for c in session_cookies_by_domain.get(domain, [])}
                new_domain_cookies = {c['name']: c for c in result_obj.cookies}
                existing_domain_cookies.update(new_domain_cookies)
                session_cookies_by_domain[domain] = list(existing_domain_cookies.values())
                logger.info(f"Stored/Updated {len(session_cookies_by_domain[domain])} cookies for domain {domain} from DOI {current_doi}", 
                            extra={"doi": current_doi, "domain": domain, "num_cookies": len(session_cookies_by_domain[domain]), "event_type": "session_cookies_updated"})

    # Check cache first
    cached_result = await get_from_cache(doi)
    if cached_result:
        if cached_result.get("status") == "success":
            logger.info(f"DOI {doi}: Cache hit and status is success.", extra={"doi": doi, "event_type": "cache_hit_success"})
            # Assuming cached results don't store tried_sources, or we don't need them for cache hits
            return cached_result.get("markdown"), "Retrieved from cache", ["Cache"] 
        elif cached_result.get("status") == "failure":
            logger.info(f"DOI {doi}: Cache hit and status is failure. Attempting fresh retrieval.", 
                        extra={"doi": doi, "event_type": "cache_hit_failure_override", "cached_reason": cached_result.get("reason")})
        else:
            logger.info(f"DOI {doi}: Cache hit but status is neither success nor failure ('{cached_result.get('status')}'). Proceeding with fresh retrieval.",
                        extra={"doi": doi, "event_type": "cache_hit_unknown_status_override", "cached_status": cached_result.get('status')})

    current_markdown_content: Optional[str] = None
    current_status_message: str = ""

    # Initialize shared HTTP clients
    logger.debug(f"DOI {doi}: Initializing HTTP clients for fresh retrieval attempt.", extra={"doi": doi, "event_type": "http_client_init"})
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http_client, \
               aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20.0)) as aio_session:

        # Step 1: Europe PMC (JSON)
        tried_sources.append("EuropePMC_JSON")
        logger.debug(f"DOI {doi}: Attempting Europe PMC.", extra={"doi": doi, "event_type": "europepmc_attempt"})
        async with SEMAPHORE_EUROPEPMC:
            try:
                europepmc_json = await retrieve_europepmc.fetch_europepmc(doi)
                if europepmc_json:
                    structured_elements = parse_json_xml.parse_europe_pmc_json(europepmc_json)
                    if structured_elements and not (len(structured_elements) == 1 and structured_elements[0].get("type") == "error"):
                        current_markdown_content = parse_json_xml._render_elements_to_markdown_string(structured_elements)
                        if current_markdown_content and current_markdown_content.strip():
                            logger.info(f"DOI {doi}: Successfully parsed Europe PMC JSON to Markdown.", extra={"doi": doi, "event_type": "europepmc_success"})
                            await write_to_cache(doi, {"status": "success", "markdown": current_markdown_content, "source": "EuropePMC_JSON"})
                            return current_markdown_content, "Retrieved and parsed from Europe PMC (JSON)", tried_sources
            except Exception as e:
                logger.error(f"DOI {doi}: Error fetching/parsing Europe PMC: {e}", exc_info=True, extra={"doi": doi, "event_type": "europepmc_exception"})

        # Step 2: PMC XML
        if pmcid_full:
            tried_sources.append("PMC_XML")
            logger.debug(f"DOI {doi}: Attempting PMC XML for PMCID: {pmcid_full}.", extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_attempt"})
            async with SEMAPHORE_NCBI_PMC:
                try:
                    pmc_xml_str = await retrieve_pmc.fetch_pmc_xml(pmcid_full, session=aio_session)
                    if pmc_xml_str:
                        structured_elements = parse_json_xml.parse_pmc_xml(pmc_xml_str)
                        if structured_elements and not (len(structured_elements) == 1 and structured_elements[0].get("type") == "error"):
                            current_markdown_content = parse_json_xml._render_elements_to_markdown_string(structured_elements)
                            if current_markdown_content and current_markdown_content.strip():
                                logger.info(f"DOI {doi}: Successfully parsed PMC XML to Markdown for PMCID: {pmcid_full}.", extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_success"})
                                await write_to_cache(doi, {"status": "success", "markdown": current_markdown_content, "source": "PMC_XML"})
                                return current_markdown_content, f"Retrieved and parsed from PMC XML (PMCID: {pmcid_full})", tried_sources
                except Exception as e:
                    logger.error(f"DOI {doi}: Error fetching/parsing PMC XML for PMCID {pmcid_full}: {e}", exc_info=True, extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_exception"})
        else:
            logger.debug(f"DOI {doi}: No PMCID available, skipping PMC XML.", extra={"doi": doi, "event_type": "pmcxml_skip_no_pmcid"})

        # Step 3: Elink 'prlinks' (XML) & Scrapers
        identifier_for_elink_xml = pmid if pmid else doi
        id_type_for_elink_xml = "pmid" if pmid else "doi"

        if identifier_for_elink_xml:
            tried_sources.append("Elink_prlinks_XML")
            logger.debug(f"DOI {doi}: Attempting Elink (prlinks XML) for {id_type_for_elink_xml}: {identifier_for_elink_xml}.", extra={"doi": doi, "identifier": identifier_for_elink_xml, "id_type": id_type_for_elink_xml, "event_type": "elink_prlinks_attempt"})
            async with SEMAPHORE_NCBI_ELINK:
                try:
                    # Use the renamed XML-specific function
                    article_links_xml = await elink_pubmed._get_article_links_by_id_type_xml(identifier=identifier_for_elink_xml, id_type=id_type_for_elink_xml)
                    if article_links_xml:
                        logger.info(f"DOI {doi}: Elink (prlinks XML) found {len(article_links_xml)} links for {id_type_for_elink_xml}: {identifier_for_elink_xml}.", extra={"doi": doi, "count": len(article_links_xml), "event_type": "elink_prlinks_links_count_found"})
                        for link_url in article_links_xml:
                            logger.debug(f"DOI {doi}: Processing Elink (prlinks XML) URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_prlinks_url_processing_start"})
                            current_domain_for_call = urlparse(link_url).netloc
                            cookies_for_this_call = session_cookies_by_domain.get(current_domain_for_call)
                            async with SEMAPHORE_GENERAL_SCRAPING:
                                content_result = await resolve_pdf_link.resolve_content(link_url, client=http_client, original_doi_for_referer=doi, session_cookies=cookies_for_this_call)
                            _update_session_cookies(content_result, doi)

                            if content_result.type == "file":
                                md = await parse_pdf_to_markdown(content_result.path)
                                if md:
                                    logger.info(f"DOI {doi}: Elink (prlinks XML) PDF to Markdown success from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_prlinks_pdf_success"})
                                    await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Elink_prlinks_PDF ({link_url})"})
                                    return md, f"Retrieved PDF via Elink (prlinks XML) ({link_url}) and parsed", tried_sources
                            elif content_result.type == "html":
                                # Check length before returning
                                if content_result.text and len(content_result.text) > resolve_pdf_link.MIN_HTML_CONTENT_LENGTH:
                                    logger.info(f"DOI {doi}: Elink (prlinks XML) HTML success from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_prlinks_html_success"})
                                    await write_to_cache(doi, {"status": "success", "markdown": content_result.text, "source": f"Elink_prlinks_HTML ({link_url})"})
                                    return content_result.text, f"Retrieved HTML via Elink (prlinks XML) ({link_url})", tried_sources
                            # ... (rest of advanced_scraper fallback for prlinks)
                            else: 
                                logger.warning(f"DOI {doi}: resolve_content failed for Elink (prlinks XML) URL: {link_url}. Reason: {content_result.reason}. Trying advanced_scraper.", extra={"doi": doi, "url": link_url, "reason": content_result.reason, "event_type": "elink_prlinks_resolve_fail_try_advanced"})
                                cookies_for_this_call = session_cookies_by_domain.get(current_domain_for_call) 
                                async with SEMAPHORE_GENERAL_SCRAPING:
                                    advanced_result = await advanced_scraper.scrape_with_fallback(
                                        link_url, original_doi_for_referer=doi, session_cookies=cookies_for_this_call
                                    )
                                _update_session_cookies(advanced_result, doi)
                                if advanced_result.type == "file":
                                    md = await parse_pdf_to_markdown(advanced_result.path)
                                    if md:
                                        logger.info(f"DOI {doi}: Advanced scraper PDF success (from Elink prlinks path) from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_prlinks_advanced_pdf_success"})
                                        await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Elink_prlinks_AdvancedScraper_PDF ({link_url})"})
                                        return md, f"Retrieved PDF via Elink (prlinks) > Advanced Scraper ({link_url}) and parsed", tried_sources
                                elif advanced_result.type == "html":
                                    if advanced_result.text and len(advanced_result.text) > resolve_pdf_link.MIN_HTML_CONTENT_LENGTH:
                                        logger.info(f"DOI {doi}: Advanced scraper HTML success (from Elink prlinks path) from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_prlinks_advanced_html_success"})
                                        await write_to_cache(doi, {"status": "success", "markdown": advanced_result.text, "source": f"Elink_prlinks_AdvancedScraper_HTML ({link_url})"})
                                        return advanced_result.text, f"Retrieved HTML via Elink (prlinks) > Advanced Scraper ({link_url})", tried_sources
                                elif advanced_result.type == "pdf_bytes":
                                    temp_pdf_path = None
                                    try:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                            tmp_file.write(advanced_result.content)
                                            temp_pdf_path = tmp_file.name
                                        md = await parse_pdf_to_markdown(temp_pdf_path)
                                        if md:
                                            logger.info(f"DOI {doi}: Advanced scraper PDFBytes success (from Elink prlinks path) from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_prlinks_advanced_pdfbytes_success"})
                                            await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Elink_prlinks_AdvancedScraper_PDFBytes ({link_url})"})
                                            return md, f"Retrieved PDF (bytes) via Elink (prlinks) > Advanced Scraper ({link_url}) and parsed", tried_sources
                                    finally:
                                        if temp_pdf_path and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
                                else:
                                    logger.warning(f"DOI {doi}: Advanced scraper also failed for Elink (prlinks XML) URL: {link_url}. Reason: {advanced_result.reason}", extra={"doi": doi, "url": link_url, "reason": advanced_result.reason, "event_type": "elink_prlinks_advanced_fail"})
                    else:
                        logger.debug(f"DOI {doi}: No links found by Elink (prlinks XML) for {id_type_for_elink_xml}: {identifier_for_elink_xml}", extra={"doi": doi, "event_type": "elink_prlinks_no_links"})
                except Exception as e:
                    logger.error(f"DOI {doi}: Error during Elink (prlinks XML) processing for {id_type_for_elink_xml} {identifier_for_elink_xml}: {e}", exc_info=True, extra={"doi": doi, "event_type": "elink_prlinks_exception"})
        else:
            logger.debug(f"DOI {doi}: No PMID or DOI available for Elink (prlinks XML), skipping.", extra={"doi": doi, "event_type": "elink_prlinks_skip_no_id"})

        # Step 4: Unpaywall & Scrapers
        tried_sources.append("Unpaywall")
        logger.debug(f"DOI {doi}: Attempting Unpaywall.", extra={"doi": doi, "event_type": "unpaywall_attempt"})
        async with SEMAPHORE_UNPAYWALL:
            try:
                oa_url = await retrieve_unpaywall.get_unpaywall_oa_url(doi, session=aio_session)
                if oa_url and not isinstance(oa_url, str): 
                    logger.error(f"DOI {doi}: Unpaywall returned non-string OA URL: {type(oa_url)}. Treating as no URL.", extra={"doi": doi, "event_type": "unpaywall_non_string_url"})
                    oa_url = None
                if oa_url:
                    logger.debug(f"DOI {doi}: Unpaywall found OA URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_url_found"})
                    current_domain_for_call = urlparse(oa_url).netloc
                    cookies_for_this_call = session_cookies_by_domain.get(current_domain_for_call)
                    async with SEMAPHORE_GENERAL_SCRAPING:
                        content_result = await resolve_pdf_link.resolve_content(oa_url, client=http_client, original_doi_for_referer=doi, session_cookies=cookies_for_this_call)
                    _update_session_cookies(content_result, doi)

                    if content_result.type == "file":
                        md = await parse_pdf_to_markdown(content_result.path)
                        if md:
                            logger.info(f"DOI {doi}: Unpaywall PDF to Markdown success from URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_pdf_success"})
                            await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Unpaywall_PDF ({oa_url})"})
                            return md, f"Retrieved PDF via Unpaywall ({oa_url}) and parsed", tried_sources
                    elif content_result.type == "html":
                        if content_result.text and len(content_result.text) > resolve_pdf_link.MIN_HTML_CONTENT_LENGTH:
                            logger.info(f"DOI {doi}: Unpaywall HTML success from URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_html_success"})
                            await write_to_cache(doi, {"status": "success", "markdown": content_result.text, "source": f"Unpaywall_HTML ({oa_url})"})
                            return content_result.text, f"Retrieved HTML via Unpaywall ({oa_url})", tried_sources
                    else: 
                        logger.warning(f"DOI {doi}: resolve_content failed for Unpaywall URL: {oa_url}. Reason: {content_result.reason}. Trying advanced_scraper.", extra={"doi": doi, "url": oa_url, "reason": content_result.reason, "event_type": "unpaywall_resolve_fail_try_advanced"})
                        cookies_for_this_call = session_cookies_by_domain.get(current_domain_for_call)
                        async with SEMAPHORE_GENERAL_SCRAPING:
                            advanced_result = await advanced_scraper.scrape_with_fallback(
                                url=oa_url, original_doi_for_referer=doi, session_cookies=cookies_for_this_call
                            )
                        _update_session_cookies(advanced_result, doi)
                        if advanced_result.type == "file":
                            md = await parse_pdf_to_markdown(advanced_result.path)
                            if md:
                                logger.info(f"DOI {doi}: Advanced scraper PDF success (from Unpaywall path) for URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_advanced_pdf_success"})
                                await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Unpaywall_AdvancedScraper_PDF ({oa_url})"})
                                return md, f"Retrieved PDF via Unpaywall > Advanced Scraper ({oa_url}) and parsed", tried_sources
                        elif advanced_result.type == "html":
                             if advanced_result.text and len(advanced_result.text) > resolve_pdf_link.MIN_HTML_CONTENT_LENGTH:
                                logger.info(f"DOI {doi}: Advanced scraper HTML success (from Unpaywall path) for URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_advanced_html_success"})
                                await write_to_cache(doi, {"status": "success", "markdown": advanced_result.text, "source": f"Unpaywall_AdvancedScraper_HTML ({oa_url})"})
                                return advanced_result.text, f"Retrieved HTML via Unpaywall > Advanced Scraper ({oa_url})", tried_sources
                        elif advanced_result.type == "pdf_bytes":
                            temp_pdf_path = None
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                    tmp_file.write(advanced_result.content)
                                    temp_pdf_path = tmp_file.name
                                md = await parse_pdf_to_markdown(temp_pdf_path)
                                if md:
                                    logger.info(f"DOI {doi}: Advanced scraper PDFBytes success (from Unpaywall path) for URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_advanced_pdfbytes_success"})
                                    await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Unpaywall_AdvancedScraper_PDFBytes ({oa_url})"})
                                    return md, f"Retrieved PDF (bytes) via Unpaywall > Advanced Scraper ({oa_url}) and parsed", tried_sources
                            finally:
                                if temp_pdf_path and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
                        else:
                            logger.warning(f"DOI {doi}: Advanced scraper also failed for Unpaywall URL: {oa_url}. Reason: {advanced_result.reason}", extra={"doi": doi, "url": oa_url, "reason": advanced_result.reason, "event_type": "unpaywall_advanced_fail"})
                else:
                    logger.debug(f"DOI {doi}: No OA URL found by Unpaywall.", extra={"doi": doi, "event_type": "unpaywall_no_url"})
            except Exception as e:
                logger.error(f"DOI {doi}: Error during Unpaywall processing: {e}", exc_info=True, extra={"doi": doi, "event_type": "unpaywall_exception"})
        
        # Step 5: PubMed LinkOut (llinks JSON) - NEW
        if pmid and not current_markdown_content: # Only try if PMID exists and no content found yet
            tried_sources.append("PubMed_LinkOut_llinks")
            logger.debug(f"DOI {doi}, PMID {pmid}: Attempting PubMed LinkOut (llinks).", extra={"doi": doi, "pmid": pmid, "event_type": "pubmed_linkout_llinks_attempt"})
            async with SEMAPHORE_NCBI_ELINK: # Reuse Elink semaphore or define a new one if different rate limits apply
                try:
                    linkout_urls = await elink_pubmed.get_article_links(pmid=pmid) # Call the new JSON-based function
                    if linkout_urls:
                        logger.info(f"DOI {doi}, PMID {pmid}: PubMed LinkOut (llinks) found {len(linkout_urls)} links.", extra={"doi": doi, "pmid": pmid, "count": len(linkout_urls), "event_type": "pubmed_linkout_llinks_found"})
                        for link_url in linkout_urls:
                            logger.debug(f"DOI {doi}: Processing PubMed LinkOut (llinks) URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "pubmed_linkout_llinks_url_processing_start"})
                            current_domain_for_call = urlparse(link_url).netloc
                            cookies_for_this_call = session_cookies_by_domain.get(current_domain_for_call)
                            async with SEMAPHORE_GENERAL_SCRAPING:
                                content_result = await resolve_pdf_link.resolve_content(link_url, client=http_client, original_doi_for_referer=doi, session_cookies=cookies_for_this_call)
                            _update_session_cookies(content_result, doi)

                            if content_result.type == "file":
                                md = await parse_pdf_to_markdown(content_result.path)
                                if md:
                                    logger.info(f"DOI {doi}: PubMed LinkOut (llinks) PDF to Markdown success from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "pubmed_linkout_llinks_pdf_success"})
                                    await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"PubMed_LinkOut_llinks_PDF ({link_url})"})
                                    return md, f"Retrieved PDF via PubMed LinkOut (llinks) ({link_url}) and parsed", tried_sources
                            elif content_result.type == "html":
                                if content_result.text and len(content_result.text) > resolve_pdf_link.MIN_HTML_CONTENT_LENGTH:
                                    logger.info(f"DOI {doi}: PubMed LinkOut (llinks) HTML success from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "pubmed_linkout_llinks_html_success"})
                                    await write_to_cache(doi, {"status": "success", "markdown": content_result.text, "source": f"PubMed_LinkOut_llinks_HTML ({link_url})"})
                                    return content_result.text, f"Retrieved HTML via PubMed LinkOut (llinks) ({link_url})", tried_sources
                            else: # resolve_content failed for this LinkOut URL
                                logger.warning(f"DOI {doi}: resolve_content failed for PubMed LinkOut (llinks) URL: {link_url}. Reason: {content_result.reason}. Trying advanced_scraper.", extra={"doi": doi, "url": link_url, "reason": content_result.reason, "event_type": "pubmed_linkout_llinks_resolve_fail_try_advanced"})
                                cookies_for_this_call = session_cookies_by_domain.get(current_domain_for_call)
                                async with SEMAPHORE_GENERAL_SCRAPING:
                                    advanced_result = await advanced_scraper.scrape_with_fallback(
                                        link_url, original_doi_for_referer=doi, session_cookies=cookies_for_this_call
                                    )
                                _update_session_cookies(advanced_result, doi)
                                if advanced_result.type == "file":
                                    md = await parse_pdf_to_markdown(advanced_result.path)
                                    if md:
                                        logger.info(f"DOI {doi}: Advanced scraper PDF success (from PubMed LinkOut llinks path) from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "pubmed_linkout_llinks_advanced_pdf_success"})
                                        await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"PubMed_LinkOut_llinks_AdvancedScraper_PDF ({link_url})"})
                                        return md, f"Retrieved PDF via PubMed LinkOut (llinks) > Advanced Scraper ({link_url}) and parsed", tried_sources
                                elif advanced_result.type == "html":
                                    if advanced_result.text and len(advanced_result.text) > resolve_pdf_link.MIN_HTML_CONTENT_LENGTH:
                                        logger.info(f"DOI {doi}: Advanced scraper HTML success (from PubMed LinkOut llinks path) from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "pubmed_linkout_llinks_advanced_html_success"})
                                        await write_to_cache(doi, {"status": "success", "markdown": advanced_result.text, "source": f"PubMed_LinkOut_llinks_AdvancedScraper_HTML ({link_url})"})
                                        return advanced_result.text, f"Retrieved HTML via PubMed LinkOut (llinks) > Advanced Scraper ({link_url})", tried_sources
                                elif advanced_result.type == "pdf_bytes":
                                    temp_pdf_path = None
                                    try:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                            tmp_file.write(advanced_result.content)
                                            temp_pdf_path = tmp_file.name
                                        md = await parse_pdf_to_markdown(temp_pdf_path)
                                        if md:
                                            logger.info(f"DOI {doi}: Advanced scraper PDFBytes success (from PubMed LinkOut llinks path) from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "pubmed_linkout_llinks_advanced_pdfbytes_success"})
                                            await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"PubMed_LinkOut_llinks_AdvancedScraper_PDFBytes ({link_url})"})
                                            return md, f"Retrieved PDF (bytes) via PubMed LinkOut (llinks) > Advanced Scraper ({link_url}) and parsed", tried_sources
                                    finally:
                                        if temp_pdf_path and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
                                else:
                                    logger.warning(f"DOI {doi}: Advanced scraper also failed for PubMed LinkOut (llinks) URL: {link_url}. Reason: {advanced_result.reason}", extra={"doi": doi, "url": link_url, "reason": advanced_result.reason, "event_type": "pubmed_linkout_llinks_advanced_fail"})
                    else:
                        logger.debug(f"DOI {doi}, PMID {pmid}: No links found by PubMed LinkOut (llinks).", extra={"doi": doi, "pmid": pmid, "event_type": "pubmed_linkout_llinks_no_links"})
                except Exception as e:
                    logger.error(f"DOI {doi}, PMID {pmid}: Error during PubMed LinkOut (llinks) processing: {e}", exc_info=True, extra={"doi": doi, "pmid": pmid, "event_type": "pubmed_linkout_llinks_exception"})
        else:
            if not pmid:
                logger.debug(f"DOI {doi}: No PMID available, skipping PubMed LinkOut (llinks).", extra={"doi": doi, "event_type": "pubmed_linkout_llinks_skip_no_pmid"})
            # If pmid exists but current_markdown_content is already found, this block is skipped.

    # If all steps fail
    failure_reason = f"All retrieval methods failed. Tried: {', '.join(tried_sources) if tried_sources else 'None'}."
    logger.warning(f"DOI {doi}: {failure_reason}", extra={"doi": doi, "tried_sources": tried_sources, "event_type": "doi_retrieval_failed_all_steps"})
    await write_to_cache(doi, {"status": "failure", "reason": failure_reason})
    return None, "Full text not found after all attempts", tried_sources


# --- Main Entry Point ---
async def retrieve_full_texts_for_dois( # This was missing the function definition line
    query_refiner_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main function to retrieve full texts for DOIs from query_refiner_output.
    Filters DOIs based on relevance_score (4 or 5).
    """
    doi = article_data.get("doi")
    pmid = article_data.get("pmid")
    pmcid_numeric = article_data.get("pmcid") # Numeric part, e.g., "8905495"
    pmcid_full = f"PMC{pmcid_numeric}" if pmcid_numeric else None

    if not doi:
        return None, "DOI not provided in article data"

    logger.debug(f"Starting full text retrieval for DOI: {doi}", extra={"doi": doi, "event_type": "doi_retrieval_start"})

    # Check cache first
    cached_result = await get_from_cache(doi)
    if cached_result:
        if cached_result.get("status") == "success":
            logger.info(f"DOI {doi}: Cache hit and status is success.", extra={"doi": doi, "event_type": "cache_hit_success"})
            return cached_result.get("markdown"), "Retrieved from cache"
        elif cached_result.get("status") == "failure":
            # For debugging false negatives, let's re-attempt even if a failure is cached.
            logger.info(f"DOI {doi}: Cache hit and status is failure. Attempting fresh retrieval.", 
                        extra={"doi": doi, "event_type": "cache_hit_failure_override", "cached_reason": cached_result.get("reason")})
            # Do not return here; proceed to fresh retrieval
        else:
            logger.info(f"DOI {doi}: Cache hit but status is neither success nor failure ('{cached_result.get('status')}'). Proceeding with fresh retrieval.",
                        extra={"doi": doi, "event_type": "cache_hit_unknown_status_override", "cached_status": cached_result.get('status')})
            # Proceed to fresh retrieval

    # Initialize shared HTTP clients (consider passing these from the main orchestrator for efficiency)
    logger.debug(f"DOI {doi}: Initializing HTTP clients for fresh retrieval attempt.", extra={"doi": doi, "event_type": "http_client_init"})
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http_client, \
               aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20.0)) as aio_session: # Example timeout

        # Step 1: Europe PMC (JSON)
        logger.debug(f"DOI {doi}: Attempting Europe PMC.", extra={"doi": doi, "event_type": "europepmc_attempt"})
        async with SEMAPHORE_EUROPEPMC:
            try:
                europepmc_json = await retrieve_europepmc.fetch_europepmc(doi) # Uses httpx internally
                if europepmc_json:
                    logger.debug(f"DOI {doi}: Europe PMC JSON found.", extra={"doi": doi, "event_type": "europepmc_json_found"})
                    # parse_europe_pmc_json returns List[Dict[str, Any]] (structured elements)
                    structured_elements = parse_json_xml.parse_europe_pmc_json(europepmc_json)
                    if structured_elements and not (len(structured_elements) == 1 and structured_elements[0].get("type") == "error"):
                        markdown_content = parse_json_xml._render_elements_to_markdown_string(structured_elements)
                        if markdown_content and markdown_content.strip():
                            logger.info(f"DOI {doi}: Successfully parsed Europe PMC JSON to Markdown.", extra={"doi": doi, "event_type": "europepmc_success"})
                            await write_to_cache(doi, {"status": "success", "markdown": markdown_content, "source": "EuropePMC_JSON"})
                            return markdown_content, "Retrieved and parsed from Europe PMC (JSON)"
                        else:
                            logger.warning(f"DOI {doi}: Europe PMC JSON parsed to empty/whitespace markdown.", extra={"doi": doi, "event_type": "europepmc_json_parse_empty_md"})
                    else:
                        logger.warning(f"DOI {doi}: Europe PMC JSON parsing failed or returned error structure.", extra={"doi": doi, "event_type": "europepmc_json_parse_fail_or_error_struct"})
                else:
                    logger.debug(f"DOI {doi}: No JSON data returned from Europe PMC.", extra={"doi": doi, "event_type": "europepmc_no_json_data"})
            except Exception as e:
                logger.error(f"DOI {doi}: Error fetching/parsing Europe PMC: {e}", exc_info=True, extra={"doi": doi, "event_type": "europepmc_exception"})

        # Step 2: PMC XML
        if pmcid_full:
            logger.debug(f"DOI {doi}: Attempting PMC XML for PMCID: {pmcid_full}.", extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_attempt"})
            async with SEMAPHORE_NCBI_PMC:
                try:
                    # retrieve_pmc.fetch_pmc_xml uses aiohttp
                    pmc_xml_str = await retrieve_pmc.fetch_pmc_xml(pmcid_full, session=aio_session)
                    if pmc_xml_str:
                        logger.debug(f"DOI {doi}: PMC XML found for PMCID: {pmcid_full}.", extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_found"})
                        structured_elements = parse_json_xml.parse_pmc_xml(pmc_xml_str)
                        if structured_elements and not (len(structured_elements) == 1 and structured_elements[0].get("type") == "error"):
                            markdown_content = parse_json_xml._render_elements_to_markdown_string(structured_elements)
                            if markdown_content and markdown_content.strip():
                                logger.info(f"DOI {doi}: Successfully parsed PMC XML to Markdown for PMCID: {pmcid_full}.", extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_success"})
                                await write_to_cache(doi, {"status": "success", "markdown": markdown_content, "source": "PMC_XML"})
                                return markdown_content, f"Retrieved and parsed from PMC XML (PMCID: {pmcid_full})"
                            else:
                                logger.warning(f"DOI {doi}: PMC XML parsed to empty/whitespace markdown for PMCID: {pmcid_full}.", extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_parse_empty_md"})
                        else:
                            logger.warning(f"DOI {doi}: PMC XML parsing failed or returned error structure for PMCID: {pmcid_full}.", extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_parse_fail_or_error_struct"})
                    else:
                        logger.debug(f"DOI {doi}: No XML data returned from PMC for PMCID: {pmcid_full}.", extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_no_xml_data"})
                except Exception as e:
                    logger.error(f"DOI {doi}: Error fetching/parsing PMC XML for PMCID {pmcid_full}: {e}", exc_info=True, extra={"doi": doi, "pmcid": pmcid_full, "event_type": "pmcxml_exception"})
        else:
            logger.debug(f"DOI {doi}: No PMCID available, skipping PMC XML.", extra={"doi": doi, "event_type": "pmcxml_skip_no_pmcid"})


        # Step 3: Elink & Scrapers
        identifier_for_elink = pmid if pmid else doi
        id_type_for_elink = "pmid" if pmid else "doi"

        if identifier_for_elink:
            logger.debug(f"DOI {doi}: Attempting Elink for {id_type_for_elink}: {identifier_for_elink}.", extra={"doi": doi, "identifier": identifier_for_elink, "id_type": id_type_for_elink, "event_type": "elink_attempt"})
            async with SEMAPHORE_NCBI_ELINK:
                try:
                    article_links = await elink_pubmed.get_article_links(identifier=identifier_for_elink, id_type=id_type_for_elink)
                    if article_links:
                        logger.info(f"DOI {doi}: Elink found {len(article_links)} links for {id_type_for_elink}: {identifier_for_elink}.", extra={"doi": doi, "count": len(article_links), "event_type": "elink_links_count_found"})
                        logger.debug(f"DOI {doi}: Elink links: {article_links}", extra={"doi": doi, "links": article_links, "event_type": "elink_links_list_debug"})
                        for link_url in article_links:
                            logger.debug(f"DOI {doi}: Processing Elink URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_url_processing_start"})
                            async with SEMAPHORE_GENERAL_SCRAPING:
                                content_result = await resolve_pdf_link.resolve_content(link_url, client=http_client)
                            logger.debug(f"DOI {doi}: Elink URL {link_url} resolve_content result: type='{content_result.type}', reason='{getattr(content_result, 'reason', None)}'", extra={"doi": doi, "url": link_url, "resolve_type": content_result.type, "resolve_reason": getattr(content_result, 'reason', None), "event_type": "elink_resolve_content_result"})

                            if content_result.type == "file":
                                md = await parse_pdf_to_markdown(content_result.path)
                                if md:
                                    logger.info(f"DOI {doi}: Elink PDF to Markdown success from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_pdf_success"})
                                    await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Elink_PDF ({link_url})"})
                                    return md, f"Retrieved PDF via Elink ({link_url}) and parsed"
                            elif content_result.type == "html":
                                logger.info(f"DOI {doi}: Elink HTML success from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_html_success"})
                                await write_to_cache(doi, {"status": "success", "markdown": content_result.text, "source": f"Elink_HTML ({link_url})"})
                                return content_result.text, f"Retrieved HTML via Elink ({link_url})"
                            else: # Failure from resolve_content
                                logger.warning(f"DOI {doi}: resolve_content failed for Elink URL: {link_url}. Reason: {content_result.reason}. Trying advanced_scraper.", extra={"doi": doi, "url": link_url, "reason": content_result.reason, "event_type": "elink_resolve_fail_try_advanced"})
                                async with SEMAPHORE_GENERAL_SCRAPING:
                                    advanced_result = await advanced_scraper.scrape_with_fallback(link_url)
                                logger.debug(f"DOI {doi}: Elink URL {link_url} advanced_scraper result: type='{advanced_result.type}', reason='{getattr(advanced_result, 'reason', None)}'", extra={"doi": doi, "url": link_url, "advanced_type": advanced_result.type, "advanced_reason": getattr(advanced_result, 'reason', None), "event_type": "elink_advanced_scraper_result"})
                                if advanced_result.type == "file":
                                    md = await parse_pdf_to_markdown(advanced_result.path)
                                    if md:
                                        logger.info(f"DOI {doi}: Advanced scraper PDF success (from Elink path) from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_advanced_pdf_success"})
                                        await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Elink_AdvancedScraper_PDF ({link_url})"})
                                        return md, f"Retrieved PDF via Elink > Advanced Scraper ({link_url}) and parsed"
                                elif advanced_result.type == "html":
                                    logger.info(f"DOI {doi}: Advanced scraper HTML success (from Elink path) from URL: {link_url}", extra={"doi": doi, "url": link_url, "event_type": "elink_advanced_html_success"})
                                    await write_to_cache(doi, {"status": "success", "markdown": advanced_result.text, "source": f"Elink_AdvancedScraper_HTML ({link_url})"})
                                    return advanced_result.text, f"Retrieved HTML via Elink > Advanced Scraper ({link_url})"
                                else:
                                    logger.warning(f"DOI {doi}: Advanced scraper also failed for Elink URL: {link_url}. Reason: {advanced_result.reason}", extra={"doi": doi, "url": link_url, "reason": advanced_result.reason, "event_type": "elink_advanced_fail"})
                    else:
                        logger.debug(f"DOI {doi}: No links found by Elink for {id_type_for_elink}: {identifier_for_elink}", extra={"doi": doi, "event_type": "elink_no_links"})
                except Exception as e:
                    logger.error(f"DOI {doi}: Error during Elink processing for {id_type_for_elink} {identifier_for_elink}: {e}", exc_info=True, extra={"doi": doi, "event_type": "elink_exception"})
        else:
            logger.debug(f"DOI {doi}: No PMID or DOI available for Elink, skipping.", extra={"doi": doi, "event_type": "elink_skip_no_id"})


        # Step 4: Unpaywall & Scrapers
        logger.debug(f"DOI {doi}: Attempting Unpaywall.", extra={"doi": doi, "event_type": "unpaywall_attempt"})
        async with SEMAPHORE_UNPAYWALL:
            try:
                oa_url = await retrieve_unpaywall.get_unpaywall_oa_url(doi, session=aio_session)
                if oa_url:
                    logger.debug(f"DOI {doi}: Unpaywall found OA URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_url_found"})
                    async with SEMAPHORE_GENERAL_SCRAPING:
                        content_result = await resolve_pdf_link.resolve_content(oa_url, client=http_client)
                    logger.debug(f"DOI {doi}: Unpaywall OA URL {oa_url} resolve_content result: type='{content_result.type}', reason='{getattr(content_result, 'reason', None)}'", extra={"doi": doi, "url": oa_url, "resolve_type": content_result.type, "resolve_reason": getattr(content_result, 'reason', None), "event_type": "unpaywall_resolve_content_result"})

                    if content_result.type == "file":
                        md = await parse_pdf_to_markdown(content_result.path)
                        if md:
                            logger.info(f"DOI {doi}: Unpaywall PDF to Markdown success from URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_pdf_success"})
                            await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Unpaywall_PDF ({oa_url})"})
                            return md, f"Retrieved PDF via Unpaywall ({oa_url}) and parsed"
                    elif content_result.type == "html":
                        logger.info(f"DOI {doi}: Unpaywall HTML success from URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_html_success"})
                        await write_to_cache(doi, {"status": "success", "markdown": content_result.text, "source": f"Unpaywall_HTML ({oa_url})"})
                        return content_result.text, f"Retrieved HTML via Unpaywall ({oa_url})"
                    else: # Failure from resolve_content
                        logger.warning(f"DOI {doi}: resolve_content failed for Unpaywall URL: {oa_url}. Reason: {content_result.reason}. Trying advanced_scraper.", extra={"doi": doi, "url": oa_url, "reason": content_result.reason, "event_type": "unpaywall_resolve_fail_try_advanced"})
                        async with SEMAPHORE_GENERAL_SCRAPING:
                            advanced_result = await advanced_scraper.scrape_with_fallback(oa_url)
                        logger.debug(f"DOI {doi}: Unpaywall OA URL {oa_url} advanced_scraper result: type='{advanced_result.type}', reason='{getattr(advanced_result, 'reason', None)}'", extra={"doi": doi, "url": oa_url, "advanced_type": advanced_result.type, "advanced_reason": getattr(advanced_result, 'reason', None), "event_type": "unpaywall_advanced_scraper_result"})
                        if advanced_result.type == "file":
                            md = await parse_pdf_to_markdown(advanced_result.path)
                            if md:
                                logger.info(f"DOI {doi}: Advanced scraper PDF success (from Unpaywall path) for URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_advanced_pdf_success"})
                                await write_to_cache(doi, {"status": "success", "markdown": md, "source": f"Unpaywall_AdvancedScraper_PDF ({oa_url})"})
                                return md, f"Retrieved PDF via Unpaywall > Advanced Scraper ({oa_url}) and parsed"
                        elif advanced_result.type == "html":
                            logger.info(f"DOI {doi}: Advanced scraper HTML success (from Unpaywall path) for URL: {oa_url}", extra={"doi": doi, "url": oa_url, "event_type": "unpaywall_advanced_html_success"})
                            await write_to_cache(doi, {"status": "success", "markdown": advanced_result.text, "source": f"Unpaywall_AdvancedScraper_HTML ({oa_url})"})
                            return advanced_result.text, f"Retrieved HTML via Unpaywall > Advanced Scraper ({oa_url})"
                        else:
                            logger.warning(f"DOI {doi}: Advanced scraper also failed for Unpaywall URL: {oa_url}. Reason: {advanced_result.reason}", extra={"doi": doi, "url": oa_url, "reason": advanced_result.reason, "event_type": "unpaywall_advanced_fail"})
                else:
                    logger.debug(f"DOI {doi}: No OA URL found by Unpaywall.", extra={"doi": doi, "event_type": "unpaywall_no_url"})
            except Exception as e:
                logger.error(f"DOI {doi}: Error during Unpaywall processing: {e}", exc_info=True, extra={"doi": doi, "event_type": "unpaywall_exception"})

    # If all steps fail
    logger.warning(f"DOI {doi}: All retrieval attempts failed.", extra={"doi": doi, "event_type": "doi_retrieval_failed_all_steps"})
    await write_to_cache(doi, {"status": "failure", "reason": "All retrieval methods failed"})
    return None, "Full text not found after all attempts"


# --- Main Entry Point ---
async def retrieve_full_texts_for_dois(
    query_refiner_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main function to retrieve full texts for DOIs from query_refiner_output.
    Filters DOIs based on relevance_score (4 or 5).
    """
    if not query_refiner_output or "triaged_articles" not in query_refiner_output:
        logger.error("Invalid input: 'triaged_articles' not found in query_refiner_output.", extra={"event_type": "invalid_input_structure"})
        return query_refiner_output # Or raise error

    articles_to_process = [
        article for article in query_refiner_output.get("triaged_articles", [])
        if isinstance(article, dict) and article.get("average_relevance_score", 0.0) >= 4.0 and article.get("doi")
    ]

    original_articles_list = query_refiner_output.get("triaged_articles", [])
    output_articles = []

    if not articles_to_process:
        logger.info("No articles with relevance score 4 or 5 found to process. Marking all as skipped.", extra={"event_type": "no_relevant_articles"})
        for article in original_articles_list:
            updated_article = article.copy()
            updated_article["fulltext_retrieval_status"] = "skipped_relevance"
            updated_article["fulltext_retrieval_message"] = "Not processed due to relevance score"
            output_articles.append(updated_article)
        query_refiner_output["triaged_articles"] = output_articles
        return query_refiner_output

    logger.info(f"Processing {len(articles_to_process)} articles for full text retrieval.", extra={"count": len(articles_to_process), "event_type": "processing_start"})

    # Create a list of tasks to run concurrently
    tasks = []
    for article_data in articles_to_process:
        # Each task will call get_full_text_for_doi
        # We need to ensure that the article_data dictionary itself is updated or a new list is formed.
        # Let's store (index, task) to map results back.
        tasks.append(get_full_text_for_doi(article_data.copy())) # Pass a copy to avoid modification issues if original list is used elsewhere

    # Run tasks concurrently, respecting MAX_CONCURRENT_DOI_PROCESSING
    # This semaphore limits how many get_full_text_for_doi coroutines run at once.
    # Individual semaphores inside get_full_text_for_doi control service-specific concurrency.
    processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOI_PROCESSING)
    
    async def worker(task_coro):
        async with processing_semaphore:
            return await task_coro

    # Gather results
    # raw_results will be a list of (markdown_content, status_message, tried_sources) tuples
    # or Exception instances, in the same order as articles_to_process
    raw_results = await asyncio.gather(*(worker(task) for task in tasks), return_exceptions=True)
    
    # Create a map of DOI to its result for easier lookup
    # The value in the map will be (markdown_content, status_message, tried_sources_list)
    doi_to_result_map: Dict[str, Tuple[Optional[str], str, List[str]]] = {}
    for i, article_data in enumerate(articles_to_process):
        doi = article_data["doi"]
        result_item = raw_results[i]
        if isinstance(result_item, Exception):
            logger.error(f"Unhandled exception for DOI {doi}: {result_item}", exc_info=result_item, extra={"doi": doi, "event_type": "doi_processing_unhandled_exception"})
            # Ensure a list of tried_sources (empty if error before any attempt)
            doi_to_result_map[doi] = (None, f"Error: {str(result_item)}", []) 
        elif isinstance(result_item, tuple) and len(result_item) == 3:
            doi_to_result_map[doi] = result_item # (markdown, status, tried_sources)
        else:
            # Should not happen if get_full_text_for_doi always returns 3 items or raises
            logger.error(f"Unexpected result format for DOI {doi}: {result_item}", extra={"doi": doi, "event_type": "doi_processing_unexpected_result_format"})
            doi_to_result_map[doi] = (None, "Error: Unexpected result format from retrieval function", [])


    # Iterate through the original list of triaged_articles and update them
    for original_article in original_articles_list: 
        updated_article = original_article.copy() 
        doi = original_article.get("doi")
        
        if doi and doi in doi_to_result_map: 
            markdown_content, status_message, tried_sources_list = doi_to_result_map[doi]
            
            # Update status message if it's a failure and we have tried_sources
            if not markdown_content and tried_sources_list:
                 status_message = f"Full text not found. Tried: {', '.join(tried_sources_list)}."
            elif not markdown_content: # No markdown, no tried_sources (e.g. early error)
                 status_message = status_message # Keep original error message
            
            if markdown_content:
                updated_article["fulltext"] = markdown_content
                updated_article["fulltext_retrieval_status"] = "success"
                updated_article["fulltext_retrieval_message"] = status_message
            else:
                updated_article["fulltext"] = None
                updated_article["fulltext_retrieval_status"] = "failure"
                updated_article["fulltext_retrieval_message"] = status_message
            
            # Add tried_sources to the article for transparency, if desired
            # updated_article["fulltext_retrieval_sources_attempted"] = tried_sources_list

        elif original_article.get("average_relevance_score", 0.0) < 4.0: # Changed from 'not in [4,5]' to '< 4.0'
            updated_article["fulltext_retrieval_status"] = "skipped_relevance"
            updated_article["fulltext_retrieval_message"] = "Not processed due to relevance score"
        else:
            updated_article["fulltext_retrieval_status"] = "skipped_no_doi"
            updated_article["fulltext_retrieval_message"] = "Not processed, DOI missing or other issue pre-processing"
            
        output_articles.append(updated_article)

    query_refiner_output["triaged_articles"] = output_articles
    logger.info("Finished processing all selected DOIs.", extra={"event_type": "processing_end"})
    return query_refiner_output


# --- Command-line interface for testing ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/retrieve_full_text.py <path_to_query_refiner_output_json>")
        # Create a dummy input for basic testing
        dummy_input_path = "workspace/dummy_query_refiner_output.json"
        dummy_data = {
            "query": "investigation of false negatives",
            "refined_queries": ["false negative investigation"],
            "triaged_articles": [
                {
                    "title": "Investigative Article 1 (False Negative)",
                    "doi": "10.1021/acs.jmedchem.9b01279",
                    "pmid": "", # Will be fetched if necessary by Elink
                    "pmcid": "", # Will be fetched if necessary
                    "relevance_score": 5 
                }
                # Add more problematic DOIs here for batch testing if needed later
            ]
        }
        with open(dummy_input_path, 'w') as f:
            json.dump(dummy_data, f, indent=2)
        print(f"Created dummy input file: {dummy_input_path}")
        input_file_path = dummy_input_path
    else:
        input_file_path = sys.argv[1]

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found: {input_file_path}")
        sys.exit(1)

    with open(input_file_path, 'r') as f:
        test_input_data = json.load(f)

    # Configure logging for CLI test
    # Removed (%(event_type)s) to prevent errors if event_type is not in all log records from all modules
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    # Ensure our specific logger (and its handlers) are also at INFO if they were set to something else.
    # However, the logger instance at the top of the file already sets its level.
    # This basicConfig will affect the root logger and other loggers that don't have specific handlers.
    # logger.setLevel(logging.INFO) # This line is redundant if the top-level logger is already configured

    print(f"Starting retrieval process for file: {input_file_path}")
    results = asyncio.run(retrieve_full_texts_for_dois(test_input_data))
    
    output_file_path = "workspace/retrieve_full_text_output.json"
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing complete. Results saved to: {output_file_path}")

    # Print summary of results
    for article in results.get("triaged_articles", []):
        doi = article.get('doi', 'N/A')
        status = article.get('fulltext_retrieval_status', 'N/A')
        message = article.get('fulltext_retrieval_message', '')
        has_fulltext = "Yes" if article.get("fulltext") else "No"
        print(f"DOI: {doi:<30} | Status: {status:<20} | Has Fulltext: {has_fulltext:<3} | Message: {message}")


# --- AutoGen Agent Definition ---
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from typing import Sequence


class FullTextRetrievalAgent(BaseChatAgent):
    def __init__(
        self,
        name: str,
        description: Optional[str] = "Retrieves full text for a list of articles.",
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """The types of messages that the agent produces."""
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        """Handles incoming messages and returns a response with enriched articles."""
        if not messages:
            return Response(
                chat_message=TextMessage(content="Error: No input messages received.", source=self.name),
                inner_messages=[],
            )

        # Assuming the input is in the last message, as a JSON string of List[Dict[str, Any]]
        last_message = messages[-1]
        if not isinstance(last_message, TextMessage) or not isinstance(last_message.content, str):
            return Response(
                chat_message=TextMessage(
                    content="Error: Expected a TextMessage with a JSON string content.", source=self.name
                ),
                inner_messages=[],
            )

        try:
            input_articles: List[Dict[str, Any]] = json.loads(last_message.content)
            if not isinstance(input_articles, list):
                raise ValueError("Parsed JSON is not a list.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse input JSON: {e}", extra={"input_content": last_message.content})
            return Response(
                chat_message=TextMessage(content=f"Error: Invalid JSON input. {e}", source=self.name),
                inner_messages=[],
            )
        except ValueError as e:
            logger.error(f"Parsed JSON is not a list: {e}", extra={"input_content": last_message.content})
            return Response(
                chat_message=TextMessage(content=f"Error: Input JSON must be a list of articles. {e}", source=self.name),
                inner_messages=[],
            )

        # The retrieve_full_texts_for_dois function expects a specific dictionary structure.
        # We'll wrap our input_articles list into that structure.
        # We use a dummy query and refined_queries as they are part of the expected structure
        # but not directly used by the full-text retrieval logic for each article.
        mock_query_refiner_output = {
            "query": "Full-text retrieval task",
            "refined_queries": [],
            "triaged_articles": input_articles  # This is the list we want to process
        }

        try:
            # Call the existing function to retrieve full texts
            # This function modifies mock_query_refiner_output in-place
            # and also returns it.
            enriched_output_dict = await retrieve_full_texts_for_dois(mock_query_refiner_output)
            
            # Extract the enriched articles list
            enriched_articles_list = enriched_output_dict.get("triaged_articles", [])

            output_json = json.dumps(enriched_articles_list, indent=2)
            response_message = TextMessage(content=output_json, source=self.name)
        except Exception as e:
            logger.error(f"Error during full text retrieval: {e}", exc_info=True)
            response_message = TextMessage(
                content=f"Error during full text retrieval: {str(e)}", source=self.name
            )

        return Response(chat_message=response_message, inner_messages=[])

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Resets the agent to its initialization state."""
        # This agent is stateless, so nothing to do here.
        pass
