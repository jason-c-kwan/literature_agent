"""
Resolves a URL to determine if it yields HTML full text or a downloadable PDF.
"""
import asyncio
import datetime
import os
import logging
import json
from typing import Literal, NamedTuple, Union, List, Dict, Optional # Added Dict, Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse, quote_plus

import httpx
import yaml
from bs4 import BeautifulSoup
from readability import Document # readability-lxml
import pymupdf4llm # For checking PDF content
import fitz # PyMuPDF, pymupdf4llm is a wrapper around this

from consts.http_headers import DEFAULT_HEADERS

# --- Configuration ---
CONFIG_PATH = "config/settings.yaml"
PDF_UNLOCK_PARAMS_PATH = "config/pdf_unlock_params.yaml"

# Default values if not in config
DEFAULT_TIMEOUT = 15
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_FACTOR = 0.5
DEFAULT_RETRY_STATUS_FORCELIST = [500, 502, 503, 504]
DEFAULT_PER_DOMAIN_CONCURRENCY = 2
MIN_PDF_SIZE_KB = 10
MIN_HTML_CONTENT_LENGTH = 200
MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE = 7000
DOWNLOADS_DIR = "workspace/downloads"

def load_config():
    """Loads configuration from settings.yaml."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

config = load_config()
REQUEST_TIMEOUT = config.get("resolve_content_timeout", DEFAULT_TIMEOUT)
MAX_RETRIES = config.get("resolve_max_retries", DEFAULT_MAX_RETRIES)
RETRY_BACKOFF_FACTOR = config.get("resolve_retry_backoff_factor", DEFAULT_RETRY_BACKOFF_FACTOR)
RETRY_STATUS_FORCELIST = config.get("resolve_retry_status_forcelist", DEFAULT_RETRY_STATUS_FORCELIST)
PER_DOMAIN_CONCURRENCY = config.get("resolve_per_domain_concurrency", DEFAULT_PER_DOMAIN_CONCURRENCY)

# --- Logging Setup ---
class JsonFormatter(logging.Formatter):
    CORE_LOG_KEYS = {
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'message', 'module',
        'msecs', 'msg', 'name', 'pathname', 'process', 'processName',
        'relativeCreated', 'stack_info', 'thread', 'threadName',
        'timestamp', 'level', 'function', 'line', 
    }

    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(), 
        }
        log_entry['event'] = getattr(record, 'event_type', record.msg.split(' ')[0] if isinstance(record.msg, str) else 'generic_event')
        for key, value in record.__dict__.items():
            if key not in self.CORE_LOG_KEYS and key not in log_entry:
                log_entry[key] = value
        return json.dumps(log_entry)

logger = logging.getLogger("resolve_pdf_link")
if not logger.handlers: 
    handler = logging.StreamHandler() 
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- PDF Unlock Params Loading ---
_pdf_unlock_params_cache = None
def load_pdf_unlock_params():
    global _pdf_unlock_params_cache
    if _pdf_unlock_params_cache is not None:
        return _pdf_unlock_params_cache
    try:
        with open(PDF_UNLOCK_PARAMS_PATH, "r", encoding="utf-8") as f:
            _pdf_unlock_params_cache = yaml.safe_load(f)
            if _pdf_unlock_params_cache is None:
                 _pdf_unlock_params_cache = {}
            logger.info(f"Successfully loaded PDF unlock params from {PDF_UNLOCK_PARAMS_PATH}", extra={"event_type": "pdf_unlock_params_loaded"})
            return _pdf_unlock_params_cache
    except FileNotFoundError:
        logger.warning(f"PDF unlock params file not found: {PDF_UNLOCK_PARAMS_PATH}. Using empty config.", extra={"event_type": "pdf_unlock_params_not_found"})
        _pdf_unlock_params_cache = {}
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing PDF unlock params file {PDF_UNLOCK_PARAMS_PATH}: {e}. Using empty config.", extra={"event_type": "pdf_unlock_params_parse_error"})
        _pdf_unlock_params_cache = {}
        return {}

# --- DOI Extraction Helper (for Referer) ---
import re # Add to imports if not already there (should be at top)
# Optional is already imported above now

def _extract_doi_from_acs_url(url_str: str) -> Optional[str]:
    # Regex for typical ACS DOI patterns. ACS DOIs usually start with 10.1021/
    # This regex is designed to be quite general for things that look like DOIs.
    match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)', url_str, re.IGNORECASE)
    return match.group(1) if match else None

# --- PDF Retry URL Generation ---
def generate_pdf_retry_urls(original_url: str) -> List[str]:
    urls_to_try = []
    try:
        parsed_original_url = urlparse(original_url)
        original_path = parsed_original_url.path
        original_query_params = parse_qs(parsed_original_url.query, keep_blank_values=True)

        unlock_params_config = load_pdf_unlock_params()
        # Ensure 'default' and domain specific params are lists, even if not present in YAML
        
        netloc = parsed_original_url.netloc
        domain_specific_params_list = unlock_params_config.get(netloc, []) or []
        # If no params found for full netloc (e.g., www.example.com), try base domain (e.g., example.com)
        if not domain_specific_params_list and netloc.startswith("www."):
            base_domain = netloc[4:]
            domain_specific_params_list = unlock_params_config.get(base_domain, []) or []
        
        default_params_list = unlock_params_config.get("default", []) or []
        
        seen_param_tuples = set()
        ordered_params_to_try_tuples = []
        for p_list in [domain_specific_params_list, default_params_list]:
            for p_str in p_list:
                try:
                    param_key, param_value = p_str.split("=", 1)
                    if (param_key, param_value) not in seen_param_tuples:
                        ordered_params_to_try_tuples.append((param_key, param_value))
                        seen_param_tuples.add((param_key, param_value))
                except ValueError:
                    logger.warning(f"Malformed parameter string '{p_str}' in PDF unlock config.", extra={"param_string": p_str, "event_type": "pdf_unlock_param_malformed"})

        path_variants = [original_path]
        if "/pdf/" in original_path.lower(): # Only add variants if /pdf/ is in the original path
            path_variants.append(original_path.lower().replace("/pdf/", "/epdf/", 1))
            path_variants.append(original_path.lower().replace("/pdf/", "/pdfdirect/", 1))
        
        # Deduplicate path_variants in case original_path itself was like '/epdf/'
        path_variants = list(dict.fromkeys(path_variants))

        # Always start with the original URL as provided
        if original_url not in urls_to_try:
            urls_to_try.append(original_url)

        for path_variant in path_variants:
            # URL with current path_variant and original query parameters
            url_with_path_variant_original_query = urlunparse(
                (parsed_original_url.scheme,
                 parsed_original_url.netloc,
                 path_variant,
                 parsed_original_url.params,
                 urlencode(original_query_params, doseq=True),
                 parsed_original_url.fragment)
            )
            if url_with_path_variant_original_query not in urls_to_try:
                urls_to_try.append(url_with_path_variant_original_query)

            # Apply domain-specific parameters to this path_variant
            for p_str in domain_specific_params_list:
                try:
                    param_key, param_value = p_str.split("=", 1)
                    current_variant_query_params = parse_qs(urlparse(url_with_path_variant_original_query).query, keep_blank_values=True)
                    current_variant_query_params[param_key] = [param_value] # Add/override
                    
                    retry_url_domain_specific = urlunparse(
                        (parsed_original_url.scheme,
                         parsed_original_url.netloc,
                         path_variant, # Current path variant
                         parsed_original_url.params,
                         urlencode(current_variant_query_params, doseq=True),
                         parsed_original_url.fragment)
                    )
                    if retry_url_domain_specific not in urls_to_try:
                        urls_to_try.append(retry_url_domain_specific)
                except ValueError: # Already logged if p_str is malformed
                    pass
            
            # Apply default parameters to this path_variant (if not already applied by domain-specific)
            for p_str in default_params_list:
                try:
                    param_key, param_value = p_str.split("=", 1)
                    # Check if this exact param key-value was already added by domain-specific logic for this path_variant
                    # This is a bit complex to check perfectly without re-parsing all generated URLs.
                    # A simpler approach: add it, and deduplicate later. Or, check if the param key is already in original_query_params
                    # or was part of domain_specific_params_list.
                    # For now, let's add and rely on final deduplication, but be mindful of order.
                    
                    current_variant_query_params = parse_qs(urlparse(url_with_path_variant_original_query).query, keep_blank_values=True)
                    
                    # Only add default param if not already present from original or domain-specific for this key
                    # This logic might be too simple if multiple defaults for same key exist.
                    # The ordered_params_to_try_tuples was better for handling precedence.
                    # Reverting to a similar loop structure for params for clarity:

                    # Let's refine: iterate through combined unique params, applying them
                    # This part needs to be careful not to duplicate effort or create too many variants.
                    # The original ordered_params_to_try_tuples was good.
                    # We need to apply these ordered_params_to_try_tuples to each path_variant.

                except ValueError:
                    pass # Already logged

        # Re-iterate with the ordered_params_to_try_tuples for each path variant to ensure correct precedence
        # This will replace the separate domain/default loops above for parameter application.
        
        # Clear urls_to_try except for the initial set of path_variants with original queries
        base_urls_for_param_application = []
        if original_url not in base_urls_for_param_application: base_urls_for_param_application.append(original_url)
        for path_variant in path_variants:
            url_with_path_variant_original_query = urlunparse(
                (parsed_original_url.scheme, parsed_original_url.netloc, path_variant, 
                 parsed_original_url.params, urlencode(original_query_params, doseq=True), parsed_original_url.fragment)
            )
            if url_with_path_variant_original_query not in base_urls_for_param_application:
                base_urls_for_param_application.append(url_with_path_variant_original_query)
        
        urls_to_try = list(dict.fromkeys(base_urls_for_param_application)) # Start with unique base URLs (original + path variants)

        # Now, for each of these base URLs (original + path variants with original query), apply the ordered unlock parameters
        newly_generated_with_unlock_params = []
        for base_url_for_params in urls_to_try: # urls_to_try currently holds (original_url + path_variants_with_original_query)
            parsed_base_for_params = urlparse(base_url_for_params)
            # Important: when applying unlock params, we usually want to apply them to the path without original query params,
            # or ensure they override. The current base_query_params are from the original URL.
            # Let's try applying to path without original query, and also path + original_query + unlock_param.

            # Scenario 1: Path variant + unlock param (ignoring original query)
            path_for_unlock_param = parsed_base_for_params.path # Path from current base_url_for_params
            
            for param_key, param_value in ordered_params_to_try_tuples:
                query_dict_unlock_param_only = {param_key: [param_value]} # Create query with only the unlock param
                
                url_path_variant_with_unlock_param = urlunparse(
                    (parsed_base_for_params.scheme,
                     parsed_base_for_params.netloc,
                     path_for_unlock_param,
                     parsed_base_for_params.params, # Usually empty
                     urlencode(query_dict_unlock_param_only, doseq=True),
                     parsed_base_for_params.fragment) # Usually empty
                )
                newly_generated_with_unlock_params.append(url_path_variant_with_unlock_param)

                # Scenario 2: Path variant + original query + unlock param (unlock param overrides if key matches)
                # This is what the previous logic did with new_query_params_for_variant = base_query_params.copy()
                # base_query_params here are from the original URL, applied to the current path_variant
                original_query_for_current_path = parse_qs(parsed_base_for_params.query, keep_blank_values=True)
                query_combined = original_query_for_current_path.copy()
                query_combined[param_key] = [param_value] # Override/add unlock param

                url_path_variant_orig_q_plus_unlock_param = urlunparse(
                    (parsed_base_for_params.scheme,
                     parsed_base_for_params.netloc,
                     path_for_unlock_param,
                     parsed_base_for_params.params,
                     urlencode(query_combined, doseq=True),
                     parsed_base_for_params.fragment)
                )
                newly_generated_with_unlock_params.append(url_path_variant_orig_q_plus_unlock_param)

        # Combine initial list (original_url + path_variants_with_original_query)
        # with the newly_generated_with_unlock_params.
        # The order should be: initial urls, then newly_generated_with_unlock_params.
        # Deduplication at the end preserves the first occurrence.
        combined_urls = urls_to_try + newly_generated_with_unlock_params
        final_urls_to_try = list(dict.fromkeys(combined_urls))

    except Exception as e:
        logger.error(f"Error generating PDF retry URLs for {original_url}: {e}", exc_info=True, extra={"original_url": original_url, "event_type": "generate_pdf_retry_url_error"})
        # Fallback to just the original URL if generation fails catastrophically
        final_urls_to_try = [original_url] if original_url not in (urls_to_try or []) else (urls_to_try or [original_url])
        if original_url not in final_urls_to_try: # Ensure original is always there on error
             final_urls_to_try.insert(0, original_url)
        final_urls_to_try = list(dict.fromkeys(final_urls_to_try))


    logger.info(f"Generated {len(final_urls_to_try)} unique URLs for PDF retry for {original_url}", extra={"original_url": original_url, "count": len(final_urls_to_try), "generated_urls": final_urls_to_try, "event_type": "pdf_retry_urls_generated"})
    return final_urls_to_try

# --- Per-Domain Rate Limiting ---
domain_semaphores = {}
domain_semaphores_lock = asyncio.Lock()

async def get_domain_semaphore(url: str) -> asyncio.Semaphore:
    domain = urlparse(url).netloc
    if not domain: 
        logger.warning("Could not parse domain from URL", extra={"url": url, "event_type": "domain_parse_failure"})
        domain = "__no_domain__"
    async with domain_semaphores_lock:
        if domain not in domain_semaphores:
            logger.info(f"Creating new semaphore for domain: {domain}", extra={"domain": domain, "concurrency": PER_DOMAIN_CONCURRENCY, "event_type": "semaphore_create"})
            domain_semaphores[domain] = asyncio.Semaphore(PER_DOMAIN_CONCURRENCY)
        return domain_semaphores[domain]

# --- Result types ---
class Failure(NamedTuple):
    type: Literal["failure"]
    reason: str
class HTMLResult(NamedTuple):
    type: Literal["html"]
    text: str
class FileResult(NamedTuple):
    type: Literal["file"]
    path: str
ResolveResult = Union[Failure, HTMLResult, FileResult]

PAYWALL_KEYWORDS = [
    "access this article", "buy this article", "purchase access", "subscribe to view", 
    "institutional login", "access options", "get access", "full text access", 
    "journal subscription", "pay per view", "purchase pdf", "rent this article",
    "limited preview", "unlock this article", "sign in to read", "£", "$", "€", "usd", "eur", "gbp"
]

def is_pdf_content_valid(file_path: str) -> tuple[bool, str]:
    try:
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb < MIN_PDF_SIZE_KB:
            return False, f"File size ({file_size_kb:.2f}KB) is less than minimum ({MIN_PDF_SIZE_KB}KB)."
        doc = fitz.open(file_path)
        page_count = len(doc)
        if page_count == 0:
            doc.close(); return False, "PDF has 0 pages."
        if page_count == 1:
            text = ""
            try: page = doc.load_page(0); text = page.get_text("text")
            finally: doc.close()
            text_original_for_len_check = text
            text_lower_stripped = text.lower().strip()
            if text_lower_stripped == "dummy pdf file" or \
               ("dummy" in text_lower_stripped and "pdf" in text_lower_stripped and len(text_original_for_len_check) < 200) or \
               (len(text_original_for_len_check) < 100 and any(kw in text_lower_stripped for kw in ["placeholder", "abstract only", "cover page"])):
                 return False, "PDF content suggests it's a dummy, placeholder, or abstract-only."
            if len(text_lower_stripped) < 20:
                return False, f"Single-page PDF has very little text content (approx. {len(text_lower_stripped)} chars)."
        else: # More than 1 page
            doc.close()
        return True, ""
    except Exception as e:
        logger.error(f"Error checking PDF content for {file_path}", exc_info=True, extra={"file_path": file_path, "event_type": "pdf_validation_error"})
        return False, f"Error checking PDF content: {e!s}"

def is_html_potentially_paywalled(full_html_content: str) -> bool:
    html_lower = full_html_content.lower()
    matches = sum(1 for keyword in PAYWALL_KEYWORDS if keyword in html_lower)
    if any(currency in html_lower for currency in ["£", "$", "€", "usd", "eur", "gbp"]) and matches >= 1: return True
    if matches >= 2: return True
    if "access this article for" in html_lower and "buy this article" in html_lower: return True
    if "institutional login" in html_lower and ("subscribe" in html_lower or "purchase" in html_lower): return True
    return False

async def _make_request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    request_specific_headers = kwargs.pop("headers", {})
    final_headers = {**DEFAULT_HEADERS, **request_specific_headers}
    if "User-Agent" not in final_headers: # Should be covered by DEFAULT_HEADERS
        final_headers["User-Agent"] = DEFAULT_HEADERS["User-Agent"]

    # --- Cookie Injection (Task 2b) ---
    passed_cookies = kwargs.pop("passed_cookies", None) # Extract from kwargs if present
    if passed_cookies:
        target_domain = urlparse(url).netloc
        relevant_cookies_for_domain = [
            c for c in passed_cookies
            if c.get('domain') and target_domain.endswith(c['domain'].lstrip('.'))
        ]
        if relevant_cookies_for_domain:
            cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in relevant_cookies_for_domain if c.get('name') and c.get('value')])
            if cookie_str:
                if "Cookie" in final_headers: # Merge
                    final_headers["Cookie"] = f"{final_headers['Cookie']}; {cookie_str}"
                else:
                    final_headers["Cookie"] = cookie_str
                logger.info(f"Using {len(relevant_cookies_for_domain)} cookies for domain {target_domain} for URL {url}", extra={"url":url, "event_type":"cookie_injection_resolve_pdf_link", "num_cookies": len(relevant_cookies_for_domain)})
    
    for attempt in range(MAX_RETRIES + 1): 
        log_extra = {"url": url, "method": method, "attempt": attempt + 1, "max_retries": MAX_RETRIES}
        try:
            logger.info(f"Requesting {method} {url}, attempt {attempt + 1}", extra={**log_extra, "headers_sent": final_headers, "event_type": f"{method.lower()}_attempt"})
            response = await client.request(method, url, timeout=REQUEST_TIMEOUT, headers=final_headers, **kwargs)
            response.raise_for_status()
            logger.info(f"{method} request to {url} successful (status {response.status_code})", extra={**log_extra, "status_code": response.status_code, "event_type": f"{method.lower()}_success"})
            return response
        except httpx.HTTPStatusError as e:
            log_extra["status_code"] = e.response.status_code; log_extra["error"] = str(e)
            if e.response.status_code in RETRY_STATUS_FORCELIST and attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(f"{method} request to {url} failed with {e.response.status_code}, retrying in {delay:.2f}s...", extra={**log_extra, "delay": delay, "event_type": "retry_scheduled_status"})
                await asyncio.sleep(delay)
            else:
                logger.error(f"{method} request to {url} failed with {e.response.status_code}, no more retries or non-retriable.", extra={**log_extra, "event_type": "request_failure_status"})
                raise 
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            log_extra["error"] = str(e)
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(f"{method} request to {url} failed with {type(e).__name__}, retrying in {delay:.2f}s...", extra={**log_extra, "delay": delay, "event_type": "retry_scheduled_network"})
                await asyncio.sleep(delay)
            else:
                logger.error(f"{method} request to {url} failed with {type(e).__name__}, no more retries.", extra={**log_extra, "event_type": "request_failure_network"})
                raise 
    raise httpx.RequestError(f"Max retries ({MAX_RETRIES}) exceeded for {method} {url}")

async def resolve_content(
    url: str, 
    client: httpx.AsyncClient = None, 
    original_doi_for_referer: Optional[str] = None, # For Task 4
    session_cookies: Optional[List[Dict]] = None # For Task 2c
) -> ResolveResult:
    provided_client = bool(client)
    if not client: client = httpx.AsyncClient(follow_redirects=True)

    logger.info(f"Starting to resolve content for URL: {url}", extra={"url": url, "event_type": "resolve_start"})
    semaphore = await get_domain_semaphore(url)
    
    async with semaphore:
        logger.info(f"Semaphore acquired for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_acquired"})
        try:
            unlock_params_config = load_pdf_unlock_params()
            block_head_domains = unlock_params_config.get("block_head", [])
            parsed_url_for_head_check = urlparse(url)
            skip_head = parsed_url_for_head_check.netloc in block_head_domains

            if not skip_head:
                try:
                    logger.info(f"Attempting HEAD request for {url}", extra={"url": url, "event_type": "head_attempt"})
                    head_response = await _make_request_with_retry(client, "HEAD", url, passed_cookies=session_cookies) # Pass cookies
                    content_type_head = head_response.headers.get("content-type", "").lower()
                    if content_type_head.startswith("application/pdf"):
                        logger.info(f"HEAD indicates PDF for {url}. Proceeding to GET.", extra={"url": url, "event_type": "head_pdf_detected_get_attempt"})
                        pdf_response_from_head_path = await _make_request_with_retry(client, "GET", url, headers={"Accept": "application/pdf,*/*"}, passed_cookies=session_cookies) # Pass cookies
                        if not os.path.exists(DOWNLOADS_DIR): os.makedirs(DOWNLOADS_DIR)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        url_part = "".join(c if c.isalnum() else "_" for c in url.split("://")[-1][:50])
                        file_name = f"{timestamp}_{url_part}_head.pdf" # Indicate source
                        file_path = os.path.join(DOWNLOADS_DIR, file_name)
                        with open(file_path, "wb") as f: f.write(pdf_response_from_head_path.content)
                        is_valid, reason = is_pdf_content_valid(file_path)
                        if is_valid:
                            logger.info(f"PDF from HEAD path {file_path} is valid.", extra={"url": url, "path": file_path, "event_type": "pdf_head_valid_after_get"})
                            return FileResult(type="file", path=file_path)
                        else:
                            logger.warning(f"PDF from HEAD path {file_path} invalid: {reason}. Deleting.", extra={"url": url, "path": file_path, "reason": reason, "event_type": "pdf_head_invalid_after_get"})
                            try: os.remove(file_path)
                            except OSError: pass # Fall through to main GET logic
                except httpx.HTTPStatusError as e_head:
                    logger.info(f"HEAD for {url} failed with {e_head.response.status_code}. Proceeding with GETs.", extra={"url": url, "status": e_head.response.status_code, "event_type": "head_fail_status_try_get"})
                except httpx.RequestError as e_head_req:
                     logger.warning(f"HEAD for {url} failed: {e_head_req!s}. Proceeding with GETs.", extra={"url": url, "error": str(e_head_req), "event_type": "head_fail_request_try_get"})
            else:
                logger.info(f"Skipping HEAD for {url} (domain in block_head).", extra={"url": url, "domain": parsed_url_for_head_check.netloc, "event_type": "head_skip_blocked_domain"})

            urls_to_attempt = generate_pdf_retry_urls(url)
            last_get_exception = None

            for attempt_idx, current_url_to_try in enumerate(urls_to_attempt):
                get_request_headers = {} # Specific headers for this GET attempt
                path_lower = urlparse(current_url_to_try).path.lower()
                is_pdf_like_path = any(p in path_lower for p in ["/pdf/", "/epdf/", "/pdfdirect/"])
                
                # Prioritize PDF if path suggests it or if it's a non-original retry URL (could be param variant)
                if is_pdf_like_path or current_url_to_try != url:
                    get_request_headers["Accept"] = "application/pdf,*/*"
                # Else, _make_request_with_retry will use DEFAULT_HEADERS["Accept"] which is "*/*"
                # If a more specific HTML accept is desired for the first original URL:
                # elif current_url_to_try == url and attempt_idx == 0: # First attempt on original URL
                #    get_request_headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7"

                logger.info(f"Attempting GET {current_url_to_try} ({attempt_idx+1}/{len(urls_to_attempt)})", extra={"url_attempt": current_url_to_try, "event_type": "resolve_get_attempt"})
                try:
                    headers_for_this_get = get_request_headers.copy() # Start with general headers for this attempt type

                    # --- Referer Header for ACS PDF-like paths (Task 4) ---
                    current_url_parsed = urlparse(current_url_to_try)
                    if "pubs.acs.org" in current_url_parsed.netloc and \
                       any(p_segment in current_url_parsed.path.lower() for p_segment in ["/pdf/", "/epdf/"]):
                        
                        # Determine DOI for Referer: use passed original_doi_for_referer if available,
                        # otherwise try to extract from the initial URL given to resolve_content,
                        # or finally from the current URL being attempted.
                        doi_for_referer = original_doi_for_referer
                        if not doi_for_referer:
                            doi_for_referer = _extract_doi_from_acs_url(url) # 'url' is the original URL to resolve_content
                        if not doi_for_referer: # Fallback to current URL if still not found
                            doi_for_referer = _extract_doi_from_acs_url(current_url_to_try)

                        if doi_for_referer:
                            referer_value = f"https://pubs.acs.org/doi/{doi_for_referer}"
                            headers_for_this_get["Referer"] = referer_value
                            logger.info(f"Added Referer: {referer_value} for ACS URL: {current_url_to_try}", 
                                        extra={"url": current_url_to_try, "referer": referer_value, "event_type": "referer_added_acs"})
                    
                    get_response = await _make_request_with_retry(client, "GET", current_url_to_try, headers=headers_for_this_get, passed_cookies=session_cookies)
                    content_type_get = get_response.headers.get("content-type", "").lower()

                    if "html" in content_type_get:
                        # Process HTML only if it's not a clearly PDF-intended path that returned HTML (which is an error for that path)
                        if not is_pdf_like_path or (is_pdf_like_path and current_url_to_try == url and attempt_idx == 0) : # Original URL might be HTML even if path has /pdf/
                            html_content = get_response.text
                            logger.info(f"GET for {current_url_to_try} returned HTML.", extra={"url": current_url_to_try, "event_type": "get_html_detected"})
                            paywalled = is_html_potentially_paywalled(html_content)
                            extracted_text = None
                            try:
                                doc = Document(html_content); main_content_html = doc.summary(html_partial=True)
                                summary_soup = BeautifulSoup(main_content_html, "html.parser"); extracted_text = summary_soup.get_text(separator="\\n", strip=True)
                                if extracted_text and len(extracted_text) > MIN_HTML_CONTENT_LENGTH: logger.info(f"Extracted HTML (readability) from {current_url_to_try}", extra={"len":len(extracted_text)})
                            except Exception as e_read: logger.warning(f"Readability failed for {current_url_to_try}: {e_read!s}", extra={"err":str(e_read)})

                            if not extracted_text or len(extracted_text) < MIN_HTML_CONTENT_LENGTH:
                                article_tag = BeautifulSoup(html_content, "html.parser").find("article")
                                if article_tag: extracted_text = article_tag.get_text(separator="\\n", strip=True)
                                if extracted_text and len(extracted_text) > MIN_HTML_CONTENT_LENGTH: logger.info(f"Extracted HTML (<article>) from {current_url_to_try}", extra={"len":len(extracted_text)})
                            
                            if extracted_text:
                                if len(extracted_text) >= MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE: return HTMLResult(type="html", text=extracted_text)
                                if paywalled: last_get_exception = Failure(type="failure", reason="Paywalled HTML and short content."); continue
                                if len(extracted_text) >= MIN_HTML_CONTENT_LENGTH: return HTMLResult(type="html", text=extracted_text)
                                last_get_exception = Failure(type="failure", reason="HTML too short."); continue
                            else: last_get_exception = Failure(type="failure", reason="No main content extracted from HTML."); continue
                        else: # HTML from a PDF-specific path variant - treat as failure for this path
                            logger.warning(f"Expected PDF, got HTML from PDF-path variant {current_url_to_try}", extra={"url":current_url_to_try, "event_type":"html_from_pdf_path_variant"})
                            last_get_exception = Failure(type="failure", reason=f"Expected PDF, got HTML from {current_url_to_try}")
                            continue
                    
                    elif content_type_get.startswith("application/pdf"):
                        logger.info(f"GET indicates PDF for {current_url_to_try}. Saving.", extra={"url": current_url_to_try, "event_type": "get_pdf_detected"})
                        if not os.path.exists(DOWNLOADS_DIR): os.makedirs(DOWNLOADS_DIR)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        url_part = "".join(c if c.isalnum() else "_" for c in current_url_to_try.split("://")[-1][:50])
                        file_name = f"{timestamp}_{url_part}.pdf"
                        file_path = os.path.join(DOWNLOADS_DIR, file_name)
                        with open(file_path, "wb") as f: f.write(get_response.content) # Use get_response.content
                        is_valid, reason = is_pdf_content_valid(file_path)
                        if is_valid:
                            logger.info(f"PDF (GET from {current_url_to_try}) {file_path} is valid.", extra={"path": file_path, "event_type": "pdf_get_valid"})
                            return FileResult(type="file", path=file_path)
                        else:
                            logger.warning(f"PDF (GET from {current_url_to_try}) {file_path} invalid: {reason}. Deleting.", extra={"path": file_path, "reason":reason, "event_type": "pdf_get_invalid"})
                            try: os.remove(file_path)
                            except OSError: pass
                            last_get_exception = Failure(type="failure", reason=f"Downloaded PDF from {current_url_to_try} invalid: {reason}")
                            continue
                    else:
                        logger.warning(f"Unsupported Content-Type '{content_type_get}' for {current_url_to_try}.", extra={"url": current_url_to_try, "type": content_type_get, "event_type": "unsupported_content_type_get"})
                        last_get_exception = Failure(type="failure", reason=f"Unsupported Content-Type '{content_type_get}' from {current_url_to_try}.")
                        continue
                except httpx.HTTPStatusError as e_get:
                    last_get_exception = e_get
                    logger.warning(f"GET for {current_url_to_try} failed: HTTP {e_get.response.status_code}", extra={"url": current_url_to_try, "status": e_get.response.status_code, "event_type": "get_http_status_error"})
                    continue
                except httpx.RequestError as e_get_req:
                    last_get_exception = e_get_req
                    logger.warning(f"GET for {current_url_to_try} failed: {e_get_req!s}", extra={"url": current_url_to_try, "error": str(e_get_req), "event_type": "get_request_error"})
                    continue
            
            if last_get_exception:
                if isinstance(last_get_exception, Failure): return last_get_exception
                if isinstance(last_get_exception, httpx.HTTPStatusError): return Failure(type="failure", reason=f"All GETs failed. Last: HTTP {last_get_exception.response.status_code} on {last_get_exception.request.url}")
                if isinstance(last_get_exception, httpx.RequestError): return Failure(type="failure", reason=f"All GETs failed. Last: RequestError {last_get_exception!s} on {last_get_exception.request.url}")
            return Failure(type="failure", reason=f"All GET attempts failed for {url} and variants.")

        except Exception as e_outer: # Catch-all for unexpected issues in resolve_content
            logger.critical(f"Unexpected error in resolve_content for {url}: {e_outer!s}", exc_info=True, extra={"url": url, "event_type": "resolve_unexpected_error_outer"})
            return Failure(type="failure", reason=f"Unexpected error: {e_outer!s}")
        finally:
            logger.info(f"Semaphore released for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_released"})
            if not provided_client and client: await client.aclose()
            logger.info(f"Finished resolving content for URL: {url}", extra={"url": url, "event_type": "resolve_end"})

if __name__ == "__main__":
    async def main():
        if not logger.handlers or not isinstance(logger.handlers[0].formatter, JsonFormatter):
            logger.handlers.clear()
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(JsonFormatter())
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)

        test_urls = [
            "https://www.bmj.com/content/372/bmj.n386", # HTML
            "https://arxiv.org/pdf/2303.10130.pdf", # Direct PDF
            "https://www.nejm.org/doi/pdf/10.1056/NEJMoa2035389", 
            "https://www.tandfonline.com/doi/pdf/10.1080/00222930802004478",
            "http://nonexistenturl12345.com/article.html", 
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "http://httpstat.us/503", 
            "http://httpstat.us/404", 
            "http://httpstat.us/403",
        ]
        results = await asyncio.gather(*(resolve_content(url) for url in test_urls), return_exceptions=True)

        for test_url, result in zip(test_urls, results):
            print(f"\\n--- Result for: {test_url} ---")
            if isinstance(result, Exception):
                print(f"  Unexpected Exception: {result!s}")
            elif result.type == "file":
                print(f"  Success (File): {result.path}")
            elif result.type == "html":
                print(f"  Success (HTML): Extracted text. First 100 chars: {result.text[:100]}...")
                if not os.path.exists(DOWNLOADS_DIR):
                    os.makedirs(DOWNLOADS_DIR)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                url_part = "".join(c if c.isalnum() else "_" for c in test_url.split("://")[-1][:50])
                file_name = f"{timestamp}_{url_part}.txt"
                file_path = os.path.join(DOWNLOADS_DIR, file_name)
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(result.text)
                    print(f"  HTML content saved to: {file_path}")
                except Exception as e:
                    print(f"  Error saving HTML content: {e!s}")
            elif result.type == "failure":
                print(f"  Failure: {result.reason}")
            print("-" * 20)
    
    asyncio.run(main())
