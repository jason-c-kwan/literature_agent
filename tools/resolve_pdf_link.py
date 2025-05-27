"""
Resolves a URL to determine if it yields HTML full text or a downloadable PDF.
"""
import asyncio
import datetime
import os
import logging
import json
from typing import Literal, NamedTuple, Union, List # Added List
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse, quote_plus # Added more from urllib.parse

import httpx
import yaml
from bs4 import BeautifulSoup
from readability import Document # readability-lxml
import pymupdf4llm # For checking PDF content
import fitz # PyMuPDF, pymupdf4llm is a wrapper around this

# --- Configuration ---
CONFIG_PATH = "config/settings.yaml"
PDF_UNLOCK_PARAMS_PATH = "config/pdf_unlock_params.yaml" # Added

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
        # Keys manually added to log_entry in this formatter:
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
        
        # 'event' is a special field. If 'event_type' is in extra, use it.
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
    global _pdf_unlock_params_cache # Use global to modify the module-level cache
    if _pdf_unlock_params_cache is not None:
        return _pdf_unlock_params_cache
    try:
        with open(PDF_UNLOCK_PARAMS_PATH, "r", encoding="utf-8") as f:
            _pdf_unlock_params_cache = yaml.safe_load(f)
            if _pdf_unlock_params_cache is None: # Handle empty file case
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

# --- PDF Retry URL Generation ---
def generate_pdf_retry_urls(original_url: str) -> List[str]:
    urls_to_try = [] # Start with empty, add original first if not already covered by a param variant
    try:
        parsed_original_url = urlparse(original_url)
        original_query_params = parse_qs(parsed_original_url.query, keep_blank_values=True)
        
        unlock_params_config = load_pdf_unlock_params()
        domain_specific_params = unlock_params_config.get(parsed_original_url.netloc, [])
        default_params = unlock_params_config.get("default", [])
        
        # Combine domain-specific and default, prioritizing domain-specific
        seen_param_tuples = set() # To store (key, value) tuples for uniqueness
        ordered_params_to_try_tuples = []

        for p_list in [domain_specific_params, default_params]:
            for p_str in p_list:
                try:
                    param_key, param_value = p_str.split("=", 1)
                    if (param_key, param_value) not in seen_param_tuples:
                        ordered_params_to_try_tuples.append((param_key, param_value))
                        seen_param_tuples.add((param_key, param_value))
                except ValueError:
                    logger.warning(f"Malformed parameter string '{p_str}' in PDF unlock config for domain '{parsed_original_url.netloc}' or default.", extra={"param_string": p_str, "event_type": "pdf_unlock_param_malformed"})


        # Add original URL first
        urls_to_try.append(original_url)

        for param_key, param_value in ordered_params_to_try_tuples:
            new_query_params = original_query_params.copy()
            
            # Update/add the new parameter. parse_qs stores values as lists.
            new_query_params[param_key] = [param_value] 
            
            # Create new query string. urlencode handles quoting.
            new_query_string = urlencode(new_query_params, doseq=True) 
            
            retry_url = urlunparse(
                (parsed_original_url.scheme,
                 parsed_original_url.netloc,
                 parsed_original_url.path,
                 parsed_original_url.params, # usually empty for HTTP URLs
                 new_query_string,
                 parsed_original_url.fragment)
            )
            if retry_url not in urls_to_try: # Avoid adding if it's identical to original or another variant
                urls_to_try.append(retry_url)
                
    except Exception as e:
        logger.error(f"Error generating PDF retry URLs for {original_url}: {e}", exc_info=True, extra={"original_url": original_url, "event_type": "generate_pdf_retry_url_error"})
        if not urls_to_try or urls_to_try[0] != original_url: # Ensure original is always an option if generation fails
            urls_to_try.insert(0, original_url)
            urls_to_try = list(dict.fromkeys(urls_to_try)) # Remove duplicates, keep order

    logger.info(f"Generated {len(urls_to_try)} unique URLs (incl. original) for PDF retry for {original_url}", extra={"original_url": original_url, "count": len(urls_to_try), "generated_urls": urls_to_try, "event_type": "pdf_retry_urls_generated"})
    return list(dict.fromkeys(urls_to_try)) # Ensure uniqueness and preserve order

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
    "access this article", "buy this article", "purchase access",
    "subscribe to view", "institutional login", "access options",
    "get access", "full text access", "journal subscription",
    "pay per view", "purchase pdf", "rent this article",
    "limited preview", "unlock this article", "sign in to read",
    "£", "$", "€", "usd", "eur", "gbp"
]

def is_pdf_content_valid(file_path: str) -> tuple[bool, str]:
    try:
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb < MIN_PDF_SIZE_KB:
            return False, f"File size ({file_size_kb:.2f}KB) is less than minimum ({MIN_PDF_SIZE_KB}KB)."
        doc = fitz.open(file_path)
        page_count = len(doc)
        if page_count == 0:
            doc.close()
            return False, "PDF has 0 pages."
        if page_count == 1:
            text = ""
            try:
                page = doc.load_page(0)
                text = page.get_text("text")
            finally:
                doc.close()
            text_original_for_len_check = text
            text_lower_stripped = text.lower().strip()
            if text_lower_stripped == "dummy pdf file":
                 return False, "PDF content suggests it's a dummy PDF (e.g., contains 'dummy PDF')."
            if "dummy" in text_lower_stripped and "pdf" in text_lower_stripped and len(text_original_for_len_check) < 200:
                 return False, "PDF content suggests it's a dummy PDF (e.g., contains 'dummy PDF')."
            if len(text_original_for_len_check) < 100 and \
               ("placeholder" in text_lower_stripped or \
                "abstract only" in text_lower_stripped or \
                "cover page" in text_lower_stripped):
                 return False, "PDF content suggests it's a placeholder or abstract-only."
            if len(text_lower_stripped) < 20:
                return False, f"Single-page PDF has very little text content (approx. {len(text_lower_stripped)} chars)."
            return True, ""
        else:
            doc.close()
            return True, ""
    except Exception as e:
        logger.error(f"Error checking PDF content for {file_path}", exc_info=True, extra={"file_path": file_path, "event_type": "pdf_validation_error"})
        return False, f"Error checking PDF content: {e!s}"

def is_html_potentially_paywalled(full_html_content: str) -> bool:
    html_lower = full_html_content.lower()
    matches = 0
    for keyword in PAYWALL_KEYWORDS:
        if keyword in html_lower:
            matches +=1
    if any(currency in html_lower for currency in ["£", "$", "€", "usd", "eur", "gbp"]) and matches >= 1:
        return True
    if matches >= 2:
        return True
    if "access this article for" in html_lower and "buy this article" in html_lower:
        return True
    if "institutional login" in html_lower and ("subscribe" in html_lower or "purchase" in html_lower):
        return True
    return False

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

async def _make_request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    if "User-Agent" not in headers:
        headers["User-Agent"] = DEFAULT_USER_AGENT
    
    for attempt in range(MAX_RETRIES + 1): 
        log_extra = {"url": url, "method": method, "attempt": attempt + 1, "max_retries": MAX_RETRIES}
        try:
            logger.info(f"Requesting {method} {url}, attempt {attempt + 1}", extra={**log_extra, "event_type": f"{method.lower()}_attempt"})
            response = await client.request(method, url, timeout=REQUEST_TIMEOUT, headers=headers, **kwargs)
            response.raise_for_status()
            logger.info(f"{method} request to {url} successful (status {response.status_code})", extra={**log_extra, "status_code": response.status_code, "event_type": f"{method.lower()}_success"})
            return response
        except httpx.HTTPStatusError as e:
            log_extra["status_code"] = e.response.status_code
            log_extra["error"] = str(e)
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


async def resolve_content(url: str, client: httpx.AsyncClient = None) -> ResolveResult:
    provided_client = bool(client)
    if not client:
        client = httpx.AsyncClient(follow_redirects=True)

    logger.info(f"Starting to resolve content for URL: {url}", extra={"url": url, "event_type": "resolve_start"})
    
    semaphore = await get_domain_semaphore(url)
    logger.info(f"Attempting to acquire semaphore for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_acquire_attempt"})
    async with semaphore:
        logger.info(f"Semaphore acquired for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_acquired"})
        try:
            # Initial HEAD request (optional, but can quickly identify PDFs)
            try:
                head_response = await _make_request_with_retry(client, "HEAD", url)
                content_type_head = head_response.headers.get("content-type", "").lower()
                if content_type_head.startswith("application/pdf"):
                    logger.info(f"HEAD indicates PDF for {url}. Proceeding to download with GET.", extra={"url": url, "content_type": content_type_head, "event_type": "head_pdf_detected_get_attempt"})
                    # Even if HEAD says PDF, we GET it to ensure content and handle potential changes/redirects
                    pdf_response = await _make_request_with_retry(client, "GET", url, headers={"Accept": "application/pdf,*/*"}) # Add Accept header
                    
                    # Validate and save (refactored to a helper or inline for now)
                    if not os.path.exists(DOWNLOADS_DIR): os.makedirs(DOWNLOADS_DIR)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    url_part = "".join(c if c.isalnum() else "_" for c in url.split("://")[-1][:50])
                    file_name = f"{timestamp}_{url_part}.pdf"
                    file_path = os.path.join(DOWNLOADS_DIR, file_name)
                    with open(file_path, "wb") as f: f.write(pdf_response.content)
                    
                    is_valid, reason = is_pdf_content_valid(file_path)
                    if not is_valid:
                        logger.warning(f"Downloaded PDF (after HEAD success) {file_path} invalid: {reason}. Deleting.", extra={"url": url, "path": file_path, "reason": reason, "event_type": "pdf_head_invalid_after_get"})
                        try: os.remove(file_path)
                        except OSError: pass
                        # Do not return Failure yet, fall through to main GET logic which might try unlock
                    else:
                        logger.info(f"PDF (after HEAD success) {file_path} is valid.", extra={"url": url, "path": file_path, "event_type": "pdf_head_valid_after_get"})
                        return FileResult(type="file", path=file_path)
            except httpx.HTTPStatusError as e_head:
                if e_head.response.status_code in [403, 405]:
                    logger.info(f"HEAD request for {url} failed with {e_head.response.status_code}. Proceeding with GET.", extra={"url": url, "status_code": e_head.response.status_code, "event_type": "head_fail_try_get"})
                else:
                    # For other HEAD errors, log but proceed to GET, as GET is more definitive.
                    logger.warning(f"HEAD request for {url} failed with {e_head.response.status_code}: {e_head!s}. Proceeding with GET.", extra={"url": url, "status_code": e_head.response.status_code, "error": str(e_head), "event_type": "head_fail_non_403_405_try_get"})
            except httpx.RequestError as e_head_req:
                 logger.warning(f"HEAD request for {url} failed with RequestError: {e_head_req!s}. Proceeding with GET.", extra={"url": url, "error": str(e_head_req), "event_type": "head_request_error_try_get"})


            # Main GET attempt and PDF unlock logic
            # The generate_pdf_retry_urls function includes the original URL as the first item.
            urls_to_attempt = generate_pdf_retry_urls(url)
            
            last_get_exception = None

            for attempt_idx, current_url_to_try in enumerate(urls_to_attempt):
                is_original_url_attempt = (current_url_to_try == url and attempt_idx == 0)
                request_headers = {"User-Agent": DEFAULT_USER_AGENT} # Base headers
                if "/pdf/" in current_url_to_try.lower() or "/epdf/" in current_url_to_try.lower() or not is_original_url_attempt:
                    # For PDF paths or any retry URL (non-original), prioritize PDF.
                    request_headers["Accept"] = "application/pdf,*/*"
                else:
                    # For original URL that might be HTML, broader accept.
                    request_headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7"


                logger.info(f"Attempting GET for {current_url_to_try} (Attempt {attempt_idx + 1}/{len(urls_to_attempt)})", 
                            extra={"url_attempt": current_url_to_try, "original_url": url, "attempt_num": attempt_idx + 1, "total_attempts": len(urls_to_attempt), "event_type": "resolve_get_attempt"})
                try:
                    get_response = await _make_request_with_retry(client, "GET", current_url_to_try, headers=request_headers)
                    content_type_get = get_response.headers.get("content-type", "").lower()

                    if "html" in content_type_get:
                        # Only process HTML if it's from the original URL attempt and not a PDF-specific retry
                        if is_original_url_attempt or not ( "/pdf/" in current_url_to_try.lower() or "/epdf/" in current_url_to_try.lower()):
                            html_content = get_response.text
                            # ... (existing HTML processing logic, slightly adapted)
                            logger.info(f"GET request for {current_url_to_try} returned HTML.", extra={"url": current_url_to_try, "content_type": content_type_get, "event_type": "get_html_detected"})
                            potentially_paywalled_full_html = is_html_potentially_paywalled(html_content)
                            if potentially_paywalled_full_html:
                                logger.info(f"HTML for {current_url_to_try} shows paywall indicators.", extra={"url": current_url_to_try, "event_type": "html_paywall_indicator"})

                            soup = BeautifulSoup(html_content, "html.parser")
                            extracted_text = None
                            try:
                                doc = Document(html_content)
                                main_content_html = doc.summary(html_partial=True)
                                summary_soup = BeautifulSoup(main_content_html, "html.parser")
                                current_extracted_text = summary_soup.get_text(separator="\\n", strip=True)
                                if current_extracted_text and len(current_extracted_text) > MIN_HTML_CONTENT_LENGTH:
                                    extracted_text = current_extracted_text
                                    logger.info(f"Extracted HTML content from {current_url_to_try} using readability.", extra={"url": current_url_to_try, "length": len(extracted_text), "event_type": "html_extract_readability_success"})
                            except Exception as e_readability:
                                logger.warning(f"Readability failed for {current_url_to_try}: {e_readability!s}", extra={"url": current_url_to_try, "error": str(e_readability), "event_type": "html_extract_readability_fail"})

                            if not extracted_text or len(extracted_text) < MIN_HTML_CONTENT_LENGTH:
                                article_tag = soup.find("article")
                                if article_tag:
                                    current_extracted_text = article_tag.get_text(separator="\\n", strip=True)
                                    if current_extracted_text and len(current_extracted_text) > MIN_HTML_CONTENT_LENGTH:
                                        extracted_text = current_extracted_text
                                        logger.info(f"Extracted HTML content from {current_url_to_try} using <article> tag.", extra={"url": current_url_to_try, "length": len(extracted_text), "event_type": "html_extract_article_success"})

                            if extracted_text:
                                if len(extracted_text) >= MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE:
                                    logger.info(f"Long HTML content from {current_url_to_try} overrides paywall flag.", extra={"url": current_url_to_try, "length": len(extracted_text), "event_type": "html_long_override_paywall"})
                                    return HTMLResult(type="html", text=extracted_text)
                                elif potentially_paywalled_full_html:
                                    logger.warning(f"Paywall indicators and short HTML content for {current_url_to_try}.", extra={"url": current_url_to_try, "length": len(extracted_text), "event_type": "html_paywall_short_content_fail"})
                                    # Don't return Failure yet, continue if other URLs to try
                                    last_get_exception = Failure(type="failure", reason="Paywall indicators in full HTML and extracted content is short.") # Store as potential failure
                                    continue # Try next URL in urls_to_attempt
                                elif len(extracted_text) >= MIN_HTML_CONTENT_LENGTH:
                                    logger.info(f"Sufficient HTML content from {current_url_to_try}.", extra={"url": current_url_to_try, "length": len(extracted_text), "event_type": "html_sufficient_content_success"})
                                    return HTMLResult(type="html", text=extracted_text)
                                else: # HTML too short, no strong paywall
                                    logger.warning(f"HTML content extracted from {current_url_to_try} but too short.", extra={"url": current_url_to_try, "length": len(extracted_text), "event_type": "html_extract_too_short_fail"})
                                    last_get_exception = Failure(type="failure", reason="HTML detected, content extracted but too short, no strong paywall signs on page.")
                                    continue # Try next URL
                            else: # No main content extracted
                                reason_msg = "Paywall indicators in full HTML and no main content extracted." if potentially_paywalled_full_html else "HTML detected, but no main content could be extracted."
                                logger.warning(f"No main HTML content extracted from {current_url_to_try}. Paywalled: {potentially_paywalled_full_html}", extra={"url": current_url_to_try, "paywalled_full_html": potentially_paywalled_full_html, "event_type": "html_no_content_extracted_fail"})
                                last_get_exception = Failure(type="failure", reason=reason_msg)
                                continue # Try next URL
                        else: # HTML from a PDF-specific retry URL, likely an error page
                            logger.warning(f"Received HTML from a PDF-specific retry URL: {current_url_to_try}. Content-Type: {content_type_get}. Skipping as non-PDF.", extra={"url":current_url_to_try, "content_type":content_type_get, "event_type":"html_from_pdf_retry_url"})
                            last_get_exception = Failure(type="failure", reason=f"Expected PDF, got HTML from {current_url_to_try}")
                            continue


                    elif content_type_get.startswith("application/pdf"):
                        logger.info(f"GET indicates PDF for {current_url_to_try}. Proceeding to save.", extra={"url": current_url_to_try, "content_type": content_type_get, "event_type": "get_pdf_detected"})
                        if not os.path.exists(DOWNLOADS_DIR): os.makedirs(DOWNLOADS_DIR)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Use current_url_to_try for filename uniqueness if params changed
                        url_part_for_file = "".join(c if c.isalnum() else "_" for c in current_url_to_try.split("://")[-1][:50])
                        file_name = f"{timestamp}_{url_part_for_file}.pdf"
                        file_path = os.path.join(DOWNLOADS_DIR, file_name)
                        with open(file_path, "wb") as f: f.write(get_response.content)
                        logger.info(f"PDF (via GET) downloaded to {file_path} from {current_url_to_try}", extra={"url": current_url_to_try, "path": file_path, "event_type": "pdf_get_download_success"})
                        
                        is_valid, reason = is_pdf_content_valid(file_path)
                        if not is_valid:
                            logger.warning(f"Downloaded PDF (via GET from {current_url_to_try}) {file_path} invalid: {reason}. Deleting.", extra={"url": current_url_to_try, "path": file_path, "reason": reason, "event_type": "pdf_get_invalid"})
                            try: os.remove(file_path)
                            except OSError: pass
                            last_get_exception = Failure(type="failure", reason=f"Downloaded PDF (from {current_url_to_try}) appears invalid: {reason}")
                            continue # Try next URL
                        logger.info(f"PDF (via GET from {current_url_to_try}) {file_path} is valid.", extra={"url": current_url_to_try, "path": file_path, "event_type": "pdf_get_valid"})
                        return FileResult(type="file", path=file_path)
                    
                    else: # Neither HTML nor PDF from GET
                        logger.warning(f"Unsupported Content-Type '{content_type_get}' for {current_url_to_try}.", extra={"url": current_url_to_try, "content_type": content_type_get, "event_type": "unsupported_content_type_fail_get"})
                        last_get_exception = Failure(type="failure", reason=f"Content-Type '{content_type_get}' from {current_url_to_try} is not PDF or HTML.")
                        continue # Try next URL

                except httpx.HTTPStatusError as e_get:
                    last_get_exception = e_get # Store the exception
                    logger.warning(f"GET request for {current_url_to_try} failed with {e_get.response.status_code}.", extra={"url": current_url_to_try, "status_code": e_get.response.status_code, "error": str(e_get), "event_type": "get_http_status_error"})
                    # If it's a 403 on a PDF path, the loop will continue to try other params.
                    # If it's another error, or not a PDF path, or last attempt, it might fail.
                    if e_get.response.status_code not in RETRY_STATUS_FORCELIST and e_get.response.status_code != 403:
                        # If it's a hard error (e.g. 404 on a retry URL), maybe stop early for this set of attempts?
                        # For now, let the loop continue, it will fail at the end if no success.
                        pass
                    # Continue to the next URL in urls_to_attempt
                    continue
                except httpx.RequestError as e_get_req:
                    last_get_exception = e_get_req
                    logger.warning(f"GET request for {current_url_to_try} failed with RequestError: {e_get_req!s}", extra={"url": current_url_to_try, "error": str(e_get_req), "event_type": "get_request_error"})
                    # Continue to the next URL in urls_to_attempt
                    continue
            
            # If loop finished and no success, handle the last_get_exception or return generic failure
            if last_get_exception:
                if isinstance(last_get_exception, Failure): # Our custom Failure type from HTML processing
                    return last_get_exception
                elif isinstance(last_get_exception, httpx.HTTPStatusError):
                     return Failure(type="failure", reason=f"All GET attempts failed. Last error: HTTP {last_get_exception.response.status_code} on {last_get_exception.request.url}")
                elif isinstance(last_get_exception, httpx.RequestError):
                     return Failure(type="failure", reason=f"All GET attempts failed. Last error: RequestError {last_get_exception!s} on {last_get_exception.request.url}")
            
            return Failure(type="failure", reason=f"All GET attempts failed for original URL {url} and its variants.")

        except httpx.HTTPStatusError as e: # Catch errors from _make_request_with_retry if they weren't handled by PDF unlock
            logger.error(f"HTTP error resolving {url}: {e.response.status_code}", extra={"url": url, "status_code": e.response.status_code, "error": str(e), "event_type": "resolve_http_status_error_final"})
            return Failure(type="failure", reason=f"HTTP error: {e.response.status_code} {e.response.reason_phrase} for URL: {url}")
        except httpx.RequestError as e: 
            logger.error(f"Request error resolving {url}: {e!s}", extra={"url": url, "error": str(e), "event_type": "resolve_request_error_final"})
            return Failure(type="failure", reason=f"Request failed: {e!s}")
        except Exception as e:
            logger.critical(f"Unexpected error resolving {url}: {e!s}", exc_info=True, extra={"url": url, "error": str(e), "event_type": "resolve_unexpected_error"})
            return Failure(type="failure", reason=f"An unexpected error occurred: {e!s}")
        finally:
            logger.info(f"Semaphore released for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_released"})
            if not provided_client and client:
                await client.aclose()
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
            # Test a URL that might 403 and then unlock with a param
            # For this, we'd need a mock server or a live test URL known to behave this way.
            # Example (hypothetical, assuming nejm.org needs ?download=true from config):
            "https://www.nejm.org/doi/pdf/10.1056/NEJMoa2035389", # Fictional DOI
            "https://www.tandfonline.com/doi/pdf/10.1080/00222930802004478", # Fictional DOI for tandf
            "http://nonexistenturl12345.com/article.html", 
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", # Dummy PDF
            "http://httpstat.us/503", 
            "http://httpstat.us/404", 
            "http://httpstat.us/403", # Generic 403
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
                if not os.path.exists(DOWNLOADS_DIR): os.makedirs(DOWNLOADS_DIR)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                url_part = "".join(c if c.isalnum() else "_" for c in test_url.split("://")[-1][:50])
                file_name = f"{timestamp}_{url_part}.txt"
                file_path = os.path.join(DOWNLOADS_DIR, file_name)
                try:
                    with open(file_path, "w", encoding="utf-8") as f: f.write(result.text)
                    print(f"  HTML content saved to: {file_path}")
                except Exception as e:
                    print(f"  Error saving HTML content: {e!s}")
            elif result.type == "failure":
                print(f"  Failure: {result.reason}")
            print("-" * 20)

    asyncio.run(main())
