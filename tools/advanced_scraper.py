"""
Advanced scraper for resolving URLs to HTML full text or downloadable PDFs,
with support for domain-specific rules, headless browsing, proxy rotation,
and LLM-based analysis fallback.
"""
import asyncio
import datetime
import os
import logging
import json
from typing import Literal, NamedTuple, Union, Optional, List, Dict, Any
from urllib.parse import urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from readability import Document  # readability-lxml
import fitz  # PyMuPDF
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

# --- Configuration ---
CONFIG_PATH_SETTINGS = "config/settings.yaml"
CONFIG_PATH_DOMAINS = "config/domains.yaml"

# Default values from resolve_pdf_link.py, can be overridden by settings.yaml
DEFAULT_TIMEOUT = 15
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_FACTOR = 0.5
DEFAULT_RETRY_STATUS_FORCELIST = [500, 502, 503, 504]
DEFAULT_PER_DOMAIN_CONCURRENCY = 2
MIN_PDF_SIZE_KB = 10
MIN_HTML_CONTENT_LENGTH = 200
MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE = 7000 # from resolve_pdf_link.py
DOWNLOADS_DIR = "workspace/downloads/advanced_scraper" # Separate download dir

def load_yaml_config(path: str, default: Dict = None) -> Dict:
    """Loads a YAML configuration file."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or default
    except FileNotFoundError:
        logger.warning(f"Config file not found: {path}. Using defaults.")
        return default
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {e}. Using defaults.")
        return default

settings_config = load_yaml_config(CONFIG_PATH_SETTINGS)
domains_config = load_yaml_config(CONFIG_PATH_DOMAINS, {"domains": []})

REQUEST_TIMEOUT = settings_config.get("resolve_content_timeout", DEFAULT_TIMEOUT)
MAX_RETRIES = settings_config.get("resolve_max_retries", DEFAULT_MAX_RETRIES)
RETRY_BACKOFF_FACTOR = settings_config.get("resolve_retry_backoff_factor", DEFAULT_RETRY_BACKOFF_FACTOR)
RETRY_STATUS_FORCELIST = settings_config.get("resolve_retry_status_forcelist", DEFAULT_RETRY_STATUS_FORCELIST)
PER_DOMAIN_CONCURRENCY = settings_config.get("resolve_per_domain_concurrency", DEFAULT_PER_DOMAIN_CONCURRENCY)

# Proxy settings
PROXY_SETTINGS = settings_config.get("proxy_settings", {"enabled": False, "proxies": []})

# --- Logging Setup (similar to resolve_pdf_link.py) ---
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

logger = logging.getLogger("advanced_scraper")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Per-Domain Rate Limiting (similar to resolve_pdf_link.py) ---
domain_semaphores: Dict[str, asyncio.Semaphore] = {}
domain_semaphores_lock = asyncio.Lock()

async def get_domain_semaphore(url: str) -> asyncio.Semaphore:
    domain = urlparse(url).netloc
    if not domain:
        logger.warning("Could not parse domain from URL", extra={"url": url, "event_type": "domain_parse_failure"})
        domain = "__no_domain__" # Fallback domain

    async with domain_semaphores_lock:
        if domain not in domain_semaphores:
            logger.info(f"Creating new semaphore for domain: {domain}", extra={"domain": domain, "concurrency": PER_DOMAIN_CONCURRENCY, "event_type": "semaphore_create"})
            domain_semaphores[domain] = asyncio.Semaphore(PER_DOMAIN_CONCURRENCY)
        return domain_semaphores[domain]

# --- Result types (copied from resolve_pdf_link.py) ---
class Failure(NamedTuple):
    type: Literal["failure"]
    reason: str
    status_code: Optional[int] = None # Added status_code

class HTMLResult(NamedTuple):
    type: Literal["html"]
    text: str
    url: str # Added final URL

class FileResult(NamedTuple):
    type: Literal["file"]
    path: str
    url: str # Added final URL

ResolveResult = Union[Failure, HTMLResult, FileResult]

# --- Constants and Helper Functions from resolve_pdf_link.py (adapted) ---
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
        if not os.path.exists(file_path):
            return False, "File does not exist."
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
                page = doc.load_page(0) # type: ignore
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
            if len(text_lower_stripped) < 20: # Stricter check for single page
                return False, f"Single-page PDF has very little text content (approx. {len(text_lower_stripped)} chars)."
            # Add more checks if needed for single-page PDFs
        doc.close()
        return True, ""
    except Exception as e:
        logger.error(f"Error checking PDF content for {file_path}", exc_info=True, extra={"file_path": file_path, "event_type": "pdf_validation_error"})
        return False, f"Error checking PDF content: {e!s}"

def is_html_potentially_paywalled(full_html_content: str) -> bool:
    # This function can be enhanced based on common paywall patterns
    html_lower = full_html_content.lower()
    matches = 0
    for keyword in PAYWALL_KEYWORDS:
        if keyword in html_lower:
            matches +=1
    # More specific checks
    if any(currency in html_lower for currency in ["£", "$", "€", "usd", "eur", "gbp"]) and matches >= 1:
        return True
    if matches >= 2: # Two general keywords might indicate a paywall
        return True
    if "access this article for" in html_lower and "buy this article" in html_lower: # Specific phrase combination
        return True
    if "institutional login" in html_lower and ("subscribe" in html_lower or "purchase" in html_lower):
        return True
    return False

async def _make_request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    # This function is copied and adapted from resolve_pdf_link.py
    # It should incorporate proxy selection if PROXY_SETTINGS["enabled"] is true.
    # For now, placeholder for proxy logic.
    
    current_proxies = None
    if PROXY_SETTINGS.get("enabled") and PROXY_SETTINGS.get("proxies"):
        # Basic round-robin or random selection for now.
        # This needs to be more robust (e.g., per-request proxy, retry with new proxy)
        import random
        proxy_url = random.choice(PROXY_SETTINGS["proxies"])
        current_proxies = {"http://": proxy_url, "https://": proxy_url}
        logger.info(f"Using proxy: {proxy_url} for {method} {url}", extra={"url": url, "proxy": proxy_url, "event_type": "proxy_use"})

    effective_client = httpx.AsyncClient(follow_redirects=True, proxies=current_proxies) if current_proxies else client

    for attempt in range(MAX_RETRIES + 1):
        log_extra = {"url": url, "method": method, "attempt": attempt + 1, "max_retries": MAX_RETRIES}
        try:
            logger.info(f"Requesting {method} {url}, attempt {attempt + 1}", extra={**log_extra, "event_type": f"{method.lower()}_attempt"})
            response = await effective_client.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            response.raise_for_status()
            logger.info(f"{method} request to {url} successful (status {response.status_code})", extra={**log_extra, "status_code": response.status_code, "event_type": f"{method.lower()}_success"})
            if effective_client is not client: # Close the temporary client if one was created for proxy
                await effective_client.aclose()
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
                if effective_client is not client: await effective_client.aclose()
                raise
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProxyError) as e: # Added ProxyError
            log_extra["error"] = str(e)
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(f"{method} request to {url} failed with {type(e).__name__}, retrying in {delay:.2f}s...", extra={**log_extra, "delay": delay, "event_type": "retry_scheduled_network"})
                await asyncio.sleep(delay)
                # Potentially switch proxy here if proxy error
                if isinstance(e, httpx.ProxyError) and PROXY_SETTINGS.get("enabled") and PROXY_SETTINGS.get("proxies"):
                    logger.warning(f"Proxy error with {current_proxies}. Attempting to switch proxy.", extra={"url": url, "event_type": "proxy_error_switch_attempt"})
                    # Re-select proxy for next attempt (basic)
                    proxy_url = random.choice(PROXY_SETTINGS["proxies"])
                    current_proxies = {"http://": proxy_url, "https://": proxy_url}
                    if effective_client is not client: await effective_client.aclose() # close old proxy client
                    effective_client = httpx.AsyncClient(follow_redirects=True, proxies=current_proxies)

            else:
                logger.error(f"{method} request to {url} failed with {type(e).__name__}, no more retries.", extra={**log_extra, "event_type": "request_failure_network"})
                if effective_client is not client: await effective_client.aclose()
                raise
    if effective_client is not client: await effective_client.aclose()
    raise httpx.RequestError(f"Max retries ({MAX_RETRIES}) exceeded for {method} {url}")


def _get_domain_rules(url: str) -> Optional[Dict[str, Any]]:
    """Gets domain-specific parsing rules from domains_config."""
    parsed_url = urlparse(url)
    domain_to_match = parsed_url.netloc
    for rule in domains_config.get("domains", []):
        if rule.get("domain") == domain_to_match:
            return rule
    return None

async def _extract_html_text(html_content: str, url: str) -> Optional[str]:
    """Extracts main text content from HTML."""
    extracted_text = None
    try:
        doc = Document(html_content) # readability-lxml
        main_content_html = doc.summary(html_partial=True)
        summary_soup = BeautifulSoup(main_content_html, "html.parser")
        current_extracted_text = summary_soup.get_text(separator="\\n", strip=True)
        if current_extracted_text and len(current_extracted_text) >= MIN_HTML_CONTENT_LENGTH:
            extracted_text = current_extracted_text
            logger.info(f"Extracted HTML content from {url} using readability.", extra={"url": url, "length": len(extracted_text), "event_type": "html_extract_readability_success"})
    except Exception as e:
        logger.warning(f"Readability failed for {url}: {e!s}", extra={"url": url, "error": str(e), "event_type": "html_extract_readability_fail"})

    if not extracted_text or len(extracted_text) < MIN_HTML_CONTENT_LENGTH:
        # Fallback to <article> tag if readability fails or yields short content
        soup = BeautifulSoup(html_content, "html.parser")
        article_tag = soup.find("article")
        if article_tag:
            current_extracted_text = article_tag.get_text(separator="\\n", strip=True)
            if current_extracted_text and len(current_extracted_text) >= MIN_HTML_CONTENT_LENGTH:
                extracted_text = current_extracted_text
                logger.info(f"Extracted HTML content from {url} using <article> tag.", extra={"url": url, "length": len(extracted_text), "event_type": "html_extract_article_success"})
    return extracted_text

async def _handle_pdf_download(response_content: bytes, url: str, source_description: str = "direct") -> ResolveResult:
    """Saves and validates a PDF from bytes."""
    if not os.path.exists(DOWNLOADS_DIR):
        os.makedirs(DOWNLOADS_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for uniqueness
    url_part = "".join(c if c.isalnum() else "_" for c in urlparse(url).netloc[:30] + urlparse(url).path.replace('/', '_')[:30])
    file_name = f"{timestamp}_{url_part}.pdf"
    file_path = os.path.join(DOWNLOADS_DIR, file_name)

    try:
        with open(file_path, "wb") as f:
            f.write(response_content)
        logger.info(f"PDF (via {source_description}) downloaded to {file_path}", extra={"url": url, "path": file_path, "event_type": f"pdf_{source_description}_download_success"})

        is_valid, reason = is_pdf_content_valid(file_path)
        if not is_valid:
            logger.warning(f"Downloaded PDF (via {source_description}) {file_path} invalid: {reason}. Deleting.", extra={"url": url, "path": file_path, "reason": reason, "event_type": f"pdf_{source_description}_invalid"})
            try: os.remove(file_path)
            except OSError as e: logger.error(f"Error deleting invalid PDF {file_path}: {e}", extra={"path": file_path, "event_type": "pdf_delete_fail"})
            return Failure(type="failure", reason=f"Downloaded PDF (via {source_description}) appears invalid: {reason}")
        logger.info(f"PDF (via {source_description}) {file_path} is valid.", extra={"url": url, "path": file_path, "event_type": f"pdf_{source_description}_valid"})
        return FileResult(type="file", path=file_path, url=url)
    except IOError as e:
        logger.error(f"IOError saving PDF to {file_path}: {e}", extra={"url": url, "path": file_path, "event_type": "pdf_save_io_error"})
        return Failure(type="failure", reason=f"IOError saving PDF: {e}")


# --- Main Public API ---
async def scrape_with_fallback(
    url: str,
    # Parameters to control behavior, potentially passed by the agent
    attempt_playwright: bool = True, # Agent might set this to False if it wants to do LLM analysis first
    playwright_page_content_for_llm: Optional[str] = None # If agent already got Playwright content
) -> ResolveResult:
    """
    Attempts to scrape a URL for full text HTML or a PDF, with multiple fallbacks.
    Order of operations:
    1. Direct HTTP GET: Check for HTML full text.
    2. Direct HTTP GET: Check for PDF content-type.
    3. (If attempt_playwright is True) Playwright: Render page, check for HTML full text.
    4. (If attempt_playwright is True) Playwright: Render page, check for PDF download/link.
    5. LLM analysis (orchestrated by the calling Agent, not directly in this function but this
       function should be callable in a way that supports the agent's workflow).
    """
    logger.info(f"Starting advanced scrape for URL: {url}", extra={"url": url, "event_type": "scrape_start"})
    
    # Ensure downloads directory exists
    if not os.path.exists(DOWNLOADS_DIR):
        try:
            os.makedirs(DOWNLOADS_DIR)
            logger.info(f"Created downloads directory: {DOWNLOADS_DIR}", extra={"event_type": "dir_create_success", "path": DOWNLOADS_DIR})
        except OSError as e:
            logger.error(f"Could not create downloads directory {DOWNLOADS_DIR}: {e}", extra={"event_type": "dir_create_fail", "path": DOWNLOADS_DIR})
            return Failure(type="failure", reason=f"Cannot create download directory: {e}")

    # Per-domain concurrency control
    semaphore = await get_domain_semaphore(url)
    logger.info(f"Attempting to acquire semaphore for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_acquire_attempt"})
    
    async with httpx.AsyncClient(follow_redirects=True) as client, semaphore:
        logger.info(f"Semaphore acquired for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_acquired"})
        final_url_after_redirects = url # Will be updated after requests

        try:
            # Step 1: Direct HTTP GET for HTML full text
            logger.info(f"Attempting direct GET for HTML: {url}", extra={"url": url, "event_type": "direct_get_html_attempt"})
            try:
                get_response_html = await _make_request_with_retry(client, "GET", url, headers={"Accept": "text/html,*/*;q=0.8"})
                final_url_after_redirects = str(get_response_html.url)
                content_type_html = get_response_html.headers.get("content-type", "").lower()

                if "html" in content_type_html:
                    html_content = get_response_html.text
                    extracted_text = await _extract_html_text(html_content, final_url_after_redirects)
                    if extracted_text:
                        paywalled = is_html_potentially_paywalled(html_content)
                        # Check length conditions from resolve_pdf_link
                        if len(extracted_text) >= MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE:
                            logger.info(f"Direct GET: Long HTML content from {final_url_after_redirects} overrides paywall flag.", extra={"url": final_url_after_redirects, "length": len(extracted_text), "event_type": "direct_html_long_override_paywall"})
                            return HTMLResult(type="html", text=extracted_text, url=final_url_after_redirects)
                        elif not paywalled and len(extracted_text) >= MIN_HTML_CONTENT_LENGTH:
                             logger.info(f"Direct GET: Sufficient non-paywalled HTML from {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "length": len(extracted_text), "event_type": "direct_html_sufficient_success"})
                             return HTMLResult(type="html", text=extracted_text, url=final_url_after_redirects)
                        elif paywalled:
                            logger.info(f"Direct GET: HTML from {final_url_after_redirects} is short or paywalled.", extra={"url": final_url_after_redirects, "length": len(extracted_text), "paywalled": paywalled, "event_type": "direct_html_paywalled_or_short"})
                        # else: too short, continue
                    else:
                        logger.info(f"Direct GET: No substantial HTML text extracted from {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "event_type": "direct_html_no_extract"})
                else:
                    logger.info(f"Direct GET: Content-Type for {final_url_after_redirects} is not HTML: {content_type_html}", extra={"url": final_url_after_redirects, "content_type": content_type_html, "event_type": "direct_get_not_html"})

            except httpx.HTTPStatusError as e:
                logger.warning(f"Direct GET for HTML for {url} failed with HTTP {e.response.status_code}", extra={"url": url, "status_code": e.response.status_code, "event_type": "direct_get_html_http_fail"})
                # If 403/405, HEAD might also fail, but we try HEAD for PDF next anyway
            except httpx.RequestError as e:
                logger.warning(f"Direct GET for HTML for {url} failed with RequestError: {e}", extra={"url": url, "error": str(e), "event_type": "direct_get_html_request_fail"})
                # Continue to PDF attempt

            # Step 2: Direct HEAD/GET for PDF
            logger.info(f"Attempting direct HEAD/GET for PDF: {url}", extra={"url": url, "event_type": "direct_pdf_attempt"})
            try:
                head_response_pdf = await _make_request_with_retry(client, "HEAD", url, headers={"Accept": "application/pdf,*/*;q=0.8"})
                final_url_after_redirects = str(head_response_pdf.url) # Update final_url
                content_type_pdf_head = head_response_pdf.headers.get("content-type", "").lower()
                if content_type_pdf_head.startswith("application/pdf"):
                    logger.info(f"HEAD indicates PDF for {final_url_after_redirects}. Downloading with GET.", extra={"url": final_url_after_redirects, "event_type": "direct_head_pdf_detected"})
                    pdf_get_response = await _make_request_with_retry(client, "GET", final_url_after_redirects) # Use final_url_after_redirects
                    return await _handle_pdf_download(pdf_get_response.content, final_url_after_redirects, "direct_head_then_get")
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [403, 405]: # HEAD not allowed, try GET directly for PDF
                    logger.info(f"HEAD for PDF for {url} failed ({e.response.status_code}), trying direct GET for PDF.", extra={"url": url, "status_code": e.response.status_code, "event_type": "direct_head_pdf_fail_try_get"})
                    try:
                        get_response_pdf = await _make_request_with_retry(client, "GET", url, headers={"Accept": "application/pdf,*/*;q=0.8"})
                        final_url_after_redirects = str(get_response_pdf.url)
                        content_type_pdf_get = get_response_pdf.headers.get("content-type", "").lower()
                        if content_type_pdf_get.startswith("application/pdf"):
                            logger.info(f"Direct GET indicates PDF for {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "event_type": "direct_get_pdf_detected"})
                            return await _handle_pdf_download(get_response_pdf.content, final_url_after_redirects, "direct_get")
                    except httpx.RequestError as e_get: # Includes HTTPStatusError
                         logger.warning(f"Direct GET for PDF for {url} also failed: {e_get}", extra={"url": url, "error": str(e_get), "event_type": "direct_get_pdf_fail"})
                else:
                    logger.warning(f"HEAD for PDF for {url} failed: {e}", extra={"url": url, "error": str(e), "event_type": "direct_head_pdf_fail_other"})
            except httpx.RequestError as e:
                logger.warning(f"Direct PDF check for {url} failed with RequestError: {e}", extra={"url": url, "error": str(e), "event_type": "direct_pdf_request_fail"})


            # Step 3 & 4: Playwright fallback (if enabled)
            if attempt_playwright:
                logger.info(f"Attempting Playwright fallback for: {url}", extra={"url": url, "event_type": "playwright_attempt"})
                # Placeholder for Playwright logic
                # This will involve:
                # from playwright.async_api import async_playwright
                # async with async_playwright() as p:
                #   browser = await p.chromium.launch(proxy=...)
                #   page = await browser.new_page()
                #   await page.goto(url)
                #   rendered_html = await page.content()
                #   Check rendered_html for full text (call _extract_html_text)
                #   If HTML found and good, return HTMLResult
                #   Else, look for PDF downloads/links
                #     - page.on("download", handle_download_event)
                #     - Search DOM for PDF links
                #   await browser.close()
                playwright_result: Optional[ResolveResult] = None
                try:
                    async with async_playwright() as p:
                        browser_args = []
                        # Basic proxy integration for Playwright
                        playwright_proxy_config = None
                        if PROXY_SETTINGS.get("enabled") and PROXY_SETTINGS.get("proxies"):
                            import random
                            # Select a random proxy. More sophisticated selection could be added.
                            proxy_url_full = random.choice(PROXY_SETTINGS["proxies"])
                            parsed_proxy = urlparse(proxy_url_full)
                            playwright_proxy_config = {
                                "server": f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}",
                            }
                            if parsed_proxy.username:
                                playwright_proxy_config["username"] = parsed_proxy.username
                            if parsed_proxy.password:
                                playwright_proxy_config["password"] = parsed_proxy.password
                            logger.info(f"Playwright using proxy: {playwright_proxy_config['server']}", extra={"url": url, "event_type": "playwright_proxy_use"})
                        
                        browser = await p.chromium.launch(proxy=playwright_proxy_config, args=browser_args)
                        page = await browser.new_page()
                        
                        # PDF download interception
                        download_info = {"path": None, "url": None, "error": None}

                        async def handle_download(download):
                            try:
                                suggested_filename = download.suggested_filename
                                if not suggested_filename.lower().endswith(".pdf"):
                                    logger.warning(f"Playwright download is not a PDF: {suggested_filename}", extra={"url": url, "filename": suggested_filename, "event_type": "playwright_download_not_pdf"})
                                    download_info["error"] = "Downloaded file not a PDF."
                                    await download.cancel() # or delete() if already saved
                                    return

                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                url_part = "".join(c if c.isalnum() else "_" for c in urlparse(url).netloc[:30] + urlparse(url).path.replace('/', '_')[:30])
                                file_name = f"{timestamp}_{url_part}_playwright.pdf"
                                save_path = os.path.join(DOWNLOADS_DIR, file_name)
                                await download.save_as(save_path)
                                download_info["path"] = save_path
                                download_info["url"] = download.url # URL of the download
                                logger.info(f"Playwright download saved to {save_path}", extra={"url": url, "path": save_path, "download_url": download.url, "event_type": "playwright_download_saved"})
                            except Exception as e_download:
                                logger.error(f"Error handling Playwright download: {e_download}", exc_info=True, extra={"url": url, "event_type": "playwright_download_handler_error"})
                                download_info["error"] = str(e_download)
                            
                        page.on("download", handle_download)

                        try:
                            await page.goto(url, timeout=REQUEST_TIMEOUT * 1000 * 2, wait_until="networkidle") # Longer timeout for playwright, wait for network idle
                            final_url_after_redirects = page.url # Update with Playwright's final URL
                        except PlaywrightTimeoutError:
                            logger.warning(f"Playwright timed out navigating to {url}. Content might be partial.", extra={"url": url, "event_type": "playwright_nav_timeout"})
                            # Continue with potentially partial content
                        except PlaywrightError as e_goto:
                            logger.error(f"Playwright navigation error for {url}: {e_goto}", exc_info=True, extra={"url": url, "event_type": "playwright_nav_error"})
                            await browser.close()
                            return Failure(type="failure", reason=f"Playwright navigation error: {e_goto}")

                        # Check if PDF was downloaded via event
                        if download_info["path"]:
                            is_valid, reason = is_pdf_content_valid(download_info["path"])
                            if is_valid:
                                logger.info(f"Playwright auto-downloaded PDF is valid: {download_info['path']}", extra={"url": url, "event_type": "playwright_autodownload_pdf_valid"})
                                playwright_result = FileResult(type="file", path=download_info["path"], url=download_info["url"] or final_url_after_redirects)
                            else:
                                logger.warning(f"Playwright auto-downloaded PDF invalid: {reason}. Deleting.", extra={"url": url, "path": download_info["path"], "reason": reason, "event_type": "playwright_autodownload_pdf_invalid"})
                                try: os.remove(download_info["path"])
                                except OSError: pass
                                # Don't return failure yet, try extracting HTML or finding link
                        elif download_info["error"]:
                             logger.warning(f"Playwright download event error: {download_info['error']}", extra={"url": url, "event_type": "playwright_download_event_error"})


                        if not playwright_result: # If PDF not auto-downloaded and valid
                            rendered_html = await page.content()
                            # Attempt 3a (Playwright): Extract HTML full text from rendered page
                            extracted_text_playwright = await _extract_html_text(rendered_html, final_url_after_redirects)
                            if extracted_text_playwright:
                                paywalled_playwright = is_html_potentially_paywalled(rendered_html)
                                if len(extracted_text_playwright) >= MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE:
                                    logger.info(f"Playwright: Long HTML content from {final_url_after_redirects} overrides paywall.", extra={"url": final_url_after_redirects, "length": len(extracted_text_playwright), "event_type": "playwright_html_long_override_paywall"})
                                    playwright_result = HTMLResult(type="html", text=extracted_text_playwright, url=final_url_after_redirects)
                                elif not paywalled_playwright and len(extracted_text_playwright) >= MIN_HTML_CONTENT_LENGTH:
                                    logger.info(f"Playwright: Sufficient non-paywalled HTML from {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "length": len(extracted_text_playwright), "event_type": "playwright_html_sufficient_success"})
                                    playwright_result = HTMLResult(type="html", text=extracted_text_playwright, url=final_url_after_redirects)
                                else:
                                    logger.info(f"Playwright: HTML from {final_url_after_redirects} is short or paywalled.", extra={"url": final_url_after_redirects, "length": len(extracted_text_playwright), "paywalled": paywalled_playwright, "event_type": "playwright_html_paywalled_or_short"})
                            else:
                                logger.info(f"Playwright: No substantial HTML text extracted from {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "event_type": "playwright_html_no_extract"})
                        
                        if not playwright_result: # If HTML not good, try finding PDF links in Playwright DOM
                            # Attempt 3b (Playwright): Find PDF links in rendered DOM
                            # This could reuse domain_rules or use generic selectors
                            # For now, a simple search for <a> tags with .pdf href
                            pdf_links = await page.query_selector_all("a[href$='.pdf']")
                            if pdf_links:
                                for link_element in pdf_links:
                                    pdf_url_rel = await link_element.get_attribute("href")
                                    if pdf_url_rel:
                                        pdf_abs_url = urlparse(final_url_after_redirects)._replace(path=pdf_url_rel).geturl() if not pdf_url_rel.startswith(('http://', 'https://')) else pdf_url_rel
                                        logger.info(f"Playwright found potential PDF link: {pdf_abs_url}", extra={"url": url, "pdf_link": pdf_abs_url, "event_type": "playwright_found_pdf_link"})
                                        # Try to fetch this PDF link using httpx
                                        try:
                                            async with httpx.AsyncClient(follow_redirects=True) as pdf_client: # New client for this
                                                pdf_response = await _make_request_with_retry(pdf_client, "GET", pdf_abs_url)
                                                pdf_content_type = pdf_response.headers.get("content-type", "").lower()
                                                if pdf_content_type.startswith("application/pdf"):
                                                    dl_result = await _handle_pdf_download(pdf_response.content, pdf_abs_url, "playwright_link_get")
                                                    if dl_result.type == "file":
                                                        playwright_result = dl_result
                                                        break # Found a valid PDF
                                        except Exception as e_fetch_link:
                                            logger.warning(f"Playwright: Error fetching PDF link {pdf_abs_url}: {e_fetch_link}", extra={"url": url, "pdf_link": pdf_abs_url, "event_type": "playwright_fetch_pdf_link_error"})
                        await browser.close()
                        if playwright_result:
                            return playwright_result
                        
                except PlaywrightError as e_pw:
                    logger.error(f"Playwright processing error for {url}: {e_pw}", exc_info=True, extra={"url": url, "event_type": "playwright_general_error"})
                    # Fall through to general failure, or specific failure if desired
                except Exception as e_pw_unhandled: # Catch any other unhandled playwright errors
                    logger.error(f"Unhandled Playwright exception for {url}: {e_pw_unhandled}", exc_info=True, extra={"url": url, "event_type": "playwright_unhandled_exception"})


            # If all above fail, return a generic failure.
            # The agent might take this failure and then try LLM analysis on any fetched content.
            logger.warning(f"All direct and Playwright (if attempted) methods failed for {url}.", extra={"url": url, "event_type": "all_methods_failed"})
            return Failure(type="failure", reason="All scraping attempts (direct HTTP, Playwright if enabled) failed to yield valid content.", status_code=None)

        except httpx.HTTPStatusError as e_outer:
            logger.error(f"Outer HTTP error for {url}: {e_outer.response.status_code}", extra={"url": url, "status_code": e_outer.response.status_code, "event_type": "scrape_http_status_error_final"})
            return Failure(type="failure", reason=f"HTTP error: {e_outer.response.status_code} for URL: {url}", status_code=e_outer.response.status_code)
        except httpx.RequestError as e_outer:
            logger.error(f"Outer Request error for {url}: {e_outer!s}", extra={"url": url, "error": str(e_outer), "event_type": "scrape_request_error_final"})
            return Failure(type="failure", reason=f"Request failed: {e_outer!s}")
        except Exception as e_outer:
            logger.critical(f"Unexpected error scraping {url}: {e_outer!s}", exc_info=True, extra={"url": url, "error": str(e_outer), "event_type": "scrape_unexpected_error"})
            return Failure(type="failure", reason=f"An unexpected error occurred during scraping: {e_outer!s}")
        finally:
            logger.info(f"Semaphore released for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_released"})
            logger.info(f"Finished advanced scrape for URL: {url}", extra={"url": url, "event_type": "scrape_end"})


if __name__ == "__main__":
    async def fetch_specific_pdf_url(url: str) -> ResolveResult:
        """
        Attempts to download and validate a PDF from a specific URL,
        assuming it's a direct link to a PDF.
        """
        logger.info(f"Attempting to fetch specific PDF URL: {url}", extra={"url": url, "event_type": "fetch_specific_pdf_attempt"})
        
        if not os.path.exists(DOWNLOADS_DIR):
            try:
                os.makedirs(DOWNLOADS_DIR)
            except OSError as e:
                logger.error(f"Could not create downloads directory {DOWNLOADS_DIR} for specific PDF: {e}", extra={"event_type": "dir_create_fail_specific_pdf", "path": DOWNLOADS_DIR})
                return Failure(type="failure", reason=f"Cannot create download directory for specific PDF: {e}")

        semaphore = await get_domain_semaphore(url)
        async with httpx.AsyncClient(follow_redirects=True) as client, semaphore:
            try:
                logger.info(f"Fetching specific PDF GET: {url}", extra={"url": url, "event_type": "specific_pdf_get_attempt"})
                # Use a more specific Accept header if we are sure it's a PDF
                get_response = await _make_request_with_retry(client, "GET", url, headers={"Accept": "application/pdf,application/octet-stream,*/*;q=0.8"})
                final_url = str(get_response.url)
                content_type = get_response.headers.get("content-type", "").lower()

                # Check content type, but also try to save if it looks like PDF data even if type is generic
                if content_type.startswith("application/pdf") or \
                   (content_type.startswith("application/octet-stream") and url.lower().endswith(".pdf")):
                    logger.info(f"Specific PDF URL GET indicates PDF for {final_url} (Content-Type: {content_type}).", extra={"url": final_url, "content_type": content_type, "event_type": "specific_pdf_get_detected"})
                    return await _handle_pdf_download(get_response.content, final_url, "specific_url_get")
                else:
                    # Try saving anyway if the URL ends with .pdf, as content-type might be wrong
                    if url.lower().endswith(".pdf"):
                        logger.warning(f"Content-Type for specific PDF URL {final_url} is '{content_type}', but URL ends with .pdf. Attempting to save.", extra={"url": final_url, "content_type": content_type, "event_type": "specific_pdf_suspicious_content_type_save_attempt"})
                        return await _handle_pdf_download(get_response.content, final_url, "specific_url_suspicious_get")
                    
                    logger.warning(f"Specific PDF URL {final_url} did not return PDF content-type: {content_type}", extra={"url": final_url, "content_type": content_type, "event_type": "specific_pdf_not_pdf_content_type"})
                    return Failure(type="failure", reason=f"URL did not yield PDF content (Content-Type: {content_type})", status_code=get_response.status_code)

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching specific PDF URL {url}: {e.response.status_code}", extra={"url": url, "status_code": e.response.status_code, "event_type": "specific_pdf_http_error"})
                return Failure(type="failure", reason=f"HTTP error {e.response.status_code} for specific PDF URL: {url}", status_code=e.response.status_code)
            except httpx.RequestError as e:
                logger.error(f"Request error fetching specific PDF URL {url}: {e!s}", extra={"url": url, "error": str(e), "event_type": "specific_pdf_request_error"})
                return Failure(type="failure", reason=f"Request failed for specific PDF URL: {e!s}")
            except Exception as e:
                logger.critical(f"Unexpected error fetching specific PDF URL {url}: {e!s}", exc_info=True, extra={"url": url, "error": str(e), "event_type": "specific_pdf_unexpected_error"})
                return Failure(type="failure", reason=f"Unexpected error for specific PDF URL: {e!s}")


async def fetch_specific_pdf_url(url: str) -> ResolveResult:
    """
    Attempts to download and validate a PDF from a specific URL,
    assuming it's a direct link to a PDF.
    """
    logger.info(f"Attempting to fetch specific PDF URL: {url}", extra={"url": url, "event_type": "fetch_specific_pdf_attempt"})
    
    if not os.path.exists(DOWNLOADS_DIR):
        try:
            os.makedirs(DOWNLOADS_DIR)
        except OSError as e:
            logger.error(f"Could not create downloads directory {DOWNLOADS_DIR} for specific PDF: {e}", extra={"event_type": "dir_create_fail_specific_pdf", "path": DOWNLOADS_DIR})
            return Failure(type="failure", reason=f"Cannot create download directory for specific PDF: {e}")

    semaphore = await get_domain_semaphore(url)
    async with httpx.AsyncClient(follow_redirects=True) as client, semaphore:
        try:
            logger.info(f"Fetching specific PDF GET: {url}", extra={"url": url, "event_type": "specific_pdf_get_attempt"})
            # Use a more specific Accept header if we are sure it's a PDF
            get_response = await _make_request_with_retry(client, "GET", url, headers={"Accept": "application/pdf,application/octet-stream,*/*;q=0.8"})
            final_url = str(get_response.url)
            content_type = get_response.headers.get("content-type", "").lower()

            # Check content type, but also try to save if it looks like PDF data even if type is generic
            if content_type.startswith("application/pdf") or \
               (content_type.startswith("application/octet-stream") and url.lower().endswith(".pdf")):
                logger.info(f"Specific PDF URL GET indicates PDF for {final_url} (Content-Type: {content_type}).", extra={"url": final_url, "content_type": content_type, "event_type": "specific_pdf_get_detected"})
                return await _handle_pdf_download(get_response.content, final_url, "specific_url_get")
            else:
                # Try saving anyway if the URL ends with .pdf, as content-type might be wrong
                if url.lower().endswith(".pdf"):
                    logger.warning(f"Content-Type for specific PDF URL {final_url} is '{content_type}', but URL ends with .pdf. Attempting to save.", extra={"url": final_url, "content_type": content_type, "event_type": "specific_pdf_suspicious_content_type_save_attempt"})
                    return await _handle_pdf_download(get_response.content, final_url, "specific_url_suspicious_get")
                
                logger.warning(f"Specific PDF URL {final_url} did not return PDF content-type: {content_type}", extra={"url": final_url, "content_type": content_type, "event_type": "specific_pdf_not_pdf_content_type"})
                return Failure(type="failure", reason=f"URL did not yield PDF content (Content-Type: {content_type})", status_code=get_response.status_code)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching specific PDF URL {url}: {e.response.status_code}", extra={"url": url, "status_code": e.response.status_code, "event_type": "specific_pdf_http_error"})
            return Failure(type="failure", reason=f"HTTP error {e.response.status_code} for specific PDF URL: {url}", status_code=e.response.status_code)
        except httpx.RequestError as e:
            logger.error(f"Request error fetching specific PDF URL {url}: {e!s}", extra={"url": url, "error": str(e), "event_type": "specific_pdf_request_error"})
            return Failure(type="failure", reason=f"Request failed for specific PDF URL: {e!s}")
        except Exception as e:
            logger.critical(f"Unexpected error fetching specific PDF URL {url}: {e!s}", exc_info=True, extra={"url": url, "error": str(e), "event_type": "specific_pdf_unexpected_error"})
            return Failure(type="failure", reason=f"Unexpected error for specific PDF URL: {e!s}")


if __name__ == "__main__":
    async def main_test():
        # Configure logger for local testing if not already configured by autogen
        if not logger.handlers or not isinstance(logger.handlers[0].formatter, JsonFormatter):
            logger.handlers.clear() # Clear any existing handlers
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(JsonFormatter())
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO) # Or logging.DEBUG for more verbosity

        test_urls = [
            "https://www.bmj.com/content/372/bmj.n386", # Should give HTML
            "https://arxiv.org/pdf/2303.10130.pdf",    # Should give PDF
            # "https://www.nature.com/articles/s41586-021-03491-6", # Likely paywalled HTML
            # "http://nonexistenturl12345.com/article.html",
            # "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", # Invalid PDF
            # "http://httpstat.us/503", # Server error
        ]
        results = await asyncio.gather(*(scrape_with_fallback(url, attempt_playwright=False) for url in test_urls), return_exceptions=True)

        for test_url, result in zip(test_urls, results):
            print(f"\\n--- Result for: {test_url} ---")
            if isinstance(result, Exception):
                print(f"  Unexpected Exception: {result!s}")
            elif result.type == "file":
                print(f"  Success (File): {result.path} from {result.url}")
            elif result.type == "html":
                print(f"  Success (HTML): Extracted text from {result.url}. First 100 chars: {result.text[:100]}...")
            elif result.type == "failure":
                print(f"  Failure: {result.reason} (Status: {result.status_code})")
            print("-" * 20)

    asyncio.run(main_test())
