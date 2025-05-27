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
import random # Added import
from typing import Literal, NamedTuple, Union, Optional, List, Dict, Any
from urllib.parse import urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from readability import Document  # readability-lxml
import fitz  # PyMuPDF
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError, Download # Added Download

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
PLAYWRIGHT_NAV_TIMEOUT = 90000 # 90 seconds for Playwright navigation

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

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

async def _make_request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    if "User-Agent" not in headers:
        headers["User-Agent"] = DEFAULT_USER_AGENT

    current_proxies = None
    if PROXY_SETTINGS.get("enabled") and PROXY_SETTINGS.get("proxies"):
        import random
        proxy_url = random.choice(PROXY_SETTINGS["proxies"])
        current_proxies = {"http://": proxy_url, "https://": proxy_url}
        logger.info(f"Using proxy: {proxy_url} for {method} {url}", extra={"url": url, "proxy": proxy_url, "event_type": "proxy_use"})

    effective_client = httpx.AsyncClient(follow_redirects=True, proxies=current_proxies) if current_proxies else client

    for attempt in range(MAX_RETRIES + 1):
        log_extra = {"url": url, "method": method, "attempt": attempt + 1, "max_retries": MAX_RETRIES}
        try:
            logger.info(f"Requesting {method} {url}, attempt {attempt + 1}", extra={**log_extra, "event_type": f"{method.lower()}_attempt"})
            response = await effective_client.request(method, url, timeout=REQUEST_TIMEOUT, headers=headers, **kwargs)
            response.raise_for_status()
            logger.info(f"{method} request to {url} successful (status {response.status_code})", extra={**log_extra, "status_code": response.status_code, "event_type": f"{method.lower()}_success"})
            if effective_client is not client: 
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
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProxyError) as e: 
            log_extra["error"] = str(e)
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(f"{method} request to {url} failed with {type(e).__name__}, retrying in {delay:.2f}s...", extra={**log_extra, "delay": delay, "event_type": "retry_scheduled_network"})
                await asyncio.sleep(delay)
                if isinstance(e, httpx.ProxyError) and PROXY_SETTINGS.get("enabled") and PROXY_SETTINGS.get("proxies"):
                    logger.warning(f"Proxy error with {current_proxies}. Attempting to switch proxy.", extra={"url": url, "event_type": "proxy_error_switch_attempt"})
                    proxy_url = random.choice(PROXY_SETTINGS["proxies"])
                    current_proxies = {"http://": proxy_url, "https://": proxy_url}
                    if effective_client is not client: await effective_client.aclose() 
                    effective_client = httpx.AsyncClient(follow_redirects=True, proxies=current_proxies)
            else:
                logger.error(f"{method} request to {url} failed with {type(e).__name__}, no more retries.", extra={**log_extra, "event_type": "request_failure_network"})
                if effective_client is not client: await effective_client.aclose()
                raise
    if effective_client is not client: await effective_client.aclose()
    raise httpx.RequestError(f"Max retries ({MAX_RETRIES}) exceeded for {method} {url}")

def _get_domain_rules(url: str) -> Optional[Dict[str, Any]]:
    parsed_url = urlparse(url)
    domain_to_match = parsed_url.netloc
    for rule in domains_config.get("domains", []):
        if rule.get("domain") == domain_to_match:
            return rule
    return None

async def _extract_html_text(html_content: str, url: str) -> Optional[str]:
    extracted_text = None
    try:
        doc = Document(html_content) 
        main_content_html = doc.summary(html_partial=True)
        summary_soup = BeautifulSoup(main_content_html, "html.parser")
        current_extracted_text = summary_soup.get_text(separator="\\n", strip=True)
        if current_extracted_text and len(current_extracted_text) >= MIN_HTML_CONTENT_LENGTH:
            extracted_text = current_extracted_text
            logger.info(f"Extracted HTML content from {url} using readability.", extra={"url": url, "length": len(extracted_text), "event_type": "html_extract_readability_success"})
    except Exception as e:
        logger.warning(f"Readability failed for {url}: {e!s}", extra={"url": url, "error": str(e), "event_type": "html_extract_readability_fail"})

    if not extracted_text or len(extracted_text) < MIN_HTML_CONTENT_LENGTH:
        soup = BeautifulSoup(html_content, "html.parser")
        article_tag = soup.find("article")
        if article_tag:
            current_extracted_text = article_tag.get_text(separator="\\n", strip=True)
            if current_extracted_text and len(current_extracted_text) >= MIN_HTML_CONTENT_LENGTH:
                extracted_text = current_extracted_text
                logger.info(f"Extracted HTML content from {url} using <article> tag.", extra={"url": url, "length": len(extracted_text), "event_type": "html_extract_article_success"})
    return extracted_text

async def _handle_pdf_download(response_content: bytes, url: str, source_description: str = "direct") -> ResolveResult:
    if not os.path.exists(DOWNLOADS_DIR):
        os.makedirs(DOWNLOADS_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
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

async def scrape_with_fallback(
    url: str,
    attempt_playwright: bool = True, 
    playwright_page_content_for_llm: Optional[str] = None
) -> ResolveResult:
    logger.info(f"Starting advanced scrape for URL: {url}", extra={"url": url, "event_type": "scrape_start"})
    if not os.path.exists(DOWNLOADS_DIR):
        try:
            os.makedirs(DOWNLOADS_DIR)
            logger.info(f"Created downloads directory: {DOWNLOADS_DIR}", extra={"event_type": "dir_create_success", "path": DOWNLOADS_DIR})
        except OSError as e:
            logger.error(f"Could not create downloads directory {DOWNLOADS_DIR}: {e}", extra={"event_type": "dir_create_fail", "path": DOWNLOADS_DIR})
            return Failure(type="failure", reason=f"Cannot create download directory: {e}")

    semaphore = await get_domain_semaphore(url)
    logger.info(f"Attempting to acquire semaphore for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_acquire_attempt"})
    
    async with semaphore:
        logger.info(f"Semaphore acquired for domain of {url}", extra={"url": url, "domain": urlparse(url).netloc, "event_type": "semaphore_acquired"})
        async with httpx.AsyncClient(follow_redirects=True) as client:
            final_url_after_redirects = url 
            direct_http_exception = None
            
            # --- Direct HTTP Attempts ---
            try:
                logger.info(f"Attempting direct GET for HTML/PDF: {url}", extra={"url": url, "event_type": "direct_get_html_pdf_attempt"})
                get_response = await _make_request_with_retry(client, "GET", url, headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"})
                final_url_after_redirects = str(get_response.url)
                content_type = get_response.headers.get("content-type", "").lower()

                if "html" in content_type:
                    html_content = get_response.text
                    extracted_text = await _extract_html_text(html_content, final_url_after_redirects)
                    if extracted_text:
                        paywalled = is_html_potentially_paywalled(html_content)
                        if len(extracted_text) >= MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE:
                            logger.info(f"Direct GET: Long HTML content from {final_url_after_redirects} overrides paywall flag.", extra={"url": final_url_after_redirects, "length": len(extracted_text), "event_type": "direct_html_long_override_paywall"})
                            return HTMLResult(type="html", text=extracted_text, url=final_url_after_redirects)
                        elif not paywalled and len(extracted_text) >= MIN_HTML_CONTENT_LENGTH:
                            logger.info(f"Direct GET: Sufficient non-paywalled HTML from {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "length": len(extracted_text), "event_type": "direct_html_sufficient_success"})
                            return HTMLResult(type="html", text=extracted_text, url=final_url_after_redirects)
                        elif paywalled:
                            logger.info(f"Direct GET: HTML from {final_url_after_redirects} is short or paywalled.", extra={"url": final_url_after_redirects, "length": len(extracted_text), "paywalled": paywalled, "event_type": "direct_html_paywalled_or_short"})
                    else: 
                        logger.info(f"Direct GET: No substantial HTML text extracted from {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "event_type": "direct_html_no_extract"})
                
                elif content_type.startswith("application/pdf"):
                    logger.info(f"Direct GET indicates PDF for {final_url_after_redirects}. Downloading.", extra={"url": final_url_after_redirects, "event_type": "direct_get_pdf_detected"})
                    pdf_dl_result = await _handle_pdf_download(get_response.content, final_url_after_redirects, "direct_get")
                    if pdf_dl_result.type == "file":
                        return pdf_dl_result 
                    else: 
                        logger.warning(f"Direct GET PDF download/validation failed for {final_url_after_redirects}: {pdf_dl_result.reason}", extra={"url":final_url_after_redirects, "reason":pdf_dl_result.reason, "event_type":"direct_get_pdf_fail_validation"})
                
                else: 
                    logger.info(f"Direct GET: Content-Type for {final_url_after_redirects} is not HTML or PDF: {content_type}. Status: {get_response.status_code}", extra={"url": final_url_after_redirects, "content_type": content_type, "status_code": get_response.status_code, "event_type": "direct_get_not_html_or_pdf"})
                    if get_response.status_code != 200:
                         direct_http_exception = httpx.HTTPStatusError(f"Status {get_response.status_code} from direct GET", request=get_response.request, response=get_response)
                         logger.warning(f"Direct GET for {url} resulted in status {get_response.status_code}.", extra={"url":url, "status_code":get_response.status_code, "event_type":"direct_get_non_200_status"})

            except httpx.HTTPStatusError as e_direct_status:
                logger.warning(f"Direct GET for HTML/PDF for {url} failed with HTTP {e_direct_status.response.status_code}", extra={"url": url, "status_code": e_direct_status.response.status_code, "event_type": "direct_get_http_status_fail"})
                direct_http_exception = e_direct_status 
            except httpx.RequestError as e_direct_request:
                logger.warning(f"Direct GET for HTML/PDF for {url} failed with RequestError: {e_direct_request}", extra={"url": url, "error": str(e_direct_request), "event_type": "direct_get_request_fail"})
                direct_http_exception = e_direct_request
            except Exception as e_unexpected_direct: 
                logger.error(f"Unexpected error during direct HTTP phase for {url}: {e_unexpected_direct}", exc_info=True, extra={"url":url, "event_type":"direct_http_unexpected_error"})
                direct_http_exception = e_unexpected_direct

            # --- Playwright Attempt ---
            if attempt_playwright:
                logger.info(f"Attempting Playwright fallback for: {url} (Direct HTTP exception was: {type(direct_http_exception).__name__ if direct_http_exception else 'None'})", 
                            extra={"url": url, "direct_http_exception_type": type(direct_http_exception).__name__ if direct_http_exception else None, 
                                   "direct_http_exception_details": str(direct_http_exception) if direct_http_exception else None, 
                                   "event_type": "playwright_attempt_after_direct"})
                
                playwright_result: Optional[ResolveResult] = None
                # Ensure random is available in this scope, even if imported globally.
                # import random # This is already at the top of the file, should not be needed here.
                                # The UnboundLocalError suggests it's not seen. Let's trust the global import for now.
                try:
                    async with async_playwright() as p:
                        browser_args = []
                        playwright_proxy_config = None
                        if PROXY_SETTINGS.get("enabled") and PROXY_SETTINGS.get("proxies"):
                            # import random # Rely on global import
                            proxy_url_full = random.choice(PROXY_SETTINGS["proxies"])
                            parsed_proxy = urlparse(proxy_url_full)
                            playwright_proxy_config = { "server": f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}" }
                            if parsed_proxy.username: playwright_proxy_config["username"] = parsed_proxy.username
                            if parsed_proxy.password: playwright_proxy_config["password"] = parsed_proxy.password
                            logger.info(f"Playwright using proxy: {playwright_proxy_config['server']}", extra={"url": url, "event_type": "playwright_proxy_use"})
                        
                        browser = await p.chromium.launch(proxy=playwright_proxy_config, args=browser_args)
                        page = await browser.new_page()
                        await page.set_viewport_size({"width": 1920, "height": 1080}) # Set viewport
                        
                        # Ensure random is available for the timeout call
                        # This is belt-and-suspenders as it's imported globally
                        # import random # Already globally imported, this line is not needed if global works.
                                        # Given UnboundLocalError, let's ensure it's in scope if there's an odd issue.
                                        # Re-evaluating: the global import should be sufficient.
                                        # The error might stem from how asyncio tasks/scopes are handled.
                                        # Forcing it into local scope for this call:
                        
                        download_info = {"path": None, "url": None, "error": None}

                        async def handle_download(download):
                            try:
                                suggested_filename = download.suggested_filename
                                if not suggested_filename.lower().endswith(".pdf"):
                                    logger.warning(f"Playwright download is not a PDF: {suggested_filename}", extra={"url": url, "filename": suggested_filename, "event_type": "playwright_download_not_pdf"})
                                    download_info["error"] = "Downloaded file not a PDF."
                                    await download.cancel()
                                    return
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                url_part = "".join(c if c.isalnum() else "_" for c in urlparse(url).netloc[:30] + urlparse(url).path.replace('/', '_')[:30])
                                file_name = f"{timestamp}_{url_part}_playwright.pdf"
                                save_path = os.path.join(DOWNLOADS_DIR, file_name)
                                await download.save_as(save_path)
                                download_info["path"] = save_path
                                download_info["url"] = download.url
                                logger.info(f"Playwright download saved to {save_path}", extra={"url": url, "path": save_path, "download_url": download.url, "event_type": "playwright_download_saved"})
                            except Exception as e_download:
                                logger.error(f"Error handling Playwright download: {e_download}", exc_info=True, extra={"url": url, "event_type": "playwright_download_handler_error"})
                                download_info["error"] = str(e_download)
                            
                        page.on("download", handle_download)
                        logger.info(f"Playwright: Navigating to {url}", extra={"url": url, "event_type": "playwright_goto_attempt"})
                        pw_response_status = None
                        try:
                            pw_nav_response = await page.goto(url, timeout=PLAYWRIGHT_NAV_TIMEOUT, wait_until="networkidle")
                            # Use the globally imported random
                            await page.wait_for_timeout(random.randint(2000, 5000))
                            final_url_after_redirects = page.url
                            pw_response_status = pw_nav_response.status if pw_nav_response else None
                            logger.info(f"Playwright: Navigation to {final_url_after_redirects} completed with status {pw_response_status}", extra={"url": url, "final_url": final_url_after_redirects, "status": pw_response_status, "event_type": "playwright_goto_success"})
                        except PlaywrightTimeoutError:
                            logger.warning(f"Playwright: Timed out navigating to {url}. Content might be partial.", extra={"url": url, "event_type": "playwright_nav_timeout"})
                            final_url_after_redirects = page.url 
                        except PlaywrightError as e_goto:
                            logger.error(f"Playwright: Navigation error for {url}: {e_goto}", exc_info=True, extra={"url": url, "event_type": "playwright_nav_error"})
                            await browser.close()
                            return Failure(type="failure", reason=f"Playwright navigation error: {e_goto}", status_code=getattr(e_goto, 'response_status', None))

                        if download_info["path"]:
                            is_valid, reason = is_pdf_content_valid(download_info["path"])
                            if is_valid:
                                logger.info(f"Playwright auto-downloaded PDF is valid: {download_info['path']}", extra={"url": url, "event_type": "playwright_autodownload_pdf_valid"})
                                playwright_result = FileResult(type="file", path=download_info["path"], url=download_info["url"] or final_url_after_redirects)
                            else:
                                logger.warning(f"Playwright auto-downloaded PDF invalid: {reason}. Deleting.", extra={"url": url, "path": download_info["path"], "reason": reason, "event_type": "playwright_autodownload_pdf_invalid"})
                                try: os.remove(download_info["path"])
                                except OSError: pass
                        elif download_info["error"]:
                             logger.warning(f"Playwright download event error: {download_info['error']}", extra={"url": url, "event_type": "playwright_download_event_error"})

                        if not playwright_result:
                            logger.info(f"Playwright: Attempting to get page content from {final_url_after_redirects}", extra={"url": url, "final_url": final_url_after_redirects, "event_type": "playwright_get_content_attempt"})
                            rendered_html = await page.content()
                            logger.info(f"Playwright: Got page content, length {len(rendered_html)} chars from {final_url_after_redirects}", extra={"url": url, "final_url": final_url_after_redirects, "content_length": len(rendered_html), "event_type": "playwright_get_content_success"})
                            extracted_text_playwright = await _extract_html_text(rendered_html, final_url_after_redirects)
                            
                            # Check for common verification/error messages
                            is_verification_page = False
                            if extracted_text_playwright:
                                verification_keywords = ["verify you are human", "enable javascript and cookies", "security of your connection", "waiting for response"]
                                lc_extracted_text = extracted_text_playwright.lower()
                                if any(keyword in lc_extracted_text for keyword in verification_keywords):
                                    is_verification_page = True
                                    logger.warning(f"Playwright: Detected verification page content for {final_url_after_redirects}.", extra={"url":url, "final_url":final_url_after_redirects, "event_type":"playwright_verification_page_detected"})

                            if extracted_text_playwright and not is_verification_page:
                                logger.info(f"Playwright: Extracted text length {len(extracted_text_playwright)} from {final_url_after_redirects}", extra={"url": url, "final_url": final_url_after_redirects, "extracted_length": len(extracted_text_playwright), "event_type": "playwright_text_extraction_info"})
                                paywalled_playwright = is_html_potentially_paywalled(rendered_html)
                                if len(extracted_text_playwright) >= MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE:
                                    logger.info(f"Playwright: Long HTML content from {final_url_after_redirects} overrides paywall.", extra={"url": final_url_after_redirects, "length": len(extracted_text_playwright), "event_type": "playwright_html_long_override_paywall"})
                                    playwright_result = HTMLResult(type="html", text=extracted_text_playwright, url=final_url_after_redirects)
                                elif not paywalled_playwright and len(extracted_text_playwright) >= MIN_HTML_CONTENT_LENGTH:
                                    logger.info(f"Playwright: Sufficient non-paywalled HTML from {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "length": len(extracted_text_playwright), "event_type": "playwright_html_sufficient_success"})
                                    playwright_result = HTMLResult(type="html", text=extracted_text_playwright, url=final_url_after_redirects)
                                else: # Short or paywalled
                                    logger.info(f"Playwright: HTML from {final_url_after_redirects} is short or paywalled (or was verification page). Paywalled: {paywalled_playwright}, Length: {len(extracted_text_playwright)}", extra={"url": final_url_after_redirects, "length": len(extracted_text_playwright), "paywalled": paywalled_playwright, "event_type": "playwright_html_short_or_paywalled"})
                            elif is_verification_page:
                                # Already logged, playwright_result remains None to allow fallback
                                pass
                            else: # No substantial text extracted
                                logger.info(f"Playwright: No substantial HTML text extracted from {final_url_after_redirects}.", extra={"url": final_url_after_redirects, "event_type": "playwright_html_no_extract"})
                        
                        if not playwright_result: # If no auto-download or other early success
                            page_content_lower = ""
                            try:
                                current_page_content = await page.content()
                                page_content_lower = current_page_content.lower()
                            except PlaywrightError as e_content:
                                logger.warning(f"Playwright: Could not get page content from {final_url_after_redirects} to check for 'PDF': {e_content}", extra={"url":url, "final_url":final_url_after_redirects, "event_type":"playwright_get_content_for_pdf_check_error"})

                            if direct_http_exception and "pdf" in page_content_lower: 
                                logger.info(f"Playwright: Direct HTTP failed and 'pdf' found in page content of {final_url_after_redirects}. Attempting click_download_and_capture.", extra={"url": url, "final_url": final_url_after_redirects, "event_type": "playwright_pdf_in_content_click_attempt"})
                                
                                pdf_bytes_captured = await click_download_and_capture(page)
                                
                                if pdf_bytes_captured:
                                    logger.info(f"Playwright: Captured {len(pdf_bytes_captured)} bytes via click_download_and_capture for {final_url_after_redirects}.", extra={"url": url, "final_url": final_url_after_redirects, "bytes": len(pdf_bytes_captured), "event_type": "playwright_click_capture_success_bytes"})
                                    click_capture_result = await _handle_pdf_download(pdf_bytes_captured, final_url_after_redirects, "playwright_click_capture")
                                    if click_capture_result.type == "file":
                                        playwright_result = click_capture_result 
                                        logger.info(f"Playwright: Successfully processed PDF from click_download_and_capture for {final_url_after_redirects}.", extra={"url": url, "path": playwright_result.path, "event_type": "playwright_click_capture_processed_file"})
                                    else: 
                                        logger.warning(f"Playwright: PDF from click_download_and_capture for {final_url_after_redirects} was invalid or failed to save: {click_capture_result.reason}", extra={"url": url, "reason": click_capture_result.reason, "event_type": "playwright_click_capture_invalid_or_save_fail"})
                                else:
                                    logger.info(f"Playwright: click_download_and_capture did not return bytes for {final_url_after_redirects}.", extra={"url": url, "final_url": final_url_after_redirects, "event_type": "playwright_click_capture_no_bytes"})
                        
                        if not playwright_result: # Re-check if click_download_and_capture succeeded
                            logger.info(f"Playwright: Attempting to find PDF links in DOM of {final_url_after_redirects}", extra={"url": url, "final_url": final_url_after_redirects, "event_type": "playwright_find_pdf_links_attempt"})
                            pdf_links = await page.query_selector_all("a[href$='.pdf'], a[href*='downloadSgArticle'], a[href*='pdfLink']")
                            if pdf_links:
                                logger.info(f"Playwright: Found {len(pdf_links)} potential PDF links in DOM of {final_url_after_redirects}.", extra={"url": url, "final_url": final_url_after_redirects, "count": len(pdf_links), "event_type": "playwright_found_pdf_links_count"})
                                for i, link_element in enumerate(pdf_links):
                                    pdf_url_rel = await link_element.get_attribute("href")
                                    if pdf_url_rel:
                                        pdf_abs_url = urlparse(final_url_after_redirects)._replace(path=pdf_url_rel).geturl() if not pdf_url_rel.startswith(('http://', 'https://')) else pdf_url_rel
                                        logger.info(f"Playwright: Trying PDF link {i+1}/{len(pdf_links)}: {pdf_abs_url}", extra={"url": url, "pdf_link": pdf_abs_url, "event_type": "playwright_try_pdf_link"})
                                        try:
                                            async with httpx.AsyncClient(follow_redirects=True) as pdf_client:
                                                pdf_response = await _make_request_with_retry(pdf_client, "GET", pdf_abs_url)
                                                pdf_content_type = pdf_response.headers.get("content-type", "").lower()
                                                if pdf_content_type.startswith("application/pdf"):
                                                    dl_result = await _handle_pdf_download(pdf_response.content, pdf_abs_url, "playwright_link_get")
                                                    if dl_result.type == "file":
                                                        playwright_result = dl_result
                                                        logger.info(f"Playwright: Successfully downloaded PDF from link {pdf_abs_url}", extra={"url":url, "pdf_link":pdf_abs_url, "event_type":"playwright_pdf_link_download_success"})
                                                        break 
                                                else:
                                                    logger.warning(f"Playwright: PDF link {pdf_abs_url} did not return PDF content-type: {pdf_content_type}", extra={"url":url, "pdf_link":pdf_abs_url, "content_type":pdf_content_type, "event_type":"playwright_pdf_link_wrong_content_type"})
                                        except Exception as e_fetch_link:
                                            logger.warning(f"Playwright: Error fetching PDF link {pdf_abs_url}: {e_fetch_link}", exc_info=True, extra={"url": url, "pdf_link": pdf_abs_url, "event_type": "playwright_fetch_pdf_link_error"})
                            else:
                                logger.info(f"Playwright: No PDF links found in DOM of {final_url_after_redirects}", extra={"url": url, "final_url": final_url_after_redirects, "event_type": "playwright_no_pdf_links_found"})
                        
                        logger.info(f"Playwright: Closing browser for {url}", extra={"url": url, "event_type": "playwright_browser_close"})
                        await browser.close()
                        if playwright_result:
                            return playwright_result
                except PlaywrightError as e_pw:
                    logger.error(f"Playwright processing error for {url}: {e_pw}", exc_info=True, extra={"url": url, "event_type": "playwright_general_error"})
                except Exception as e_pw_unhandled:
                    logger.error(f"Unhandled Playwright exception for {url}: {e_pw_unhandled}", exc_info=True, extra={"url": url, "event_type": "playwright_unhandled_exception"})
                
                # If Python Playwright didn't get a result, try Node.js stealth fetcher
                if not playwright_result:
                    logger.info(f"Python Playwright failed for {url}, trying Node.js stealth fetcher.", extra={"url": url, "event_type": "nodejs_fallback_attempt"})
                    nodejs_result = await _run_nodejs_stealth_fetcher(url)
                    # _run_nodejs_stealth_fetcher now returns FileResult, HTMLResult, or Failure
                    if nodejs_result.type == "file" or nodejs_result.type == "html":
                        logger.info(f"Node.js stealth fetcher succeeded for {url} with type: {nodejs_result.type}.", 
                                    extra={"url": url, "result_type": nodejs_result.type, "event_type": "nodejs_fallback_success"})
                        return nodejs_result # This could be FileResult or HTMLResult
                    else: # Node.js fetcher also failed
                        logger.warning(f"Node.js stealth fetcher also failed for {url}. Reason: {nodejs_result.reason}", 
                                       extra={"url": url, "reason": nodejs_result.reason, "event_type": "nodejs_fallback_failure"})
                        # playwright_result remains None or the failure from Python's Playwright, so overall failure.

        # If all methods (direct HTTP, Python Playwright, and Node.js fetcher if attempted) failed:
        logger.warning(f"All scraping attempts failed for {url} after trying direct, Python Playwright, and Node.js fetcher (if enabled).",
                       extra={"url": url, "event_type": "all_methods_failed_final",
                              "direct_http_exception": str(direct_http_exception) if direct_http_exception else "None"})
        # Determine the most relevant status code if one was captured
        final_status_code = None
        if isinstance(direct_http_exception, httpx.HTTPStatusError):
            final_status_code = direct_http_exception.response.status_code
        # We could also try to get a status from Playwright failure if that was the last thing tried.
        
        return Failure(type="failure", 
                       reason="All scraping attempts (direct HTTP, Playwright if enabled) failed to yield valid content.", 
                       status_code=final_status_code)

    # `async with semaphore:` and `async with client:` will handle their own cleanup.
    # The "Semaphore released" and "Finished advanced scrape" logs are implicitly covered by exiting these blocks.
    # Or, if explicit end-of-function logging is desired, it should be outside the `async with semaphore` if it means the absolute end.
    # For now, the structure implies completion when a result is returned or the final Failure is returned.

async def click_download_and_capture(page, timeout: int = 30000) -> Optional[bytes]:
    logger.info(f"Attempting to click PDF download and capture for {page.url}", extra={"url": page.url, "event_type": "click_download_capture_start"})
    # Common selectors for PDF download links/buttons
    selectors = [
        "a[href$='.pdf'][download]",
        "button:has-text('Download PDF')",
        "a:has-text('Download PDF')",
        "a[href$='.pdf']",
        "button[aria-label*='Download PDF']",
        "a[aria-label*='Download PDF']",
        "button:has-text('PDF')",
        "a:has-text('PDF')",
        "a[href*='downloadSgArticle']",
        "a[href*='pdfLink']",
        "button[id*='download']",
        "a[id*='download']",
        "input[type='submit'][value*='Download']",
    ]

    for i, selector in enumerate(selectors):
        try:
            logger.info(f"Trying selector ({i+1}/{len(selectors)}): '{selector}' on {page.url}", extra={"url": page.url, "selector": selector, "event_type": "click_download_selector_try"})
            element = page.locator(selector).first
            
            if not await element.is_visible(timeout=2000) or not await element.is_enabled(timeout=2000):
                logger.info(f"Selector '{selector}' found but not visible/enabled on {page.url}", extra={"url": page.url, "selector": selector, "event_type": "click_download_selector_not_interactive"})
                continue

            async with page.expect_download(timeout=timeout) as download_info:
                logger.info(f"Expecting download after clicking '{selector}' on {page.url}", extra={"url": page.url, "selector": selector, "event_type": "click_download_expecting"})
                await element.click(timeout=5000) 
            
            download = await download_info.value
            
            temp_file_path = await download.path()
            if not temp_file_path:
                logger.warning(f"Download via '{selector}' on {page.url} did not provide a file path.", extra={"url": page.url, "selector": selector, "event_type": "click_download_no_path"})
                await download.delete()
                continue

            logger.info(f"Download triggered by '{selector}' on {page.url}, saved to temp path: {temp_file_path}", extra={"url": page.url, "selector": selector, "temp_path": temp_file_path, "event_type": "click_download_triggered"})
            
            pdf_bytes = None
            try:
                with open(temp_file_path, "rb") as f:
                    pdf_bytes = f.read()
            except Exception as e_read:
                logger.error(f"Error reading downloaded file {temp_file_path} for {page.url}: {e_read}", exc_info=True, extra={"url": page.url, "path": temp_file_path, "event_type": "click_download_read_error"})
                await download.delete()
                return None

            await download.delete()
            logger.info(f"Successfully captured {len(pdf_bytes)} bytes from download triggered by '{selector}' on {page.url}", extra={"url": page.url, "selector": selector, "bytes_captured": len(pdf_bytes), "event_type": "click_download_capture_success"})
            return pdf_bytes

        except PlaywrightTimeoutError:
            logger.warning(f"Timeout waiting for download or click with selector '{selector}' on {page.url}", extra={"url": page.url, "selector": selector, "event_type": "click_download_timeout"})
        except PlaywrightError as e_pw_click:
             logger.warning(f"Playwright error with selector '{selector}' on {page.url}: {str(e_pw_click)}", extra={"url": page.url, "selector": selector, "error": str(e_pw_click), "event_type": "click_download_playwright_error"})
        except Exception as e:
            logger.error(f"Unexpected error during click_download_and_capture with selector '{selector}' on {page.url}: {e}", exc_info=True, extra={"url": page.url, "selector": selector, "event_type": "click_download_unexpected_error"})
    
    logger.warning(f"Failed to click and capture download for {page.url} after trying all selectors.", extra={"url": page.url, "event_type": "click_download_all_selectors_failed"})
    return None

async def _run_nodejs_stealth_fetcher(url: str) -> ResolveResult:
    """
    Runs the Node.js stealth_fetcher.js script as a subprocess.
    """
    script_path = os.path.join("tools", "stealth_fetcher.js")
    node_executable = "node"

    if not os.path.exists(script_path):
        logger.error(f"Node.js fetcher script not found at {script_path}", 
                     extra={"url": url, "script_path": script_path, "event_type": "nodejs_script_not_found"})
        return Failure(type="failure", reason=f"Node.js stealth_fetcher.js script not found at {script_path}.")

    # Ensure the download directory for Node.js script exists (it creates it, but good to be sure)
    # The Node.js script now creates 'workspace/downloads/stealth_dl/'
    node_dl_dir = os.path.join(os.getcwd(), "workspace", "downloads", "stealth_dl")
    if not os.path.exists(node_dl_dir):
        try:
            os.makedirs(node_dl_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create Node.js download directory {node_dl_dir}: {e}", 
                         extra={"event_type": "dir_create_fail_node_dl", "path": node_dl_dir})
            # Non-fatal, as Node.js script might still succeed with HTML or PDF URL

    cmd = [node_executable, script_path, url]
    logger.info(f"Running Node.js stealth fetcher for {url}: {' '.join(cmd)}", 
                extra={"url": url, "command": " ".join(cmd), "event_type": "nodejs_fetch_start"})

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # Increased timeout for Node.js script execution, as Playwright can be slow
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120.0) # 2 minutes timeout
        except asyncio.TimeoutError:
            logger.error(f"Node.js stealth fetcher for {url} timed out after 120s.",
                         extra={"url": url, "event_type": "nodejs_fetch_timeout"})
            try:
                process.kill()
                await process.wait() # Ensure process is cleaned up
            except Exception as e_kill:
                logger.error(f"Error killing timed-out Node.js process: {e_kill}", extra={"url":url, "event_type":"nodejs_kill_error"})
            return Failure(type="failure", reason="Node.js script execution timed out.")


        if process.returncode != 0:
            stderr_str = stderr.decode('utf-8', errors='replace').strip() if stderr else "Unknown error"
            logger.error(f"Node.js stealth fetcher for {url} exited with code {process.returncode}. Stderr: {stderr_str}", 
                         extra={"url": url, "returncode": process.returncode, "stderr": stderr_str, "event_type": "nodejs_fetch_error_exit"})
            try:
                error_data = json.loads(stderr_str)
                message = error_data.get("message", stderr_str)
                return Failure(type="failure", reason=f"Node.js script error: {message}")
            except json.JSONDecodeError:
                return Failure(type="failure", reason=f"Node.js script failed (code {process.returncode}): {stderr_str}")

        stdout_str = stdout.decode('utf-8', errors='replace').strip() if stdout else ""
        if not stdout_str:
            logger.warning(f"Node.js stealth fetcher for {url} produced no stdout.", 
                           extra={"url": url, "event_type": "nodejs_fetch_no_stdout"})
            return Failure(type="failure", reason="Node.js script produced no output.")

        try:
            result_json = json.loads(stdout_str)
            status = result_json.get("status")
            final_url_from_node = result_json.get("final_url", url)

            if status == "pdf_downloaded":
                pdf_path = result_json.get("path")
                if pdf_path and os.path.exists(pdf_path):
                    # Validate the PDF downloaded by Node.js
                    is_valid, reason = is_pdf_content_valid(pdf_path)
                    if is_valid:
                        logger.info(f"Node.js stealth fetcher downloaded a valid PDF: {pdf_path}", 
                                     extra={"url": url, "path": pdf_path, "event_type": "nodejs_pdf_downloaded_valid"})
                        return FileResult(type="file", path=pdf_path, url=final_url_from_node)
                    else:
                        logger.warning(f"Node.js downloaded PDF {pdf_path} is invalid: {reason}. Deleting.", 
                                       extra={"url": url, "path": pdf_path, "reason": reason, "event_type": "nodejs_pdf_downloaded_invalid"})
                        try: os.remove(pdf_path)
                        except OSError as e_del: logger.error(f"Error deleting invalid PDF from Node.js: {e_del}", extra={"path":pdf_path})
                        return Failure(type="failure", reason=f"Node.js downloaded PDF was invalid: {reason}")
                else:
                    logger.error(f"Node.js reported PDF downloaded but path missing or invalid: {pdf_path}",
                                 extra={"url":url, "path":pdf_path, "event_type":"nodejs_pdf_path_error"})
                    return Failure(type="failure", reason="Node.js reported PDF download but path was invalid.")

            elif status == "success_pdf_url":
                pdf_direct_url = result_json.get("pdf_url")
                if pdf_direct_url:
                    logger.info(f"Node.js returned a direct PDF URL: {pdf_direct_url}. Attempting download.",
                                 extra={"url": url, "pdf_url": pdf_direct_url, "event_type": "nodejs_pdf_url_provided_attempt_dl"})
                    # Use httpx to download this URL (leverages existing retry, proxy logic if _make_request_with_retry is adapted or similar is used)
                    async with httpx.AsyncClient(follow_redirects=True, timeout=REQUEST_TIMEOUT * 2) as pdf_client: # Longer timeout for PDF download
                        try:
                            # Re-use _make_request_with_retry for robustness
                            pdf_response = await _make_request_with_retry(pdf_client, "GET", pdf_direct_url, headers={"Accept": "application/pdf,*/*"})
                            # _handle_pdf_download will save, validate, and return FileResult or Failure
                            return await _handle_pdf_download(pdf_response.content, pdf_direct_url, "nodejs_pdf_url_download")
                        except Exception as e_dl:
                            logger.error(f"Failed to download PDF from URL provided by Node.js ({pdf_direct_url}): {e_dl}",
                                         exc_info=True, extra={"url":url, "pdf_url":pdf_direct_url, "event_type":"nodejs_pdf_url_download_fail"})
                            return Failure(type="failure", reason=f"Failed to download PDF from Node.js-provided URL: {e_dl}")
                else:
                    logger.error(f"Node.js reported success_pdf_url but no pdf_url provided.",
                                 extra={"url":url, "event_type":"nodejs_pdf_url_missing"})
                    return Failure(type="failure", reason="Node.js reported success_pdf_url but pdf_url was missing.")

            elif status == "success": # HTML content
                html_text = result_json.get("html")
                # The Node.js script already uses Readability. We can trust its 'html' output
                # or re-process 'main_text' if preferred.
                # For now, use the 'html' (which is Readability's processed HTML)
                if html_text and len(html_text) >= MIN_HTML_CONTENT_LENGTH: # Check length again
                    logger.info(f"Node.js stealth fetcher success (HTML). Length: {len(html_text)}", 
                                 extra={"url": url, "final_url": final_url_from_node, "length": len(html_text), "event_type": "nodejs_fetch_success_html"})
                    # Pass the already processed HTML (from Node's Readability)
                    # The HTMLResult expects the *extracted text*, not raw HTML.
                    # We should use _extract_html_text on the html_text from Node.js
                    # to be consistent with how other HTML is processed.
                    extracted_node_html = await _extract_html_text(html_text, final_url_from_node)
                    if extracted_node_html:
                         return HTMLResult(type="html", text=extracted_node_html, url=final_url_from_node)
                    else:
                        failure_reason = "Node.js HTML content failed Python-side extraction."
                        if result_json.get("source") == "nodejs_stealth_fetcher_buffer_fail_html" and result_json.get("message"):
                            failure_reason += f" (Node.js buffer error: {result_json.get('message')})"
                        logger.warning(f"Node.js HTML content for {url} could not be further extracted by Python's _extract_html_text. Reason: {failure_reason}",
                                       extra={"url":url, "event_type":"nodejs_html_python_extract_fail", "node_message": result_json.get("message")})
                        return Failure(type="failure", reason=failure_reason)

                else: # HTML too short or missing
                    failure_reason = "Node.js script returned success (HTML) but content was too short."
                    if result_json.get("source") == "nodejs_stealth_fetcher_buffer_fail_html" and result_json.get("message"):
                        failure_reason += f" (Node.js buffer error: {result_json.get('message')})"
                    logger.warning(f"Node.js stealth fetcher for {url} returned success (HTML) but content was too short or missing. Reason: {failure_reason}",
                                   extra={"url":url, "length": len(html_text) if html_text else 0, "event_type": "nodejs_fetch_success_short_html", "node_message": result_json.get("message")})
                    return Failure(type="failure", reason=failure_reason)
            
            elif status == "error":
                error_message = result_json.get('message', 'Unknown error from Node.js script')
                logger.error(f"Node.js stealth fetcher for {url} reported an error: {error_message}", 
                             extra={"url": url, "error_message": error_message, "event_type": "nodejs_fetch_script_reported_error"})
                return Failure(type="failure", reason=f"Node.js script error: {error_message}")
            else: # Unknown status
                logger.error(f"Node.js stealth fetcher for {url} returned unknown JSON status: {status}. Output: {stdout_str[:200]}...",
                             extra={"url":url, "status": status, "output_snippet": stdout_str[:200], "event_type": "nodejs_fetch_unknown_json_status"})
                return Failure(type="failure", reason=f"Node.js script returned unknown status: {status}")

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from Node.js stealth fetcher for {url}. Output: {stdout_str[:200]}...",
                         extra={"url": url, "output_snippet": stdout_str[:200], "event_type": "nodejs_fetch_json_decode_error"})
            return Failure(type="failure", reason="Failed to decode JSON output from Node.js script.")

    except FileNotFoundError: # For node executable itself
        logger.critical(f"Node.js executable '{node_executable}' not found. Cannot run stealth fetcher.", 
                        extra={"url": url, "node_executable": node_executable, "event_type": "nodejs_executable_not_found"})
        return Failure(type="failure", reason=f"Node.js executable '{node_executable}' not found.")
    except Exception as e:
        logger.critical(f"Exception running Node.js stealth fetcher for {url}: {e}", exc_info=True, 
                        extra={"url": url, "event_type": "nodejs_fetch_exception"})
        return Failure(type="failure", reason=f"Exception running Node.js script: {e}")


if __name__ == "__main__":
    async def main_test():
        # Configure logger for local testing if not already configured by autogen
        if not logger.handlers or not isinstance(logger.handlers[0].formatter, JsonFormatter): # type: ignore
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
