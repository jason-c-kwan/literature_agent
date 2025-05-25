"""
Resolves a URL to determine if it yields HTML full text or a downloadable PDF.
"""
import asyncio
import datetime
import os
import logging
import json
from typing import Literal, NamedTuple, Union
from urllib.parse import urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from readability import Document # readability-lxml
import pymupdf4llm # For checking PDF content
import fitz # PyMuPDF, pymupdf4llm is a wrapper around this

# --- Configuration ---
CONFIG_PATH = "config/settings.yaml"

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

async def _make_request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    for attempt in range(MAX_RETRIES + 1): 
        log_extra = {"url": url, "method": method, "attempt": attempt + 1, "max_retries": MAX_RETRIES}
        try:
            logger.info(f"Requesting {method} {url}, attempt {attempt + 1}", extra={**log_extra, "event_type": f"{method.lower()}_attempt"})
            response = await client.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
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
            try:
                head_response = await _make_request_with_retry(client, "HEAD", url)
                content_type = head_response.headers.get("content-type", "").lower()
                if content_type.startswith("application/pdf"):
                    logger.info(f"HEAD indicates PDF for {url}. Proceeding to download.", extra={"url": url, "content_type": content_type, "event_type": "head_pdf_detected"})
                    pdf_response = await _make_request_with_retry(client, "GET", url) 

                    if not os.path.exists(DOWNLOADS_DIR):
                        os.makedirs(DOWNLOADS_DIR)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    url_part = "".join(c if c.isalnum() else "_" for c in url.split("://")[-1][:50])
                    file_name = f"{timestamp}_{url_part}.pdf"
                    file_path = os.path.join(DOWNLOADS_DIR, file_name)

                    with open(file_path, "wb") as f:
                        f.write(pdf_response.content)
                    logger.info(f"PDF downloaded to {file_path}", extra={"url": url, "path": file_path, "event_type": "pdf_download_success"})

                    is_valid, reason = is_pdf_content_valid(file_path)
                    if not is_valid:
                        logger.warning(f"Downloaded PDF {file_path} invalid: {reason}. Deleting.", extra={"url": url, "path": file_path, "reason": reason, "event_type": "pdf_invalid"})
                        try: os.remove(file_path)
                        except OSError: pass
                        return Failure(type="failure", reason=f"Downloaded PDF appears invalid: {reason}")
                    logger.info(f"PDF {file_path} is valid.", extra={"url": url, "path": file_path, "event_type": "pdf_valid"})
                    return FileResult(type="file", path=file_path)

            except httpx.HTTPStatusError as e:
                if e.response.status_code in [403, 405]: 
                    logger.info(f"HEAD request for {url} failed with {e.response.status_code}, trying GET.", extra={"url": url, "status_code": e.response.status_code, "event_type": "head_fail_try_get"})
                else: 
                    return Failure(type="failure", reason=f"HEAD request failed: {e!s}")
            
            get_response = await _make_request_with_retry(client, "GET", url)
            content_type = get_response.headers.get("content-type", "").lower()

            if "html" in content_type:
                html_content = get_response.text
                logger.info(f"GET request for {url} returned HTML.", extra={"url": url, "content_type": content_type, "event_type": "get_html_detected"})
                potentially_paywalled_full_html = is_html_potentially_paywalled(html_content)
                if potentially_paywalled_full_html:
                    logger.info(f"HTML for {url} shows paywall indicators.", extra={"url": url, "event_type": "html_paywall_indicator"})

                soup = BeautifulSoup(html_content, "html.parser")
                extracted_text = None
                try:
                    doc = Document(html_content)
                    main_content_html = doc.summary(html_partial=True)
                    summary_soup = BeautifulSoup(main_content_html, "html.parser")
                    current_extracted_text = summary_soup.get_text(separator="\n", strip=True)
                    if current_extracted_text and len(current_extracted_text) > MIN_HTML_CONTENT_LENGTH:
                        extracted_text = current_extracted_text
                        logger.info(f"Extracted HTML content from {url} using readability.", extra={"url": url, "length": len(extracted_text), "event_type": "html_extract_readability_success"})
                except Exception as e:
                    logger.warning(f"Readability failed for {url}: {e!s}", extra={"url": url, "error": str(e), "event_type": "html_extract_readability_fail"})

                if not extracted_text or len(extracted_text) < MIN_HTML_CONTENT_LENGTH:
                    article_tag = soup.find("article")
                    if article_tag:
                        current_extracted_text = article_tag.get_text(separator="\n", strip=True)
                        if current_extracted_text and len(current_extracted_text) > MIN_HTML_CONTENT_LENGTH:
                            extracted_text = current_extracted_text
                            logger.info(f"Extracted HTML content from {url} using <article> tag.", extra={"url": url, "length": len(extracted_text), "event_type": "html_extract_article_success"})

                if extracted_text:
                    if len(extracted_text) >= MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE:
                        logger.info(f"Long HTML content from {url} overrides paywall flag.", extra={"url": url, "length": len(extracted_text), "event_type": "html_long_override_paywall"})
                        return HTMLResult(type="html", text=extracted_text)
                    elif potentially_paywalled_full_html:
                        logger.warning(f"Paywall indicators and short HTML content for {url}.", extra={"url": url, "length": len(extracted_text), "event_type": "html_paywall_short_content_fail"})
                        return Failure(type="failure", reason="Paywall indicators in full HTML and extracted content is short.")
                    elif len(extracted_text) >= MIN_HTML_CONTENT_LENGTH:
                        logger.info(f"Sufficient HTML content from {url}.", extra={"url": url, "length": len(extracted_text), "event_type": "html_sufficient_content_success"})
                        return HTMLResult(type="html", text=extracted_text)
                    else:
                        logger.warning(f"HTML content extracted from {url} but too short.", extra={"url": url, "length": len(extracted_text), "event_type": "html_extract_too_short_fail"})
                        return Failure(type="failure", reason="HTML detected, content extracted but too short, no strong paywall signs on page.")
                else:
                    reason_msg = "Paywall indicators in full HTML and no main content extracted." if potentially_paywalled_full_html else "HTML detected, but no main content could be extracted."
                    logger.warning(f"No main HTML content extracted from {url}. Paywalled: {potentially_paywalled_full_html}", extra={"url": url, "paywalled_full_html": potentially_paywalled_full_html, "event_type": "html_no_content_extracted_fail"})
                    return Failure(type="failure", reason=reason_msg)

            if content_type.startswith("application/pdf"): 
                logger.info(f"GET indicates PDF for {url}. Proceeding to save.", extra={"url": url, "content_type": content_type, "event_type": "get_pdf_detected"})
                if not os.path.exists(DOWNLOADS_DIR): os.makedirs(DOWNLOADS_DIR)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                url_part = "".join(c if c.isalnum() else "_" for c in url.split("://")[-1][:50])
                file_name = f"{timestamp}_{url_part}.pdf"
                file_path = os.path.join(DOWNLOADS_DIR, file_name)
                with open(file_path, "wb") as f: f.write(get_response.content)
                logger.info(f"PDF (via GET) downloaded to {file_path}", extra={"url": url, "path": file_path, "event_type": "pdf_get_download_success"})
                
                is_valid, reason = is_pdf_content_valid(file_path)
                if not is_valid:
                    logger.warning(f"Downloaded PDF (via GET) {file_path} invalid: {reason}. Deleting.", extra={"url": url, "path": file_path, "reason": reason, "event_type": "pdf_get_invalid"})
                    try: os.remove(file_path)
                    except OSError: pass
                    return Failure(type="failure", reason=f"Downloaded PDF (via GET) appears invalid: {reason}")
                logger.info(f"PDF (via GET) {file_path} is valid.", extra={"url": url, "path": file_path, "event_type": "pdf_get_valid"})
                return FileResult(type="file", path=file_path)

            logger.warning(f"Unsupported Content-Type '{content_type}' for {url}.", extra={"url": url, "content_type": content_type, "event_type": "unsupported_content_type_fail"})
            return Failure(type="failure", reason=f"Content-Type '{content_type}' is not PDF or HTML.")

        except httpx.HTTPStatusError as e: 
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
            "https://www.bmj.com/content/372/bmj.n386",
            "https://arxiv.org/pdf/2303.10130.pdf",
            "https://www.nature.com/articles/s41586-021-03491-6",
            "http://nonexistenturl12345.com/article.html", 
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "http://httpstat.us/503", 
            "http://httpstat.us/404", 
        ]
        results = await asyncio.gather(*(resolve_content(url) for url in test_urls), return_exceptions=True)

        for test_url, result in zip(test_urls, results):
            print(f"\n--- Result for: {test_url} ---")
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
