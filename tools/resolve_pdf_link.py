"""
Resolves a URL to determine if it yields HTML full text or a downloadable PDF.
"""
import asyncio
import datetime
import os
from typing import Literal, NamedTuple, Union

import httpx
import yaml
from bs4 import BeautifulSoup
from readability import Document # readability-lxml
import pymupdf4llm # For checking PDF content
import fitz # PyMuPDF, pymupdf4llm is a wrapper around this

# Configuration
CONFIG_PATH = "config/settings.yaml"
DEFAULT_TIMEOUT = 15  # Default timeout in seconds if not in config
MIN_PDF_SIZE_KB = 10 # Minimum size in KB for a PDF to be considered potentially valid
MIN_HTML_CONTENT_LENGTH = 200 # Minimum character length for extracted HTML to be considered valid preview/abstract
MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE = 7000 # If extracted text is longer than this, override initial paywall flag

def load_config():
    """Loads configuration from settings.yaml."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

config = load_config()
REQUEST_TIMEOUT = config.get("resolve_content_timeout", DEFAULT_TIMEOUT)

# Result types
class Failure(NamedTuple):
    """Represents a failure to resolve content."""
    type: Literal["failure"]
    reason: str

class HTMLResult(NamedTuple):
    """Represents successfully resolved HTML content."""
    type: Literal["html"]
    text: str

class FileResult(NamedTuple):
    """Represents a successfully downloaded file."""
    type: Literal["file"]
    path: str

ResolveResult = Union[Failure, HTMLResult, FileResult]

DOWNLOADS_DIR = "workspace/downloads"
PAYWALL_KEYWORDS = [
    "access this article", "buy this article", "purchase access", 
    "subscribe to view", "institutional login", "access options", 
    "get access", "full text access", "journal subscription",
    "pay per view", "purchase pdf", "rent this article",
    "limited preview", "unlock this article", "sign in to read",
    "£", "$", "€", "usd", "eur", "gbp" # Currency symbols often indicate payment
]

def is_pdf_content_valid(file_path: str) -> tuple[bool, str]:
    """
    Checks if the downloaded PDF content is valid (not a dummy/placeholder).
    Returns a tuple: (is_valid, reason_if_not_valid)
    """
    try:
        # 1. Check file size
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb < MIN_PDF_SIZE_KB:
            return False, f"File size ({file_size_kb:.2f}KB) is less than minimum ({MIN_PDF_SIZE_KB}KB)."

        # 2. Check content using pymupdf4llm (or fitz directly for page count/text)
        # For simplicity, let's use basic fitz/pymupdf features if pymupdf4llm is too heavy for this.
        # pymupdf4llm.to_markdown might be overkill.
        doc = fitz.open(file_path) # fitz is now imported at module level
        page_count = len(doc)
        
        if page_count == 0:
            doc.close()
            return False, "PDF has 0 pages."

        if page_count == 1: # Suspicious for a full article
            text = ""
            # Load page and get text. Ensure doc is closed even if get_text fails.
            try:
                page = doc.load_page(0)
                text = page.get_text("text")
            finally:
                doc.close() # Close doc after text extraction or if it fails
            
            text_original_for_len_check = text # Preserve original text for original length checks
            text_lower_stripped = text.lower().strip()
            
            # Check 1a: Exact match for "dummy pdf file" (case-insensitive, stripped)
            if text_lower_stripped == "dummy pdf file":
                 return False, "PDF content suggests it's a dummy PDF (e.g., contains 'dummy PDF')."

            # Check 1b: General dummy phrases (original Check 1 logic)
            if "dummy" in text_lower_stripped and "pdf" in text_lower_stripped and len(text_original_for_len_check) < 200:
                 return False, "PDF content suggests it's a dummy PDF (e.g., contains 'dummy PDF')."
            
            # Check 2: Other placeholder phrases for short content
            if len(text_original_for_len_check) < 100 and \
               ("placeholder" in text_lower_stripped or \
                "abstract only" in text_lower_stripped or \
                "cover page" in text_lower_stripped):
                 return False, "PDF content suggests it's a placeholder or abstract-only."
            
            # Check 3: Very short or empty text for a single page document
            if len(text_lower_stripped) < 20: # Check length of stripped, lowercased text
                # This catches cases where text is very minimal and doesn't match specific keywords above.
                return False, f"Single-page PDF has very little text content (approx. {len(text_lower_stripped)} chars)."
            
            # If it's a 1-page PDF with sufficient, non-dummy text, it's valid.
            return True, "" 
        else: # More than 1 page, likely okay unless very small.
            doc.close()
            return True, ""
    except Exception as e:
        return False, f"Error checking PDF content: {e!s}"

def is_html_potentially_paywalled(full_html_content: str) -> bool:
    """
    Checks if the full HTML content of a page shows strong signs of a paywall.
    This is checked BEFORE trying to extract main content.
    """
    # It's important to check the raw HTML because paywall messages
    # might not be part of the "main content" extracted by readability.
    html_lower = full_html_content.lower()
    matches = 0
    for keyword in PAYWALL_KEYWORDS:
        if keyword in html_lower: # Check in full HTML
            matches +=1
    
    # More aggressive check on full HTML:
    # If currency symbols are present AND at least one other paywall keyword.
    if any(currency in html_lower for currency in ["£", "$", "€", "usd", "eur", "gbp"]) and matches >= 1:
        return True
    # If multiple (e.g., 2 or more) general paywall keywords are present in the full HTML.
    if matches >= 2: 
        return True
        
    # Check for specific pattern from user feedback, now on full HTML
    if "access this article for" in html_lower and "buy this article" in html_lower:
        return True
    if "institutional login" in html_lower and ("subscribe" in html_lower or "purchase" in html_lower):
        return True
        
    return False

async def resolve_content(url: str, client: httpx.AsyncClient = None) -> ResolveResult:
    """
    Given any landing-page URL, determines whether it yields HTML full text
    or a downloadable file (PDF), and returns one of three outcomes:
    failure, HTML text, or local file path.

    Args:
        url: The URL to resolve.
        client: An optional httpx.AsyncClient instance.

    Returns:
        A ResolveResult indicating success or failure.
    """
    provided_client = bool(client)
    if not client:
        client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True)

    try:
        # 1. HEAD request to check for PDF
        try:
            head_response = await client.head(url)
            head_response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)

            content_type = head_response.headers.get("content-type", "").lower()
            if content_type.startswith("application/pdf"):
                # Download the PDF
                pdf_response = await client.get(url)
                pdf_response.raise_for_status()

                if not os.path.exists(DOWNLOADS_DIR):
                    os.makedirs(DOWNLOADS_DIR)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize URL to create a filename part
                url_part = "".join(c if c.isalnum() else "_" for c in url.split("://")[-1][:50])
                file_name = f"{timestamp}_{url_part}.pdf"
                file_path = os.path.join(DOWNLOADS_DIR, file_name)

                with open(file_path, "wb") as f:
                    f.write(pdf_response.content)
                
                is_valid, reason = is_pdf_content_valid(file_path)
                if not is_valid:
                    try: 
                        os.remove(file_path)
                    except OSError: 
                        pass # Ignore errors during deletion, but still report failure
                    return Failure(type="failure", reason=f"Downloaded PDF appears invalid: {reason}")
                return FileResult(type="file", path=file_path)

        except httpx.HTTPStatusError as e:
            # If HEAD fails with e.g. 405 Method Not Allowed, or 403, proceed to GET
            if e.response.status_code in [403, 405]:
                pass # Continue to GET request
            else:
                return Failure(type="failure", reason=f"HEAD request failed: {e!s}")
        except httpx.RequestError as e:
            return Failure(type="failure", reason=f"HEAD request network error: {e!s}")


        # 2. GET request for HTML or other content
        get_response = await client.get(url)
        get_response.raise_for_status()

        content_type = get_response.headers.get("content-type", "").lower()

        if "html" in content_type:
            html_content = get_response.text

            # **Step 1**: Check full HTML for strong paywall indicators
            potentially_paywalled_full_html = is_html_potentially_paywalled(html_content)

            # **Step 2**: Always try to extract main content
            soup = BeautifulSoup(html_content, "html.parser")
            extracted_text = None

            # Attempt 1: readability-lxml
            try:
                doc = Document(html_content) # Process full HTML
                # Get title with readability as an additional check for relevance
                # title = doc.title() 
                main_content_html = doc.summary(html_partial=True)
                summary_soup = BeautifulSoup(main_content_html, "html.parser")
                current_extracted_text = summary_soup.get_text(separator="\n", strip=True)
                if current_extracted_text and len(current_extracted_text) > MIN_HTML_CONTENT_LENGTH:
                    # Even if full HTML didn't scream paywall, the extracted main content might still be a snippet.
                    # However, the primary paywall check is now on full HTML.
                    # If we got here, we assume it's not a blatant paywall page.
                    extracted_text = current_extracted_text
            except Exception: # pylint: disable=broad-except
                pass # readability failed, try next method

            # Attempt 2: <article> tag
            if not extracted_text or len(extracted_text) < MIN_HTML_CONTENT_LENGTH:
                article_tag = soup.find("article")
                if article_tag:
                    current_extracted_text = article_tag.get_text(separator="\n", strip=True)
                    if current_extracted_text and len(current_extracted_text) > MIN_HTML_CONTENT_LENGTH:
                        extracted_text = current_extracted_text
            
            # Final decision based on initial paywall check and extracted content length
            if extracted_text:
                if len(extracted_text) >= MIN_CHARS_FOR_FULL_ARTICLE_OVERRIDE:
                    # If content is very long, it's likely full text, override initial paywall flag
                    return HTMLResult(type="html", text=extracted_text)
                elif potentially_paywalled_full_html:
                    # Full HTML looked like a paywall, and extracted text is not long enough to override
                    return Failure(type="failure", reason="Paywall indicators in full HTML and extracted content is short.")
                elif len(extracted_text) >= MIN_HTML_CONTENT_LENGTH:
                    # Full HTML did not look like a paywall, and extracted text is reasonably long
                    return HTMLResult(type="html", text=extracted_text)
                else:
                    # Full HTML no paywall, but extracted text too short
                    return Failure(type="failure", reason="HTML detected, content extracted but too short, no strong paywall signs on page.")
            else: # No text extracted at all
                if potentially_paywalled_full_html:
                    return Failure(type="failure", reason="Paywall indicators in full HTML and no main content extracted.")
                else:
                    return Failure(type="failure", reason="HTML detected, but no main content could be extracted.")

        # If HEAD didn't identify PDF and GET isn't HTML, check GET content-type for PDF again
        if content_type.startswith("application/pdf"):
            if not os.path.exists(DOWNLOADS_DIR):
                os.makedirs(DOWNLOADS_DIR)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            url_part = "".join(c if c.isalnum() else "_" for c in url.split("://")[-1][:50])
            file_name = f"{timestamp}_{url_part}.pdf"
            file_path = os.path.join(DOWNLOADS_DIR, file_name)
            with open(file_path, "wb") as f:
                f.write(get_response.content)
            
            is_valid, reason = is_pdf_content_valid(file_path)
            if not is_valid:
                try:
                    os.remove(file_path)
                except OSError:
                    pass # Ignore errors during deletion, but still report failure
                return Failure(type="failure", reason=f"Downloaded PDF (via GET) appears invalid: {reason}")
            return FileResult(type="file", path=file_path)

        return Failure(type="failure", reason=f"Content-Type '{content_type}' is not PDF or HTML.")

    except httpx.HTTPStatusError as e:
        return Failure(type="failure", reason=f"HTTP error: {e.response.status_code} {e.response.reason_phrase} for URL: {url}")
    except httpx.RequestError as e:
        # Covers network errors, timeouts, etc.
        return Failure(type="failure", reason=f"Request failed: {e!s}")
    except Exception as e: # pylint: disable=broad-except
        return Failure(type="failure", reason=f"An unexpected error occurred: {e!s}")
    finally:
        if not provided_client and client:
            await client.aclose()

if __name__ == "__main__":
    # Example Usage (requires an event loop to run)
    async def main():
        # Test URLs (replace with actual URLs for testing)
        test_urls = [
            "https://www.bmj.com/content/372/bmj.n386",  # Example HTML article
            "https://arxiv.org/pdf/2303.10130.pdf",       # Example direct PDF link
            "https://www.nature.com/articles/s41586-021-03491-6", # Nature article (often HTML)
            "http://nonexistenturl12345.com/article.html", # Non-existent URL
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf" # Another PDF
        ]

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as shared_client:
            for test_url in test_urls:
                print(f"Resolving: {test_url}")
                result = await resolve_content(test_url, client=shared_client)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for uniqueness
                url_part = "".join(c if c.isalnum() else "_" for c in test_url.split("://")[-1][:50])

                if result.type == "file":
                    print(f"  Success (File): {result.path}")
                elif result.type == "html":
                    print(f"  Success (HTML): Extracted text. First 100 chars: {result.text[:100]}...")
                    # Save HTML text to a file
                    if not os.path.exists(DOWNLOADS_DIR):
                        os.makedirs(DOWNLOADS_DIR)
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
