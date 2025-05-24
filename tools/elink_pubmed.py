import asyncio
import httpx
import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from ratelimit import limits, RateLimitException
import time
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

NCBI_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
TOOL_NAME = "LiteratureAgentElinkToolV2"
API_EMAIL = os.getenv("API_EMAIL")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

CALLS_PER_SECOND = 10 if PUBMED_API_KEY else 3
ONE_SECOND = 1
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

if not API_EMAIL:
    print("Warning: API_EMAIL not found in .env file. NCBI ELink calls may be limited or fail if this is required by NCBI.")

@limits(calls=CALLS_PER_SECOND, period=ONE_SECOND)
async def _execute_ncbi_utility_request(client: httpx.AsyncClient, base_url: str, params: Dict[str, Any]) -> httpx.Response:
    """
    Executes an HTTP GET request to an NCBI utility, respecting rate limits.
    """
    return await client.get(base_url, params=params)

async def _convert_to_pmid(identifier: str, id_type_tag: str) -> Optional[str]:
    """
    Converts a given identifier (like DOI or PMC ID) to a PubMed ID (PMID) using ESearch.
    id_type_tag should be "DOI" or "PMCID".
    """
    if not identifier or not identifier.strip():
        print(f"Error: Identifier for {id_type_tag} is empty.")
        return None

    print(f"Attempting to convert {id_type_tag} '{identifier}' to PMID...")
    search_term = f"{identifier.strip()}[{id_type_tag}]"
    
    request_params: Dict[str, Any] = {
        "db": "pubmed",
        "term": search_term,
        "retmode": "xml",
        "tool": TOOL_NAME,
        "email": API_EMAIL,
    }
    if PUBMED_API_KEY:
        request_params["api_key"] = PUBMED_API_KEY
    
    request_params = {k: v for k, v in request_params.items() if v is not None}

    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await _execute_ncbi_utility_request(client, NCBI_ESEARCH_URL, request_params)
                response.raise_for_status()

            xml_content = response.text
            if not xml_content.strip():
                print(f"Warning: Empty ESearch XML response for {id_type_tag} {identifier} (attempt {attempt + 1}/{RETRY_ATTEMPTS}).")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                    continue
                else:
                    return None
            
            root = ET.fromstring(xml_content)
            
            error_messages = root.findall(".//ERROR")
            if error_messages:
                for error_msg_element in error_messages:
                    if error_msg_element.text:
                        print(f"NCBI ESearch API Error for {id_type_tag} {identifier}: {error_msg_element.text.strip()}")
                return None

            id_list = root.find(".//IdList")
            if id_list is not None:
                pmid_element = id_list.find("./Id")
                if pmid_element is not None and pmid_element.text:
                    found_pmid = pmid_element.text.strip()
                    print(f"Successfully converted {id_type_tag} '{identifier}' to PMID: {found_pmid}")
                    return found_pmid
            
            count_element = root.find(".//Count")
            if count_element is not None and count_element.text == "0":
                print(f"No PMID found for {id_type_tag} '{identifier}'. Search returned 0 results.")
                return None
            
            print(f"Could not extract PMID from ESearch response for {id_type_tag} {identifier}. XML structure might be unexpected.")
            return None 

        except httpx.HTTPStatusError as e:
            print(f"HTTP error during ESearch for {id_type_tag} {identifier} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e.response.status_code}")
            if e.response.status_code in [400, 404, 429]:
                if e.response.status_code == 429 and attempt < RETRY_ATTEMPTS - 1:
                    print("Explicit 429 received from ESearch, will retry after delay...")
                else:
                    break 
        except RateLimitException:
            print(f"Rate limit actively hit during ESearch for {id_type_tag} {identifier} (attempt {attempt + 1}/{RETRY_ATTEMPTS}).")
            if attempt < RETRY_ATTEMPTS - 1:
                 await asyncio.sleep(RETRY_DELAY_SECONDS)
            continue
        except httpx.RequestError as e:
            print(f"Request error during ESearch for {id_type_tag} {identifier} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {str(e)}")
        except ET.ParseError as e:
            print(f"XML parsing error during ESearch for {id_type_tag} {identifier} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred during ESearch for {id_type_tag} {identifier} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {type(e).__name__} - {e}")
        
        if attempt < RETRY_ATTEMPTS - 1:
            await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        else:
            print(f"Failed to convert {id_type_tag} '{identifier}' to PMID after {RETRY_ATTEMPTS} attempts.")
    
    return None

async def get_pubmed_prlinks(pmid: str) -> List[str]:
    """
    Obtains publisher-provided full-text or PDF links for a given PubMed ID
    via NCBI ELink's "prlinks" command using httpx and manual XML parsing.
    """
    urls: List[str] = []
    if not pmid or not pmid.strip():
        print("Error: PMID is empty or invalid.")
        return urls

    request_params: Dict[str, Any] = {
        "dbfrom": "pubmed",
        "cmd": "prlinks",
        "id": pmid.strip(),
        "retmode": "xml",
        "tool": TOOL_NAME,
        "email": API_EMAIL,
    }
    if PUBMED_API_KEY:
        request_params["api_key"] = PUBMED_API_KEY

    request_params = {k: v for k, v in request_params.items() if v is not None}

    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await _execute_ncbi_utility_request(client, NCBI_ELINK_URL, request_params)
                response.raise_for_status()

            xml_content = response.text
            if not xml_content.strip():
                print(f"Warning: Empty ELink XML response for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}).")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                    continue
                else:
                    return urls

            root = ET.fromstring(xml_content)
            error_messages = root.findall(".//ERROR")
            if error_messages:
                for error_msg_element in error_messages:
                    if error_msg_element.text:
                        print(f"NCBI ELink API Error for PMID {pmid}: {error_msg_element.text.strip()}")
                return urls

            found_urls_in_xml = False
            for id_url_set_element in root.findall(".//IdUrlSet"):
                for obj_url_element in id_url_set_element.findall("./ObjUrl"):
                    url_tag = obj_url_element.find("./Url")
                    if url_tag is not None and url_tag.text:
                        urls.append(url_tag.text.strip())
                        found_urls_in_xml = True
            
            if not found_urls_in_xml:
                 for link_element in root.findall(".//LinkSetDb/Link"):
                    url_tag = link_element.find("./Url")
                    if url_tag is not None and url_tag.text:
                        urls.append(url_tag.text.strip())
                        found_urls_in_xml = True
            
            return list(set(urls))

        except httpx.HTTPStatusError as e:
            print(f"HTTP error during ELink for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e.response.status_code} - {e.response.text}")
            if e.response.status_code in [400, 404, 429]:
                if e.response.status_code == 429 and attempt < RETRY_ATTEMPTS - 1:
                    print("Explicit 429 received from ELink, will retry after delay...")
                else:
                    break
        except RateLimitException:
            print(f"Rate limit actively hit during ELink for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}).")
            if attempt < RETRY_ATTEMPTS - 1:
                 await asyncio.sleep(RETRY_DELAY_SECONDS) 
            continue 
        except httpx.RequestError as e:
            print(f"Request error during ELink for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {str(e)}")
        except ET.ParseError as e:
            print(f"XML parsing error during ELink for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            break 
        except Exception as e:
            print(f"An unexpected error occurred during ELink for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {type(e).__name__} - {e}")
        
        if attempt < RETRY_ATTEMPTS - 1:
            await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1)) 
        else:
            print(f"Failed to get ELinks for PMID {pmid} after {RETRY_ATTEMPTS} attempts.")

    return list(set(urls))

async def get_article_links(identifier: str, id_type: str) -> List[str]:
    """
    Obtains publisher-provided full-text or PDF links for a given article identifier
    (PMID, DOI, or PMC ID).

    Args:
        identifier: The article identifier string.
        id_type: The type of identifier provided. Must be one of "pmid", "doi", or "pmc".

    Returns:
        A list of URL strings, or an empty list if no links are found or an error occurs.
    """
    pmid_to_use: Optional[str] = None
    normalized_id_type = id_type.lower().strip()

    if not identifier or not identifier.strip():
        print(f"Error: Identifier is empty for type '{normalized_id_type}'.")
        return []

    if normalized_id_type == "pmid":
        pmid_to_use = identifier.strip()
    elif normalized_id_type == "doi":
        pmid_to_use = await _convert_to_pmid(identifier, "DOI")
    elif normalized_id_type == "pmc":
        pmid_to_use = await _convert_to_pmid(identifier, "PMCID")
    else:
        print(f"Error: Unsupported id_type '{id_type}'. Must be 'pmid', 'doi', or 'pmc'.")
        return []

    if pmid_to_use:
        return await get_pubmed_prlinks(pmid_to_use)
    else:
        print(f"Could not obtain PMID for identifier '{identifier}' (type: {id_type}). Cannot fetch links.")
        return []

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch publisher links for PubMed articles using PMID, DOI, or PMC ID.")
    parser.add_argument("--identifier", type=str, help="The article identifier (PMID, DOI, or PMC ID).")
    parser.add_argument("--type", type=str, choices=["pmid", "doi", "pmc"], help="The type of identifier provided.")

    args = parser.parse_args()

    async def run_single_query(identifier: str, id_type: str):
        print(f"\n--- Querying for identifier: '{identifier}' (type: {id_type}) ---")
        try:
            links = await get_article_links(identifier, id_type)
            if links:
                print(f"Found links for '{identifier}':")
                for link_url in links:
                    print(f"- {link_url}")
            else:
                print(f"No links found or error for '{identifier}'.")
        except Exception as e:
            print(f"Error during query for '{identifier}': {e}")

    async def run_test_suite():
        print("\n--- Running Test Suite ---")
        test_cases = [
            {"id": "31345905", "type": "pmid", "desc": "Valid PMID"},
            {"id": "10.1016/j.cell.2020.01.001", "type": "doi", "desc": "Valid DOI"},
            {"id": "PMC3499990", "type": "pmc", "desc": "Valid PMC ID (PMC3499990)"}, # Corresponds to PMID 23144561
            {"id": "1", "type": "pmid", "desc": "Early PMID"},
            {"id": "invalid_id_string", "type": "pmid", "desc": "Invalid PMID string"},
            {"id": " ", "type": "pmid", "desc": "Empty PMID string"},
            {"id": "10.invalid/doi", "type": "doi", "desc": "Invalid DOI string"},
            {"id": "PMCinvalid", "type": "pmc", "desc": "Invalid PMC string"},
            {"id": "33577005", "type": "pmid", "desc": "Another valid PMID"},
            {"id": "not_a_real_id", "type": "doi", "desc": "Non-existent DOI"},
        ]

        for test_case in test_cases:
            print(f"\n--- Testing with {test_case['desc']}: '{test_case['id']}' (type: {test_case['type']}) ---")
            try:
                links = await get_article_links(test_case["id"], test_case["type"])
                if links:
                    print(f"Found links for {test_case['desc']} '{test_case['id']}':")
                    for link_url in links:
                        print(f"- {link_url}")
                else:
                    print(f"No links found or error for {test_case['desc']} '{test_case['id']}'.")
            except Exception as e:
                print(f"Error during test for {test_case['desc']} '{test_case['id']}': {e}")
            
            await asyncio.sleep(0.5) # Small delay between tests

    if args.identifier and args.type:
        asyncio.run(run_single_query(args.identifier, args.type))
    else:
        print("No specific identifier provided, running test suite instead.")
        print("Usage: python tools/elink_pubmed.py --identifier <ID> --type <pmid|doi|pmc>")
        asyncio.run(run_test_suite())
