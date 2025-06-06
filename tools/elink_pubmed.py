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

async def _get_pubmed_prlinks_xml(pmid: str) -> List[str]:
    """
    Obtains publisher-provided full-text or PDF links for a given PubMed ID
    via NCBI ELink's "prlinks" command (XML output) using httpx and manual XML parsing.
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

async def _get_article_links_by_id_type_xml(identifier: str, id_type: str) -> List[str]:
    """
    Obtains publisher-provided full-text or PDF links for a given article identifier
    (PMID, DOI, or PMC ID) using the ELink 'prlinks' command (XML output).

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
        return await _get_pubmed_prlinks_xml(pmid_to_use)
    else:
        print(f"Could not obtain PMID for identifier '{identifier}' (type: {id_type}). Cannot fetch links via XML prlinks.")
        return []

async def get_article_links(pmid: str) -> List[str]:
    """
    Retrieves full-text article links from PubMed LinkOut using the 'llinks' command.

    This function queries the NCBI ELink utility for a given PubMed ID (PMID)
    to find associated links, specifically looking for those likely to lead to
    full text (e.g., "Free full text", "PDF", "PubMed Central").

    Args:
        pmid: The PubMed ID (PMID) of the article.

    Returns:
        A list of unique URLs that are likely to provide access to the
        full text of the article, preserving the order of discovery.
        Returns an empty list if no suitable links are found or if an
        error occurs.
    """
    if not pmid or not pmid.strip():
        print(f"Error: PMID is empty or invalid for get_article_links (llinks).")
        return []

    print(f"Attempting to get article links (llinks) for PMID: {pmid}...")
    
    request_params: Dict[str, Any] = {
        "dbfrom": "pubmed",
        "id": pmid.strip(),
        "cmd": "llinks", # Command for LinkOut links
        "retmode": "json", # Request JSON response
        "tool": TOOL_NAME, 
        "email": API_EMAIL,
    }
    if PUBMED_API_KEY:
        request_params["api_key"] = PUBMED_API_KEY
    
    request_params = {k: v for k, v in request_params.items() if v is not None}

    link_keywords = ["Free full text", "Full Text", "PubMed Central", "Europe PMC", "PDF"]
    found_urls: List[str] = []

    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await _execute_ncbi_utility_request(client, NCBI_ELINK_URL, request_params)
                response.raise_for_status() 

            response_json = response.json()
            
            linksets = response_json.get("linksets")
            if not linksets or not isinstance(linksets, list) or not linksets:
                print(f"No 'linksets' found or invalid format in ELink JSON response for PMID {pmid}.")
                return []

            first_linkset = linksets[0]
            if not isinstance(first_linkset, dict):
                print(f"First linkset is not a dictionary for PMID {pmid}.")
                return []
            
            idurls_data = first_linkset.get("idurls")
            if not idurls_data or not isinstance(idurls_data, dict):
                print(f"No 'idurls' dictionary found in ELink JSON for PMID {pmid}.")
                return []

            pmid_key_in_json = pmid.strip()
            pmid_specific_links_data = idurls_data.get(pmid_key_in_json)
            if not pmid_specific_links_data or not isinstance(pmid_specific_links_data, dict):
                print(f"No link data found for PMID key '{pmid_key_in_json}' in 'idurls' for PMID {pmid}.")
                return []
            
            objurls = pmid_specific_links_data.get("objurls")
            if not objurls or not isinstance(objurls, list):
                print(f"No 'objurls' list found for PMID {pmid}, though PMID key was present.")
                return []

            for obj_url_entry in objurls:
                if not isinstance(obj_url_entry, dict):
                    continue

                url_info = obj_url_entry.get("url")
                if not url_info or not isinstance(url_info, dict) or not url_info.get("$"):
                    continue 

                url_str = url_info["$"]
                
                link_text = obj_url_entry.get("linktext", "").lower()
                provider_name = ""
                provider_info = obj_url_entry.get("provider")
                if provider_info and isinstance(provider_info, dict):
                    provider_name = provider_info.get("name", "").lower()

                if any(keyword.lower() in link_text for keyword in link_keywords) or \
                   any(keyword.lower() in provider_name for keyword in link_keywords):
                    if url_str not in found_urls: 
                        found_urls.append(url_str)
            
            print(f"Found {len(found_urls)} relevant LinkOut URLs for PMID {pmid}.")
            return found_urls

        except httpx.HTTPStatusError as e:
            print(f"HTTP error during ELink (llinks) for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429 and attempt < RETRY_ATTEMPTS - 1:
                print("Rate limit (429) hit for ELink (llinks), retrying after delay...")
            elif e.response.status_code in [400, 404] or attempt == RETRY_ATTEMPTS - 1:
                break 
        except RateLimitException:
            print(f"Rate limit actively hit by decorator during ELink (llinks) for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}).")
            if attempt == RETRY_ATTEMPTS - 1:
                print(f"Failed ELink (llinks) for PMID {pmid} due to persistent rate limiting after {RETRY_ATTEMPTS} attempts.")
                break
        except httpx.RequestError as e:
            print(f"Request error during ELink (llinks) for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {str(e)}")
        except json.JSONDecodeError as e: # Ensure json is imported
            print(f"JSON parsing error during ELink (llinks) for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            break 
        except Exception as e:
            print(f"An unexpected error occurred during ELink (llinks) for PMID {pmid} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {type(e).__name__} - {e}")
        
        if attempt < RETRY_ATTEMPTS - 1:
            await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1)) 
        else:
            print(f"Failed to get ELinks (llinks) for PMID {pmid} after {RETRY_ATTEMPTS} attempts.")
            
    return found_urls


if __name__ == "__main__":
    import argparse
    import json # Ensure json is imported for the new function if not already

    parser = argparse.ArgumentParser(description="Fetch publisher links for PubMed articles using PMID, DOI, or PMC ID.")
    parser.add_argument("--identifier", type=str, help="The article identifier (PMID, DOI, or PMC ID).")
    parser.add_argument("--type", type=str, choices=["pmid", "doi", "pmc"], help="The type of identifier provided.")

    args = parser.parse_args()

    async def run_single_query(identifier: str, id_type: str):
        print(f"\n--- Querying for identifier: '{identifier}' (type: {id_type}) using _get_article_links_by_id_type_xml ---")
        try:
            links = await _get_article_links_by_id_type_xml(identifier, id_type)
            if links:
                print(f"Found links (XML prlinks) for '{identifier}':")
                for link_url in links:
                    print(f"- {link_url}")
            else:
                print(f"No links found or error for '{identifier}'.")
        except Exception as e:
            print(f"Error during query for '{identifier}': {e}")

    async def run_new_llinks_test(pmid_to_test: str):
        print(f"\n--- Testing new get_article_links (llinks) for PMID: {pmid_to_test} ---")
        try:
            links = await get_article_links(pmid_to_test)
            if links:
                print(f"Found LinkOut links for PMID '{pmid_to_test}':")
                for link_url in links:
                    print(f"- {link_url}")
            else:
                print(f"No LinkOut links found or error for PMID '{pmid_to_test}'.")
        except Exception as e:
            print(f"Error during get_article_links (llinks) test for PMID '{pmid_to_test}': {e}")

    async def run_test_suite():
        print("\n--- Running Test Suite (XML prlinks) ---")
        test_cases_xml = [
            {"id": "31345905", "type": "pmid", "desc": "Valid PMID (XML)"},
            {"id": "10.1016/j.cell.2020.01.001", "type": "doi", "desc": "Valid DOI"},
            {"id": "PMC3499990", "type": "pmc", "desc": "Valid PMC ID (PMC3499990)"}, # Corresponds to PMID 23144561
            {"id": "1", "type": "pmid", "desc": "Early PMID"},
            {"id": "invalid_id_string", "type": "pmid", "desc": "Invalid PMID string"},
            {"id": " ", "type": "pmid", "desc": "Empty PMID string"},
            {"id": "10.invalid/doi", "type": "doi", "desc": "Invalid DOI string"},
            {"id": "PMCinvalid", "type": "pmc", "desc": "Invalid PMC string"},
            {"id": "33577005", "type": "pmid", "desc": "Another valid PMID"},
            {"id": "not_a_real_id", "type": "doi", "desc": "Non-existent DOI (XML)"},
        ]

        for test_case in test_cases_xml:
            print(f"\n--- Testing with {test_case['desc']}: '{test_case['id']}' (type: {test_case['type']}) using _get_article_links_by_id_type_xml ---")
            try:
                links = await _get_article_links_by_id_type_xml(test_case["id"], test_case["type"])
                if links:
                    print(f"Found links (XML prlinks) for {test_case['desc']} '{test_case['id']}':")
                    for link_url in links:
                        print(f"- {link_url}")
                else:
                    print(f"No links found or error for {test_case['desc']} '{test_case['id']}'.")
            except Exception as e:
                print(f"Error during test for {test_case['desc']} '{test_case['id']}': {e}")
            
            await asyncio.sleep(0.5) # Small delay between tests

    if args.identifier and args.type:
        asyncio.run(run_single_query(args.identifier, args.type))
        if args.type.lower() == "pmid":
            # Also run the new llinks test if a PMID is provided
            asyncio.run(run_new_llinks_test(args.identifier))
    else:
        print("No specific identifier provided, running test suite instead.")
        print("Usage: python tools/elink_pubmed.py --identifier <ID> --type <pmid|doi|pmc>")
        asyncio.run(run_test_suite()) # Runs XML prlinks tests
        # Add a specific test for the new llinks function
        print("\n--- Running Standalone Test for New get_article_links (llinks) ---")
        asyncio.run(run_new_llinks_test("38692467")) # Test with PMID from task
        asyncio.run(run_new_llinks_test("31345905")) # Another example
