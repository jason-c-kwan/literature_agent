import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import os
from typing import Optional
from dotenv import load_dotenv
import argparse
import sys

# Load environment variables from .env file
load_dotenv()

API_EMAIL = os.getenv("API_EMAIL")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
TOOL_NAME = "literature_agent"  # As confirmed
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

async def fetch_pmc_xml(pmc_id: str, session: Optional[aiohttp.ClientSession] = None) -> str | None:
    """
    Retrieves the full-text XML for a given PMCID from the PubMed Central Open Access subset.

    Args:
        pmc_id: The PubMed Central ID (e.g., "PMC1234567").

    Returns:
        The XML string if successful and the article is in the OA subset, 
        None otherwise (e.g., network error, article not found, not in OA, or invalid XML).
    """
    if not API_EMAIL:
        print("Error: API_EMAIL is not configured in the .env file. Cannot make NCBI requests.")
        # Depending on strictness, could raise ValueError("API_EMAIL not configured")
        return None

    params = {
        "db": "pmc",
        "rettype": "full",
        "retmode": "xml",
        "id": pmc_id,
        "tool": TOOL_NAME,
        "email": API_EMAIL
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    provided_session = bool(session)
    if not session:
        session = aiohttp.ClientSession()

    try:
        # print(f"Fetching PMCID: {pmc_id} with params: {params}") # For debugging
        async with session.get(BASE_URL, params=params) as response:
            response.raise_for_status()  # Raises an AIOHTTP error for bad status codes (4XX or 5XX)
            xml_text = await response.text()

            # ---- XML Parsing Logic (to be refined with actual XML examples) ----
            # This section needs to be robustly tested with examples of:
            # 1. Successful OA full-text XML.
            # 2. Response for articles valid but not in OA subset.
            # 3. Response for invalid PMCIDs.
            try:
                root = ET.fromstring(xml_text)

                # Check for NCBI's common error structure within a successful HTTP response
                # (e.g., <eFetchResult><ERROR>Record not found</ERROR></eFetchResult>)
                if root.tag == "eFetchResult":
                    error_node = root.find("ERROR")
                    if error_node is not None:
                        # NCBI returned an error XML (e.g., ID not found, etc.)
                        # print(f"NCBI API Error for PMCID {pmc_id}: {error_node.text}")
                        return None
                    else:
                        # It's an eFetchResult but not an error, could be an empty set or unexpected
                        # print(f"Unexpected eFetchResult structure for PMCID {pmc_id}")
                        return None


                # Check for expected root tags of valid PMC OA articles
                # Common tags are <pmc-articleset> (for multiple articles, though unlikely for single ID fetch)
                # or <article> (for a single article).
                if root.tag in ["pmc-articleset", "article"]:
                    # Further validation could be added here if needed, e.g., checking for specific child elements
                    # that confirm it's a full-text article.
                    return xml_text
                
                # If the root tag is something else, it's unexpected for a successful OA fetch.
                # print(f"Unexpected XML root tag '{root.tag}' for PMCID {pmc_id}. Not a recognized OA article format.")
                return None

            except ET.ParseError:
                # The returned text was not valid XML.
                # This could happen if NCBI returns plain text error or malformed XML.
                # print(f"Failed to parse XML for PMCID {pmc_id}. Response text: {xml_text[:200]}...") # Log snippet
                return None

    except aiohttp.ClientResponseError as e:
        # HTTP error (4xx or 5xx)
        # print(f"HTTP error {e.status} for PMCID {pmc_id}: {e.message}")
        return None
    except aiohttp.ClientError as e:
        # Other AIOHTTP client errors (e.g., connection issues)
        # print(f"AIOHTTP client error for PMCID {pmc_id}: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during the process
        # print(f"An unexpected error occurred while fetching PMCID {pmc_id}: {e}")
        return None
    finally:
        if not provided_session and session:
            await session.close()


# async def main():
#     parser = argparse.ArgumentParser(description="Fetch full-text XML for a PMCID from PubMed Central OA subset.")
#     parser.add_argument("pmcid", nargs='?', help="The PMCID to fetch (e.g., PMC1234567). If not provided, runs predefined test cases.")
#     args = parser.parse_args()

#     if args.pmcid:
#         # Informative message to stderr, so it doesn't mix with XML output on stdout
#         print(f"Attempting to fetch PMCID: {args.pmcid}", file=sys.stderr) 
#         xml_data = await fetch_pmc_xml(args.pmcid)
#         if xml_data:
#             print(xml_data) # Output XML to stdout
#         else:
#             print(f"Failed to fetch XML for {args.pmcid}, it might not be in OA subset, or an error occurred.", file=sys.stderr) # Print errors to stderr
#     else:
#         # Original test cases if no PMCID is provided
#         print("No PMCID provided, running predefined test cases...\n", file=sys.stderr)
#         # Replace with PMCID known to be in OA subset for testing
#         test_pmc_id_oa = "PMC3539239"  # Example: a known OA article
#         # Replace with PMCID known NOT to be in OA or invalid
#         test_pmc_id_not_oa = "PMC123" # Example: likely not OA or invalid
#         test_pmc_id_invalid = "PMCINVALID" # Example: invalid format

#         print(f"Attempting to fetch OA article: {test_pmc_id_oa}")
#         xml_data_oa = await fetch_pmc_xml(test_pmc_id_oa)
#         if xml_data_oa:
#             print(f"Successfully fetched XML for {test_pmc_id_oa}. Length: {len(xml_data_oa)}")
#         else:
#             print(f"Failed to fetch XML for {test_pmc_id_oa} or it's not in OA subset.")

#         print(f"\nAttempting to fetch non-OA/invalid article: {test_pmc_id_not_oa}")
#         xml_data_not_oa = await fetch_pmc_xml(test_pmc_id_not_oa)
#         if xml_data_not_oa:
#             print(f"Successfully fetched XML for {test_pmc_id_not_oa}. Length: {len(xml_data_not_oa)}")
#         else:
#             print(f"Failed to fetch XML for {test_pmc_id_not_oa} as expected (or error).")

#         print(f"\nAttempting to fetch invalid PMCID: {test_pmc_id_invalid}")
#         xml_data_invalid = await fetch_pmc_xml(test_pmc_id_invalid)
#         if xml_data_invalid:
#             print(f"Successfully fetched XML for {test_pmc_id_invalid}. Length: {len(xml_data_invalid)}")
#         else:
#             print(f"Failed to fetch XML for {test_pmc_id_invalid} as expected (or error).")


# if __name__ == '__main__':
#     # Example usage (requires running in an async context)
#     asyncio.run(main())
