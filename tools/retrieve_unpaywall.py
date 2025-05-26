import aiohttp
import asyncio
import sys # Added for command-line arguments
from typing import Dict, Optional, Any

# Consider loading from a config or environment variable for flexibility
UNPAYWALL_API_EMAIL = "jason.kwan@wisc.edu" # TODO: Load from .env or config

async def get_unpaywall_oa_url(
    doi: str, 
    email: str = UNPAYWALL_API_EMAIL, 
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    Fetches the Unpaywall record for a given DOI and returns the best OA URL found.
    Priority:
    1. best_oa_location.url_for_pdf
    2. best_oa_location.url
    3. best_oa_location.url_for_landing_page
    4. Fallback to iterating oa_locations with the same URL priority.
    """
    if not email:
        print("Error: Unpaywall email not configured.")
        return None

    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    record: Optional[Dict[str, Any]] = None

    provided_session = bool(session)
    if not session:
        session = aiohttp.ClientSession()

    try:
        async with session.get(api_url) as response:
            response.raise_for_status()
            if response.content_type == 'application/json':
                record = await response.json()
            else:
                print(f"Error: Unexpected content type from Unpaywall API: {response.content_type} for DOI {doi}")
                return None
    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error {e.status} fetching Unpaywall record for DOI {doi}: {e.message}")
        return None
    except aiohttp.ClientError as e: # Includes connection errors, timeouts
        print(f"ClientError fetching Unpaywall record for DOI {doi}: {e}")
        return None
    except Exception as e: # Catch-all for other unexpected errors
        print(f"An unexpected error occurred while fetching Unpaywall record for DOI {doi}: {e}")
        return None
    finally:
        if not provided_session and session:
            await session.close()
    
    if not record:
        return None

    # Try to get URL from best_oa_location
    best_oa_location = record.get("best_oa_location")
    if isinstance(best_oa_location, dict):
        url = best_oa_location.get("url_for_pdf")
        if url: return url
        url = best_oa_location.get("url")
        if url: return url
        url = best_oa_location.get("url_for_landing_page")
        if url: return url
        
    # Fallback: Try to get URL from the list of oa_locations
    oa_locations = record.get("oa_locations", [])
    if isinstance(oa_locations, list):
        # Prioritize url_for_pdf across all locations
        for location in oa_locations:
            if isinstance(location, dict):
                url = location.get("url_for_pdf")
                if url: return url
        # Then, prioritize url across all locations
        for location in oa_locations:
            if isinstance(location, dict):
                url = location.get("url")
                if url: return url
        # Finally, try url_for_landing_page across all locations
        for location in oa_locations:
            if isinstance(location, dict):
                url = location.get("url_for_landing_page")
                if url: return url
                
    # If no URL found after all checks
    print(f"No suitable OA URL found for DOI {doi} in Unpaywall record.")
    return None

# Example usage (optional, for testing)
async def main():
    if len(sys.argv) > 1:
        doi_to_test = sys.argv[1]
        print(f"\n--- Testing DOI: {doi_to_test} ---")
        oa_url = await get_unpaywall_oa_url(doi_to_test)
        if oa_url:
            print(f"Best OA URL found: {oa_url}")
        else:
            print(f"No OA URL retrieved for DOI: {doi_to_test}")
    else:
        print("Usage: python tools/retrieve_unpaywall.py <DOI>")
        print("\n--- Running with example DOIs ---")
        doi_examples = ["10.1038/nature12373", "10.1103/physrevlett.98.030801", "10.7554/eLife.01567", "10.1000/nonexistentdoi"]
        for doi_example in doi_examples:
            print(f"\n--- Testing DOI: {doi_example} ---")
            oa_url = await get_unpaywall_oa_url(doi_example)
            if oa_url:
                print(f"Best OA URL found: {oa_url}")
            else:
                print(f"No OA URL retrieved for DOI: {doi_example}")

if __name__ == "__main__":
    asyncio.run(main())
