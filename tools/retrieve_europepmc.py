import httpx
from typing import Optional, Dict, Any
from urllib.parse import quote as url_quote
import xmltodict # Import xmltodict

async def fetch_europepmc(doi: str) -> Optional[Dict[str, Any]]:
    """
    Query Europe PMC REST API for full-text JSON by DOI.

    Args:
        doi: The Digital Object Identifier (DOI) of the article.

    Returns:
        A dictionary containing the full-text JSON if found and successfully retrieved.
        Returns None if the article metadata is not found, no suitable full-text URL is available,
        or if fetching/parsing the full-text fails.
        Raises httpx.HTTPStatusError for unhandled HTTP errors from the initial metadata request.
    """
    encoded_doi = url_quote(doi)
    metadata_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:\"{encoded_doi}\"&format=json&resulttype=core"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        article_metadata: Optional[Dict[str, Any]] = None
        try:
            # Step 1: Fetch article metadata
            response_metadata = await client.get(metadata_url)
            
            if response_metadata.status_code == 200:
                data = response_metadata.json()
                if data.get("resultList") and data["resultList"].get("result"):
                    article_metadata = data["resultList"]["result"][0]
                else:
                    return None # DOI metadata not found
            elif response_metadata.status_code == 404:
                return None
            else:
                response_metadata.raise_for_status() # Raise for other metadata fetch errors
                return None # Should be unreachable if raise_for_status works

        except httpx.HTTPStatusError: # Errors from metadata fetch
            raise
        except httpx.RequestError as e:
            print(f"HTTPX RequestError during metadata fetch: {e}")
            return None
        except (KeyError, IndexError, TypeError, ValueError) as e: # ValueError for json decode issues
            print(f"Error parsing metadata response: {e}")
            return None

        if not article_metadata:
            return None

        # Step 2: Find and fetch full-text JSON from metadata
        full_text_json_url: Optional[str] = None
        
        # Priority 1: If PMCID is available, try the direct /PMCID/fullTextXML?format=json endpoint
        pmcid = article_metadata.get("pmcid")
        if pmcid:
            # Ensure PMCID doesn't already have "PMC" prefix, as the API might expect just the number for the path
            # However, typical PMCID format is "PMC1234567"
            # Let's assume the pmcid field from metadata is the correct identifier for the path
            # The base URL for this specific endpoint structure:
            full_text_json_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML?format=json"
            print(f"Attempting full-text retrieval using PMCID-based URL: {full_text_json_url}")
            # If this attempt fails (e.g. 404), we might want to fall back to fullTextUrlList,
            # but for now, let's try this prioritized approach.

        # Fallback or Alternative: Check fullTextUrlList if PMCID method isn't chosen or fails
        # For simplicity in this iteration, if PMCID exists, we use it. Otherwise, check list.
        if not full_text_json_url and article_metadata.get("fullTextUrlList"):
            for ft_url_info in article_metadata["fullTextUrlList"].get("fullTextUrl", []):
                url_candidate = ft_url_info.get("url")
                doc_style = ft_url_info.get("documentStyle")

                if doc_style == "json" and url_candidate:
                    full_text_json_url = url_candidate
                    print(f"Found direct JSON URL in fullTextUrlList: {full_text_json_url}")
                    break
                elif doc_style == "xml" and url_candidate:
                    # If it's an XML URL, try appending ?format=json
                    if "?" in url_candidate:
                        full_text_json_url = f"{url_candidate}&format=json"
                    else:
                        full_text_json_url = f"{url_candidate}?format=json"
                    print(f"Found XML URL in fullTextUrlList, attempting with ?format=json: {full_text_json_url}")
                    break 
                # Could add more heuristics here if needed, e.g. for 'html' or 'pdf' if we had a parser
        
        if not full_text_json_url:
            print(f"No suitable full-text JSON URL identified for DOI: {doi} (PMCID: {pmcid})")
            return None

        try:
            # Step 3: Fetch the actual full-text content
            response_full_text = await client.get(full_text_json_url) # This URL might already have ?format=json
            
            if response_full_text.status_code == 200:
                try:
                    # Attempt to parse as JSON first
                    return response_full_text.json()
                except ValueError:
                    # If JSON parsing fails, try to parse as XML if the original URL didn't specify format=json
                    # or if format=json still returned XML (which can happen)
                    print(f"Failed to parse as JSON from {full_text_json_url}, attempting XML parsing.")
                    try:
                        xml_content = response_full_text.text
                        parsed_xml = xmltodict.parse(xml_content)
                        # Potentially, we might want to check if the root of parsed_xml is <error>
                        # For now, assume successful parse means valid content.
                        # The original request was for "full-text JSON", so xmltodict provides a JSON-like dict.
                        return parsed_xml 
                    except Exception as xml_e:
                        print(f"Failed to parse as XML from {full_text_json_url}: {xml_e}")
                        return None
            elif response_full_text.status_code == 404:
                print(f"Full-text content not found at {full_text_json_url} (404)")
                return None
            else:
                print(f"Failed to fetch full-text content from {full_text_json_url}. Status: {response_full_text.status_code}")
                return None
        except httpx.RequestError as e:
            print(f"HTTPX RequestError during full-text fetch from {full_text_json_url}: {e}")
            return None
        # ValueError from response_full_text.json() is now handled above.
        # Other ValueErrors (e.g. from xmltodict) are also handled.
        except Exception as e: # Catch-all for other unexpected errors during full-text fetch
            print(f"Unexpected error fetching/parsing full text from {full_text_json_url}: {e}")
            return None

if __name__ == "__main__":
    import asyncio
    import argparse
    import json

    async def main_cli():
        parser = argparse.ArgumentParser(description="Fetch full-text JSON from Europe PMC by DOI.")
        parser.add_argument("doi", type=str, help="The DOI to fetch.")
        args = parser.parse_args()

        print(f"Fetching full-text JSON for DOI: {args.doi}")
        try:
            result = await fetch_europepmc(args.doi)
            if result:
                print("Success! Full-text JSON response:")
                print(json.dumps(result, indent=2))
            else:
                print("Full-text JSON not found or an error occurred.")
        except httpx.HTTPStatusError as e: # This would be from the metadata call
            print(f"An HTTP Status Error occurred during metadata fetch: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                try:
                    print(f"Response content: {e.response.text}")
                except Exception:
                    pass
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    asyncio.run(main_cli())
