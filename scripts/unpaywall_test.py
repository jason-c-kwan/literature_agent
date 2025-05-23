import requests
import json

def test_unpaywall(doi: str, email: str, timeout: float = 10.0):
    """
    Query Unpaywall for a given DOI and print the JSON response or error.
    """
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": email}
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()  # raises HTTPError for 4xx/5xx responses
        data = response.json()
        print(json.dumps(data, indent=2))
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error for DOI {doi}: {http_err} — {response.text}")
    except requests.exceptions.RequestException as err:
        print(f"Request failed for DOI {doi}: {err}")

if __name__ == "__main__":
    # Replace with a known OA DOI and your email
    test_unpaywall("10.1016/j.genrep.2025.102204", "jason.kwan@wisc.edu")
