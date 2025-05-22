import asyncio
import os
import sys
from typing import List, Dict, Any

# Add project root to sys.path to allow importing tools
# Assuming this script is run from the project root or its parent directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.search import search_literature
from tools.triage import TriageAgent

# Mock ChatCompletionClient for TriageAgent
# This is a simplified version, you might want to use the full MockChatCompletionClient
# from tests/test_triage.py for more comprehensive testing.
class MockChatCompletionClient:
    def __init__(self, response_content: str = "3"):
        self.response_content = response_content
        self._model_info_dict = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mock_family",
            "structured_output": False,
        }

    @property
    def model_info(self):
        return self._model_info_dict

    async def create(self, messages, **kwargs):
        # Simulate the structure AssistantAgent expects from its LLM call
        from autogen_core.models import CreateResult, RequestUsage
        return CreateResult(
            content=self.response_content,
            usage=RequestUsage(prompt_tokens=10, completion_tokens=1),
            thought=None,
            finish_reason="stop", # Added required field
            cached=False # Added required field
        )

    def create_stream(self, *args, **kwargs):
        pass
    async def close(self): pass
    def actual_usage(self): return None
    def total_usage(self): return None
    def count_tokens(self, messages, **kwargs): return 0
    def remaining_tokens(self, messages, **kwargs): return 1000
    @property
    def capabilities(self): return {"vision": False, "function_calling": False, "json_output": False}


async def main():
    # 1. Define a search query and user query for triage
    search_query = "CRISPR gene editing"
    user_triage_query = "highly relevant articles on CRISPR applications in human genetic diseases"

    print(f"Step 1: Searching literature for '{search_query}'...")
    # 2. Run search_literature to get a DataFrame
    # Ensure your .env file is configured with API_EMAIL and other API keys for search to work
    try:
        search_results_df = await asyncio.to_thread(search_literature, search_query, max_results_per_source=50)
        print(f"Search completed. Found {len(search_results_df)} articles.")
        if search_results_df.empty:
            print("No articles found. Please check your search query or API configurations.")
            return
        print("\nFirst 3 search results (DataFrame head):")
        print(search_results_df.head(3).to_string())
    except Exception as e:
        print(f"Error during literature search: {e}")
        print("Please ensure your .env file is correctly configured with API_EMAIL and other necessary API keys.")
        return

    # 3. Convert DataFrame to a list of dictionaries for TriageAgent
    articles_for_triage: List[Dict[str, Any]] = search_results_df.to_dict('records')
    print(f"\nStep 2: Preparing {len(articles_for_triage)} articles for triage.")

    # 4. Instantiate TriageAgent
    print("\nStep 3: Initializing TriageAgent with a mock LLM client...")
    mock_llm_client = MockChatCompletionClient(response_content="4") # Assume LLM always returns score 4
    triage_agent = TriageAgent(name="literature_triage_agent", model_client=mock_llm_client)
    # Relax triage settings for demonstration purposes
    triage_agent.sjr_threshold = 0 # Disable SJR filter
    triage_agent.norm_citation_threshold = 0.0 # Disable normalized citation filter
    triage_agent.oa_statuses_accepted = ['gold', 'green', 'closed', None] # Accept all OA statuses, including None

    # 5. Call triage_articles_async
    print(f"\nStep 4: Triaging articles with user query: '{user_triage_query}'...")
    triaged_articles = await triage_agent.triage_articles_async(articles_for_triage, user_triage_query)

    print(f"\nStep 5: Triage completed. {len(triaged_articles)} articles passed the triage filters.")

    if triaged_articles:
        print("\nTriaged Articles (first 3):")
        for i, article in enumerate(triaged_articles[:3]):
            print(f"--- Article {i+1} ---")
            print(f"Title: {article.get('title')}")
            print(f"Relevance Score: {article.get('relevance_score')}")
            print(f"DOI: {article.get('doi')}")
            print(f"Year: {article.get('year')}")
            print(f"OA Status: {article.get('oa_status')}")
            print("-" * 20)
    else:
        print("No articles passed the triage filters.")

if __name__ == "__main__":
    # Fix for ProactorEventLoop on Windows for asyncio
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
