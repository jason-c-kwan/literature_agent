import asyncio
import datetime
import os
from typing import List, Dict, Any, Optional

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient # For type hinting

# Determine settings file path relative to this file's location
# Assumes triage.py is in tools/ and settings.yaml is in config/ at the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SETTINGS_FILE_PATH = os.path.join(PROJECT_ROOT, 'config', 'settings.yaml')

class TriageAgent(AssistantAgent):
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        system_message: Optional[str] = "You are a biomedical literature triage assistant. Your task is to assess relevance. Respond with only a score from 1 to 5.",
        settings_path: str = SETTINGS_FILE_PATH,
        **kwargs, 
    ):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
            **kwargs 
        )
        self._load_settings(settings_path)
        # The system_message from agents.yaml will be used by AssistantAgent's LLM calls.
        # It should be generic, e.g.: "You are a scorer... Respond with score 1-5."
        # The specific query and article details will be in the user message.

    def _load_settings(self, settings_path: str):
        try:
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
            triage_settings = settings.get('triage', {})
            self.sjr_threshold = triage_settings.get('sjr_percentile_threshold', 50)
            self.norm_citation_threshold = triage_settings.get('min_age_normalized_citations', 1.0)
            self.oa_statuses_accepted = triage_settings.get('open_access_statuses', ['gold', 'hybrid', 'green'])
        except FileNotFoundError:
            print(f"Warning: Settings file not found at {settings_path}. Using default triage settings.")
            self.sjr_threshold = 50
            self.norm_citation_threshold = 1.0
            self.oa_statuses_accepted = ['gold', 'hybrid', 'green']
        except Exception as e:
            print(f"Warning: Error loading settings from {settings_path}: {e}. Using default triage settings.")
            self.sjr_threshold = 50
            self.norm_citation_threshold = 1.0
            self.oa_statuses_accepted = ['gold', 'hybrid', 'green']

    async def get_relevance_score(self, article_title: str, article_abstract: Optional[str], user_query: str) -> int:
        """
        Gets a relevance score from the LLM for a given article against a user query.
        The agent's system_message (set at initialization from agents.yaml) should instruct the LLM
        on its role and how to score. The prompt here provides the specific data.
        """
        prompt_content = (
            f"User Query: {user_query}\n\n"
            f"Title: {article_title}\n\n"
            f"Abstract: {article_abstract if article_abstract else 'N/A'}"
        )
        
        try:
            # Use the AssistantAgent's on_messages method to interact with the LLM.
            # This method uses the system_message configured during agent initialization.
            response_obj = await super().on_messages(
                messages=[TextMessage(content=prompt_content, source="user_data_provider")], 
                cancellation_token=CancellationToken()
            )
            
            # The response_obj.chat_message should contain the LLM's direct response.
            llm_response_content = response_obj.chat_message.content

            if isinstance(llm_response_content, str):
                score_str = llm_response_content.strip()
                if score_str.isdigit():
                    score = int(score_str)
                    if 1 <= score <= 5:
                        return score
                print(f"Warning: LLM returned invalid score format: '{score_str}' for title: '{article_title}' with query '{user_query}'")
            else:
                print(f"Warning: LLM response content is not a string: {type(llm_response_content)} for title: '{article_title}'")
        except Exception as e:
            print(f"Error getting relevance score for '{article_title}' with query '{user_query}': {e}")
        return 1 # Default to lowest score on error or invalid format

    async def triage_articles_async(self, articles: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
        """
        Performs relevance assessment and quality filtering on a list of articles.
        """
        triaged_articles: List[Dict[str, Any]] = []
        current_year = datetime.datetime.now().year

        for article_data in articles:
            # Make a copy to avoid modifying the original list of dicts if it's passed around
            article = article_data.copy()
            
            title = article.get('title', 'N/A')
            abstract = article.get('abstract') # May be None
            
            # 1. Get LLM Relevance Score
            relevance_score = await self.get_relevance_score(title, abstract, user_query)
            article['relevance_score'] = relevance_score

            # 2. Apply Filters
            # SJR Percentile Filter
            sjr = article.get('sjr_percentile')
            if sjr is not None:
                try:
                    if float(sjr) < self.sjr_threshold:
                        continue
                except ValueError:
                    print(f"Warning: Invalid SJR percentile value '{sjr}' for article '{title}'. Skipping SJR filter for this article.")


            # Open Access Filter
            oa_status = article.get('oa_status')
            accepted_oa_statuses = self.oa_statuses_accepted if isinstance(self.oa_statuses_accepted, list) else []
            # Only apply filter if oa_status is present and not in accepted list. If oa_status is None, article passes this filter.
            if oa_status is not None and oa_status not in accepted_oa_statuses:
                continue

            # Age-Normalized Citation Count Filter
            citations = article.get('citation_count')
            pub_year_val = article.get('year') or article.get('publication_year') # Prioritize 'year', fallback to 'publication_year'
            
            if citations is not None and pub_year_val is not None:
                try:
                    pub_year = int(pub_year_val)
                    citations_val = int(citations)
                    
                    age = current_year - pub_year + 1
                    if age <= 0: 
                        age = 1 # Avoid division by zero or negative age
                    
                    normalized_citations = float(citations_val) / age
                    if normalized_citations < self.norm_citation_threshold:
                        continue
                except ValueError:
                    print(f"Warning: Invalid citation count ('{citations}') or year ('{pub_year_val}') for article '{title}'. Skipping citation filter.")
                    pass # Changed from continue: if data is bad, skip this filter, don't drop article based on this error.
            elif citations is None or pub_year_val is None: # If data is missing for required calculation
                # Changed from continue: if data is missing, skip this filter, don't drop article based on this.
                pass 
                
            triaged_articles.append(article)

        return triaged_articles

# Note: For this TriageAgent to be correctly loaded and used by the AutoGen framework
# based on `agents.yaml`, the `system_message` in `agents.yaml` for the `triage`
# agent should be generic and not contain the `[Insert User Query Here]` placeholder.
# Example of a suitable system_message for agents.yaml:
# system_message: |
#   You are a biomedical literature triage assistant. Your task is to assess the relevance of scientific publications to a user query.
#   For each publication, you will be provided with the User Query, the Title, and the Abstract. Evaluate the relevance based on the following criteria:
#   - Presence of key terms related to the query.
#   - Study design and methodology.
#   - Population studied.
#   - Interventions and outcomes.
#   Assign a relevance score on a scale from 1 to 5:
#   1 - Not relevant
#   2 - Slightly relevant
#   3 - Moderately relevant
#   4 - Very relevant
#   5 - Highly relevant
#   Respond with only the relevance score.
