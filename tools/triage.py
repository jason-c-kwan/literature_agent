import asyncio
import datetime
import os
from typing import List, Dict, Any, Optional
import json # Added import

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
        system_message: Optional[str] = "You are a biomedical literature triage assistant. Your task is to assess relevance based on a query summary and respond with a JSON object of scores.", # Updated default
        settings_path: str = SETTINGS_FILE_PATH,
        **kwargs, 
    ):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message, # This will be overridden by agents.yaml if specified there
            **kwargs 
        )
        self._load_settings(settings_path)
        # The detailed system_message from agents.yaml (instructing JSON output etc.) will be used.

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

    async def get_detailed_relevance_scores(self, article_title: str, article_abstract: Optional[str], query_summary: dict) -> Dict[str, Optional[int]]:
        """
        Gets detailed relevance scores from the LLM for a given article against a query_summary.
        The agent's system_message (from agents.yaml) instructs the LLM on its role and JSON output format.
        """
        default_score_keys = ["research_focus", "model_preferences", "must_include", "exclusions", "time_window", "requested_outputs"]
        error_scores = {key: None for key in default_score_keys} # Return None for all categories on error

        try:
            query_summary_str = json.dumps(query_summary, indent=2)
        except TypeError:
            print(f"Error: Could not serialize query_summary to JSON for article '{article_title}'.")
            return error_scores
            
        prompt_content = (
            f"Query Summary:\n```json\n{query_summary_str}\n```\n\n"
            f"Publication Title: {article_title}\n\n"
            f"Publication Abstract: {article_abstract if article_abstract else 'N/A'}\n\n"
            f"Provide your category scores as a single JSON object based on the instructions in the system message."
        )
        
        try:
            response_obj = await super().on_messages(
                messages=[TextMessage(content=prompt_content, source="user_data_provider")],
                cancellation_token=CancellationToken()
            )
            
            llm_response_content = response_obj.chat_message.content

            if isinstance(llm_response_content, str):
                try:
                    # Attempt to parse the entire response as JSON
                    # The system prompt for triage agent now demands the *entire* response be JSON.
                    scores_dict = json.loads(llm_response_content.strip())
                    
                    # Validate structure and values
                    validated_scores = {}
                    valid_structure = True
                    for key in default_score_keys: # Ensure all expected keys are present or handled
                        if key in scores_dict:
                            value = scores_dict[key]
                            if value is None: # Explicit null for broad categories is fine
                                validated_scores[key] = None
                            elif isinstance(value, int) and 1 <= value <= 5:
                                validated_scores[key] = value
                            else: # Invalid score value for a key
                                print(f"Warning: LLM returned invalid score value '{value}' for category '{key}' in article '{article_title}'. Setting to None.")
                                validated_scores[key] = None # Or handle as error, e.g. score of 1
                                valid_structure = False # Or just for this key
                        else: # Key missing from LLM response
                            print(f"Warning: LLM response missing score for category '{key}' in article '{article_title}'. Setting to None.")
                            validated_scores[key] = None
                            valid_structure = False
                    
                    # Ensure all expected keys are in the output, even if they were missing from LLM
                    for key in default_score_keys:
                        if key not in validated_scores:
                             validated_scores[key] = None

                    return validated_scores

                except json.JSONDecodeError:
                    print(f"Warning: LLM response was not valid JSON: '{llm_response_content.strip()}' for article '{article_title}'.")
                    return error_scores # Return dict with all Nones
            else:
                print(f"Warning: LLM response content is not a string: {type(llm_response_content)} for article '{article_title}'.")
                return error_scores
        except Exception as e:
            print(f"Error getting detailed relevance scores for '{article_title}': {e}")
            return error_scores

    async def triage_articles_async(self, articles: List[Dict[str, Any]], query_summary: Optional[dict]) -> List[Dict[str, Any]]:
        """
        Performs relevance assessment based on query_summary and calculates average scores.
        """
        triaged_articles_output: List[Dict[str, Any]] = []
        # current_year = datetime.datetime.now().year # Keep if needed for other filters, but primary filters removed

        if query_summary is None:
            print("Warning: No query_summary provided to triage_articles_async. Articles will not be scored by LLM.")
            for article_data in articles:
                article = article_data.copy()
                article['detailed_relevance_scores'] = {key: None for key in ["research_focus", "model_preferences", "must_include", "exclusions", "time_window", "requested_outputs"]}
                article['average_relevance_score'] = None
                # Decide if articles should still be added to output if no summary, or just return empty.
                # For now, let's add them with None scores, filtering will happen in cli/litsearch.py
                triaged_articles_output.append(article)
            return triaged_articles_output

        for article_data in articles:
            article = article_data.copy()
            title = article.get('title', 'N/A')
            abstract = article.get('abstract')

            detailed_scores = await self.get_detailed_relevance_scores(title, abstract, query_summary)
            article['detailed_relevance_scores'] = detailed_scores

            valid_numerical_scores = []
            if isinstance(detailed_scores, dict):
                for score_value in detailed_scores.values():
                    if isinstance(score_value, (int, float)): # Check if it's a number
                        valid_numerical_scores.append(score_value)
            
            if valid_numerical_scores:
                article['average_relevance_score'] = sum(valid_numerical_scores) / len(valid_numerical_scores)
            else:
                article['average_relevance_score'] = None # No valid numerical scores to average

            # Old filtering logic (SJR, OA, Citations) is removed as per plan.
            # The new filtering will be based on average_relevance_score and detailed_relevance_scores
            # and will be handled in cli/litsearch.py after this method returns.
            
            triaged_articles_output.append(article)

        return triaged_articles_output

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
