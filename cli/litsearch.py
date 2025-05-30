import os
import yaml
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo, LLMMessage, CreateResult, SystemMessage as SystemMessageFromCore
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.messages import StopMessage, TextMessage, ToolCallSummaryMessage, BaseChatMessage
from tools.search import LiteratureSearchTool, SearchLiteratureParams
from typing import Sequence, Any, Optional, List, Union, Type, Dict, Tuple
import pandas as pd # Added for Timestamp
from pydantic import BaseModel
from tools.triage import TriageAgent
from tools.ranking import RankerAgent
from tools.export import ExporterAgent
from tools.retrieve_full_text import FullTextRetrievalAgent
from tools._base import StubConfig
import asyncio
import re
import uuid
from rich.console import Console
from rich.theme import Theme
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

from autogen_agentchat.teams import BaseGroupChat
from autogen_agentchat.conditions import FunctionalTermination
from autogen_agentchat.ui import Console as AgentChatConsole

from autogen_core.tools import FunctionTool
from tools.triage import TriageAgent

REQUIRED_METADATA_FIELDS = [
    "purpose", "scope", "audience", "article_type",
    "date_range", "open_access", "output_format"
]

class QueryRefinerJsonTermination(TerminationCondition):
    """Terminate when the QueryRefinerAgent outputs a valid JSON block
    containing refined queries."""
    def __init__(self, query_refiner_agent_name: str):
        self._query_refiner_agent_name = query_refiner_agent_name
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    def _is_valid_query_summary(self, summary: Any) -> bool:
        if not isinstance(summary, dict):
            return False
        expected_keys = {
            "research_focus": str,
            "model_preferences": list,
            "must_include": list,
            "exclusions": list,
            "time_window": str,
            "requested_outputs": str
        }
        for key, expected_type in expected_keys.items():
            if key not in summary:
                return False
            if not isinstance(summary[key], expected_type):
                return False
            if expected_type == list and not all(isinstance(item, str) for item in summary[key]):
                # Allow empty lists, but if not empty, items must be strings
                if summary[key] and not all(isinstance(item, str) for item in summary[key]):
                    return False
        return True

    async def __call__(self, messages: Sequence[BaseChatMessage]) -> Optional[StopMessage]:
        if self._terminated:
            return None

        last_message = messages[-1] if messages else None
        if last_message and last_message.source == self._query_refiner_agent_name and isinstance(last_message.content, str):
            content = last_message.content
            match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
            if match:
                json_str = match.group(1)
                try:
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, list) and len(parsed_json) > 0:
                        all_valid = True
                        for item in parsed_json:
                            if not (
                                isinstance(item, dict) and
                                "pubmed_query" in item and isinstance(item["pubmed_query"], str) and
                                "general_query" in item and isinstance(item["general_query"], str) and
                                "article_type" in item and isinstance(item["article_type"], list) and
                                all(isinstance(at, str) for at in item["article_type"]) and
                                "date_range" in item and isinstance(item["date_range"], str) and
                                "query_summary" in item and self._is_valid_query_summary(item["query_summary"])
                            ):
                                all_valid = False
                                break
                        if all_valid:
                            self._terminated = True
                            return StopMessage(content="QueryRefinerAgent produced valid JSON output with query_summary.", source=self._query_refiner_agent_name)
                except json.JSONDecodeError:
                    pass # Will be handled by the agent if it's not valid JSON
        return None

    async def reset(self) -> None:
        self._terminated = False

def _is_valid_query_summary_for_extraction(summary: Any) -> bool:
    # Simplified check for extraction, main validation in Termination class
    if not isinstance(summary, dict): return False
    expected_keys = ["research_focus", "model_preferences", "must_include", "exclusions", "time_window", "requested_outputs"]
    return all(key in summary for key in expected_keys)


def extract_queries(text: str) -> list[dict]:
    """
    Parses a fenced JSON block from the input text and returns a list of query dictionaries.
    Each dictionary is expected to have 'pubmed_query', 'general_query', 'article_type', 
    'date_range', and 'query_summary'.
    Returns an empty list if parsing fails or the structure is incorrect.
    """
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if not match:
        return []
    
    json_str = match.group(1)
    try:
        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, list):
            valid_queries = []
            for item in parsed_json:
                if isinstance(item, dict) and \
                   "pubmed_query" in item and isinstance(item["pubmed_query"], str) and \
                   "general_query" in item and isinstance(item["general_query"], str) and \
                   "article_type" in item and isinstance(item["article_type"], list) and \
                   all(isinstance(at, str) for at in item["article_type"]) and \
                   "date_range" in item and isinstance(item["date_range"], str) and \
                   "query_summary" in item and _is_valid_query_summary_for_extraction(item["query_summary"]):
                    valid_queries.append({
                        "pubmed_query": item["pubmed_query"],
                        "general_query": item["general_query"],
                        "article_type": item["article_type"],
                        "date_range": item["date_range"],
                        "query_summary": item["query_summary"]
                    })
                else:
                    # If any item doesn't match the full structure, consider the whole JSON invalid for this function's purpose.
                    console.print(f"[yellow]Warning: Invalid item structure in JSON query block (extract_queries): {item}[/yellow]")
                    return [] 
            return valid_queries
        return []
    except json.JSONDecodeError:
        console.print(f"[yellow]Warning: JSONDecodeError while parsing query block (extract_queries).[/yellow]")
        return []

def parse_chat_history_for_metadata(
    chat_history: Sequence[BaseChatMessage],
    field_question_map: Dict[str, str],
    user_proxy_name: str,
    query_refiner_name: str
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if not chat_history:
        return metadata
    for i, msg in enumerate(chat_history):
        if msg.source == query_refiner_name and isinstance(msg.content, str):
            question_text = msg.content.strip()
            asked_field = None
            for field, prompt_text in field_question_map.items():
                if prompt_text.lower() in question_text.lower() or field.lower() in question_text.lower():
                    asked_field = field
                    break
            if asked_field and (i + 1) < len(chat_history):
                answer_msg = chat_history[i+1]
                if answer_msg.source == user_proxy_name and isinstance(answer_msg.content, str):
                    metadata[asked_field] = answer_msg.content.strip()
    return metadata

def convert_metadata_to_search_params(
    collected_metadata: Dict[str, Any],
    original_query: str,
    console: Console,
    refined_query_pair: Optional[Dict[str, str]] = None
) -> Optional[SearchLiteratureParams]:
    pubmed_query_to_use = ""
    general_query_to_use = ""

    if refined_query_pair:
        pubmed_query_to_use = refined_query_pair.get("pubmed_query", "").strip()
        general_query_to_use = refined_query_pair.get("general_query", "").strip()
        console.print(f"[info]Using refined PubMed query: '{pubmed_query_to_use}'[/info]")
        console.print(f"[info]Using refined general query: '{general_query_to_use}'[/info]")
    
    if not pubmed_query_to_use and not general_query_to_use:
        console.print(f"[yellow]No valid refined query pair provided, falling back to original query: '{original_query}'[/yellow]")
        general_query_to_use = original_query
        temp_original_keywords = re.sub(r'\b(and|or|not|the|a|of|in|for|to|with)\b', '', original_query, flags=re.IGNORECASE)
        pubmed_query_to_use = ' '.join(temp_original_keywords.split())
        console.print(f"[info]Using keywords from original query for PubMed (fallback): '{pubmed_query_to_use}'[/info]")

    if not pubmed_query_to_use and not general_query_to_use:
        console.print("[red]Cannot form search query: No refined pair and no original query usable.[/red]")
        return None

    search_param_args: Dict[str, Any] = {
        "pubmed_query": pubmed_query_to_use,
        "general_query": general_query_to_use
    }
    return SearchLiteratureParams(**search_param_args)

class DebugOpenAIChatCompletionClient(OpenAIChatCompletionClient):
    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Optional[List[Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        json_output: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> CreateResult:
        print("\n" + "="*80)
        print("DEBUG: LLM Messages Payload to model_client.create():")
        print(f"Number of messages: {len(messages)}")
        tools_arg = tools
        print(f"Tools argument received by DebugClient.create: {'Yes, content:' + str(tools_arg) if tools_arg is not None else 'No (None)'}")
        if tools_arg:
            print(f"  Type of tools_arg: {type(tools_arg)}")
            if isinstance(tools_arg, list):
                for tool_idx, tool_def_item in enumerate(tools_arg):
                    tool_name_to_print = f"Unknown (type: {type(tool_def_item)})"
                    if hasattr(tool_def_item, 'name') and isinstance(getattr(tool_def_item, 'name'), str):
                        tool_name_to_print = getattr(tool_def_item, 'name')
                    elif isinstance(tool_def_item, dict) and tool_def_item.get("type") == "function":
                        func_dict = tool_def_item.get("function")
                        if isinstance(func_dict, dict):
                            tool_name_to_print = func_dict.get("name", "Name N/A in func dict")
                    print(f"  Tool {tool_idx}: {tool_name_to_print}")
            else:
                print(f"  Tools argument is not a list: {tools_arg}")
        print(f"JSON output mode: {'Yes, type: ' + str(json_output) if json_output else 'No'}")
        print("-" * 80)
        for i, msg in enumerate(messages):
            role_to_print = "unknown"
            if isinstance(msg, SystemMessageFromCore): role_to_print = "system"
            elif hasattr(msg, "role") and msg.role is not None: role_to_print = msg.role
            print(f"Message {i}: Role: {role_to_print}")
            print(f"  Content type: {type(msg.content)}")
            if isinstance(msg.content, str):
                try:
                    parsed_json = json.loads(msg.content)
                    print("  Content (parsed as JSON):")
                    print(json.dumps(parsed_json, indent=2))
                except json.JSONDecodeError:
                    print("  Content (string):")
                    print(msg.content)
            elif isinstance(msg.content, list):
                print("  Content (list of parts):")
                for part_idx, part in enumerate(msg.content):
                    print(f"    Part {part_idx}: Type: {type(part)}")
                    if isinstance(part, dict): print(f"      {json.dumps(part, indent=2)}")
                    else: print(f"      {part!r}")
            else: print(f"  Content: {msg.content!r}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls: print(f"  Tool Calls: {msg.tool_calls!r}")
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id: print(f"  Tool Call ID: {msg.tool_call_id}")
        print("="*80 + "\n")
        await asyncio.sleep(0.1)
        try:
            effective_tools_for_super = tools if tools is not None else []
            return await super().create(messages, tools=effective_tools_for_super, cancellation_token=cancellation_token, json_output=json_output, **kwargs)
        except Exception as e:
            print(f"DEBUG: Error during model_client.create AFTER printing payload: {type(e).__name__} - {e}")
            raise

def resolve_env_placeholder(value: str) -> str:
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{(.+?)(?::-([^}]+))?\}", value)
        if match:
            var_name = match.group(1)
            default_value = match.group(2)
            env_var = os.getenv(var_name)
            if env_var is not None: return env_var
            elif default_value is not None: return default_value
            else:
                print(f"Warning: Environment variable {var_name} not set and no default provided for '{value}'")
                return None
    return value

def load_rich_theme(base_path: Path) -> Theme:
    theme_file_path = base_path / "config" / "rich_theme.json"
    if theme_file_path.exists():
        with open(theme_file_path, "r") as f:
            theme_config = json.load(f)
        return Theme(theme_config)
    return Theme({})

def styled_input(prompt_message: str, console: Console) -> str:
    plain_prompt = re.sub(r'\[/?[^\]]*\]', '', prompt_message)
    return prompt(f"{plain_prompt} ").strip()

async def run_search_pipeline(
    query_team: RoundRobinGroupChat,
    literature_search_tool: LiteratureSearchTool,
    triage_agent: TriageAgent,
    agents: Dict[str, Any],
    original_query: str,
    console: Console,
    settings_config: Dict[str, Any],
    cli_args: argparse.Namespace,
    query_refiner_config_params: Dict[str, Any],
    fields_to_collect_override: Optional[List[str]] = None # Not currently used with autonomous flow
):
    all_articles_data = []
    processed_dois_for_articles = set()
    all_refined_queries_for_logging = []

    query_refiner_agent = agents.get("query_refiner")
    user_proxy_agent = agents.get("user_proxy")

    if not query_refiner_agent or not user_proxy_agent:
        console.print("[red]Error: Query Refiner or User Proxy agent not found.[/red]")
        return {"query": original_query, "refined_queries": [], "triaged_articles": []}

    field_question_map = query_refiner_config_params.get('required_fields', {})
    
    console.print(f"[secondary]Starting autonomous query refinement for:[/secondary] [highlight]'{original_query}'[/highlight]")
    initial_task_message = TextMessage(content=original_query, source=user_proxy_agent.name)
    chat_result = None
    try:
        console.print(f"[info]Initiating chat with query_team. User's query: '{original_query}'[/info]")
        chat_result = await AgentChatConsole(query_team.run_stream(task=initial_task_message))
    except Exception as e_chat:
        console.print(f"[red]Error during query_team.run_stream: {e_chat}[/red]")

    collected_metadata: Dict[str, Any] = {}
    extracted_query_pairs: List[Dict[str, str]] = []
    last_refiner_msg_content: Optional[str] = None

    if chat_result and chat_result.messages:
        console.print(f"[info]Query refinement chat completed. Processing {len(chat_result.messages)} messages.[/info]")
        actual_chat_messages = [m for m in chat_result.messages if isinstance(m, BaseChatMessage)]
        collected_metadata = parse_chat_history_for_metadata(
            actual_chat_messages, field_question_map, user_proxy_agent.name, query_refiner_agent.name
        )
        console.print(f"[info]Metadata collected during refinement (primarily for non-query fields): {collected_metadata}[/info]")
        for msg in reversed(actual_chat_messages):
            if msg.source == query_refiner_agent.name and isinstance(msg.content, str):
                last_refiner_msg_content = msg.content
                break
        
        if last_refiner_msg_content:
            extracted_query_pairs = extract_queries(last_refiner_msg_content) # This now extracts all four fields
            if extracted_query_pairs:
                console.print(f"[success]Successfully extracted {len(extracted_query_pairs)} refined query objects.[/success]")
                for i, pair in enumerate(extracted_query_pairs):
                    console.print(f"  Pair {i+1}: PubMed='{pair['pubmed_query']}', General='{pair['general_query']}', ArticleTypes={pair['article_type']}, DateRange='{pair['date_range']}', QuerySummary={json.dumps(pair.get('query_summary'))}")
                    all_refined_queries_for_logging.append({
                        "pubmed_query": pair['pubmed_query'],
                        "general_query": pair['general_query'],
                        "article_type": pair['article_type'],
                        "date_range": pair['date_range'],
                        "query_summary": pair.get('query_summary') 
                    })
            else:
                console.print(f"[red]Failed to extract valid JSON query objects from QueryRefinerAgent's last message. Content: {last_refiner_msg_content[:500]}...[/red]")
        else:
            console.print(f"[red]Could not find a final message from QueryRefinerAgent.[/red]")

    # This will store lists of article dicts, one list per search_output
    search_results_per_query_pair: List[List[Dict[str, Any]]] = []


    # Determine if fallback to original query is needed
    use_fallback_query = not extracted_query_pairs

    if use_fallback_query:
        console.print("[yellow]No valid refined queries extracted or refinement failed. Falling back to original query and collected metadata.[/yellow]")
        fallback_search_params = convert_metadata_to_search_params(
            collected_metadata, original_query, console, refined_query_pair=None
        )
        if fallback_search_params:
            # Treat the fallback as a single query pair for the loop
            extracted_query_pairs = [{
                "pubmed_query": fallback_search_params.pubmed_query,
                "general_query": fallback_search_params.general_query
            }]
            all_refined_queries_for_logging.append(f"FALLBACK USED PubMed: {fallback_search_params.pubmed_query}")
            all_refined_queries_for_logging.append(f"FALLBACK USED General: {fallback_search_params.general_query}")
        else:
            console.print("[red]CRITICAL: Could not form search parameters using fallback logic.[/red]")
            return {"query": original_query, "refined_queries": all_refined_queries_for_logging, "triaged_articles": []}

    # Loop through extracted_query_pairs (will be 1 if fallback, or up to 3 if successful refinement)
    for i, query_pair_to_use in enumerate(extracted_query_pairs):
        loop_iteration_label = f"query pair {i+1}" if not use_fallback_query else "fallback query"
        console.print(f"[info]--- Running search for {loop_iteration_label} ---[/info]")
        
        current_search_params = convert_metadata_to_search_params(
            collected_metadata, original_query, console, refined_query_pair=query_pair_to_use
        )

        if not current_search_params:
            console.print(f"[yellow]Skipping {loop_iteration_label} as search parameters could not be formed.[/yellow]")
            continue
        
        # Apply publication type filters from the current query pair
        # Priority: CLI args > Query Pair's article_type > Settings
        final_publication_types_to_use = []
        if cli_args.pub_types:
            final_publication_types_to_use = [s.strip().lower() for s in cli_args.pub_types.split(',') if s.strip()]
            console.print(f"[info]Using publication types from CLI for {loop_iteration_label}: {final_publication_types_to_use}[/info]")
        elif query_pair_to_use.get("article_type"): # Now sourced from the query pair
            article_types_from_pair = query_pair_to_use.get("article_type", [])
            # Ensure it's a list and all elements are strings, then normalize
            if isinstance(article_types_from_pair, list):
                final_publication_types_to_use = [str(s).strip().lower() for s in article_types_from_pair if str(s).strip()]
            elif isinstance(article_types_from_pair, str): # Handle if LLM mistakenly returns a string
                final_publication_types_to_use = [s.strip().lower() for s in article_types_from_pair.split(',') if s.strip()]
            console.print(f"[info]Using publication types from query pair for {loop_iteration_label}: {final_publication_types_to_use}[/info]")
        else: # Fallback to settings if not in CLI or query pair
            default_pub_types_from_settings = settings_config.get('search_settings', {}).get('default_publication_types', [])
            if default_pub_types_from_settings:
                 final_publication_types_to_use = [str(s).strip().lower() for s in default_pub_types_from_settings if str(s).strip()]
                 console.print(f"[info]Using default publication types from settings for {loop_iteration_label}: {final_publication_types_to_use}[/info]")
            else:
                console.print(f"[info]No publication type filter applied for {loop_iteration_label}.[/info]")
        
        current_search_params.publication_types = final_publication_types_to_use if final_publication_types_to_use else None
        
        # Apply max results
        current_search_params.max_results_per_source = settings_config.get('search_settings', {}).get('default_max_results_per_source', 50)
        # console.print(f"[info]Using max results per source for {loop_iteration_label}: {current_search_params.max_results_per_source}[/info]") # Already printed by search tool

        # Apply date range filters from the current query pair
        date_range_str = query_pair_to_use.get("date_range", "") # Sourced from query pair
        start_date_val, end_date_val = None, None

        if date_range_str and date_range_str.lower() not in ["no restriction", ""]:
            if "last" in date_range_str.lower() and "year" in date_range_str.lower():
                try:
                    num_years = int(re.findall(r'\d+', date_range_str)[0])
                    current_year = pd.Timestamp.now().year
                    start_date_val = f"{current_year - num_years}-01-01"
                    end_date_val = f"{current_year}-12-31"
                except Exception as e_date: 
                    console.print(f"[yellow]Could not parse 'last X years' from '{date_range_str}': {e_date}[/yellow]")
            elif re.match(r"(\d{4})-(\d{4})", date_range_str): # YYYY-YYYY
                match_yr = re.match(r"(\d{4})-(\d{4})", date_range_str)
                if match_yr:
                    start_date_val = f"{match_yr.group(1)}-01-01"
                    end_date_val = f"{match_yr.group(2)}-12-31"
            elif re.match(r"(\d{4})", date_range_str) and len(date_range_str) == 4: # Single year YYYY
                start_date_val = f"{date_range_str}-01-01"
                end_date_val = f"{date_range_str}-12-31"
            
            if start_date_val and end_date_val:
                current_search_params.start_date = start_date_val
                current_search_params.end_date = end_date_val
                # console.print(f"[info]Applying date range for {loop_iteration_label}: {start_date_val} to {end_date_val}[/info]") # Already printed by search tool
            else:
                console.print(f"[yellow]Could not parse date range '{date_range_str}' from query pair for {loop_iteration_label}. No date filter applied.[/yellow]")
        else:
            console.print(f"[info]No date range restriction applied for {loop_iteration_label} based on query pair ('{date_range_str}').[/info]")

        console.print(f"[secondary]Executing search for {loop_iteration_label} with parameters:[/secondary]")
        console.print(f"  PubMed Query: '{current_search_params.pubmed_query}'")
        console.print(f"  General Query: '{current_search_params.general_query}'")
        console.print(f"  Max Results: {current_search_params.max_results_per_source}")
        console.print(f"  Pub Types: {current_search_params.publication_types}")
        console.print(f"  Start Date: {current_search_params.start_date}, End Date: {current_search_params.end_date}")
        
        try:
            search_output = await literature_search_tool.run(args=current_search_params, cancellation_token=CancellationToken())
            # search_output["data"] is a list of article dicts
            search_results_per_query_pair.append(search_output.get("data", []))
        except Exception as e_search:
            console.print(f"[red]Error during literature search for {loop_iteration_label}: {e_search}[/red]")
            search_results_per_query_pair.append([]) # Add empty list on error to maintain structure

    # Flatten all results from all query pairs
    all_raw_articles_from_searches: List[Dict[str, Any]] = []
    for result_list in search_results_per_query_pair:
        all_raw_articles_from_searches.extend(result_list)

    if not all_raw_articles_from_searches:
        console.print("[yellow]No search results obtained from any query pair or fallback.[/yellow]")
        # Ensure all_articles_data is an empty list if no results
        all_articles_data = []
    else:
        # Deduplicate all_raw_articles_from_searches to populate all_articles_data
        # This is a simplified deduplication for now. The main deduplication happens in search.py
        # Here, we just want to avoid sending grossly duplicated items to triage if multiple query pairs found the same article.
        # A more robust deduplication would use normalized DOIs and PMIDs.
        # For now, let's use a set of (DOI, PMID, Title) tuples to track uniqueness.
        console.print(f"[info]Raw articles from all searches before deduplication for triage: {len(all_raw_articles_from_searches)}[/info]")
        temp_dedup_set = set()
        deduplicated_for_triage: List[Dict[str, Any]] = []
        for article in all_raw_articles_from_searches:
            doi = str(article.get("doi", "")).lower().strip() if article.get("doi") else ""
            pmid = str(article.get("pmid", "")).strip() if article.get("pmid") else ""
            title = str(article.get("title", "")).lower().strip() if article.get("title") else ""
            
            # Prefer DOI if available, then PMID, then title for uniqueness key
            unique_key_parts = []
            if doi: unique_key_parts.append(f"doi:{doi}")
            if pmid: unique_key_parts.append(f"pmid:{pmid}")
            if not doi and not pmid and title: # Only use title if no identifiers
                 unique_key_parts.append(f"title:{title}")
            
            # If no reliable identifier, treat as unique to avoid losing it, or skip if too noisy
            # For now, if no key parts, we'll add it, but this might need refinement.
            unique_identifier = tuple(sorted(unique_key_parts)) if unique_key_parts else f"unique_placeholder_{uuid.uuid4()}"

            if unique_identifier not in temp_dedup_set:
                temp_dedup_set.add(unique_identifier)
                deduplicated_for_triage.append(article)
        all_articles_data = deduplicated_for_triage
        console.print(f"[info]Articles after deduplication for triage: {len(all_articles_data)}[/info]")

    triaged_articles_with_scores: List[Dict[str, Any]] = []
    query_summary_for_triage: Optional[Dict[str, Any]] = None

    if extracted_query_pairs: # If we have refined queries, use the first summary
        query_summary_for_triage = extracted_query_pairs[0].get('query_summary')
        if query_summary_for_triage:
            console.print(f"[info]Using query summary for triage: {json.dumps(query_summary_for_triage)}[/info]")
        else:
            console.print("[yellow]Warning: Refined queries extracted, but no query_summary found in the first one. Triage might be affected.[/yellow]")
    elif use_fallback_query: # No refined queries, and fallback is active
        console.print("[yellow]Warning: Using fallback query. No query_summary available for triage. Triage will proceed without detailed summary.[/yellow]")
        # query_summary_for_triage remains None
    
    if not all_articles_data:
        console.print("[info]No articles to triage after deduplication.[/info]")
    else:
        console.print(f"[secondary]Triaging {len(all_articles_data)} unique articles...[/secondary]")
        try:
            triaged_articles_with_scores = await triage_agent.triage_articles_async(
                articles=all_articles_data, query_summary=query_summary_for_triage # Pass the extracted or None summary
            )
            console.print(f"[secondary]Triage complete. {len(triaged_articles_with_scores)} articles were processed by triage.[/secondary]")
        except Exception as e:
            console.print(f"[red]Error during article triage: {e}[/red]")
            # Fallback: use all_articles_data but mark scores as None or error state
            triaged_articles_with_scores = all_articles_data 
            for article_item in triaged_articles_with_scores:
                article_item['detailed_relevance_scores'] = {key: None for key in ["research_focus", "model_preferences", "must_include", "exclusions", "time_window", "requested_outputs"]}
                article_item['average_relevance_score'] = None
    
    highly_relevant_articles = []
    if triaged_articles_with_scores:
        console.print(f"[secondary]Filtering {len(triaged_articles_with_scores)} triaged articles based on new criteria...[/secondary]")
        for article in triaged_articles_with_scores:
            avg_score = article.get('average_relevance_score')
            detailed_scores = article.get('detailed_relevance_scores', {})
            passes_overall_filter = True

            # Condition 1: Average Score Check
            if avg_score is None or avg_score < 4.0:
                passes_overall_filter = False

            # Condition 2: Individual Category Score Check (only if Condition 1 is met)
            if passes_overall_filter:
                has_at_least_one_valid_detailed_score = False
                for category_score_value in detailed_scores.values():
                    if isinstance(category_score_value, (int, float)): # It's a numerical score
                        has_at_least_one_valid_detailed_score = True
                        if category_score_value < 3.0:
                            passes_overall_filter = False
                            break # Fails individual category threshold
                
                # If avg_score was >= 4.0, but there were no valid detailed scores to check against >=3.0
                # (e.g., all categories were broad, resulting in avg_score being None, which fails Condition 1)
                # This check ensures that if an article somehow got a high avg_score without any scorable categories, it's caught.
                # However, if avg_score is None (all broad), it's already filtered.
                # If avg_score is not None, it means there was at least one valid detailed score.
                # So, this specific 'if not has_at_least_one_valid_detailed_score' might be redundant
                # if avg_score calculation correctly results in None when no valid_numerical_scores exist.
                # For safety, if avg_score passed but no detailed scores were numeric, it should fail.
                if avg_score is not None and not has_at_least_one_valid_detailed_score:
                     passes_overall_filter = False


            if passes_overall_filter:
                highly_relevant_articles.append(article)
        
        console.print(f"[info]Found {len(highly_relevant_articles)} articles meeting the new relevance criteria.[/info]")
    else:
        console.print(f"[secondary]No triaged articles with scores to filter.[/secondary]")

    output_data = {
        "query": original_query,
        "refined_queries": all_refined_queries_for_logging, 
        "triaged_articles": highly_relevant_articles
    }
    
    workspace_dir = Path("workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = workspace_dir / "triage_results.json"
    with open(output_file_path, "w") as f:
        json.dump(output_data, f, indent=2)
    console.print(f"[primary]Search and triage pipeline completed. Results saved to {output_file_path}[/primary]")
    console.print(f"[secondary]Total unique articles (relevance 4 or 5) saved:[/secondary] [highlight]{len(highly_relevant_articles)}[/highlight]")

    # Create a unique run ID and directory for this specific run's detailed outputs
    run_id = str(uuid.uuid4())
    workspace_debug_dir = Path("workspace") / run_id 
    workspace_debug_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[info]Run-specific output directory created: {workspace_debug_dir}[/info]")

    # Determine articles that were triaged but not deemed highly relevant
    other_triaged_articles = []
    if triaged_articles_with_scores:
        highly_relevant_ids = set()
        for article_hr in highly_relevant_articles:
            hr_doi = str(article_hr.get("doi", "")).lower().strip() if article_hr.get("doi") else ""
            hr_pmid = str(article_hr.get("pmid", "")).strip() if article_hr.get("pmid") else ""
            # Using a tuple of (doi, pmid) as a unique key.
            highly_relevant_ids.add((hr_doi, hr_pmid))

        for article_t in triaged_articles_with_scores:
            t_doi = str(article_t.get("doi", "")).lower().strip() if article_t.get("doi") else ""
            t_pmid = str(article_t.get("pmid", "")).strip() if article_t.get("pmid") else ""
            if (t_doi, t_pmid) not in highly_relevant_ids:
                other_triaged_articles.append(article_t)
    
    if other_triaged_articles:
        other_triaged_output_path = workspace_debug_dir / "triaged_results.json"
        with open(other_triaged_output_path, "w") as f:
            json.dump(other_triaged_articles, f, indent=2)
        console.print(f"[info]Other triaged articles ({len(other_triaged_articles)}) saved to {other_triaged_output_path}[/info]")
    else:
        console.print("[info]No 'other' triaged articles to save.[/info]")
    
    full_text_agent_instance = agents.get("FullTextRetrievalAgent")
    full_text_results_list = highly_relevant_articles
    if full_text_agent_instance and highly_relevant_articles:
        console.print(f"[secondary]Attempting full text retrieval for {len(highly_relevant_articles)} articles...[/secondary]")
        try:
            input_json_str = json.dumps(highly_relevant_articles)
            task_message = TextMessage(content=input_json_str, source="pipeline_orchestrator")
            task_result = await full_text_agent_instance.run(task=task_message)
            if task_result.messages and isinstance(task_result.messages[-1], TextMessage):
                response_content = task_result.messages[-1].content
                if isinstance(response_content, str):
                    full_text_results_list = json.loads(response_content)
                    console.print(f"[info]Full text retrieval agent processed {len(full_text_results_list)} articles.[/info]")
                    # run_id and workspace_debug_dir are now defined and created earlier.
                    # workspace_debug_dir.mkdir(parents=True, exist_ok=True) # This directory is already created.
                    full_text_output_path = workspace_debug_dir / "full_text_results.json" # Use Path object
                    with open(full_text_output_path, "w") as f:
                        json.dump(full_text_results_list, f, indent=2)
                    console.print(f"[debug]Full text results saved to {full_text_output_path}[/debug]")
                else:
                    console.print(f"[red]FullTextRetrievalAgent response content was not a string: {type(response_content)}[/red]")
            else:
                console.print(f"[red]FullTextRetrievalAgent did not return a final TextMessage as expected.[/red]")
                if task_result.messages: console.print(f"[debug]Last message from agent: {task_result.messages[-1]}[/debug]")
        except Exception as e:
            console.print(f"[red]Error during full text retrieval agent execution: {e}[/red]")
    elif not highly_relevant_articles:
        console.print("[info]No highly relevant articles to process for full text retrieval.[/info]")
    else:
        console.print("[yellow]FullTextRetrievalAgent not found or not loaded. Skipping full text retrieval.[/yellow]")

    output_data["triaged_articles"] = full_text_results_list # This updates the main summary file data
    with open(output_file_path, "w") as f: # output_file_path is "workspace/triage_results.json"
        json.dump(output_data, f, indent=2)
    console.print(f"[primary]Main pipeline results (including full text attempt) saved to {output_file_path}[/primary]")
    console.print(f"[secondary]Total articles in final output (after full text attempt):[/secondary] [highlight]{len(full_text_results_list)}[/highlight]")
    return output_data

async def main():
    parser = argparse.ArgumentParser(description="Run the literature search pipeline.")
    parser.add_argument("--pub-types", "-t", type=str, default=None, help="Comma-separated list of publication types.")
    args = parser.parse_args()
    base_path = Path(".").resolve()
    env_file_path = base_path / ".env"
    agents_file_path = base_path / "config" / "agents.yaml"
    settings_file_path = base_path / "config" / "settings.yaml"
    if env_file_path.exists(): load_dotenv(dotenv_path=env_file_path, override=True)
    custom_theme = load_rich_theme(base_path)
    console = Console(theme=custom_theme)
    if not agents_file_path.exists(): console.print(f"[red]Error: Agents config not found: {agents_file_path}[/red]"); return
    if not settings_file_path.exists(): console.print(f"[red]Error: Settings config not found: {settings_file_path}[/red]"); return
    with open(agents_file_path, "r") as f: agents_config_list = yaml.safe_load(f)
    with open(settings_file_path, "r") as f: settings_config = yaml.safe_load(f)
    if settings_config is None: settings_config = {}
    
    instantiated_agents = {}
    query_team_instance = None
    query_refiner_cfg_params = None
    publication_type_mappings = settings_config.get('publication_type_mappings', {})

    if agents_config_list:
        for agent_cfg in agents_config_list:
            if not isinstance(agent_cfg, dict) or "name" not in agent_cfg: continue
            agent_name = agent_cfg.get('name', 'Unknown')
            agent_params = agent_cfg.get('config', {})
            model_client_inst = None
            if "model_client" in agent_params:
                mc_cfg_comp = agent_params["model_client"]
                mc_cfg_dict = mc_cfg_comp.get("config", {})
                for key in ["model", "base_url", "api_key"]:
                    if key in mc_cfg_dict: mc_cfg_dict[key] = resolve_env_placeholder(mc_cfg_dict[key])
                if "model_info" in mc_cfg_dict and isinstance(mc_cfg_dict["model_info"], dict):
                    mc_cfg_dict["model_info"] = ModelInfo(**mc_cfg_dict["model_info"])
                try: model_client_inst = DebugOpenAIChatCompletionClient(**mc_cfg_dict)
                except Exception as e: console.print(f"[red]Error instantiating model_client for {agent_name}: {e}[/red]"); continue
            
            component_instance = None
            try:
                filtered_params = {k: v for k, v in agent_params.items() if k not in ["human_input_mode", "code_execution_config"]}
                if agent_name == "user_proxy": component_instance = UserProxyAgent(**filtered_params)
                elif agent_name == "query_refiner":
                    query_refiner_cfg_params = agent_params.copy()
                    assistant_args = { "name": "query_refiner", "model_client": model_client_inst, **{k:v for k,v in agent_params.items() if k in ["tools", "description", "system_message", "reflect_on_tool_use", "tool_call_summary_format"]}}
                    assistant_args.pop('required_fields', None)
                    component_instance = AssistantAgent(**assistant_args)
                elif agent_name == "query_team":
                    participants = [instantiated_agents[p_cfg["name"]] for p_cfg in agent_params["participants"] if p_cfg["name"] in instantiated_agents]
                    if not instantiated_agents.get("query_refiner"): raise ValueError("Query Refiner agent must be defined before query_team.")
                    term_cond = QueryRefinerJsonTermination(query_refiner_agent_name=instantiated_agents["query_refiner"].name)
                    component_instance = RoundRobinGroupChat(participants=participants, termination_condition=term_cond, max_turns=40)
                elif agent_name == "triage": component_instance = TriageAgent(model_client=model_client_inst, **{k:v for k,v in filtered_params.items() if k != "model_client"})
                elif agent_name == "ranker": component_instance = RankerAgent(cfg=StubConfig(**filtered_params))
                elif agent_name == "summariser": component_instance = AssistantAgent(model_client=model_client_inst, **{k:v for k,v in filtered_params.items() if k != "model_client"})
                elif agent_name == "exporter": component_instance = ExporterAgent(cfg=StubConfig(**filtered_params))
                elif agent_name == "search_literature": component_instance = LiteratureSearchTool(publication_type_mappings=publication_type_mappings, **filtered_params)
                elif agent_name == "FullTextRetrievalAgent": component_instance = FullTextRetrievalAgent(**filtered_params)
                
                if component_instance: 
                    instantiated_agents[agent_name] = component_instance
                    if agent_name == "query_team" and isinstance(component_instance, BaseGroupChat): query_team_instance = component_instance
            except Exception as e: console.print(f"[red]Error loading component {agent_name}: {e}[/red]")
    else: console.print("[yellow]Warning: agents_config_list is empty.[/yellow]"); return

    search_tool_inst = instantiated_agents.get("search_literature")
    if not isinstance(search_tool_inst, LiteratureSearchTool):
        console.print(f"[red]Error: 'search_literature' is not a LiteratureSearchTool instance.[/red]"); return
    if not query_team_instance: console.print("[red]Error: 'query_team' not loaded.[/red]"); return
    triage_agent_inst = instantiated_agents.get("triage")
    if not isinstance(triage_agent_inst, TriageAgent): console.print(f"[red]Error: 'triage' agent not TriageAgent instance.[/red]"); return

    console.print("\n[bold blue]──────────────────────────────────────────────────────────────[/bold blue]")
    console.print("[bold blue]                 Literature Search Assistant                [/bold blue]")
    console.print("[bold blue]──────────────────────────────────────────────────────────────[/bold blue]\n")
    
    user_orig_query = await asyncio.get_event_loop().run_in_executor(None, styled_input, "[primary]What would you like to research?[/primary]", console)
    if not user_orig_query: return

    await run_search_pipeline(
        query_team_instance, search_tool_inst, triage_agent_inst, instantiated_agents,
        user_orig_query, console, settings_config, args, query_refiner_cfg_params
    )

if __name__ == "__main__":
    asyncio.run(main())
