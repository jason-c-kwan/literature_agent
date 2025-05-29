import os
import yaml
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo, LLMMessage, CreateResult, SystemMessage as SystemMessageFromCore # Added SystemMessageFromCore
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import TerminationCondition
from tools.search import LiteratureSearchTool, SearchLiteratureParams
from typing import Sequence, Any, Optional, List, Union, Type, Dict # Added Dict
from pydantic import BaseModel # Added for DebugClient
from tools.triage import TriageAgent
from tools.ranking import RankerAgent
from tools.export import ExporterAgent
from tools.retrieve_full_text import FullTextRetrievalAgent # Added for FullTextRetrievalAgent
from tools._base import StubConfig # Added import
import asyncio
import re
import json
import uuid # For generating run_id
from rich.console import Console
from rich.theme import Theme
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
# Using prompt_toolkit for proper readline support with multi-line editing

# Import GroupChat for the new query logic
from autogen_agentchat.teams import BaseGroupChat
from autogen_agentchat.ui import Console as AgentChatConsole # Renamed to avoid conflict
#from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.messages import ToolCallRequestEvent, TextMessage, ToolCallSummaryMessage, BaseChatMessage # Added BaseChatMessage

from autogen_core.tools import FunctionTool
# clarifier_stub and its FunctionTool definition will be replaced by actual_clarifier_tool_func
# and its FunctionTool definition within the main() function, as it needs access to 'console'.
from tools.triage import TriageAgent # Added for TriageAgent type hint

REQUIRED_METADATA_FIELDS = [
    "purpose", "scope", "audience", "article_type",
    "date_range", "open_access", "output_format" # Removed "keywords"
]

def build_prompt_for_metadata_collection(
    original_query: str,
    collected_metadata: Dict[str, Any],
    all_required_fields: List[str],
    field_prompts: Dict[str, str] # The actual prompt text for each field
) -> str:
    """
    Builds the message to send to the QueryRefinerAgent to guide metadata collection.
    """
    if not collected_metadata:
        first_field_to_ask = all_required_fields[0]
        prompt_for_first_field = field_prompts.get(first_field_to_ask, f"Please provide information for {first_field_to_ask}.")
        return (
            f"My initial research query is: '{original_query}'. "
            f"Please start by asking me: \"{prompt_for_first_field}\" "
            "Ask only one question per turn until all required information is gathered."
        )
    else:
        next_field_to_ask = None
        for field in all_required_fields:
            if field not in collected_metadata:
                next_field_to_ask = field
                break
        
        collected_info_str = "; ".join(f"'{k}': '{v}'" for k, v in collected_metadata.items())
        if next_field_to_ask:
            prompt_for_next_field = field_prompts.get(next_field_to_ask, f"Please provide information for {next_field_to_ask}.")
            # Make the instruction very direct
            return (
                f"We have already discussed: {collected_info_str}. "
                f"The original query was: '{original_query}'. "
                f"Your next immediate task is to ask me the following question, and only this question: \"{prompt_for_next_field}\" "
                "Do not summarize. Do not proceed with research. Only ask this one question."
            )
        else: # All fields collected
            return (
                f"All required information has been collected: {collected_info_str}. "
                f"The original query was: '{original_query}'. "
                "You can now proceed based on this information." # This message signals completion to QRA
            )

def parse_chat_history_for_metadata(
    chat_history: Sequence[BaseChatMessage], # Changed type hint
    field_question_map: Dict[str, str], # Maps field name to the question text (prompt) for that field
    user_proxy_name: str, # Name of the UserProxyAgent
    query_refiner_name: str # Name of the QueryRefinerAgent
) -> Dict[str, Any]:
    """
    Parses the chat history to extract metadata based on questions and answers.
    """
    metadata: Dict[str, Any] = {}
    if not chat_history:
        return metadata

    for i, msg in enumerate(chat_history):
        # Check if the message is a question from the QueryRefinerAgent
        if msg.source == query_refiner_name and isinstance(msg.content, str):
            question_text = msg.content.strip()
            # Try to find which field this question corresponds to
            asked_field = None
            for field, prompt_text in field_question_map.items():
                # This matching needs to be robust. The agent might not ask the exact prompt text.
                # For now, we assume the agent asks a question that *contains* the core prompt text.
                # Or, if the agent is well-behaved, it uses the exact prompt.
                # A more robust way would be if the agent tagged its question.
                if prompt_text.lower() in question_text.lower() or field.lower() in question_text.lower(): # Simple check
                    asked_field = field
                    break
            
            # If a known field was asked, look for the next message from UserProxyAgent as the answer
            if asked_field and (i + 1) < len(chat_history):
                answer_msg = chat_history[i+1]
                if answer_msg.source == user_proxy_name and isinstance(answer_msg.content, str):
                    metadata[asked_field] = answer_msg.content.strip()
                    # print(f"DEBUG: Matched field '{asked_field}' with question '{question_text}' and answer '{metadata[asked_field]}'") # Debug
    
    # print(f"DEBUG: [parse_chat_history_for_metadata] History len {len(chat_history)}, Extracted: {metadata}")
    return metadata

def convert_metadata_to_search_params(
    metadata: Dict[str, Any],
    original_query: str,
    console: Console,
    refined_terms: Optional[str] = None
) -> Optional[SearchLiteratureParams]:
    """
    Converts collected metadata and refined terms into SearchLiteratureParams.
    """
    general_query_to_use = original_query # Default
    pubmed_query_to_use = "" # Default to empty, can be refined

    if refined_terms and refined_terms.strip():
        console.print(f"[info]Using refined search terms for general query: '{refined_terms}'[/info]")
        general_query_to_use = refined_terms
        # Potentially, refined_terms could also be structured for PubMed, or inspire its creation.
        # For now, let's assume refined_terms are primarily for general search.
        # A more sophisticated approach might involve the LLM generating specific PubMed syntax.
        # If refined_terms look like a PubMed query, we could use them.
        # Example: if "cancer[MeSH Terms] AND therapy" in refined_terms.
        # For simplicity, we'll keep pubmed_query basic for now or derive it.
        # One simple strategy: if refined_terms are just keywords, use them for pubmed_query too.
        if not any(char in refined_terms for char in "[]()*"): # Basic check if it's not complex PubMed syntax
            pubmed_query_to_use = refined_terms
            console.print(f"[info]Using refined terms for PubMed query as well: '{refined_terms}'[/info]")
        else:
            # If refined_terms seem complex, maybe they are already a PubMed query
            # Or we need a separate step to generate a PubMed query from metadata + refined_terms
            # For now, if refined_terms are complex, we'll use them for general and try to make a simpler one for pubmed
            # or leave pubmed_query as potentially empty if it's hard to derive.
            # A simple fallback for PubMed could be keywords from 'scope' or 'purpose' if available.
            scope_text = metadata.get("scope", "")
            purpose_text = metadata.get("purpose", "")
            if scope_text or purpose_text:
                # This is a very naive keyword extraction for PubMed.
                # Consider using an LLM for better PubMed query formulation based on all metadata.
                potential_pubmed_keywords = f"{scope_text} {purpose_text} {original_query}".strip()
                # Replace common conjunctions that might not work well as standalone pubmed terms
                potential_pubmed_keywords = re.sub(r'\b(and|or|not|the|a|of|in|for|to|with)\b', '', potential_pubmed_keywords, flags=re.IGNORECASE)
                potential_pubmed_keywords = ' '.join(potential_pubmed_keywords.split()) # Clean up multiple spaces
                if potential_pubmed_keywords:
                    pubmed_query_to_use = potential_pubmed_keywords
                    console.print(f"[info]Derived potential PubMed query from scope/purpose/original: '{pubmed_query_to_use}'[/info]")
                else:
                    console.print(f"[yellow]Could not derive a specific PubMed query from metadata, using refined_terms if simple, or empty.[/yellow]")
                    if not any(char in refined_terms for char in "[]()*"):
                         pubmed_query_to_use = refined_terms
                    else: # refined_terms are complex, don't use for pubmed directly
                         pubmed_query_to_use = "" # Or try to extract keywords from original_query
                         if original_query:
                            temp_original_keywords = re.sub(r'\b(and|or|not|the|a|of|in|for|to|with)\b', '', original_query, flags=re.IGNORECASE)
                            pubmed_query_to_use = ' '.join(temp_original_keywords.split())
                            console.print(f"[info]Using keywords from original query for PubMed: '{pubmed_query_to_use}'[/info]")


    elif original_query: # No refined_terms, fallback to original_query
        console.print(f"[yellow]No refined terms provided, falling back to original query for general search: '{original_query}'[/yellow]")
        general_query_to_use = original_query
        # Try to make a simple PubMed query from original_query
        temp_original_keywords = re.sub(r'\b(and|or|not|the|a|of|in|for|to|with)\b', '', original_query, flags=re.IGNORECASE)
        pubmed_query_to_use = ' '.join(temp_original_keywords.split())
        console.print(f"[info]Using keywords from original query for PubMed: '{pubmed_query_to_use}'[/info]")
    else:
        console.print("[red]Cannot form search query: No refined terms and no original query.[/red]")
        return None

    return SearchLiteratureParams(
        pubmed_query=pubmed_query_to_use.strip(),
        general_query=general_query_to_use.strip()
    )

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
        
        # Use the 'tools' argument as it's passed to this create method
        tools_arg = tools # The argument received by this DebugClient.create method
        
        print(f"Tools argument received by DebugClient.create: {'Yes, content:' + str(tools_arg) if tools_arg is not None else 'No (None)'}")
        if tools_arg: # Check if tools_arg is not None and not empty
            print(f"  Type of tools_arg: {type(tools_arg)}")
            if isinstance(tools_arg, list):
                for tool_idx, tool_def_item in enumerate(tools_arg):
                    tool_name_to_print = f"Unknown (type: {type(tool_def_item)})"
                    if hasattr(tool_def_item, 'name') and isinstance(getattr(tool_def_item, 'name'), str): # Covers FunctionTool
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
            # Determine role for printing
            role_to_print = "unknown"
            if isinstance(msg, SystemMessageFromCore):
                role_to_print = "system"
            elif hasattr(msg, "role") and msg.role is not None:
                role_to_print = msg.role
            
            print(f"Message {i}: Role: {role_to_print}")
            print(f"  Content type: {type(msg.content)}")
            if isinstance(msg.content, str):
                try:
                    # Try to pretty print if content is JSON string
                    parsed_json = json.loads(msg.content)
                    print("  Content (parsed as JSON):")
                    print(json.dumps(parsed_json, indent=2))
                except json.JSONDecodeError:
                    print("  Content (string):")
                    print(msg.content)
            elif isinstance(msg.content, list): # For multimodal or tool messages with list content
                print("  Content (list of parts):")
                for part_idx, part in enumerate(msg.content):
                    print(f"    Part {part_idx}: Type: {type(part)}")
                    if isinstance(part, dict): # e.g. text part, image_url part
                         print(f"      {json.dumps(part, indent=2)}")
                    else: # e.g. FunctionCall in AssistantMessage
                         print(f"      {part!r}") # Use repr for other types
            else:
                print(f"  Content: {msg.content!r}") # Use repr for other types

            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  Tool Calls: {msg.tool_calls!r}")
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id: # For ToolMessage
                print(f"  Tool Call ID: {msg.tool_call_id}")
        print("="*80 + "\n")
        
        await asyncio.sleep(0.1) # Ensure print buffer flushes

        try:
            # If tools is None (e.g. during reflection), pass an empty list to super().create.
            # This ensures convert_tools receives an iterable.
            effective_tools_for_super = tools if tools is not None else []
            return await super().create(
                messages,
                tools=effective_tools_for_super,
                cancellation_token=cancellation_token,
                json_output=json_output,
                **kwargs,
            )
        except Exception as e:
            print(f"DEBUG: Error during model_client.create AFTER printing payload: {type(e).__name__} - {e}")
            raise

# Helper function to resolve environment variable placeholders
def resolve_env_placeholder(value: str) -> str:
    if isinstance(value, str):
        # Match ${VAR_NAME} or ${VAR_NAME:-default_value}
        match = re.fullmatch(r"\$\{(.+?)(?::-([^}]+))?\}", value)
        if match:
            var_name = match.group(1)
            default_value = match.group(2)
            env_var = os.getenv(var_name)
            if env_var is not None:
                return env_var
            elif default_value is not None:
                return default_value
            else:
                # If no default and env var not set, return None or raise error,
                # or return the placeholder itself to see if downstream handles it
                # For now, let's return None, as API keys/base URLs shouldn't be empty strings
                print(f"Warning: Environment variable {var_name} not set and no default provided for '{value}'")
                return None
    return value

# Load custom rich theme
def load_rich_theme(base_path: Path) -> Theme:
    theme_file_path = base_path / "config" / "rich_theme.json"
    if theme_file_path.exists():
        with open(theme_file_path, "r") as f:
            theme_config = json.load(f)
        return Theme(theme_config)
    return Theme({})

def styled_input(prompt_message: str, console: Console) -> str:
    """
    Display a styled prompt using Rich console and capture input using prompt_toolkit
    for proper readline support (cursor movement, history, etc.)
    """
    # Extract the text content from Rich markup for plain input prompt
    import re
    # Remove Rich markup tags to get plain text
    plain_prompt = re.sub(r'\[/?[^\]]*\]', '', prompt_message)
    
    # Use prompt_toolkit for proper multi-line editing support
    return prompt(f"{plain_prompt} ").strip()

async def run_search_pipeline(
    query_team: BaseGroupChat, # May become unused for refinement, but kept for signature consistency for now
    literature_search_tool: LiteratureSearchTool,
    triage_agent: TriageAgent,
    agents: Dict[str, Any],
    original_query: str,
    console: Console,
    settings_config: Dict[str, Any],
    cli_args: argparse.Namespace,
    query_refiner_config_params: Dict[str, Any],
    fields_to_collect_override: Optional[List[str]] = None
):
    all_articles_data = []
    processed_dois_for_articles = set()
    all_refined_queries = []

    query_refiner_agent = agents.get("query_refiner")
    user_proxy_agent = agents.get("user_proxy")

    if not query_refiner_agent or not user_proxy_agent:
        console.print("[red]Error: Query Refiner or User Proxy agent not found in 'agents' dict.[/red]")
        return {"query": original_query, "refined_queries": [], "triaged_articles": []}

    field_question_map = query_refiner_config_params.get('required_fields', {})
    
    active_fields_to_collect = fields_to_collect_override if fields_to_collect_override is not None else REQUIRED_METADATA_FIELDS

    # Ensure field_question_map (from config) has prompts for all active_fields_to_collect.
    if not active_fields_to_collect:
        console.print(f"[red]Error: No fields specified for metadata collection.[/red]")
        search_params = convert_metadata_to_search_params({}, original_query, console) # Fallback
    elif not all(field in field_question_map for field in active_fields_to_collect):
        missing_prompts = [field for field in active_fields_to_collect if field not in field_question_map]
        console.print(f"[red]Error: 'required_fields' in QueryRefinerAgent config is missing prompts for: {missing_prompts} (from active list: {active_fields_to_collect}). Config has prompts for: {list(field_question_map.keys())}[/red]")
        search_params = convert_metadata_to_search_params({}, original_query, console) # Fallback
    else:
        # Proceed with metadata collection
        collected_metadata: Dict[str, Any] = {}
        MAX_CLARIFICATION_ITERATIONS = len(active_fields_to_collect) + 3
        clarification_iteration = 0
        console.print(f"[secondary]Starting metadata collection for query:[/secondary] [highlight]'{original_query}'[/highlight]")
        console.print(f"[info]Fields to collect: {active_fields_to_collect}[/info]")

        while not all(field in collected_metadata for field in active_fields_to_collect) and clarification_iteration < MAX_CLARIFICATION_ITERATIONS:
            clarification_iteration += 1
            console.print(f"[info]Metadata collection iteration {clarification_iteration}/{MAX_CLARIFICATION_ITERATIONS}. Collected: {collected_metadata}[/info]")

            current_prompt_to_refiner = build_prompt_for_metadata_collection(
                original_query,
                collected_metadata,
                active_fields_to_collect, # Use the determined list of fields
                field_question_map
            )
            
            if "All required information has been collected" in current_prompt_to_refiner:
                 console.print("[info]Metadata collection prompt indicates completion based on current state.")
                 break

            console.print(f"[debug]UserProxyAgent sending to QueryRefinerAgent (iteration {clarification_iteration}): {current_prompt_to_refiner}[/debug]")
            
            try:
                # Use query_team.run_stream wrapped with AgentChatConsole
                # The task for query_team.run_stream() should be a message object or string.
                # current_prompt_to_refiner is already a string.
                chat_result = await AgentChatConsole(query_team.run_stream(
                    task=TextMessage(content=current_prompt_to_refiner, source="pipeline_orchestrator")
                ))
            except Exception as e_chat:
                console.print(f"[red]Error during AgentChatConsole(query_team.run_stream(...)): {e_chat}[/red]")
                break

            if chat_result and chat_result.messages: # Changed from chat_result.chat_history to chat_result.messages
                actual_chat_messages = [m for m in chat_result.messages if isinstance(m, BaseChatMessage)]
                console.print(f"[debug]Chat history for iteration {clarification_iteration} (last {len(actual_chat_messages)} chat messages):")
                for msg_idx, msg_obj in enumerate(actual_chat_messages):
                     console.print(f"[debug]  Msg {msg_idx} (Source: {getattr(msg_obj, 'source', 'N/A')}): {str(msg_obj.content)[:200]}...")

                newly_collected = parse_chat_history_for_metadata(
                    actual_chat_messages, # Pass filtered list
                    field_question_map,
                    user_proxy_agent.name,
                    query_refiner_agent.name
                )
                if newly_collected:
                    all_needed_present_in_newly_collected = True
                    if not active_fields_to_collect: # Handle empty active_fields_to_collect
                        all_needed_present_in_newly_collected = False
                    else:
                        for fld in active_fields_to_collect:
                            if fld not in newly_collected:
                                all_needed_present_in_newly_collected = False
                                break
                    
                    if all_needed_present_in_newly_collected and \
                       len(newly_collected) >= len(active_fields_to_collect) and \
                       all(f in newly_collected for f in active_fields_to_collect): # Ensure all active fields are in newly_collected
                        # If newly_collected has at least all the active fields,
                        # and all active fields are indeed in newly_collected.
                        # Prioritize taking all values from newly_collected for the active fields.
                        temp_collected_metadata = collected_metadata.copy()
                        for fld in active_fields_to_collect:
                            if fld in newly_collected: # Should always be true if all_needed_present_in_newly_collected is true
                                temp_collected_metadata[fld] = newly_collected[fld]
                        collected_metadata = temp_collected_metadata
                        # console.print(f"[debug] Optimization: newly_collected contains all active fields. Updated collected_metadata. Collected: {collected_metadata}")
                    else:
                        # Original update logic if the above condition isn't met
                        # console.print(f"[debug] Standard update. Newly: {newly_collected}, Active: {active_fields_to_collect}, Current Collected: {collected_metadata}")
                        for k, v in newly_collected.items():
                            if k in active_fields_to_collect and k not in collected_metadata: # Add only if not already present from a previous partial collection
                                collected_metadata[k] = v
                            elif k in active_fields_to_collect and k in collected_metadata and collected_metadata[k] != v : # If present but different, update (optional, depends on desired behavior)
                                # This part is tricky: should new values override old ones if a field is "re-collected"?
                                # For now, let's stick to adding only if not already present to match original logic more closely for partials.
                                # If a field is asked for again, parse_chat_history_for_metadata should ideally handle the latest answer.
                                # The current loop structure implies we ask until a field is present.
                                pass # Sticking to "add if not present"

                    console.print(f"[info]Collected in this iteration (potentially merged): {newly_collected}. Current total collected: {collected_metadata}[/info]")
                else:
                    console.print(f"[yellow]No new metadata parsed in iteration {clarification_iteration}. Check chat history and parsing logic.[/yellow]")
            else:
                console.print(f"[yellow]No chat history from user_proxy_agent.initiate_chat in iteration {clarification_iteration}.[/yellow]")
                break
        
        if not all(field in collected_metadata for field in active_fields_to_collect):
            console.print(f"[red]Failed to collect all required metadata for {active_fields_to_collect} after {clarification_iteration} iterations. Collected: {collected_metadata}[/red]")
            search_params = convert_metadata_to_search_params({}, original_query, console)
        else:
            console.print(f"[success]All required metadata collected for {active_fields_to_collect}: {collected_metadata}[/success]")
            search_params = convert_metadata_to_search_params(collected_metadata, original_query, console)

    # After metadata collection loop, try to get refined search terms
    refined_search_terms = None
    if chat_result and chat_result.messages:
        # Get the last message from query_refiner_agent
        last_refiner_msg_content = None
        for msg in reversed(chat_result.messages):
            if isinstance(msg, BaseChatMessage) and msg.source == query_refiner_agent.name:
                if isinstance(msg.content, str):
                    last_refiner_msg_content = msg.content
                    break
        
        if last_refiner_msg_content:
            match = re.search(r"Refined Search Terms: (.*)", last_refiner_msg_content, re.IGNORECASE)
            if match:
                refined_search_terms = match.group(1).strip()
                console.print(f"[info]Extracted refined search terms: '{refined_search_terms}'[/info]")

    # Pass refined terms (if any) to convert_metadata_to_search_params
    search_params = convert_metadata_to_search_params(collected_metadata, original_query, console, refined_search_terms)

    # Fallback if search_params could not be formed
    if not search_params:
        if original_query: # Try one last time with original query if search_params is None
            console.print("[yellow]Search params were None, attempting fallback to original query directly.[/yellow]")
            search_params = convert_metadata_to_search_params({}, original_query, console, None) # Pass None for refined_terms
        
        if not search_params: # If still None, then critical failure
             console.print("[red]CRITICAL: Could not form any search parameters.[/red]")
             return {"query": original_query, "refined_queries": [], "triaged_articles": []}

    if search_params and search_params.general_query:
        all_refined_queries.append(f"General: {search_params.general_query}")
    if search_params.pubmed_query: # This might be empty if not well-formulated
        all_refined_queries.append(f"PubMed: {search_params.pubmed_query}")
    
    # Ensure original_query is added if no refined queries were generated, or add refined if available
    if not all_refined_queries and original_query:
        all_refined_queries.append(f"Original (used as fallback): {original_query}")
    elif refined_search_terms and f"General: {refined_search_terms}" not in all_refined_queries: # Avoid duplicate if refined_terms became general_query
        # This logic might need refinement based on how search_params uses refined_terms
        pass


    try: # OUTER TRY BLOCK for the rest of the pipeline (search, triage, etc.)
        final_publication_types_to_use = []
        if cli_args.pub_types:
            final_publication_types_to_use = [s.strip().lower() for s in cli_args.pub_types.split(',') if s.strip()]
            console.print(f"[info]Using publication types from CLI: {final_publication_types_to_use}[/info]")
        else:
            default_pub_types_from_settings = settings_config.get('search_settings', {}).get('default_publication_types', [])
            if default_pub_types_from_settings:
                 final_publication_types_to_use = [str(s).strip().lower() for s in default_pub_types_from_settings if str(s).strip()]
                 console.print(f"[info]Using default publication types from settings: {final_publication_types_to_use}[/info]")
            else:
                console.print(f"[info]No publication type filter applied.[/info]")

        default_max_results = settings_config.get('search_settings', {}).get('default_max_results_per_source', 50)
        max_results_to_use = default_max_results
        console.print(f"[info]Using max results per source: {max_results_to_use}[/info]")

        search_params.max_results_per_source = max_results_to_use
        search_params.publication_types = final_publication_types_to_use if final_publication_types_to_use else None
        
        console.print(f"[secondary]Executing literature search with parameters:[/secondary]")
        console.print(f"  [highlight]PubMed Query: '{search_params.pubmed_query}'[/highlight]")
        console.print(f"  [highlight]General Query: '{search_params.general_query}'[/highlight]")
        console.print(f"  [highlight]Max Results: {search_params.max_results_per_source}[/highlight]")
        console.print(f"  [highlight]Pub Types: {search_params.publication_types}[/highlight]")

        search_output = await literature_search_tool.run(
            args=search_params,
            cancellation_token=CancellationToken()
        )
        
        search_results_data = search_output.get("data", [])
        
        for record in search_results_data:
            if isinstance(record, dict):
                doi_val = record.get('doi')
                if isinstance(doi_val, str) and doi_val.strip():
                    if doi_val not in processed_dois_for_articles:
                        all_articles_data.append(record)
                        processed_dois_for_articles.add(doi_val)
            else:
                console.print(f"[yellow]Warning: search_results_data contained a non-dictionary item: {type(record)}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during the main query refinement or search process: {e}[/red]")
        # Ensure all_articles_data is initialized if an error occurs before search loop
        # or if we want to proceed with empty/partially filled data for triage.
        # For now, if an error happens here, triage will likely operate on an empty list.

    # 5. Triage all collected articles
    triaged_articles_with_scores = []
    if all_articles_data:
        console.print(f"[secondary]Triaging {len(all_articles_data)} unique articles...[/secondary]")
        try:
            triaged_articles_with_scores = await triage_agent.triage_articles_async(
                articles=all_articles_data,
                user_query=original_query
            )
            console.print(f"[secondary]Triage complete. {len(triaged_articles_with_scores)} articles passed filters and were scored.[/secondary]")
        except Exception as e:
            console.print(f"[red]Error during article triage: {e}[/red]")
            # Fallback to using all_articles_data if triage fails, but without scores
            triaged_articles_with_scores = all_articles_data 
            # Add a note that scores are missing if triage failed
            for article in triaged_articles_with_scores:
                if 'relevance_score' not in article:
                    article['relevance_score'] = "N/A (Triage Error)"

    # NEW: Filter triaged articles by relevance score
    highly_relevant_articles = []
    if triaged_articles_with_scores:
        console.print(f"[secondary]Filtering triaged articles for relevance scores 4 or 5...[/secondary]")
        for article in triaged_articles_with_scores:
            score = article.get('relevance_score')
            # Ensure score is an integer and is 4 or 5
            if isinstance(score, int) and score in [4, 5]:
                highly_relevant_articles.append(article)
        console.print(f"[info]Found {len(highly_relevant_articles)} articles with relevance score 4 or 5 out of {len(triaged_articles_with_scores)} triaged articles.[/info]")
    else:
        console.print(f"[secondary]No triaged articles to filter.[/secondary]")

    output_data = {
        "query": original_query,
        "refined_queries": all_refined_queries,
        "triaged_articles": highly_relevant_articles # MODIFIED: Use filtered list
    }
    
    # Create workspace directory if it doesn't exist
    workspace_dir = Path("workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = workspace_dir / "triage_results.json"

    with open(output_file_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"[primary]Search and triage pipeline completed. Results saved to {output_file_path}[/primary]")
    # MODIFIED: Update count and message
    console.print(f"[secondary]Total unique articles (relevance 4 or 5) saved:[/secondary] [highlight]{len(highly_relevant_articles)}[/highlight]")
    
    # --- Full Text Retrieval Step ---
    full_text_agent_instance = agents.get("FullTextRetrievalAgent")
    full_text_results_list = highly_relevant_articles # Default to previous results if agent fails

    if full_text_agent_instance and highly_relevant_articles:
        console.print(f"[secondary]Attempting full text retrieval for {len(highly_relevant_articles)} articles...[/secondary]")
        try:
            # Prepare input for FullTextRetrievalAgent
            # It expects a TextMessage with JSON string content
            input_json_str = json.dumps(highly_relevant_articles)
            task_message = TextMessage(content=input_json_str, source="pipeline_orchestrator")
            
            # Run the agent
            # The agent's run method returns a TaskResult
            task_result = await full_text_agent_instance.run(task=task_message)
            
            if task_result.messages and isinstance(task_result.messages[-1], TextMessage):
                response_content = task_result.messages[-1].content
                if isinstance(response_content, str):
                    full_text_results_list = json.loads(response_content)
                    console.print(f"[info]Full text retrieval agent processed {len(full_text_results_list)} articles.[/info]")
                    
                    # Write full_text_results to disk for debugging
                    run_id = str(uuid.uuid4()) # Generate a unique run ID
                    workspace_debug_dir = os.path.join("workspace", run_id)
                    os.makedirs(workspace_debug_dir, exist_ok=True)
                    full_text_output_path = os.path.join(workspace_debug_dir, "full_text_results.json")
                    with open(full_text_output_path, "w") as f:
                        json.dump(full_text_results_list, f, indent=2)
                    console.print(f"[debug]Full text results saved to {full_text_output_path}[/debug]")
                else:
                    console.print(f"[red]FullTextRetrievalAgent response content was not a string: {type(response_content)}[/red]")
            else:
                console.print(f"[red]FullTextRetrievalAgent did not return a final TextMessage as expected.[/red]")
                if task_result.messages:
                     console.print(f"[debug]Last message from agent: {task_result.messages[-1]}[/debug]")


        except Exception as e:
            console.print(f"[red]Error during full text retrieval agent execution: {e}[/red]")
            # full_text_results_list remains highly_relevant_articles
    elif not highly_relevant_articles:
        console.print("[info]No highly relevant articles to process for full text retrieval.[/info]")
    else:
        console.print("[yellow]FullTextRetrievalAgent not found or not loaded. Skipping full text retrieval.[/yellow]")

    # Update output_data with the results from full text retrieval
    # The 'triaged_articles' key will now hold articles potentially enriched with 'fulltext'
    output_data["triaged_articles"] = full_text_results_list
    
    # Re-save the triage_results.json, now potentially with full text data
    with open(output_file_path, "w") as f: # output_file_path was defined earlier
        json.dump(output_data, f, indent=2)
    console.print(f"[primary]Main pipeline results (including full text attempt) saved to {output_file_path}[/primary]")
    console.print(f"[secondary]Total articles in final output (after full text attempt):[/secondary] [highlight]{len(full_text_results_list)}[/highlight]")

    return output_data

async def main():
    """
    Main asynchronous function to load configurations, instantiate agents, and run the search pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the literature search pipeline.")
    parser.add_argument(
        "--pub-types", "-t", 
        type=str, 
        default=None, 
        help="Comma-separated list of publication types (e.g., research,review). Overrides settings.yaml default."
    )
    # Add other CLI arguments here if needed, e.g., for query, max_results_override
    args = parser.parse_args()

    base_path = Path(".").resolve()
    env_file_path = base_path / ".env"
    agents_file_path = base_path / "config" / "agents.yaml"
    settings_file_path = base_path / "config" / "settings.yaml"
    theme_file_path = base_path / "config" / "rich_theme.json"

    # Load custom rich theme and create console
    custom_theme = load_rich_theme(base_path)
    console = Console(theme=custom_theme)

    # 1. Load .env, config/agents.yaml & config/settings.yaml
    if env_file_path.exists():
        dotenv_loaded = load_dotenv(dotenv_path=env_file_path, override=True)
    else:
        pass # console.print(f"[yellow].env file NOT found at: {env_file_path}[/yellow]")
        # dotenv_loaded = False

    if not agents_file_path.exists():
        console.print(f"[red]Error: Agents configuration file not found at {agents_file_path}[/red]")
        return
    if not settings_file_path.exists():
        console.print(f"[red]Error: Settings configuration file not found at {settings_file_path}[/red]")
        return

    with open(agents_file_path, "r") as f:
        agents_config_list = yaml.safe_load(f)
        # console.print(f"[primary]Agents loaded (raw from YAML): {agents_config_list}[/primary]")
        
    with open(settings_file_path, "r") as f:
        settings_config = yaml.safe_load(f)
        if settings_config is None: settings_config = {} # Ensure it's a dict

    # 2. Manually instantiate agents and teams
    agents = {}
    query_team = None
    query_refiner_config_params_for_pipeline = None # To store QueryRefiner's specific config
    
    # Extract publication_type_mappings from settings_config for LiteratureSearchTool
    publication_type_mappings_from_settings = settings_config.get('publication_type_mappings', {})

    if agents_config_list:
        for agent_config_original in agents_config_list:
            if not isinstance(agent_config_original, dict) or "name" not in agent_config_original:
                continue

            agent_name = agent_config_original.get('name', 'Unknown')
            agent_config_params = agent_config_original.get('config', {})
            
            model_client_instance = None
            if "model_client" in agent_config_params:
                model_client_component_config = agent_config_params["model_client"]
                model_client_config_dict = model_client_component_config.get("config", {})
                
                # Resolve env vars for model client config
                if "model" in model_client_config_dict:
                    model_client_config_dict["model"] = resolve_env_placeholder(model_client_config_dict["model"])
                if "base_url" in model_client_config_dict:
                    model_client_config_dict["base_url"] = resolve_env_placeholder(model_client_config_dict["base_url"])
                if "api_key" in model_client_config_dict:
                    model_client_config_dict["api_key"] = resolve_env_placeholder(model_client_config_dict["api_key"])
                
                # Explicitly convert model_info to ModelInfo TypedDict if present
                if "model_info" in model_client_config_dict and isinstance(model_client_config_dict["model_info"], dict):
                    model_client_config_dict["model_info"] = ModelInfo(**model_client_config_dict["model_info"])

                # Instantiate OpenAIChatCompletionClient
                try:
                    model_client_instance = DebugOpenAIChatCompletionClient(**model_client_config_dict) # Use Debug Client
                    # console.print(f"  [primary]Manually instantiated model_client for {agent_name}.[/primary]")
                except Exception as e:
                    console.print(f"[red]Error instantiating model_client for {agent_name}: {e}[/red]")
                    continue # Skip this agent if model_client fails

            component = None
            try:
                # Filter out 'human_input_mode' and 'code_execution_config' for UserProxyAgent and AssistantAgent
                filtered_agent_config_params = {k: v for k, v in agent_config_params.items() if k not in ["human_input_mode", "code_execution_config"]}

                if agent_name == "user_proxy":
                    component = UserProxyAgent(**filtered_agent_config_params)
                elif agent_name == "query_refiner":
                    # Store the raw config for query_refiner to pass to pipeline
                    query_refiner_config_params_for_pipeline = agent_config_params.copy() # Use .copy()
                    
                    # The clarify_query tool is removed from QueryRefinerAgent's config in agents.yaml
                    # and its human_input_mode is ALWAYS. So, no FunctionTool needed here for it.
                    # The 'tools' key might not even be in agent_config_params if not defined in YAML.
                    # We should ensure 'tools' is an empty list if not present or if we want to override.
                    assistant_params = {
                        "name": "query_refiner", # Name from config
                        "model_client": model_client_instance,
                        "tools": agent_config_params.get("tools", []), # Use tools from YAML if any, else empty
                        "description": agent_config_params.get("description"),
                        "system_message": agent_config_params.get("system_message"),
                        "reflect_on_tool_use": agent_config_params.get("reflect_on_tool_use", False),
                        "tool_call_summary_format": agent_config_params.get("tool_call_summary_format", "{result}")
                    }
                    # Remove 'required_fields' from params passed to AssistantAgent constructor if it exists
                    assistant_params.pop('required_fields', None)

                    component = AssistantAgent(**assistant_params)
                    
                elif agent_name == "query_team":
                    # Participants need to be instantiated first
                    participants_instances = []
                    for p_config in agent_config_params["participants"]:
                        if p_config["name"] in agents:
                            participants_instances.append(agents[p_config["name"]])
                        else:
                            raise ValueError(f"Participant {p_config['name']} not found for query_team. Ensure order in agents.yaml.")
                    
                    term_config = agent_config_params["termination_condition"]["config"]

                    component = RoundRobinGroupChat(
                        participants=participants_instances,
                        termination_condition=None,  
                        max_turns=2,  # Allow for question and answer
                    )
                elif agent_name == "triage":
                    component = TriageAgent(model_client=model_client_instance, **{k:v for k,v in filtered_agent_config_params.items() if k != "model_client"})
                elif agent_name == "ranker":
                    # Wrap config in StubConfig
                    component = RankerAgent(cfg=StubConfig(**filtered_agent_config_params)) # Changed to pass StubConfig
                elif agent_name == "summariser":
                    component = AssistantAgent(model_client=model_client_instance, **{k:v for k,v in filtered_agent_config_params.items() if k != "model_client"})
                elif agent_name == "exporter":
                    # Wrap config in StubConfig
                    component = ExporterAgent(cfg=StubConfig(**filtered_agent_config_params)) # Changed to pass StubConfig
                elif agent_name == "search_literature":
                    # Pass mappings to LiteratureSearchTool constructor
                    component = LiteratureSearchTool(
                        publication_type_mappings=publication_type_mappings_from_settings,
                        **filtered_agent_config_params
                    )
                elif agent_name == "FullTextRetrievalAgent":
                    # FullTextRetrievalAgent does not require a model_client
                    component = FullTextRetrievalAgent(**filtered_agent_config_params)
                else:
                    # console.print(f"[yellow]Warning: Unhandled component type for {agent_name}. Skipping.[/yellow]")
                    pass

                if component:
                    agents[agent_name] = component
                    # Ensure query_team is assigned if it's the component being processed
                    if agent_name == "query_team" and isinstance(component, BaseGroupChat): # Added type check
                        query_team = component
            except Exception as e:
                console.print(f"[red]Error manually loading component {agent_name}: {e}[/red]")
                agents[agent_name] = None

        refiner = agents.get("query_refiner")
        if not refiner:
            console.print("[red]query_refiner never loaded; check your instantiation logic[/red]")
        else:
            # The list of Handoff configs lives in a private `_handoffs` attr:
            pass # console.print(f"[debug] query_refiner._handoffs = {refiner._handoffs!r}")
            # And any FunctionTools or other tools show up under `_tool_configs`:
            pass # console.print(f"[debug] query_refiner._tool_configs = {getattr(refiner, '_tool_configs', None)!r}")

            
    else:
        console.print("[yellow]Warning: agents_config_list is empty or None.[/yellow]")
        return

    # Retrieve the instantiated LiteratureSearchTool from agents dict
    literature_search_tool_instance = agents.get("search_literature")
    if not literature_search_tool_instance:
        # Fallback if not defined in agents.yaml, though it should be
        console.print("[yellow]Warning: LiteratureSearchTool not found in agents.yaml, instantiating with mappings.[/yellow]")
        literature_search_tool_instance = LiteratureSearchTool(publication_type_mappings=publication_type_mappings_from_settings)
    elif not isinstance(literature_search_tool_instance, LiteratureSearchTool):
        console.print(f"[red]Error: 'search_literature' component is not an instance of LiteratureSearchTool. Got {type(literature_search_tool_instance)}[/red]")
        return


    if not query_team:
        console.print("[red]Error: 'query_team' not found or failed to load.[/red]")
        return
    
    triage_agent_instance = agents.get("triage")
    if not triage_agent_instance:
        console.print("[red]Error: 'triage' agent not found or failed to load.[/red]")
        return
    if not isinstance(triage_agent_instance, TriageAgent):
        console.print(f"[red]Error: 'triage' agent is not an instance of TriageAgent. Got {type(triage_agent_instance)}[/red]")
        return

    # Friendly Rich-styled header
    console.print("\n[bold blue]──────────────────────────────────────────────────────────────[/bold blue]")
    console.print("[bold blue]                 Literature Search Assistant                [/bold blue]")
    console.print("[bold blue]──────────────────────────────────────────────────────────────[/bold blue]\n")

    # 1. Read a free-text search query from the user.
    # Run styled_input in a separate thread to avoid blocking the asyncio event loop
    loop = asyncio.get_event_loop()
    original_query = await loop.run_in_executor(None, styled_input, "[primary]What would you like to research?[/primary]", console)
    if not original_query:
        # console.print("[yellow]No query entered. Exiting.[/yellow]")
        return

    # Pass the full 'agents' dictionary to run_search_pipeline
    # so it can retrieve FullTextRetrievalAgent or any other agent it might need.
    await run_search_pipeline(
        query_team,
        literature_search_tool_instance,
        triage_agent_instance,
        agents,
        original_query,
        console,
        settings_config,
        args,
        query_refiner_config_params=query_refiner_config_params_for_pipeline # Pass the specific config
    )


if __name__ == "__main__":
    asyncio.run(main())
