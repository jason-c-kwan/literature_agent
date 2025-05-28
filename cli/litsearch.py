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
#from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.messages import ToolCallRequestEvent, TextMessage, ToolCallSummaryMessage # Added ToolCallSummaryMessage

from autogen_core.tools import FunctionTool
# clarifier_stub and its FunctionTool definition will be replaced by actual_clarifier_tool_func
# and its FunctionTool definition within the main() function, as it needs access to 'console'.
from tools.triage import TriageAgent # Added for TriageAgent type hint


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
    query_team: BaseGroupChat,
    literature_search_tool: LiteratureSearchTool,
    triage_agent: TriageAgent,
    agents: Dict[str, Any], # Added agents dictionary
    original_query: str,
    console: Console,
    settings_config: Dict[str, Any],
    cli_args: argparse.Namespace
):
    all_articles_data = []
    processed_dois_for_articles = set()
    all_refined_queries = []
    
    MAX_REFINEMENT_ATTEMPTS = 3
    refinement_attempts = 0
    refined_query_objects = []
    
    console.print(f"[secondary]Initiating research query with GroupChat:[/secondary] [highlight]'{original_query}'[/highlight]")
    
    # Determine publication types to use
    final_publication_types_to_use = []
    if cli_args.pub_types: # CLI flag is provided
        final_publication_types_to_use = [s.strip().lower() for s in cli_args.pub_types.split(',') if s.strip()]
        console.print(f"[info]Using publication types from CLI: {final_publication_types_to_use}[/info]")
    else: # CLI flag not provided, use default from settings
        default_pub_types_from_settings = settings_config.get('search_settings', {}).get('default_publication_types', [])
        if default_pub_types_from_settings: # Ensure it's a list and clean
             final_publication_types_to_use = [str(s).strip().lower() for s in default_pub_types_from_settings if str(s).strip()]
             console.print(f"[info]Using default publication types from settings: {final_publication_types_to_use}[/info]")
        else:
            console.print(f"[info]No publication type filter applied (neither CLI nor settings default provided).[/info]")

    # Determine max_results_per_source
    default_max_results = settings_config.get('search_settings', {}).get('default_max_results_per_source', 50)
    # Potentially add a CLI flag for max_results here if desired in future, to override settings
    max_results_to_use = default_max_results
    console.print(f"[info]Using max results per source: {max_results_to_use}[/info]")

    try: # OUTER TRY BLOCK
        # Query Refinement Loop
        while refinement_attempts < MAX_REFINEMENT_ATTEMPTS:
            refinement_attempts += 1
            console.print(f"[info]Query refinement attempt {refinement_attempts}/{MAX_REFINEMENT_ATTEMPTS}...[/info]")
            
            task_for_refiner = original_query
            if refinement_attempts > 1 and 'last_refined_queries_str' in locals() and last_refined_queries_str:
                task_for_refiner = (
                    f"The previous attempt to generate refined queries resulted in a JSON parsing error. "
                    f"The problematic output was: \n```json\n{last_refined_queries_str}\n```\n"
                    f"Please ensure your output is a valid JSON array of three objects, each with 'pubmed_query' and 'general_query' string fields. "
                    f"The original query was: '{original_query}'"
                )

            refined_queries_str = "" 
            try: # INNER TRY for team_result and parsing
                team_result = await query_team.run(task=task_for_refiner)
                console.print(f"[debug]Team run completed (attempt {refinement_attempts}). Messages: {team_result.messages}[/debug]")

                if team_result.messages and isinstance(team_result.messages, list) and len(team_result.messages) > 0:
                    for msg in reversed(team_result.messages):
                        if msg.source == "query_refiner" and (isinstance(msg, TextMessage) or isinstance(msg, ToolCallSummaryMessage)):
                            if hasattr(msg, "content") and isinstance(msg.content, str):
                                code_block_pattern = re.compile(r"```(?:json|python|text)?\n(.*?)\n```", re.DOTALL)
                                match = code_block_pattern.search(msg.content)
                                if match:
                                    refined_queries_str = match.group(1).strip()
                                    break
                                elif msg.content.strip().startswith("[") and msg.content.strip().endswith("]"):
                                    refined_queries_str = msg.content.strip()
                                    break
                
                last_refined_queries_str = refined_queries_str

                if not refined_queries_str:
                    console.print(f"[red]Query refiner did not produce a parsable output string on attempt {refinement_attempts}.[/red]")
                    if refinement_attempts == MAX_REFINEMENT_ATTEMPTS:
                        console.print(f"[red]Max refinement attempts reached. No output from query refiner.[/red]")
                    continue 

                console.print(f"[debug]Attempting to parse refined_queries_str (attempt {refinement_attempts}): {refined_queries_str}[/debug]")
                
                parsed_json = json.loads(refined_queries_str)
                if not isinstance(parsed_json, list):
                    raise ValueError("Agent did not return a JSON array.")
                if not all(isinstance(obj, dict) and "pubmed_query" in obj and "general_query" in obj for obj in parsed_json):
                    raise ValueError("Each object in the JSON array must have 'pubmed_query' and 'general_query' fields.")
                if len(parsed_json) != 3:
                     console.print(f"[yellow]Warning: Expected 3 refined query objects, but got {len(parsed_json)} on attempt {refinement_attempts}. Using as is.[/yellow]")

                refined_query_objects = parsed_json
                console.print(f"[success]Successfully parsed refined queries on attempt {refinement_attempts}.[/success]")
                break 
                
            except (json.JSONDecodeError, ValueError) as e_parse:
                console.print(f"[red]Error parsing Query-RefinerAgent output on attempt {refinement_attempts}: {e_parse}. Raw output: {refined_queries_str}[/red]")
                refined_query_objects = [] 
                if refinement_attempts == MAX_REFINEMENT_ATTEMPTS:
                    console.print(f"[red]Max refinement attempts reached. Could not parse refined queries.[/red]")
            except Exception as e_team_run: 
                console.print(f"[red]Unexpected error during query_team.run or parsing on attempt {refinement_attempts}: {e_team_run}[/red]")
                refined_query_objects = []
                if refinement_attempts == MAX_REFINEMENT_ATTEMPTS:
                    console.print(f"[red]Max refinement attempts reached due to unexpected error.[/red]")
        
        if not refined_query_objects: # If loop finished without break (i.e. all attempts failed)
            console.print("[yellow]Falling back: Using original query as general query due to refinement failure.[/yellow]")
            refined_query_objects = [{"pubmed_query": "", "general_query": original_query if original_query else ""}]

        console.print(f"[secondary]Using Refined Query Objects:[/secondary] {refined_query_objects}")
        all_refined_queries.extend([obj.get("general_query", "") for obj in refined_query_objects])

        # Literature Search Loop (still within the outer try block)
        for j, query_obj in enumerate(refined_query_objects):
            pubmed_q = query_obj.get("pubmed_query", "")
            general_q = query_obj.get("general_query", "")

            if not pubmed_q and not general_q:
                console.print(f"[yellow]Skipping empty refined query object in query {j+1}.[/yellow]")
                continue
            
            console.print(f"[secondary]Calling search_literature with:[/secondary]")
            console.print(f"  [highlight]PubMed Query: '{pubmed_q}'[/highlight]")
            console.print(f"  [highlight]General Query: '{general_q}'[/highlight]")
            
            search_params = SearchLiteratureParams(
                pubmed_query=pubmed_q,
                general_query=general_q,
                max_results_per_source=max_results_to_use,
                publication_types=final_publication_types_to_use if final_publication_types_to_use else None
            )
            search_output = await literature_search_tool.run(
                args=search_params,
                cancellation_token=CancellationToken()
            )
            
            search_results_data = search_output.get("data", [])
            
            for record in search_results_data:
                # Ensure record is a dictionary and 'doi' exists
                if isinstance(record, dict):
                    doi_val = record.get('doi')
                    # Ensure doi_val is a string and not empty before adding to set and checking membership
                    if isinstance(doi_val, str) and doi_val.strip():
                        if doi_val not in processed_dois_for_articles:
                            all_articles_data.append(record)
                            processed_dois_for_articles.add(doi_val)
                    elif doi_val is None: # Handle articles with no DOI (add them if not otherwise identifiable as duplicate)
                        # This part is tricky without another unique ID. For now, let's assume if DOI is None, we add it.
                        # A more robust solution might involve checking title/authors if DOI is None.
                        # However, the current logic relies on DOI for deduplication.
                        # If we simply add all records with None DOI, we might get duplicates if other sources provide the same article without DOI.
                        # For now, let's stick to DOI-based deduplication. If DOI is None, it won't be added to processed_dois_for_articles
                        # and won't be processed by this specific 'if' block for deduplication.
                        # A simpler approach for now: if it has a DOI, deduplicate. If not, add it.
                        # This might lead to duplicates if an article appears from two sources, one with DOI and one without.
                        # The current structure of processed_dois_for_articles is for DOIs.
                        # Let's refine: only add if DOI is present and unique.
                        pass # Articles without a valid string DOI won't be added via this DOI-based deduplication.
                            # This means all_articles_data will only contain articles that had a valid, unique DOI.
                else:
                    console.print(f"[yellow]Warning: search_results_data contained a non-dictionary item: {type(record)}[/yellow]")

    except Exception as e: # OUTER EXCEPT, correctly aligned with the OUTER TRY
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
    query_team = None # Initialize query_team to None
    
    # Extract publication_type_mappings from settings_config for LiteratureSearchTool
    publication_type_mappings_from_settings = settings_config.get('publication_type_mappings', {})

    if agents_config_list:
        for agent_config_original in agents_config_list:
            if not isinstance(agent_config_original, dict) or "name" not in agent_config_original:
                # console.print(f"[yellow]Skipping invalid agent configuration: {agent_config_original}[/yellow]")
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
                    # Define the actual clarifier tool function here, so it has access to 'console'
                    async def actual_clarifier_tool_func(content: str) -> str:
                        # 'content' will be the question from the query_refiner's tool call
                        loop = asyncio.get_event_loop()
                        # 'console' is the Rich Console instance from the outer scope (main)
                        answer = await loop.run_in_executor(None, styled_input, f"{content}", console)
                        console.print(f"[debug]User answer via actual_clarifier_tool_func: {answer}[/debug]")
                        return answer

                    clarify_tool_for_agent = FunctionTool(
                        name="clarify_query", # Must match the name used in query_refiner's prompts/system_message
                        func=actual_clarifier_tool_func,
                        description="Ask a plain-language follow-up to the user to clarify their query. The 'content' parameter should be the question you want to ask the user.",
                        strict=True, 
                    )

                    component = AssistantAgent(
                        name="query_refiner",
                        model_client=model_client_instance,
                        tools=[clarify_tool_for_agent], # Use the new tool
                        description=agent_config_params.get("description"),
                        system_message=agent_config_params.get("system_message"),
                        reflect_on_tool_use=True, # IMPORTANT: Set to True for reflection
                        tool_call_summary_format="{result}", # This will be used by the reflection step if needed
                    )
                    
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
                        max_turns=1,  # stop immediately after the first agent turn
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
        agents, # Pass the whole agents dictionary
        original_query, 
        console,
        settings_config, 
        args 
    )


if __name__ == "__main__":
    asyncio.run(main())
