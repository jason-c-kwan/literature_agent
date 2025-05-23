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
from typing import Sequence, Any, Optional, List, Union, Type # Added for DebugClient
from pydantic import BaseModel # Added for DebugClient
from tools.triage import TriageAgent
from tools.ranking import RankerAgent
from tools.export import ExporterAgent
from tools._base import StubConfig # Added import
import asyncio
import re
import json
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

async def run_search_pipeline(query_team: BaseGroupChat, literature_search_tool: LiteratureSearchTool, original_query: str, console: Console):
    all_dois = set()
    all_refined_queries = []

    # 2. Call the query_team (GroupChat) to clarify ambiguities and expand the query
    console.print(f"[secondary]Initiating research query with GroupChat:[/secondary] [highlight]'{original_query}'[/highlight]")
    
    refined_queries_str = ""
    
    try:
        # Single run call; clarifications will be handled internally by query_refiner using its new tool
        team_result = await query_team.run(task=original_query)
        console.print(f"[debug]Team run completed. Messages: {team_result.messages}[/debug]")

        # Extract refined_queries_str from the last message of query_refiner
        # The query_refiner, with reflect_on_tool_use=True, should output a TextMessage with the JSON.
        if team_result.messages and isinstance(team_result.messages, list) and len(team_result.messages) > 0:
            for msg in reversed(team_result.messages):
                if msg.source == "query_refiner" and (isinstance(msg, TextMessage) or isinstance(msg, ToolCallSummaryMessage)):
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        # Attempt to find a JSON code block
                        code_block_pattern = re.compile(r"```(?:json|python|text)?\n(.*?)\n```", re.DOTALL)
                        match = code_block_pattern.search(msg.content)
                        if match:
                            extracted_json_content = match.group(1).strip()
                            # Verify if this extracted content is indeed a JSON array
                            if extracted_json_content.startswith("[") and extracted_json_content.endswith("]"):
                                refined_queries_str = extracted_json_content # Store only the JSON part
                                console.print(f"[debug]refined_queries_str set from {msg.type} by {msg.source} (extracted from code block): {refined_queries_str}[/debug]")
                                break
                        elif msg.content.strip().startswith("[") and msg.content.strip().endswith("]"): 
                            # Fallback for plain JSON string without markdown fences
                            refined_queries_str = msg.content.strip()
                            console.print(f"[debug]refined_queries_str set from {msg.type} by {msg.source} (plain JSON string): {refined_queries_str}[/debug]")
                            break
        
        if not refined_queries_str:
            console.print(f"[red]Warning: Could not find valid JSON refined_queries_str from query_refiner in team_result.messages. Last relevant message from query_refiner: {next((m for m in reversed(team_result.messages) if m.source == 'query_refiner'), 'No message from query_refiner')}[/red]")

        console.print(f"[debug]Attempting to parse refined_queries_str: {refined_queries_str}[/debug]")
        refined_query_objects = []
        if refined_queries_str:
            # Attempt to parse the refined queries.
            # The refined_queries_str should now be just the JSON array string.
            # The agent is expected to return a JSON array of objects with 'pubmed_query' and 'general_query' fields.
            try:
                # refined_queries_str is already the extracted JSON string or plain JSON
                refined_query_objects = json.loads(refined_queries_str)
                if not isinstance(refined_query_objects, list):
                    raise ValueError("Agent did not return a JSON array.")
                
                # Ensure each object has 'pubmed_query' and 'general_query'
                for obj in refined_query_objects:
                    if not isinstance(obj, dict) or "pubmed_query" not in obj or "general_query" not in obj:
                        raise ValueError("Each object in the JSON array must have 'pubmed_query' and 'general_query' fields.")

                # The query_refiner is specified to return exactly 3 JSON objects.
                if len(refined_query_objects) != 3:
                    console.print(f"[yellow]Warning: Expected 3 refined query objects, but got {len(refined_query_objects)}.[/yellow]")
                
            except (json.JSONDecodeError, ValueError) as e:
                console.print(f"[red]Error parsing Query-RefinerAgent output: {e}. Raw output: {refined_queries_str}[/red]")
                console.print("[yellow]Attempting fallback parsing (may not be accurate).[/yellow]")
                # Fallback parsing if JSON parsing fails (less robust, but prevents crash)
                refined_query_objects = []
                lines = [line.strip() for line in refined_queries_str.split('\n') if line.strip()]
                for line in lines:
                    if "pubmed_query" in line or "general_query" in line:
                        match_pubmed = re.search(r"pubmed_query:\s*['\"](.*?)['\"]", line)
                        match_general = re.search(r"general_query:\s*['\"](.*?)['\"]", line)
                        
                        pq = match_pubmed.group(1) if match_pubmed else ""
                        gq = match_general.group(1) if match_general else ""
                        
                        if pq or gq:
                            refined_query_objects.append({"pubmed_query": pq, "general_query": gq})
                
                if not refined_query_objects:
                    refined_query_objects.append({"pubmed_query": "", "general_query": ""})
                # If fallback yields more than 3, take the first 3. If less, use what we have.
                refined_query_objects = refined_query_objects[:3] 

        console.print(f"[secondary]Refined Query Objects from Query Team:[/secondary] {refined_query_objects}")
        all_refined_queries.extend([obj["general_query"] for obj in refined_query_objects])

        # 3. For each refined query object, call the `search_literature` tool
        for j, query_obj in enumerate(refined_query_objects):
            pubmed_q = query_obj.get("pubmed_query", "")
            general_q = query_obj.get("general_query", "")

            if not pubmed_q and not general_q:
                console.print(f"[yellow]Skipping empty refined query object in query {j+1}.[/yellow]")
                continue
            
            console.print(f"[secondary]Calling search_literature with:[/secondary]")
            console.print(f"  [highlight]PubMed Query: '{pubmed_q}'[/highlight]")
            console.print(f"  [highlight]General Query: '{general_q}'[/highlight]") # Fixed variable name here
            
            search_output = await literature_search_tool.run(
                args=SearchLiteratureParams(pubmed_query=pubmed_q, general_query=general_q),
                cancellation_token=CancellationToken()
            )
            
            search_results_data = search_output.get("data", [])
            
            # 4. Extract the DOIs from all results
            for record in search_results_data:
                if 'doi' in record and record['doi']:
                    all_dois.add(record['doi'])
    except Exception as e:
        console.print(f"[red]Error during query refinement or search: {e}[/red]")

    unique_dois = sorted(list(all_dois))

    output_data = {
        "query": original_query,
        "refined_queries": all_refined_queries,
        "dois": unique_dois
    }
    
    output_file_path = Path("dois.json")
    with open(output_file_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"[primary]Search pipeline completed. Results saved to {output_file_path}[/primary]")
    console.print(f"[secondary]Total unique DOIs found:[/secondary] [highlight]{len(unique_dois)}[/highlight]")
    
    return output_data

async def main():
    """
    Main asynchronous function to load configurations, instantiate agents, and run the search pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the literature search pipeline.")
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

    # 2. Manually instantiate agents and teams
    agents = {}
    query_team = None # Initialize query_team to None

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
                    component = LiteratureSearchTool(**filtered_agent_config_params) # Also filter for tools
                else:
                    # console.print(f"[yellow]Warning: Unhandled component type for {agent_name}. Skipping.[/yellow]")
                    pass

                if component:
                    agents[agent_name] = component
                    if agent_name == "query_team":
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

    literature_search_tool = LiteratureSearchTool()

    if not query_team:
        console.print("[red]Error: 'query_team' not found or failed to load.[/red]")
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

    await run_search_pipeline(query_team, literature_search_tool, original_query, console)


if __name__ == "__main__":
    asyncio.run(main())
