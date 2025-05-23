import os
import yaml
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import TerminationCondition
from tools.search import LiteratureSearchTool, SearchLiteratureParams
from tools.triage import TriageAgent
from tools.ranking import RankerAgent
from tools.export import ExporterAgent
from tools._base import StubConfig # Added import
import asyncio
import pdb
import re
import json

from rich.console import Console
from rich.theme import Theme

# Import GroupChat for the new query logic
from autogen_agentchat.teams import BaseGroupChat
#from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.messages import ToolCallRequestEvent

from autogen_core.tools import FunctionTool

def clarifier_stub(content: str) -> None:
    """No-op placeholder; we’ll catch the callEvent ourselves."""
    return None

clarify_tool = FunctionTool(
    name="clarify_query",
    func=clarifier_stub,
    description="Ask a plain-language follow-up",
    strict=True,
)



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

async def run_search_pipeline(query_team: BaseGroupChat, literature_search_tool: LiteratureSearchTool, original_query: str, console: Console):
    all_dois = set()
    all_refined_queries = []

    # 2. Call the query_team (GroupChat) to clarify ambiguities and expand the query
    console.print(f"[secondary]Initiating research query with GroupChat:[/secondary] [highlight]'{original_query}'[/highlight]")
    
    MAX_CLARIFICATION_TURNS = 3
    clarification_turn_count = 0
    refined_queries_str = ""
    
    try:
        team_result = await query_team.run(task=original_query)

        while clarification_turn_count < MAX_CLARIFICATION_TURNS:
            # 1) Did it ask for clarification?
            stub_events = [
                m for m in team_result.messages
                if isinstance(m, ToolCallRequestEvent)
                and m.content and m.content[0].name == "clarify_query"
            ]
            
            if stub_events:
                clarification_turn_count += 1
                fcall = stub_events[-1].content[0]
                params = json.loads(fcall.arguments)
                question = params.get("content", "Could you clarify?")
                answer = console.input(f"{question} ")

                await query_team.reset()
                team_result = await query_team.run(task=answer)
                continue # Continue loop for next turn or JSON check
            else:
                # If no clarification asked, check for JSON output
                json_message = None
                for m in team_result.messages:
                    if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip().startswith("["):
                        json_message = m.content
                        break
                
                if json_message:
                    refined_queries_str = json_message
                    break # Exit loop if JSON found
                else:
                    # If no stub and no JSON, abort loop
                    console.print("[yellow]Query team did not ask for clarification or provide JSON. Aborting clarification loop.[/yellow]")
                    break

        # After the loop, try to get the refined queries string
        if not refined_queries_str and team_result.messages and isinstance(team_result.messages, list) and len(team_result.messages) > 0:
            last_message = team_result.messages[-1]
            if hasattr(last_message, 'content'):
                refined_queries_str = last_message.content
            else:
                console.print(f"[red]Warning: Query team returned unexpected message format: {last_message}[/red]")
        
        refined_query_objects = []
        if refined_queries_str:
            # Attempt to parse the refined queries.
            # The agent is expected to return a JSON array of objects with 'pubmed_query' and 'general_query' fields.
            try:
                # Extract content from markdown code blocks first
                code_block_pattern = re.compile(r"```(?:json|python|text)?\n(.*?)\n```", re.DOTALL)
                extracted_json_str = ""
                for block in code_block_pattern.findall(refined_queries_str):
                    try:
                        # Try to load as JSON. If successful, this is our block.
                        json.loads(block)
                        extracted_json_str = block
                        break
                    except json.JSONDecodeError:
                        continue

                if not extracted_json_str:
                    raise ValueError("No valid JSON code block found in agent's response.")

                refined_query_objects = json.loads(extracted_json_str)
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
            console.print(f"  [highlight]General Query: '{general_q}'[/highlight]")
            
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

def main():
    """
    Main function to load configurations, instantiate agents, and run the search pipeline.
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
    console.print(f"[primary]Attempting to load .env file from: {env_file_path}[/primary]")
    if env_file_path.exists():
        console.print(f"[primary].env file found at: {env_file_path}[/primary]")
        dotenv_loaded = load_dotenv(dotenv_path=env_file_path, override=True)
        console.print(f"[primary]load_dotenv result: {dotenv_loaded}[/primary]")
    else:
        console.print(f"[yellow].env file NOT found at: {env_file_path}[/yellow]")
        dotenv_loaded = False

    if not agents_file_path.exists():
        console.print(f"[red]Error: Agents configuration file not found at {agents_file_path}[/red]")
        return
    if not settings_file_path.exists():
        console.print(f"[red]Error: Settings configuration file not found at {settings_file_path}[/red]")
        return

    with open(agents_file_path, "r") as f:
        agents_config_list = yaml.safe_load(f)
        console.print(f"[primary]Agents loaded (raw from YAML): {agents_config_list}[/primary]")


    with open(settings_file_path, "r") as f:
        settings_config = yaml.safe_load(f)

    # 2. Manually instantiate agents and teams
    agents = {}
    query_team = None # Initialize query_team to None

    if agents_config_list:
        for agent_config_original in agents_config_list:
            if not isinstance(agent_config_original, dict) or "name" not in agent_config_original:
                console.print(f"[yellow]Skipping invalid agent configuration: {agent_config_original}[/yellow]")
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
                    model_client_instance = OpenAIChatCompletionClient(**model_client_config_dict)
                    console.print(f"  [primary]Manually instantiated model_client for {agent_name}.[/primary]")
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
                    component = AssistantAgent(
                        name="query_refiner",
                        model_client=model_client_instance,
                        tools=[clarify_tool],            # our stub
                        description=agent_config_params.get("description"),
                        system_message=agent_config_params.get("system_message"),
                        reflect_on_tool_use=False,
                        tool_call_summary_format="{result}",
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
                    console.print(f"[yellow]Warning: Unhandled component type for {agent_name}. Skipping.[/yellow]")

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
            console.print(f"[debug] query_refiner._handoffs = {refiner._handoffs!r}")
            # And any FunctionTools or other tools show up under `_tool_configs`:
            console.print(f"[debug] query_refiner._tool_configs = {getattr(refiner, '_tool_configs', None)!r}")

            
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
    original_query = console.input("[primary]What would you like to research?[/primary] ")
    if not original_query:
        console.print("[yellow]No query entered. Exiting.[/yellow]")
        return

    asyncio.run(run_search_pipeline(query_team, literature_search_tool, original_query, console))


if __name__ == "__main__":
    main()
