import os
import yaml
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from autogen_core import ComponentLoader, CancellationToken # Import CancellationToken
from tools.search import LiteratureSearchTool, SearchLiteratureParams
import asyncio
import pdb
import re # For parsing placeholders
import json

from rich.console import Console
from rich.theme import Theme

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
                return None # Or raise ValueError(f"Environment variable {var_name} not set for {value}")
    return value # Return original if not a string or no match

# Load custom rich theme
def load_rich_theme(base_path: Path) -> Theme:
    theme_file_path = base_path / "config" / "rich_theme.json"
    if theme_file_path.exists():
        with open(theme_file_path, "r") as f:
            theme_config = json.load(f)
        return Theme(theme_config)
    return Theme({}) # Return empty theme if not found

async def run_search_pipeline(query_refiner_agent, literature_search_tool, original_query: str, rounds: int, console: Console):
    all_dois = set()
    all_refined_queries = []

    for i in range(rounds):
        console.print(f"[primary]--- Refining Query Round {i+1}/{rounds} ---[/primary]")
        
        # 2. Call the Query-RefinerAgent to clarify ambiguities and expand the query
        console.print(f"[secondary]Input to Query-RefinerAgent:[/secondary] [highlight]'{original_query}'[/highlight]")
        
        # The Query-RefinerAgent is expected to return a list of three Boolean/MeSH-style search strings.
        # Assuming the agent's 'run' method returns a TaskResult with messages, and the last message content
        # can be parsed as a list of strings. This might need adjustment based on actual agent output format.
        try:
            refiner_result = await query_refiner_agent.run(task=f"Refine the following research query into three distinct Boolean/MeSH-style search strings: {original_query}")
            
            refined_queries_str = ""
            if refiner_result.messages and isinstance(refiner_result.messages, list) and len(refiner_result.messages) > 0:
                last_message = refiner_result.messages[-1]
                if hasattr(last_message, 'content'):
                    refined_queries_str = last_message.content
                else:
                    console.print(f"[red]Warning: Query-RefinerAgent returned unexpected message format: {last_message}[/red]")
            
            # Attempt to parse the refined queries. This is a placeholder and might need robust parsing.
            # For now, assuming the agent returns a string that can be split into lines, or a JSON string.
            # A more robust solution would involve the agent returning a structured JSON.
            # For the purpose of this task, let's assume it returns a comma-separated string or similar.
            # If the agent is designed to return a JSON array string, json.loads would be appropriate.
            # For now, a simple split for demonstration.
            
            # Example: Agent returns "query1, query2, query3" or "['query1', 'query2', 'query3']"
            # Let's assume it returns a string that can be evaluated as a list or is comma-separated.
            # A safer approach would be to define a specific output format for the agent.
            
            # For now, let's assume the agent returns a string like "['query1', 'query2', 'query3']"
            # or a simple string that we can split.
            
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
                        break # Found the JSON block
                    except json.JSONDecodeError:
                        continue # Not a valid JSON block, try next

                if not extracted_json_str:
                    raise ValueError("No valid JSON code block found in agent's response.")

                refined_query_objects = json.loads(extracted_json_str)
                if not isinstance(refined_query_objects, list):
                    raise ValueError("Agent did not return a JSON array.")
                
                # Ensure each object has 'pubmed_query' and 'general_query'
                for obj in refined_query_objects:
                    if not isinstance(obj, dict) or "pubmed_query" not in obj or "general_query" not in obj:
                        raise ValueError("Each object in the JSON array must have 'pubmed_query' and 'general_query' fields.")

                # Take up to 'rounds' number of refined query objects
                refined_query_objects = refined_query_objects[:rounds]
                
            except (json.JSONDecodeError, ValueError) as e:
                console.print(f"[red]Error parsing Query-RefinerAgent output: {e}. Raw output: {refined_queries_str}[/red]")
                console.print("[yellow]Attempting fallback parsing (may not be accurate).[/yellow]")
                # Fallback parsing if JSON parsing fails (less robust, but prevents crash)
                refined_query_objects = []
                # This fallback is a heuristic and might not perfectly align with the new structured output.
                # It's primarily to prevent a crash if the agent deviates from the expected JSON.
                # A better long-term solution is to ensure the agent strictly adheres to the JSON format.
                lines = [line.strip() for line in refined_queries_str.split('\n') if line.strip()]
                for line in lines:
                    # Simple heuristic: if a line contains "pubmed_query" or "general_query", try to extract.
                    # This is very fragile and mostly for graceful degradation.
                    if "pubmed_query" in line or "general_query" in line:
                        # Attempt to extract key-value pairs, e.g., from "pubmed_query: 'term'"
                        match_pubmed = re.search(r"pubmed_query:\s*['\"](.*?)['\"]", line)
                        match_general = re.search(r"general_query:\s*['\"](.*?)['\"]", line)
                        
                        pq = match_pubmed.group(1) if match_pubmed else ""
                        gq = match_general.group(1) if match_general else ""
                        
                        if pq or gq:
                            refined_query_objects.append({"pubmed_query": pq, "general_query": gq})
                
                # Ensure we have 'rounds' number of query objects, padding if necessary
                while len(refined_query_objects) < rounds:
                    refined_query_objects.append({"pubmed_query": "", "general_query": ""})
                refined_query_objects = refined_query_objects[:rounds] # Trim if too many from fallback

            console.print(f"[secondary]Refined Query Objects from Query-RefinerAgent:[/secondary] {refined_query_objects}")
            # Store the general queries for the final output JSON
            all_refined_queries.extend([obj["general_query"] for obj in refined_query_objects])

            # 3. For each refined query object, call the `search_literature` tool
            for j, query_obj in enumerate(refined_query_objects):
                pubmed_q = query_obj.get("pubmed_query", "")
                general_q = query_obj.get("general_query", "")

                if not pubmed_q and not general_q:
                    console.print(f"[yellow]Skipping empty refined query object in round {i+1}, query {j+1}.[/yellow]")
                    continue
                
                console.print(f"[secondary]Calling search_literature with:[/secondary]")
                console.print(f"  [highlight]PubMed Query: '{pubmed_q}'[/highlight]")
                console.print(f"  [highlight]General Query: '{general_q}'[/highlight]")
                
                # Call the run method of LiteratureSearchTool with structured queries
                search_output = await literature_search_tool.run(
                    args=SearchLiteratureParams(pubmed_query=pubmed_q, general_query=general_q),
                    cancellation_token=CancellationToken()
                )
                
                # Extract the 'data' field from the new structured output
                search_results_data = search_output.get("data", [])
                
                # 4. Extract the DOIs from all results
                for record in search_results_data:
                    if 'doi' in record and record['doi']:
                        all_dois.add(record['doi'])
        except Exception as e:
            console.print(f"[red]Error during query refinement or search in round {i+1}: {e}[/red]")
            break

    # Deduplicate DOIs (already handled by using a set)
    unique_dois = sorted(list(all_dois)) # Sort for consistent output

    # Output a single JSON array named `dois.json`
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
    
    return output_data # Return for testing purposes

def main():
    """
    Main function to load configurations, instantiate agents, and run the search pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the literature search pipeline.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds for query refinement and search.")
    args = parser.parse_args()

    base_path = Path(".").resolve() # Resolve to absolute path for clarity
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
        dotenv_loaded = load_dotenv(dotenv_path=env_file_path, override=True) # Added override=True
        console.print(f"[primary]load_dotenv result: {dotenv_loaded}[/primary]")
    else:
        console.print(f"[yellow].env file NOT found at: {env_file_path}[/yellow]")
        dotenv_loaded = False

    # console.print(f"OPENAI_API_BASE after load_dotenv: {os.getenv('OPENAI_API_BASE')}")
    # console.print(f"OPENAI_API_KEY after load_dotenv (first 5 chars): {os.getenv('OPENAI_API_KEY')[:5] if os.getenv('OPENAI_API_KEY') else None}")


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
    # console.print(f"Settings loaded: {settings_config}") # Optional: for debugging

    # 2. Instantiate agents via autogen_core.ComponentLoader.load_component
    loader = ComponentLoader()
    agents = {}
    if agents_config_list:  # Ensure it's not None or empty
        for agent_config_original in agents_config_list:
            if not isinstance(agent_config_original, dict) or "name" not in agent_config_original:
                console.print(f"[yellow]Skipping invalid agent configuration: {agent_config_original}[/yellow]")
                continue

            agent_config = agent_config_original.copy()
            agent_name = agent_config.get('name', 'Unknown')
            
            if "config" in agent_config and "model_client" in agent_config["config"]:
                model_client_config = agent_config["config"]["model_client"].get("config")
                if model_client_config:
                    console.print(f"[primary]Resolving env vars for {agent_name}'s model_client...[/primary]")
                    if "model" in model_client_config:
                        model_client_config["model"] = resolve_env_placeholder(model_client_config["model"])
                        console.print(f"  [primary]Resolved model to: {model_client_config['model']}[/primary]")
                    if "base_url" in model_client_config:
                        model_client_config["base_url"] = resolve_env_placeholder(model_client_config["base_url"])
                        console.print(f"  [primary]Resolved base_url to: {model_client_config['base_url']}[/primary]")
                    if "api_key" in model_client_config:
                        model_client_config["api_key"] = resolve_env_placeholder(model_client_config["api_key"])
                        console.print(f"  [primary]Resolved api_key (is set: {model_client_config['api_key'] is not None})[/primary]")

            console.print(f"[primary]Attempting to load component: {agent_name} with resolved config[/primary]")
            try:
                agents[agent_config["name"]] = loader.load_component(agent_config)
            except Exception as e:
                console.print(f"[red]Error loading component {agent_name}: {e}[/red]")
                agents[agent_config["name"]] = None
    else:
        console.print("[yellow]Warning: agents_config_list is empty or None.[/yellow]")
        return

    query_refiner_agent = agents.get("query_refiner")
    literature_search_tool = LiteratureSearchTool() # Instantiate the tool

    if not query_refiner_agent:
        console.print("[red]Error: 'query_refiner' agent not found or failed to load.[/red]")
        return
    

    # 1. Read a free-text search query from the user.
    original_query = console.input("[primary]What would you like to research?[/primary] ")
    if not original_query:
        console.print("[yellow]No query entered. Exiting.[/yellow]")
        return

    asyncio.run(run_search_pipeline(query_refiner_agent, literature_search_tool, original_query, args.rounds, console))


if __name__ == "__main__":
    main()
