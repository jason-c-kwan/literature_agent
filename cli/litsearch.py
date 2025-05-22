import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from autogen_core import ComponentLoader # Assuming this is the correct import path
import asyncio
import pdb
import re # For parsing placeholders

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

def main(base_path_str: str = "."):
    """
    Main function to load configurations, instantiate agents, and run a simple chat.
    """
    base_path = Path(base_path_str).resolve() # Resolve to absolute path for clarity
    env_file_path = base_path / ".env"
    agents_file_path = base_path / "config" / "agents.yaml"
    settings_file_path = base_path / "config" / "settings.yaml"

    # 1. Load .env, config/agents.yaml & config/settings.yaml
    print(f"Attempting to load .env file from: {env_file_path}")
    if env_file_path.exists():
        print(f".env file found at: {env_file_path}")
        dotenv_loaded = load_dotenv(dotenv_path=env_file_path, override=True) # Added override=True
        print(f"load_dotenv result: {dotenv_loaded}")
    else:
        print(f".env file NOT found at: {env_file_path}")
        dotenv_loaded = False

    print(f"OPENAI_API_BASE after load_dotenv: {os.getenv('OPENAI_API_BASE')}")
    print(f"OPENAI_API_KEY after load_dotenv (first 5 chars): {os.getenv('OPENAI_API_KEY')[:5] if os.getenv('OPENAI_API_KEY') else None}")


    if not agents_file_path.exists():
        print(f"Error: Agents configuration file not found at {agents_file_path}")
        return
    if not settings_file_path.exists():
        print(f"Error: Settings configuration file not found at {settings_file_path}")
        return

    with open(agents_file_path, "r") as f:
        agents_config_list = yaml.safe_load(f)
        # This print is already in your original error, good for comparison
        print(f"Agents loaded (raw from YAML): {agents_config_list}")


    with open(settings_file_path, "r") as f:
        settings_config = yaml.safe_load(f)
    # print(f"Settings loaded: {settings_config}") # Optional: for debugging

    # 2. Instantiate agents via autogen_core.ComponentLoader.load_component
    loader = ComponentLoader()
    agents = {}
    if agents_config_list:  # Ensure it's not None or empty
        for agent_config_original in agents_config_list:
            if not isinstance(agent_config_original, dict) or "name" not in agent_config_original:
                print(f"Skipping invalid agent configuration: {agent_config_original}")
                continue

            # Deep copy the config to modify it, or modify in place if that's acceptable
            agent_config = agent_config_original.copy() # Shallow copy is enough if we only modify nested dicts by replacing them
            
            agent_name = agent_config.get('name', 'Unknown')
            
            # Manually resolve environment variables for model_client config
            if "config" in agent_config and "model_client" in agent_config["config"]:
                model_client_config = agent_config["config"]["model_client"].get("config")
                if model_client_config:
                    print(f"Resolving env vars for {agent_name}'s model_client...")
                    if "model" in model_client_config:
                        model_client_config["model"] = resolve_env_placeholder(model_client_config["model"])
                        print(f"  Resolved model to: {model_client_config['model']}")
                    if "base_url" in model_client_config:
                        model_client_config["base_url"] = resolve_env_placeholder(model_client_config["base_url"])
                        print(f"  Resolved base_url to: {model_client_config['base_url']}")
                    if "api_key" in model_client_config:
                        model_client_config["api_key"] = resolve_env_placeholder(model_client_config["api_key"])
                        # Avoid printing full API key, even if resolved
                        print(f"  Resolved api_key (is set: {model_client_config['api_key'] is not None})")


            print(f"Attempting to load component: {agent_name} with resolved config")
            # The diagnostic prints for OPENAI_API_BASE/KEY before loading are less critical now,
            # but can be kept for sanity checking os.environ if needed.
            # if "model_client" in agent_config.get("config", {}):
            #      print(f"  OPENAI_API_BASE (os.environ) before loading {agent_name}: {os.getenv('OPENAI_API_BASE')}")
            #      print(f"  OPENAI_API_KEY (os.environ) before loading {agent_name} (first 5 chars): {os.getenv('OPENAI_API_KEY')[:5] if os.getenv('OPENAI_API_KEY') else None}")

            try:
                agents[agent_config["name"]] = loader.load_component(agent_config) # Use the modified agent_config
            except Exception as e:
                print(f"Error loading component {agent_name}: {e}")
                # pdb.set_trace() # Optional: uncomment for interactive debugging if error persists
                agents[agent_config["name"]] = None # Or skip adding it
    else:
        print("Warning: agents_config_list is empty or None.")
        return

    user_proxy = agents.get("user_proxy")
    assistant = agents.get("query_refiner")

    if not assistant:  # Fallback if query_refiner is not found
        for name, agent_instance in agents.items():
            if name != "user_proxy" and agent_instance is not None:
                # A more robust check would be isinstance(agent_instance, BaseAssistantAgent)
                # For now, picking the first non-user_proxy, non-None agent.
                assistant = agent_instance
                print(f"Using fallback assistant: {name}")
                break
    
    if not user_proxy:
        print("Error: user_proxy agent not found or failed to load.")
        return
    if not assistant:
        print("Error: No suitable assistant agent found or failed to load (e.g., query_refiner).")
        return

    print("Running hello_round_trip...")
    asyncio.run(hello_round_trip(user_proxy, assistant))

async def hello_round_trip(user, assistant):
    # send “Hello” to the assistant and wait for the TaskResult
    print("Assistant about to run...")
    result = await assistant.run(task="Hello")          # single message
    print("Assistant run completed.")
    if result.messages and isinstance(result.messages, list) and len(result.messages) > 0:
        last_message = result.messages[-1]
        # Assuming last_message is an object with a 'content' attribute
        # (e.g., autogen_core.TextMessage)
        if hasattr(last_message, 'content'):
            print("Assistant reply →", last_message.content)
        else:
            print("Assistant reply (unknown format) →", last_message)
    else:
        print("No messages found in assistant's result or result format is unexpected.")
        print("Full result object:", result)


if __name__ == "__main__":
    # This allows running the script from the project root directory (e.g., literature_agent/)
    # where .env and config/ are expected.
    # If running from cli/ directory, main(".") would look for ./config, ./.env
    # To run from project root: python cli/litsearch.py
    # To run from cli dir: python litsearch.py (then base_path should be "..")
    # For simplicity, assume it's run from project root.
    main()
