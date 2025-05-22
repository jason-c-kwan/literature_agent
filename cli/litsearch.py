import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from autogen_core import ComponentLoader # Assuming this is the correct import path
import asyncio
import pdb

def main(base_path_str: str = "."):
    """
    Main function to load configurations, instantiate agents, and run a simple chat.
    """
    base_path = Path(base_path_str)
    env_file_path = base_path / ".env"
    agents_file_path = base_path / "config" / "agents.yaml"
    settings_file_path = base_path / "config" / "settings.yaml"

    # 1. Load .env, config/agents.yaml & config/settings.yaml
    load_dotenv(dotenv_path=env_file_path)

    if not agents_file_path.exists():
        print(f"Error: Agents configuration file not found at {agents_file_path}")
        return
    if not settings_file_path.exists():
        print(f"Error: Settings configuration file not found at {settings_file_path}")
        return

    with open(agents_file_path, "r") as f:
        agents_config_list = yaml.safe_load(f)
        print(f"Agents loaded: {agents_config_list}")

    with open(settings_file_path, "r") as f:
        settings_config = yaml.safe_load(f)
    # print(f"Settings loaded: {settings_config}") # Optional: for debugging

    # 2. Instantiate agents via autogen_core.ComponentLoader.load_component
    loader = ComponentLoader()
    agents = {}
    if agents_config_list:  # Ensure it's not None or empty
        for agent_config in agents_config_list:
            if not isinstance(agent_config, dict) or "name" not in agent_config:
                print(f"Skipping invalid agent configuration: {agent_config}")
                continue
            try:
                agents[agent_config["name"]] = loader.load_component(agent_config)
            except Exception as e:
                print(f"Error loading component {agent_config.get('name', 'Unknown')}: {e}")
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

    asyncio.run(hello_round_trip(user_proxy, assistant))

async def hello_round_trip(user, assistant):
    # send “Hello” to the assistant and wait for the TaskResult
    result = await assistant.run(task="Hello")          # single message
    print("Assistant reply →", result.messages[-1]["content"])

if __name__ == "__main__":
    # This allows running the script from the project root directory (e.g., literature_agent/)
    # where .env and config/ are expected.
    # If running from cli/ directory, main(".") would look for ./config, ./.env
    # To run from project root: python cli/litsearch.py
    # To run from cli dir: python litsearch.py (then base_path should be "..")
    # For simplicity, assume it's run from project root.
    main()
