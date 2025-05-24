from dotenv import load_dotenv
load_dotenv()           # now GOOGLE_API_KEY is in os.environ
# scripts/test_agents.py
import yaml
import copy # Added for deepcopy
from pathlib import Path
from autogen_core import ComponentLoader  # new loader

# Load every stanza in the YAML and instantiate it
specs = yaml.safe_load(Path("config/agents.yaml").read_text())
agents = {}
for s_spec in specs:
    agent_name = s_spec.get("name")
    if not agent_name:
        print(f"Warning: Skipping spec without a name: {s_spec}")
        continue

    current_spec_to_load = copy.deepcopy(s_spec) # Use a deep copy to avoid modifying original spec list

    if agent_name == "query_refiner":
        # For query_refiner, remove the 'tools' key from its config if it exists,
        # as it's dynamically added in cli/litsearch.py and not suitable for direct ComponentLoader.
        if "config" in current_spec_to_load and isinstance(current_spec_to_load["config"], dict):
            if "tools" in current_spec_to_load["config"]:
                del current_spec_to_load["config"]["tools"]
                print(f"Note: Removed 'tools' from '{agent_name}' config for this test load.")
    
    try:
        agents[agent_name] = ComponentLoader.load_component(current_spec_to_load) # <- magic happens here
    except Exception as e:
        print(f"Error loading component '{agent_name}': {e}")
        agents[agent_name] = None # Or handle error as appropriate

print("Loaded agents:", list(agents.keys())) # Print keys of the agents dict
