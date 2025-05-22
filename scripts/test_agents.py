from dotenv import load_dotenv
load_dotenv()           # now GOOGLE_API_KEY is in os.environ
# scripts/test_agents.py
import yaml
from pathlib import Path
from autogen_core import ComponentLoader  # new loader
# Load every stanza in the YAML and instantiate it
specs = yaml.safe_load(Path("config/agents.yaml").read_text())
agents = {
    s["name"]: ComponentLoader.load_component(s)      # <- magic happens here
    for s in specs
}
print("Loaded agents:", list(agents))
