import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import MagicMock, call # For more detailed call assertions if needed

# Assuming tests are run from the project root, so cli.litsearch is importable
# Ensure __init__.py exists in cli/ and tests/ if not already present and needed for older Pythons
# For modern pytest, direct import should work if project root is in sys.path
try:
    from cli.litsearch import main as litsearch_main
    from autogen_core import ComponentLoader # To patch it correctly
except ImportError as e:
    # This might happen if PYTHONPATH is not set up correctly for tests
    # or if __init__.py files are missing in relevant directories for older Python versions.
    print(f"ImportError in test setup: {e}. Ensure cli/litsearch.py is accessible.")
    litsearch_main = None
    ComponentLoader = None


# Helper to create dummy config files
def create_dummy_configs(tmp_path: Path, agents_content=None, settings_content=None, env_content=None):
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)

    if agents_content is None:
        agents_content = [
            {"name": "user_proxy", "provider": "mock.UserProxy", "component_type": "agent", "config": {"name": "user_proxy"}},
            {"name": "query_refiner", "provider": "mock.Assistant", "component_type": "agent", "config": {"name": "query_refiner"}},
        ]
    (config_dir / "agents.yaml").write_text(yaml.dump(agents_content))

    if settings_content is None:
        settings_content = {"version": "0.1-test"}
    (config_dir / "settings.yaml").write_text(yaml.dump(settings_content))

    if env_content is not None:
        (tmp_path / ".env").write_text(env_content)


@pytest.mark.skipif(litsearch_main is None, reason="cli.litsearch.main could not be imported")
def test_dotenv_loading(monkeypatch, tmp_path):
    """Asserts .env variables are loaded."""
    sentinel_key = "TEST_SENTINEL_VAR_FROM_FILE"
    sentinel_value = "loaded_successfully"
    env_content = f"{sentinel_key}={sentinel_value}\n"
    
    create_dummy_configs(tmp_path, env_content=env_content)

    # Ensure the variable is not already in the environment
    monkeypatch.delenv(sentinel_key, raising=False)
    
    # Call the main function of the script, pointing to the temp directory
    litsearch_main(base_path_str=str(tmp_path))
    
    assert os.getenv(sentinel_key) == sentinel_value

@pytest.mark.skipif(litsearch_main is None or ComponentLoader is None, reason="Imports failed")
def test_component_loader_mocking(mocker, tmp_path):
    """Mocks ComponentLoader.load_component."""
    dummy_agents_data = [
        {"name": "agent_one", "provider": "provider.One", "component_type": "agent", "config": {"id": 1}},
        {"name": "agent_two", "provider": "provider.Two", "component_type": "agent", "config": {"id": 2}},
    ]
    create_dummy_configs(tmp_path, agents_content=dummy_agents_data)

    # Patch load_component on the ComponentLoader class from autogen_core
    # This will affect instances of ComponentLoader created in litsearch_main
    mocked_load_component_method = mocker.patch.object(ComponentLoader, "load_component", return_value=MagicMock())
    # Alternative if the above doesn't work as expected due to import nuances:
    # mocked_load_component_method = mocker.patch("autogen_core.ComponentLoader.load_component", return_value=MagicMock())
    # Or even more specific if litsearch.py imports it as `from autogen_core import ComponentLoader`:
    # mocked_load_component_method = mocker.patch("cli.litsearch.ComponentLoader.load_component", return_value=MagicMock())
    # Sticking with patch.object for now as it's often robust for class methods.

    litsearch_main(base_path_str=str(tmp_path))

    assert mocked_load_component_method.call_count == len(dummy_agents_data)
    
    expected_calls = []
    for agent_cfg in dummy_agents_data:
        expected_calls.append(
            call(
                provider=agent_cfg["provider"],
                component_type=agent_cfg["component_type"],
                config=agent_cfg["config"]
            )
        )
    # Check if all expected calls were made, regardless of order for any_call
    # For specific order, use `mocked_load_component_method.assert_has_calls(expected_calls, any_order=False)`
    # For checking presence of each call:
    for expected_call_args in expected_calls:
         mocked_load_component_method.assert_any_call(*expected_call_args.args, **expected_call_args.kwargs)


@pytest.mark.skipif(litsearch_main is None or ComponentLoader is None, reason="Imports failed")
def test_script_output_captures_reply(capsys, mocker, tmp_path):
    """Confirms the script prints the assistant’s reply."""
    user_proxy_name = "user_proxy"
    assistant_name = "query_refiner" # Matches the default assistant litsearch.py tries to get
    expected_reply_text = "Mocked assistant reply for Hello"

    agents_config_for_test = [
        {"name": user_proxy_name, "provider": "mock.UserProxy", "component_type": "agent", "config": {"name": user_proxy_name}},
        {"name": assistant_name, "provider": "mock.Assistant", "component_type": "agent", "config": {"name": assistant_name}},
    ]
    create_dummy_configs(tmp_path, agents_content=agents_config_for_test)

    mock_user_proxy_agent = MagicMock(name="MockUserProxyAgent")
    mock_assistant_agent = MagicMock(name="MockAssistantAgent")

    # Configure mock_user_proxy_agent.initiate_chat
    # It needs to populate its own chat_messages attribute as the script expects
    def mock_initiate_chat_impl(recipient, message):
        if message == "Hello" and recipient == mock_assistant_agent:
            # Simulate storing the message history
            # The script expects user_proxy.chat_messages[assistant][-1]['content']
            if not hasattr(mock_user_proxy_agent, 'chat_messages') or not isinstance(mock_user_proxy_agent.chat_messages, dict):
                 mock_user_proxy_agent.chat_messages = {} # Ensure it's a dict

            mock_user_proxy_agent.chat_messages[mock_assistant_agent] = [
                {"role": "assistant", "content": expected_reply_text} # Assistant's reply
            ]
        # else:
            # print(f"Unexpected chat: to {recipient} msg: {message}")


    mock_user_proxy_agent.initiate_chat = MagicMock(side_effect=mock_initiate_chat_impl)
    
    # Patch ComponentLoader.load_component to return our mocks
    def side_effect_load_component(provider, component_type, config):
        if config.get("name") == user_proxy_name:
            return mock_user_proxy_agent
        elif config.get("name") == assistant_name:
            return mock_assistant_agent
        return MagicMock() # Default mock for any other agents

    # Using patch.object on the ComponentLoader class from autogen_core
    mocker.patch.object(ComponentLoader, "load_component", side_effect=side_effect_load_component)
    # As above, consider "autogen_core.ComponentLoader.load_component" or "cli.litsearch.ComponentLoader.load_component"
    # if patch.object doesn't behave as expected.

    litsearch_main(base_path_str=str(tmp_path))

    captured = capsys.readouterr()
    
    # Check that initiate_chat was called correctly
    mock_user_proxy_agent.initiate_chat.assert_called_once_with(
        recipient=mock_assistant_agent,
        message="Hello"
    )
    
    # Check that the expected reply is in the script's output
    assert f"Assistant's final reply: {expected_reply_text}" in captured.out

# To run these tests, navigate to the project root and run:
# pytest
# or
# pytest tests/test_litsearch.py
