import pytest
import os
import asyncio 
import yaml
import json
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, call, AsyncMock 
from rich.console import Console
from rich.theme import Theme
from dotenv import load_dotenv 
import sys 

sys.path.append('/Users/jkwan2/miniconda3/envs/litagent/lib/python3.10/site-packages/')

try:
    from cli.litsearch import main as litsearch_main_coro, run_search_pipeline, load_rich_theme
    from autogen_core import ComponentLoader, CancellationToken
    from tools.search import LiteratureSearchTool, SearchLiteratureParams
    from tools.triage import TriageAgent
    from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage
    import autogen_agentchat.agents as agents_module
    from autogen_agentchat.teams import BaseGroupChat # Added for spec
except ImportError as e:
    print(f"ImportError in test setup: {e}. Ensure cli/litsearch.py and tools/search.py are accessible.")
    litsearch_main_coro = None
    run_search_pipeline = None
    load_rich_theme = None
    ComponentLoader = None
    CancellationToken = None
    LiteratureSearchTool = None
    SearchLiteratureParams = None
    TriageAgent = None 

def create_dummy_configs(tmp_path: Path, agents_content=None, settings_content=None, env_content=None, rich_theme_content=None):
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    if agents_content is None: agents_content = []
    (config_dir / "agents.yaml").write_text(yaml.dump(agents_content))
    if settings_content is None: settings_content = {"version": "0.1-test"}
    (config_dir / "settings.yaml").write_text(yaml.dump(settings_content))
    if env_content is not None: (tmp_path / ".env").write_text(env_content)
    if rich_theme_content is None: rich_theme_content = {"primary": "cyan"}
    (config_dir / "rich_theme.json").write_text(json.dumps(rich_theme_content))

@pytest.mark.asyncio 
@pytest.mark.skipif(litsearch_main_coro is None, reason="cli.litsearch.main could not be imported")
async def test_dotenv_loading(monkeypatch, tmp_path, mocker): 
    env_content = "TEST_VAR_DOTENV_ASYNC=test_value_dotenv_async\n"
    dummy_env_file_path = tmp_path / ".env" 
    dummy_env_file_path.write_text(env_content)
    create_dummy_configs(tmp_path, agents_content=[])

    mocked_dotenv_load = mocker.patch("cli.litsearch.load_dotenv", return_value=True)
    
    monkeypatch.setattr("cli.litsearch.styled_input", lambda _p, _c: "test_query")
    mocker.patch("cli.litsearch.run_search_pipeline", new_callable=AsyncMock) 
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(spec=[]))

    with monkeypatch.context() as m:
        m.chdir(tmp_path) 
        await litsearch_main_coro() 
    
    mocked_dotenv_load.assert_called_once_with(dotenv_path=dummy_env_file_path, override=True)

@pytest.mark.skipif(litsearch_main_coro is None or ComponentLoader is None, reason="Imports failed")
@pytest.mark.asyncio 
async def test_component_loader_mocking(mocker, tmp_path, monkeypatch):
    dummy_agents_data = [{"name": "agent_one", "provider": "provider.One", "config": {"id": 1}}] 
    create_dummy_configs(tmp_path, agents_content=dummy_agents_data)
    mocked_cl_load = mocker.patch.object(ComponentLoader, "load_component", return_value=MagicMock())

    monkeypatch.setattr("cli.litsearch.styled_input", lambda _p, _c: "test")
    mocker.patch("cli.litsearch.run_search_pipeline", new_callable=AsyncMock)
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(spec=[]))
    mocker.patch("cli.litsearch.load_dotenv") 
    mocker.patch("cli.litsearch.UserProxyAgent")
    mocker.patch("cli.litsearch.AssistantAgent")
    mocker.patch("cli.litsearch.DebugOpenAIChatCompletionClient")
    mocker.patch("cli.litsearch.TriageAgent")
    mocker.patch("cli.litsearch.RoundRobinGroupChat")

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        await litsearch_main_coro()
    assert mocked_cl_load.call_count == 0 
    
@pytest.mark.skipif(litsearch_main_coro is None or TriageAgent is None, reason="Imports failed for litsearch_main or TriageAgent") 
@pytest.mark.asyncio 
async def test_script_output_captures_reply(capsys, mocker, tmp_path, monkeypatch):
    user_proxy_name, assistant_name = "user_proxy", "query_refiner"
    valid_agents_config = [
        {"name": user_proxy_name, "provider": "autogen_agentchat.agents.UserProxyAgent", "config": {"name": user_proxy_name, "human_input_mode": "NEVER"}},
        {"name": assistant_name, "provider": "autogen_agentchat.agents.AssistantAgent", "config": {"name": assistant_name, "system_message": "System message for query_refiner.", "model_client": {"provider": "mock.Client", "config": {}}}},
        {"name": "query_team", "provider": "autogen_agentchat.teams.RoundRobinGroupChat", "config": {"participants": [{"name": assistant_name}, {"name": user_proxy_name}], "termination_condition": {"provider": "autogen_agentchat.conditions.HandoffTermination", "config": {"target": "user"}}}},
        {"name": "triage", "provider": "tools.triage.TriageAgent", "config": {"name": "triage", "model_client": {"provider": "mock.Client", "config": {}}}},
        # Add missing agents that main() expects or has fallbacks for
        {"name": "search_literature", "provider": "tools.search.LiteratureSearchTool", "config": {}},
        {"name": "FullTextRetrievalAgent", "provider": "tools.retrieve_full_text.FullTextRetrievalAgent", "config": {}}
    ]
    create_dummy_configs(tmp_path, agents_content=valid_agents_config)
    
    mock_user_proxy = MagicMock(name="MockUserProxyAgent")
    mock_query_refiner = MagicMock(name="MockQueryRefinerAssistantAgent")
    # mock_query_refiner.system_message = "" # No longer needed here, will be set by side_effect

    mocker.patch("cli.litsearch.UserProxyAgent", return_value=mock_user_proxy)
    def mock_assistant_constructor_side_effect_reply(*args, **kwargs):
        if kwargs.get("name") == assistant_name:
            # Set the system_message on the mock_query_refiner instance
            # based on what's passed to the constructor.
            mock_query_refiner.system_message = kwargs.get("system_message", "")
            return mock_query_refiner
        return MagicMock(name=f"OtherAssistant_{kwargs.get('name')}")
    mocker.patch("cli.litsearch.AssistantAgent", side_effect=mock_assistant_constructor_side_effect_reply)
    
    mocker.patch("cli.litsearch.DebugOpenAIChatCompletionClient", return_value=MagicMock())
    mocker.patch("cli.litsearch.FunctionTool", return_value=MagicMock())
    # Patch TriageAgent with the MagicMock class itself, not an instance.
    # This ensures that when TriageAgent is instantiated in cli.litsearch,
    # it creates a MagicMock instance, and isinstance(mock_instance, MagicMock_class) is True.
    mocker.patch("cli.litsearch.TriageAgent", MagicMock) 
    # Mock FullTextRetrievalAgent as well, as it's in the config now
    mocker.patch("cli.litsearch.FullTextRetrievalAgent", MagicMock)
    
    mock_query_team_inst = MagicMock(name="MockQueryTeamInstance", spec=BaseGroupChat) # Added spec
    mocker.patch("cli.litsearch.RoundRobinGroupChat", return_value=mock_query_team_inst)
    
    # Mock Console.print directly to ensure capsys captures it
    mock_console_print = mocker.patch("rich.console.Console.print")

    async def mock_run_pipeline_for_output(*args, **kwargs):
        # Corrected indices:
        # args[0]=query_team, args[1]=literature_search_tool, args[2]=triage_agent, 
        # args[3]=agents_dict, args[4]=original_query, args[5]=console, 
        # args[6]=settings_config, args[7]=cli_args
        original_query_arg = args[4]
        console_instance = args[5]

        output_file = tmp_path / "workspace" / "triage_results.json"
        (tmp_path / "workspace").mkdir(exist_ok=True)
        with open(output_file, "w") as f: json.dump({}, f)
    
        # Simulate the print call that would happen in run_search_pipeline
        console_instance.print(f"[primary]Search and triage pipeline completed. Results saved to {output_file}[/primary]")
        return {"query": original_query_arg, "refined_queries": [], "triaged_articles": []}
    
    mocker.patch("cli.litsearch.run_search_pipeline", side_effect=mock_run_pipeline_for_output)
    
    # Patch LiteratureSearchTool with MagicMock class.
    # Instances created from LiteratureSearchTool in cli.litsearch.py will be MagicMock instances.
    # The isinstance(instance, LiteratureSearchTool) check in cli.litsearch.py
    # will effectively become isinstance(mock_instance, MagicMock), which is valid.
    mocker.patch("cli.litsearch.LiteratureSearchTool", MagicMock)
    
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(spec=[]))
    mocker.patch("cli.litsearch.styled_input", return_value="test query")

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        await litsearch_main_coro()

    captured = capsys.readouterr()
    
    # Check if mock_console_print was called with the expected string
    # This is more robust than checking capsys if Rich's behavior with capsys is tricky.
    found_print_call = False
    expected_print_fragment = "Search and triage pipeline completed. Results saved to"
    expected_path_fragment = "workspace/triage_results.json"
    # Debug: print all calls to mock_console_print
    # print("DEBUG: mock_console_print.call_args_list:")
    # for i, call_args_item in enumerate(mock_console_print.call_args_list):
    #     print(f"Call {i}: {call_args_item}")
    #     if call_args_item[0]: # Positional arguments
    #         print(f"  Arg 0 type: {type(call_args_item[0][0])}")
    #         print(f"  Arg 0 content: {str(call_args_item[0][0])}")


    for call_args in mock_console_print.call_args_list:
        # The first positional argument to Console.print()
        arg = call_args[0][0]
        
        # Convert Rich Text object to plain string if necessary
        plain_text_arg = str(arg) # Works for str and rich.text.Text

        if expected_print_fragment in plain_text_arg and expected_path_fragment in plain_text_arg:
            found_print_call = True
            break
    assert found_print_call, f"Expected console print containing '{expected_print_fragment}' and '{expected_path_fragment}' was not found. Calls: {mock_console_print.call_args_list}"

    assert "System message for query_refiner." in mock_query_refiner.system_message


@pytest.mark.skipif(run_search_pipeline is None or LiteratureSearchTool is None or CancellationToken is None or SearchLiteratureParams is None or TriageAgent is None, reason="Required imports failed for test_search_pipeline_logic")
@pytest.mark.asyncio
async def test_search_pipeline_logic(mocker, tmp_path, monkeypatch):
    original_query = "impact of climate change on marine ecosystems"
    mock_query_team = MagicMock()
    mock_query_team.run = AsyncMock() 
    refined_queries_payload = [{"pubmed_query": "q1_pm", "general_query": "q1_gen"}, {"pubmed_query": "q2_pm", "general_query": "q2_gen"}, {"pubmed_query": "q3_pm", "general_query": "q3_gen"}]
    mock_query_team.run.return_value = MagicMock(messages=[TextMessage(source="query_refiner", content=f"```json\n{json.dumps(refined_queries_payload)}\n```")])

    mock_literature_search_tool = MagicMock(spec=LiteratureSearchTool)
    mock_literature_search_tool.run.side_effect = [{"data": [], "meta": {}} for _ in range(3)]
    mock_triage_agent = MagicMock(spec=TriageAgent)
    mock_triage_agent.triage_articles_async.return_value = []
    mock_console = MagicMock(spec=Console)
    # Add mocks for the new parameters
    mock_agents_dict = {"FullTextRetrievalAgent": MagicMock()} 
    mock_settings_config = {"search_settings": {"default_publication_types": ["journal article"], "default_max_results_per_source": 10}}
    mock_cli_args = MagicMock(pub_types=None) # Simulate no CLI pub_types

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        await run_search_pipeline(
            mock_query_team, 
            mock_literature_search_tool, 
            mock_triage_agent, 
            mock_agents_dict, # Added
            original_query, 
            mock_console,
            mock_settings_config, # Added
            mock_cli_args # Added
        )
    mock_query_team.run.assert_called_once_with(task=original_query)

@pytest.mark.skipif(load_rich_theme is None, reason="load_rich_theme could not be imported")
def test_load_rich_theme(tmp_path):
    theme_content = {"info": "blue", "warning": "bright_yellow"}
    create_dummy_configs(tmp_path, rich_theme_content=theme_content)
    loaded_theme = load_rich_theme(tmp_path)
    assert loaded_theme.styles["info"].color.name == "blue"
    assert loaded_theme.styles["warning"].color.name == "bright_yellow"
    (tmp_path / "config" / "rich_theme.json").unlink()
    empty_theme = load_rich_theme(tmp_path)
    assert empty_theme.styles == Theme({}).styles 

@pytest.mark.skipif(run_search_pipeline is None or LiteratureSearchTool is None or CancellationToken is None or SearchLiteratureParams is None or TriageAgent is None or TextMessage is None, reason="Required imports failed for test_groupchat_query_pipeline")
@pytest.mark.asyncio
async def test_groupchat_query_pipeline(mocker, tmp_path, monkeypatch):
    original_query = "impact of climate change on marine ecosystems"
    mock_query_team = MagicMock()
    mock_query_team.run = AsyncMock()
    refined_queries = [{"pubmed_query": "q1_pm", "general_query": "q1_gen"}, {"pubmed_query": "q2_pm", "general_query": "q2_gen"}, {"pubmed_query": "q3_pm", "general_query": "q3_gen"}]
    mock_query_team.run.return_value = MagicMock(messages=[TextMessage(source="query_refiner", content=f"```json\n{json.dumps(refined_queries)}\n```")])
    
    mock_literature_search_tool = MagicMock(spec=LiteratureSearchTool)
    mock_literature_search_tool.run.return_value = {"data": [], "meta": {}}
    mock_console = MagicMock(spec=Console)
    mock_triage_agent = MagicMock(spec=TriageAgent)
    mock_triage_agent.triage_articles_async.return_value = []
    # Add mocks for the new parameters
    mock_agents_dict = {"FullTextRetrievalAgent": MagicMock()}
    mock_settings_config = {"search_settings": {"default_publication_types": ["journal article"], "default_max_results_per_source": 10}}
    mock_cli_args = MagicMock(pub_types=None)

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        await run_search_pipeline(
            mock_query_team, 
            mock_literature_search_tool, 
            mock_triage_agent, 
            mock_agents_dict, # Added
            original_query, 
            mock_console,
            mock_settings_config, # Added
            mock_cli_args # Added
        )
    mock_query_team.run.assert_called_once_with(task=original_query)
