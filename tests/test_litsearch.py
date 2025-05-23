import pytest
import os
import yaml
import json
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, call
from rich.console import Console
from rich.theme import Theme

# Assuming tests are run from the project root, so cli.litsearch is importable
try:
    from cli.litsearch import main as litsearch_main, run_search_pipeline, load_rich_theme
    from autogen_core import ComponentLoader, CancellationToken
    from tools.search import LiteratureSearchTool, SearchLiteratureParams
except ImportError as e:
    print(f"ImportError in test setup: {e}. Ensure cli/litsearch.py and tools/search.py are accessible.")
    litsearch_main = None
    run_search_pipeline = None
    load_rich_theme = None
    ComponentLoader = None
    CancellationToken = None
    LiteratureSearchTool = None
    SearchLiteratureParams = None


# Helper to create dummy config files
def create_dummy_configs(tmp_path: Path, agents_content=None, settings_content=None, env_content=None, rich_theme_content=None):
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
    
    if rich_theme_content is None:
        rich_theme_content = {
            "primary": "cyan",
            "secondary": "green",
            "highlight": "yellow"
        }
    (config_dir / "rich_theme.json").write_text(json.dumps(rich_theme_content))


@pytest.mark.skipif(litsearch_main is None, reason="cli.litsearch.main could not be imported")
def test_dotenv_loading(monkeypatch, tmp_path):
    """Asserts .env variables are loaded."""
    sentinel_key = "TEST_SENTINEL_VAR_FROM_FILE"
    sentinel_value = "loaded_successfully"
    env_content = f"{sentinel_key}={sentinel_value}\n"
    
    create_dummy_configs(tmp_path, env_content=env_content)

    monkeypatch.delenv(sentinel_key, raising=False)
    
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        litsearch_main()
    
    assert os.getenv(sentinel_key) == sentinel_value

@pytest.mark.skipif(litsearch_main is None or ComponentLoader is None, reason="Imports failed")
def test_component_loader_mocking(mocker, tmp_path, monkeypatch):
    """Mocks ComponentLoader.load_component."""
    dummy_agents_data = [
        {"name": "agent_one", "provider": "provider.One", "component_type": "agent", "config": {"id": 1}},
        {"name": "agent_two", "provider": "provider.Two", "component_type": "agent", "config": {"id": 2}},
    ]
    create_dummy_configs(tmp_path, agents_content=dummy_agents_data)

    mocked_load_component_method = mocker.patch.object(ComponentLoader, "load_component", return_value=MagicMock())

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        litsearch_main()

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
    for expected_call_args in expected_calls:
         mocked_load_component_method.assert_any_call(*expected_call_args.args, **expected_call_args.kwargs)


@pytest.mark.skipif(litsearch_main is None or ComponentLoader is None, reason="Imports failed")
def test_script_output_captures_reply(capsys, mocker, tmp_path, monkeypatch):
    """Confirms the script prints the assistant’s reply."""
    user_proxy_name = "user_proxy"
    assistant_name = "query_refiner"

    agents_config_for_test = [
        {"name": user_proxy_name, "provider": "mock.UserProxy", "component_type": "agent", "config": {"name": user_proxy_name}},
        {"name": assistant_name, "provider": "mock.Assistant", "component_type": "agent", "config": {"name": assistant_name}},
    ]
    create_dummy_configs(tmp_path, agents_content=agents_config_for_test)

    mock_user_proxy_agent = MagicMock(name="MockUserProxyAgent")
    mock_assistant_agent = MagicMock(name="MockAssistantAgent")
    # Mock the system_message attribute as it's set dynamically
    mock_assistant_agent.system_message = "" 

    def side_effect_load_component(config):
        if config.get("name") == user_proxy_name:
            return mock_user_proxy_agent
        elif config.get("name") == assistant_name:
            return mock_assistant_agent
        return MagicMock()

    mocker.patch.object(ComponentLoader, "load_component", side_effect=side_effect_load_component)
    
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(rounds=1))
    mocker.patch("rich.console.Console.input", return_value="test query")

    # Mock the query_refiner_agent.run to return expected JSON output
    mock_assistant_agent.run.return_value = MagicMock(messages=[MagicMock(content=json.dumps([
        {"pubmed_query": "test_pubmed_q", "general_query": "test_general_q"}
    ]))])
    
    # Mock LiteratureSearchTool.run to return the new structured output
    mock_literature_search_tool = MagicMock(spec=LiteratureSearchTool)
    mock_literature_search_tool.run.return_value = {
        "data": [{"doi": "10.123/test", "title": "Test Title", "abstract": "Test Abstract"}],
        "meta": {"total_hits": 1, "query": "test_general_q", "timestamp": "2023-01-01T00:00:00Z"}
    }
    mocker.patch("cli.litsearch.LiteratureSearchTool", return_value=mock_literature_search_tool)


    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        litsearch_main()

    captured = capsys.readouterr()
    
    assert "Search pipeline completed. Results saved to dois.json" in captured.out
    # Verify the system message was set
    assert "For a given user question, generate exactly 1 refined search strings." in mock_assistant_agent.system_message


@pytest.mark.skipif(run_search_pipeline is None or LiteratureSearchTool is None or CancellationToken is None or SearchLiteratureParams is None, reason="Required imports failed for test_search_pipeline_logic")
@pytest.mark.asyncio
async def test_search_pipeline_logic(mocker, tmp_path, monkeypatch):
    """
    Verifies the core search pipeline logic with new structured queries:
    - Query-RefinerAgent calls and output parsing
    - LiteratureSearchTool calls with correct arguments
    - DOI extraction and deduplication
    - JSON output format and content
    """
    original_query = "impact of climate change on marine ecosystems"
    num_rounds = 2
    
    # Mock the Query-RefinerAgent
    mock_query_refiner_agent = MagicMock()
    # Simulate the agent returning a JSON string of structured queries
    mock_query_refiner_agent.run.side_effect = [
        MagicMock(messages=[MagicMock(content=json.dumps([
            {"pubmed_query": "climate change[Mesh] AND marine ecosystems[Mesh]", "general_query": "climate change marine ecosystems"},
            {"pubmed_query": "ocean acidification[Mesh] AND coral reefs[Mesh]", "general_query": "ocean acidification coral reefs"},
            {"pubmed_query": "sea level rise[Mesh] AND coastal habitats[Mesh]", "general_query": "sea level rise coastal habitats"}
        ]))]),
        MagicMock(messages=[MagicMock(content=json.dumps([
            {"pubmed_query": "marine biodiversity[Mesh] AND loss[tiab]", "general_query": "marine biodiversity loss"},
            {"pubmed_query": "polar ice melt[Mesh] AND effects[tiab]", "general_query": "polar ice melt effects"},
            {"pubmed_query": "fisheries[Mesh] AND collapse[tiab]", "general_query": "fisheries collapse"}
        ]))]),
    ]

    # Mock the LiteratureSearchTool
    mock_literature_search_tool = MagicMock(spec=LiteratureSearchTool)
    # The run method returns the new structured output
    mock_literature_search_tool.run.side_effect = [
        {"data": [{"doi": "10.1000/doi1", "title": "Doc 1", "abstract": "Abs 1"}, {"doi": "10.1000/doi2", "title": "Doc 2", "abstract": "Abs 2"}], "meta": {"total_hits": 2, "query": "q1", "timestamp": "ts1"}},
        {"data": [{"doi": "10.1000/doi2", "title": "Doc 2 (duplicate)", "abstract": "Abs 2 Dup"}, {"doi": "10.1000/doi3", "title": "Doc 3", "abstract": "Abs 3"}], "meta": {"total_hits": 2, "query": "q2", "timestamp": "ts2"}},
        {"data": [{"doi": "10.1000/doi4", "title": "Doc 4", "abstract": "Abs 4"}], "meta": {"total_hits": 1, "query": "q3", "timestamp": "ts3"}},
        {"data": [{"doi": "10.1000/doi5", "title": "Doc 5", "abstract": "Abs 5"}], "meta": {"total_hits": 1, "query": "q4", "timestamp": "ts4"}},
        {"data": [{"doi": "10.1000/doi1", "title": "Doc 1 (duplicate)", "abstract": "Abs 1 Dup"}, {"doi": "10.1000/doi6", "title": "Doc 6", "abstract": "Abs 6"}], "meta": {"total_hits": 2, "query": "q5", "timestamp": "ts5"}},
        {"data": [{"doi": "10.1000/doi7", "title": "Doc 7", "abstract": "Abs 7"}], "meta": {"total_hits": 1, "query": "q6", "timestamp": "ts6"}},
    ]

    # Mock rich.console.Console for input/output
    mock_console = MagicMock(spec=Console)
    mock_console.input.return_value = original_query
    mock_console.print.return_value = None

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        result_data = await run_search_pipeline(
            mock_query_refiner_agent,
            mock_literature_search_tool,
            original_query,
            num_rounds,
            mock_console
        )

        # Verify Query-RefinerAgent calls
        assert mock_query_refiner_agent.run.call_count == num_rounds
        expected_refiner_task_calls = [
            call(task=f"Refine the following research query into three distinct Boolean/MeSH-style search strings: {original_query}")
        ] * num_rounds
        mock_query_refiner_agent.run.assert_has_calls(expected_refiner_task_calls, any_order=False)

        # Verify LiteratureSearchTool calls
        expected_search_tool_calls = [
            call(args=SearchLiteratureParams(pubmed_query="climate change[Mesh] AND marine ecosystems[Mesh]", general_query="climate change marine ecosystems"), cancellation_token=mocker.ANY),
            call(args=SearchLiteratureParams(pubmed_query="ocean acidification[Mesh] AND coral reefs[Mesh]", general_query="ocean acidification coral reefs"), cancellation_token=mocker.ANY),
            call(args=SearchLiteratureParams(pubmed_query="sea level rise[Mesh] AND coastal habitats[Mesh]", general_query="sea level rise coastal habitats"), cancellation_token=mocker.ANY),
            call(args=SearchLiteratureParams(pubmed_query="marine biodiversity[Mesh] AND loss[tiab]", general_query="marine biodiversity loss"), cancellation_token=mocker.ANY),
            call(args=SearchLiteratureParams(pubmed_query="polar ice melt[Mesh] AND effects[tiab]", general_query="polar ice melt effects"), cancellation_token=mocker.ANY),
            call(args=SearchLiteratureParams(pubmed_query="fisheries[Mesh] AND collapse[tiab]", general_query="fisheries collapse"), cancellation_token=mocker.ANY),
        ]
        assert mock_literature_search_tool.run.call_count == num_rounds * 3
        mock_literature_search_tool.run.assert_has_calls(expected_search_tool_calls, any_order=False)

        # Verify dois.json content
        output_file_path = tmp_path / "dois.json"
        assert output_file_path.exists()
        with open(output_file_path, "r") as f:
            output_json = json.load(f)

        expected_dois = sorted([
            "10.1000/doi1", "10.1000/doi2", "10.1000/doi3", "10.1000/doi4",
            "10.1000/doi5", "10.1000/doi6", "10.1000/doi7"
        ])
        expected_refined_queries = [
            "climate change marine ecosystems", "ocean acidification coral reefs", "sea level rise coastal habitats",
            "marine biodiversity loss", "polar ice melt effects", "fisheries collapse"
        ]

        assert output_json["query"] == original_query
        assert output_json["refined_queries"] == expected_refined_queries
        assert sorted(output_json["dois"]) == expected_dois
        assert len(output_json["dois"]) == len(expected_dois)

        # Verify the returned data from run_search_pipeline
        assert result_data["query"] == original_query
        assert result_data["refined_queries"] == expected_refined_queries
        assert sorted(result_data["dois"]) == expected_dois


@pytest.mark.skipif(load_rich_theme is None, reason="load_rich_theme could not be imported")
def test_load_rich_theme(tmp_path):
    """Verify rich theme loading."""
    theme_content = {
        "info": "blue",
        "warning": "orange"
    }
    create_dummy_configs(tmp_path, rich_theme_content=theme_content)
    
    loaded_theme = load_rich_theme(tmp_path)
    assert loaded_theme.styles["info"].color.name == "blue"
    assert loaded_theme.styles["warning"].color.name == "orange"

    (tmp_path / "config" / "rich_theme.json").unlink()
    empty_theme = load_rich_theme(tmp_path)
    assert empty_theme.styles == {}


# To run these tests, navigate to the project root and run:
# pytest
# or
# pytest tests/test_litsearch.py
