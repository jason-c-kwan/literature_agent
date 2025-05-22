import asyncio
import datetime
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel # Added import

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, RequestUsage
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response as AgentChatResponse


# Ensure tools.triage can be imported.
# This might require adjusting sys.path or ensuring __init__.py files are correct
# For now, assuming direct import works if tests are run from project root.
from tools.triage import TriageAgent

# Mock ChatCompletionClient
class MockChatCompletionClient(ChatCompletionClient):
    def __init__(self, response_content: Union[str, List[Any]] = "3", raise_exception: bool = False):
        self.response_content = response_content
        self.raise_exception = raise_exception
        self._model_info_dict = { # Store as a private dict
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mock_family",
            "structured_output": False,
        }

    @property
    def model_info(self): # Implement as a property
        return self._model_info_dict

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        if self.raise_exception:
            raise Exception("Mock LLM Error")
        
        # Simulate the structure AssistantAgent expects from its LLM call
        # The content here is what TriageAgent's get_relevance_score will parse
        return CreateResult(
            content=self.response_content, # This is the direct score string
            usage=RequestUsage(prompt_tokens=10, completion_tokens=1),
            thought=None 
        )

    def create_stream(self, *args, **kwargs): # Not used by TriageAgent directly in this setup
        pass
    async def close(self): pass
    def actual_usage(self) -> RequestUsage: return RequestUsage(prompt_tokens=0, completion_tokens=0)
    def total_usage(self) -> RequestUsage: return RequestUsage(prompt_tokens=0, completion_tokens=0)
    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int: return 0
    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int: return 1000
    @property
    def capabilities(self): return {"vision": False, "function_calling": False, "json_output": False} # Deprecated

    def dump_component(self) -> Dict[str, Any]: return {} # For ComponentBase
    @classmethod
    def load_component(cls, config: Dict[str, Any]) -> "MockChatCompletionClient": return cls() # For ComponentBase
    def _to_config(self) -> Dict[str, Any]: return {} # For ComponentBase
    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "MockChatCompletionClient": return cls() # For ComponentBase


@pytest.fixture
def mock_settings_file_content():
    return {
        'triage': {
            'sjr_percentile_threshold': 60,
            'min_age_normalized_citations': 1.5,
            'open_access_statuses': ['gold', 'green']
        }
    }

@pytest.fixture
def temp_settings_file(mock_settings_file_content):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmpfile:
        yaml.dump(mock_settings_file_content, tmpfile)
        filepath = tmpfile.name
    yield filepath
    os.remove(filepath)

@pytest.fixture
def sample_articles():
    return [
        {
            'title': 'Article A', 'abstract': 'Abstract A', 'journal': 'Journal X', 
            'oa_status': 'gold', 'citation_count': 10, 'year': 2020, 
            'sjr_percentile': 70, 'doi': '10.1/a'
        },
        {
            'title': 'Article B', 'abstract': 'Abstract B', 'journal': 'Journal Y', 
            'oa_status': 'closed', 'citation_count': 5, 'year': 2021, 
            'sjr_percentile': 80, 'doi': '10.1/b'
        },
        {
            'title': 'Article C', 'abstract': 'Abstract C', 'journal': 'Journal Z', 
            'oa_status': 'green', 'citation_count': 2, 'year': 2022, 
            'sjr_percentile': 50, 'doi': '10.1/c' # Fails SJR
        },
        {
            'title': 'Article D', 'abstract': 'Abstract D', 'journal': 'Journal W', 
            'oa_status': 'gold', 'citation_count': 1, 'year': 2023, 
            'sjr_percentile': 90, 'doi': '10.1/d' # Fails norm_citation (1 / (CUR_YEAR-2023+1))
        },
        {
            'title': 'Article E Missing Data', 'abstract': 'Abstract E', 'journal': 'Journal V',
            'oa_status': 'gold', 'year': 2020, 
            'sjr_percentile': 90, 'doi': '10.1/e' # Missing citation_count
        },
         {
            'title': 'Article F High Citations', 'abstract': 'Abstract F', 'journal': 'Journal Q', 
            'oa_status': 'gold', 'citation_count': 100, 'year': 2020, 
            'sjr_percentile': 90, 'doi': '10.1/f'
        },
    ]

@pytest.mark.asyncio
async def test_triage_agent_load_settings(temp_settings_file, mock_settings_file_content):
    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client, settings_path=temp_settings_file)
    
    triage_settings = mock_settings_file_content['triage']
    assert agent.sjr_threshold == triage_settings['sjr_percentile_threshold']
    assert agent.norm_citation_threshold == triage_settings['min_age_normalized_citations']
    assert agent.oa_statuses_accepted == triage_settings['open_access_statuses']

@pytest.mark.asyncio
async def test_triage_agent_load_settings_file_not_found():
    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client, settings_path="non_existent_file.yaml")
    
    assert agent.sjr_threshold == 50 # Default
    assert agent.norm_citation_threshold == 1.0 # Default
    assert agent.oa_statuses_accepted == ['gold', 'hybrid', 'green'] # Default

@pytest.mark.asyncio
@patch('autogen_agentchat.agents.AssistantAgent.on_messages', new_callable=AsyncMock)
async def test_get_relevance_score_valid(mock_super_on_messages):
    mock_client = MockChatCompletionClient() 
    agent = TriageAgent(name="test_triage", model_client=mock_client)

    # Configure the mock for super().on_messages
    # It should return an AgentChatResponse whose chat_message.content is the score string
    mock_super_on_messages.return_value = AgentChatResponse(
        chat_message=TextMessage(content="4", source="assistant")
    )

    score = await agent.get_relevance_score("Title", "Abstract", "Query")
    assert score == 4
    mock_super_on_messages.assert_called_once()
    # We can add more assertions here about the content of messages passed to on_messages

@pytest.mark.asyncio
@patch('autogen_agentchat.agents.AssistantAgent.on_messages', new_callable=AsyncMock)
async def test_get_relevance_score_invalid_string(mock_super_on_messages):
    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client)
    mock_super_on_messages.return_value = AgentChatResponse(
        chat_message=TextMessage(content="not a score", source="assistant")
    )
    score = await agent.get_relevance_score("Title", "Abstract", "Query")
    assert score == 1 # Default on error

@pytest.mark.asyncio
@patch('autogen_agentchat.agents.AssistantAgent.on_messages', new_callable=AsyncMock)
async def test_get_relevance_score_out_of_range(mock_super_on_messages):
    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client)
    mock_super_on_messages.return_value = AgentChatResponse(
        chat_message=TextMessage(content="7", source="assistant")
    )
    score = await agent.get_relevance_score("Title", "Abstract", "Query")
    assert score == 1 # Default on error

@pytest.mark.asyncio
@patch('autogen_agentchat.agents.AssistantAgent.on_messages', new_callable=AsyncMock)
async def test_get_relevance_score_llm_exception(mock_super_on_messages):
    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client)
    mock_super_on_messages.side_effect = Exception("LLM API Error")
    score = await agent.get_relevance_score("Title", "Abstract", "Query")
    assert score == 1 # Default on error

@pytest.mark.asyncio
@patch('tools.triage.datetime') # To mock datetime.datetime.now()
@patch('tools.triage.TriageAgent.get_relevance_score', new_callable=AsyncMock) # Mock the LLM call
async def test_triage_articles_async_filtering(
    mock_get_score: AsyncMock, 
    mock_datetime: MagicMock, 
    sample_articles: List[Dict[str, Any]], 
    temp_settings_file: str
):
    # Setup mocks
    mock_get_score.return_value = 3 # Assume all articles get a score of 3
    mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 1) # For consistent age calculation

    mock_client = MockChatCompletionClient() # This won't be called due to mocking get_relevance_score
    agent = TriageAgent(name="test_triage", model_client=mock_client, settings_path=temp_settings_file)
    
    # Override settings for this specific test if needed, or rely on temp_settings_file
    agent.sjr_threshold = 60
    agent.norm_citation_threshold = 1.5 
    agent.oa_statuses_accepted = ['gold', 'green']

    user_query = "test query"
    results = await agent.triage_articles_async(sample_articles, user_query)

    result_titles = [r['title'] for r in results]

    # Article A: score=3, sjr=70(ok), oa=gold(ok), norm_cit=10/(2024-2020+1)=10/5=2.0(ok) -> PASS
    assert "Article A" in result_titles
    
    # Article B: oa=closed(fail) -> FAIL
    assert "Article B" not in result_titles
    
    # Article C: sjr=50(fail, <60) -> FAIL
    assert "Article C" not in result_titles
    
    # Article D: norm_cit=1/(2024-2023+1)=1/2=0.5(fail, <1.5) -> FAIL
    assert "Article D" not in result_titles

    # Article E Missing Data: citation_count is None -> FAIL (due to current logic)
    assert "Article E Missing Data" not in result_titles

    # Article F High Citations: score=3, sjr=90(ok), oa=gold(ok), norm_cit=100/(2024-2020+1)=100/5=20.0(ok) -> PASS
    assert "Article F High Citations" in result_titles

    assert len(results) == 2 # A and F

    for r in results:
        assert 'relevance_score' in r
        assert r['relevance_score'] == 3


@pytest.mark.asyncio
@patch('tools.triage.datetime')
@patch('tools.triage.TriageAgent.get_relevance_score', new_callable=AsyncMock)
async def test_triage_article_passes_all_filters(mock_get_score, mock_datetime, temp_settings_file):
    mock_get_score.return_value = 5
    mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 1)
    
    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client, settings_path=temp_settings_file)
    agent.sjr_threshold = 50
    agent.norm_citation_threshold = 1.0
    agent.oa_statuses_accepted = ['gold']

    article = [{
        'title': 'Super Article', 'abstract': 'Amazing abstract', 'journal': 'Top Journal',
        'oa_status': 'gold', 'citation_count': 20, 'publication_year': 2020,
        'sjr_percentile': 80
    }]
    results = await agent.triage_articles_async(article, "query")
    assert len(results) == 1
    assert results[0]['title'] == 'Super Article'
    assert results[0]['relevance_score'] == 5

@pytest.mark.asyncio
@patch('tools.triage.datetime')
@patch('tools.triage.TriageAgent.get_relevance_score', new_callable=AsyncMock)
async def test_triage_article_invalid_sjr_value(mock_get_score, mock_datetime, temp_settings_file):
    mock_get_score.return_value = 4
    mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 1)

    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client, settings_path=temp_settings_file)
    agent.sjr_threshold = 50
    agent.norm_citation_threshold = 1.0
    agent.oa_statuses_accepted = ['gold']

    article_invalid_sjr = [{ # Should pass if SJR is skipped
        'title': 'Test Invalid SJR', 'abstract': 'Abstract', 'journal': 'Journal',
        'oa_status': 'gold', 'citation_count': 10, 'publication_year': 2022,
        'sjr_percentile': 'not_a_number' 
    }]
    # Expecting a warning to be printed, but the article should pass other filters
    results = await agent.triage_articles_async(article_invalid_sjr, "query")
    assert len(results) == 1 
    assert results[0]['title'] == 'Test Invalid SJR'

@pytest.mark.asyncio
@patch('tools.triage.datetime')
@patch('tools.triage.TriageAgent.get_relevance_score', new_callable=AsyncMock)
async def test_triage_article_missing_year_for_citation_filter(mock_get_score, mock_datetime, temp_settings_file):
    mock_get_score.return_value = 4
    mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 1)

    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client, settings_path=temp_settings_file)
    
    article_missing_year = [{
        'title': 'Test Missing Year', 'abstract': 'Abstract', 'journal': 'Journal',
        'oa_status': 'gold', 'citation_count': 10, # 'publication_year' is missing
        'sjr_percentile': 60 
    }]
    results = await agent.triage_articles_async(article_missing_year, "query")
    assert len(results) == 0 # Fails because year is needed for citation filter

@pytest.mark.asyncio
@patch('tools.triage.datetime')
@patch('tools.triage.TriageAgent.get_relevance_score', new_callable=AsyncMock)
async def test_triage_article_zero_age_citation_normalization(mock_get_score, mock_datetime, temp_settings_file):
    mock_get_score.return_value = 4
    # Set current year to be the same as publication year, leading to age = 1 (after +1 adjustment)
    mock_datetime.datetime.now.return_value = datetime.datetime(2023, 1, 1) 

    mock_client = MockChatCompletionClient()
    agent = TriageAgent(name="test_triage", model_client=mock_client, settings_path=temp_settings_file)
    agent.norm_citation_threshold = 0.5 # Lower threshold for this test

    article_current_year = [{
        'title': 'Current Year Article', 'abstract': 'Abstract', 'journal': 'Journal',
        'oa_status': 'gold', 'citation_count': 1, 'publication_year': 2023, # pub_year = current_year
        'sjr_percentile': 60 
    }]
    # Age = 2023 - 2023 + 1 = 1. Norm_cit = 1/1 = 1.0. Should pass if threshold is <= 1.0
    results = await agent.triage_articles_async(article_current_year, "query")
    assert len(results) == 1
    assert results[0]['title'] == 'Current Year Article'

    agent.norm_citation_threshold = 1.1 # Now it should fail
    results_fail = await agent.triage_articles_async(article_current_year, "query")
    assert len(results_fail) == 0
