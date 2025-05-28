import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock

from autogen_agentchat.messages import TextMessage
from tools.retrieve_full_text import FullTextRetrievalAgent, retrieve_full_texts_for_dois

@pytest.fixture
def full_text_agent():
    """Fixture to create an instance of FullTextRetrievalAgent."""
    return FullTextRetrievalAgent(name="TestFullTextAgent")

@pytest.mark.asyncio
async def test_full_text_retrieval_agent_enriches_records(full_text_agent):
    """
    Tests that FullTextRetrievalAgent correctly calls retrieve_full_texts_for_dois
    and returns the enriched records.
    """
    sample_input_articles = [
        {"doi": "10.123/test.doi.1", "title": "Article 1", "relevance_score": 5},
        {"doi": "10.123/test.doi.2", "title": "Article 2", "relevance_score": 4},
    ]

    expected_enriched_articles = [
        {"doi": "10.123/test.doi.1", "title": "Article 1", "relevance_score": 5, "fulltext": "Full text for article 1", "fulltext_retrieval_status": "success", "fulltext_retrieval_message": "Mocked success"},
        {"doi": "10.123/test.doi.2", "title": "Article 2", "relevance_score": 4, "fulltext": None, "fulltext_retrieval_status": "failure", "fulltext_retrieval_message": "Mocked failure"},
    ]

    # This is the structure that retrieve_full_texts_for_dois returns
    mock_retrieval_output = {
        "query": "Full-text retrieval task",
        "refined_queries": [],
        "triaged_articles": expected_enriched_articles
    }

    # Path to the function to be mocked within tools.retrieve_full_text module
    mock_target_function = "tools.retrieve_full_text.retrieve_full_texts_for_dois"

    with patch(mock_target_function, new_callable=MagicMock) as mock_retrieve_func:
        # Configure the mock to return a coroutine that resolves to mock_retrieval_output
        async def async_mock_retrieve_func(*args, **kwargs):
            return mock_retrieval_output
        
        mock_retrieve_func.side_effect = async_mock_retrieve_func

        # Prepare input message for the agent
        input_json_str = json.dumps(sample_input_articles)
        input_message = TextMessage(content=input_json_str, source="test_runner")

        # Call the agent's on_messages method
        response = await full_text_agent.on_messages(messages=[input_message], cancellation_token=MagicMock())

        # Assertions
        mock_retrieve_func.assert_called_once()
        
        # Check the argument passed to the mocked function
        # The agent wraps the input list into a dict before calling retrieve_full_texts_for_dois
        expected_call_arg = {
            "query": "Full-text retrieval task",
            "refined_queries": [],
            "triaged_articles": sample_input_articles
        }
        mock_retrieve_func.assert_called_with(expected_call_arg)

        assert response.chat_message is not None
        assert isinstance(response.chat_message, TextMessage)
        
        response_content_list = json.loads(response.chat_message.content)
        assert response_content_list == expected_enriched_articles

@pytest.mark.asyncio
async def test_full_text_agent_handles_empty_input(full_text_agent):
    """Tests agent's behavior with no input messages."""
    response = await full_text_agent.on_messages(messages=[], cancellation_token=MagicMock())
    assert "Error: No input messages received." in response.chat_message.content

@pytest.mark.asyncio
async def test_full_text_agent_handles_invalid_json_input(full_text_agent):
    """Tests agent's behavior with invalid JSON in TextMessage."""
    input_message = TextMessage(content="this is not json", source="test_runner")
    response = await full_text_agent.on_messages(messages=[input_message], cancellation_token=MagicMock())
    assert "Error: Invalid JSON input" in response.chat_message.content

@pytest.mark.asyncio
async def test_full_text_agent_handles_non_list_json_input(full_text_agent):
    """Tests agent's behavior with JSON that is not a list."""
    input_json_str = json.dumps({"not_a": "list"})
    input_message = TextMessage(content=input_json_str, source="test_runner")
    response = await full_text_agent.on_messages(messages=[input_message], cancellation_token=MagicMock())
    assert "Error: Input JSON must be a list of articles" in response.chat_message.content

@pytest.mark.asyncio
async def test_full_text_agent_handles_retrieval_exception(full_text_agent):
    """Tests agent's error handling when retrieve_full_texts_for_dois raises an exception."""
    sample_input_articles = [{"doi": "10.123/test.doi.1"}]
    input_json_str = json.dumps(sample_input_articles)
    input_message = TextMessage(content=input_json_str, source="test_runner")

    mock_target_function = "tools.retrieve_full_text.retrieve_full_texts_for_dois"
    with patch(mock_target_function, new_callable=MagicMock) as mock_retrieve_func:
        async def async_mock_raise_exception(*args, **kwargs):
            raise ValueError("Simulated retrieval error")
        mock_retrieve_func.side_effect = async_mock_raise_exception
        
        response = await full_text_agent.on_messages(messages=[input_message], cancellation_token=MagicMock())
        
        assert "Error during full text retrieval: Simulated retrieval error" in response.chat_message.content
