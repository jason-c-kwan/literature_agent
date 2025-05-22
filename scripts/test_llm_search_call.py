# scripts/test_llm_search_call.py
import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools.search import search_literature
from autogen_core import CancellationToken # For manual loop
from autogen_agentchat.messages import TextMessage # For manual loop

# Load environment variables from .env file
load_dotenv()

async def run_interactive_tool_test():
    """
    Interactively prompts for a search query and tests if the LLM
    calls the search_literature tool using Gemini via an OpenAI-compatible endpoint.
    This version implements a manual interactive chat loop.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set or not found in .env file.")
        return
    else:
        # Print partial API key for verification
        print(f"DEBUG: Using API Key (first 5, last 5 chars): {api_key[:5]}...{api_key[-5:]}")

    # Get BASE_URL from OPENAI_API_BASE, defaulting to the one from test_gemini.py
    # This is CRUCIAL for directing requests to Gemini's OpenAI-compatible endpoint.
    gemini_base_url = os.environ.get("OPENAI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
    print(f"DEBUG: Using API Key (first 5, last 5 chars): {api_key[:5]}...{api_key[-5:]}")
    print(f"DEBUG: Using Base URL: {gemini_base_url}")

    # Use the model name confirmed to work in test_gemini.py
    # We can change this back to "gemini-2.5-pro-preview-05-06" later if needed.
    working_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17")
    print(f"DEBUG: Using Model Name: {working_model_name}")

    gemini_model_info = {
        "model": working_model_name, # Reflect the actual model being used
        "family": "gemini",
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
    }
    model_client = OpenAIChatCompletionClient(
        model=working_model_name, # Use the confirmed working model name
        api_key=api_key,
        base_url=gemini_base_url, # Crucial: Use the correct base URL
        model_info=gemini_model_info
    )
    assistant = AssistantAgent(
        name="LiteratureSearchAssistant",
        model_client=model_client,
        tools=[search_literature],
        system_message="You are a helpful assistant. Your primary goal is to use the 'search_literature' tool based on the user's query. Respond with the tool's findings. Reply with TERMINATE when the task is done, or if you cannot fulfill the request."
    )

    # UserProxyAgent instance. Its default input_func will be used if we were to call its on_messages.
    # In this manual loop, we get input directly but use its name.
    user_proxy = UserProxyAgent(
        name="HumanUser",
        # No human_input_mode or code_execution_config for this agent version
    )

    print("\n--- Interactive LLM Tool Call Test (Chat Mode) ---")
    print("Type 'exit' to end the chat.")
    
    cancellation_token = CancellationToken()

    for turn in range(10): # Limit to 10 turns for this example
        print(f"\n--- Turn {turn + 1} ---")
        try:
            # Get human input
            human_input_text = await asyncio.to_thread(input, f"{user_proxy.name}: ")
            if human_input_text.strip().lower() == "exit":
                print("Exiting chat.")
                break
            
            human_message = TextMessage(content=human_input_text, source=user_proxy.name)

            # Send the latest human message to the assistant
            # The AssistantAgent's on_messages expects only new messages, 
            # it maintains its own internal context.
            print(f"Sending to {assistant.name}: \"{human_input_text}\"")
            print(f"--- {assistant.name} processing... ---")
            
            # The on_messages method is the correct one for component-based agents
            # to process incoming messages and get a response.
            assistant_response_container = await assistant.on_messages(
                messages=[human_message], # Send only the latest message from the user
                cancellation_token=cancellation_token
            )
            
            assistant_reply_message = assistant_response_container.chat_message

            print(f"{assistant.name}: {assistant_reply_message.content}")

            # --- Detailed logging of assistant's turn ---
            tool_called_this_turn = False
            # Check inner messages for tool call requests/executions
            if assistant_response_container.inner_messages:
                print(f"  Inner messages from {assistant.name}:")
                for inner_msg in assistant_response_container.inner_messages:
                    print(f"    - Type: {type(inner_msg).__name__}, Content: {inner_msg.content}")
                    # ToolCallRequestEvent often has content as a list of FunctionCall objects
                    if hasattr(inner_msg, 'content') and isinstance(inner_msg.content, list):
                        for item in inner_msg.content:
                            if hasattr(item, 'name') and item.name == 'search_literature':
                                tool_called_this_turn = True
                                break
                    if tool_called_this_turn: break
            
            # Check the final reply message itself (e.g., ToolCallSummaryMessage)
            if not tool_called_this_turn and hasattr(assistant_reply_message, 'content'):
                 if isinstance(assistant_reply_message.content, list): # Direct tool call in response (less common for final)
                    for item in assistant_reply_message.content:
                        if hasattr(item, 'name') and item.name == 'search_literature':
                            tool_called_this_turn = True
                            break
                 elif type(assistant_reply_message).__name__ == 'ToolCallSummaryMessage' and \
                    'search_literature' in str(assistant_reply_message.content):
                    tool_called_this_turn = True
            
            if tool_called_this_turn:
                print(f"  INFO: 'search_literature' tool was called by {assistant.name} in this turn.")
            # --- End detailed logging ---

            if "TERMINATE" in str(assistant_reply_message.content).upper():
                print(f"--- {assistant.name} indicated TERMINATE. Ending chat. ---")
                break
        
        except Exception as e:
            print(f"\n--- An error occurred during turn {turn + 1}: ---")
            print(e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Load .env before checking for the key
    load_dotenv() 
    # Ensure OPENAI_API_KEY is available before running async
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable (e.g., in your .env file) before running this script.")
    else:
        try:
            asyncio.run(run_interactive_tool_test())
        except KeyboardInterrupt:
            print("\n--- Script interrupted by user. Exiting. ---")
