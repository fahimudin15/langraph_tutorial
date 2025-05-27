import os
from langchain.chat_models import init_chat_model
from typing import Annotated, Dict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langsmith import Client
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langgraph.types import Command, interrupt
from langchain.tools import tool
from datetime import datetime

# Set up API keys
os.environ["ANTHROPIC_API_KEY"] = "your_api_key"
os.environ["TAVILY_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_PROJECT"] = "langgraph_tutorial"

# Initialize tracers and callbacks
tracer = LangChainTracer(project_name="langgraph_tutorial")
console_tracer = ConsoleCallbackHandler()
callback_manager = CallbackManager([tracer])

class ConversationHistory:
    def __init__(self):
        self.states: Dict[int, dict] = {}
        self.current_index: int = 0
        self.max_index: int = 0

    def save_state(self, state: dict):
        self.current_index += 1
        self.max_index = self.current_index
        self.states[self.current_index] = {
            'state': state.copy(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return self.current_index

    def get_state(self, index: int) -> dict:
        if index in self.states:
            return self.states[index]['state']
        raise ValueError(f"No state found for index {index}")

    def list_states(self) -> List[str]:
        return [f"[{i}] {self.states[i]['timestamp']}: {self._summarize_state(self.states[i]['state'])}"
                for i in sorted(self.states.keys())]

    def _summarize_state(self, state: dict) -> str:
        if 'messages' in state:
            messages = state['messages']
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, (HumanMessage, AIMessage)):
                    content = last_message.content
                    if isinstance(content, list):
                        content = next((item['text'] for item in content if item['type'] == 'text'), '')
                    return f"{content[:50]}..." if len(content) > 50 else content
        return "Empty state"

    def time_travel(self, index: int) -> dict:
        if index in self.states:
            self.current_index = index
            return self.states[index]['state']
        raise ValueError(f"Cannot travel to index {index}. Valid indices are: {sorted(self.states.keys())}")

# Initialize conversation history
conversation_history = ConversationHistory()

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    try:
        human_response = interrupt({"query": query})
        return human_response["data"]
    except Exception as e:
        return "Waiting for human input..."

# Initialize tools
from langchain_tavily import TavilySearch
tavily_tool = TavilySearch(max_results=2)
tools = [human_assistance, tavily_tool]

# Initialize memory saver
memory = InMemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM with tools
llm = init_chat_model(
    "anthropic:claude-3-sonnet-20240229",
    callbacks=[tracer],
    verbose=False
)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    messages = state.get("messages", [])
    if not messages:
        return state
    
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        response = llm_with_tools.invoke(messages)
        return {"messages": messages + [response]}
    return state

class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            return inputs
        
        message = messages[-1]
        if not hasattr(message, "tool_calls"):
            return inputs
            
        outputs = []
        for tool_call in message.tool_calls:
            try:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=str(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content="Waiting for human input...",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"messages": messages + outputs}

# Initialize the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

def route_tools(state: State):
    messages = state.get("messages", [])
    if not messages:
        return END
    
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)

def handle_time_travel_command(command: str, current_state: dict) -> dict:
    if command.startswith("!history"):
        print("\nConversation History:")
        for entry in conversation_history.list_states():
            print(entry)
        return current_state
    
    if command.startswith("!goto "):
        try:
            index = int(command.split()[1])
            new_state = conversation_history.time_travel(index)
            print(f"\nTraveled to state [{index}]")
            return new_state
        except (IndexError, ValueError) as e:
            print(f"\nError: {str(e)}")
            return current_state
    
    return current_state

# Start the chatbot
if __name__ == "__main__":
    print("\nChatbot initialized! Type 'quit', 'exit', or 'q' to end the conversation.")
    print("Time Travel Commands:")
    print("  !history - Show conversation history")
    print("  !goto N  - Travel to state N\n")
    
    current_state = {"messages": []}
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                print("Assistant: Please type something to continue our conversation.")
                continue
            
            # Handle time travel commands
            if user_input.startswith("!"):
                current_state = handle_time_travel_command(user_input, current_state)
                continue
                
            config = {
                "configurable": {
                    "thread_id": "chat-1",
                    "checkpoint_ns": "",
                },
                "metadata": {
                    "run_name": "Chatbot Conversation",
                    "conversation_id": "chat-1"
                },
                "callbacks": [tracer]
            }
            
            try:
                # Add user message to current state
                current_state["messages"].append(HumanMessage(content=user_input))
                
                events = graph.stream(
                    current_state,
                    config,
                    stream_mode="values"
                )
                
                for event in events:
                    if "messages" in event:
                        message = event["messages"][-1]
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            print(f"\nTool Used: {message.tool_calls[0]['name']}")
                        
                        content = message.content
                        if isinstance(content, list):
                            content = next((item['text'] for item in content if item['type'] == 'text'), '')
                        
                        if content and content.strip() != user_input:
                            print(f"Assistant: {content}")
                        
                        # Update current state and save to history
                        current_state = event
                        conversation_history.save_state(current_state)
                print()
                        
            except Exception as e:
                if "GraphInterrupt" in str(e):
                    print("Assistant: I'm waiting for human input to provide expert guidance. Please provide your response.")
                else:
                    print(f"An error occurred: {e}")
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            break
