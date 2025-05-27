# Simple-Agent Chatbot System

A simple-agent chatbot system built using LangGraph and LangChain, featuring time-travel capabilities and tool integration.

## Features

- Interactive chat interface with Claude 3 Sonnet model
- Time travel functionality to revisit previous conversation states
- Integrated tools including:
  - Tavily Search for web queries
  - Human assistance tool for expert guidance
- Conversation history management
- State tracking and checkpointing

## Prerequisites

- Python 3.8+
- Anthropic API key
- Tavily API key
- LangChain API key

## Environment Variables Required

```env
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langgraph_tutorial
```

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd langraph_tutorial
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the chatbot:
```bash
python chatbot.py
```

### Time Travel Commands
- `!history` - Show conversation history
- `!goto N` - Travel to state N
- Type 'quit', 'exit', or 'q' to end the conversation

## Project Structure

```
.
├── chatbot.py          # Main chatbot implementation
├── .env.example        # Example environment variables
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore rules
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
