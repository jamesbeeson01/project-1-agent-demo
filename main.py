import asyncio

# Suppresses a Google SDK warning - likely to change
import logging
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.mcp import MCPAdapter

CONFIG = {
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:/Users/DELL/Documents/Programming/project-1-agent-demo/agent-files"
      ]
    },
    "memory": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-memory"
      ],
      "env": {
        "MEMORY_FILE_PATH": "C:/Users/DELL/Documents/Programming/project-1-agent-demo/memory.jsonl"
      }
    },
    "docs-langchain": {
      "url": "https://docs.langchain.com/mcp"
    },
    "reference-langchain": {
      "url": "https://reference.langchain.com/mcp"
    }
  }
}

async def main():
    adapter = MCPAdapter(CONFIG)
    tools = await adapter.list_tools()

    agent = create_agent(
        model="google_genai:gemini-flash-lite-latest",
        tools=tools,
        system_prompt="You are a helpful assistant",
        checkpointer=InMemorySaver(),
    )

    thread_config = {"configurable": {"thread_id": "1"}}   
    

    while True:
        print("----User----")
        prompt = input("You: ")

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            thread_config
        )

        print("----AI----")
        print(response["messages"][-1].text)




if __name__ == "__main__":
    asyncio.run(main())
