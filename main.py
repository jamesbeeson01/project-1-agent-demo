from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent

def main():
    agent = create_agent(
        model="google_genai:gemini-flash-lite-latest",
        tools=[],
        system_prompt="You are a helpful assistant",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello there?"}]}
    )
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
