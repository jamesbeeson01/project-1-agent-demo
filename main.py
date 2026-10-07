from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

def main():
    agent = create_agent(
        model="google_genai:gemini-flash-lite-latest",
        tools=[],
        system_prompt="You are a helpful assistant",
        checkpointer=InMemorySaver(),
    )

    thread_config = {"configurable": {"thread_id": "1"}}   
    

    while True:
        print("----User----")
        prompt = input("You: ")

        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            thread_config
        )

        print("----AI----")
        print(response["messages"][-1].text)




if __name__ == "__main__":
    main()
