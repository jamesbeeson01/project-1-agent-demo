from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = """
You are a dog. Congratulations! 

You are a good boy and love playing fetch. 

You speak like Scooby-Doo. 
"""

BALL_STATUS = "The ball is in owner's hand"

def fetch() -> str:
    """Go get the ball!"""
    global BALL_STATUS
    BALL_STATUS = "The ball is in dog's mouth"
    return "You got the ball! Good dog."

def give_ball() -> str:
    """Give ball to owner"""
    global BALL_STATUS
    BALL_STATUS = "The ball is in owner's hand"
    return "You gave the ball back."

def get_ball_status() -> str:
    """Where is the ball?"""
    return BALL_STATUS

DOG = create_agent(
    model="google_genai:gemini-flash-lite-latest",
    tools=[fetch, give_ball, get_ball_status],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)

thread_config = {"configurable": {"thread_id": "doggo"}}   

def talk_to_dog(prompt: str) -> str:
    """Talk to the dog and get a response. `prompt` is what you tell the dog."""
    response = DOG.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        thread_config
    )

    return response["messages"][-1].text