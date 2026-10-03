import os
from typing import TypedDict, Annotated


from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    max_tokens=1000,
)


def get_text(content) -> str:
    """Return plain text whether content is a string or a list of blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]


def prompt_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=get_text(response.content))]}


graph = StateGraph(State)
graph.add_node("prompt", prompt_node)
graph.add_edge(START, "prompt")
graph.add_edge("prompt", END)

workflow = graph.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}

while True:
    user_message = input("\nYou: ").strip()

    if user_message.lower() in ["exit", "quit", "bye"]:
        break
    if not user_message:
        continue

    print("AI: ", end="", flush=True)

    for chunk, metadata in workflow.stream(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "prompt":
            print(get_text(chunk.content), end="", flush=True)

    print()