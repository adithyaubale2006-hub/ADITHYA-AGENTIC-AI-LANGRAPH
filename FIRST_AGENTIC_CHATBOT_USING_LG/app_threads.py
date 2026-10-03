import os
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import uuid 




# Load environment variables
load_dotenv()

# Initialize Streamlit Page
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")

#generate unique thread id for different chats
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):

    #prevent the same thread to get add in multiple chats
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

st.title("LangGraph Chatbot")

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []


# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


st.sidebar.title("My Conversations")

# Cache the graph setup so it doesn't re-initialize on every Streamlit rerun
@st.cache_resource
def get_workflow():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash", # Note: Updated from gemini-3.6-flash to a valid model
        api_key=os.getenv("GEMINI_API_KEY"),
        max_tokens=1000,
    )

    def get_text_from_content(content) -> str:
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
        return {"messages": [AIMessage(content=get_text_from_content(response.content))]}

    graph = StateGraph(State)
    graph.add_node("prompt", prompt_node)
    graph.add_edge(START, "prompt")
    graph.add_edge("prompt", END)

    return graph.compile(checkpointer=MemorySaver()), get_text_from_content

workflow, get_text = get_workflow()
config = {"configurable": {"thread_id": "1"}}

# Initialize chat history for Streamlit UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Type here..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to UI chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response from LangGraph
        for chunk, metadata in workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") == "prompt":
                full_response += get_text(chunk.content)
                # Update placeholder with streaming text and a blinking cursor
                message_placeholder.markdown(full_response + "▌")
        
        # Final update to remove the cursor
        message_placeholder.markdown(full_response)
    
    # Add assistant response to UI chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})