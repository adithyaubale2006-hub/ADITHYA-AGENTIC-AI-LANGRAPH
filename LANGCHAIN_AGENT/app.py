import os
import requests
from dotenv import load_dotenv

# LangChain & Tools Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langsmith import Client

# ==========================================
# 1. SETUP & API KEYS
# ==========================================
load_dotenv()
tavily_key = os.getenv("TAVILY_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
WEATHER_STACK_API = os.getenv("WEATHER_STACK_API")


# ==========================================
# 2. BUILD A WEATHER TOOL
# ==========================================
@tool
def get_weather_tool(city: str):
    """
    Fetch the current weather for the city.
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHER_STACK_API}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# ==========================================
# 3. IMPLEMENT TOOLS
# ==========================================
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool, get_weather_tool]


# ==========================================
# 4. INITIALIZE LLM (Gemini)
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",  # Verify this model name; usually it is "gemini-1.5-flash"
    api_key=gemini_key,
    max_tokens=1086,
    temperature=0.2
)


# ==========================================
# 5. BRING IN THE PROMPT
# ==========================================
# Initialize the LangSmith client
client = Client()

# Pull the prompt directly using the client
prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)


# ==========================================
# 6. CREATE THE AGENT
# ==========================================
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# ==========================================
# 7. EXECUTE THE AGENT
# ==========================================
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True 
)


# ==========================================
# 8. RUN
# ==========================================
response = agent_executor.invoke({
    "input": (
        "Find the capital of India "
        "and then find its current weather."
    )
})

print(response["output"])