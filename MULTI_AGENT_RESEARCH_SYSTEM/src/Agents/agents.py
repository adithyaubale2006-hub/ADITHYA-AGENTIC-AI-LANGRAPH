from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import search_web, scrape_url
import os
from dotenv import load_dotenv

load_dotenv()

#MODEL CREATION
llm = ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',
    api_key=os.getenv("GEMINI_API_KEY"),
    max_tokens=1000,
    temperature=0
)

#Agent 1 : First Agent with Web Search
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[search_web]
    )

#Agent 2 : Reader Agent Scraper
def reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )




# writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful..."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual, and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()



#critic_chain

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line Verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()