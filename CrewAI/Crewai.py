import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents.react.agent import create_react_agent
from crewai import Agent, Crew, Task
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai.tools import tool

from langchain.chat_models import ChatLiteLLM

HUGGINGFACE_API_KEY=os.environ["HUGGINGFACE_API_KEY"]="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"

llm = ChatLiteLLM(model="huggingface/mistralai/Mistral-7B-Instruct-v0.1")

# Pinecone configuration
TAVILY_API_KEY=os.environ["TAVILY_API_KEY"]="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"
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"] ="pcsk_6L8ZmY_fh3AkwzUNVg87rneaKU2HgoApGK91wJJoVoYtX4HSdnKdxaWpuktSyjkApsBeZ"
index_name = "agenticragmodel"
embed = embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(
    index_name=index_name,
    namespace="main",
    embedding=embed
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=1,
    return_messages=True
)

qa_db = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

@tool("Pinecone Document Store")
def pinecone_tool(query: str) -> str:
    """Searches Pinecone vector DB using the RetrievalQA chain."""
    return qa_db.run(query)

tavily = TavilySearchResults(max_results=10, tavily_api_key=os.getenv("TAVILY_API_KEY"))
@tool("Tavily")
def tavily_tool(query: str) -> str:
    """Searches the web using Tavily for up-to-date information."""
    return tavily.run(query)


research_agent = Agent(
    role="Research Analyst",
    goal="Search and analyze information from TED talks database and online sources",
    backstory="Expert in researching educational and scientific talks with a focus on extracting meaningful insights.",
    tools=[pinecone_tool, tavily_tool],
    llm=llm,  
    verbose=True
)

review_agent = Agent(
    role="Content Verifier",
    goal="Verify and summarize the accuracy of the retrieved TED talk information",
    backstory="Specializes in reviewing and validating research content with references.",
    tools=[pinecone_tool],
    llm=llm, 
    verbose=True
)

task = Task(
    description="Find the title of Al Gore’s TED talk and summarize it in 2 sentences.",
    expected_output="The title of the TED talk by Al Gore and a short summary with sources or citations.",
    agent=research_agent
)

crew = Crew(
    agents=[research_agent, review_agent],
    tasks=[task],
    manager_llm=llm
)

result = crew.kickoff()
print("\n Final Result:")
print(result)