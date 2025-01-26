from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API keys from .env file
openai_api_key = os.getenv('OPENAI_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize OpenAI and Groq models
openai = ChatOpenAI(model='gpt-4o-mini', openai_api_key=openai_api_key)
groq_llm = ChatGroq(model='llama-3.3-70b-versatile', groq_api_key=groq_api_key)

# Setup search tool
search_tool = TavilySearchResults(max_results=2, tavily_api_key=tavily_api_key)

# System prompt to guide the AI's behavior
system_prompt = """
You are an AI chatbot designed to assist users by providing helpful, friendly, and insightful responses. 
You have access to powerful language models and external tools to help you answer questions.
Your task is to respond in a way that is both informative and engaging, acting as a knowledgeable assistant.
Use the Tavily search tool to retrieve relevant information if needed.
Be concise, polite, and professional in your replies.
"""

# Create the agent with Groq model and Tavily search tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
def get_response_from_ai_agent(llm_id,query,allowed_search,system_promt,provider):
    if provider=='Groq':
        llm=ChatGroq(model=llm_id)
    elif provider=='OpenAI':
        llm=ChatOpenAI(model=llm_id)
    tools=[TavilySearchResults(max_results=2)] if allowed_search else []
    
    
        
# Create the react agent
    agent = create_react_agent(
        model=groq_llm,
        tools=[search_tool],
        state_modifier=system_prompt
    )



    # Set the state with the query
    state = {'messages': query}

    # Example interaction with the agent
    response = agent.invoke(state)
    messages = response.get('messages')

    # Get the content of the AI message
    ai_message = [message.content for message in messages if isinstance(message, AIMessage)]

    return ai_message[-1]  # This will print the last AI message
