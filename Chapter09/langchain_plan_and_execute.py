# ENV VARIABLES
import os
os.environ['SERPAPI_API_KEY'] = 'YOUR_SERPAPI_API_KEY_HERE'
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY_HERE'
# END ENV VARIABLES

# Define the model
from langchain.chat_models import ChatOpenAI
model = ChatOpenAI(temperature=0)

# Tools
from langchain.agents.tools import Tool

# Search tool
from langchain import SerpAPIWrapper

search = Tool(
    name='Search',
    func=SerpAPIWrapper().run,
    description='Google search tool')

# Calculator tool
from langchain import LLMMathChain

llm_math_chain = LLMMathChain.from_llm(
    llm=model,
    verbose=True)

calculator = Tool(
    name='Calculator',
    func=llm_math_chain.run,
    description='Calculator tool')

# Plan ane execute
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

agent = PlanAndExecute(
    planner=load_chat_planner(
        llm=model),
    executor=load_agent_executor(
        llm=model,
        tools=[search, calculator],
        verbose=True),
    verbose=True)

agent.run('What is the sum of the elevations of the deepest section of the ocean and the highest peak on Earth? Use metric units only.')
