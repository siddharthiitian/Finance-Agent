from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

web_search_agent = Agent(
name = 'web-search-agent',
role = 'Search Web for the infomation',
model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
tools = [DuckDuckGo()],
instructions=['Always include the source of the information'],
markdown=True,
)


financial_agent = Agent(
 name = 'financial-agent',
 model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
 tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
 instructions=['Use tables to display the data'],
 show_tool_calls=True,
 markdown=True,
)
 
multi_agent = Agent(
team=[web_search_agent, financial_agent],
instructions=['Always include the source of the information','Use tables to display the data'],
show_tool_calls=True,
markdown=True,
)

multi_agent.print_response('Summarize analyst recommendations and share the latest news for AAPL stock',stream=True)