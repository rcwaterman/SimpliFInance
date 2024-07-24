import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from tools import PolygonAPIWrapper, PolygonAggregates, PolygonFinancials, PolygonTickerNews, RAGAgent
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.prebuilt import ToolExecutor
from langchain.tools import tool
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import FunctionMessage, HumanMessage
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import BaseMessage
import datetime
import yfinance as yf
import pandas as pd
import requests

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']
POLYGON_API_KEY = os.environ['POLYGON_API_KEY']
FMP_API_KEY = os.environ['FMP_API_KEY']

#-----DEFINE ADDITIONAL TOOLS AND FUNCTIONS-----#

@tool
def get_datetime() -> str:
    """Get the current date and time in YYYY-MM-DD HH:MM:SS format."""
    return str(datetime.datetime.now())

@tool
def get_date() -> str:
    """Get the current date in YYYY-MM-DD format. Also useful when determining the current quarter."""
    return str(datetime.datetime.now()).split(" ")[0]

@tool
def get_time() -> str:
    """Get the current time in HH:MM:SS format."""
    return str(datetime.datetime.now()).split(" ")[1]

@tool
def get_quarter(date:str) -> str:
    """This tool takes a date in YYYY-MM-DD format as an argument and returns the quarter and year in the format 'QQ YYYY'."""
    quarters = {
        "01" : "Q1",
        "02" : "Q1",
        "03" : "Q1",
        "04" : "Q2",
        "05" : "Q2",
        "06" : "Q2",
        "07" : "Q3",
        "08" : "Q3",
        "09" : "Q3",
        "10" : "Q4",
        "11" : "Q4",
        "12" : "Q4",
    }
    return quarters[date.split("-")[1]] + f" {date.split('-')[0]}"    

@tool
def calculate_percent_valuation(intrinsic_value:float, current_stock_price:float) -> float:
    """This tool can be used to calculate how overvalued or undervalued a stock is. It takes the calculated intrinsic value and the current stock price as arguments and returns the valuation percentage, in a format similar to '0.50' for 50%. The
    math performed by this function is (intrinsic_value-current_stock_price)/abs(intrinsic_value). The current stock price must be retrieved using the 'get_date' tool (to get the current date) and then using that date to access the 'polygon_aggregates' tool. 
    A positive percentage indicates an undervalued stock and a negative percentage indicates an overvalued stock."""
    return (intrinsic_value-current_stock_price)/abs(intrinsic_value)

@tool
def calculate_intrinsic_value(ticker:str, average_growth_rate):
    """This tool is helpful for calculating the intrinsic value of a stock. It takes the stock ticker, the average growth rate based on revenue (retrieved from financial reports or with the polygon API. This should be capped at plus or minus 300% per year.)"""
    wacc = calculate_wacc(ticker)

@tool
def calculate_wacc( #refer to https://www.gurufocus.com/term/wacc/SOFI#:~:text=SoFi%20Technologies%20WACC%20%25%20Calculation,the%20firm's%20cost%20of%20capital.
    ticker:str, 
    market_cap:float, 
    interest_expense:float, 
    tax_expense:float, 
    pre_tax_income:float, 
    long_term_debt:float
    ):
    """This tool is used to determine the weighted average cost of capital (WACC) when performing a DCF analysis. It takes the following arguments:

    ticker
    market capitalization - The market capitalization should be retrieved using the duckduckgo_search tool. Explicitly state 'nvidia market cap today'
    interest expense - trailing twelve month interest expense calculated from the response of the polygon_financials tool.
    tax expense - trailing twelve month tax expense calculated from the response of the polygon_financials tool.
    pre-tax income -  trailing twelve month pre-tax income calculated from the response of the polygon_financials tool.
    long term debt - long term debt calculated from the response of the polygon_financials tool.

    WACC is returned as a percentage in the format '0.057'."""
    
    treasury_yield10 = yf.Ticker("^TNX") 
    risk_free_rate = round(treasury_yield10.info['regularMarketPreviousClose']/100,2) 
    sp500_teturn = 0.10
    stock = yf.Ticker(f"{ticker}")
    beta = stock.info["beta"]

    cost_of_equity = round(risk_free_rate + beta*(sp500_teturn - risk_free_rate),2)
    weight_of_equity, weight_of_debt = get_weights(market_cap, long_term_debt)
    cost_of_debt = get_cost_of_debt(interest_expense, long_term_debt)
    tax_rate = get_tax_rate(tax_expense, pre_tax_income)
    wacc = round((weight_of_equity * cost_of_equity) + ((weight_of_debt * cost_of_debt ) * (1-tax_rate)),3)
    return wacc

@tool
def get_dcf(ticker:str) -> float:
    """This tool takes a single stock ticker as an argument and returns the discounted cash flow valuation, in dollars. This tool is helpful when trying to determine the intrinsic value of a company, or if a company is overvalued or undervalued."""
    url = f'https://financialmodelingprep.com/api/v3/discounted-cash-flow/{ticker}?apikey={FMP_API_KEY}'

    # Make the request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        dcf_data = response.json()[0]
        # return the DCF data
        return dcf_data["dcf"]
    else:
        # return the error message
        return f"Failed to retrieve data: {response.status_code}"

def get_weights(market_cap, long_term_debt):
    e = market_cap
    d = long_term_debt
    weight_of_equity = e/(e+d)
    weight_of_debt = d/(e+d)
    return weight_of_equity, weight_of_debt

def get_cost_of_debt(interest_expense, long_term_debt) -> float:
    return interest_expense/long_term_debt

def get_tax_rate(tax_expense, pre_tax_income):
    tax_rate = tax_expense/pre_tax_income
    if tax_rate>1:
        return 1.00
    if tax_rate<0:
        return 0.00
    return tax_rate

def get_wacc(ticker):
    treasury_yield10 = yf.Ticker("^TNX") 
    risk_free_rate = round(treasury_yield10.info['regularMarketPrice']/100,2)
    sp500_teturn = 0.10
    stock = yf.Ticker(f"{ticker}")
    beta = stock.info["beta"]
    cost_of_equity = round(risk_free_rate + beta*(sp500_teturn - risk_free_rate),2)
    stock_bal = stock.balance_sheet

#-----CREATE TOOL BELT AND EXECUTOR-----#
api_wrapper = PolygonAPIWrapper(polygon_api_key=POLYGON_API_KEY)

tool_belt = [
    get_datetime,
    get_date,
    get_time,
    get_quarter,
    calculate_percent_valuation,
    get_dcf,
    RAGAgent(),
    DuckDuckGoSearchRun(),
    PolygonAggregates(api_wrapper=api_wrapper),
    PolygonFinancials(api_wrapper=api_wrapper),
    PolygonTickerNews(api_wrapper=api_wrapper),
]

tool_executor = ToolExecutor(tool_belt)

#-----INSTANTIATE MODEL AND BIND FUNCTIONS-----#

model = ChatOpenAI(model="gpt-4o", temperature=0)

functions = [convert_to_openai_function(t) for t in tool_belt]
model = model.bind_functions(functions)

#-----INSTANTIATE AGENT-----#

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

#-----CREATE NODES-----#
def call_model(state):
  messages = state["messages"]
  response = model.invoke(messages)
  return {"messages" : [response]}

def call_tool(state):
  last_message = state["messages"][-1]

  action = ToolInvocation(
      tool=last_message.additional_kwargs["function_call"]["name"],
      tool_input=json.loads(
          last_message.additional_kwargs["function_call"]["arguments"]
      )
  )

  response = tool_executor.invoke(action)

  function_message = FunctionMessage(content=str(response), name=action.tool)

  return {"messages" : [function_message]}

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("use tool", call_tool)
workflow.set_entry_point("agent")

def should_continue(state):
  last_message = state["messages"][-1]

  if "function_call" not in last_message.additional_kwargs:
    return "end"

  return "continue"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue" : "use tool",
        "end" : END
    }
)

@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "SimpliFinance"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():

    workflow.add_edge("use tool", "agent")
    app = workflow.compile()

    cl.user_session.set("agent", app)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")

    if message.elements:
        dfs = [read_csv(file.path) for file in message.elements if "csv" in file.mime]
        message_to_send = {"messages": HumanMessage(content=message.content + '\nThe following is a pandas dataframe containing information that is relevant to the preceeding message:\n' + dfs[0])}
    else:
        message_to_send = {"messages": HumanMessage(content=message.content)}

    async for chunk in agent.astream(
        message_to_send,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        for key in chunk:
            if key == "agent" and chunk[key].get("messages")[0].content is not None:
                await msg.stream_token(chunk[key].get("messages")[0].content)

    await msg.send()

def read_csv(csv):
    df = pd.read_csv(csv,dtype=str)
    return df