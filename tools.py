"""
Util that calls several of Polygon's stock market REST APIs.
Docs: https://polygon.io/docs/stocks/getting-started
"""

import requests
import json
from typing import Any, Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, root_validator, Field
from langchain_core.utils import get_from_dict_or_env
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

POLYGON_BASE_URL = "https://api.polygon.io/"

#load the vectorstore and initialize the retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local("./data", embeddings, allow_dangerous_deserialization=True)

class PolygonAPIWrapper(BaseModel):
    """Wrapper for Polygon API."""

    polygon_api_key: Optional[str] = None

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key in environment."""
        polygon_api_key = get_from_dict_or_env(
            values, "polygon_api_key", "POLYGON_API_KEY"
        )
        values["polygon_api_key"] = polygon_api_key

        return values

    def get_financials(self, ticker: str) -> Optional[dict]:
        """
        Get fundamental financial data, which is found in balance sheets,
        income statements, and cash flow statements for a given ticker.

        /vX/reference/financials
        """
        url = (
            f"{POLYGON_BASE_URL}vX/reference/financials?"
            f"ticker={ticker}&"
            f"apiKey={self.polygon_api_key}"
        )
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status != "OK":
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)

    def get_last_quote(self, ticker: str) -> Optional[dict]:
        """
        Get the most recent National Best Bid and Offer (Quote) for a ticker.

        /v2/last/nbbo/{ticker}
        """
        url = f"{POLYGON_BASE_URL}v2/last/nbbo/{ticker}?apiKey={self.polygon_api_key}"
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status != "OK":
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)

    def get_ticker_news(self, ticker: str) -> Optional[dict]:
        """
        Get the most recent news articles relating to a stock ticker symbol,
        including a summary of the article and a link to the original source.

        /v2/reference/news
        """
        url = (
            f"{POLYGON_BASE_URL}v2/reference/news?"
            f"ticker={ticker}&"
            f"apiKey={self.polygon_api_key}"
        )
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status != "OK":
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)

    def get_aggregates(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get aggregate bars for a stock over a given date range
        in custom time window sizes.

        /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}
        """
        timespan = kwargs.get("timespan", "day")
        multiplier = kwargs.get("timespan_multiplier", 1)
        from_date = kwargs.get("from_date", None)
        to_date = kwargs.get("to_date", None)
        adjusted = kwargs.get("adjusted", True)
        sort = kwargs.get("sort", "asc")

        url = (
            f"{POLYGON_BASE_URL}v2/aggs"
            f"/ticker/{ticker}"
            f"/range/{multiplier}"
            f"/{timespan}"
            f"/{from_date}"
            f"/{to_date}"
            f"?apiKey={self.polygon_api_key}"
            f"&adjusted={adjusted}"
            f"&sort={sort}"
        )
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        
        return data.get("results", None)

    def run(self, mode: str, ticker: str, **kwargs: Any) -> str:
        if mode == "get_financials":
            return json.dumps(self.get_financials(ticker))
        elif mode == "get_last_quote":
            return json.dumps(self.get_last_quote(ticker))
        elif mode == "get_ticker_news":
            return json.dumps(self.get_ticker_news(ticker))
        elif mode == "get_aggregates":
            return json.dumps(self.get_aggregates(ticker, **kwargs))
        else:
            raise ValueError(f"Invalid mode {mode} for Polygon API.")

class PolygonAggregatesSchema(BaseModel):
    """Input for PolygonAggregates."""

    ticker: str = Field(
        description="The ticker symbol to fetch aggregates for.",
    )
    timespan: str = Field(
        description="The size of the time window. "
        "Possible values are: "
        "second, minute, hour, day, week, month, quarter, year. "
        "Default is 'day'",
    )
    timespan_multiplier: int = Field(
        description="The number of timespans to aggregate. "
        "For example, if timespan is 'day' and "
        "timespan_multiplier is 1, the result will be daily bars. "
        "If timespan is 'day' and timespan_multiplier is 5, "
        "the result will be weekly bars.  "
        "Default is 1.",
    )
    from_date: str = Field(
        description="The start of the aggregate time window. "
        "Either a date with the format YYYY-MM-DD or "
        "a millisecond timestamp.",
    )
    to_date: str = Field(
        description="The end of the aggregate time window. "
        "Either a date with the format YYYY-MM-DD or "
        "a millisecond timestamp.",
    )


class PolygonAggregates(BaseTool):
    """
    Tool that gets aggregate bars (stock prices) over a
    given date range for a given ticker from Polygon.
    """

    mode: str = "get_aggregates"
    name: str = "polygon_aggregates"
    description: str = (
        "A wrapper around Polygon's Aggregates API. "
        "This tool is useful for fetching aggregate bars (stock prices) for a ticker. "
        "Input should be the ticker, date range, timespan, and timespan multiplier"
        " that you want to get the aggregate bars for. This should be done when "
        "attempting to retreive the current stock price for any valuation calculation."
    )
    args_schema: Type[PolygonAggregatesSchema] = PolygonAggregatesSchema

    api_wrapper: PolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        timespan: str,
        timespan_multiplier: int,
        from_date: str,
        to_date: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            timespan=timespan,
            timespan_multiplier=timespan_multiplier,
            from_date=from_date,
            to_date=to_date,
        )
    
class FinancialInputs(BaseModel):
    """Inputs for Polygon's Financials API"""

    query: str


class PolygonFinancials(BaseTool):
    """Tool that gets the financials of a ticker from Polygon"""

    mode: str = "get_financials"
    name: str = "polygon_financials"
    description: str = (
        "A wrapper around Polygon's Stock Financials API. "
        "This tool is useful for fetching fundamental financials from "
        "balance sheets, income statements, and cash flow statements "
        "for a stock ticker. The input should be the ticker that you want "
        "to get the latest fundamental financial data for. "
        "If a duckduckgo_search call returns a list of stocks, "
        "The tickers for those stocks can be passed to this function to "
        "retreive financial data and provide an accurate response "
        "to the user."
    )
    args_schema: Type[BaseModel] = FinancialInputs

    api_wrapper: PolygonAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(self.mode, ticker=query)
    
class TickerInputs(BaseModel):
    """Inputs for Polygon's Ticker News API"""

    query: str

class PolygonTickerNews(BaseTool):
    """Tool that gets the latest news for a given ticker from Polygon"""

    mode: str = "get_ticker_news"
    name: str = "polygon_ticker_news"
    description: str = (
        "A wrapper around Polygon's Ticker News API. "
        "This tool is useful for fetching the latest news for a stock. "
        "Input should be the ticker that you want to get the latest news for."
    )
    args_schema: Type[BaseModel] = TickerInputs

    api_wrapper: PolygonAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(self.mode, ticker=query)
    
class RAGInput(BaseModel):
    """Input for the local data retrieval tool."""

    query: str = Field(description="retreive data from vectorstore")

class RAGAgent(BaseTool):
    """Tool that retrieves information from a local vectorstore of financial reports. These reports include 
    11-K, 10-K, 10-Q, 8-K, and SD filings from all of the S&P 500 companies, as of July 18th, 2024. Whenever specific financial 
    information is requested in the query, ensure to use the Condensed Consolidated Statements of Cash Flows sections of the financial 
    documents."""

    name: str = "vectorstore_retrieval"
    description: str = (
        """Tool that retrieves information from a local vectorstore of financial reports. These reports include 
    11-K, 10-K, 10-Q, and SD (Special disclosure) filings from all of the S&P 500 companies, as of July 18th, 2024. The information in this 
    vector store only pertains to fiscal years 2023 and 2024. Whenever specific financial 
    information is requested in the query, ensure to use the phrase 'Financial Information'. 
    If necessary, modify the user query to contain this phrase."""
    )
    args_schema: Type[BaseModel] = RAGInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Retrieve data from the vectorstore."""
        retriever = db.as_retriever()
        return retriever.invoke(query)
