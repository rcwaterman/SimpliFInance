import requests
import json
import os
from dotenv import find_dotenv, dotenv_values

keys = list(dotenv_values(find_dotenv('.env')).items())
FMP_API_KEY = os.environ['FMP_API_KEY'] = keys[0][1]

def get_dcf(ticker:str):
    """This tool takes a stock ticker as an argument and returns the discounted cash flow valuation, in dollars. This tool is helpful when trying to determine the intrinsic value of a company, or if a company is overvalued or undervalued."""
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

data = get_dcf("AAPL")
print(data)