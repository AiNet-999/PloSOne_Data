
!pip install swig
!pip install wrds
!pip install pyportfolioopt

!pip install -q condacolab
import condacolab
condacolab.install()
!apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
# from within a Jupyter cell
!pip install pandas_market_calendars

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
snp500_df = pd.read_html(wiki_url)[0]
tickers = snp500_df['Symbol'].to_list()

print(f"Number of tickers fetched: {len(tickers)}")

TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2018-01-01'
TEST_START_DATE = '2018-09-01'
TEST_END_DATE = '2024-09-01'

closing_prices = {}

for ticker in tickers:
    try:
        df = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE, ticker_list=[ticker]).fetch_data()
        if not df.empty:
            closing_prices[ticker] = df.set_index('date')['close']
            print(f"Successfully fetched data for {ticker}")
        else:
            print(f"No data found for {ticker}")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

if closing_prices:
    closing_prices_df = pd.DataFrame(closing_prices)
    closing_prices_df.index.name = 'Date'
    closing_prices_df.reset_index(inplace=True)
    closing_prices_df.to_csv("SP500_Closing_Prices.csv", index=False)
    print("Data saved to S&P500_Closing_Prices.csv")
else:
    print("No valid ticker data available for saving.")

