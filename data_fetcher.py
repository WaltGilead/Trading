import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.tickers = {
            # Major Stocks
            'AAPL': 'AAPL',
            'GOOGL': 'GOOGL',
            'MSFT': 'MSFT',
            'TSLA': 'TSLA',
            'AMZN': 'AMZN',
            # Market Indices & Commodities
            'BTC/USD': 'BTC-USD',
            'US30': '^DJI',
            'US100': '^IXIC', 
            'XAU/USD': 'GC=F', 
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB' 
        }
        
    def fetch_market_data(self, period="1mo", interval="1d"):
        """
        Fetches historical market data for all tracked tickers.
        """
        data = {}
        for name, symbol in self.tickers.items():
            try:
                logger.info(f"Fetching {interval} data for {name} ({symbol})...")
                ticker = yf.Ticker(symbol)
                history = ticker.history(period=period, interval=interval)
                
                if history.empty:
                    logger.warning(f"No data found for {name} ({symbol})")
                    continue
                    
                # Calculate some basic stats to ensure data quality
                history['Returns'] = history['Close'].pct_change()
                data[name] = history
                
            except Exception as e:
                logger.error(f"Error fetching data for {name}: {e}")
                
        return data

    def fetch_multi_timeframe_data(self):
        """
        Fetches 1h and 1d data for all tickers.
        Returns a dict: {'1h': {ticker: df}, '1d': {ticker: df}}
        """
        data = {'1h': {}, '1d': {}}
        
        for ticker_name, ticker_symbol in self.tickers.items():
            try:
                # 1 Hour Data
                # Note: yfinance limitation, 1h data only available for last 730 days. 
                # We fetch 1mo to be safe and fast.
                df_1h = yf.download(ticker_symbol, period="1mo", interval="60m", progress=False)
                if not df_1h.empty:
                    if isinstance(df_1h.columns, pd.MultiIndex):
                        df_1h.columns = df_1h.columns.get_level_values(0)
                    data['1h'][ticker_name] = df_1h
                
                # Daily Data
                df_1d = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
                if not df_1d.empty:
                    if isinstance(df_1d.columns, pd.MultiIndex):
                        df_1d.columns = df_1d.columns.get_level_values(0)
                    data['1d'][ticker_name] = df_1d
                    
            except Exception as e:
                logger.error(f"Error fetching MTF data for {ticker_name}: {e}")
                
        return data

    def fetch_latest_price(self):
        """
        Fetches the latest price for all tickers.
        """
        prices = {}
        for name, symbol in self.tickers.items():
            try:
                ticker = yf.Ticker(symbol)
                # Try fast_info first
                if hasattr(ticker, 'fast_info') and ticker.fast_info.last_price:
                     prices[name] = ticker.fast_info.last_price
                else:
                    # Fallback to history
                    history = ticker.history(period="1d")
                    if not history.empty:
                        prices[name] = history['Close'].iloc[-1]
                    else:
                         prices[name] = None
            except Exception as e:
                logger.error(f"Error fetching price for {name}: {e}")
                prices[name] = None
        return prices

    def fetch_news(self):
        """
        Fetches news for the tickers.
        """
        all_news = []
        for name, symbol in self.tickers.items():
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                for item in news:
                    # Handle different yfinance news formats
                    title = item.get('title')
                    if not title:
                        # Try 'content' or other keys if title is missing
                        title = item.get('content', {}).get('title')
                        
                    if title:
                        all_news.append({
                            'title': title,
                            'link': item.get('link'),
                            'publisher': item.get('publisher'),
                            'related_ticker': name,
                            'published': item.get('providerPublishTime')
                        })
            except Exception as e:
                logger.error(f"Error fetching news for {name}: {e}")
        return all_news

    def fetch_economic_data(self):
        """
        Fetches key economic indicators from FRED.
        """
        series_ids = {
            'CPI': 'CPIAUCSL',      # Consumer Price Index
            'Unemployment': 'UNRATE', # Unemployment Rate
            'FedFunds': 'FEDFUNDS'   # Federal Funds Rate
        }
        
        econ_data = {}
        # Fetch last 2 years to ensure we have enough for YoY calcs
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        for name, series_id in series_ids.items():
            try:
                logger.info(f"Fetching {name} from FRED...")
                df = web.DataReader(series_id, 'fred', start_date)
                econ_data[name] = df
            except Exception as e:
                logger.error(f"Error fetching {name} from FRED: {e}")
                econ_data[name] = None
                
        return econ_data

if __name__ == "__main__":
    # Test the fetcher
    fetcher = DataFetcher()
    print("Fetching Market Data...")
    market_data = fetcher.fetch_market_data(period="5d")
    for ticker, df in market_data.items():
        print(f"\n{ticker} Data Head:")
        print(df.head())
