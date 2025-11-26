import time
import logging
from data_fetcher import DataFetcher
from sentiment_engine import SentimentEngine
from signal_generator import SignalGenerator
from monitor import Monitor
from ml_predictor import MLPredictor
from market_aggregator import MarketAggregator

logger = logging.getLogger(__name__)

class TradeAgent:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.sentiment_engine = SentimentEngine()
        self.signal_generator = SignalGenerator()
        self.monitor = Monitor()
        self.ml_predictor = MLPredictor()
        self.market_aggregator = MarketAggregator()
        
    def run_cycle(self):
        """
        Runs one cycle of the agent: Fetch -> Score -> Signal -> Monitor -> Output
        """
        logger.info("Starting Agent Cycle...")
        
        # 1. Fetch Data
        # Fetch Multi-Timeframe Data for Signals
        mtf_data = self.data_fetcher.fetch_multi_timeframe_data()
        market_data = mtf_data['1d'] # Use daily for macro/monitor
        
        latest_prices = self.data_fetcher.fetch_latest_price()
        news_items = self.data_fetcher.fetch_news()
        econ_data = self.data_fetcher.fetch_economic_data()
        
        # 2. Score Sentiment
        news_score, news_summary = self.sentiment_engine.compute_news_score(news_items)
        econ_score, econ_label = self.sentiment_engine.compute_economic_score(econ_data)
        macro_score, macro_label = self.sentiment_engine.compute_macro_score(market_data)
        
        sentiment_scores = {
            'news': news_score,
            'economic': econ_score,
            'macro': macro_score
        }
        
        # 3. Generate Signals & Monitor
        results = {}
        
        # Global Risk Alerts
        global_alerts = self.monitor.monitor_risk(market_data, sentiment_scores)
        
        for ticker_name in self.data_fetcher.tickers.keys():
            if ticker_name in ['VIX', 'DXY']: 
                continue
                
            # ML Prediction
            # We use the Daily data for ML training/prediction
            df_daily = mtf_data['1d'].get(ticker_name)
            ml_prediction = 0.0
            ml_insights = None
            indicators = {}
            ml_context = None
            
            if df_daily is not None and not df_daily.empty:
                # Ensure indicators are present for ML and UI
                df_daily = self.signal_generator.calculate_technical_indicators(df_daily)
                
                # Get detailed ML prediction
                ml_result = self.ml_predictor.get_detailed_prediction(df_daily)
                if ml_result:
                    ml_prediction = ml_result['prediction']
                    ml_insights = ml_result
                    ml_context = {
                        'prediction': ml_prediction,
                        'confidence': ml_result['confidence']
                    }
                
                # Extract latest indicators for UI
                latest = df_daily.iloc[-1]
                indicators = {
                    'RSI': round(float(latest.get('RSI', 0)), 2),
                    'SMA_50': round(float(latest.get('SMA_50', 0)), 2),
                    'SMA_200': round(float(latest.get('SMA_200', 0)), 2),
                    'MACD': round(float(latest.get('MACD_Line', 0)), 2),
                    'MACD_Signal': round(float(latest.get('MACD_Signal', 0)), 2),
                    'BB_Upper': round(float(latest.get('BB_Upper', 0)), 2),
                    'BB_Lower': round(float(latest.get('BB_Lower', 0)), 2)
                }

            # Generate Signal (Now with ML Context)
            signal, bias, confidence, explanation = self.signal_generator.generate_signal(
                ticker_name, mtf_data, sentiment_scores, ml_context=ml_context
            )
            
            results[ticker_name] = {
                'price': latest_prices.get(ticker_name),
                'signal': signal,
                'bias': bias,
                'confidence': confidence,
                'explanation': explanation,
                'ml_prediction': ml_prediction,
                'ml_insights': ml_insights,  # NEW: Full ML insights
                'indicators': indicators
            }
            
        # Calculate aggregate market score
        report = {
            'sentiment': sentiment_scores,
            'sentiment_labels': {'news': news_summary, 'economic': econ_label, 'macro': macro_label},
            'alerts': global_alerts,
            'signals': results
        }
        
        # Add aggregate market outlook
        market_aggregate = self.market_aggregator.calculate_aggregate_score(report)
        report['market_aggregate'] = market_aggregate
        
        return report

    def print_report(self, report):
        print("\n" + "="*50)
        print(f"AUTONOMOUS TRADE AGENT REPORT - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        print(f"\nSENTIMENT SCORES:")
        print(f"  * Economic: {report['sentiment']['economic']} ({report['sentiment_labels']['economic']})")
        print(f"  * News:     {report['sentiment']['news']} ({report['sentiment_labels']['news'][:50]}...)")
        print(f"  * Macro:    {report['sentiment']['macro']} ({report['sentiment_labels']['macro']})")
        
        if report['alerts']:
            print(f"\nRISK ALERTS:")
            for alert in report['alerts']:
                print(f"  * {alert}")
        else:
            print(f"\nNO MAJOR RISK ALERTS")
            
        print(f"\nTRADING SIGNALS:")
        for ticker, data in report['signals'].items():
            price = data['price']
            price_str = f"{price:.2f}" if price else "N/A"
            print(f"\n  > {ticker} (Price: {price_str})")
            print(f"     Signal: {data['signal']} | Bias: {data['bias']} | Conf: {data['confidence']}%")
            print(f"     Reason: {data['explanation']}")
            
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    agent = TradeAgent()
    report = agent.run_cycle()
    agent.print_report(report)
