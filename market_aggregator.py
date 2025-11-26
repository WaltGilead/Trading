"""
Market Aggregator - Compiles overall market sentiment from all sources
"""
import numpy as np
from typing import Dict, List, Tuple

class MarketAggregator:
    def __init__(self):
        pass
    
    def calculate_aggregate_score(self, report: dict) -> dict:
        """
        Calculates an aggregate market score from all available data sources.
        
        Returns a dict with:
        - overall_score: 0-100 (0=very bearish, 50=neutral, 100=very bullish)
        - outlook: "Bullish", "Bearish", or "Neutral"
        - confidence: 0-100
        - breakdown: dict showing contribution of each factor
        """
        scores = []
        weights = []
        breakdown = {}
        
        # 1. SENTIMENT SCORES (Weight: 25%)
        sentiment = report.get('sentiment', {})
        sentiment_avg = (
            sentiment.get('economic', 50) + 
            sentiment.get('news', 50) + 
            sentiment.get('macro', 50)
        ) / 3
        scores.append(sentiment_avg)
        weights.append(0.25)
        breakdown['Sentiment'] = {
            'score': sentiment_avg,
            'weight': '25%',
            'status': self._get_status(sentiment_avg)
        }
        
        # 2. SIGNAL AGGREGATION (Weight: 35%)
        signals = report.get('signals', {})
        signal_scores = []
        buy_count = 0
        sell_count = 0
        neutral_count = 0
        
        for asset, data in signals.items():
            signal = data.get('signal', 'NEUTRAL')
            confidence = data.get('confidence', 50)
            
            # Convert signal to score
            if signal == 'BUY':
                signal_scores.append(70 + (confidence - 50) * 0.6)  # 70-88
                buy_count += 1
            elif signal == 'SELL':
                signal_scores.append(30 - (confidence - 50) * 0.6)  # 12-30
                sell_count += 1
            else:  # NEUTRAL
                bias = data.get('bias', 'Neutral')
                if bias == 'Bullish':
                    signal_scores.append(55)
                elif bias == 'Bearish':
                    signal_scores.append(45)
                else:
                    signal_scores.append(50)
                neutral_count += 1
        
        signal_avg = np.mean(signal_scores) if signal_scores else 50
        scores.append(signal_avg)
        weights.append(0.35)
        breakdown['Signals'] = {
            'score': signal_avg,
            'weight': '35%',
            'status': self._get_status(signal_avg),
            'details': f"{buy_count} BUY, {sell_count} SELL, {neutral_count} NEUTRAL"
        }
        
        # 3. TECHNICAL INDICATORS (Weight: 25%)
        tech_scores = []
        for asset, data in signals.items():
            indicators = data.get('indicators', {})
            if indicators:
                # RSI scoring
                rsi = indicators.get('RSI', 50)
                if rsi > 70:
                    rsi_score = 30  # Overbought = bearish
                elif rsi < 30:
                    rsi_score = 70  # Oversold = bullish
                else:
                    rsi_score = 50 + (rsi - 50) * 0.4
                tech_scores.append(rsi_score)
                
                # MACD scoring
                macd = indicators.get('MACD', 0)
                macd_signal = indicators.get('MACD_Signal', 0)
                if macd > macd_signal:
                    tech_scores.append(65)  # Bullish cross
                else:
                    tech_scores.append(35)  # Bearish cross
        
        tech_avg = np.mean(tech_scores) if tech_scores else 50
        scores.append(tech_avg)
        weights.append(0.25)
        breakdown['Technicals'] = {
            'score': tech_avg,
            'weight': '25%',
            'status': self._get_status(tech_avg)
        }
        
        # 4. ML PREDICTIONS (Weight: 15%)
        ml_scores = []
        for asset, data in signals.items():
            ml_pred = data.get('ml_prediction', 0)
            ml_insights = data.get('ml_insights')
            
            if ml_insights:
                confidence = ml_insights.get('confidence', 50)
                # Convert prediction to score (predictions are typically -0.05 to +0.05)
                pred_score = 50 + (ml_pred * 1000)  # Scale to 0-100
                pred_score = max(0, min(100, pred_score))  # Clamp
                
                # Weight by confidence
                weighted_pred = pred_score * (confidence / 100)
                ml_scores.append(weighted_pred)
        
        ml_avg = np.mean(ml_scores) if ml_scores else 50
        scores.append(ml_avg)
        weights.append(0.15)
        breakdown['ML Predictions'] = {
            'score': ml_avg,
            'weight': '15%',
            'status': self._get_status(ml_avg)
        }
        
        # Calculate weighted average
        overall_score = sum(s * w for s, w in zip(scores, weights))
        overall_score = max(0, min(100, overall_score))  # Clamp to 0-100
        
        # Determine outlook
        if overall_score >= 60:
            outlook = "Bullish"
            confidence = min(95, 60 + (overall_score - 60) * 0.875)
        elif overall_score <= 40:
            outlook = "Bearish"
            confidence = min(95, 60 + (40 - overall_score) * 0.875)
        else:
            outlook = "Neutral"
            confidence = 50 + abs(50 - overall_score) * 0.5
        
        return {
            'overall_score': round(overall_score, 1),
            'outlook': outlook,
            'confidence': round(confidence, 1),
            'breakdown': breakdown,
            'signal_distribution': {
                'buy': buy_count,
                'sell': sell_count,
                'neutral': neutral_count
            }
        }
    
    def _get_status(self, score: float) -> str:
        """Convert score to status label"""
        if score >= 60:
            return "Bullish"
        elif score <= 40:
            return "Bearish"
        else:
            return "Neutral"

if __name__ == "__main__":
    # Test with sample data
    sample_report = {
        'sentiment': {'economic': 65, 'news': 70, 'macro': 60},
        'signals': {
            'AAPL': {'signal': 'BUY', 'confidence': 80, 'bias': 'Bullish', 
                     'indicators': {'RSI': 55, 'MACD': 3, 'MACD_Signal': 2},
                     'ml_prediction': 0.023, 'ml_insights': {'confidence': 85}},
            'GOOGL': {'signal': 'BUY', 'confidence': 75, 'bias': 'Bullish',
                      'indicators': {'RSI': 60, 'MACD': 9, 'MACD_Signal': 8},
                      'ml_prediction': 0.018, 'ml_insights': {'confidence': 80}},
            'MSFT': {'signal': 'NEUTRAL', 'confidence': 55, 'bias': 'Bullish',
                     'indicators': {'RSI': 45, 'MACD': -2, 'MACD_Signal': -1},
                     'ml_prediction': 0.005, 'ml_insights': {'confidence': 65}}
        }
    }
    
    agg = MarketAggregator()
    result = agg.calculate_aggregate_score(sample_report)
    
    print("=" * 60)
    print("MARKET AGGREGATE SCORE")
    print("=" * 60)
    print(f"\nOverall Score: {result['overall_score']}/100")
    print(f"Outlook: {result['outlook']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"\nSignal Distribution: {result['signal_distribution']}")
    print(f"\nBreakdown:")
    for factor, data in result['breakdown'].items():
        print(f"  {factor}: {data['score']:.1f}/100 ({data['status']}) - Weight: {data['weight']}")
