import logging

logger = logging.getLogger(__name__)

class Monitor:
    def __init__(self):
        pass

    def monitor_risk(self, market_data, sentiment_scores):
        """
        Checks for risk conditions like high volatility or sentiment flips.
        Returns a list of alerts.
        """
        alerts = []
        
        # Check VIX
        vix_data = market_data.get('VIX')
        if vix_data is not None and not vix_data.empty:
            current_vix = vix_data['Close'].iloc[-1]
            if current_vix > 30:
                alerts.append(f"HIGH RISK: VIX is {current_vix:.2f} (>30)")
            elif current_vix > 20:
                alerts.append(f"CAUTION: VIX is {current_vix:.2f} (>20)")
                
        # Check DXY
        dxy_data = market_data.get('DXY')
        if dxy_data is not None and not dxy_data.empty:
            current_dxy = dxy_data['Close'].iloc[-1]
            if current_dxy > 106:
                 alerts.append(f"RISK: DXY is {current_dxy:.2f} (Strong Dollar)")

        # Check Sentiment Shifts
        macro_score = sentiment_scores.get('macro', 50)
        if macro_score < 30:
            alerts.append("RISK: Macro Sentiment is Bearish")
            
        return alerts
