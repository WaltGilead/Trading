import streamlit as st
import pandas as pd
import time
from agent import TradeAgent

# Page Config
st.set_page_config(
    page_title="Trade Agent Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Card Style */
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #2b3144;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    p, div {
        color: #b0b3b8;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00cc96;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #888;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid #2b3144;
        border-radius: 5px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161924;
        border-right: 1px solid #2b3144;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: #888;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #00cc96;
        border-bottom: 2px solid #00cc96;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_agent():
    return TradeAgent()

def main():
    # Sidebar
    with st.sidebar:
        st.title("⚡ Trade Agent Pro")
        st.markdown("---")
        st.markdown("**Status:** 🟢 Online")
        
        if st.button("🔄 Run Analysis Cycle", type="primary", use_container_width=True):
            st.session_state['run_analysis'] = True
        
        st.markdown("---")
        st.caption(f"Last Update: {time.strftime('%H:%M:%S')}")
        st.caption("v2.1.0 | Powered by Python")

    # Main Content
    st.title("Market Intelligence Dashboard")
    
    agent = get_agent()
    
    # Run Logic
    if st.session_state.get('run_analysis', False):
        with st.spinner("Analyzing Market Data..."):
            report = agent.run_cycle()
            st.session_state['last_report'] = report
            st.session_state['run_analysis'] = False # Reset
            
    # Display Logic
    if 'last_report' in st.session_state:
        report = st.session_state['last_report']
        
        # ===== MARKET OUTLOOK AGGREGATE =====
        market_agg = report.get('market_aggregate', {})
        if market_agg:
            st.markdown("## 📊 Market Outlook")
            
            overall_score = market_agg.get('overall_score', 50)
            outlook = market_agg.get('outlook', 'Neutral')
            confidence = market_agg.get('confidence', 50)
            
            # Color coding based on outlook
            if outlook == "Bullish":
                color = "#00cc96"  # Green
                emoji = "📈"
            elif outlook == "Bearish":
                color = "#ef553b"  # Red
                emoji = "📉"
            else:
                color = "#636efa"  # Blue
                emoji = "➡️"
            
            # Main Gauge Row
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"### {emoji} {outlook} Market")
                st.progress(overall_score / 100)
                st.caption(f"Aggregate Score: {overall_score:.1f}/100 | Confidence: {confidence:.0f}%")
            
            with col2:
                signal_dist = market_agg.get('signal_distribution', {})
                st.metric("BUY Signals", signal_dist.get('buy', 0))
            
            with col3:
                st.metric("NEUTRAL", signal_dist.get('neutral', 0))
            
            with col4:
                st.metric("SELL Signals", signal_dist.get('sell', 0))
            
            # Breakdown Expander
            with st.expander("🔍 View Score Breakdown"):
                breakdown = market_agg.get('breakdown', {})
                
                if breakdown:
                    b_cols = st.columns(4)
                    for idx, (factor, data) in enumerate(breakdown.items()):
                        with b_cols[idx % 4]:
                            score = data.get('score', 50)
                            status = data.get('status', 'Neutral')
                            weight = data.get('weight', 'N/A')
                            
                            st.metric(
                                f"{factor}",
                                f"{score:.1f}",
                                f"{status} ({weight})"
                            )
                            
                            # Show details if available
                            if 'details' in data:
                                st.caption(data['details'])
            
            st.markdown("---")
        else:
            st.info("✨ New Feature: Market Outlook is available! Click 'Run Analysis Cycle' to generate the aggregate score.")
        
        # Tabs
        tab1, tab2 = st.tabs(["🔍 Asset Drill-Down", "🛡️ Risk Monitor"])
        
        with tab1:
            st.markdown("### Asset Analysis")
            
            # Asset Selector
            assets = list(report['signals'].keys())
            selected_asset = st.selectbox("Select Asset", assets)
            
            if selected_asset:
                data = report['signals'][selected_asset]
                indicators = data.get('indicators', {})
                ml_insights = data.get('ml_insights')
                
                # Header
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"${data['price']:,.2f}")
                c2.metric("Signal", data['signal'])
                
                # Enhanced ML Prediction Display
                ml_pred_value = data.get('ml_prediction', 0)
                ml_label = "Neutral"
                ml_emoji = "➡️"
                
                if ml_pred_value > 0.001:
                    ml_label = "Bullish"
                    ml_emoji = "📈"
                elif ml_pred_value < -0.001:
                    ml_label = "Bearish"
                    ml_emoji = "📉"
                
                if ml_insights and ml_insights.get('confidence'):
                    confidence_pct = ml_insights['confidence']
                    c3.metric(
                        "ML Prediction", 
                        f"{ml_emoji} {ml_label}",
                        f"Return: {ml_pred_value*100:+.2f}% | Conf: {confidence_pct:.0f}%"
                    )
                else:
                    c3.metric("ML Prediction", f"{ml_emoji} {ml_label}", f"Return: {ml_pred_value*100:+.2f}%")
                
                st.markdown("---")
                
                # Technicals Grid
                st.markdown("#### Technical Indicators")
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("RSI (14)", indicators.get('RSI', 'N/A'))
                t2.metric("MACD", indicators.get('MACD', 'N/A'))
                t3.metric("SMA 50", f"{indicators.get('SMA_50', 0):,.2f}")
                t4.metric("SMA 200", f"{indicators.get('SMA_200', 0):,.2f}")
                
                st.markdown("---")
                
                # 🤖 ML INSIGHTS SECTION
                if ml_insights:
                    st.markdown("#### 🤖 Machine Learning Insights")
                    
                    # Row 1: Confidence and Performance Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        confidence = ml_insights.get('confidence', 0)
                        st.metric("Prediction Confidence", f"{confidence:.1f}%")
                        # Visual confidence bar
                        confidence_color = "#00cc96" if confidence > 70 else "#ffa500" if confidence > 40 else "#ef553b"
                        st.progress(confidence / 100)
                    
                    with col2:
                        mse = ml_insights.get('metrics', {}).get('mse', 0)
                        st.metric("Model MSE", f"{mse:.6f}")
                        st.caption("Mean Squared Error (lower is better)")
                    
                    with col3:
                        r2 = ml_insights.get('metrics', {}).get('r2_score', 0)
                        st.metric("R² Score", f"{r2:.4f}")
                        st.caption("Coefficient of Determination")
                    
                    st.markdown("")
                    
                    # Row 2: Feature Importance
                    st.markdown("**Feature Importance**")
                    st.caption("Which technical indicators most influence this prediction?")
                    
                    feature_importance = ml_insights.get('feature_importance', {})
                    if feature_importance:
                        # Sort by importance
                        sorted_features = sorted(
                            feature_importance.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        # Take top 8 features for cleaner display
                        top_features = sorted_features[:8]
                        
                        # Create DataFrame for chart
                        import pandas as pd
                        chart_df = pd.DataFrame({
                            'Feature': [f[0] for f in top_features],
                            'Importance': [f[1] for f in top_features]
                        })
                        
                        # Display top features as metrics (avoiding problematic bar_chart)
                        st.markdown("**Top Features:**")
                        cols = st.columns(min(len(top_features), 4))
                        for idx, (feature, importance) in enumerate(top_features[:4]):
                            with cols[idx]:
                                st.metric(feature, f"{importance:.4f}")
                        
                        # Show remaining features if more than 4
                        if len(top_features) > 4:
                            cols2 = st.columns(min(len(top_features) - 4, 4))
                            for idx, (feature, importance) in enumerate(top_features[4:8]):
                                with cols2[idx]:
                                    st.metric(feature, f"{importance:.4f}")
                        
                        # Show detailed breakdown in expandable section
                        with st.expander("📊 View All Feature Importance Values"):
                            for feature, importance in sorted_features:
                                st.write(f"**{feature}**: {importance:.4f}")
                    else:
                        st.info("Feature importance data not available")
                
                else:
                    st.info("💡 ML insights unavailable for this asset (insufficient data for training)")
                
                st.markdown("---")
                st.markdown(f"**Analysis:** {data['explanation']}")

        with tab2:
            st.markdown("### System Health & Risk")
            
            # Display Sentiment Scores here as they are useful context
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Economic Score", report['sentiment']['economic'], report['sentiment_labels']['economic'])
            with col2:
                st.metric("News Sentiment", report['sentiment']['news'], "VADER Score")
            with col3:
                st.metric("Macro Score", report['sentiment']['macro'], report['sentiment_labels']['macro'])
            
            st.markdown("---")
            
            if report['alerts']:
                for alert in report['alerts']:
                    st.error(f"⚠️ {alert}")
            else:
                st.success("✅ System Normal. No critical risk events detected.")
                
            st.markdown("#### Raw Data Logs")
            st.json(report['sentiment'])

    else:
        st.info("👈 Click 'Run Analysis Cycle' in the sidebar to start.")

if __name__ == "__main__":
    main()
