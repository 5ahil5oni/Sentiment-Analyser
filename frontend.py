# enhanced_frontend_app.py

import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

try:
    import plotly
    plotly_version_str = plotly.__version__
except ImportError:
    plotly_version_str = "Plotly Not Installed"

# --- Helper Functions ---
def get_sentiment_color_and_emoji(score: float) -> Tuple[str, str, str]:
    """Maps sentiment score to a color, emoji, and descriptive label."""
    if score > 0.6: return "#2E7D32", "😊", "Strongly Positive"
    elif score > 0.1: return "#66BB6A", "🙂", "Mildly Positive"
    elif score >= -0.1: return "#757575", "😐", "Neutral"
    elif score > -0.6: return "#EF5350", "😕", "Mildly Negative"
    else: return "#C62828", "😠", "Strongly Negative"

def get_confidence_level(score: float) -> Tuple[str, str]:
    """Determine confidence level based on score magnitude."""
    abs_score = abs(score)
    if abs_score > 0.8: return "Very High", "#1B5E20"
    elif abs_score > 0.6: return "High", "#4CAF50"
    elif abs_score > 0.3: return "Medium", "#FFC107"
    else: return "Low", "#FF9800"

def analyze_text_stats(text: str) -> Dict[str, Any]:
    """Analyze basic text statistics."""
    if not text:
        return {"char_count": 0, "word_count": 0, "sentence_count": 0, "avg_word_length": 0, "complexity_score": 0}
    words = text.split()
    sentences = [s for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "complexity_score": len(set(w.lower() for w in words)) / len(words) if words else 0
    }

def create_sentiment_gauge(score: float) -> go.Figure:
    """Create a gauge chart for sentiment visualization."""
    bar_color, _, _ = get_sentiment_color_and_emoji(score)
    # print(f"DEBUG: Gauge bar_color for score {score} is {bar_color}") # For debugging
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score", 'font': {'size': 20}},
        number={'font': {'size': 36}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.6], 'color': "#80E8FB"}, {'range': [-0.6, -0.1], 'color': "#8EDCFB"},
                {'range': [-0.1, 0.1], 'color': "#88D0FA"}, {'range': [0.1, 0.6], 'color': "#82F1FD"},
                {'range': [0.6, 1], 'color': "#86F3FF"}],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 1, 'value': 0}
        }))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=30), font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_history_chart(history: List[Dict]) -> go.Figure:
    """Create a line chart showing sentiment history."""
    if not history:
        fig = go.Figure()
        fig.update_layout(title='Sentiment Analysis History (No data yet)', height=400,
                          xaxis_visible=False, yaxis_visible=False,
                          annotations=[dict(text="No analyses performed yet.", showarrow=False)])
        return fig
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    fig = px.line(df, x='timestamp', y='score', title='Sentiment Analysis History',
                  hover_data=['text_preview', 'strategy', 'label'], markers=True)
    fig.update_traces(line_color='#1f77b4', line_width=2)
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20),
                      xaxis_title="Time of Analysis", yaxis_title="Sentiment Score")
    return fig

# --- Initialize Session State ---
if "analysis_history" not in st.session_state: st.session_state.analysis_history = []
if "single_sentiment_data" not in st.session_state: st.session_state.single_sentiment_data = None
if "api_health" not in st.session_state: st.session_state.api_health = {"data": None, "connected": False, "auto_refresh": False}
if "current_single_text" not in st.session_state: st.session_state.current_single_text = ""
if "last_analyzed_text" not in st.session_state: st.session_state.last_analyzed_text = None # For auto-analyze debounce
if "auto_analyze" not in st.session_state: st.session_state.auto_analyze = False # Initialize auto_analyze

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Sentiment Analyzer", page_icon="🔑",
    layout="wide", initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.sentiment-result-box {
    padding: 25px; border-radius: 15px; color: white; text-align: center;
    font-size: 1.8em; font-weight: bold; margin: 20px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: all 0.3s ease;
}
.sentiment-result-box:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
.stats-card { background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea; margin: 10px 0; }
.confidence-badge { display: inline-block; padding: 5px 12px; border-radius: 20px; color: white; font-weight: bold; font-size: 0.9em; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
.stTabs [data-baseweb="tab"] { height: 50px; padding-left: 20px; padding-right: 20px; background-color: #f0f2f6; border-radius: 10px 10px 0 0; }
.stTabs [aria-selected="true"] { background-color: #667eea; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>Advanced Sentiment Analyzer</h1>
    <p>Intelligent text sentiment analysis with comprehensive insights and analytics</p>
</div>
""", unsafe_allow_html=True)

# --- API Health Check ---
@st.cache_data(ttl=30)
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return {"data": response.json(), "connected": True}
        return {"data": None, "connected": False, "status_code": response.status_code}
    except requests.exceptions.RequestException as e:
        return {"data": None, "connected": False, "error": str(e)}

if st.session_state.api_health.get("data") is None or st.session_state.api_health.get("auto_refresh", False):
    st.session_state.api_health = check_api_health()
api_health_data = st.session_state.api_health.get("data")
api_connected = st.session_state.api_health.get("connected", False)

# --- Sidebar ---
with st.sidebar:
    st.header("🔧 System Status")
    if api_connected and api_health_data:
        st.success("API Connected")
        st.info(f"**Active Strategy:** {api_health_data.get('active_analyzer_instance_strategy', 'N/A').title()}")
        st.info(f"**VADER Available:** {'Yes' if api_health_data.get('vader_available', False) else 'No'}")
    else:
        st.error("API Disconnected")
        if st.session_state.api_health.get("status_code"): st.warning(f"API Status: {st.session_state.api_health['status_code']}")
        elif st.session_state.api_health.get("error"): st.warning(f"Connection Error: {st.session_state.api_health['error'][:100]}...") # Truncate long errors
        else: st.warning(f"Cannot connect to {API_BASE_URL}")
    
    if st.button("Refresh API Status"):
        st.session_state.api_health = check_api_health()
        st.rerun()
    st.divider()
    st.header("Analysis Options")
    enable_batch_tab = st.checkbox("Enable Batch Analysis Tab", value=True, help="Show the Batch Analysis tab.")
    st.session_state.auto_analyze = st.checkbox("Auto-analyze on text change (Single)", value=st.session_state.auto_analyze, help="Experimental: Analyzes as you type.")
    
    st.header("History")
    if st.button("Clear History", type="secondary"): # Changed type for safety
        st.session_state.analysis_history = []
        st.rerun()
    history_count = len(st.session_state.analysis_history)
    st.metric("Total Analyses in History", history_count)
    if history_count > 0:
        avg_sentiment_hist = sum(item['score'] for item in st.session_state.analysis_history) / history_count
        st.metric("Average Sentiment (History)", f"{avg_sentiment_hist:.3f}")

# --- Main Content ---
tab_list = ["Single Analysis"]
if enable_batch_tab: tab_list.append("Batch Analysis")
tab_list.extend(["Analytics", "Settings", "API Access"])
tabs = st.tabs(tab_list)

# Tab 1: Single Analysis
with tabs[0]:
    st.subheader("Single Text Sentiment Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.session_state.current_single_text = st.text_area(
            "Enter text here:", value=st.session_state.current_single_text, height=200,
            key="single_text_input_widget", placeholder="Example: 'The service was exceptional...'",
            help="The API's configured strategy will be used for analysis."
        )
        input_text_single = st.session_state.current_single_text
        if input_text_single:
            stats = analyze_text_stats(input_text_single)
            expander_title = f"Text Statistics ({stats['word_count']} words, {stats['sentence_count']} sentences)"
            with st.expander(expander_title):
                cols_stats = st.columns(4)
                cols_stats[0].metric("Characters", stats["char_count"])
                cols_stats[1].metric("Words", stats["word_count"])
                cols_stats[2].metric("Sentences", stats["sentence_count"])
                cols_stats[3].metric("Complexity", f"{stats['complexity_score']:.2f}")
    with col2:
        st.write("### Quick Actions")
        analyze_single_button_clicked = st.button(
            "Analyze Sentiment", type="primary", use_container_width=True,
            key="analyze_single_btn", disabled=not api_connected or not input_text_single
        )
        st.write("#### Load Sample Text")
        sample_texts = {
            "Positive": "I'm absolutely thrilled with this amazing product! The quality is outstanding and it exceeded all my expectations.",
            "Negative": "This is the worst experience I've ever had. The product is completely broken and the service is terrible.",
            "Neutral": "The product arrived on time and appears to be functioning as described in the documentation."
        }
        for label, text_sample in sample_texts.items():
            if st.button(f"Use {label} Sample", key=f"sample_btn_{label.lower()}", use_container_width=True):
                st.session_state.current_single_text = text_sample
                st.rerun()
    
    # Auto-analyze logic
    perform_auto_analysis = False
    if st.session_state.auto_analyze and input_text_single and len(input_text_single) > 10: # Min length for auto
        if st.session_state.last_analyzed_text != input_text_single:
            # This is a very basic debounce. A proper one would use JavaScript or more complex session state.
            # For now, it triggers on text change if it's different from last analyzed.
            if "auto_analyze_timer" not in st.session_state:
                st.session_state.auto_analyze_timer = time.time()
            
            if time.time() - st.session_state.auto_analyze_timer > 1.0: # 1 second delay
                perform_auto_analysis = True
                st.session_state.auto_analyze_timer = time.time() # Reset timer


    if (analyze_single_button_clicked or perform_auto_analysis) and input_text_single and api_connected:
        st.session_state.single_sentiment_data = None
        spinner_message = " Auto-analyzing..." if perform_auto_analysis else " Analyzing single text..."
        with st.spinner(spinner_message):
            try:
                payload = {"text": input_text_single}
                start_time = time.time()
                response = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=30)
                response_time = time.time() - start_time
                st.session_state.last_analyzed_text = input_text_single # Update for auto-analyze debounce
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.single_sentiment_data = result
                    history_item = {
                        "timestamp": datetime.now().isoformat(), "text": input_text_single,
                        "text_preview": input_text_single[:50] + "..." if len(input_text_single) > 50 else input_text_single,
                        "score": result.get("sentiment_score", 0.0), "label": result.get("sentiment_label", "N/A"),
                        "strategy": result.get("strategy_used", "N/A"), "response_time": response_time
                    }
                    st.session_state.analysis_history.insert(0, history_item)
                else:
                    st.error(f"API Error (Status {response.status_code}):")
                    try: st.json(response.json())
                    except: st.text(response.text)
                    st.session_state.single_sentiment_data = {"error": True, "message": response.text}
            except requests.exceptions.Timeout:
                st.error("Single text analysis request timed out."); st.session_state.single_sentiment_data = {"error": True, "message": "Request Timeout"}
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}"); st.session_state.single_sentiment_data = {"error": True, "message": str(e)}
            except Exception as e:
                st.error(f"Unexpected error: {e}"); st.session_state.single_sentiment_data = {"error": True, "message": str(e)}

    if st.session_state.single_sentiment_data and not st.session_state.single_sentiment_data.get("error"):
        result = st.session_state.single_sentiment_data
        sentiment_score_api_val = result.get("sentiment_score", 0.0) # Use a different variable name
        sentiment_label_api = result.get("sentiment_label", "N/A")
        strategy_used = result.get("strategy_used", "N/A")
        color, emoji, descriptive_label = get_sentiment_color_and_emoji(sentiment_score_api_val)
        confidence_level, confidence_color = get_confidence_level(sentiment_score_api_val)
        st.divider()
        st.subheader(" Single Analysis Result")
        st.markdown(
            f'<div class="sentiment-result-box" style="background: linear-gradient(135deg, {color}99, {color});">'
            f'{emoji} {descriptive_label}<br>'
            f'<span style="font-size: 0.7em;">Score: {sentiment_score_api_val:.3f} (Strategy: {strategy_used.title()})</span>'
            f'</div>', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1: st.plotly_chart(create_sentiment_gauge(sentiment_score_api_val), use_container_width=True)
        with res_col2:
            st.markdown("#### Key Metrics:")
            st.markdown(f"**API Sentiment Label:** `{sentiment_label_api.title()}`")
            st.markdown(f'**Confidence in Polarity:** <span class="confidence-badge" style="background-color: {confidence_color};">{confidence_level}</span>', unsafe_allow_html=True)
            st.markdown("---"); st.markdown("#### Interpretation Guide:")
            st.markdown("- **> 0.6:** Strongly Positive\n- **0.1 to 0.6:** Mildly Positive\n- **-0.1 to 0.1:** Neutral\n- **-0.6 to -0.1:** Mildly Negative\n- **< -0.6:** Strongly Negative")
        with st.expander("Show Raw API Response", expanded=False): st.json(result)
    elif st.session_state.single_sentiment_data and st.session_state.single_sentiment_data.get("error"):
        st.warning(f"Could not display result due to an error: {st.session_state.single_sentiment_data.get('message', 'Unknown error')}")


# Tab 2: Batch Analysis
if enable_batch_tab:
    batch_tab_index = tab_list.index("Batch Analysis")
    with tabs[batch_tab_index]:
        st.subheader("Batch Text Sentiment Analysis")
        st.info("This feature uses the API's `/analyze-batch` endpoint for efficient processing.")
        batch_method = st.radio(
            "Choose input method for batch:", ["Text Area (Multiple Lines)", "File Upload (.txt, .csv)"],
            horizontal=True, key="batch_method_radio"
        )
        batch_texts_input = []
        if batch_method == "Text Area (Multiple Lines)":
            batch_input_area = st.text_area(
                "Enter multiple texts (one per line):", height=250,
                placeholder="Text 1...\nText 2...\nAnother text...", key="batch_text_area")
            if batch_input_area: batch_texts_input = [line.strip() for line in batch_input_area.split('\n') if line.strip()]
        elif batch_method == "File Upload (.txt, .csv)":
            uploaded_file = st.file_uploader("Upload a file:", type=['txt', 'csv'], key="batch_file_uploader")
            if uploaded_file:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    batch_texts_input = [line.strip() for line in content.split('\n') if line.strip()]
                    st.success(f"Read {len(batch_texts_input)} lines from '{uploaded_file.name}'.")
                except Exception as e: st.error(f"Error reading file: {e}")
        
        if batch_texts_input:
            st.info(f"Prepared {len(batch_texts_input)} texts for batch analysis.")
            api_batch_size_val = st.slider("API Batch Size (for backend processing)", 1, 32, 8, key="api_batch_slider_val",
                                       help="How many texts the API backend processes in one internal go.")
            if st.button("🔬 Analyze Full Batch (via API)", type="primary", use_container_width=True, key="analyze_batch_api_btn_main", disabled=not api_connected):
                with st.spinner(f"Processing {len(batch_texts_input)} texts in batch via API..."):
                    try:
                        payload = {"texts": batch_texts_input, "batch_size": api_batch_size_val}
                        start_time = time.time()
                        # Determine a reasonable timeout based on number of texts
                        timeout_seconds = max(60, int(len(batch_texts_input) * 0.5)) # 0.5s per text, min 60s
                        response = requests.post(f"{API_BASE_URL}/analyze-batch", json=payload, timeout=timeout_seconds)
                        response_time_batch = time.time() - start_time
                        if response.status_code == 200:
                            batch_api_response = response.json()
                            batch_results_data = batch_api_response.get("results", [])
                            overall_strategy = batch_api_response.get("overall_strategy_used", "N/A")
                            st.success(f"Batch complete! {len(batch_results_data)} texts processed in {response_time_batch:.2f}s. API Strategy: {overall_strategy.title()}")
                            if batch_results_data:
                                df_batch = pd.DataFrame(batch_results_data)
                                st.dataframe(df_batch, use_container_width=True, column_config={
                                    "text": st.column_config.TextColumn("Analyzed Text", width="large"),
                                    "sentiment_label": "Label",
                                    "sentiment_score": st.column_config.NumberColumn("Score", format="%.3f")})
                                for item_data in batch_results_data:
                                    st.session_state.analysis_history.insert(0, {
                                        "timestamp": datetime.now().isoformat(), "text": item_data.get("text","N/A"),
                                        "text_preview": item_data.get("text","N/A")[:50] + "...",
                                        "score": item_data.get("sentiment_score", 0.0), "label": item_data.get("sentiment_label", "N/A"),
                                        "strategy": overall_strategy, "response_time": response_time_batch / len(batch_texts_input) if batch_texts_input else 0
                                    })
                                scores_for_hist = [res.get("sentiment_score", 0.0) for res in batch_results_data]
                                if scores_for_hist:
                                    fig_hist = px.histogram(scores_for_hist, nbins=max(5, len(set(scores_for_hist)) // 2), title="Batch Score Distribution", labels={"value": "Sentiment Score"})
                                    st.plotly_chart(fig_hist, use_container_width=True)
                        else:
                            st.error(f"API Batch Error (Status {response.status_code}):")
                            try: st.json(response.json())
                            except: st.text(response.text)
                    except requests.exceptions.Timeout: st.error(f"Batch API request timed out after {timeout_seconds}s.")
                    except requests.exceptions.RequestException as e: st.error(f"Connection error (batch): {e}")
                    except Exception as e: st.error(f"Unexpected error (batch): {e}")
            elif not api_connected: st.warning("API not connected. Cannot perform batch analysis.")
        elif st.button("Analyze Full Batch (via API)", use_container_width=True, disabled=True): # Show disabled button if no text
            pass


# Tab 3: Analytics
analytics_tab_index = tab_list.index("Analytics")
with tabs[analytics_tab_index]:
    st.subheader("Sentiment Analytics Dashboard")
    if st.session_state.analysis_history:
        df_history = pd.DataFrame(st.session_state.analysis_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        df_history.sort_values('timestamp', ascending=False, inplace=True)
        st.markdown("#### Overview")
        col1, col2, col3, col4 = st.columns(4)
        total_analyses = len(df_history)
        col1.metric("Total Analyses", total_analyses)
        if total_analyses > 0:
            avg_score_hist = df_history["score"].mean()
            col2.metric("Average Sentiment", f"{avg_score_hist:.3f}")
            positive_count_hist = len(df_history[df_history["score"] > 0.1])
            col3.metric("Positive Texts", f"{positive_count_hist} ({positive_count_hist/total_analyses*100:.1f}%)")
            if "response_time" in df_history.columns and df_history["response_time"].notna().any(): # Check if column exists and has non-NA values
                avg_response_time_hist = df_history["response_time"].dropna().mean() # Drop NA before mean
                col4.metric("Avg API Time/Text", f"{avg_response_time_hist:.2f}s")
            else:
                col4.metric("Avg API Time/Text", "N/A")

        st.markdown("#### Visualizations")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1: st.plotly_chart(create_history_chart(st.session_state.analysis_history), use_container_width=True)
        with chart_col2:
            if total_analyses > 0:
                strategy_counts = df_history["strategy"].value_counts()
                fig_pie_strategy = px.pie(values=strategy_counts.values, names=strategy_counts.index, title="Strategy Usage (History)", hole=0.3)
                st.plotly_chart(fig_pie_strategy, use_container_width=True)
            else: st.info("No strategy usage data yet.")
        st.markdown("#### Detailed History")
        display_df_hist = df_history.copy()
        display_df_hist["timestamp"] = display_df_hist["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(display_df_hist[["timestamp", "text_preview", "score", "label", "strategy", "response_time"]],
            column_config={
                "timestamp": "Time", "text_preview": "Text Preview",
                "score": st.column_config.NumberColumn("Score", format="%.3f"), "label": "Label", "strategy": "Strategy",
                "response_time": st.column_config.NumberColumn("API Time (s)", format="%.2f")},
            height=300, use_container_width=True)
        if st.button("📥 Export Full History as CSV", key="export_csv_btn_main"):
            csv_data = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download History CSV", data=csv_data,
                file_name=f"sentiment_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    else: st.info("No analysis history recorded yet.")

# Tab 4: Settings
settings_tab_index = tab_list.index("Settings")
with tabs[settings_tab_index]:
    st.subheader("Application Settings")
    st.markdown("##### API Configuration")
    st.text_input("API Base URL (Current):", value=API_BASE_URL, disabled=True, help="To change, modify the script's API_BASE_URL variable.")
    st.session_state.api_health["auto_refresh"] = st.checkbox("Auto-refresh API Health in Sidebar", value=st.session_state.api_health.get("auto_refresh", False))
    st.markdown("##### Analysis Defaults (UI Suggestions)")
    st.slider("Default UI Batch Size Suggestion", 1, 50, 10, help="Suggests batch size for UI elements, API may use different for its own processing.")
    st.slider("Default Request Timeout (seconds)", 5, 120, 30, help="Timeout for API requests from this frontend.")
    st.markdown("##### Data Management")
    st.warning("**Action: Clear All Data**\n\nThis will remove all analysis history and current session results. This action is irreversible.")
    if st.button("Proceed to Clear All Session Data", type="secondary", key="clear_all_data_btn_final"):
        st.session_state.analysis_history = []
        st.session_state.single_sentiment_data = None
        st.session_state.current_single_text = ""
        st.success("All session data has been cleared!")
        st.rerun()
    st.markdown("##### ℹ System Information")
    st.info(f"Streamlit Version: {st.__version__}\nRequests Version: {requests.__version__}\n"
            f"Pandas Version: {pd.__version__}\nPlotly Version: {plotly_version_str}") # Used plotly_version_str

# Tab 5: API Access Info
api_access_tab_index = tab_list.index("API Access")
with tabs[api_access_tab_index]:
    st.subheader("Programmatic API Access")
    st.markdown(f"Interact with the Sentiment Analyzer API directly. Base URL: `{API_BASE_URL}`")
    st.markdown(f"Full API docs: [Swagger UI]({API_BASE_URL}/docs) | [ReDoc]({API_BASE_URL}/redoc).")
    st.markdown("---"); st.markdown("#### Endpoint: `/analyze` (Single Text Analysis)")
    st.markdown("**Method:** `POST`"); st.markdown("**Request Body (JSON):**"); st.json({"text": "Your sample text."})
    st.markdown("**Example `curl`:**"); st.code(f"curl -X POST -H \"Content-Type: application/json\" -d '{{\"text\": \"Streamlit is fun!\"}}' {API_BASE_URL}/analyze", language="bash")
    st.markdown("**Example Python `requests`:**"); st.code(f"import requests, json\napi_url = \"{API_BASE_URL}/analyze\"\npayload = {{\"text\": \"Test sentence.\"}}\nresponse = requests.post(api_url, json=payload)\nprint(json.dumps(response.json(), indent=2) if response.ok else f'Error: {{response.status_code}}')", language="python")
    st.markdown("**Success Response Example (JSON):**"); st.json({"text": "Streamlit is fun!", "sentiment_label": "positive", "sentiment_score": 0.987, "strategy_used": "balanced"})
    st.markdown("---"); st.markdown("#### Endpoint: `/analyze-batch` (Batch Text Analysis)")
    st.markdown("**Method:** `POST`"); st.markdown("**Request Body (JSON):**"); st.json({"texts": ["Text 1.", "Text 2."], "batch_size": 2})
    st.markdown("**Example `curl`:**"); st.code(f"curl -X POST -H \"Content-Type: application/json\" -d '{{\"texts\": [\"API access.\", \"Batch efficient.\"], \"batch_size\": 2}}' {API_BASE_URL}/analyze-batch", language="bash")
    st.markdown("**Example Python `requests`:**"); st.code(f"import requests, json\napi_url = \"{API_BASE_URL}/analyze-batch\"\npayload = {{\"texts\": [\"One.\", \"Two.\"], \"batch_size\": 2}}\nresponse = requests.post(api_url, json=payload)\nprint(json.dumps(response.json(), indent=2) if response.ok else f'Error: {{response.status_code}}')", language="python")
    st.markdown("**Success Response Example (JSON):**"); st.json({"results": [{"text": "API access.", "sentiment_label": "positive", "sentiment_score": 0.95}, {"text": "Batch efficient.", "sentiment_label": "positive", "sentiment_score": 0.92}], "overall_strategy_used": "balanced", "batch_size_used": 2})
    st.markdown("---"); st.markdown("#### Endpoint: `/health` (Health Check)")
    st.markdown("**Method:** `GET`"); st.markdown("**Example `curl`:**"); st.code(f"curl {API_BASE_URL}/health", language="bash")
    st.markdown("**Success Response Example (JSON):**"); st.json({"api_status": "ok", "sentiment_analyzer_status": "ok", "configured_strategy_on_load": "balanced", "active_analyzer_instance_strategy": "balanced", "vader_available": True})

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>🔬 Advanced Sentiment Analyzer © 2024-2025 • Interface powered by Streamlit</div>", unsafe_allow_html=True)

