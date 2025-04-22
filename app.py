"""
Main application file for the Enhanced Employee Attrition Prediction web application.
"""

import streamlit as st
from src.pages import home, data_exploration, model_training, predictions, insights, retention_planner
from src.utils import config
import os
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
def load_css():
    """Load custom CSS to beautify the app."""
    
    st.markdown(
        """
        <style>
        /* Main styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Headings */
        h1, h2, h3 {
            font-family: 'Source Sans Pro', sans-serif;
        }
        h1 {
            font-weight: 600;
            margin-bottom: 1.5rem;
        }
        h2 {
            font-weight: 500;
            color: #333;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        /* Custom button styling */
        .stButton button {
            background-color: white !important;
            color: #555 !important;
            border: none !important;
            padding: 0.75rem 1rem !important;
            border-radius: 0.5rem !important;
            transition: all 0.2s ease !important;
            text-align: left !important;
            font-weight: 500 !important;
            margin: 0.2rem 0 !important;
            font-size: 1.05rem !important;
            width: 100% !important;
            cursor: pointer !important;
        }
        .stButton button:hover {
            background-color: rgba(78, 205, 196, 0.1) !important;
            color: #4ecdc4 !important;
            box-shadow: none !important;
            transform: translateX(5px) !important;
        }
        .stButton button:active {
            background-color: rgba(78, 205, 196, 0.2) !important;
            transform: translateX(0px) !important;
        }
        
        /* Primary button - active navigation item */
        .stButton button[kind="primary"] {
            background-color: rgba(78, 205, 196, 0.15) !important;
            color: #4ecdc4 !important;
            border-left: 3px solid #4ecdc4 !important;
            font-weight: 600 !important;
        }
        
        /* Hide the radio button label */
        .sidebar .row-widget.stRadio > div {
            display: none;
        }
        
        /* Fix for z-index issues with dropdowns */
        .stSelectbox [data-baseweb="select"] {
            z-index: 999;
        }
        
        /* Metric styling - more modern look */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: bold !important;
            color: #4ecdc4 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #666 !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.9rem !important;
        }
        
        /* Card-like elements */
        div.card {
            border-radius: 0.5rem;
            border: 1px solid #f0f0f0;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        div.card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        /* Enhanced Navigation styling */
        .sidebar .sidebar-content {
            background-color: #fafafa;
        }
        
        /* Custom navigation */
        .nav-link {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            margin: 0.2rem 0;
            border-radius: 0.5rem;
            color: #555;
            font-weight: 500;
            transition: all 0.2s ease;
            text-decoration: none;
            font-size: 1.05rem;
        }
        .nav-link:hover {
            background-color: rgba(78, 205, 196, 0.1);
            color: #4ecdc4;
            padding-left: 1.3rem;
        }
        .nav-link.active {
            background-color: rgba(78, 205, 196, 0.15);
            color: #4ecdc4;
            border-left: 3px solid #4ecdc4;
            font-weight: 600;
        }
        .nav-icon {
            margin-right: 10px;
            font-size: 1.2rem;
            width: 24px;
            text-align: center;
        }
        
        /* Better sidebar layout */
        .sidebar-header {
            padding: 1.5rem 1rem;
            margin-bottom: 1.5rem;
            text-align: center;
            border-bottom: 1px solid #f0f0f0;
        }
        .sidebar-footer {
            padding: 1rem;
            text-align: center;
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            font-size: 0.8rem;
            color: #888;
        }
        
        /* Dashboard KPI cards */
        .kpi-card {
            background: linear-gradient(145deg, #fafafa, #ffffff);
            border-radius: 1rem;
            padding: 1.2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            border-left: 4px solid #4ecdc4;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        .kpi-card.warning {
            border-left: 4px solid #ff6f61;
        }
        .kpi-card.info {
            border-left: 4px solid #3498db;
        }
        
        /* Footer styling */
        footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #f0f0f0;
            text-align: center;
            color: #666;
            font-size: 0.8rem;
        }
        
        /* Data tables */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            border-radius: 0.5rem;
            overflow: hidden;
        }
        .dataframe th {
            background-color: #4ecdc4;
            color: white;
            padding: 0.75rem;
            border: 1px solid #e0e0e0;
        }
        .dataframe td {
            padding: 0.75rem;
            border: 1px solid #e0e0e0;
        }
        .dataframe tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .dataframe tr:hover {
            background-color: #f0f0f0;
        }
        
        /* Plotly charts */
        .js-plotly-plot {
            border-radius: 0.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Make expanders stand out more */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #4ecdc4;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            background-color: rgba(78, 205, 196, 0.05);
            transition: all 0.2s ease;
        }
        .streamlit-expanderHeader:hover {
            background-color: rgba(78, 205, 196, 0.1);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f7f7f7;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4ecdc4 !important;
            color: white !important;
        }
        
        /* Add more whitespace */
        .section-spacing {
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        
        /* Scrollable areas */
        .scrollable {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        .scrollable::-webkit-scrollbar {
            width: 6px;
        }
        .scrollable::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        .scrollable::-webkit-scrollbar-thumb {
            background: #4ecdc4;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    """Main function to run the application."""
    
    # Load custom CSS
    load_css()
    
    # Create necessary directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    
    # Add header/navigation in sidebar
    st.sidebar.markdown(
        """
        <div class="sidebar-header">
            <h1 style="color: #4ecdc4; margin-bottom: 0; font-size: 1.8rem;">HR Analytics</h1>
            <p style="color: #666; font-size: 1rem;">Employee Attrition Prediction</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Navigation
    st.sidebar.markdown("<h3 style='margin-left: 10px; margin-bottom: 15px; font-size: 1.2rem;'>Navigation</h3>", unsafe_allow_html=True)
    
    # Main pages
    pages = {
        "Home": {"icon": "🏠", "page": home},
        "Data Exploration": {"icon": "📊", "page": data_exploration},
        "Model Training": {"icon": "🧠", "page": model_training},
        "Predictions": {"icon": "🔮", "page": predictions},
        "Insights": {"icon": "💡", "page": insights},
        "Retention Planner": {"icon": "📝", "page": retention_planner},
        "Team Analysis": {"icon": "👥", "page": home},      # Will route to home for now
        "Career Path Optimizer": {"icon": "📈", "page": home}  # Will route to home for now
    }
    
    # Initialize session state for current page if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Check for query parameters (for direct links)
    query_params = st.query_params
    if "page" in query_params:
        page_from_query = query_params["page"][0]
        
        # Map query param to actual page
        page_mapping = {
            "home": "Home",
            "data_exploration": "Data Exploration",
            "model_training": "Model Training",
            "predictions": "Predictions",
            "insights": "Insights",
            "retention_planner": "Retention Planner",
            "team_analysis": "Team Analysis",
            "career_path_optimizer": "Career Path Optimizer"
        }
        
        if page_from_query in page_mapping:
            # Only update if different from current
            if st.session_state.current_page != page_mapping[page_from_query]:
                st.session_state.current_page = page_mapping[page_from_query]
    
    # Core Features Section
    st.sidebar.markdown("<div style='margin-left: 10px; margin-bottom: 5px; color: #888; font-size: 0.9rem;'>CORE FEATURES</div>", unsafe_allow_html=True)
    
    # Function to handle page navigation
    def nav_to(page_name):
        st.session_state.current_page = page_name
        st.rerun()
    
    # Core features navigation
    for idx, (page_name, page_info) in enumerate(list(pages.items())[:5]):
        is_active = st.session_state.current_page == page_name
        
        if st.sidebar.button(
            f"{page_info['icon']} {page_name}", 
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=nav_to,
            args=(page_name,)
        ):
            pass
    
    # Advanced Features Section
    st.sidebar.markdown("<div style='margin-left: 10px; margin-top: 20px; margin-bottom: 5px; color: #888; font-size: 0.9rem;'>ADVANCED FEATURES</div>", unsafe_allow_html=True)
    
    # Advanced features navigation
    for idx, (page_name, page_info) in enumerate(list(pages.items())[5:]):
        is_active = st.session_state.current_page == page_name
        new_badge = " 🆕"
        
        if st.sidebar.button(
            f"{page_info['icon']} {page_name}{new_badge}", 
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=nav_to,
            args=(page_name,)
        ):
            pass
    
    # Display current page
    current_page = st.session_state.current_page
    if current_page in pages:
        pages[current_page]["page"].show()
    else:
        # Fallback to home if page not found
        pages["Home"]["page"].show()
    
    # Add information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        This advanced HR analytics platform predicts employee attrition using machine learning algorithms.
        It helps HR professionals identify at-risk employees and take proactive measures with data-driven retention strategies.
        """
    )
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <small style="color: #666;">
                Version {config.APP_VERSION} | Build {config.BUILD_NUMBER}
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 