import streamlit as st
from pages import data_exploration, model_training, model_comparison, prediction

# Set page configuration
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide", initial_sidebar_state="expanded")

# Load custom CSS
try:
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("CSS file not found. Ensure 'static/style.css' exists.")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Model Comparison", "Prediction"])

# Page routing
if page == "Data Exploration":
    data_exploration.show()
elif page == "Model Training":
    model_training.show()
elif page == "Model Comparison":
    model_comparison.show()
elif page == "Prediction":
    prediction.show()

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: #666666;">
        Built by Nihar Walawalkar | © VESIT 2025
    </div>
    """,
    unsafe_allow_html=True
)