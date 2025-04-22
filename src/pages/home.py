"""
Home page providing an overview of the Employee Attrition Prediction application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.utils import data_loader, config

def show():
    """Display the home page."""
    
    # Hero section with title and description
    st.markdown(
        """
        <div style="text-align: center; padding: 40px 0;">
            <h1 style="color: #4ecdc4; font-size: 2.5rem;">Employee Attrition Prediction</h1>
            <p style="font-size: 1.2rem; margin-bottom: 30px;">An AI-powered tool for predicting and understanding employee attrition</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Overview
    st.markdown(
        """
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #333;">About This Tool</h2>
            <p>This application provides data-driven insights on employee attrition, helping HR departments and managers identify at-risk employees and take proactive retention measures.</p>
            <p>Using machine learning algorithms, the tool analyzes various factors contributing to employee turnover and provides actionable recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load the dataset for statistics
    df = data_loader.load_dataset()
    if df is not None:
        # Calculate some key statistics
        total_employees = len(df)
        attrition_count = df[df['Attrition'] == 'Yes'].shape[0] if 'Attrition' in df.columns else 0
        attrition_rate = (attrition_count / total_employees) * 100 if total_employees > 0 else 0
        
        # Check if model is trained
        model_trained = all([
            os.path.exists(config.BEST_MODEL_PATH),
            os.path.exists(config.SCALER_PATH),
            os.path.exists(config.FEATURES_PATH),
            os.path.exists(config.METRICS_PATH)
        ])
        
        # Key statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div style="background-color: #4ecdc4; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="margin: 0; font-size: 1.2rem;">Total Employees</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 10px 0;">{total_employees}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(
                f"""
                <div style="background-color: #ff6f61; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="margin: 0; font-size: 1.2rem;">Attrition Rate</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 10px 0;">{attrition_rate:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col3:
            st.markdown(
                f"""
                <div style="background-color: {'#4ecdc4' if model_trained else '#ffb347'}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="margin: 0; font-size: 1.2rem;">Model Status</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 10px 0;">{"Trained" if model_trained else "Not Trained"}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Quick overview chart
        if 'Department' in df.columns and 'Attrition' in df.columns:
            st.subheader("Quick Department Overview")
            
            # Attrition by department
            dept_attrition = df.groupby('Department')['Attrition'].value_counts().unstack().reset_index()
            dept_attrition.columns = ['Department', 'No', 'Yes']
            dept_attrition['Total'] = dept_attrition['No'] + dept_attrition['Yes']
            dept_attrition['Attrition Rate (%)'] = (dept_attrition['Yes'] / dept_attrition['Total']) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    dept_attrition,
                    x='Department',
                    y=['No', 'Yes'],
                    title='Employee Distribution',
                    barmode='stack',
                    color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'},
                    labels={'value': 'Number of Employees', 'variable': 'Attrition'},
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    dept_attrition,
                    x='Department',
                    y='Attrition Rate (%)',
                    title='Attrition Rate by Department',
                    color='Attrition Rate (%)',
                    color_continuous_scale='RdBu_r',
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Features and navigation
    st.subheader("Features")
    
    feature_cols = st.columns(4)
    
    with feature_cols[0]:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; border: 1px solid #eee; border-radius: 10px; height: 100%;">
                <h3 style="color: #4ecdc4;">Data Exploration</h3>
                <p>Interactive visualizations to understand attrition patterns and employee data.</p>
                <a href="/?page=data_exploration" target="_self" style="color: #4ecdc4;">Explore Data →</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with feature_cols[1]:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; border: 1px solid #eee; border-radius: 10px; height: 100%;">
                <h3 style="color: #4ecdc4;">Model Training</h3>
                <p>Train and evaluate machine learning models with customizable parameters.</p>
                <a href="/?page=model_training" target="_self" style="color: #4ecdc4;">Train Models →</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with feature_cols[2]:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; border: 1px solid #eee; border-radius: 10px; height: 100%;">
                <h3 style="color: #4ecdc4;">Predictions</h3>
                <p>Make individual or batch predictions with detailed explanations.</p>
                <a href="/?page=predictions" target="_self" style="color: #4ecdc4;">Make Predictions →</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with feature_cols[3]:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; border: 1px solid #eee; border-radius: 10px; height: 100%;">
                <h3 style="color: #4ecdc4;">Insights</h3>
                <p>Get actionable insights and recommendations for reducing attrition.</p>
                <a href="/?page=insights" target="_self" style="color: #4ecdc4;">View Insights →</a>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Getting started guide
    st.subheader("Getting Started Guide")
    
    with st.expander("How to Use This Tool", expanded=True):
        st.markdown(
            """
            Follow these steps to get the most out of the Employee Attrition Prediction tool:
            
            1. **Explore the Data**: Start with the Data Exploration page to understand patterns and relationships in the employee data.
            
            2. **Train a Model**: Visit the Model Training page to train a machine learning model with customized parameters.
            
            3. **Make Predictions**: Use the Predictions page to:
               - Predict attrition risk for individual employees
               - Run batch predictions for multiple employees
               
            4. **Review Insights**: Check the Insights page for in-depth analysis and actionable recommendations.
            """
        )
    
    # Use case scenarios
    with st.expander("Use Case Scenarios"):
        st.markdown(
            """
            ### HR Management
            - Identify employees with high attrition risk
            - Develop targeted retention strategies
            - Monitor department-level attrition trends
            
            ### Resource Planning
            - Forecast potential turnover for succession planning
            - Allocate resources for retention programs
            - Plan recruitment needs proactively
            
            ### Employee Engagement
            - Identify factors affecting job satisfaction
            - Develop personalized engagement strategies
            - Measure the impact of HR initiatives
            """
        )
    
    # About project section
    st.markdown(
        """
        ---
        <div style="text-align: center; padding: 20px;">
            <p style="color: #666;">
                Developed by <a href="https://github.com/yourusername" target="_blank" style="color: #4ecdc4;">Nihar Walawalkar</a> | 
                Data Source: <a href="https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset" target="_blank" style="color: #4ecdc4;">IBM HR Analytics Dataset</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    ) 