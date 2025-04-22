"""
Predictions page for the Employee Attrition Prediction application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
from src.utils import data_loader, config

def load_model_artifacts():
    """
    Load the trained model and artifacts.
    
    Returns:
        tuple: (model, scaler, features) or (None, None, None) if not found
    """
    try:
        # Check if model files exist
        if not os.path.exists(config.BEST_MODEL_PATH) or \
           not os.path.exists(config.SCALER_PATH) or \
           not os.path.exists(config.FEATURES_PATH):
            return None, None, None
        
        # Load model
        model = joblib.load(config.BEST_MODEL_PATH)
        
        # Load scaler
        scaler = joblib.load(config.SCALER_PATH)
        
        # Load features
        with open(config.FEATURES_PATH, 'r') as f:
            features = json.load(f)
        
        return model, scaler, features
    
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None

def get_feature_names_from_dataset(df):
    """
    Generate potential feature names from the dataset by simulating preprocessing.
    
    Args:
        df (pd.DataFrame): The dataset
        
    Returns:
        list: List of potential feature names
    """
    data = df.copy()
    
    # Remove target column if exists
    if config.TARGET_COLUMN in data.columns:
        data = data.drop(config.TARGET_COLUMN, axis=1)
    
    # Handle categorical variables
    cat_columns = data.select_dtypes(include=['object', 'category']).columns
    dummy_data = pd.get_dummies(data, columns=cat_columns, drop_first=True)
    
    # Return list of column names
    return dummy_data.columns.tolist()

def individual_prediction():
    """Provide interface for individual employee prediction."""
    
    st.subheader("Predict Individual Employee Attrition")
    
    # Load model artifacts
    model, scaler, features = load_model_artifacts()
    
    if model is None:
        st.warning("No trained model found. Please train a model first.")
        if st.button("Go to Model Training"):
            st.session_state.current_page = "Model Training"
            st.rerun()
        return
    
    # Load dataset to get possible values for categorical features
    df = data_loader.load_dataset()
    if df is None:
        st.error("Failed to load dataset for feature extraction.")
        return
    
    # Check if features is a list or None
    if features is None:
        st.warning("No feature list found. Using features from the dataset.")
        features = get_feature_names_from_dataset(df)
    
    # Create input form
    col1, col2 = st.columns(2)
    
    # Personal Information
    with col1:
        st.write("#### Personal Information")
        
        age = st.slider("Age", 18, 65, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 10)
        education_level = st.slider("Education Level", 1, 5, 3, help="1 = Below College, 5 = Doctor")
        
    # Work Information
    with col2:
        st.write("#### Work Information")
        
        job_level = st.slider("Job Level", 1, 5, 2)

def show():
    """Display the predictions page."""
    
    st.markdown("<h1 style='text-align: center; color: #4ecdc4;'>Attrition Predictions</h1>", unsafe_allow_html=True)
    st.write("Make predictions on employee attrition using the trained model.")

    # Check if model is trained
    model_exists = os.path.exists(config.BEST_MODEL_PATH)
    scaler_exists = os.path.exists(config.SCALER_PATH)
    features_exists = os.path.exists(config.FEATURES_PATH)
    
    if not all([model_exists, scaler_exists, features_exists]):
        st.error("No trained model found. Please train a model first.")
        st.info("Go to the 'Model Training' page to train a model.")
        return

    # Load model and artifacts
    try:
        model = joblib.load(config.BEST_MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        
        with open(config.FEATURES_PATH, "r") as f:
            features = json.load(f)
            
        with open(config.METRICS_PATH, "r") as f:
            metrics = json.load(f)
            
        # Display model info
        st.sidebar.subheader("Model Information")
        st.sidebar.info(f"Model: {metrics.get('Model', 'Unknown')}")
        st.sidebar.info(f"F1 Score: {metrics.get('F1 Score', 0.0):.4f}")
        st.sidebar.info(f"Features: {len(features)}")
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Prediction tabs
    pred_type = st.radio(
        "Prediction Type",
        ["Individual Prediction", "Batch Prediction"]
    )
    
    # Load sample dataset for reference
    df = data_loader.load_dataset()
    
    # Individual Prediction
    if pred_type == "Individual Prediction":
        st.subheader("Predict Individual Employee Attrition")
        
        # Create columns for a better layout
        col1, col2 = st.columns(2)
        
        # Initialize employee data dictionary
        employee_data = {}
        
        # Personal Information
        with col1:
            st.write("**Personal Information**")
            employee_data['Age'] = st.number_input("Age", min_value=18, max_value=70, value=35)
            employee_data['Gender'] = st.selectbox("Gender", ["Male", "Female"])
            employee_data['MaritalStatus'] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            
            if 'DistanceFromHome' in features:
                employee_data['DistanceFromHome'] = st.slider("Distance From Home (miles)", 1, 30, 10)
                
            if 'EducationField' in features:
                employee_data['EducationField'] = st.selectbox(
                    "Education Field", 
                    ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
                )
                
            if 'Education' in features:
                employee_data['Education'] = st.slider("Education Level", 1, 5, 3, 
                                                      help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctoral")
        
        # Work Information
        with col2:
            st.write("**Work Information**")
            
            if 'Department' in features:
                employee_data['Department'] = st.selectbox(
                    "Department", 
                    ["Sales", "Research & Development", "Human Resources"]
                )
                
            if 'JobRole' in features:
                job_roles = [
                    "Sales Executive", "Research Scientist", "Laboratory Technician", 
                    "Manufacturing Director", "Healthcare Representative", "Manager", 
                    "Sales Representative", "Research Director", "Human Resources"
                ]
                employee_data['JobRole'] = st.selectbox("Job Role", job_roles)
            
            if 'JobLevel' in features:
                employee_data['JobLevel'] = st.slider("Job Level", 1, 5, 2)
                
            if 'MonthlyIncome' in features:
                employee_data['MonthlyIncome'] = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500)
                
            if 'YearsAtCompany' in features:
                employee_data['YearsAtCompany'] = st.slider("Years at Company", 0, 40, 5)
                
            if 'YearsInCurrentRole' in features:
                employee_data['YearsInCurrentRole'] = st.slider("Years in Current Role", 0, 20, 3)
        
        # Additional Factors
        st.write("**Additional Factors**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'OverTime' in features:
                employee_data['OverTime'] = st.selectbox("Works Overtime", ["Yes", "No"])
                
            if 'BusinessTravel' in features:
                employee_data['BusinessTravel'] = st.selectbox(
                    "Business Travel", 
                    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
                )
                
            if 'TrainingTimesLastYear' in features:
                employee_data['TrainingTimesLastYear'] = st.slider("Training Times Last Year", 0, 6, 2)
        
        with col2:
            if 'JobSatisfaction' in features:
                employee_data['JobSatisfaction'] = st.slider("Job Satisfaction", 1, 4, 3, 
                                                           help="1=Low, 4=High")
                
            if 'WorkLifeBalance' in features:
                employee_data['WorkLifeBalance'] = st.slider("Work-Life Balance", 1, 4, 3,
                                                           help="1=Bad, 4=Excellent")
                
            if 'EnvironmentSatisfaction' in features:
                employee_data['EnvironmentSatisfaction'] = st.slider("Environment Satisfaction", 1, 4, 3,
                                                                    help="1=Low, 4=High")
        
        with col3:
            if 'JobInvolvement' in features:
                employee_data['JobInvolvement'] = st.slider("Job Involvement", 1, 4, 3,
                                                          help="1=Low, 4=High")
                
            if 'PerformanceRating' in features:
                employee_data['PerformanceRating'] = st.slider("Performance Rating", 1, 4, 3,
                                                             help="1=Low, 4=Excellent")
                
            if 'StockOptionLevel' in features:
                employee_data['StockOptionLevel'] = st.slider("Stock Option Level", 0, 3, 1)
        
        # Make prediction when button is clicked
        if st.button("Predict Attrition", key="individual_predict"):
            # Prepare data for prediction
            with st.spinner("Predicting..."):
                # Create a DataFrame with the employee data
                employee_df = pd.DataFrame([employee_data])
                
                # Preprocess the data (same way as during training) with silent mode enabled
                X_pred = data_loader.safe_preprocess_for_prediction(employee_df, features, scaler, silent=True)
                
                if X_pred is not None:
                    # Make prediction
                    prediction_proba = model.predict_proba(X_pred)[0, 1]
                    prediction = "Yes" if prediction_proba >= 0.5 else "No"
                    
                    # Display prediction
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Create a gauge chart for probability
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction_proba * 100,
                            title={"text": "Attrition Risk"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#ff6f61" if prediction_proba >= 0.5 else "#4ecdc4"},
                                "steps": [
                                    {"range": [0, 33], "color": "#4ecdc4"},
                                    {"range": [33, 66], "color": "#ffb347"},
                                    {"range": [66, 100], "color": "#ff6f61"}
                                ],
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Display prediction details
                        risk_level = "High" if prediction_proba >= 0.7 else "Medium" if prediction_proba >= 0.3 else "Low"
                        risk_color = "#ff6f61" if risk_level == "High" else "#ffb347" if risk_level == "Medium" else "#4ecdc4"
                        
                        st.markdown(
                            f"""
                            <div style="background-color: {risk_color}; padding: 20px; border-radius: 10px; color: white;">
                                <h3 style="margin: 0; color: white;">Prediction: {prediction}</h3>
                                <p style="margin: 10px 0 0 0; font-size: 18px;">This employee has a {prediction_proba*100:.1f}% probability of leaving.</p>
                                <p style="margin: 5px 0 0 0;">Risk Level: <strong>{risk_level}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Add call to action for high-risk employees
                        if risk_level == "High":
                            st.markdown(
                                """
                                <div style="margin-top: 20px; padding: 15px; background-color: #f8d7da; border-radius: 5px; color: #721c24;">
                                    <h4 style="margin-top: 0;">Recommended Action</h4>
                                    <p>This employee is at high risk of attrition. Consider creating a retention plan.</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            if st.button("Create Retention Plan"):
                                st.session_state.current_page = "Retention Planner"
                                st.rerun()
                    
                    # Display feature importance instead of SHAP values
                    st.write("### Feature Importance")
                    
                    # Get feature importance from model if available
                    feature_importance = data_loader.get_feature_importance(model, features)
                    
                    if feature_importance is not None:
                        # Show top 10 features
                        fig = px.bar(
                            feature_importance.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Features by Importance',
                            color='Importance',
                            color_continuous_scale='RdBu'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Actionable insights based on most influential features
                        st.write("### Actionable Insights")
                        st.write("Based on the most important features for this prediction, consider the following:")
                        
                        # Extract top influential features
                        top_features = feature_importance.head(5)['Feature'].tolist()
                        
                        # Provide insights based on important features
                        insights = []
                        
                        if any('OverTime' in feature for feature in top_features):
                            insights.append("- Consider reviewing overtime policies or workload distribution")
                            
                        if any('JobSatisfaction' in feature for feature in top_features):
                            insights.append("- Schedule a job satisfaction discussion")
                            
                        if any('WorkLifeBalance' in feature for feature in top_features):
                            insights.append("- Review work-life balance initiatives")
                            
                        if any('YearsAtCompany' in feature for feature in top_features) or any('YearsInCurrentRole' in feature for feature in top_features):
                            insights.append("- Consider career progression opportunities")
                            
                        if any('MonthlyIncome' in feature for feature in top_features):
                            insights.append("- Review compensation package")
                            
                        if any('JobInvolvement' in feature for feature in top_features):
                            insights.append("- Explore opportunities for increased involvement in projects")
                            
                        if not insights:
                            insights = ["- Schedule a general check-in meeting", "- Review career development plan", "- Discuss potential concerns"]
                            
                        for insight in insights:
                            st.write(insight)
                else:
                    st.error("Error preprocessing data for prediction. Please check the input values.")
    
    # Batch Prediction
    else:
        st.subheader("Batch Prediction")
        
        # File upload option
        st.write("Upload a CSV file with employee data to predict attrition for multiple employees.")
        
        upload_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Option to download a template
            st.write("**Need a template?**")
            
            if st.button("Download Template CSV"):
                # Create a template DataFrame with the required columns
                template_df = pd.DataFrame(columns=features)
                
                # Convert to CSV
                csv = template_df.to_csv(index=False)
                
                # Create download link
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="employee_template.csv">Download Template CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Option to use sample data
            st.write("**Or use sample data**")
            
            if st.button("Use Sample Data") and df is not None:
                try:
                    # Use a subset of the actual dataset
                    sample_size = min(10, len(df))
                    sample_df = df.sample(sample_size).reset_index(drop=True)
                    
                    # Don't try to filter columns to model features directly
                    # Instead, just use sample data as is and let preprocessing handle it
                    
                    # Display sample data for user
                    st.write("**Sample Data Preview:**")
                    st.dataframe(sample_df.head())
                    
                    # Store in session state
                    st.session_state.batch_df = sample_df
                except Exception as e:
                    st.error(f"Error preparing sample data: {str(e)}")
                    st.info("Please check that the dataset has the necessary columns.")
        
        # Process uploaded file
        if upload_file is not None:
            try:
                batch_df = pd.read_csv(upload_file)
                st.success("File uploaded successfully!")
                
                # Check for required columns
                missing_cols = [col for col in features if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.info(f"Required columns: {', '.join(features)}")
                else:
                    # Show preview
                    st.write("**Preview:**")
                    st.dataframe(batch_df.head())
                    
                    # Store in session state
                    st.session_state.batch_df = batch_df
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Make batch prediction
        if st.button("Run Batch Prediction") and getattr(st.session_state, 'batch_df', None) is not None:
            batch_df = st.session_state.batch_df
            
            with st.spinner("Processing batch predictions..."):
                # Preprocess for prediction (show info messages for batch mode)
                X_batch = data_loader.safe_preprocess_for_prediction(batch_df, features, scaler, silent=False)
                
                if X_batch is not None:
                    # Predict probabilities
                    probas = model.predict_proba(X_batch)[:, 1]
                    predictions = ["Yes" if p >= 0.5 else "No" for p in probas]
                    
                    # Create result DataFrame
                    result_df = batch_df.copy()
                    result_df['Attrition_Probability'] = probas
                    result_df['Predicted_Attrition'] = predictions
                    result_df['Risk_Level'] = pd.cut(
                        probas, 
                        bins=[0, 0.33, 0.66, 1], 
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Summary statistics
                    total = len(result_df)
                    high_risk = len(result_df[result_df['Risk_Level'] == 'High'])
                    medium_risk = len(result_df[result_df['Risk_Level'] == 'Medium'])
                    low_risk = len(result_df[result_df['Risk_Level'] == 'Low'])
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("High Risk Employees", f"{high_risk} ({high_risk/total*100:.1f}%)")
                    
                    with col2:
                        st.metric("Medium Risk Employees", f"{medium_risk} ({medium_risk/total*100:.1f}%)")
                        
                    with col3:
                        st.metric("Low Risk Employees", f"{low_risk} ({low_risk/total*100:.1f}%)")
                    
                    # Results table with conditional formatting
                    st.write("**Detailed Results:**")
                    
                    # Sort by risk (high to low)
                    result_df = result_df.sort_values('Attrition_Probability', ascending=False)
                    
                    # Format the probabilities as percentages
                    result_df['Attrition_Probability'] = result_df['Attrition_Probability'].apply(lambda x: f"{x*100:.1f}%")
                    
                    # Display the styled dataframe (use newer styling API)
                    def style_risk_level(df):
                        return df.style.map(
                            lambda val: 'background-color: #ff6f61; color: white' if val == 'High' 
                                   else 'background-color: #ffb347; color: white' if val == 'Medium'
                                   else 'background-color: #4ecdc4; color: white' if val == 'Low'
                                   else '', 
                            subset=['Risk_Level']
                        )
                    
                    st.dataframe(style_risk_level(result_df), height=400)
                    
                    # Visualization
                    st.subheader("Risk Distribution")
                    
                    # Distribution of risk levels
                    risk_counts = result_df['Risk_Level'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Level', 'Count']
                    
                    fig = px.pie(
                        risk_counts, 
                        values='Count', 
                        names='Risk Level',
                        title='Distribution of Attrition Risk Levels',
                        color='Risk Level',
                        color_discrete_map={
                            'High': '#ff6f61',
                            'Medium': '#ffb347',
                            'Low': '#4ecdc4'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = result_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"attrition_predictions_{timestamp}.csv"
                    href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}" class="btn">Download Prediction Results</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Store in session state for further analysis
                    st.session_state.prediction_results = result_df
                else:
                    st.error("Error preprocessing data for batch prediction.") 