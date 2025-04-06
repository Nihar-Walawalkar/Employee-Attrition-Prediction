import streamlit as st
import pandas as pd
import numpy as np
import joblib

def show():
    st.markdown("<h1>Prediction</h1>", unsafe_allow_html=True)
    st.write("Enter employee details below to predict attrition risk.")

    # Load model, scaler, and top features
    try:
        model = joblib.load("model/best_attrition_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        top_features = joblib.load("model/top_features.pkl")
    except FileNotFoundError:
        st.error("Model files not found. Ensure 'model/' contains the required .pkl files.")
        st.stop()

    # Display the expected features for debugging
    st.write("Top Features Expected by Model:", top_features)

    # Input form
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 65, 35)
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
            work_life_balance = st.selectbox("Work-Life Balance (1-4)", [1, 2, 3, 4])
            daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800, step=10)
            monthly_rate = st.number_input("Monthly Rate ($)", min_value=1000, max_value=30000, value=15000, step=100)

        with col2:
            years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            job_role = st.selectbox("Job Role", range(9), format_func=lambda x: ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"][x])
            distance_from_home = st.slider("Distance from Home (miles)", 1, 29, 10)
            overtime = st.checkbox("Overtime (Yes=1, No=0)")
            hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65, step=1)
            num_companies_worked = st.slider("Number of Companies Worked", 0, 9, 1)

    # Prediction button
    if st.button("Predict Attrition"):
        # Prepare input data
        input_data = {
            "Age": age,
            "MonthlyIncome": monthly_income,
            "YearsAtCompany": years_at_company,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "TotalWorkingYears": total_working_years,
            "JobRole": job_role,
            "DistanceFromHome": distance_from_home,
            "OverTime": 1 if overtime else 0,
            "DailyRate": daily_rate,
            "MonthlyRate": monthly_rate,
            "HourlyRate": hourly_rate,
            "NumCompaniesWorked": num_companies_worked
        }
        
        # Validate that all required features are present
        missing_features = [feature for feature in top_features if feature not in input_data]
        if missing_features:
            st.error(f"Missing features in input data: {missing_features}")
            st.stop()
        
        # Convert to DataFrame and select top features
        input_df = pd.DataFrame([input_data])[top_features]
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
        
        # Display result
        result_class = "success" if prediction == 0 else "warning"
        result_text = "Employee is likely to stay" if prediction == 0 else "Employee is likely to leave"
        probability_text = f"Probability of leaving: {probability:.2%}" if probability is not None else ""
        
        st.markdown(
            f"""
            <div class="result-box {result_class}">
                <h2>{result_text}</h2>
                <p>{probability_text}</p>
            </div>
            """,
            unsafe_allow_html=True
        )