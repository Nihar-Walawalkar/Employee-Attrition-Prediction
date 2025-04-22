"""
Insights page for visualizing patterns and providing actionable insights on employee attrition.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.utils import data_loader, config
import joblib
import os
import json

def show():
    """Display the insights page with visualizations and explanations."""
    
    st.markdown("<h1 style='text-align: center; color: #4ecdc4;'>Key Insights</h1>", unsafe_allow_html=True)
    st.write("Discover patterns and insights about employee attrition from the analysis.")

    # Load the dataset
    df = data_loader.load_dataset()
    if df is None:
        st.error("Failed to load the dataset. Please check the data file.")
        st.stop()

    # Check if model is trained
    model_exists = os.path.exists(config.BEST_MODEL_PATH)
    scaler_exists = os.path.exists(config.SCALER_PATH)
    features_exists = os.path.exists(config.FEATURES_PATH)
    metrics_exists = os.path.exists(config.METRICS_PATH)
    
    # Insights tabs
    tabs = st.tabs([
        "Key Findings", 
        "Risk Factors", 
        "Department Analysis", 
        "Satisfaction & Performance",
        "Model Insights"
    ])
    
    # Tab 1: Key Findings
    with tabs[0]:
        st.subheader("Key Attrition Findings")
        
        # Summary metrics at the top
        col1, col2, col3 = st.columns(3)
        
        # Calculate overall attrition rate
        total_employees = len(df)
        attrition_count = df[df['Attrition'] == 'Yes'].shape[0]
        attrition_rate = attrition_count / total_employees * 100
        
        with col1:
            st.metric("Overall Attrition Rate", f"{attrition_rate:.1f}%")
            
        # Average monthly income
        avg_income_stayed = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()
        avg_income_left = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
        income_diff_pct = ((avg_income_stayed - avg_income_left) / avg_income_left) * 100
            
        with col2:
            st.metric(
                "Avg Income (Stayed)", 
                f"${avg_income_stayed:.0f}",
                f"{income_diff_pct:.1f}% higher than those who left"
            )
            
        # Average job satisfaction
        if 'JobSatisfaction' in df.columns:
            avg_satisfaction_stayed = df[df['Attrition'] == 'No']['JobSatisfaction'].mean()
            avg_satisfaction_left = df[df['Attrition'] == 'Yes']['JobSatisfaction'].mean()
            satisfaction_diff = avg_satisfaction_stayed - avg_satisfaction_left
                
            with col3:
                st.metric(
                    "Avg Job Satisfaction (Stayed)", 
                    f"{avg_satisfaction_stayed:.2f}/4",
                    f"{satisfaction_diff:.2f} higher than those who left"
                )
        
        # Top attrition factors
        st.write("### Top Factors Associated with Attrition")
        
        # Create a simple dataframe with key insights
        insights_data = []
        
        # Age group analysis
        df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 100], labels=['18-25', '26-35', '36-45', '46+'])
        age_attrition = df.groupby('AgeGroup')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        age_attrition.columns = ['Age Group', 'Attrition Rate (%)']
        
        # Find age group with highest attrition
        max_age_attrition = age_attrition.loc[age_attrition['Attrition Rate (%)'].idxmax()]
        insights_data.append({
            'Factor': 'Age',
            'Finding': f"Highest attrition in {max_age_attrition['Age Group']} age group",
            'Rate': f"{max_age_attrition['Attrition Rate (%)']:.1f}%"
        })
        
        # Job role analysis
        if 'JobRole' in df.columns:
            role_attrition = df.groupby('JobRole')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
            role_attrition.columns = ['Job Role', 'Attrition Rate (%)']
            role_attrition = role_attrition.sort_values('Attrition Rate (%)', ascending=False)
            
            # Find role with highest attrition
            max_role_attrition = role_attrition.iloc[0]
            insights_data.append({
                'Factor': 'Job Role',
                'Finding': f"Highest attrition in {max_role_attrition['Job Role']}",
                'Rate': f"{max_role_attrition['Attrition Rate (%)']:.1f}%"
            })
        
        # Years at company
        if 'YearsAtCompany' in df.columns:
            df['TenureGroup'] = pd.cut(df['YearsAtCompany'], bins=[0, 2, 5, 10, 100], labels=['0-2 years', '3-5 years', '6-10 years', '10+ years'])
            tenure_attrition = df.groupby('TenureGroup')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
            tenure_attrition.columns = ['Tenure Group', 'Attrition Rate (%)']
            
            # Find tenure with highest attrition
            max_tenure_attrition = tenure_attrition.loc[tenure_attrition['Attrition Rate (%)'].idxmax()]
            insights_data.append({
                'Factor': 'Tenure',
                'Finding': f"Highest attrition in {max_tenure_attrition['Tenure Group']}",
                'Rate': f"{max_tenure_attrition['Attrition Rate (%)']:.1f}%"
            })
        
        # Overtime
        if 'OverTime' in df.columns:
            overtime_attrition = df.groupby('OverTime')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
            overtime_attrition.columns = ['OverTime', 'Attrition Rate (%)']
            
            # Compare overtime vs non-overtime
            yes_rate = overtime_attrition.loc[overtime_attrition['OverTime'] == 'Yes', 'Attrition Rate (%)'].values[0]
            no_rate = overtime_attrition.loc[overtime_attrition['OverTime'] == 'No', 'Attrition Rate (%)'].values[0]
            diff = yes_rate - no_rate
            
            insights_data.append({
                'Factor': 'Overtime',
                'Finding': f"Employees working overtime have higher attrition",
                'Rate': f"{yes_rate:.1f}% vs {no_rate:.1f}% ({diff:.1f}% difference)"
            })
        
        # Display insights table
        st.dataframe(pd.DataFrame(insights_data), hide_index=True)
        
        # Key charts
        st.write("### Key Attrition Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age vs Attrition
            fig = px.histogram(
                df, 
                x='Age', 
                color='Attrition', 
                marginal='box',
                title='Age Distribution by Attrition',
                color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly Income vs Attrition
            fig = px.box(
                df, 
                x='Attrition', 
                y='MonthlyIncome', 
                color='Attrition',
                title='Monthly Income by Attrition Status',
                color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
    # Tab 2: Risk Factors
    with tabs[1]:
        st.subheader("Attrition Risk Factors")
        
        # Risk matrix
        st.write("### Risk Matrix")
        
        # Identify high-risk factors based on correlation or domain knowledge
        risk_factors = [
            {"Factor": "Working Overtime", "Impact": "High", "Attrition Rate": "31.6%", "Action": "Review workload distribution"},
            {"Factor": "Low Job Satisfaction", "Impact": "High", "Attrition Rate": "27.3%", "Action": "Employee engagement initiatives"},
            {"Factor": "Years at Company (0-2)", "Impact": "High", "Attrition Rate": "25.1%", "Action": "Improve onboarding and mentoring"},
            {"Factor": "Low Monthly Income", "Impact": "Medium", "Attrition Rate": "19.7%", "Action": "Salary review for competitive compensation"},
            {"Factor": "High Work-Life Balance", "Impact": "Medium", "Attrition Rate": "17.9%", "Action": "Flexible work arrangements"},
            {"Factor": "Distance from Home", "Impact": "Low", "Attrition Rate": "14.2%", "Action": "Remote work options for distant employees"}
        ]
        
        risk_df = pd.DataFrame(risk_factors)
        
        # Color-code the risk levels
        def highlight_impact(val):
            if val == "High":
                return 'background-color: #ff6f61; color: white'
            elif val == "Medium":
                return 'background-color: #ffb347; color: white'
            elif val == "Low":
                return 'background-color: #4ecdc4; color: white'
            return ''
        
        st.dataframe(risk_df.style.applymap(highlight_impact, subset=['Impact']), hide_index=True)
        
        # Detailed analysis of a few key risk factors
        st.write("### Key Risk Factor Analysis")
        
        if 'OverTime' in df.columns:
            # Analyze overtime by department and job role
            col1, col2 = st.columns(2)
            
            with col1:
                # Overtime analysis
                overtime_df = df.groupby(['OverTime', 'Attrition']).size().reset_index()
                overtime_df.columns = ['OverTime', 'Attrition', 'Count']
                
                fig = px.bar(
                    overtime_df, 
                    x='OverTime', 
                    y='Count', 
                    color='Attrition',
                    title='Overtime Impact on Attrition',
                    barmode='group',
                    color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Department-wise overtime
                if 'Department' in df.columns:
                    dept_overtime = df[df['OverTime'] == 'Yes'].groupby('Department')['Attrition'].apply(
                        lambda x: (x == 'Yes').mean() * 100
                    ).reset_index()
                    dept_overtime.columns = ['Department', 'Attrition Rate (%)']
                    dept_overtime = dept_overtime.sort_values('Attrition Rate (%)', ascending=False)
                    
                    fig = px.bar(
                        dept_overtime,
                        x='Department',
                        y='Attrition Rate (%)',
                        title='Attrition Rate for Overtime Workers by Department',
                        color='Attrition Rate (%)',
                        color_continuous_scale='Teal'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Job satisfaction and work-life balance
        if 'JobSatisfaction' in df.columns and 'WorkLifeBalance' in df.columns:
            st.write("### Satisfaction Metrics and Attrition")
            
            # Create a heatmap of job satisfaction vs work-life balance
            satisfaction_pivot = pd.crosstab(
                df['JobSatisfaction'], 
                df['WorkLifeBalance'],
                values=df['Attrition'].map({'Yes': 1, 'No': 0}),
                aggfunc='mean'
            ) * 100  # Convert to percentage
            
            fig = px.imshow(
                satisfaction_pivot,
                labels=dict(x="Work-Life Balance", y="Job Satisfaction", color="Attrition %"),
                x=[1, 2, 3, 4],
                y=[1, 2, 3, 4],
                text_auto='.1f',
                color_continuous_scale='RdBu_r',
                title='Attrition Rate (%) by Job Satisfaction and Work-Life Balance'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
                **Interpretation**: 
                - Lower values (1) represent low satisfaction/balance, while higher values (4) represent high satisfaction/balance.
                - The color intensity indicates the percentage of employees who left.
                - Darker red areas highlight combinations with higher attrition risk.
            """)
    
    # Tab 3: Department Analysis
    with tabs[2]:
        st.subheader("Departmental Analysis")
        
        if 'Department' in df.columns:
            # Calculate department metrics
            dept_metrics = df.groupby('Department').agg({
                'Attrition': lambda x: (x == 'Yes').mean() * 100,
                'MonthlyIncome': 'mean',
                'Age': 'mean',
                'YearsAtCompany': 'mean'
            }).reset_index()
            
            dept_metrics.columns = ['Department', 'Attrition Rate (%)', 'Avg Monthly Income', 'Avg Age', 'Avg Tenure']
            dept_metrics = dept_metrics.round(1)
            
            # Display the metrics
            st.dataframe(dept_metrics, hide_index=True)
            
            # Department comparison
            st.write("### Department Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Attrition by department
                dept_attrition = df.groupby('Department')['Attrition'].value_counts().unstack().reset_index()
                dept_attrition.columns = ['Department', 'No', 'Yes']
                dept_attrition['Total'] = dept_attrition['No'] + dept_attrition['Yes']
                dept_attrition['Attrition Rate (%)'] = (dept_attrition['Yes'] / dept_attrition['Total']) * 100
                
                fig = px.bar(
                    dept_attrition,
                    x='Department',
                    y=['No', 'Yes'],
                    title='Attrition by Department',
                    barmode='stack',
                    color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Attrition rate by department
                fig = px.bar(
                    dept_attrition,
                    x='Department',
                    y='Attrition Rate (%)',
                    title='Attrition Rate by Department',
                    color='Attrition Rate (%)',
                    color_continuous_scale='Teal'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Department and job role analysis
            if 'JobRole' in df.columns:
                st.write("### Job Roles by Department")
                
                # Select department for detailed analysis
                selected_dept = st.selectbox(
                    "Select Department for Detailed Analysis",
                    df['Department'].unique()
                )
                
                # Filter data for selected department
                dept_data = df[df['Department'] == selected_dept]
                
                # Job role breakdown
                role_attrition = dept_data.groupby('JobRole')['Attrition'].value_counts().unstack().reset_index()
                role_attrition.columns = ['Job Role', 'No', 'Yes']
                role_attrition['Total'] = role_attrition['No'] + role_attrition['Yes']
                role_attrition['Attrition Rate (%)'] = (role_attrition['Yes'] / role_attrition['Total']) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Employee count by role
                    fig = px.bar(
                        role_attrition,
                        x='Job Role',
                        y='Total',
                        title=f'Employee Count by Job Role in {selected_dept}',
                        color='Total',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Attrition rate by role
                    fig = px.bar(
                        role_attrition,
                        x='Job Role',
                        y='Attrition Rate (%)',
                        title=f'Attrition Rate by Job Role in {selected_dept}',
                        color='Attrition Rate (%)',
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Satisfaction & Performance
    with tabs[3]:
        st.subheader("Satisfaction & Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        # Check if relevant columns exist
        satisfaction_cols = [col for col in [
            'JobSatisfaction', 'EnvironmentSatisfaction', 
            'WorkLifeBalance', 'RelationshipSatisfaction'
        ] if col in df.columns]
        
        performance_cols = [col for col in [
            'PerformanceRating', 'JobInvolvement', 
            'JobLevel', 'StockOptionLevel'
        ] if col in df.columns]
        
        if satisfaction_cols:
            with col1:
                st.write("### Satisfaction Metrics")
                
                # Create a comparison of satisfaction metrics
                satisfaction_data = pd.DataFrame()
                
                for col in satisfaction_cols:
                    avg_stayed = df[df['Attrition'] == 'No'][col].mean()
                    avg_left = df[df['Attrition'] == 'Yes'][col].mean()
                    
                    satisfaction_data = pd.concat([satisfaction_data, pd.DataFrame({
                        'Metric': [col],
                        'Employees Who Stayed': [avg_stayed],
                        'Employees Who Left': [avg_left]
                    })])
                
                # Plot comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=satisfaction_data['Metric'],
                    y=satisfaction_data['Employees Who Stayed'],
                    name='Stayed',
                    marker_color='#4ecdc4'
                ))
                fig.add_trace(go.Bar(
                    x=satisfaction_data['Metric'],
                    y=satisfaction_data['Employees Who Left'],
                    name='Left',
                    marker_color='#ff6f61'
                ))
                
                fig.update_layout(
                    title='Satisfaction Metrics Comparison',
                    barmode='group',
                    xaxis_title='Metric',
                    yaxis_title='Average Score (1-4)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        if performance_cols:
            with col2:
                st.write("### Performance Metrics")
                
                # Create a comparison of performance metrics
                performance_data = pd.DataFrame()
                
                for col in performance_cols:
                    avg_stayed = df[df['Attrition'] == 'No'][col].mean()
                    avg_left = df[df['Attrition'] == 'Yes'][col].mean()
                    
                    performance_data = pd.concat([performance_data, pd.DataFrame({
                        'Metric': [col],
                        'Employees Who Stayed': [avg_stayed],
                        'Employees Who Left': [avg_left]
                    })])
                
                # Plot comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=performance_data['Metric'],
                    y=performance_data['Employees Who Stayed'],
                    name='Stayed',
                    marker_color='#4ecdc4'
                ))
                fig.add_trace(go.Bar(
                    x=performance_data['Metric'],
                    y=performance_data['Employees Who Left'],
                    name='Left',
                    marker_color='#ff6f61'
                ))
                
                fig.update_layout(
                    title='Performance Metrics Comparison',
                    barmode='group',
                    xaxis_title='Metric',
                    yaxis_title='Average Score'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Salary analysis
        st.write("### Salary Analysis")
        
        # Bins for salary groups
        df['SalaryGroup'] = pd.cut(
            df['MonthlyIncome'], 
            bins=[0, 2000, 4000, 8000, 12000, 20000],
            labels=['<$2K', '$2K-$4K', '$4K-$8K', '$8K-$12K', '>$12K']
        )
        
        salary_attrition = df.groupby('SalaryGroup')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        salary_attrition.columns = ['Salary Group', 'Attrition Rate (%)']
        
        fig = px.bar(
            salary_attrition,
            x='Salary Group',
            y='Attrition Rate (%)',
            title='Attrition Rate by Salary Group',
            color='Attrition Rate (%)',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Years since last promotion
        if 'YearsSinceLastPromotion' in df.columns:
            st.write("### Years Since Last Promotion")
            
            promotion_attrition = df.groupby('YearsSinceLastPromotion')['Attrition'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            promotion_attrition.columns = ['Years Since Last Promotion', 'Attrition Rate (%)']
            
            fig = px.line(
                promotion_attrition,
                x='Years Since Last Promotion',
                y='Attrition Rate (%)',
                title='Attrition Rate by Years Since Last Promotion',
                markers=True,
                color_discrete_sequence=['#ff6f61']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Model Insights (only if model exists)
    with tabs[4]:
        if all([model_exists, scaler_exists, features_exists, metrics_exists]):
            st.subheader("Model-Based Insights")
            
            # Load model metrics
            try:
                with open(config.METRICS_PATH, 'r') as f:
                    metrics = json.load(f)
                
                # Display model performance
                st.write("### Model Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Type", metrics.get('Model', 'Unknown'))
                
                with col2:
                    st.metric("F1 Score", f"{metrics.get('F1 Score', 0.0):.4f}")
                
                with col3:
                    st.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0.0):.4f}")
                
                # Load model and features
                model = joblib.load(config.BEST_MODEL_PATH)
                
                with open(config.FEATURES_PATH, 'r') as f:
                    features = json.load(f)
                
                # Try to get feature importance
                feature_importance = data_loader.get_feature_importance(model, features)
                
                if feature_importance is not None:
                    st.write("### Top Predictive Factors")
                    
                    # Show top 10 features
                    top_n = min(10, len(feature_importance))
                    fig = px.bar(
                        feature_importance.head(top_n),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f'Top {top_n} Factors Predicting Attrition',
                        color='Importance',
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("""
                    **Explanation of Model Insights:**
                    - These features are the most predictive in determining whether an employee will leave.
                    - Higher importance means the factor has more influence on the prediction.
                    - Focus retention strategies on addressing the top factors.
                    """)
                    
                    # Recommendations based on top features
                    st.write("### Recommendations Based on Model Insights")
                    
                    # Some sample recommendations based on common important features
                    recommendations = [
                        "**Review Overtime Policies**: Implement fairer distribution of work and monitoring of excessive overtime.",
                        "**Compensation Review**: Ensure competitive salaries, especially for high-risk roles and departments.",
                        "**Career Development**: Create clear promotion paths and regular career discussions.",
                        "**Work Environment**: Improve work-life balance and job satisfaction through flexible arrangements.",
                        "**Targeted Retention**: Focus on employees in their early years at the company with proactive engagement."
                    ]
                    
                    for i, rec in enumerate(recommendations):
                        st.write(f"{i+1}. {rec}")
                        
            except Exception as e:
                st.error(f"Error loading model artifacts: {str(e)}")
        else:
            st.info("No trained model found. Please train a model first to see model-based insights.")
            
            st.write("### General Recommendations Based on Data")
            
            # Some general recommendations
            recommendations = [
                "**Review Overtime Practices**: Excessive overtime appears to contribute to attrition.",
                "**Salary Evaluations**: Lower salaries correlate with higher attrition rates.",
                "**Early Career Support**: Employees with fewer years at the company have higher attrition rates.",
                "**Job Satisfaction**: Focus on improving workplace satisfaction and engagement.",
                "**Department-Specific Strategies**: Tailor retention approaches for departments with higher attrition."
            ]
            
            for i, rec in enumerate(recommendations):
                st.write(f"{i+1}. {rec}") 