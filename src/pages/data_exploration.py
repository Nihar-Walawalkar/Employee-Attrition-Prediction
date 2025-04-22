"""
Data Exploration page for visualizing and analyzing the HR dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.utils import data_loader, config

def show():
    """Display the data exploration page."""
    
    st.markdown("<h1 style='text-align: center; color: #4ecdc4;'>Data Exploration</h1>", unsafe_allow_html=True)
    st.write("Explore the IBM HR Analytics Employee Attrition Dataset with interactive visualizations.")

    # Load the dataset
    df = data_loader.load_dataset()
    if df is None:
        st.error("Failed to load the dataset. Please check the data file.")
        st.stop()

    # Sidebar for exploration options
    st.sidebar.subheader("Exploration Options")
    exploration_option = st.sidebar.radio(
        "Select View",
        ["Dataset Overview", "Descriptive Statistics", "Key Metrics", "Correlation Analysis", "Custom Analysis"]
    )

    # Dataset Overview
    if exploration_option == "Dataset Overview":
        st.subheader("Dataset Overview")
        
        # Display sample of the dataset
        st.write("### First 5 rows of the dataset")
        st.dataframe(df.head())
        
        # Dataset shape
        st.write(f"**Dataset shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for missing values
        st.write("### Missing Values")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.dataframe(missing_values[missing_values > 0])
        else:
            st.success("No missing values found in the dataset.")
        
        # Data types
        st.write("### Data Types")
        
        # Group columns by data type for better organization
        dtypes = pd.DataFrame(df.dtypes, columns=['Data Type'])
        dtypes = dtypes.reset_index().rename(columns={'index': 'Column'})
        
        # Display in three columns
        col1, col2, col3 = st.columns(3)
        
        # Numeric columns
        numeric_cols = dtypes[dtypes['Data Type'].isin(['int64', 'float64'])]
        with col1:
            st.write("**Numeric Columns**")
            st.dataframe(numeric_cols, height=300)
            
        # Categorical columns
        cat_cols = dtypes[dtypes['Data Type'].isin(['object', 'category', 'bool'])]
        with col2:
            st.write("**Categorical Columns**")
            st.dataframe(cat_cols, height=300)
            
        # Other columns
        other_cols = dtypes[~dtypes['Data Type'].isin(['int64', 'float64', 'object', 'category', 'bool'])]
        with col3:
            st.write("**Other Columns**")
            st.dataframe(other_cols, height=300)
        
        # Unique values for categorical columns
        st.write("### Unique Values for Categorical Columns")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            for col in categorical_cols:
                unique_values = df[col].unique()
                st.write(f"**{col}:** {', '.join(map(str, unique_values))}")
        else:
            st.info("No categorical columns found.")

    # Descriptive Statistics
    elif exploration_option == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        
        # Summary statistics
        st.write("### Summary Statistics for Numeric Columns")
        st.dataframe(df.describe().round(2))
        
        # Categorical columns summary
        st.write("### Categorical Columns Summary")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            # Create tabs for each categorical column
            tabs = st.tabs(categorical_cols)
            
            for i, col in enumerate(categorical_cols):
                with tabs[i]:
                    # Calculate value counts
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, 'Count']
                    value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
                    
                    # Display as table and chart
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(value_counts)
                    
                    with col2:
                        fig = px.pie(value_counts, values='Count', names=col, title=f"{col} Distribution", 
                                    color_discrete_sequence=px.colors.sequential.Teal)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")
        
        # Data distribution for numeric columns
        st.write("### Data Distribution for Numeric Columns")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            selected_column = st.selectbox("Select a numeric column for distribution analysis:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_column, color_discrete_sequence=['#4ecdc4'],
                                  marginal="box", title=f"Distribution of {selected_column}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Basic statistics
                stats = df[selected_column].describe().to_dict()
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Mean", f"{stats['mean']:.2f}")
                    st.metric("Median", f"{stats['50%']:.2f}")
                    st.metric("Std Dev", f"{stats['std']:.2f}")
                
                with metrics_col2:
                    st.metric("Min", f"{stats['min']:.2f}")
                    st.metric("Max", f"{stats['max']:.2f}")
                    st.metric("IQR", f"{stats['75%'] - stats['25%']:.2f}")
                
                # Skewness and kurtosis
                skewness = df[selected_column].skew()
                kurtosis = df[selected_column].kurt()
                
                st.write(f"**Skewness:** {skewness:.2f}")
                st.write(f"**Kurtosis:** {kurtosis:.2f}")

    # Key Metrics
    elif exploration_option == "Key Metrics":
        st.subheader("Key Metrics and Visualizations")
        
        # Attrition Distribution (Overall)
        st.write("### Overall Attrition Distribution")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            attrition_counts = df['Attrition'].value_counts().reset_index()
            attrition_counts.columns = ['Attrition', 'Count']
            
            fig = px.pie(attrition_counts, values='Count', names='Attrition', title='Attrition Distribution',
                       color='Attrition', color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            total = len(df)
            attrition_count = df[df['Attrition'] == 'Yes'].shape[0]
            retention_count = df[df['Attrition'] == 'No'].shape[0]
            attrition_rate = attrition_count / total * 100
            
            st.markdown(
                f"""
                <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="text-align:center; color:#4ecdc4;">Attrition Statistics</h4>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span>Total Employees:</span>
                        <span style="font-weight: bold;">{total}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span>Stayed:</span>
                        <span style="font-weight: bold; color:#4ecdc4;">{retention_count} ({100-attrition_rate:.1f}%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span>Left:</span>
                        <span style="font-weight: bold; color:#ff6f61;">{attrition_count} ({attrition_rate:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Attrition by Department, Job Role, etc.
        st.write("### Attrition by Categorical Variables")
        
        categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'Attrition']
        selected_categorical = st.selectbox("Select a categorical variable:", categorical_cols)
        
        fig = px.histogram(df, x=selected_categorical, color='Attrition',
                         barmode='group', color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'},
                         title=f'Attrition by {selected_categorical}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Attrition rates by category
        category_attrition = df.groupby(selected_categorical)['Attrition'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        category_attrition.columns = [selected_categorical, 'Attrition Rate (%)']
        category_attrition = category_attrition.sort_values('Attrition Rate (%)', ascending=False)
        
        fig = px.bar(category_attrition, x=selected_categorical, y='Attrition Rate (%)',
                   title=f'Attrition Rate by {selected_categorical}',
                   color='Attrition Rate (%)', color_continuous_scale='Teal')
        st.plotly_chart(fig, use_container_width=True)
        
        # Numerical variables and attrition
        st.write("### Numerical Variables by Attrition")
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_numerical = st.selectbox("Select a numerical variable:", numerical_cols)
        
        fig = px.box(df, x='Attrition', y=selected_numerical, color='Attrition',
                   color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'},
                   title=f'{selected_numerical} by Attrition')
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Analysis
    elif exploration_option == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        # Data preprocessing for correlation
        df_corr = df.copy()
        
        # Convert categorical columns to numeric
        categorical_cols = df_corr.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_corr[col] = df_corr[col].astype('category').cat.codes
        
        # Calculate correlation matrix
        corr_matrix = df_corr.corr().round(2)
        
        # Display full correlation matrix
        st.write("### Full Correlation Matrix")
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                      zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with target variable (Attrition)
        st.write("### Correlation with Attrition")
        
        # Sort correlations with Attrition
        attrition_corr = corr_matrix['Attrition'].drop('Attrition').sort_values(ascending=False)
        
        # Top positive and negative correlations
        top_positive = attrition_corr[attrition_corr > 0].head(10)
        top_negative = attrition_corr[attrition_corr < 0].sort_values().head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Positive Correlations**")
            
            fig = px.bar(top_positive, orientation='h', 
                       color=top_positive.values, color_continuous_scale='Teal',
                       title="Variables Positively Correlated with Attrition",
                       labels={'value': 'Correlation Coefficient', 'index': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Top Negative Correlations**")
            
            fig = px.bar(top_negative, orientation='h',
                       color=top_negative.values, color_continuous_scale='Teal_r',
                       title="Variables Negatively Correlated with Attrition",
                       labels={'value': 'Correlation Coefficient', 'index': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)

    # Custom Analysis
    elif exploration_option == "Custom Analysis":
        st.subheader("Custom Analysis")
        
        # Custom scatter plot
        st.write("### Custom Scatter Plot")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            x_axis = st.selectbox("Select X-axis variable:", numeric_cols, index=numeric_cols.index('Age') if 'Age' in numeric_cols else 0)
        
        with col2:
            y_axis = st.selectbox("Select Y-axis variable:", numeric_cols, index=numeric_cols.index('MonthlyIncome') if 'MonthlyIncome' in numeric_cols else 0)
        
        with col3:
            categorical_cols = ['Attrition'] + df.select_dtypes(include=['object']).columns.tolist()
            categorical_cols = list(dict.fromkeys(categorical_cols))  # Remove duplicates while preserving order
            color_by = st.selectbox("Color by:", categorical_cols)
        
        # Create scatter plot
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                       title=f'{y_axis} vs {x_axis} by {color_by}',
                       opacity=0.7, size_max=10, 
                       color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'} if color_by == 'Attrition' else None)
        st.plotly_chart(fig, use_container_width=True)
        
        # Custom histogram
        st.write("### Custom Histogram")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hist_variable = st.selectbox("Select variable for histogram:", numeric_cols)
        
        with col2:
            group_by = st.selectbox("Group by:", categorical_cols)
        
        # Create histogram
        fig = px.histogram(df, x=hist_variable, color=group_by,
                         title=f'Distribution of {hist_variable} by {group_by}',
                         barmode='group', histnorm='percent',
                         color_discrete_map={'Yes': '#ff6f61', 'No': '#4ecdc4'} if group_by == 'Attrition' else None)
        st.plotly_chart(fig, use_container_width=True)
        
        # Custom groupby analysis
        st.write("### Custom Group Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            group_var = st.selectbox("Group by variable:", categorical_cols)
        
        with col2:
            measure_var = st.selectbox("Measure variable:", numeric_cols)
        
        with col3:
            agg_func = st.selectbox("Aggregation function:", ["Mean", "Median", "Min", "Max", "Count", "Sum", "Std Dev"])
        
        # Map selected aggregation function to pandas function
        agg_map = {
            "Mean": "mean",
            "Median": "median",
            "Min": "min",
            "Max": "max",
            "Count": "count",
            "Sum": "sum",
            "Std Dev": "std"
        }
        
        # Perform groupby aggregation
        grouped_data = df.groupby(group_var)[measure_var].agg(agg_map[agg_func]).reset_index()
        grouped_data = grouped_data.sort_values(measure_var, ascending=False)
        grouped_data.columns = [group_var, f"{agg_func} of {measure_var}"]
        
        # Display as bar chart
        fig = px.bar(grouped_data, x=group_var, y=f"{agg_func} of {measure_var}",
                   title=f"{agg_func} of {measure_var} by {group_var}",
                   color=f"{agg_func} of {measure_var}", color_continuous_scale='Teal')
        st.plotly_chart(fig, use_container_width=True) 