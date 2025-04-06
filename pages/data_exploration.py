import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.markdown("<h1>Data Exploration</h1>", unsafe_allow_html=True)
    st.write("Explore the IBM HR Analytics Employee Attrition Dataset with interactive visualizations.")

    # Load the dataset
    try:
        df = pd.read_csv("data/employee_attrition.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Ensure 'data/employee_attrition.csv' exists.")
        st.stop()

    # Data Preprocessing (same as Colab)
    st.subheader("Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Drop unnecessary columns
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # Visualizations
    st.subheader("Visualizations")

    # Distribution of Attrition
    st.write("### Attrition Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Attrition', data=df, ax=ax)
    ax.set_title('Attrition Distribution')
    st.pyplot(fig)

    # Correlation Matrix
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

    # Attrition by Department
    st.write("### Attrition by Department")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Department', hue='Attrition', data=df, ax=ax)
    ax.set_title('Attrition by Department')
    st.pyplot(fig)

    # Attrition by Job Role
    st.write("### Attrition by Job Role")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='JobRole', hue='Attrition', data=df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title('Attrition by Job Role')
    st.pyplot(fig)

    # Age Distribution
    st.write("### Age Distribution by Attrition")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='Age', hue='Attrition', kde=True, ax=ax)
    ax.set_title('Age Distribution by Attrition')
    st.pyplot(fig)

    # Monthly Income vs Attrition
    st.write("### Monthly Income vs Attrition")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=ax)
    ax.set_title('Monthly Income vs Attrition')
    st.pyplot(fig)

    # Job Satisfaction vs Attrition
    st.write("### Job Satisfaction vs Attrition")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='JobSatisfaction', hue='Attrition', data=df, ax=ax)
    ax.set_title('Job Satisfaction vs Attrition')
    st.pyplot(fig)

    # Years at Company vs Attrition
    st.write("### Years at Company vs Attrition")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='YearsAtCompany', hue='Attrition', kde=True, ax=ax)
    ax.set_title('Years at Company vs Attrition')
    st.pyplot(fig)