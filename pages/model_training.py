import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def show():
    st.markdown("<h1>Model Training</h1>", unsafe_allow_html=True)
    st.write("Select a model to train for predicting employee attrition and evaluate its performance.")

    # Load the dataset
    try:
        df = pd.read_csv("data/employee_attrition.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Ensure 'data/employee_attrition.csv' exists.")
        st.stop()

    # Preprocessing
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # Separate features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature Selection with Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, y)

    # Get feature importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    # Display feature importance as a plot
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax, palette='Blues_d')
    ax.set_title('Top 10 Feature Importance')
    st.pyplot(fig)

    # Select top 10 features
    top_features = feature_importance['Feature'].head(10).tolist()
    st.write("### Top 10 Features Selected")
    st.write(top_features)

    X_selected = X[top_features]
    X_selected_scaled = scaler.fit_transform(X_selected)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_selected_scaled, y, test_size=0.2, random_state=42)

    # Model selection dropdown
    model_options = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    selected_model_name = st.selectbox("Select Model to Train", list(model_options.keys()))
    selected_model = model_options[selected_model_name]

    # Train the model
    if st.button("Train Model"):
        model = selected_model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Display evaluation metrics
        st.subheader(f"Evaluation of {selected_model_name}")
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None
        }
        st.write(metrics)

        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        # ROC Curve
        if y_prob is not None:
            st.write("### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)

        # Save the model, scaler, and top features
        joblib.dump(model, 'model/best_attrition_model.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        joblib.dump(top_features, 'model/top_features.pkl')
        st.success("Model, scaler, and top features saved successfully!")