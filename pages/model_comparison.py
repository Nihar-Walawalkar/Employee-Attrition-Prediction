import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.markdown("<h1>Model Comparison</h1>", unsafe_allow_html=True)
    st.write("Compare different machine learning models for predicting employee attrition.")

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
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    top_features = feature_importance.sort_values('Importance', ascending=False)['Feature'].head(10).tolist()
    X_selected = X[top_features]
    X_selected_scaled = scaler.fit_transform(X_selected)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_selected_scaled, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    # Compare models
    if st.button("Compare Models"):
        results = []
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            mean_cv_score = np.mean(cv_scores)
            
            # Train and evaluate on test set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            result = {
                'Model': name,
                'Cross-Validation F1 Score': mean_cv_score,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            results.append(result)

        # Display results
        results_df = pd.DataFrame(results).drop(columns=['y_pred', 'y_prob'])
        st.subheader("Model Comparison Results")
        st.dataframe(results_df.style.highlight_max(subset=['Cross-Validation F1 Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], color='lightgreen'))

        # Highlight the best model
        best_model = results_df.loc[results_df['F1 Score'].idxmax()]['Model']
        st.write(f"**Best Model (based on F1 Score):** {best_model}")

        # Visualize multiple metrics
        st.write("### Metric Comparison")
        metrics_df = results_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], var_name='Metric', value_name='Score')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Score', y='Model', hue='Metric', data=metrics_df, ax=ax)
        ax.set_title('Metric Comparison Across Models')
        st.pyplot(fig)

        # ROC Curves
        st.write("### ROC Curves")
        fig, ax = plt.subplots(figsize=(8, 5))
        for result in results:
            if result['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
                ax.plot(fpr, tpr, label=f"{result['Model']} (AUC = {roc_auc_score(y_test, result['y_prob']):.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves for All Models')
        ax.legend()
        st.pyplot(fig)

        # Confusion Matrices
        st.write("### Confusion Matrices")
        cols = st.columns(len(models))
        for idx, result in enumerate(results):
            with cols[idx]:
                st.write(f"**{result['Model']}**")
                cm = confusion_matrix(y_test, result['y_pred'])
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)