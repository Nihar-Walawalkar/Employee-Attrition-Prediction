"""
Model Training page for training and evaluating machine learning models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import time
import os
from src.utils import data_loader, config

def show():
    """Display the model training page."""
    
    st.markdown("<h1 style='text-align: center; color: #4ecdc4;'>Model Training</h1>", unsafe_allow_html=True)
    st.write("Train and evaluate machine learning models for predicting employee attrition.")

    # Load the dataset
    df = data_loader.load_dataset()
    if df is None:
        st.error("Failed to load the dataset. Please check the data file.")
        st.stop()

    # Preprocess data
    X, X_scaled, y, scaler, categorical_cols = data_loader.preprocess_data(df)
    if X is None:
        st.error("Error during data preprocessing.")
        st.stop()

    # Sidebar for training options
    st.sidebar.subheader("Training Options")
    
    # Feature selection options
    feature_selection = st.sidebar.radio(
        "Feature Selection Method",
        ["All Features", "Top N Features", "Manual Selection"]
    )
    
    if feature_selection == "Top N Features":
        n_features = st.sidebar.slider("Number of top features", 5, 20, 10)
        
        # Use Random Forest for feature importance
        with st.spinner("Calculating feature importance..."):
            rf = RandomForestClassifier(random_state=config.RANDOM_STATE)
            rf.fit(X_scaled, y)
            
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            top_features = feature_importance['Feature'].head(n_features).tolist()
            
            # Update features
            X_selected = X[top_features]
            X_selected_scaled = scaler.transform(X_selected)
    elif feature_selection == "Manual Selection":
        all_features = X.columns.tolist()
        selected_features = st.sidebar.multiselect(
            "Select features to include",
            all_features,
            default=all_features[:min(10, len(all_features))]
        )
        
        if not selected_features:
            st.warning("Please select at least one feature to continue.")
            return
            
        # Update features
        X_selected = X[selected_features]
        X_selected_scaled = scaler.transform(X_selected)
        top_features = selected_features
    else:  # All Features
        X_selected = X
        X_selected_scaled = X_scaled
        top_features = X.columns.tolist()

    # Model selection
    selected_model_name = st.sidebar.selectbox(
        "Select Model to Train",
        list(config.MODEL_OPTIONS.keys())
    )
    
    # Hyperparameter tuning
    st.sidebar.subheader("Hyperparameter Tuning")
    
    # Model-specific hyperparameters
    hyperparams = {}
    
    if selected_model_name == "Logistic Regression":
        hyperparams['C'] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
        hyperparams['solver'] = st.sidebar.selectbox(
            "Solver", 
            ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
        )
        hyperparams['max_iter'] = st.sidebar.slider("Max Iterations", 100, 1000, 100, 100)
    
    elif selected_model_name == "Random Forest":
        hyperparams['n_estimators'] = st.sidebar.slider("Number of Trees", 50, 300, 100, 10)
        hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 3, 20, 10)
        hyperparams['min_samples_split'] = st.sidebar.slider("Min Samples Split", 2, 10, 2)
        hyperparams['max_features'] = st.sidebar.selectbox(
            "Max Features", 
            ['auto', 'sqrt', 'log2', None]
        )
    
    elif selected_model_name == "XGBoost":
        hyperparams['n_estimators'] = st.sidebar.slider("Number of Trees", 50, 300, 100, 10)
        hyperparams['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 3, 15, 6)
        hyperparams['gamma'] = st.sidebar.slider("Gamma", 0.0, 1.0, 0.0, 0.1)
    
    elif selected_model_name == "Support Vector Machine":
        hyperparams['C'] = st.sidebar.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
        hyperparams['kernel'] = st.sidebar.selectbox(
            "Kernel", 
            ['linear', 'rbf', 'poly', 'sigmoid']
        )
        hyperparams['probability'] = True  # For prediction probabilities
    
    elif selected_model_name == "Gradient Boosting":
        hyperparams['n_estimators'] = st.sidebar.slider("Number of Trees", 50, 300, 100, 10)
        hyperparams['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 3, 15, 3)
        hyperparams['subsample'] = st.sidebar.slider("Subsample", 0.5, 1.0, 1.0, 0.1)
    
    elif selected_model_name == "Decision Tree":
        hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 3, 20, 10)
        hyperparams['min_samples_split'] = st.sidebar.slider("Min Samples Split", 2, 10, 2)
        hyperparams['min_samples_leaf'] = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)
        hyperparams['criterion'] = st.sidebar.selectbox(
            "Criterion", 
            ['gini', 'entropy']
        )
    
    elif selected_model_name == "Neural Network":
        hyperparams['hidden_layer_sizes'] = tuple(
            st.sidebar.slider("Hidden Layer Neurons", 5, 100, 50)
            for _ in range(st.sidebar.slider("Number of Hidden Layers", 1, 3, 1))
        )
        hyperparams['activation'] = st.sidebar.selectbox(
            "Activation Function", 
            ['relu', 'tanh', 'logistic']
        )
        hyperparams['alpha'] = st.sidebar.slider("Alpha", 0.0001, 0.01, 0.0001, 0.0001)
        hyperparams['max_iter'] = st.sidebar.slider("Max Iterations", 200, 1000, 200, 100)
    
    # Add random state to all models
    hyperparams['random_state'] = config.RANDOM_STATE
    
    # Training settings
    st.sidebar.subheader("Training Settings")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, config.TEST_SIZE, 0.05)
    cv_folds = st.sidebar.slider("Cross-Validation Folds", 3, 10, config.CV_FOLDS)
    
    # Display feature importance
    if feature_selection != "All Features":
        st.subheader("Selected Features")
        
        # Create feature importance plot
        if feature_selection == "Top N Features":
            # Only show selected features
            selected_importance = feature_importance[feature_importance['Feature'].isin(top_features)]
            
            fig = px.bar(
                selected_importance, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Feature Importance",
                color='Importance',
                color_continuous_scale='Teal'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Display manually selected features
            st.write(f"You selected {len(top_features)} features:", ", ".join(top_features))
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected_scaled, y, test_size=test_size, random_state=config.RANDOM_STATE
    )
    
    # Create model instance based on selection
    model_class = globals()[config.MODEL_OPTIONS[selected_model_name]]
    model = model_class(**hyperparams)
    
    # Train the model
    if st.button("Train Model", key="train_model_button"):
        with st.spinner(f"Training {selected_model_name} model..."):
            # Start timer
            start_time = time.time()
            
            # Train the model
            model.fit(X_train, y_train)
            
            # End timer
            training_time = time.time() - start_time
            
            # Cross-validation
            cv_start_time = time.time()
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1')
            cv_time = time.time() - cv_start_time
            
            # Predict on test set
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None
            }
            
            # Display evaluation metrics
            st.subheader(f"Model Evaluation")
            
            # Display training information
            st.write(f"**Model:** {selected_model_name}")
            st.write(f"**Training Time:** {training_time:.2f} seconds")
            st.write(f"**Cross-Validation Time:** {cv_time:.2f} seconds")
            st.write(f"**Cross-Validation F1 Scores:** {', '.join([f'{score:.4f}' for score in cv_scores])}")
            st.write(f"**Mean CV F1 Score:** {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            
            # Create metrics display in a nice grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                
            with col2:
                st.metric("Precision", f"{metrics['Precision']:.4f}")
                
            with col3:
                st.metric("Recall", f"{metrics['Recall']:.4f}")
                
            col4, col5 = st.columns(2)
            
            with col4:
                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                
            with col5:
                if metrics['ROC-AUC'] is not None:
                    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Create a nicer confusion matrix with plotly
            cm_labels = ['No', 'Yes']  # Assuming 0=No, 1=Yes for attrition
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=cm_labels,
                y=cm_labels,
                text_auto=True,
                color_continuous_scale='Teal'
            )
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification Report
            st.subheader("Classification Report")
            
            # Get classification report as dictionary
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Extract class metrics
            class_metrics = pd.DataFrame({
                'Class': ['No (0)', 'Yes (1)', 'Avg/Total'],
                'Precision': [report['0']['precision'], report['1']['precision'], report['macro avg']['precision']],
                'Recall': [report['0']['recall'], report['1']['recall'], report['macro avg']['recall']],
                'F1-Score': [report['0']['f1-score'], report['1']['f1-score'], report['macro avg']['f1-score']],
                'Support': [report['0']['support'], report['1']['support'], report['macro avg']['support']]
            })
            
            st.dataframe(class_metrics.set_index('Class').style.highlight_max(axis=0, subset=['Precision', 'Recall', 'F1-Score']))
            
            # ROC Curve
            if y_prob is not None:
                st.subheader("ROC Curve")
                
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {metrics["ROC-AUC"]:.4f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
                
                fig.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    legend=dict(x=0.1, y=0.9)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (if applicable)
            feature_importance = data_loader.get_feature_importance(model, top_features)
            if feature_importance is not None:
                st.subheader("Feature Importance")
                
                # Create feature importance plot
                top_n = min(len(top_features), 15)  # Show at most 15 features
                fig = px.bar(
                    feature_importance.head(top_n), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title=f"Top {top_n} Feature Importance",
                    color='Importance',
                    color_continuous_scale='Teal'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Save model and artifacts
            save_model = st.checkbox("Save this model for predictions", value=True)
            
            if save_model:
                # Prepare metrics for saving
                save_metrics = {k: float(v) if v is not None else None for k, v in metrics.items()}
                save_metrics['CV_F1_Mean'] = float(np.mean(cv_scores))
                save_metrics['CV_F1_Std'] = float(np.std(cv_scores))
                save_metrics['Training_Time'] = float(training_time)
                save_metrics['CV_Time'] = float(cv_time)
                save_metrics['Model'] = selected_model_name
                save_metrics['Features_Count'] = len(top_features)
                save_metrics['Test_Size'] = test_size
                save_metrics['CV_Folds'] = cv_folds
                save_metrics['Hyperparameters'] = {k: str(v) for k, v in hyperparams.items()}
                
                # Save model artifacts
                data_loader.save_model_artifacts(model, scaler, top_features, save_metrics)
                
                st.success(f"✅ Model, scaler, and top features saved successfully!")
                
                # Display model info
                st.info(f"Model saved to: {config.BEST_MODEL_PATH}")
                st.info(f"Number of features: {len(top_features)}")
                
                # Add download buttons for the model artifacts
                model_path = config.BEST_MODEL_PATH
                scaler_path = config.SCALER_PATH
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        model_bytes = f.read()
                    
                    with open(scaler_path, 'rb') as f:
                        scaler_bytes = f.read()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="Download Model",
                            data=model_bytes,
                            file_name="best_attrition_model.pkl",
                            mime="application/octet-stream",
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Scaler",
                            data=scaler_bytes,
                            file_name="scaler.pkl",
                            mime="application/octet-stream",
                        ) 