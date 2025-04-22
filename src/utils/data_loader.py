"""
Data loading and preprocessing utilities for the HR Analytics application.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from src.utils import config
import warnings
import streamlit as st

@st.cache_data
def load_dataset():
    """
    Load the employee attrition dataset with caching for performance.
    
    Returns:
        pd.DataFrame: The loaded dataset or None if loading fails
    """
    try:
        # Check if dataset exists
        if not os.path.exists(config.DATASET_PATH):
            st.info("Dataset not found. Attempting to locate dataset...")
            
            # Check if there's a dataset in the parent directory
            parent_data_path = os.path.join(os.path.dirname(os.path.dirname(config.BASE_DIR)), 
                                          "Employee-Attrition-Prediction", "data", 
                                          "employee_attrition.csv")
            
            if os.path.exists(parent_data_path):
                st.info(f"Found dataset at {parent_data_path}. Copying to application data directory...")
                # Copy the dataset to our data directory
                df = pd.read_csv(parent_data_path)
                os.makedirs(os.path.dirname(config.DATASET_PATH), exist_ok=True)
                df.to_csv(config.DATASET_PATH, index=False)
                st.success("Dataset copied successfully!")
            else:
                # Try the default dataset location as a last resort
                try:
                    sample_data = {
                        'EmployeeNumber': list(range(1, 21)),
                        'FirstName': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry',
                                    'Ivy', 'Jack', 'Kelly', 'Leo', 'Mary', 'Nathan', 'Olivia', 'Paul', 'Quinn', 'Rachel'],
                        'LastName': ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia', 'Rodriguez', 'Wilson',
                                   'Martinez', 'Anderson', 'Taylor', 'Thomas', 'Hernandez', 'Moore', 'Martin', 'Jackson', 'Thompson', 'White'],
                        'Department': ['Sales', 'HR', 'IT', 'Marketing', 'R&D', 'Sales', 'HR', 'IT', 'Marketing', 'R&D',
                                     'Sales', 'HR', 'IT', 'Marketing', 'R&D', 'Sales', 'HR', 'IT', 'Marketing', 'R&D'],
                        'JobRole': ['Sales Rep', 'HR Manager', 'Developer', 'Marketing Specialist', 'Research Scientist',
                                  'Sales Manager', 'HR Specialist', 'IT Manager', 'Marketing Manager', 'Research Director',
                                  'Sales Executive', 'HR Director', 'Senior Developer', 'Content Manager', 'Lead Scientist',
                                  'Account Manager', 'Recruiter', 'System Admin', 'Digital Marketer', 'Data Scientist'],
                        'Attrition': ['No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No',
                                    'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No'],
                        'Age': [35, 42, 29, 31, 45, 38, 27, 52, 33, 48, 
                              30, 44, 39, 35, 28, 41, 36, 32, 37, 43],
                        'YearsAtCompany': [5, 10, 2, 3, 12, 7, 1, 15, 4, 11,
                                         2, 9, 6, 4, 1, 8, 5, 3, 5, 10],
                        'JobSatisfaction': [3, 4, 2, 3, 4, 3, 2, 4, 3, 4,
                                          2, 3, 4, 3, 2, 3, 4, 2, 3, 4],
                        'WorkLifeBalance': [3, 4, 2, 3, 3, 4, 2, 3, 4, 3,
                                          2, 3, 4, 3, 2, 3, 4, 2, 3, 4],
                        'PerformanceRating': [3, 4, 3, 3, 4, 3, 2, 4, 3, 4,
                                            3, 3, 4, 3, 2, 4, 3, 3, 3, 4]
                    }
                    
                    sample_df = pd.DataFrame(sample_data)
                    os.makedirs(os.path.dirname(config.DATASET_PATH), exist_ok=True)
                    sample_df.to_csv(config.DATASET_PATH, index=False)
                    st.warning("Created sample dataset for demonstration purposes.")
                    return sample_df
                except Exception as e:
                    st.error(f"Failed to create sample dataset: {str(e)}")
                    return None
        
        # Load the dataset
        df = pd.read_csv(config.DATASET_PATH)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the dataset for training.
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        tuple: (X, X_scaled, y, scaler, categorical_cols)
    """
    try:
        if df is None or config.TARGET_COLUMN not in df.columns:
            return None, None, None, None, None
        
        # Create a copy of the dataframe
        data = df.copy()
        
        # Convert target to numeric (assuming 'Yes'/'No' values)
        if data[config.TARGET_COLUMN].dtype == 'object':
            le = LabelEncoder()
            data[config.TARGET_COLUMN] = le.fit_transform(data[config.TARGET_COLUMN])
            
            # Map class names (0: No, 1: Yes)
            target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Separate features and target
        X = data.drop(config.TARGET_COLUMN, axis=1)
        y = data[config.TARGET_COLUMN]
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X, X_scaled, y, scaler, categorical_cols
    
    except Exception as e:
        warnings.warn(f"Error preprocessing data: {str(e)}")
        return None, None, None, None, None

def get_feature_importance(model, features):
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model
        features (list): List of features used in the model
        
    Returns:
        pandas.DataFrame: DataFrame with feature names and importance scores
    """
    try:
        # Try to extract feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (Random Forest, XGBoost, etc.)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models (LogisticRegression, etc.)
            importances = np.abs(model.coef_[0])
        else:
            st.warning("Feature importance not available for this model type")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        st.error(f"Error extracting feature importance: {str(e)}")
        return None

def safe_preprocess_for_prediction(df, features, scaler, silent=False):
    """
    A more robust version of preprocess_for_prediction that handles missing features
    and other common preprocessing errors.
    
    Args:
        df (pandas.DataFrame): Input data
        features (list): List of features used in the model
        scaler (StandardScaler): Fitted scaler
        silent (bool): Whether to hide informational messages
        
    Returns:
        numpy.ndarray: Preprocessed data ready for prediction
    """
    try:
        # Create a copy of the dataframe
        data = df.copy()
        
        # Handle target column if present
        if config.TARGET_COLUMN in data.columns:
            data = data.drop(columns=[config.TARGET_COLUMN])
        
        # Split features into base columns and one-hot encoded features
        base_columns = []
        encoded_prefixes = []
        
        for feature in features:
            # Check if it's a one-hot encoded feature
            if '_' in feature:
                prefix = feature.split('_')[0]
                if prefix not in encoded_prefixes:
                    encoded_prefixes.append(prefix)
            else:
                # Regular numerical column
                base_columns.append(feature)
        
        # Extract categorical and numerical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Add default values for missing numerical columns
        for col in base_columns:
            if col not in data.columns and col not in categorical_cols:
                data[col] = 0
                if not silent:
                    st.info(f"Added missing numerical column: {col}")
        
        # Handle categorical variables
        for prefix in encoded_prefixes:
            if prefix in categorical_cols:
                # Column is already in the data, need to one-hot encode it
                pass
            elif prefix in numerical_cols:
                # This might be a numerical column that needs binning
                if not silent:
                    st.warning(f"Column {prefix} is numerical but expected to be categorical")
            else:
                # Column is missing, add with a default value
                data[prefix] = 'Unknown'
                if not silent:
                    st.info(f"Added missing categorical column: {prefix}")
        
        # One-hot encode all categorical columns
        if categorical_cols:
            data = pd.get_dummies(data, columns=categorical_cols)
        
        # Create a DataFrame with all necessary features
        X = pd.DataFrame(0, index=data.index, columns=features)
        
        # Fill in values for columns that exist in the data
        for col in X.columns:
            if col in data.columns:
                X[col] = data[col]
        
        # Apply scaler
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    except Exception as e:
        if not silent:
            st.error(f"Error preprocessing data for prediction: {str(e)}")
        return None

def save_model_artifacts(model, scaler, features, metrics):
    """
    Save model artifacts to disk.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        features (list): List of features used
        metrics (dict): Model performance metrics
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directories exist
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
        
        # Save model
        joblib.dump(model, config.BEST_MODEL_PATH)
        
        # Save scaler
        joblib.dump(scaler, config.SCALER_PATH)
        
        # Save features
        with open(config.FEATURES_PATH, 'w') as f:
            json.dump(features, f)
        
        # Save metrics
        with open(config.METRICS_PATH, 'w') as f:
            json.dump(metrics, f)
        
        return True
    
    except Exception as e:
        warnings.warn(f"Error saving model artifacts: {str(e)}")
        return False

def safe_display_dataframe(data, hide_index=False, use_container_width=True, height=None):
    """
    Safely display a dataframe by ensuring all data is of compatible types for Streamlit.
    
    Args:
        data: A list of dictionaries or a pandas DataFrame
        hide_index: Whether to hide the index
        use_container_width: Whether to use the full container width
        height: Optional height for the dataframe display
    
    Returns:
        None - displays the dataframe in the Streamlit app
    """
    try:
        # Convert to DataFrame if it's a list
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Handle empty dataframe
        if df.empty:
            st.write("No data to display")
            return
        
        # Convert all columns to strings to avoid Arrow conversion issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Display the dataframe
        kwargs = {'use_container_width': use_container_width}
        if hide_index:
            kwargs['hide_index'] = hide_index
        if height:
            kwargs['height'] = height
            
        st.dataframe(df, **kwargs)
    
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")
        st.write("Raw data:")
        st.write(data) 