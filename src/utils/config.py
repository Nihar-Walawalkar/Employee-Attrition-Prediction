"""
Configuration parameters for the application.
"""

import os
from pathlib import Path
from datetime import datetime

# Project structure
PROJECT_ROOT = BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).resolve()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Data files
DATA_FILE = os.path.join(DATA_DIR, "HR-Employee-Attrition.csv")
DATASET_PATH = os.path.join(DATA_DIR, "employee_attrition.csv")

# Model related settings
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
TOP_FEATURES_PATH = os.path.join(MODEL_DIR, "top_features.pkl")
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# Random state for reproducibility
RANDOM_STATE = 42

# Train/test split ratio
TEST_SIZE = 0.25

# Cross-validation settings
CV_FOLDS = 5

# Model training settings
MODEL_OPTIONS = {
    "Logistic Regression": "LogisticRegression",
    "Random Forest": "RandomForestClassifier",
    "XGBoost": "XGBClassifier",
    "Gradient Boosting": "GradientBoostingClassifier",
    "Decision Tree": "DecisionTreeClassifier",
    "Support Vector Machine": "SVC",
    "Neural Network": "MLPClassifier"
}

# Application metadata
APP_NAME = "HR Analytics - Employee Attrition Prediction"
APP_VERSION = "2.0.0"
BUILD_NUMBER = "2023.1"
BUILD_DATE = datetime.now().strftime("%Y-%m-%d")
DATA_UPDATE_DATE = "April 2025"
THEME_COLOR = "#4ecdc4"
SECONDARY_COLOR = "#ff6f61"
SUCCESS_COLOR = "#4ecdc4"
WARNING_COLOR = "#ff6f61"
INFO_COLOR = "#3498db"

# Default target column
TARGET_COLUMN = "Attrition"

# Model parameters
TOP_FEATURES_COUNT = 10

# Evaluation metrics to display
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Data preprocessing
CATEGORICAL_THRESHOLD = 10  # Columns with fewer unique values will be treated as categorical
DATE_COLUMNS = []  # List of date columns that need special handling
DROP_COLUMNS = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']

# Cache settings
CACHE_TTL = 3600  # Time to live for cached data in seconds 