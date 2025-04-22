# **EMPLOYEE ATTRITION PREDICTION**

An advanced HR analytics web application for predicting employee attrition with interactive visualizations and actionable insights.

![HR Analytics Dashboard](https://img.shields.io/badge/HR%20Analytics-Dashboard-4ecdc4)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Machine Learning](https://img.shields.io/badge/ML-Prediction-yellow)

## Overview

The Enhanced Employee Attrition Prediction tool helps HR professionals and managers identify employees at risk of leaving the organization and provides data-driven insights to improve retention strategies. The application uses machine learning to analyze various factors that contribute to employee attrition and offers actionable recommendations.

## Features

- **Interactive Data Exploration**: Visualize and analyze employee data with dynamic charts and filters.
- **Advanced ML Model Training**: Train and evaluate multiple machine learning models with customizable parameters.
- **Individual Prediction**: Assess attrition risk for individual employees with detailed explanations.
- **Batch Prediction**: Upload a CSV file to get attrition predictions for multiple employees at once.
- **Actionable Insights**: Get data-driven recommendations to reduce employee turnover.
- **SHAP Explanations**: Understand which factors most influence attrition risk predictions.

## Installation

### Prerequisites

- Python 3.9 or higher
- Pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Nihar-Walawalkar/Employee-Attrition-Prediction.git
   cd Employee-Attrition-Prediction
   ```

2. Create a virtual environment:
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the data:
   - Place the IBM HR Analytics Employee Attrition dataset in the `data/` folder.
   - The expected filename is `HR-Employee-Attrition.csv`.
   - You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

## Usage

Run the application:

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

### Navigation

The application has the following main sections:

1. **Home**: Overview of the application and key statistics.
2. **Data Exploration**: Interactive visualizations of the dataset.
3. **Model Training**: Train and evaluate machine learning models.
4. **Predictions**: Make individual or batch predictions.
5. **Insights**: Get actionable insights and recommendations.

## Data Exploration

The Data Exploration page offers:

- Basic dataset overview and statistics
- Distribution of categorical and numerical variables
- Correlations between variables
- Key metrics related to attrition
- Custom analysis options

## Model Training

In the Model Training section, you can:

- Choose from various machine learning algorithms:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine
  - Gradient Boosting
  - Decision Tree
  - Neural Network
- Select feature selection methods:
  - All features
  - Top N features
  - Manual selection
- Customize hyperparameters for each model
- Evaluate model performance using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- View confusion matrix and ROC curve
- Save trained models for prediction

## Predictions

The Predictions page allows you to:

- **Individual Prediction**:
  - Enter employee details
  - Get attrition risk assessment with probability
  - View detailed explanation of influential factors
  - Receive specific recommendations

- **Batch Prediction**:
  - Upload a CSV file with multiple employees
  - Get predictions for all employees at once
  - Download results as CSV
  - View summary statistics and visualizations

## Insights

The Insights page provides:

- Key findings about attrition patterns
- Risk factors analysis
- Department-level analysis
- Satisfaction and performance metrics
- Model-based insights
- Actionable recommendations

## Project Structure

```
Enhanced-Employee-Attrition-Prediction/
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/                   # Data directory
│   └── HR-Employee-Attrition.csv
├── models/                 # Saved models
│   └── best_model.pkl
├── artifacts/              # Model artifacts
│   ├── scaler.pkl
│   ├── features.json
│   └── metrics.json
└── src/                    # Source code
    ├── pages/              # Application pages
    │   ├── home.py
    │   ├── data_exploration.py
    │   ├── model_training.py
    │   ├── predictions.py
    │   └── insights.py
    └── utils/              # Utility functions
        ├── config.py
        └── data_loader.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Data source: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Built with Streamlit, Scikit-learn, and Plotly
- SHAP library for model explanations 
