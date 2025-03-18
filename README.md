# Credit Risk Modelling

[![Live App](https://img.shields.io/badge/Live_App-Click_Here-blue)](https://creditrisk-modelling.streamlit.app/)

## 📌 Overview
Credit risk modeling is essential for financial institutions to assess a borrower's likelihood of defaulting on a loan. This project provides a data-driven approach to credit risk assessment using machine learning techniques.

## 🚀 Live Demo
Experience the live application here: [Credit Risk Modelling](https://creditrisk-modelling.streamlit.app/)

## 📂 Project Structure
```
credit-risk-modelling/
│-- credit-risk-modelling.ipynb  # Jupyter Notebook with analysis & model training
│-- artifacts/
│   ├── model_data.joblib        # Trained model artifact
│-- data/
│   ├── bureau_data.csv          # Bureau data
│   ├── customers.csv            # Customer details
│   ├── loans.csv                # Loan records
│-- app/
│   ├── main.py                  # Streamlit application
│   ├── prediction_helper.py      # Helper functions for prediction
│   ├── requirements.txt          # Dependencies list
```

## ⚡ Features
- Exploratory Data Analysis (EDA) for risk factors
- Machine Learning model training and evaluation
- API for making risk predictions
- Interactive web-based application with Streamlit

## 🛠 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rajat-kumar-mondal/credit-risk-modelling.git
cd credit-risk-modelling
```
### 2️⃣ Install Dependencies
```bash
pip install -r app/requirements.txt
```
### 3️⃣ Run the Application
```bash
streamlit run app/main.py
```

## 📊 Dataset
The project uses multiple datasets, each containing 50K records, to analyze customer profiles and predict credit risk:
- **customers.csv**: Basic details of loan applicants
- **bureau_data.csv**: Credit history from various sources
- **loans.csv**: Information on previous and current loans

## 📦 Libraries & Packages
The project utilizes the following key libraries:
- **pandas** – Data manipulation and analysis  
- **numpy** – Numerical computations  
- **matplotlib.pyplot & seaborn** – Data visualization  
- **optuna** – Hyperparameter optimization  
- **scikit-learn**:
  - `train_test_split` – Splitting data into training and testing sets  
  - `MinMaxScaler` – Feature scaling  
  - `LogisticRegression` – Classification model  
  - `classification_report`, `f1_score`, `roc_curve`, `auc` – Model evaluation metrics  
  - `RandomForestClassifier` – Ensemble learning model  
  - `RandomizedSearchCV` – Hyperparameter tuning  
- **statsmodels**:
  - `variance_inflation_factor` – Detecting multicollinearity  
- **xgboost** – Gradient boosting model  
- **imblearn**:
  - `RandomUnderSampler` – Handling imbalanced datasets  
  - `SMOTETomek` – Hybrid oversampling and undersampling method
- **joblib**:
  - `dump` – Store the models and scaler objects
- **streamlit** - App user interface (UI)

## 🤖 Model Training & Evaluation
The model is trained using supervised learning techniques with a focus on classification. The training process involves:
### 1️⃣ Data Preprocessing
- Handling missing values through imputation strategies.
- Encoding categorical variables using one-hot encoding.
- Feature scaling with **MinMaxScaler**.
- Removing multicollinearity using **Variance Inflation Factor (VIF)**.

### 2️⃣ Class Balancing
- Addressing dataset imbalance using **SMOTETomek** and **RandomUnderSampler**.

### 3️⃣ Model Selection
Three machine learning models were trained and compared:
- **Logistic Regression**: A baseline model for classification.
- **Random Forest Classifier**: An ensemble method improving prediction stability.
- **XGBoost Classifier**: A gradient boosting method providing high accuracy.

### 4️⃣ Hyperparameter Tuning
- **Optuna** was used for optimizing hyperparameters efficiently.
- **RandomizedSearchCV** was applied to refine model parameters.

### 5️⃣ Model Evaluation
The models were assessed using:
- **Accuracy**: Overall correctness of predictions.
- **Precision & Recall**: Trade-off between false positives and false negatives.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC Score**: Measures model discrimination ability.
- **Cross-validation**: Ensured model robustness across different data splits.

## 📈 Model Performance
The best-performing model (XGBoost) achieved:
- **Accuracy**: 93%
- **Precision**: 78%
- **Recall**: 94%
- **F1-Score**: 83%
- **ROC-AUC Score**: 98%

The model provides a balanced assessment of credit risk, minimizing false positives and negatives effectively.

---

