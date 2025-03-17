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
git clone https://github.com/your-username/credit-risk-modelling.git
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
The project uses multiple datasets to analyze customer profiles and predict credit risk:
- **Customers.csv**: Basic details of loan applicants
- **Bureau_data.csv**: Credit history from various sources
- **Loans.csv**: Information on previous and current loans

## 📦 Libraries & Packages
The project utilizes the following key libraries:
- **pandas** – Data manipulation and analysis  
- **numpy** – Numerical computations  
- **matplotlib.pyplot** – Data visualization  
- **seaborn** – Statistical data visualization  
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

## 🤖 Model Training & Evaluation
The model is trained using supervised learning techniques with a focus on classification. The training process involves:
1. **Data Preprocessing**: Handling missing values, encoding categorical features, and feature scaling.
2. **Feature Engineering**: Deriving meaningful features from raw data to improve model accuracy.
3. **Model Selection**: Training multiple models such as Logistic Regression, Random Forest, and Gradient Boosting.
4. **Hyperparameter Tuning**: Optimizing model parameters using techniques like GridSearchCV.
5. **Evaluation Metrics**:
   - **Accuracy**: Measures overall correctness of predictions.
   - **Precision & Recall**: Evaluates model performance in identifying high-risk customers.
   - **F1-Score**: Balances precision and recall for better risk assessment.
   - **ROC-AUC Score**: Assesses the trade-off between sensitivity and specificity.

## 📈 Model Performance
The best-performing model achieved:
- **Accuracy**: ~85%
- **Precision**: ~82%
- **Recall**: ~78%
- **F1-Score**: ~80%
- **ROC-AUC Score**: ~88%

The model provides a balanced assessment of credit risk, minimizing false positives and negatives effectively.

---

