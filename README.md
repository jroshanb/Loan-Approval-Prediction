# 🏦 Loan Approval Prediction App

This is a Streamlit web application that predicts whether a loan will be approved based on user input. It uses multiple machine learning models to provide accurate predictions.

---

## 📌 Features

- Predict loan approval using:
  - Decision Tree
  - Logistic Regression
  - Random Forest
- Simple and interactive user interface
- Displays prediction result and model score

---

## 📥 Input Parameters

The following inputs are required:

- Annual Income
- Loan Amount
- Debt to Income Ratio
- Total Assets
- Net Worth
- Monthly Loan Payment
- Risk Score

---

## 🧠 Models and Accuracy

- **Decision Tree**: 98.77%
- **Logistic Regression**: 92.13%
- **Random Forest**: 99.51%

---

## 📁 Files in the Repo

- `Loan.py` – Streamlit app code
- `decision_tree_model.pkl`
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `requirements.txt`
- `README.md`

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/jroshanb/Loan-Approval-Prediction.git
   cd Loan-Approval-Prediction
   pip install -r requirements.txt
   streamlit run Loan.py

Make sure that the model files (decision_tree_model.pkl, logistic_regression_model.pkl, random_forest_model.pkl) are in the same directory as Loan.py.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
