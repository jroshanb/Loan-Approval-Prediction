import streamlit as st
import pickle
import pandas as pd

# Load the models
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_regression_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)
    
# You can predefine model scores based on training or validation
decision_tree_score = 0.9877 # Placeholder: Replace with actual R^2 score or CV score
logistic_regression_model_score = 0.92133   # Placeholder: Replace with actual R^2 score or CV score
random_forest_model_score = 0.9951     # Placeholder: Replace with actual R^2 score or CV score

# Define the feature names
feature_names = [
    'AnnualIncome', 'LoanAmount', 'DebtToIncomeRatio', 'TotalAssets', 
    'NetWorth','MonthlyLoanPayment', 'RiskScore'
]

# Streamlit app
st.title('Loan Approval Prediction for Banks')

# Input fields for the features
def user_input_features():
    AnnualIncome = st.number_input('Annual Income', min_value=0.0, value=50000.0)
    LoanAmount = st.number_input('Loan Amount', min_value=0.0, value=200000.0)
    DebtToIncomeRatio = st.number_input('Debt to Income Ratio', min_value=0.0, value=30.0)
    TotalAssets = st.number_input('Total Assets', min_value=0.0, value=500000.0)
    NetWorth = st.number_input('Net Worth', min_value=0.0, value=100000.0)
    MonthlyLoanPayment = st.number_input('Monthly Loan Payment', min_value=0.0, value=1000.0)
    RiskScore = st.number_input('Risk Score', min_value=0.0, max_value=100.0, value=60.0)
    
    data = {
        'AnnualIncome': AnnualIncome,
        'LoanAmount': LoanAmount,
        'DebtToIncomeRatio': DebtToIncomeRatio,
        'TotalAssets': TotalAssets,
        'NetWorth': NetWorth,
        'MonthlyLoanPayment': MonthlyLoanPayment,
        'RiskScore': RiskScore
    }
    
    return pd.DataFrame(data, index=[0])

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Dropdown for model selection
algorithm = st.selectbox(
    'Choose an algorithm for prediction',
    ('Decision Tree', 'Logistic Regression', 'Random Forest')
)

# Prediction function
def predict_loan_approval(model, df):
    # Convert DataFrame to NumPy array to ensure compatibility with scikit-learn models
    input_data = df[feature_names].to_numpy()

    # Make sure that the input array shape matches the model's expected input shape
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        return 'Loan Approved'
    else:
        return 'Loan Not Approved'

# Prediction button
if st.button('Predict'):
    if algorithm == 'Decision Tree':
        result = predict_loan_approval(decision_tree_model, df)
        st.subheader('Decision Tree Prediction')
        st.write(result)
        score = decision_tree_score
    elif algorithm == 'Logistic Regression':
        result = predict_loan_approval(logistic_regression_model, df)
        st.subheader('Logistic Regression Prediction')
        st.write(result)
        score = logistic_regression_model_score
    elif algorithm == 'Random Forest':
        result = predict_loan_approval(random_forest_model, df)
        st.subheader('Random Forest Prediction')
        st.write(result)
        score =random_forest_model_score