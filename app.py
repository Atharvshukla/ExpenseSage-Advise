import streamlit as st
import pandas as pd
import joblib
import investment_model

# Load the trained model
model = joblib.load('investment_advisor_model.pkl')

# Define the mapping for "Recommended Investment Products"
investment_products_mapping = investment_model.investment_products_mapping

def map_investment_products(predictions):
    """Map numerical values to investment product categories."""
    predictions['Recommended Investment Products'] = predictions['Recommended Investment Products'].round().astype(int).map(investment_products_mapping)
    return predictions

def make_prediction(new_data):
    """Make predictions on new data using the trained model."""
    # Convert new data to DataFrame
    new_data_df = pd.DataFrame(new_data)

    # Make predictions
    predictions = model.predict(new_data_df)

    # Convert predictions to DataFrame
    prediction_df = pd.DataFrame(predictions, columns=investment_model.TARGET_COLUMNS)

    # Map numerical values to investment product categories
    prediction_df = map_investment_products(prediction_df)

    return prediction_df

# Streamlit app
st.title('Investment Advisor')

st.header('Input your financial details')

# Input fields for user data
income_sources = st.selectbox('Income Sources', ['Salary', 'Business', 'Investments', 'Others'])
monthly_income = st.number_input('Monthly Income', min_value=0)
monthly_expenses = st.number_input('Monthly Expenses', min_value=0)
prior_investment_amount = st.number_input('Prior Investment Amount', min_value=0)
risk_tolerance = st.selectbox('Risk Tolerance', ['Low', 'Medium', 'High'])
investment_goals = st.selectbox('Investment Goals', ['Wealth Building', 'Retirement', 'Education', 'Others'])
time_horizon = st.selectbox('Time Horizon', ['Short-term', 'Medium-term', 'Long-term'])
age = st.number_input('Age', min_value=0)
debt = st.number_input('Debt', min_value=0)
savings = st.number_input('Savings', min_value=0)

# Create a dictionary from the input data
new_data = {
    'Income Sources': [income_sources],
    'Monthly Income': [monthly_income],
    'Monthly Expenses': [monthly_expenses],
    'Prior Investment Amount': [prior_investment_amount],
    'Risk Tolerance': [risk_tolerance],
    'Investment Goals': [investment_goals],
    'Time Horizon': [time_horizon],
    'Age': [age],
    'Debt': [debt],
    'Savings': [savings]
}

# When the user clicks the button, make the prediction
if st.button('Predict'):
    prediction = make_prediction(new_data)
    st.header('Prediction')
    st.write(prediction)
