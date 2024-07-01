import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the data from CSV file
data = pd.read_csv('investment_data.csv')

# Define the feature columns and target columns
feature_columns = ['Income Sources', 'Monthly Income', 'Monthly Expenses', 'Prior Investment Amount', 
                   'Risk Tolerance', 'Investment Goals', 'Time Horizon', 'Age', 'Debt', 'Savings']
target_columns = ['Suggested Investment Amount', 'Asset Allocation', 'Recommended Investment Products', 
                  'Expected Returns', 'Risk Level']

# Separate features and targets
X = data[feature_columns]
# Generate random target data for demonstration purposes
y = pd.DataFrame({
    'Suggested Investment Amount': np.random.randint(500, 5000, size=1000),
    'Asset Allocation': np.random.randint(0, 3, size=1000),
    'Recommended Investment Products': np.random.randint(0, 4, size=1000),
    'Expected Returns': np.random.uniform(5.0, 15.0, size=1000),
    'Risk Level': np.random.randint(0, 3, size=1000)
})

# Define the preprocessing for categorical features
categorical_features = ['Income Sources', 'Risk Tolerance', 'Investment Goals', 'Time Horizon']
categorical_transformer = OneHotEncoder()

# Define the preprocessing for numerical features
numerical_features = ['Monthly Income', 'Monthly Expenses', 'Prior Investment Amount', 'Age', 'Debt', 'Savings']
numerical_transformer = StandardScaler()

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create and combine preprocessing and modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, 'investment_advisor_model.pkl')

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f"Mean Squared Error: {mse}")

# Output the predicted values
predicted_output = pd.DataFrame(y_pred, columns=target_columns)
print(predicted_output.head())