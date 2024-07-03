import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Define global variables for feature and target columns
FEATURE_COLUMNS = ['Income Sources', 'Monthly Income', 'Monthly Expenses', 'Prior Investment Amount', 
                   'Risk Tolerance', 'Investment Goals', 'Time Horizon', 'Age', 'Debt', 'Savings']
TARGET_COLUMNS = ['Suggested Investment Amount', 'Asset Allocation', 'Recommended Investment Products', 
                  'Expected Returns', 'Risk Level']

# Define the mapping for "Recommended Investment Products"
investment_products_mapping = {
    0: 'Growth Stocks',
    1: 'High-yield Bonds',
    2: 'Index Funds',
    3: 'Real Estate',
    4: 'Mutual Funds'
}

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def generate_random_targets(num_samples):
    """Generate random target data for demonstration purposes."""
    return pd.DataFrame({
        'Suggested Investment Amount': np.random.randint(500, 5000, size=num_samples),
        'Asset Allocation': np.random.randint(0, 3, size=num_samples),
        'Recommended Investment Products': np.random.randint(0, 5, size=num_samples),
        'Expected Returns': np.random.uniform(5.0, 15.0, size=num_samples),
        'Risk Level': np.random.randint(0, 3, size=num_samples)
    })

def preprocess_and_train_model(X, y):
    """Preprocess the data and train the model."""
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
    predicted_output = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)
    print(predicted_output.head())
