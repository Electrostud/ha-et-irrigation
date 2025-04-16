import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ET_model import ETModel  # Assuming ETModel is the class defined in ET_model.py
import joblib

def load_data(data_path):
    # Load the processed data
    data = pd.read_csv(data_path)
    return data

def train_model(model, X_train, y_train):
    # Train the model
    model.fit(X_train, y_train)

def save_model(model, model_path):
    # Save the trained model
    joblib.dump(model, model_path)

def main():
    # Define paths
    data_path = os.path.join('data', 'processed', 'processed_data.csv')  # Adjust filename as necessary
    model_path = os.path.join('models', 'et_model.pkl')  # Ensure 'models' directory exists

    # Load data
    data = load_data(data_path)
    X = data.drop('target', axis=1)  # Adjust 'target' to your actual target column name
    y = data['target']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = ETModel()  # Adjust initialization as necessary

    # Train the model
    train_model(model, X_train, y_train)

    # Save the trained model
    save_model(model, model_path)

if __name__ == "__main__":
    main()