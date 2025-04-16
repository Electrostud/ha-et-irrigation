def load_data(file_path):
    # Function to load data from a specified file path
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Function to clean the dataset
    # Example: Remove duplicates and handle missing values
    data = data.drop_duplicates()
    data = data.dropna()
    return data

def transform_data(data):
    # Function to transform the dataset
    # Example: Feature scaling or encoding categorical variables
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    # Function to split the dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test