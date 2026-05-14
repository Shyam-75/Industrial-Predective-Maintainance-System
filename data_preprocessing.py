import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess(data_path='D:\\projects\\update older projects\\Project\\Predective maintenance\\data\\raw\\ai4i2020.csv'):
    df = pd.read_csv(data_path)
    
    # Feature Engineering
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
    df['wear_per_power'] = df['Tool wear [min]'] / (df['power'] + 1)
    df['log_tool_wear'] = np.log1p(df['Tool wear [min]'])
    
    # Drop useless columns
    drop_cols = ['UDI', 'Product ID']
    df = df.drop(drop_cols, axis=1)
    
    # Target and features
    X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Column transformer
    numeric_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                       'Torque [Nm]', 'Tool wear [min]', 'temp_diff', 'power', 
                       'wear_per_power', 'log_tool_wear']
    
    categorical_features = ['Type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess()
    print("Preprocessing pipeline ready!")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")