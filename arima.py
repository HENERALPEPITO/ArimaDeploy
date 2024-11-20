import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib  
import warnings
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score



def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    data['Time'] = pd.to_datetime(data['Time'] + '-1', format='%Y-w%W-%w')
    
    # Extract features and target
    features = data[['Rainfall', 'Temperature', 'Humidity']]
    target = data['Cases']

    # Scale features and target
    scaler_features = joblib.load('scaler_features.pkl') if os.path.exists('scaler_features.pkl') else MinMaxScaler()
    scaler_target = joblib.load('scaler_target.pkl') if os.path.exists('scaler_target.pkl') else MinMaxScaler()

    normalized_features = scaler_features.fit_transform(features)
    normalized_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

    # Combine normalized data into one DataFrame
    normalized_data = pd.DataFrame(normalized_features, columns=features.columns, index=data.index)
    normalized_data['Cases'] = normalized_target

    return normalized_data, scaler_target, scaler_features


def find_best_arima_order(data, exog):
    warnings.filterwarnings("ignore")

    p_values = range(0, 4)
    d_values = range(0, 3)
    q_values = range(0, 4)
    best_score, best_cfg = float("inf"), None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(data, exog=exog, order=(p, d, q))
            model_fit = model.fit()
            mse = mean_squared_error(data, model_fit.fittedvalues)
            if mse < best_score:
                best_score, best_cfg = mse, (p, d, q)
        except:
            continue
    return best_cfg


def train_and_save_arima(data, exog, file_name, scaler_target, scaler_features):
    best_order = find_best_arima_order(data, exog)
    model = ARIMA(data, exog=exog, order=best_order)
    model_fit = model.fit()

    # Save the model
    joblib.dump(model_fit, file_name)

    # Save the scalers
    joblib.dump(scaler_target, 'scaler_target.pkl')
    joblib.dump(scaler_features, 'scaler_features.pkl')

    print(f"Model saved as {file_name} with order {best_order}")


if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'completefinaldatasets.csv'
    normalized_data, scaler_target, scaler_features = load_and_preprocess_data(file_path)

    # Split data
    train_size = int(len(normalized_data) * 0.8)
    train, test = normalized_data.iloc[:train_size], normalized_data.iloc[train_size:]

    y_train = train['Cases']
    exog_train = train[['Rainfall', 'Temperature', 'Humidity']]

    # Train and save ARIMA model
    train_and_save_arima(y_train, exog_train, 'arima_model.pkl', scaler_target, scaler_features)
