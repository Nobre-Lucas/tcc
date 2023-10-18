import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def set_initial_params(model: RandomForestRegressor, X_train, y_train):
    model.fit(X_train, y_train)

def set_model_params(
    model: RandomForestRegressor, params: list
) -> RandomForestRegressor:
    model.estimators_ = params
    return model

def get_model_parameters(model: RandomForestRegressor):
    params = model.estimators_
    return params

def load_house():
    path = "data/energydata_complete.csv"
    energy_data_complete = pd.read_csv(path)
    columns_for_training = []

    temperature_columns = [f"T{i}" for i in range(1, 10)]
    humidity_columns = [f"RH_{i}" for i in range(1, 10)]

    for temperature in temperature_columns:
        columns_for_training.append(temperature)
        
    for humidity in humidity_columns:
        columns_for_training.append(humidity)
        
    columns_for_training.append("T_out")
    columns_for_training.append("RH_out")
    columns_for_training.append("Press_mm_hg")
    columns_for_training.append("Visibility")

    X = energy_data_complete[columns_for_training]
    y = energy_data_complete["Appliances"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return (X_train, y_train), (X_test, y_test)

def load_house_server_side():
    (X_train, y_train), (_, _)  = load_house()[:3]
    return X_train, y_train

def shuffle(X: np.ndarray, y: np.ndarray):
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )