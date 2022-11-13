"""Script for training a simple linear model only features present in the row (no rolling features)."""
import json
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import requests

FOOTY_API_BASE_URL = "https://api.footy-tracker.live"
MODEL_TRAINING_OUT_DIR = "model_training_artifacts"
FEATURES = [
    'team1_defender_defensive_rating_before_game',
    'team1_attacker_offensive_rating_before_game',
    'team2_defender_defensive_rating_before_game',
    'team2_attacker_offensive_rating_before_game'
]
TARGET = 'goal_diff'


def get_footy_training_df() -> pd.DataFrame:
    """Get training data for footy model from """
    response = requests.get(f"{FOOTY_API_BASE_URL}/ml/training_data/json")
    return pd.DataFrame(response.json()["data"])


if __name__ == '__main__':
    print("Load data")
    df = get_footy_training_df()

    print("Initialize model")
    footy_model = LinearRegression()

    print("Calculate and save cross validation scores for the model")
    cv_scores_fm = cross_val_score(footy_model, df, df[TARGET], cv=5, scoring='neg_mean_squared_error')
    metrics = {"Cross validation Mean Absolute Error": np.sqrt(-cv_scores_fm.mean())}
    with open(f"{MODEL_TRAINING_OUT_DIR}/metrics.json", "w") as f:
       json.dump(metrics, f)

    print("Fit model on entire data set")
    footy_model.fit(df, df.goal_diff)

    print("Save model")
    trained_model_dict = footy_model.to_minimal_representation()
    with open(f"{MODEL_TRAINING_OUT_DIR}/model.pickle", "wb") as f:
        pickle.dump(trained_model_dict, f)

