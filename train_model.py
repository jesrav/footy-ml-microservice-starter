"""Script for training a simple linear model using a rolling aggregate features."""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import requests

FOOTY_API_BASE_URL = "https://api.footy-tracker.live"
TARGET = 'goal_diff'
MODEL_TRAINING_OUT_DIR = "api/model_training_artifacts"


def get_footy_training_df() -> pd.DataFrame:
    """Get training data for footy model from """
    response = requests.get(f"{FOOTY_API_BASE_URL}/ml/training_data/json")
    return pd.DataFrame(response.json()["data"])



def train_model(df: pd.DataFrame) -> LinearRegression:
    """Train a linear model using a rolling aggregate features."""
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")
    return model

if __name__ == '__main__':
    print("Load data")
    df = get_footy_training_df()

    print("Initialize model")
    footy_model = UserStrengthModel()

    print("Calculate and save cross validation scores for the model")
    cv_scores_fm = cross_val_score(footy_model, df, df[TARGET], cv=5, scoring='neg_mean_squared_error')
    metrics = {"mae_cv": np.sqrt(-cv_scores_fm.mean())}
    with open(f"{MODEL_TRAINING_OUT_DIR}/metrics.json", "w") as f:
       json.dump(metrics, f)

    print("Fit model on entire data set")
    footy_model.fit(df, df.goal_diff)

    print("Save model")
    trained_model_dict = footy_model.to_minimal_representation()
    with open(f"{MODEL_TRAINING_OUT_DIR}/model.pickle", "wb") as f:
        pickle.dump(trained_model_dict, f)

    print("Plot offensive and defensive strength parameters of users")
    save_user_parameter_plot(footy_model, "defensive_strength", MODEL_TRAINING_OUT_DIR)
    save_user_parameter_plot(footy_model, "attack_strength", MODEL_TRAINING_OUT_DIR)
