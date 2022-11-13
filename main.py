"""
FootyTracker ML microservice example

Note, that you would normally move the logic to separate files and import them here.
We do everything in one file for simplicity.
"""
import pickle
from random import choice

from fastapi import FastAPI

from schemas import RowForML, DataForML

MODEL_TRAINING_OUT_DIR = "model_training_artifacts"

# Initialize FAST API app
app = FastAPI()

# Load trained linear regression model
with open(f"{MODEL_TRAINING_OUT_DIR}/model.pickle", "rb") as f:
    model = pickle.load(f)


#############################################################################
# Examples of rule based examples that does not use any machine learning.
#############################################################################
def jesus_rule_prediction(result_to_predict: RowForML) -> float:
    """Predict that Jesus (User id 1) always wins by 5 if he is on offence.
    Otherwise, guess random goal diff between -3 and 3, but never 0.
    """
    if result_to_predict.team1_attacker_user_id == 1:
        return 5.0
    elif result_to_predict.team2_attacker_user_id == 1:
        return -5.0
    else:
        return choice([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])


def baseline_bob_prediction(result_to_predict: RowForML) -> float:
    """Always predict a goal difference of 0"""
    return 0.0


@app.post("/jesus_rules/predict")
def predict(body: DataForML) -> float:
    result_to_predict = [r for r in body.data if r.result_to_predict][0]
    return jesus_rule_prediction(result_to_predict)


@app.post("/baseline_bob/predict")
def predict(body: DataForML) -> float:
    result_to_predict = [r for r in body.data if r.result_to_predict][0]
    return jesus_rule_prediction(result_to_predict)


#############################################################################
# Example of using a trained ML model
#############################################################################
@app.post("/baseline_bob/predict")
def predict(body: DataForML) -> float:
    result_to_predict = [r for r in body.data if r.result_to_predict][0]
    return jesus_rule_prediction(result_to_predict)