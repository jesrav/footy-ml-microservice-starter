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


####################################################################################################
# Examples of rule based examples that does not use any machine learning.
####################################################################################################
def predict_goal_diff_jesus_rule(result_to_predict: RowForML) -> int:
    """Predict the goal diff off the game using only the row of features for the actual game.

    If Jesus (user id 1) is on offence, his team will always win by 5
    Otherwise we predict a random goal diff between -3 and 3, but never 0.
    """

    if result_to_predict.team1_attacker_user_id == 1:
        return 5
    elif result_to_predict.team2_attacker_user_id == 1:
        return -5
    else:
        return choice([-3, -2, -1, 1, 2, 3])


@app.post("/jesus_rule_predict")
def predict(body: DataForML) -> int:
    result_to_predict = [r for r in body.data if r.result_to_predict][0]
    return predict_goal_diff_jesus_rule(result_to_predict)
