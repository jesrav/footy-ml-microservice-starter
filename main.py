"""
FootyTracker ML microservice example
"""
import math
from random import choice

from fastapi import FastAPI

from schemas import RowForML, DataForML

app = FastAPI()


def predict_goal_diff_rule(result_to_predict: RowForML) -> int:
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


@app.post("/rule_based_predict")
def predict(body: DataForML) -> int:
    result_to_predict = [r for r in body.data if r.result_to_predict][0]
    return predict_goal_diff_rule(result_to_predict)
