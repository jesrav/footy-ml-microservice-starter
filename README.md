# FootyTracker starter ML microservice 
Workshop participants will train and deploy their own ML model to be registered at [FootyTracker](https://github.com/jesrav/footy-tracker).
Here it will be making real time predictions of match results and participate on the ML leaderboard🤖


**Giving your model a cool name is equally important as it's predictive performance!**


## Getting started

### Requirements
- Python 3.10

### Install dependencies in your favorite virtual Python env
```bash
pip install -r requirements.txt
```

### Train a new model
```bash
python train_model.py
```

### Run API locally
```bash
uvicorn main:app --port 8000 --reload
```

Check out the API docs at http://127.0.0.1:8000/docs

## Excercices:
- Sign up on FootyTracker here: https://www.footy-tracker.live/account/register
- Change the train_model.py script to use another model / other features.
    - You can see examples of the data your model will recieve when making a prediction here: https://api.footy-tracker.live/ml/example_prediction_data/json
- Make the model API serve your new model locally
- Build docker image with API and deploy your model to Azure App service
- Set up a deployment pipeline with either Github Actions or ADO. Potentially see how it's done in the Kapacity ML Framework.
- Set up countinous training on a schedule. See how its done for Richard Prior here: https://github.com/jesrav/footy-tracker/blob/main/.github/workflows/train-and-deploy-ml-api.yml
