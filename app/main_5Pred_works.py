from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

model = load('../models/mlr_scaled_5predictorsONLY.joblib')

@app.get("/")
def read_root():
    return 'ML Model for Predicting Beer Style based on user rating criterias (API Expected Parameters) such as Brewery Name AND Beer Appearance, Aroma, Palate, Taste and Volume. Endpoints = /health/beer/type/prediction. github repo link - https://github.com/JKaur1992/adsi_at2_heroku_fastapi'

@app.get('/health', status_code=200)
def healthcheck():
    return 'Multinomial Logistic Regression Model is all ready to go!'

def format_features(aroma: float, appearance: float, palate: float, taste: float, volume: float):
    return {
        'review_aroma': [aroma],
        'review_appearance': [appearance],
        'review_palate': [palate],
        'review_taste': [taste],
        'beer_abv': [volume]
    }
    
@app.get("/beer/type/prediction")
def predict(aroma: float, appearance: float, palate: float, taste: float, volume: float):
    features = format_features(aroma, appearance, palate, taste, volume)
    variables = pd.DataFrame(features)
    pred = model.predict(variables)
    return JSONResponse(pred.tolist())

