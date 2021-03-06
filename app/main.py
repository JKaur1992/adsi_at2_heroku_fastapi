from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

model = load('../models/mlr_scaled_6Pred_Binary-Label_Pipeline.joblib')

@app.get("/")
def read_root():
    return 'ML Model for Predicting Beer Style based on user rating criterias (API Expected Parameters) such as Brewery Name AND Beer Appearance, Aroma, Palate, Taste and Volume. Endpoints = /health/beer/type/multiple/predictors. github repo link - https://github.com/JKaur1992/adsi_at2_heroku_fastapi'

@app.get('/health', status_code=200)
def healthcheck():
    return 'Multinomial Logistic Regression Model is all ready to go!'

def format_feature(name: str, aroma: float, appearance: float, palate: float, taste: float, volume: float):
    return {
        'brewery_name': [name],
        'review_aroma': [aroma],
        'review_appearance': [appearance],
        'review_palate': [palate],
        'review_taste': [taste],
        'beer_abv': [volume]
    }

@app.get("/beer/type/single/predictor")
def predict(name: str):
    feature = format_feature(name, aroma, appearance, palate, taste, volume)
    variable = pd.DataFrame(feature)
    predict = model.predict(variable)
    return JSONResponse(predict.tolist())

def format_features(name: str, aroma: float, appearance: float, palate: float, taste: float, volume: float):
    return {
        'brewery_name': [name],
        'review_aroma': [aroma],
        'review_appearance': [appearance],
        'review_palate': [palate],
        'review_taste': [taste],
        'beer_abv': [volume]
    }

@app.get("/beer/type/multiple/predictors")
def predict(name: str, aroma: float, appearance: float, palate: float, taste: float, volume: float):
    features = format_features(name, aroma, appearance, palate, taste, volume)
    variables = pd.DataFrame(features)
    pred = model.predict(variables)
    return JSONResponse(pred.tolist())

