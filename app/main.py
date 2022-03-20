from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

model = load('../models/mlr_scaled_6Pred_Binary-Label_Pipeline.joblib')

@app.get("/")
def read_root():
    return 'ML Model for Predicting Beer Style based on user rating criterias (API Expected Parameters) such as name, appearance, aroma, palate, taste and volume. Endpoints = /health/beer/type/prediction'

@app.get('/health', status_code=200)
def healthcheck():
    return 'Multinomial Logistic Regression Model is all ready to go!'

def format_features(name: str, aroma: float, appearance: float, palate: float, taste: float, volume: float):
    return {
        'brewery_name': [name],
        'review_aroma': [aroma],
        'review_appearance': [appearance],
        'review_palate': [palate],
        'review_taste': [taste],
        'beer_abv': [volume]
    }
    
@app.get("/beer/type/prediction")
def predict(name: str, aroma: float, appearance: float, palate: float, taste: float, volume: float):
    features = format_features(name, aroma, appearance, palate, taste, volume)
    variables = pd.DataFrame(features)
    pred = model.predict(variables)
    return JSONResponse(pred.tolist())

