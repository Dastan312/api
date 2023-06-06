from fastapi import FastAPI
import numpy as np
import json
from pydantic import BaseModel
from joblib import load
import uvicorn
import xgboost as xgb
from typing import List
import pandas as pd


app = FastAPI()

reg = xgb.Booster()
reg.load_model("xgb_properties") 


# class DataPoint(BaseModel):
#     bedrooms: float 
#     living_area: float
#     post_code: float
#     number_of_floors: float
#     primary_energy_consumption: float
#     construction_year: float


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post('/predict')
def predict(
    # data: DataPoint
    bedrooms: float = 1, 
    living_area: float = 200, 
    post_code: float = 1000, 
    number_of_floors: float = 2, 
    primary_energy_consumption: float = 50, 
    construction_year: float = 2020
):
    data = np.array([[bedrooms, living_area, post_code, number_of_floors, primary_energy_consumption, construction_year]])
    dmatrix = xgb.DMatrix(data)
    prediction = reg.predict(dmatrix)
    converted_prediction = float(prediction[0])
    print('prediction: ', converted_prediction)
    return {"prediction": converted_prediction}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
