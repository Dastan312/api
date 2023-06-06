from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import uvicorn
import xgboost as xgb
from typing import List


app = FastAPI()

class DataPoint(BaseModel):
    Bedrooms: float 
    Living_area: float
    Postcode: float
    Number_of_floors: float
    Primary_energy_consumption: float
    Construction_year: float


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post('/api/predict')
def predict(data: List[float]):
    data = [data]  
    dmatrix = xgb.DMatrix(data)
    prediction = reg.predict(dmatrix)
    return {"prediction": prediction[0]}


reg = xgb.XGBRegressor()  
reg.load_model("regXGBoost.pkl") 


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
