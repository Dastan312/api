from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from joblib import load
import pickle
from typing import List


# ['Bedrooms', 'Living_area', 'Postcode', 'Number_of_floors','Primary_energy_consumption', 'Construction_year']
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello, World'}

class DataPoint(BaseModel):

    Bedrooms: int 
    Living_area: float
    Postcode: int
    Number_of_floors: int
    Primary_eergy_consumption: float
    Construction_year: int


@app.post("/predict")
def predict(data: List[float]):
    prediction = predict(data)
    return {"prediction": prediction}



pickle_in = open("regXGBoost.pkl","rb")
reg=pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post('/predict')
def Price(data:DataPoint):
    data = data.dict()
    Bedrooms=data['Bedrooms']
    Living_area=data['Living_area']
    Number_of_floors=data['Number_of_floors']
    Construction_year=data['Construction_year']




    # prediction = reg.predict([['Bedrooms', 'Living_area', 'Postcode', 'Number_of_floors','Primary_energy_consumption', 'Construction_year']])
    # if(prediction[0]>0.5):
    #     prediction="BAD"
    # else:
    #     prediction="GOOD"
    # return {
    #     'prediction': prediction
    # }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000




#http://127.0.0.1:8000/docs


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


#uvicorn main:app --reload

   
