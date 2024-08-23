from sklearn.datasets import load_iris
from typing import Union
from fastapi import FastAPI, Body
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load the model
filename = 'carprice_model.joblib'
loaded_model = joblib.load(filename)

class CarPriceData(BaseModel):
    brand: float = Body(...)
    model: float = Body(...)
    type: float = Body(...)
    color: float = Body(...)
    year: float = Body(...)
    mileage: float = Body(...)
    transmission_1: float = Body(...)
    transmission_2: float = Body(...)
    condition_1: float = Body(...)
    condition_2: float = Body(...)
    province: float = Body(...)
    region: float = Body(...)
    age: float = Body(...)

@app.post("/predict-carprice")
def predit_carprice(data: CarPriceData):
    new_data = np.array([
        data.brand,
        data.model,
        data.type,
        data.color,
        data.year,
        data.mileage,
        data.transmission_1,
        data.transmission_2,
        data.condition_1,
        data.condition_2,
        data.province,
        data.region,
        data.age,
    ]).reshape(1, -1)
    predictions = loaded_model.predict(new_data)

    return {"predictions": predictions.tolist()[0]}
