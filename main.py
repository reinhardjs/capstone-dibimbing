from sklearn.datasets import load_iris
from typing import Union
from fastapi import FastAPI, Body
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load the Iris dataset
# carprice = load_carprices()

# Load the model
filename = 'carprice_model.joblib'
loaded_model = joblib.load(filename)

class CarPriceData(BaseModel):
    brand: float = Body(..., gt=0, description="brand")
    model: float = Body(..., gt=0, description="model")
    type: float = Body(..., gt=0, description="type")
    color: float = Body(..., gt=0, description="color")
    year: float = Body(..., gt=0, description="year")
    mileage: float = Body(..., gt=0, description="mileage")
    transmission: float = Body(..., gt=0, description="transmission")
    condition: float = Body(..., gt=0, description="condition")
    price: float = Body(..., gt=0, description="price")
    province: float = Body(..., gt=0, description="province")
    region: float = Body(..., gt=0, description="region")

@app.post("/predict-carprice")
def predit_carprice(data: CarPriceData):
    new_data = np.array([
        [
            data.brand, 
            data.model, 
            data.type, 
            data.color, 
            data.year, 
            data.mileage, 
            data.transmission, 
            data.condition, 
            data.price, 
            data.province, 
            data.region
        ]
    ]) 
    predictions = loaded_model.predict(new_data)
    # predicted_species = carprice.target_names[predictions[0]]

    return {}
    # return { "prediction": predicted_species }
