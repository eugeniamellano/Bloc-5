import uvicorn
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from pydantic import BaseModel
from typing import List
import pandas as pd
from fastapi import Query

app = FastAPI(
    title="Getaround API",
    description="¡Welcome to Getaround API! This API provides endpoints to predict car prices based on their features. Additionally, you can view some examples of the information!"
)




class PredictionFeatures(BaseModel):
    model_key: str = Query(..., description="Model key of the car (e.g., Citroën Peugeot, PGO, Renault, Audi, BMW, Ford, Mercedes)")
    mileage: int = Query(..., description="Mileage of the car (numerical value)")
    engine_power: int = Query(..., description="Engine power of the car (numerical value)")
    private_parking_available: bool = Query(..., description="Whether private parking is available (True/False)")
    has_gps: bool = Query(..., description="Whether the car has GPS (True/False)")
    fuel: str = Query(..., description="Type of fuel (e.g., petrol, hybrid_petrol, electro)")
    paint_color: str = Query(..., description="Paint color of the car (e.g., black, grey, white, red, silver, blue, orange, beige, brown, green)")
    car_type: str = Query(..., description="Type of car (e.g., convertible, coupe, estate, hatchback, sedan, subcompact, suv, van)")
    has_air_conditioning: bool = Query(..., description="Whether the car has air conditioning (True/False)")
    automatic_car: bool = Query(..., description="Whether the car is automatic (True/False)")
    has_getaround_connect: bool = Query(..., description="Whether the car has Getaround Connect (True/False)")
    has_speed_regulator: bool = Query(..., description="Whether the car has speed regulator (True/False)")
    winter_tires: bool = Query(..., description="Whether the car has winter tires (True/False)")


class PredictionResponse(BaseModel):
    predictions: List[float]

# Load MLflow model as a PyFuncModel
logged_model = 'runs:/1733be20356241f0840dc65bebf4d089/getaround_project'
loaded_model = mlflow.pyfunc.load_model(logged_model)

def make_prediction(model, data):
    # Prepare DataFrame for prediction
    df = pd.DataFrame(data, columns=["model_key", "mileage", "engine_power", "private_parking_available", 
                                     "has_gps", "fuel", "paint_color", "car_type", "has_air_conditioning", 
                                     "automatic_car", "has_getaround_connect", "has_speed_regulator", 
                                     "winter_tires"])
    # Make prediction
    prediction = model.predict(df)
    return prediction.tolist()

# Read dataset and drop unnecessary column
df = pd.read_csv("get_around_pricing_project.csv")
df = df.drop(axis=1, columns="Unnamed: 0") 


@app.get("/", tags=["Root"])
async def root(name: str):
    return {"message": f"Welcome {name} to Getaround API!", "documentation": "/docs"}

#@app.get("/", tags=["Root"])
#async def root():
#    return {"message": "¡Bienvenido a la API de Getaround!", "documentation": "/docs"}

@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
async def predict(prediction_features: PredictionFeatures):
    # Prepare data for prediction
    data = [{
        "model_key": prediction_features.model_key,
        "mileage": prediction_features.mileage,
        "engine_power": prediction_features.engine_power,
        "private_parking_available": prediction_features.private_parking_available,
        "has_gps": prediction_features.has_gps,
        "fuel": prediction_features.fuel,
        "paint_color": prediction_features.paint_color,
        "car_type": prediction_features.car_type,
        "has_air_conditioning": prediction_features.has_air_conditioning,
        "automatic_car": prediction_features.automatic_car,
        "has_getaround_connect": prediction_features.has_getaround_connect,
        "has_speed_regulator": prediction_features.has_speed_regulator,
        "winter_tires": prediction_features.winter_tires
    }]

    # Make prediction
    predictions = make_prediction(loaded_model, data)

    # Prepare response
    response = PredictionResponse(predictions=predictions)

    return response

@app.get("/preview/{number_row}", tags=["Data-Preview"], response_model=List[dict])
async def preview_dataset(number_row: int = 5):
    """
    Display a sample of rows of the dataset.
    `number_row` parameter allows to specify the number of rows you would like to display (default value: 5).
    """
    return df.head(number_row).to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4006)