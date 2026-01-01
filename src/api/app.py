import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipelines.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Vehicle Price Prediction API",
    description="Prediction API for Vehicle Price Prediction",
    version="1.0.0"
)

pipeline = PredictionPipeline()

class VehicleInput(BaseModel):
    year: int = Field(..., example=2017, description="Year of manufacture of the vehicle")
    manufacturer: str = Field(..., example="ford", description="Brand")
    model: str = Field(..., example="f-150", description="Model")
    condition: str = Field(..., example="excellent", description="Vehicle Status")
    cylinders: str = Field(..., example="8 cylinders", description="Number of cylinders")
    fuel: str = Field(..., example="gas", description="Fuel type")
    odometer: float = Field(..., example=48000, description="Kilometers (Miles)")
    title_status: str = Field(..., example="clean", description="License status")
    transmission: str = Field(..., example="automatic", description="Gear type")
    drive: str = Field(..., example="4wd", description="Traction type")
    size: str = Field(..., example="full-size", description="Vehicle size")
    type: str = Field(..., example="pickup", description="Case type")
    paint_color: str = Field(..., example="black", description="Colour")
    state: str = Field(..., example="tx", description="State code")

@app.get("/")
def home():
    return {"message": "Vehicle Price Predictor API is Running! Go to /docs for Swagger UI."}

@app.post("/predict")
def predict_price(vehicle: VehicleInput):
    try:
        input_data = vehicle.dict()

        price = pipeline.predict(input_data)

        return {
            "status": "success",
            "estimated_price": round(price, 2),
            "currency": "USD",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)