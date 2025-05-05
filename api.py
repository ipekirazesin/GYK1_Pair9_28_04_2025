from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Optional
import joblib
import os
from model_training import train_model, predict

app = FastAPI(title="Return Prediction API",
             description="API for predicting product returns",
             version="1.0.0")

class PredictionRequest(BaseModel):
    quantity: float
    unit_price: float
    discount: float
    net_spent: float
    avg_discount_by_product: float
    avg_net_spent_by_product: float

class PredictionResponse(BaseModel):
    prediction: bool
    probability: float
    features: dict

class TrainingResponse(BaseModel):
    message: str
    metrics: dict

@app.get("/")
async def root():
    return {"message": "Welcome to Return Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([{
            'quantity': request.quantity,
            'unit_price': request.unit_price,
            'discount': request.discount,
            'net_spent': request.net_spent,
            'avg_discount_by_product': request.avg_discount_by_product,
            'avg_net_spent_by_product': request.avg_net_spent_by_product
        }])
        
        # Make prediction
        prediction, probability = predict(input_data)
        
        return PredictionResponse(
            prediction=bool(prediction),
            probability=float(probability),
            features=request.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train():
    try:
        metrics = train_model()
        return TrainingResponse(
            message="Model trained successfully",
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 