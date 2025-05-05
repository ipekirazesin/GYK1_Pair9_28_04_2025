from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from return_preprocessing import df
from return_model import train_return_model

app = FastAPI()

RETURN_MODEL_PATH = "models/return_model.h5"

class ReturnInput(BaseModel):
    quantity: int
    unit_price: float
    discount: float
    net_spent: float
    avg_discount_by_product: float
    avg_net_spent_by_product: float

@app.post("/train/return")
def train_model():
    model, scaler, metrics = train_return_model()
    return {"status": "Model trained successfully"}

@app.post("/predict/return")
def predict_return(data: ReturnInput):
    model = tf.keras.models.load_model(RETURN_MODEL_PATH)
    X = np.array([[
        data.quantity,
        data.unit_price,
        data.discount,
        data.net_spent,
        data.avg_discount_by_product,
        data.avg_net_spent_by_product
    ]])
    prediction = model.predict(X, verbose=0)[0][0]
    return {
        "return_probability": round(float(prediction), 3),
        "will_return": bool(prediction > 0.3),
        "prediction": "İade Edilir" if prediction > 0.3 else "İade Edilmez"
    }

@app.get("/analysis/return")
def get_return_analysis():
    return {
        "total_orders": len(df),
        "return_count": int(df['is_returned'].sum()),
        "return_rate": f"%{(df['is_returned'].mean() * 100):.1f}",
        "top_returned_products": df[df['is_returned'] == 1]['product_name'].value_counts().head(5).to_dict()
    }
