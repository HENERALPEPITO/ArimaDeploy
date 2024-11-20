from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import logging
from fastapi import FastAPI, HTTPException, Request


app = FastAPI()

# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# File paths for model and scalers
MODEL_PATH = "arima_model.pkl"
SCALER_FEATURES_PATH = "scaler_features.pkl"
SCALER_TARGET_PATH = "scaler_target.pkl"

# Load the trained ARIMA model and scalers
try:
    arima_model = joblib.load(MODEL_PATH)
    scaler_features = joblib.load(SCALER_FEATURES_PATH)
    scaler_target = joblib.load(SCALER_TARGET_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Required file not found: {e.filename}")

class FutureData(BaseModel):
    Rainfall: list
    Temperature: list
    Humidity: list

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Render index.html as the main page
    with open("index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict")
async def predict_cases(future_data: FutureData, request: Request):
    # Log incoming request data for debugging
    data = await request.json()
    logging.info(f"Received data: {data}")

    try:
        # Validate input length
        if len(future_data.Rainfall) != 4 or len(future_data.Temperature) != 4 or len(future_data.Humidity) != 4:
            raise HTTPException(status_code=400, detail="Input lists must have 4 elements each.")

        # Create DataFrame for exogenous variables
        data = pd.DataFrame({
            "Rainfall": future_data.Rainfall,
            "Temperature": future_data.Temperature,
            "Humidity": future_data.Humidity
        })

        # Scale the input data
        scaled_data = scaler_features.transform(data)

        # Predict the next 4 cases
        predictions_scaled = arima_model.forecast(steps=4, exog=scaled_data)

        # Convert predictions_scaled to a NumPy array and reshape for inverse transformation
        predictions = scaler_target.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

        return {"predicted_cases": predictions.flatten().tolist()}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
