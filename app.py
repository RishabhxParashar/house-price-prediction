from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# 1. Load your trained pipeline
# Ensure pipeline.pkl is in the same folder as this script
pipeline = joblib.load("pipeline.pkl")

# 2. Create FastAPI app
app = FastAPI()

# 3. Define input schema
class HouseData(BaseModel):
    GrLivArea: int
    BedroomAbvGr: int
    FullBath: int

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(data: HouseData):
    try:
        # Convert Pydantic object to dict, then to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # FIX: Handle missing columns required by the model
        # This automatically fills any feature your model needs with 0 
        model_features = pipeline.feature_names_in_
        for col in model_features:
            if col not in df.columns:
                df[col] = 0 
        
        # Ensure columns are in the exact order the model expects
        df = df[model_features]
        
        # Make prediction (Result is likely in Log scale, e.g., 11.01)
        log_prediction = pipeline.predict(df)[0]
        
        # Convert Log Scale back to Dollars: e^11.01
        actual_price = np.exp(log_prediction)
        
        return {
            "status": "success",
            "log_value": float(log_prediction),
            "predicted_price_usd": round(float(actual_price), 2),
            "formatted_price": f"${actual_price:,.2f}"
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
