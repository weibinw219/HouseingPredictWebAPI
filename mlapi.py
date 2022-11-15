from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import json



app = FastAPI()


class Predict_Inputs(BaseModel):
    Latitude: float
    Longitude: float
    LandSize: float
    SizeInterior: float
    Bedrooms: int
    Bathrooms: int


with open('modelRFR-0.1.0.pkl', 'rb') as f:
    model = pickle.load(f)




@app.post('/')
async def predicting_endpoint(predict_inputs:Predict_Inputs):
    df = pd.DataFrame([predict_inputs.dict().values()], columns=predict_inputs.dict().keys())
    predicted_result = model.predict(df)
    output_result = json.dumps(predicted_result.tolist())
    return {"Prediction":output_result}
