from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD YOUR CUSTOM ML MODEL ---
# This loads the model into the server's memory the moment it starts up up
xgb_model = joblib.load('xgboost_price_model.pkl')

# --- 1. THE FLEXIBLE INPUT SCHEMA ---
# Notice we set default values. If the Flutter app doesn't send a province, 
# it defaults to 'Unknown' instead of crashing!
class WinePriceRequest(BaseModel):
    variety: str = "Unknown"
    region: str = "Unknown"
    province: str = "Unknown"
    winery: str = "Unknown"
    points: int = 88  # 88 is roughly the average score in the dataset

# ... (Keep your root health check endpoint)

# --- 2. THE NEW PREDICTION ENDPOINT ---
@app.post("/predict-price")
async def predict_custom_model(request: WinePriceRequest):
    try:
        # Convert the incoming JSON into a 1-row Pandas DataFrame
        input_data = pd.DataFrame([{
            'variety': request.variety,
            'region_1': request.region, # Matching the column name the model trained on
            'province': request.province,
            'winery': request.winery,
            'points': request.points
        }])

        # Tell Pandas these are categorical columns (Required for XGBoost)
        for col in ['variety', 'region_1', 'province', 'winery']:
            input_data[col] = input_data[col].astype('category')

        # Run the math!
        prediction = xgb_model.predict(input_data)
        
        # Round the price to 2 decimal places so it looks like real money
        final_price = round(float(prediction[0]), 2)

        return {"predicted_price": final_price}

    except Exception as e:
        return {"error": str(e)}



@app.post("/predict-quality")
def predict_quality(wine: WineFeatures):
    try:
        v_encoded = variety_encoder.transform([wine.variety])[0]
        r_encoded = region_encoder.transform([wine.region])[0]
        
        prediction = quality_model.predict([[v_encoded, r_encoded, wine.price]])
        quality_label = "High Quality (90+)" if prediction[0] == 1 else "Average (Under 90)"
        return {"quality_prediction": quality_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
def chat_with_sommelier(chat: ChatMessage):
    try:
        prompt = f"You are an elite, slightly snobby but helpful Sommelier. Keep your answer under 3 sentences. The user says: {chat.message}"
        
        # New syntax for the updated SDK using the latest Flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")