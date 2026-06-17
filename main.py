from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import joblib
import pandas as pd
import chromadb
from dotenv import load_dotenv
from typing import Optional

#encoders used to conver string literals to expect numerical values
variety_encoder = joblib.load('variety_encoder.pkl')
region_encoder = joblib.load('region_1_encoder.pkl')
province_encoder = joblib.load('province_encoder.pkl')
winery_encoder = joblib.load('winery_encoder.pkl')

def safe_transform(encoder, value):
    # If the value is in our known list, transform it
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    # Otherwise, map it to the "Unknown" category (assuming "Unknown" was in your training data)
    return encoder.transform(["Unknown"])[0]

# Load environment variables (like your Gemini API key)
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS to allow Flutter web app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODELS INTO MEMORY ---
# Load the XGBoost Price Predictor
xgb_model = joblib.load('xgboost_price_model.pkl')

# Load the ChromaDB Vector Database for the Chatbot
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="wine_sommelier")


# --- ROOT ENDPOINT (Health Check) ---
@app.get("/")
async def root():
    return {"message": "Sommelier API is live and running!"}


# --- XGBOOST PRICE PREDICTOR ---
class WinePriceRequest(BaseModel):
    variety: str
    region: str
    province: Optional[str] = "Unknown"  # Optional
    winery: Optional[str] = "Unknown"    # Optional

@app.post("/predict-price")
async def predict_custom_model(request: WinePriceRequest):
    try:
        # If the user leaves them out, they default to "Unknown"
        # Ensure your encoders are trained to handle the string "Unknown"
        # or map None to the most common category (mode).
        input_data = pd.DataFrame([{
            'variety_encoded': safe_transform(variety_encoder, request.variety),
            'region_encoded': safe_transform(region_encoder, request.region),
            'province_encoded': safe_transform(province_encoder, request.province or "Unknown"),
            'winery_encoded': safe_transform(winery_encoder, request.winery or "Unknown")
        }])

        prediction = xgb_model.predict(input_data)
        final_price = round(float(prediction[0]), 2)

        return {"predicted_price": final_price}

    except Exception as e:
        return {"error": str(e)}


# --- RAG CHATBOT ---
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_sommelier(request: ChatRequest):
    try:
        # 1. Search the Vector DB for the top 3 matching real wines
        results = collection.query(
            query_texts=[request.message],
            n_results=3
        )

        # 2. Format those 3 wines into a text list for Gemini to read
        real_wines_in_stock = ""
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            desc = results['documents'][0][i]
            real_wines_in_stock += f"- {meta['title']} ({meta['variety']}): ${meta['price']}\n  Review: {desc}\n\n"

        # 3. Build the strict, heavily-guarded prompt for Gemini
        system_prompt = f"""
        You are a master sommelier. A customer just asked you: "{request.message}"
        
        You MUST ONLY recommend one of the following real wines from our cellar:
        {real_wines_in_stock}
        
        Pick the single best match from that list. Tell the customer its exact name and price, and explain why its flavor profile fits their request based on the review notes. Be charming, professional, and concise. Do NOT make up any wines.
        """

        # 4. Send the prompt to Gemini
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(system_prompt)

        return {"reply": response.text}

    except Exception as e:
        return {"error": str(e)}