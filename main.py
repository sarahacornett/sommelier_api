from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from google import genai
from dotenv import load_dotenv

load_dotenv() # loads variables from .env file into os.environ

# --- Setup and Loading ---
app = FastAPI(title="Sommelier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any web app to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],
)

print("Loading ML Models...")
price_model = joblib.load('price_model.pkl')
quality_model = joblib.load('quality_model.pkl')
variety_encoder = joblib.load('variety_encoder.pkl')
region_encoder = joblib.load('region_encoder.pkl')

# Configure Gemini API (We will set the key in your terminal later)
print("Initializing Sommelier Brain...")
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- 2. Define Data Structures (What the app sends us) ---
class WineFeatures(BaseModel):
    variety: str
    region: str
    points: int = 85 # Default value
    price: float = 20.0 # Default value

class ChatMessage(BaseModel):
    message: str

# --- 3. The Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Sommelier API is running!"}

@app.post("/predict-price")
def predict_price(wine: WineFeatures):
    try:
        # Convert text to numbers using the encoders from Day 1
        v_encoded = variety_encoder.transform([wine.variety])[0]
        r_encoded = region_encoder.transform([wine.region])[0]
        
        # Make prediction
        prediction = price_model.predict([[v_encoded, r_encoded, wine.points]])
        return {"predicted_price": round(prediction[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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