from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create FastAPI app
app = FastAPI(title="HuggingFace Translation Service")

# Request model
class TranslationRequest(BaseModel):
    text: str

# Load Hugging Face model once (at startup)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")

# POST endpoint to translate text
@app.post("/translate")
async def translate(req: TranslationRequest):
    result = translator(req.text)
    return {"translation": result[0]['translation_text']}