from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create FastAPI app
app = FastAPI(title="Call HuggingFace Models")

# Load Hugging Face model once (at startup)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")

classifier = pipeline("text-classification", model="abdulmatinomotoso/English_Grammar_Checker")

class TextRequest(BaseModel):
    text: str

# POST Endpoint For text translation
@app.post("/translate")
async def translate(req: TextRequest):
    result = translator(req.text)
    return {"translation": result[0]['translation_text']}


# POST Endpoint For text classification
@app.post("/classify")
async def classify(req: TextRequest):

    """ if u want to set the labels
    label_map = {
        "LABEL_0": "Correct Grammar",
        "LABEL_1": "Grammar Error"
    }
    return {
        "input_text": req.text,
        "result": label_map.get(result["label"], "Unknown"),
        "confidence": round(result["score"], 4)
    }
    """

    result = classifier(req.text)
    print(result)
    return {
        "label": result[0]["label"],
        "score": result[0]["score"]
    }