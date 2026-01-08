from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create FastAPI app
app = FastAPI(title="Call HuggingFace Models")

# Load Hugging Face model once (at startup)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")

classifier = pipeline("text-classification", model="abdulmatinomotoso/English_Grammar_Checker")

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

QA = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

zero_shot = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

class TextRequest(BaseModel):
    text: str

class QuestionAndAnswerRequest(BaseModel):
    text: str
    question: str

# POST Endpoint For text translation
@app.post("/translate")
async def translate(req: TextRequest):
    result = translator(req.text)
    #print(result)
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
    #print(result)
    return {
        "label": result[0]["label"],
        "score": result[0]["score"]
    }

# POST Endpoint For text summarization
@app.post("/summarize")
async def summarize(req: TextRequest):
    words = len(req.text.split())
    max_len = max(20, int(words * 0.4))
    min_len = max(10, int(words * 0.2))    
    result = summarizer(req.text, max_length=max_len, min_length=min_len, do_sample=False)
    #print(result)
    return {"summary": result[0]['summary_text']}

# POST Endpoint For Question And Answer
@app.post("/questionAndAnswer")
async def questionAndAnswer(req: QuestionAndAnswerRequest):  
    result = QA(question = req.question,context=req.text)
    #print(result)
    return {"Answer": result['answer']}

# POST Endpoint For zero shot classification
@app.post("/zeroShotClassifier")
async def zeroShotClassifier(req: TextRequest):
    my_labels= ["sport","politics","social","health"]
    output=zero_shot(req.text,my_labels)
    #print(result)
    return {"result": output}

