from fastapi import FastAPI
from pydantic import BaseModel
from models import translator, classifier, summarizer, QA,zero_shot,tokenizer,model,sentence_model,sentence_model2
from sentence_transformers import util

# Create FastAPI app to call Hugging Face models
app = FastAPI(title="Call HuggingFace Models")

class TextRequest(BaseModel):
    text: str

class QuestionAndAnswerRequest(BaseModel):
    text: str
    question: str

class CompareSentences(BaseModel):
    sentence1: str
    sentence2: str

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

# POST Endpoint For test tokenizer in Contextual Embedding
@app.post("/contextEmbedding")
async def contextEmbedding(req: TextRequest):
    inputs=tokenizer(req.text, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)
    print(token_strings)
    outputs=model(**inputs)
    embedding=outputs.last_hidden_state[0][5]
    #print(len(embedding))
    #print(embedding[:10])
    return {"tokens": token_strings,"tokensIds": input_ids.tolist()}

# POST Endpoint For test tokenizer in sentence Embedding
@app.post("/sentenceEmbedding")
async def sentenceEmbedding(req: CompareSentences):
    e1=sentence_model.encode(req.sentence1, convert_to_tensor=True)
    e2=sentence_model.encode(req.sentence2 , convert_to_tensor=True)
    similarities=util.cos_sim(e1,e2)
    similarity_percentage = float(similarities.item()) * 100
    print(similarity_percentage)
    print(similarities)
    print(len(e1))
    return {"result": round(similarity_percentage, 2)}

# POST Endpoint For test tokenizer in sentence Embedding
@app.post("/arSentenceEmbedding")
async def arSentenceEmbedding(req: CompareSentences):
    e1=sentence_model2.encode(req.sentence1, convert_to_tensor=True)
    e2=sentence_model2.encode(req.sentence2 , convert_to_tensor=True)
    similarities=util.cos_sim(e1,e2)
    similarity_percentage = float(similarities.item()) * 100
    print(similarity_percentage)
    print(similarities)
    print(len(e1))
    return {"result": round(similarity_percentage, 2)}
