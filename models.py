from transformers import pipeline,BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
import torch

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")

classifier = pipeline("text-classification", model="abdulmatinomotoso/English_Grammar_Checker")

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

QA = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

zero_shot = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence_model=SentenceTransformer("all-MiniLM-L6-v2")
sentence_model2=SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")






