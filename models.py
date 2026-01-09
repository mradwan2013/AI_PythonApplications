from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")

classifier = pipeline("text-classification", model="abdulmatinomotoso/English_Grammar_Checker")

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

QA = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

zero_shot = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")


