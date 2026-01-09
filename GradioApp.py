import gradio as gr
import numpy as np
from transformers import pipeline
from models import translator, classifier, summarizer, QA,zero_shot
from PIL import Image

def translate_to_arabic(text):
    result = translator(text)
    return result[0]['translation_text']

def detect_image(image):
    return "شكرا لك"

with gr.Blocks() as app:
    gr.Markdown("Testing Gradio")

    with gr.Row():
        with gr.Column():
            image_input=gr.Image(type="pil",label="Upload the Image please" )
            detect_image_btn=gr.Button("Detect the Image")
            pred=gr.Textbox(label="Thanks Result")
        detect_image_btn.click(fn=detect_image , inputs=image_input , outputs=pred)

    with gr.Row():
        translated_output=gr.Textbox(label="Translated Text")
        translate_btn=gr.Button("Translate to Arabic")
        translate_btn.click(fn=translate_to_arabic , inputs=pred , outputs=translated_output)

app.launch() 

#iface = gr.Interface(fn=translate_to_arabic, inputs="text", outputs="text")
#iface.launch()  