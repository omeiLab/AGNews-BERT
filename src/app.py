import gradio as gr
from predict import predict

demo = gr.Interface(
    fn = predict,
    inputs = gr.Textbox(lines=2, placeholder = 'Please enter the title of news...'),
    outputs = gr.Textbox(lines = 2),
    title = 'Demo: News Classifier',
)

demo.launch()