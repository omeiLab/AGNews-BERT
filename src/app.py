import gradio as gr
from predict import predict

examples = [
    ["First images from world's largest digital camera reveal galaxies and cosmic collisions"],
    ["Worker fatally electrocuted at TSMC's under-construction Hsinchu fab"],
    ["Keegan Bradley mulls idea of being playing captain for Ryder Cup"],
    ["Housing market map: Zillow just released its updated home price forecast for 400-plus housing markets"],
    ["Infectious disease found in stranded dolphins poses risk to humans, UH researchers say"],
    ["We Need to Talk About the Massively Hung Zombie in 28 Years Later"]
]

demo = gr.Interface(
    fn = predict,
    inputs = gr.Textbox(lines=5, placeholder = 'Please enter the title of news...'),
    outputs = gr.Plot(label = "Classification Confidence"),
    title = 'Demo: News Classifier',
    examples=examples
)

demo.launch()