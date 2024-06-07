import os
import requests
import json
import gradio as gr
import time

# Define the function to make the API request
def fetch_response(messages, api_key):
    start_time = time.time()
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/qwen2-72b-instruct",
        "max_tokens": 4096,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [{"role": "user", "content": messages}]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    end_time = time.time()
    response_json = response.json()
    answer_time = end_time - start_time
    return response_json['choices'][0]['message']['content'], answer_time

# Define the Gradio interface
def chatbot_interface(messages, api_key):
    try:
        answer, answer_time = fetch_response(messages, api_key)
        return answer, f"<b><font color='blue'>Answer Time: {answer_time} seconds</font></b>"
    except Exception as e:
        return "Your API key is wrong. Try again!", None

# Create Gradio interface
iface = gr.Interface(fn=chatbot_interface,
                     inputs=[gr.Textbox(label="Message"), gr.Textbox(label="API Key", type="password")],
                     outputs=[gr.Textbox(label="Answer"), gr.Markdown(label="Answer Time")],
                     title="Qwen2 AI Chatbot - Test App - QWEN 2 is released 06.06.2024",
                     description="Feel free to chat with Qwen2 model from Alibaba. Please enter your API key.",
                     theme='Arkaine/Carl_Glow')

# Launch the interface
iface.launch()
