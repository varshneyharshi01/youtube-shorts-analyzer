import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_parameters(text_chunk):
    prompt = f"""
Analyze the following transcript text and extract any present content/creative parameters. 
These include but are not limited to: emotions (e.g. nostalgia, inspiration), tone, style, delivery, audience hooks, and storytelling techniques.

Give me a list of **at least 20-30** such parameters from this text:

{text_chunk}
"""
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text
