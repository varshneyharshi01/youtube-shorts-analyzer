import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load the API key from .env
load_dotenv()
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def get_clip_suggestions(transcript_text):
    prompt = f"""
You are a YouTube Shorts expert. From the following transcript, extract the **top 5 short-worthy clips**. 

For each clip, provide:
- Title
- Start Time (00:MM:SS)
- End Time
- Hook line
- Why it works (engagement logic)
- Suggested strategy (text overlays, tone, music, etc.)

Transcript:
{transcript_text}
"""

    model = genai.GenerativeModel("gemini-3.1-pro-preview")
    response = model.generate_content(prompt)
    return response.text

# Run manually
if __name__ == "__main__":
    with open("transcript.txt", "r") as f:
        transcript = f.read()[:3000]  # ✅ Properly indented here

    output = get_clip_suggestions(transcript)
    print("🎯 Suggested Clips:\n")
    print(output)
