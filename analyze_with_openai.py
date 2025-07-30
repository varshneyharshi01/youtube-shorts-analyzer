import os
import re
from datetime import timedelta
from dotenv import load_dotenv
from openai import OpenAI  # ✅ New SDK import

# Load API key from .env
load_dotenv()
client = OpenAI()  # ✅ Automatically uses OPENAI_API_KEY from environment

def parse_timestamp(ts):
    h, m, s = map(int, ts.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)

def split_transcript(transcript, interval_minutes=10):
    lines = transcript.splitlines()
    chunks = []
    current_chunk = []
    current_time = timedelta()
    next_cutoff = timedelta(minutes=interval_minutes)

    for line in lines:
        match = re.match(r"(\d{2}:\d{2}:\d{2})", line)
        if match:
            ts = parse_timestamp(match.group(1))
            if ts >= next_cutoff:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                next_cutoff += timedelta(minutes=interval_minutes)
        current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def analyze_chunk(chunk, index):
    prompt = f"""
I have a raw video and its transcript. My goal is to create effective YouTube Shorts from this content.

Transcript Chunk {index} (10-min interval):
{chunk}

Can you help me analyze the transcript? Please identify potential short clips and provide the following information for each one:

Timestamps: The start and end times for the clip.
Content Focus: A brief summary of the key message or theme of the clip.
Strategy: How to best present this clip for a short-form audience (e.g., focus on a shocking statement, use a quick pace, etc.).
Hook: A short, engaging sentence or phrase that could be used as the video's title or opening line.
Success Potential: An assessment of how likely the clip is to perform well (e.g., High, Very High, etc.).

Please present the final data in a markdown table format, with each suggestion on a separate row.
Also, include three suggested titles for each clip in the table. The titles should be attention-grabbing and suitable for YouTube Shorts.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a YouTube Shorts strategist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=3000
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    with open("transcript.txt", "r") as f:
        transcript = f.read()

    chunks = split_transcript(transcript, interval_minutes=10)

    for idx, chunk in enumerate(chunks, 1):
        print(f"\n🔍 Analyzing chunk {idx}/{len(chunks)}...\n")
        result = analyze_chunk(chunk, idx)
        print(result)
