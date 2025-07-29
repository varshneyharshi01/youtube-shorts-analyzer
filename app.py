import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
from datetime import timedelta
import yt_dlp
import whisper
import pandas as pd

# --- Load API Key ---
# Make sure you have GEMINI_API_KEY in your .env file
load_dotenv()
try:
    # It's a good practice to check if the key exists before configuring
    if "GEMINI_API_KEY" in os.environ:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        st.error("GEMINI_API_KEY not found in your .env file. Please add it to continue.")
        st.stop() # Stop the app if the key is not available
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please make sure your GEMINI_API_KEY is set correctly. Error: {e}")
    st.stop()

st.title("YouTube Shorts Analyzer (Gemini Edition) 🤖")
st.markdown("Analyze a YouTube video or upload a transcript (`.txt`, `.srt`) to auto-generate short-worthy clips.")

# --------- Helper Functions ---------

@st.cache_data
def format_timestamp(seconds):
    """Converts seconds into SRT timestamp format (HH:MM:SS,ms)."""
    # Ensure seconds is a float
    seconds = float(seconds)
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def download_audio_from_youtube(url, output_path):
    """Downloads the best quality audio from a YouTube URL and saves it as an MP3."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"{os.path.splitext(output_path)[0]}.mp3"
    except Exception as e:
        st.warning(f"Could not download audio from {url}. Skipping. Error: {e}")
        return None

@st.cache_data
def transcribe_audio_with_whisper(video_url):
    """
    Transcribes audio using Whisper and returns a timestamped SRT string.
    This is the key function for getting timestamps.
    """
    audio_file = download_audio_from_youtube(video_url)

    # Let the user know the process has started
    st.info("Whisper model loaded. Starting transcription... (This can take a few minutes for long videos)")

    model = whisper.load_model("base")
    # The 'word_timestamps=True' argument is more intensive but available.
    # For clip-level analysis, segment-level is usually sufficient.
    result = model.transcribe(audio_file, fp16=False)

    st.success("Transcription complete. Formatting SRT...")

    srt_content = []
    # Iterate through each text segment detected by Whisper
    for i, segment in enumerate(result['segments'], 1):
        # Get start and end times for the segment
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])

        # Get the actual text of the segment
        text = segment['text'].strip()

        # Append the SRT block
        srt_content.append(str(i))
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(f"{text}\n")

    return "\n".join(srt_content)

def analyze_transcript_for_clips(chunk, model_name, num_clips):
    """
    Analyzes a transcript chunk to find viral clips using a highly-detailed, specific prompt
    designed to replicate the high-quality output from the user's Google Sheet.
    """
    # This is the new, high-accuracy prompt.
    prompt = f"""
You are an expert viral video editor and social media strategist, specializing in creating compelling YouTube Shorts from long-form content. Your analysis must be sharp, insightful, and directly actionable for a video editing team.

Analyze the provided SRT transcript and identify at least {num_clips} potential viral clips. For each clip, provide a detailed breakdown formatted EXACTLY as a markdown table with the following 9 columns. Do not include any introductory or concluding text outside the table.

**Column-wise Instructions (Crucial for success):**

1.  **`Clip #`**: Sequential numbering starting from 1.
2.  **`Timestamps (Start - End)`**: Provide the EXACT start and end timestamps from the SRT file for the identified clip. This is non-negotiable.
3.  **`Why It Will Work (The Core Idea)`**: In one or two sentences, explain the core concept of the clip. What makes it compelling? (e.g., "Reveals a surprising statistic," "Tells a relatable failure story," "Offers a counter-intuitive piece of advice").
4.  **`Hooks for the Short (3 Options)`**: Provide exactly three short, punchy, scroll-stopping text hooks (3-7 words each) that could be used in the first 2 seconds of the video. Use bullet points or a numbered list within the cell.
5.  **`Strategy (Editing & Pacing)`**: Give specific, actionable editing advice. Mention pacing, visual elements, and audio cues. (e.g., "Start with a quick zoom on the speaker. Use kinetic typography for key phrases. Add a subtle 'whoosh' sound effect. Keep the pace fast with jump cuts.").
6.  **`Success Potential (High/Medium)`**: Rate the clip's potential for going viral as either 'High' or 'Medium'. Justify your rating in one sentence with a because in it(e.g., "High: Creates curiosity, plays on common perceptions of mentalists, and includes a strong, somewhat mysterious personal statement.").
7.  **`On-Screen Text (Headers)`**: Suggest exactly three bold, capitalized, on-screen headers (3-5 words max) that can appear during the video to retain attention. Use bullet points or a numbered list.
8.  **`Titles (For YouTube)`**: Write exactly three click-worthy, SEO-friendly YouTube Short titles. Include intriguing numbers, powerful quotes, or create a curiosity gap.
9.  **`Relevant Quote`**: Extract the single most powerful and representative sentence or phrase from the transcript for the identified clip.

**Transcript to Analyze:**
---
{chunk}
---
"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return None

def parse_markdown_table(markdown_text):
    """Parses a Markdown table into a Pandas DataFrame."""
    if not markdown_text:
        st.warning("Received empty response from the model.")
        return pd.DataFrame()

    lines = markdown_text.strip().split('\n')
    table_lines = [line for line in lines if '|' in line and '---' not in line]

    if not table_lines:
        st.warning("Could not find any table data in the model's response.")
        return pd.DataFrame()

    header = [h.strip() for h in table_lines[0].split('|') if h.strip()]
    data = []
    for line in table_lines[1:]:
        row = [r.strip() for r in line.split('|') if r.strip()]
        if len(row) == len(header):
            data.append(row)

    if not data:
        st.warning("Markdown table parsed, but no data rows were found.")
        return pd.DataFrame(columns=header)

    df = pd.DataFrame(data, columns=header)
    return df

# --------- Streamlit UI Section ---------

# Initialize session state to hold the transcript
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""

# Create tabs for different input methods
tab1, tab2, tab3, tab4, tab5 = st.tabs(["From YouTube URL", "From File Upload", "🕵️‍♀️ Analyze Patterns", "✍️ Generate Titles", "📈 Trend Analysis"])

with tab1:
    st.subheader("Analyze from a YouTube Video")
    video_url = st.text_input("Enter YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=...")
    if st.button("Generate Transcript", key="generate_transcript_btn"):
        if video_url:
            try:
                with st.spinner("Downloading audio and transcribing... This may take a few minutes."):
                    transcript_text = transcribe_audio_with_whisper(video_url)
                    st.session_state['transcript'] = transcript_text
                st.success("Transcript generated successfully! ✅")
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
        else:
            st.warning("Please enter a YouTube URL first.")

with tab2:
    st.subheader("Analyze from a Transcript File")
    uploaded_file = st.file_uploader(
        "Choose a .txt or .srt file",
        type=['txt', 'srt'],
        help="Upload a plain text or SubRip subtitle file."
    )
    if uploaded_file is not None:
        # To read file as string:
        try:
            transcript_text = uploaded_file.getvalue().decode("utf-8")
            st.session_state['transcript'] = transcript_text
            st.success(f"Successfully loaded `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- This section is shared and appears only after a transcript is loaded ---
if st.session_state['transcript']:
    st.markdown("---")
    st.subheader("Transcript Loaded")
    with st.expander("View Full Transcript"):
        st.text_area("Transcript", st.session_state['transcript'], height=300)

    st.markdown("---")
    st.subheader("Analysis Configuration")

    col1, col2 = st.columns(2)

    with col1:
        num_clips = st.number_input(
            "Number of clip suggestions",
            min_value=5, max_value=50, value=10, step=5
        )

    with col2:
        model_choice = st.selectbox(
            "Choose Gemini Model",
            (
                "gemini-1.5-flash-latest", # Fast and capable
                "gemini-1.5-pro-latest"   # Most powerful
            ),
            index=0,
            help="Flash is faster, Pro is more powerful. Start with Flash."
        )

    # --- Analyze Button ---
    if st.button("Analyze Transcript", key="analyze_btn", type="primary"):
        try:
            transcript = st.session_state.get('transcript', '')
            with st.spinner(f"Gemini is analyzing the transcript..."):
                raw_table = analyze_transcript_for_clips(transcript, model_choice, num_clips)

            if raw_table:
                df = parse_markdown_table(raw_table)
                if not df.empty:
                    st.success("Analysis Complete!")
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Analysis as CSV",
                        data=csv,
                        file_name="gemini_shorts_analysis.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Analysis finished, but no valid data was returned from the model.")
            else:
                st.error("Analysis failed. No response from the model.")

        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")

          # --------- New Tab: Viral Pattern Analysis ---------
with tab3:
    st.subheader("Deconstruct What Makes a Short Go Viral")
    st.markdown(
        "Add up to 5 successful shorts. The tool will **transcribe each one** and then analyze the content to find common patterns and strategies."
    )

    # --- Helper Functions (these are unchanged) ---

    def get_transcript_for_single_video(url, whisper_model):
        """Downloads, transcribes, and cleans up a single video URL."""
        try:
            output_template = f"temp_audio_{hash(url)}"
            mp3_file_path = download_audio_from_youtube(url, output_template)
            if mp3_file_path and os.path.exists(mp3_file_path):
                result = whisper_model.transcribe(mp3_file_path, fp16=False)
                full_text = " ".join([segment['text'].strip() for segment in result['segments']])
                os.remove(mp3_file_path)
                return full_text
            return None
        except Exception as e:
            st.warning(f"Failed to process {url}. Error: {e}")
            return None

    def analyze_viral_patterns(transcripts_str, model_name):
        """Analyzes a string of transcripts to find viral patterns."""
        prompt = f"""
You are an expert viral video analyst and YouTube Shorts strategist. Your mission is to deconstruct the mechanics of the successful YouTube Shorts provided below, based on their full transcripts. Produce a two-part strategic report. Your analysis must be sharp, insightful, and move beyond surface-level observations.

**PART 1: INDIVIDUAL VIDEO DECONSTRUCTION**

For each of the YouTube Short transcripts provided, provide a detailed analysis covering these specific points:

* **Core Concept & Hook:** Based on the transcript, what is the video's one-sentence idea? What is the verbal hook in the first few lines that grabs the listener?
* **Emotional Driver:** What is the primary emotion the transcript's language and story triggers? (e.g., Curiosity, Humor, Awe, Relatability, FOMO, Anger). Why is this emotion powerful for sharing?
* **Storytelling & Pacing:** Analyze the transcript's structure. Is it a fast-paced list? A slow-burn story? How does the language create pace and retain attention?
* **Value Proposition:** What value does the viewer get from this content? (e.g., a new insight, a quick laugh, a solution to a problem).
* **Unique Differentiator:** What is it about the message or the way it's phrased in the transcript that makes it stand out?

**PART 2: SYNTHESIZED PATTERNS & ACTIONABLE BLUEPRINT**

After analyzing all transcripts, synthesize your findings into a summary report:

* **Common Patterns:** What are the most frequent patterns you observed across all transcripts in terms of narrative structure, topic, language style, or hook formulas?
* **Emerging Trends:** Do these patterns point to a larger, emerging content trend on YouTube Shorts right now?
* **Actionable Blueprint:** Based on your complete analysis, provide a step-by-step blueprint for scripting a new, potentially viral short.
* **Key Takeaways:** What are the top 3-5 takeaways that any content creator should apply to their next YouTube Short based on these transcripts?
* **IMPORTANT:** Do not include any introductory or concluding text outside the analysis sections. Focus solely on the analysis and actionable insights.
* **Important:** list down all the parameters you found common in the transcripts (e.g. emotional driver, storytelling, value proposition, etc.) in a bullet point list.

**TRANSCRIPTS FOR ANALYSIS:**
{transcripts_str}
"""
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"An error occurred with the Gemini API: {e}")
            return None

    # --- UI for the new tab with DYNAMIC inputs ---

    st.markdown("##### Video URLs to Analyze")

    # Initialize the number of URL inputs in session state if it doesn't exist
    if 'num_url_inputs' not in st.session_state:
        st.session_state.num_url_inputs = 1

    # Functions to modify the number of inputs, called by the buttons
    def add_url_input():
        if st.session_state.num_url_inputs < 5:
            st.session_state.num_url_inputs += 1

    def remove_url_input():
        if st.session_state.num_url_inputs > 1:
            # Clear the value of the input field being removed
            key_to_remove = f"url_{st.session_state.num_url_inputs - 1}"
            if key_to_remove in st.session_state:
                st.session_state[key_to_remove] = ""
            st.session_state.num_url_inputs -= 1

    # Display the text inputs dynamically based on the number in session state
    for i in range(st.session_state.num_url_inputs):
        st.text_input(f"URL for Short #{i+1}", key=f"url_{i}")

    # Display buttons to add/remove inputs side-by-side
    col1, col2, _ = st.columns([1, 2, 3])
    with col1:
        st.button("Add URL ➕", on_click=add_url_input, use_container_width=True)
    with col2:
        st.button(
            "Remove Last ➖",
            on_click=remove_url_input,
            use_container_width=True,
            disabled=(st.session_state.num_url_inputs <= 1)
        )

    st.markdown("---")

    model_choice_tab3 = st.selectbox(
        "Choose Gemini Model ",
        ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"),
        index=0,
        help="Flash is faster for quick analysis, Pro is more powerful for deep insights.",
        key="model_choice_tab3",
    )

    if st.button("Find Patterns from Transcripts", type="primary", key="find_patterns_btn"):
        # Collect all URLs from the dynamically generated input fields
        urls_to_process = []
        for i in range(st.session_state.num_url_inputs):
            url_value = st.session_state.get(f"url_{i}", "")
            if url_value and url_value.strip():
                urls_to_process.append(url_value)
        
        valid_urls = urls_to_process

        if valid_urls:
            with st.spinner("Loading Whisper model... (This happens once)"):
                whisper_model = whisper.load_model("base")

            progress_bar = st.progress(0, text="Starting transcription process...")
            all_transcripts_for_prompt = []
            
            for i, url in enumerate(valid_urls):
                progress_text = f"Processing video {i+1}/{len(valid_urls)}: Transcribing..."
                progress_bar.progress((i) / len(valid_urls), text=progress_text)
                transcript_text = get_transcript_for_single_video(url, whisper_model)
                if transcript_text:
                    formatted_transcript = f"--- TRANSCRIPT FOR VIDEO {i+1} ({url}) ---\n{transcript_text}\n--- END TRANSCRIPT ---\n"
                    all_transcripts_for_prompt.append(formatted_transcript)

            progress_bar.progress(1.0, text="All transcripts generated. Analyzing with Gemini...")

            if all_transcripts_for_prompt:
                final_prompt_content = "\n".join(all_transcripts_for_prompt)
                analysis_result = analyze_viral_patterns(final_prompt_content, model_choice_tab3)
                progress_bar.empty()

                if analysis_result:
                    st.markdown("---")
                    st.subheader("🔬 Analysis Report")
                    st.markdown(analysis_result)
                else:
                    st.error("Analysis failed. No response was received from the model.")
            else:
                progress_bar.empty()
                st.error("Could not generate transcripts for any of the provided URLs. Please check the URLs and try again.")
        else:
            st.warning("Please enter at least one YouTube Short URL to analyze.")

            # --------- New Tab: Shorts Title Generator ---------
with tab4:
    st.subheader("✍️ YouTube Shorts Title Generator")
    st.markdown(
        "Paste the full transcript of your Short below, and Gemini will suggest a variety of titles based on proven viral strategies."
    )

    # Function to generate titles using the detailed prompt
    def generate_shorts_titles(transcript, model_name):
        """Generates Shorts titles from a transcript using a specific prompt."""
        # The detailed prompt for generating titles
        prompt = f"""
You are an expert YouTube Shorts title writer. You specialize in crafting short, punchy, and highly clickable titles that stop viewers from scrolling and maximize engagement.

Your task is to analyze the provided transcript of a YouTube Short and generate **5-7 diverse title options**. You must strictly adhere to the following strategies and formatting rules.

**--- Title Generation Strategies ---**

**1. 🏆 Find the "Golden Line" (Direct quotes from the transcript):**
* **The Punchline/Reveal:** Find the single most surprising or valuable statement and use it as the title.
* **The Strongest Opinion:** Pull the most controversial, direct, or absolute opinion from the text.
* **The First Line Hook:** If the first sentence of the transcript is short and effective, use it directly.

**2. ⚔️ Frame the Core Conflict (Problem/Solution):**
* **State the Problem:** Identify the core problem the transcript solves and frame it as a relatable question.
* **Hint at the Solution:** Allude to the simple, fast, or surprising solution found in the transcript.

**3. 🤔 Create Intense Curiosity (Intrigue):**
* **Promise a Clear Outcome:** Find the end result of an action or experiment in the transcript and build the title around it.
* **Highlight a Contradiction:** Focus on any statement in the transcript that goes against common knowledge or expectations.

**--- Formatting & Style Rules ---**

* **Extreme Brevity:** All titles MUST be under 50 characters.
* **Core Keyword First:** Whenever possible, start the title with the main topic for instant algorithmic categorization (e.g., "iPhone Tip: ...", "Cooking Hack: ...").
* **Minimalism:** Avoid generic, low-effort phrases like "Watch This!", "Mind-Blowing!", or "OMG!". The title should derive its power from the content itself.

**--- Output Format ---**

Present the generated titles in a categorized list based on the strategy used.

**--- TRANSCRIPT FOR ANALYSIS ---**

{transcript}
"""
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"An error occurred with the Gemini API: {e}")
            return None

    # --- UI for the new tab ---
    transcript_input = st.text_area(
        "Paste your full Short transcript here",
        height=250,
        key="transcript_input_tab4",
        placeholder="e.g., Here are three mistakes everyone makes when visiting Japan..."
    )

    model_choice_tab4 = st.selectbox(
        "Choose Gemini Model   ",  # Extra spaces to ensure a unique widget key
        ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"),
        index=0,
        help="Flash is faster for quick ideas. Pro is more powerful for creative depth.",
        key="model_choice_tab4",
    )

    if st.button("Suggest Titles", type="primary", key="suggest_titles_btn"):
        if transcript_input and transcript_input.strip():
            with st.spinner("Gemini is crafting some titles..."):
                suggested_titles = generate_shorts_titles(transcript_input, model_choice_tab4)

            if suggested_titles:
                st.markdown("---")
                st.subheader("🔥 Suggested Titles")
                st.markdown(suggested_titles)
            else:
                st.error("Title generation failed. No response was received from the model.")
        else:
            st.warning("Please paste a transcript into the text area above.")

 # --------- New Tab 5: Trend Analysis ---------
with tab5:
    st.subheader("📈 YouTube Shorts Trend Analysis")
    st.markdown(
        "Analyze up to 5 Shorts to uncover **deep strategic insights, theme combinations, and hidden opportunities** for your next viral video."
    )

    # --- Helper Functions for Trend Analysis ---

    def analyze_common_themes_impressive(formatted_transcripts, model_name):
        """Analyzes combined transcripts to find deep insights and theme stacks."""
        # The re-engineered prompt for an impressive, strategic report
        prompt = f"""
You are a world-class YouTube content strategist and trend forecaster based in Noida as of July 2025. Your clients at major media houses rely on your sharp, counter-intuitive insights to gain a competitive edge. Your specialty is identifying not just trends, but the deeper cultural currents and "Theme Stacks" that drive virality.

Your mission is to perform a deep-level strategic analysis of the 5 provided YouTube Shorts transcripts. Go beyond simple pattern-matching. Deliver a "Boardroom-Ready" strategic brief that is both insightful and inspiring. Focus on identifying how these creators combine multiple themes (e.g., 'Life Hack' + 'Vulnerability') to create a more powerful emotional impact.

Use this **Theme Reference Library** to guide your internal analysis:
* **Foundational Themes:** Educational (Life Hack, Common Mistakes), Narrative (Transformation, A Day in the Life), Emotional (Oddly Satisfying, Relatable Humor), Comparative (Cheap vs. Expensive).
* **Advanced/Niche Themes:** Niche Expertise (The Pro's Secret), Human Connection (Storytime, Vulnerability), Novelty & Spectacle ("What If...?" Experiment, Unique Skill Showcase).

Present your final report **EXACTLY** in the following strategic format:

**--- STRATEGIC TREND BRIEF ---**

**Executive Summary:**
[A single, powerful sentence that summarizes the core insight a busy executive needs to know about this set of videos.]

**The Dominant Trend:**
[Name the primary recurring theme. In 1-2 sentences, explain the deeper psychological reason *why* this trend is resonating with audiences right now.]

**The 'Theme Stack' Uncovered:**
[Identify the most powerful combination of themes you observed. Explain how the creators are "stacking" a primary theme with a secondary one to create a unique angle. For example: "Creators are stacking the **'Life Hack'** theme (Primary) with **'Relatable Humor'** (Secondary) to make educational content feel less like a lecture and more like advice from a friend."]

**The Hidden Opportunity (Contrarian Take):**
[Based on this trend, what is the angle everyone else is missing? If this trend is 'A', what is the 'B' that no one is doing yet? Provide one actionable, non-obvious idea.]

**The Creative Director's Brief:**
[Provide a short, inspiring mission for a creative team. e.g., "This week's challenge: Create a Short that teaches one of our core concepts, but frame it as a 'Confession' or a 'Mistake We Once Made'."]


**--- TRANSCRIPTS & URLS FOR ANALYSIS ---**
{formatted_transcripts}
"""
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"An error occurred with the Gemini API: {e}")
            return None

    # This helper function is copied here to make this tab self-contained
    def get_transcript_for_single_video_tab5(url, whisper_model):
        """Downloads, transcribes, and cleans up a single video URL for Tab 5."""
        try:
            output_template = f"temp_audio_{hash(url)}"
            mp3_file_path = download_audio_from_youtube(url, output_template)
            if mp3_file_path and os.path.exists(mp3_file_path):
                result = whisper_model.transcribe(mp3_file_path, fp16=False)
                full_text = " ".join([segment['text'].strip() for segment in result['segments']])
                os.remove(mp3_file_path)
                return full_text
            return None
        except Exception:
            return None

    # --- UI for the new tab with dynamic inputs ---

    st.markdown("##### Video URLs for Trend Analysis")

    if 'num_trend_inputs' not in st.session_state:
        st.session_state.num_trend_inputs = 1

    def add_trend_input():
        if st.session_state.num_trend_inputs < 5:
            st.session_state.num_trend_inputs += 1

    def remove_trend_input():
        if st.session_state.num_trend_inputs > 1:
            key_to_remove = f"trend_url_{st.session_state.num_trend_inputs - 1}"
            if key_to_remove in st.session_state:
                st.session_state[key_to_remove] = ""
            st.session_state.num_trend_inputs -= 1

    for i in range(st.session_state.num_trend_inputs):
        st.text_input(f"URL for Short #{i+1}", key=f"trend_url_{i}")

    col1, col2, _ = st.columns([1, 2, 3])
    with col1:
        st.button("Add URL ➕", on_click=add_trend_input, use_container_width=True, key="add_trend_btn")
    with col2:
        st.button(
            "Remove Last ➖",
            on_click=remove_trend_input,
            use_container_width=True,
            disabled=(st.session_state.num_trend_inputs <= 1),
            key="remove_trend_btn"
        )

    st.markdown("---")

    model_choice_tab5 = st.selectbox(
        "Choose Gemini Model ",
        ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"),
        index=0,
        help="Pro is strongly recommended for this deep strategic analysis.",
        key="model_choice_tab5",
    )

    if st.button("Generate Strategic Brief", type="primary", key="find_common_themes_btn"):
        urls_to_process = []
        for i in range(st.session_state.num_trend_inputs):
            url_value = st.session_state.get(f"trend_url_{i}", "")
            if url_value and url_value.strip():
                urls_to_process.append(url_value)
        
        valid_urls = urls_to_process

        if valid_urls:
            with st.spinner("Loading Whisper model... (This happens once)"):
                whisper_model = whisper.load_model("base")

            progress_bar = st.progress(0, text="Starting transcription process...")
            all_transcripts_for_prompt = []

            for i, url in enumerate(valid_urls):
                progress_text = f"Processing video {i+1}/{len(valid_urls)}: Transcribing..."
                progress_bar.progress((i) / len(valid_urls), text=progress_text)
                transcript_text = get_transcript_for_single_video_tab5(url, whisper_model)

                if transcript_text:
                    formatted_string = f"**Video {i+1}: {url}**\n*Transcript:*\n{transcript_text}\n---"
                    all_transcripts_for_prompt.append(formatted_string)

            progress_bar.progress(1.0, text="All transcripts generated. Performing deep strategic analysis...")

            if all_transcripts_for_prompt:
                final_prompt_content = "\n".join(all_transcripts_for_prompt)
                analysis_result = analyze_common_themes_impressive(final_prompt_content, model_choice_tab5)
                progress_bar.empty()

                if analysis_result:
                    st.markdown("---")
                    st.subheader("📈 Strategic Trend Brief")
                    st.markdown(analysis_result)
                else:
                    st.error("Analysis failed. No response was received from the model.")
            else:
                progress_bar.empty()
                st.error("Could not generate transcripts for any of the provided URLs. Please check the URLs and try again.")
        else:
            st.warning("Please enter at least one YouTube Short URL to analyze.")