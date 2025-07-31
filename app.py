
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
from datetime import timedelta
import yt_dlp
import whisper
import pandas as pd

# --- Page Configuration and Minor Style Polish ---
st.set_page_config(layout="wide")
st.markdown("""
<style>
.stButton>button { border-radius: 20px; }
.st-expander { border: 1px solid #E2E8F0; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


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

st.title("YouTube Shorts Analyzer 🤖")
st.markdown("Analyze a YouTube video or upload a transcript (`.txt`, `.srt`) to auto-generate short-worthy clips.")

# --------- Helper Functions (Your original functions are unchanged) ---------

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

def download_audio_from_youtube(url, output_path="temp_audio"):
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
    if not audio_file or not os.path.exists(audio_file):
        st.error("Failed to download or locate the audio file.")
        return ""

    st.info("Whisper model loaded. Starting transcription... (This can take a few minutes for long videos)")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, fp16=False)
    os.remove(audio_file) # Clean up the audio file after transcription

    st.success("Transcription complete. Formatting SRT...")
    srt_content = []
    # Iterate through each text segment detected by Whisper
    for i, segment in enumerate(result['segments'], 1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

    return "\n".join(srt_content)

def analyze_transcript_for_clips(chunk, model_name, num_clips):
    """
    Analyzes a transcript chunk to find viral clips using a highly-detailed, specific prompt
    designed to replicate the high-quality output from the user's Google Sheet.
    """
    prompt = f"""
    ROLE AND GOAL:
    You are an Expert YouTube Content Strategist and Video Editor, specializing in creating viral short-form content. Your primary goal is to analyze the provided video transcript and identify at least {num_clips} self-contained clips that can be edited into compelling YouTube Shorts.

    CONTEXT:
    Your analysis must be sharp, insightful, and directly actionable. The output must be a single Markdown table, and nothing else.

    INSTRUCTIONS:
    1.  Read through the entire transcript to find segments that are emotionally resonant, surprising, or highly valuable.
    2.  Focus on clips that are ideally 45-60 seconds long.
    3.  For each identified clip, provide a detailed breakdown formatted EXACTLY into the 7 columns specified below. Adherence to this format is crucial.

    **COLUMN DEFINITIONS AND EXAMPLES:**
    1.  **`Timestamp`**: The start and end time of the clip from the transcript.
    2.  **`Duration`**: The calculated raw duration of the clip in seconds.
    3.  **`Strategy`**: A one line explaination how that short can be made so it can give potential (for example: "Start with the direct question "Do you actually get people to fear you?" and emphasize the secretive life aspect.").
    4.  **`Why it works`**: An explaination why that suggested clip will work (for example, "Fear of Mentalists: Highlights the intriguing and slightly intimidating nature of being a mentalist. Intriguing and personal.").
    5.  **`Pacing/Editing Notes`**: Specific, actionable editing advice.
    6.  **`Hook / Opening Line`**: The exact first sentence from the transcript (for example, ""People are scared of me because I'm a mentalist... I live a very secretive life. #Mentalist #Magic")
    7.  **`Success Potential`**: A rating (Very High, High, Medium) followed by a justification (for example, "High: Creates curiosity, plays on common perceptions of mentalists, and includes a strong, somewhat mysterious personal statement.")

    **FINAL RULE:**
    Sort the final table by the `Timestamp` column in ascending order (from earliest to latest).

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
        return pd.DataFrame()
    lines = markdown_text.strip().split('\n')
    table_lines = [line for line in lines if '|' in line and '---' not in line]
    if not table_lines:
        return pd.DataFrame()
    header = [h.strip() for h in table_lines[0].split('|') if h.strip()]
    data = []
    for line in table_lines[1:]:
        row = [r.strip() for r in line.split('|') if r.strip()]
        if len(row) == len(header):
            data.append(row)
    if not data:
        return pd.DataFrame(columns=header)
    df = pd.DataFrame(data, columns=header)
    return df

# --------- Streamlit UI Section ---------

# Initialize session state to hold the transcript
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""

# --- TABS DEFINITION (CHANGED) ---
# The old 5 tabs are now 4, with the first two merged.
# --- TABS DEFINITION (CHANGED) ---
# The "Trend Analysis" tab is now replaced with "Headers Generator"
tab1, tab2, tab3, tab4 = st.tabs(["🎬 Generate Clips", "🕵️‍♀️ Analyze Patterns", "✍️ Generate Titles", "💡 Headers Generator"])

# --- TAB 1: GENERATE CLIPS (MERGED) ---
# --- TAB 1: GENERATE CLIPS ---
with tab1:
    st.header("Generate Clip Ideas from a Transcript")
    
    # Let the user choose how to provide the transcript
    input_method = st.radio(
        "Choose transcript source:",
        ("From YouTube URL", "From File Upload"),
        horizontal=True,
        key="clip_gen_source"
    )

    # --- UI for YouTube URL Input ---
    if input_method == "From YouTube URL":
        video_url = st.text_input("Enter YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=...")
        if st.button("Generate Transcript from URL", key="generate_transcript_btn"):
            if video_url:
                # Call the transcription function
                transcript_text = transcribe_audio_with_whisper(video_url)
                if transcript_text:
                    st.session_state['transcript'] = transcript_text
                    st.success("Transcript generated successfully!")
                    st.rerun() # Refresh the app to show the analysis section
            else:
                st.warning("Please enter a YouTube URL.")

    # --- UI for File Upload Input ---
    else: # input_method == "From File Upload"
        uploaded_file = st.file_uploader("Choose a .txt or .srt file", type=['txt', 'srt'], key="clip_gen_uploader")
        if uploaded_file:
            st.session_state['transcript'] = uploaded_file.getvalue().decode("utf-8")
            st.success(f"Successfully loaded `{uploaded_file.name}`")
            # The UI will update automatically below once the session state is set.

    # --- Analysis Section (Appears only after a transcript is loaded) ---
    if st.session_state.get('transcript'):
        st.markdown("---")
        st.info("✅ Transcript loaded. Configure the analysis below and click the 'Find Clip Ideas' button.")
        
        with st.expander("View Full Transcript"):
            st.text_area("Transcript Content", st.session_state['transcript'], height=300, key="transcript_display")

        st.markdown("---")
        st.subheader("Analysis Configuration")

        col1, col2 = st.columns(2)
        with col1:
            num_clips = st.number_input("Number of clip ideas", min_value=3, max_value=50, value=7, step=1, key="num_clips_input")
        with col2:
            model_choice_t1 = st.selectbox(
                "Choose Model",
                ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"),
                key="model_choice_t1",
                help="Flash is faster, Pro provides more detailed analysis."
            )
        
        if st.button("✨ Find Clip Ideas", type="primary", key="analyze_btn_t1"):
            with st.spinner("Gemini is analyzing the transcript for clips..."):
                raw_table = analyze_transcript_for_clips(st.session_state.transcript, model_choice_t1, num_clips)
            
            if raw_table:
                df = parse_markdown_table(raw_table)
                if not df.empty:
                    st.success("Analysis Complete!")
                    st.dataframe(df, use_container_width=True)
                    # Offer CSV download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Analysis as CSV",
                        data=csv,
                        file_name="clip_analysis.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Analysis finished, but no valid clip data was found in the response.")
            else:
                st.error("Analysis failed. No response was received from the model.")

# --- TAB 2: ANALYZE PATTERNS (Previously Tab 3) ---
with tab2:
    st.subheader("Deconstruct What Makes a Short Go Viral")
    st.markdown("Add up to 5 successful shorts. The tool will **transcribe each one** and then analyze the content to find common patterns and strategies.")

    def get_transcript_for_single_video(url, whisper_model):
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
        prompt = f"""
        Analyzes a string of transcripts to find viral patterns."""
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

    st.markdown("##### Video URLs to Analyze")
    if 'num_url_inputs' not in st.session_state:
        st.session_state.num_url_inputs = 1

    def add_url_input():
        if st.session_state.num_url_inputs < 5:
            st.session_state.num_url_inputs += 1

    def remove_url_input():
        if st.session_state.num_url_inputs > 1:
            key_to_remove = f"url_{st.session_state.num_url_inputs - 1}"
            if key_to_remove in st.session_state:
                st.session_state[key_to_remove] = ""
            st.session_state.num_url_inputs -= 1

    for i in range(st.session_state.num_url_inputs):
        st.text_input(f"URL for Short #{i+1}", key=f"url_{i}")

    col1, col2, _ = st.columns([1, 2, 3])
    with col1:
        st.button("Add URL ➕", on_click=add_url_input, use_container_width=True, key="add_pattern_url")
    with col2:
        st.button("Remove Last ➖", on_click=remove_url_input, use_container_width=True, disabled=(st.session_state.num_url_inputs <= 1), key="remove_pattern_url")

    st.markdown("---")
    model_choice_tab2 = st.selectbox("Choose Gemini Model", ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"), index=0, help="Flash is faster, Pro is more powerful.", key="model_choice_tab2")

    if st.button("Find Patterns from Transcripts", type="primary", key="find_patterns_btn"):
        urls_to_process = [st.session_state.get(f"url_{i}", "") for i in range(st.session_state.num_url_inputs)]
        valid_urls = [url for url in urls_to_process if url and url.strip()]
        if valid_urls:
            with st.spinner("Loading Whisper model..."):
                whisper_model = whisper.load_model("base")
            progress_bar = st.progress(0, text="Starting transcriptions...")
            all_transcripts_for_prompt = []
            for i, url in enumerate(valid_urls):
                progress_bar.progress((i) / len(valid_urls), text=f"Processing video {i+1}/{len(valid_urls)}...")
                transcript_text = get_transcript_for_single_video(url, whisper_model)
                if transcript_text:
                    all_transcripts_for_prompt.append(f"--- TRANSCRIPT FOR VIDEO {i+1} ({url}) ---\n{transcript_text}\n--- END ---\n")
            progress_bar.progress(1.0, text="Analyzing with Gemini...")
            if all_transcripts_for_prompt:
                final_prompt_content = "\n".join(all_transcripts_for_prompt)
                analysis_result = analyze_viral_patterns(final_prompt_content, model_choice_tab2)
                progress_bar.empty()
                st.markdown("---")
                st.subheader("🔬 Analysis Report")
                st.markdown(analysis_result)
            else:
                progress_bar.empty()
                st.error("Could not generate transcripts. Please check the URLs.")
        else:
            st.warning("Please enter at least one YouTube Short URL.")

# --- TAB 3: GENERATE TITLES (Previously Tab 4) ---
def generate_shorts_titles(transcript, audience, takeaway, tone, model_name):
    """Generates Shorts titles using an advanced framework and user-provided context."""
    prompt = f"""
    ROLE AND GOAL:
    You are an expert viral content strategist based in Noida, specializing in writing high-engagement, "scroll-stopping" titles for YouTube Shorts. Your goal is to generate 15-20 powerful titles based on the provided transcript and context.

    CONTEXT:
    - **Target Audience:** {audience}
    - **Main Takeaway/Message:** {takeaway}
    - **Desired Tone/Style:** {tone}

    ADVANCED TITLE FRAMEWORKS TO USE:
    [Shock Value], [Big Promise], [FOMO], [Secret], [Transformation], [Listicle], [Common Mistake], [Pain Point], [Curiosity Gap]

    INSTRUCTIONS:
    1.  Read the entire transcript and the context provided.
    2.  Generate 15-20 titles, applying a mix of the frameworks above.
    3.  **Your final output must be ONLY a Markdown table with two columns: "Strategy" and "Suggested Title". Do not include any other text or explanation.**

    EXAMPLE OUTPUT:
    | Strategy | Suggested Title |
    |---|---|
    | [Shock Value] | You're Losing Money by Not Doing This |
    | [Pain Point] | Tired of Clients Ghosting You? |
    | [Listicle] | 3 Tools That Changed My Freelance Life |


    --- TRANSCRIPT FOR ANALYSIS ---
    {transcript}
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return None
    
    # --- TAB 3: GENERATE TITLES (REVAMPED) ---
# --- TAB 3: GENERATE TITLES (REVAMPED WITH NEW PARAMETERS) ---
with tab3:
    st.subheader("✍️ Generate High-Performing Titles")
    st.markdown("Provide context to get strategic, scroll-stopping titles based on proven frameworks.")
    st.markdown("---")

    def generate_shorts_titles(transcript, audience, takeaway, tone, model_name):
        """Generates Shorts titles using an advanced framework and user-provided context."""
        prompt = f"""
        ROLE AND GOAL:
        You are an expert viral content strategist based in Noida, specializing in writing high-engagement, "scroll-stopping" titles for YouTube Shorts. Your goal is to generate 15-20 powerful titles based on the provided transcript and context, using the specific strategies listed below.

        CONTEXT:
        - **Target Audience:** {audience}
        - **Main Takeaway/Message:** {takeaway}
        - **Desired Tone/Style:** {tone}

        TITLE GENERATION STRATEGIES TO USE:
        1.  **Punchline / Reveal:** Drop a surprising or bold fact early (e.g., “50% of My Income Comes from Social Media?!”)
        2.  **Controversial Opinion:** Spark debate or strong reactions (e.g., “Freelancing Is Dead – Here's Why”)
        3.  **Clear Outcome / Result:** Show tangible success or transformation (e.g., “How I Made ₹10L in 6 Months Freelancing”)
        4.  **Problem Statement:** Call out a relatable pain point (e.g., “Struggling to Get Clients? Watch This.”)
        5.  **Contradiction / Irony:** Challenge common assumptions (e.g., “Clients Pay Less Than My Instagram Posts Do”)
        6.  **Curiosity Hook:** Create an information gap people want to close (e.g., “I Did THIS Before Every Big Client Deal”)
        7.  **Secret / Hidden Strategy:** Tease insider tips or unknown hacks (e.g., “The Tool No Freelancer Talks About”)
        8.  **Urgency / FOMO:** Build pressure to act now or miss out (e.g., “Do This Before It’s Too Late!”)
        9.  **List or Framework:** Use structure like steps, tips, or tools (e.g., “3 Steps to Build a High-Income Side Hustle”)
        10. **Transformation / Before-After:** Show clear change over time or effort (e.g., “From ₹0 to ₹1L/Month in 90 Days”)

        INSTRUCTIONS:
        Your final output must be ONLY a Markdown table with two columns: "Strategy" and "Suggested Title". Do not include any other text, explanation, or introduction.

        EXAMPLE OUTPUT:
        | Strategy | Suggested Title |
        |---|---|
        | Punchline / Reveal | My Biggest Client Was a Scam |
        | Problem Statement | Tired of Unpaid Invoices? |
        | List or Framework | 3 Tools That Saved My Business |


        --- TRANSCRIPT FOR ANALYSIS ---
        {transcript}
        """
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"An error occurred with the Gemini API: {e}")
            return None

    st.subheader("Step 1: Provide the Transcript")
    transcript_input = st.text_area(
        "Paste your full Short transcript here",
        height=200,
        key="transcript_input_tab3",
        placeholder="e.g., Today, I'm going to show you the biggest mistake freelancers make..."
    )

    st.markdown("---")
    st.subheader("Step 2: Provide Context for Better Titles")
    
    col1, col2 = st.columns(2)
    with col1:
        audience = st.text_input("Target Audience", placeholder="e.g., Beginner freelancers, students")
        tone = st.selectbox(
            "Desired Tone / Style",
            ("Educational", "Bold & Controversial", "Inspirational", "Humorous", "Relatable")
        )
    with col2:
        takeaway = st.text_input("Main Takeaway or Message", placeholder="e.g., Build a personal brand to earn more")

    st.markdown("---")
    model_choice_tab3 = st.selectbox(
        "Choose Gemini Model",
        ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"),
        index=0,
        help="Pro is recommended for creative and strategic tasks.",
        key="model_choice_tab3"
    )

    if st.button("💡 Generate Titles", type="primary", key="suggest_titles_btn"):
        if not transcript_input.strip():
            st.warning("Please paste a transcript first.")
        elif not audience.strip() or not takeaway.strip():
            st.warning("Please provide the Target Audience and Main Takeaway for best results.")
        else:
            with st.spinner("Gemini is crafting strategic titles..."):
                suggested_titles = generate_shorts_titles(
                    transcript=transcript_input,
                    audience=audience,
                    takeaway=takeaway,
                    tone=tone,
                    model_name=model_choice_tab3
                )
            if suggested_titles:
                st.markdown("---")
                st.subheader("🔥 Suggested Titles (with Strategy)")
                st.markdown(suggested_titles)
            else:
                st.error("Title generation failed. No response was received from the model.")

# --- TAB 4: TREND ANALYSIS (Previously Tab 5) ---
# --- TAB 4: HEADERS GENERATOR (REPLACES TREND ANALYSIS) ---
with tab4:
    st.subheader("💡 Headers Generator")
    st.markdown("Generate powerful, context-aware thumbnail headers based on your video's transcript and core message.")
    st.markdown("---")

    def generate_headers(transcript, core_message, peak_moment, audience_tone, model_name):
        """Generates thumbnail headers using the context-aware framework."""
        prompt = f"""
        ROLE AND GOAL:
        You are an expert thumbnail text generator. Your sole focus is to create 10-15 short, powerful, and high-impact headers for a YouTube Short thumbnail based on the provided transcript and context. The headers should be 3-7 words max.

        CONTEXT BRIEF:
        - **Core Message:** {core_message}
        - **Peak Moment / Hook:** {peak_moment}
        - **Audience & Tone:** {audience_tone}

        HEADER STRATEGIES TO USE:
        - **Problem/Solution:** State the problem directly (e.g., "Your Code Is Buggy?")
        - **Curiosity Gap:** Create a mystery (e.g., "The Secret No One Tells You")
        - **Bold Statement:** Make a controversial or strong claim (e.g., "AI Will Replace You")
        - **Result-Oriented:** Promise a clear outcome (e.g., "Write Perfect Code, Every Time")
        - **Emotional Trigger:** Use words that evoke strong feelings (e.g., "My Biggest Failure")
        - **Direct Question:** Ask a question the audience wants answered (e.g., "Is This The Future?")

        INSTRUCTIONS:
        Your final output must be ONLY a Markdown table with two columns: "Strategy" and "Suggested Header". Do not include any other text or explanation.

        --- TRANSCRIPT FOR ANALYSIS ---
        {transcript}
        """
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"An error occurred with the Gemini API: {e}")
            return None

    st.subheader("Step 1: Provide the Transcript")
    transcript_input_tab4 = st.text_area(
        "Paste your full Short transcript here",
        height=200,
        key="transcript_input_tab4",
        placeholder="Paste the full text from your video here..."
    )

    st.markdown("---")
    st.subheader("Step 2: Provide the Context Brief")
    
    core_message = st.text_input("Core Message", placeholder="e.g., The importance of networking for career growth.", key="core_message_tab4")
    peak_moment = st.text_input("Peak Moment / Hook", placeholder="e.g., 'I got my dream job from a single conversation.'", key="peak_moment_tab4")
    audience_tone = st.text_input("Audience & Tone", placeholder="e.g., For young professionals; tone is inspirational and actionable.", key="audience_tone_tab4")

    st.markdown("---")
    model_choice_tab4 = st.selectbox(
        "Choose Gemini Model",
        ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"),
        index=0,
        help="Pro is recommended for creative and strategic tasks.",
        key="model_choice_tab4"
    )

    if st.button("Generate Headers", type="primary", key="generate_headers_btn"):
        if not transcript_input_tab4.strip():
            st.warning("Please paste a transcript first.")
        elif not core_message.strip() or not peak_moment.strip() or not audience_tone.strip():
            st.warning("Please provide all three points of the Context Brief for the best results.")
        else:
            with st.spinner("Gemini is generating powerful headers..."):
                suggested_headers = generate_headers(
                    transcript=transcript_input_tab4,
                    core_message=core_message,
                    peak_moment=peak_moment,
                    audience_tone=audience_tone,
                    model_name=model_choice_tab4
                )
            if suggested_headers:
                st.markdown("---")
                st.subheader("🚀 Suggested Thumbnail Headers")
                st.markdown(suggested_headers)
            else:
                st.error("Header generation failed. No response was received from the model.")