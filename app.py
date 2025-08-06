import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
from datetime import timedelta
import yt_dlp
import whisper
import pandas as pd
import tempfile
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Page Configuration and Minor Style Polish ---
st.set_page_config(layout="wide")
st.markdown("""
<style>
.stButton>button { border-radius: 20px; }
.st-expander { border: 1px solid #E2E8F0; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# --- Load API Keys ---
load_dotenv()
try:
    if "GEMINI_API_KEY" in os.environ:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        st.error("GEMINI_API_KEY not found in your .env file. Please add it to continue.")
        st.stop()
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please make sure your GEMINI_API_KEY is set correctly. Error: {e}")
    st.stop()

# --- Main App Title ---
st.title("YouTube Shorts Analyzer 🤖")
st.markdown("Your all-in-one toolkit for creating, reviewing, and diagnosing YouTube Shorts.")

# ##############################################################################
# --- GLOBAL HELPER FUNCTIONS ---
# All functions are defined here for better organization and code structure.
# ##############################################################################

@st.cache_data
def format_timestamp(seconds):
    """Converts seconds into SRT timestamp format (HH:MM:SS,ms)."""
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
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"{os.path.splitext(output_path)[0]}.mp3"
    except Exception as e:
        st.warning(f"Could not download audio from {url}. Skipping. Error: {e}")
        return None

@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model and caches it for the session."""
    st.info("Loading Whisper model for the first time... This might take a moment.")
    model = whisper.load_model("base")
    return model

@st.cache_data
def transcribe_audio_with_whisper(video_url):
    """Transcribes audio from a YouTube URL using a cached Whisper model."""
    audio_file = download_audio_from_youtube(video_url)
    if not audio_file or not os.path.exists(audio_file):
        st.error("Failed to download or locate the audio file.")
        return ""

    model = load_whisper_model()
    result = model.transcribe(audio_file, fp16=False)
    os.remove(audio_file)

    srt_content = []
    for i, segment in enumerate(result['segments'], 1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_content)

def transcribe_uploaded_video(video_file_bytes):
    """Saves uploaded video bytes to a temporary file and transcribes it."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file_bytes)
            temp_video_path = tmp.name

        model = load_whisper_model()
        result = model.transcribe(temp_video_path, fp16=False)
        os.remove(temp_video_path)
        full_text = " ".join([segment['text'].strip() for segment in result['segments']])
        return full_text
    except Exception as e:
        st.error(f"Failed during transcription: {e}")
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return None

def analyze_transcript_for_clips(chunk, model_name, num_clips):
    """Analyzes a transcript chunk to find viral clips."""
    prompt = f"""
    ROLE AND GOAL:
    You are an Expert YouTube Content Strategist and Video Editor, specializing in creating viral short-form content. Your primary goal is to analyze the provided video transcript and identify {num_clips} self-contained clips that can be edited into compelling YouTube Shorts.

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

def select_strongest_clips(clips_markdown, model_name, num_top_clips):
    """Analyzes a markdown table of clips to select the top N strongest ones."""
    prompt = f"""
    ROLE AND GOAL:
    You are an Executive Producer and Viral Content Expert based in Noida. Your goal is to review a pre-analyzed list of potential YouTube Shorts clips and select the top {num_top_clips} clips with the highest probability of going viral.

    CONTEXT:
    You have been given a list of clips identified by a content strategist. Your job is to perform the final selection, prioritizing clips that are most likely to achieve high engagement, watch time, and shareability.

    INSTRUCTIONS:
    1.  Carefully review each clip in the provided Markdown table.
    2.  Evaluate each clip based on its 'Hook', 'Strategy', and 'Why it works' columns.
    3.  Select the {num_top_clips} clips that are the most emotionally compelling, have the strongest narrative arc, or present the most surprising/valuable information.
    4.  You must then generate a new Markdown table containing ONLY your top selections.
    5.  This new table must include all the original columns, plus **three new columns at the beginning**:
        * **`Rank`**: Your ranking of the clip (1, 2, 3...).
        * **`Reason for Recommendation`**: A concise, powerful sentence explaining *why this specific clip was chosen over the others*. Focus on its viral potential or emotional impact.
        * **`Success Potential`**: Your expert rating (e.g., Very High, High, Medium) for the clip's likelihood of going viral.
    6.  Your final output must be ONLY this new Markdown table and nothing else.

    **List of Clips to Evaluate:**
    ---
    {clips_markdown}
    ---
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred during strong clip selection: {e}")
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
    11. **Emotional Trigger:** Use words that evoke strong feelings (e.g., “My Biggest Failure”)
    12. **Direct Question:** Ask a question the audience wants answered (e.g., “Is This The Future?”)
    13. **Surprising/Unexpected:** Surprise the audience with a surprising fact or statement (e.g., “I’m a Mentalist”)
    14. **Motivational:** Motivate the audience to take action (e.g., “Don’t Let Fear Hold You Back”)
    15. **Nostalgic/Sentimental:** Evoke nostalgia or sentimentality (e.g., “The Best Advice I Ever Got”)
    16. **Aspirational / Luxurious:** Inspire the audience to aspire to something (e.g., “The Best Way to Make Money”)
    17. **Intriguing/Mysterious:** Intrigue the audience with a mysterious or intriguing statement (e.g., “The Secret to Success”)
    18. **Urgent/Timely:** Create a sense of urgency or timeliness (e.g., “Do This Before It’s Too Late!”)

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
    - **Surprising/Unexpected:** Surprise the audience with a surprising fact or statement (e.g., "I’m a Mentalist")
    - **Motivational:** Motivate the audience to take action (e.g., "Don’t Let Fear Hold You Back")
    - **Nostalgic/Sentimental:** Evoke nostalgia or sentimentality (e.g., "The Best Advice I Ever Got")
    - **Aspirational / Luxurious:** Inspire the audience to aspire to something (e.g., "The Best Way to Make Money")
    - **Intriguing/Mysterious:** Intrigue the audience with a mysterious or intriguing statement (e.g., "The Secret to Success")
    - **Urgent/Timely:** Create a sense of urgency or timeliness (e.g., "Do This Before It’s Too Late!")

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

def get_pre_upload_feedback(short_transcript, title, header, audience, model_name, long_form_transcript=""):
    """Generates a comprehensive pre-upload analysis for a YouTube Short."""
    long_form_context_block = f"\n**Full Raw Video Transcript (for context):**\n---\n{long_form_transcript}\n---" if long_form_transcript else ""
    prompt = f"""
    ROLE AND GOAL:
    You are an expert YouTube Content Strategist and Analyst based in Noida, India, with a deep understanding of the YouTube Shorts algorithm as of August 2025. Your goal is to provide a comprehensive review of a short video, analyzing its content's viral potential and then providing actionable advice to maximize its performance.

    CORE TASK:
    Your primary job is to first analyze the core content for its inherent strengths and weaknesses. Then, based on that analysis, provide specific, actionable advice on how to package and edit the video to amplify its strengths and attract the target audience.
    Note: The short we are extarcting is a part of podcast. So, give the rating based on the podcast's content. So, ending can be a bit abrupt. So, do not consider the ending as a weakness.

    CONTEXT:
    - **Proposed Title:** "{title}"
    - **Target Audience:** "{audience}"
    - **The Final Short Clip's Transcript:**
    ---
    {short_transcript}
    ---
    {long_form_context_block}

    INSTRUCTIONS:
    Generate a detailed report with the following specific sections, in this exact order. Use clear headings and bullet points. Use LaTeX for the rating (e.g., `$8.5/10$`).

    **### Video Analysis & Review**
    Start with a brief paragraph summarizing the video's content and core message. Then, create two bulleted lists:
    - **Strengths:** Analyze why the content is compelling. (e.g., "Powerful Hook," "Credible Speaker," "Controversial Topic," etc.)
    - **Areas for Consideration:** Note any minor weaknesses or points to be aware of. (e.g., "Repurposed Formatting," "Abrupt End," etc.)

    **### Rating**
    Give the short a rating out of 10, using LaTeX format (`$X/10$`). Briefly justify the score based on your analysis above. What changes would you make to the video to improve its rating?
    
    ### Viral Potential & Expected Reach Tier
        Based on the content's quality, the target audience, and its potential for engagement, classify the video's potential reach into one of the following tiers. You must justify your choice of tier by connecting it directly to your video analysis.

        - **Tier 1: Niche Appeal (Potential: 1k - 10k views):** The content is solid for a specific, core audience but is unlikely to break out to a wider viewership.
        - **Tier 2: Strong Performer (Potential: 10k - 100k views):** The content has strong elements (e.g., a great hook, high relatability) that could help it perform above average and reach beyond your subscriber base.
        - **Tier 3: High Viral Potential (Potential: 100k - 1M+ views):** The content has multiple viral triggers (e.g., controversy, strong emotion, mass appeal) and is highly likely to be picked up and heavily promoted by the algorithm.

    **### Will This Work for a YouTube Audience?**
    Provide a clear "Yes," "No," or "It's possible" answer. Follow up with a paragraph explaining the **probability of success**. Explain *why* it will or will not work, referencing the YouTube algorithm, audience psychology, and engagement potential (comments, shares, etc.).

    **### Recommendations for Uploading:**
    This section is for practical advice.
    - **Title:** Provide 3 distinct, compelling title options that are SEO-friendly.
    - **Description:** Write a short, optimized YouTube description.
    - **Call to Action (CTA):** Suggest a specific question to ask in the description or a pinned comment to drive engagement.
    - **Hashtags:** Provide 3-5 relevant hashtags.

    **### Conclusion:**
    End with a short, encouraging summary paragraph about the video's potential.
    """
    try:
        model = genai.GenerativeModel(model_name)
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating feedback: {e}")
        return None

def parse_video_id(url):
    """Uses regular expressions to extract the YouTube video ID from various URL formats."""
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_video_details(api_key, video_id):
    """Fetches video details from the YouTube Data API v3."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.videos().list(part="snippet,statistics", id=video_id)
        response = request.execute()
        return response["items"][0] if response.get("items") else None
    except HttpError as e:
        st.error(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None
    except Exception as e:
        st.error(f"An error occurred calling the YouTube API: {e}")
        return None

def generate_post_upload_diagnosis(video_data, manual_data, model_name):
    """Calls Gemini with consolidated data to diagnose video performance."""
    prompt = f"""
    ROLE: You are an expert YouTube Shorts Content Strategist from Noida, India.

    TASK: Analyze the performance of the following YouTube Short and diagnose WHY it performed the way it did. Provide a sharp, actionable report.

    DATA:

    --- Automatic Data (from YouTube API) ---
    - Video Title: {video_data['snippet']['title']}
    - Published On: {video_data['snippet']['publishedAt']}
    - View Count: {video_data['statistics'].get('viewCount', 'N/A')}
    - Like Count: {video_data['statistics'].get('likeCount', 'N/A')}
    - Comment Count: {video_data['statistics'].get('commentCount', 'N/A')}

    --- Manual Data (from Creator's Studio) ---
    - Channel's Main Niche: {manual_data['niche']}
    - Average View Duration (AVD) %: {manual_data['avd_percentage']}%
    - Retention at 3-seconds %: {manual_data['hook_retention']}%

    DIAGNOSIS REPORT STRUCTURE:

    **### 1. Overall Performance Diagnosis**
    Based on the view count relative to the niche, give a one-line summary (e.g., "Underperformed," "Met Expectations," "Good Performance").

    **### 2. The Core Reason: The "Why"**
    This is the most important part. Directly correlate the retention data to the public stats.
    * **Analyze the Hook:** Based on the 3-second retention, was the hook strong or weak? Explain what this told the YouTube algorithm.
    * **Analyze the Content Body:** Based on the AVD, did the video hold attention after the hook? Or did people drop off?
    * **Example Diagnosis:** "The video underperformed primarily due to a weak hook. With only {manual_data['hook_retention']}% retention at 3 seconds, a majority of potential viewers swiped away immediately. This sent a strong negative signal to the algorithm, which stopped pushing the short to a wider audience, resulting in the low view count of {video_data['statistics'].get('viewCount', 'N/A')}."

    **### 3. Actionable Blueprint for Next Video**
    Provide 3 specific, numbered recommendations for the creator to implement in their NEXT Short to fix the identified problems.

    **### 4. Conclusion**
    End with a short, encouraging summary paragraph about the video's potential.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred generating the AI diagnosis: {e}")
        return None

def parse_markdown_table(markdown_text):
    """Parses a Markdown table into a Pandas DataFrame."""
    if not markdown_text: return pd.DataFrame()
    lines = [line for line in markdown_text.strip().split('\n') if '|' in line and '---' not in line]
    if not lines: return pd.DataFrame()
    header = [h.strip() for h in lines[0].split('|') if h.strip()]
    data = [[r.strip() for r in line.split('|') if r.strip()] for line in lines[1:]]
    data = [row for row in data if len(row) == len(header)]
    return pd.DataFrame(data, columns=header) if data else pd.DataFrame(columns=header)

# ##############################################################################
# --- STREAMLIT UI SECTION ---
# ##############################################################################

if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎬 Generate Clips", "🕵️‍♀️ Analyze Patterns", "✍️ Generate Titles",
    "💡 Headers Generator", "🚀 Pre-Upload Review", "📈 Post-Upload Diagnosis"
])

# --- TAB 1: GENERATE CLIPS ---
with tab1:
    st.header("Generate Clip Ideas from a Transcript")
    input_method = st.radio("Choose transcript source:", ("From YouTube URL", "From File Upload"), horizontal=True, key="clip_gen_source", on_change=lambda: st.session_state.update(raw_analysis_table=None, strong_clips_table=None))

    if input_method == "From YouTube URL":
        video_url = st.text_input("Enter YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=...")
        if st.button("Generate Transcript from URL", key="generate_transcript_btn"):
            if video_url:
                with st.spinner("Generating transcript... This may take a few minutes."):
                    st.session_state['transcript'] = transcribe_audio_with_whisper(video_url)
                if st.session_state['transcript']: st.success("Transcript generated successfully!")
                else: st.error("Failed to generate transcript. Please check the URL and try again.")
            else: st.warning("Please enter a YouTube URL.")
    else:
        uploaded_file = st.file_uploader("Choose a .txt or .srt file", type=['txt', 'srt'], key="clip_gen_uploader")
        if uploaded_file:
            st.session_state['transcript'] = uploaded_file.getvalue().decode("utf-8")
            st.success(f"Successfully loaded `{uploaded_file.name}`")

    if st.session_state.get('transcript'):
        st.markdown("---")
        st.info("✅ Transcript loaded. Configure and run the analysis below.")
        with st.expander("View Full Transcript"):
            st.text_area("Transcript Content", st.session_state['transcript'], height=300, key="transcript_display")
        
        st.markdown("---")
        st.subheader("Step 1: Find All Potential Clips")
        col1, col2 = st.columns(2)
        num_clips = col1.number_input("Number of clip ideas to find", 3, 50, 7, 1, key="num_clips_input")
        model_choice_t1 = col2.selectbox("Choose Model", ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"), key="model_choice_t1", help="Flash is faster. Pro provides more detailed analysis.")
        
        if st.button("✨ Find Clip Ideas", type="primary", key="analyze_btn_t1"):
            with st.spinner("Gemini is analyzing the transcript for clips..."):
                st.session_state['raw_analysis_table'] = analyze_transcript_for_clips(st.session_state.transcript, model_choice_t1, num_clips)
                st.session_state['strong_clips_table'] = None
        
        if st.session_state.get('raw_analysis_table'):
            st.success("Analysis Complete! Here are all potential clips.")
            df = parse_markdown_table(st.session_state['raw_analysis_table'])
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.download_button("Download Full Analysis as CSV", df.to_csv(index=False).encode('utf-8'), "full_clip_analysis.csv", "text/csv")
                
                st.markdown("---")
                st.subheader("Step 2: Select the Strongest Clips for Upload")
                num_top_clips = st.number_input("Number of top clips to select", 1, 10, 3, key="num_top_clips")
                if st.button("🔥 Find Strongest Clips", key="find_strongest_btn"):
                    with st.spinner("Producer AI is reviewing the list..."):
                        st.session_state['strong_clips_table'] = select_strongest_clips(st.session_state['raw_analysis_table'], model_choice_t1, num_top_clips)
            else:
                st.error("Analysis finished, but no valid clip data was found in the response.")

        if st.session_state.get('strong_clips_table'):
            st.markdown("---")
            st.subheader("🏆 Top Recommended Clips")
            strong_df = parse_markdown_table(st.session_state['strong_clips_table'])
            if not strong_df.empty:
                st.dataframe(strong_df, use_container_width=True)
                st.download_button("Download Top Clips as CSV", strong_df.to_csv(index=False).encode('utf-8'), "top_clips.csv", "text/csv")
            else:
                st.warning("The strong clip selection process did not return a valid table.")

# --- TAB 2: ANALYZE PATTERNS ---
with tab2:
    st.subheader("Deconstruct What Makes a Short Go Viral")
    st.markdown("Add up to 5 successful shorts. The tool will **transcribe each one** and then analyze the content to find common patterns and strategies.")
    
    if 'num_url_inputs' not in st.session_state: st.session_state.num_url_inputs = 1
    
    def add_url_input():
        if st.session_state.num_url_inputs < 5: st.session_state.num_url_inputs += 1
    def remove_url_input():
        if st.session_state.num_url_inputs > 1: st.session_state.num_url_inputs -= 1

    for i in range(st.session_state.num_url_inputs):
        st.text_input(f"URL for Short #{i+1}", key=f"url_{i}")
    
    col1, col2, _ = st.columns([1, 1, 4])
    col1.button("Add URL ➕", on_click=add_url_input, use_container_width=True)
    col2.button("Remove Last ➖", on_click=remove_url_input, use_container_width=True, disabled=(st.session_state.num_url_inputs <= 1))

    model_choice_t2 = st.selectbox("Choose Gemini Model", ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"), key="model_choice_tab2")

    if st.button("Find Patterns from Transcripts", type="primary", key="find_patterns_btn"):
        valid_urls = [st.session_state.get(f"url_{i}") for i in range(st.session_state.num_url_inputs) if st.session_state.get(f"url_{i}")]
        if valid_urls:
            all_transcripts_for_prompt = []
            progress_bar = st.progress(0, text="Starting transcriptions...")
            for i, url in enumerate(valid_urls):
                progress_text = f"Processing video {i+1}/{len(valid_urls)}..."
                progress_bar.progress((i) / len(valid_urls), text=progress_text)
                transcript_text = transcribe_audio_with_whisper(url)
                if transcript_text:
                    all_transcripts_for_prompt.append(f"--- TRANSCRIPT FOR VIDEO {i+1} ({url}) ---\n{transcript_text}\n--- END ---\n")
            
            progress_bar.progress(1.0, text="Analyzing with Gemini...")
            if all_transcripts_for_prompt:
                final_prompt_content = "\n".join(all_transcripts_for_prompt)
                analysis_result = analyze_viral_patterns(final_prompt_content, model_choice_t2)
                st.markdown("---")
                st.subheader("🔬 Analysis Report")
                st.markdown(analysis_result)
            else:
                st.error("Could not generate any transcripts. Please check the URLs.")
            progress_bar.empty()
        else:
            st.warning("Please enter at least one YouTube Short URL.")

# --- TAB 3: GENERATE TITLES ---
with tab3:
    st.subheader("✍️ Generate High-Performing Titles")
    transcript_input_t3 = st.text_area("Paste your full Short transcript here", height=200, key="transcript_input_tab3")
    col1, col2 = st.columns(2)
    audience = col1.text_input("Target Audience", placeholder="e.g., Beginner freelancers")
    tone = col1.selectbox("Desired Tone", ("Educational", "Bold & Controversial", "Inspirational", "Humorous", "Relatable", "Urgent/Timely", "Intriguing/Mysterious", "Motivational", "Nostalgic/Sentimental", "Aspirational / Luxurious", "Surprising/Unexpected"))
    takeaway = col2.text_input("Main Takeaway or Message", placeholder="e.g., Build a personal brand to earn more")
    model_choice_t3 = st.selectbox("Choose Gemini Model", ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"), key="model_choice_tab3")
    
    if st.button("💡 Generate Titles", type="primary", key="suggest_titles_btn"):
        if transcript_input_t3 and audience and takeaway:
            with st.spinner("Gemini is crafting strategic titles..."):
                titles = generate_shorts_titles(transcript_input_t3, audience, takeaway, tone, model_choice_t3)
                if titles:
                    st.markdown("---")
                    st.subheader("🔥 Suggested Titles (with Strategy)")
                    st.markdown(titles)
                else:
                    st.error("Title generation failed.")
        else:
            st.warning("Please fill in all fields for the best results.")

# --- TAB 4: HEADERS GENERATOR ---
with tab4:
    st.subheader("💡 Headers Generator")
    transcript_input_t4 = st.text_area("Paste your full Short transcript here", height=200, key="transcript_input_tab4")
    core_message = st.text_input("Core Message", placeholder="e.g., The importance of networking", key="core_message_tab4")
    peak_moment = st.text_input("Peak Moment / Hook", placeholder="e.g., 'I got my dream job from one conversation.'", key="peak_moment_tab4")
    audience_tone = st.text_input("Audience & Tone", placeholder="e.g., For young professionals; inspirational.", key="audience_tone_tab4")
    model_choice_t4 = st.selectbox("Choose Gemini Model", ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"), key="model_choice_tab4")

    if st.button("Generate Headers", type="primary", key="generate_headers_btn"):
        if all([transcript_input_t4, core_message, peak_moment, audience_tone]):
            with st.spinner("Gemini is generating powerful headers..."):
                headers = generate_headers(transcript_input_t4, core_message, peak_moment, audience_tone, model_choice_t4)
                if headers:
                    st.markdown("---")
                    st.subheader("🚀 Suggested Thumbnail Headers")
                    st.markdown(headers)
                else:
                    st.error("Header generation failed.")
        else:
            st.warning("Please fill in all fields for the best results.")

# --- TAB 5: PRE-UPLOAD REVIEW ---
with tab5:
    st.subheader("🚀 Get a Final Pre-Upload Review")
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'], key="review_video_uploader")
    
    if uploaded_video:
        # Create a two-column layout: Form on left, Video on right
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.markdown("### 📝 Video Context & Analysis")
            review_title = st.text_input("Proposed Title", placeholder="e.g., DON'T Make This Freelance Mistake!", key="review_title")
            review_header = st.text_input("Thumbnail Header Text", placeholder="e.g., THE #1 MISTAKE", key="review_header")
            review_audience = st.text_input("Target Audience", placeholder="e.g., Aspiring freelancers in India", key="review_audience")
            review_model_choice = st.selectbox("Choose Gemini Model", ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"), key="review_model_choice")

            if st.button("🔬 Analyze and Review Short", type="primary", key="get_review_btn"):
                if review_title and review_audience:
                    with st.spinner("Transcribing video..."):
                        video_bytes = uploaded_video.getvalue()
                        review_transcript = transcribe_uploaded_video(video_bytes)
                    if review_transcript:
                        st.success("Transcription complete! Now getting feedback...")
                        with st.spinner("Your personal strategist is reviewing the content..."):
                            feedback = get_pre_upload_feedback(review_transcript, review_title, review_header, review_audience, review_model_choice)
                        if feedback:
                            st.markdown("---")
                            st.subheader("📋 Feedback Report")
                            st.markdown(feedback)
                        else:
                            st.error("The review could not be generated.")
                    else:
                        st.error("Could not transcribe the video.")
                else:
                    st.warning("Please fill in the Title and Audience fields.")
        
        with right_col:
            st.markdown("### 📹 Video Preview")
            st.video(uploaded_video)
            st.caption(f"📁 {uploaded_video.name}")

# --- TAB 6: POST-UPLOAD DIAGNOSIS ---
# Initialize session state variables
if 'diag_url_input' not in st.session_state:
    st.session_state.diag_url_input = ""
if 'diag_video_data' not in st.session_state:
    st.session_state.diag_video_data = None
if 'diag_show_form' not in st.session_state:
    st.session_state.diag_show_form = False
if 'diag_final_report' not in st.session_state:
    st.session_state.diag_final_report = None
if 'avd_value' not in st.session_state:
    st.session_state.avd_value = 0
if 'hook_value' not in st.session_state:
    st.session_state.hook_value = 0
if 'niche_value' not in st.session_state:
    st.session_state.niche_value = ""


with tab6:
    st.subheader("📈 Post-Upload Performance Diagnosis")
    st.markdown("Find out *why* your Short performed the way it did. This tool combines public API data with your private Studio analytics for a precise diagnosis.")

    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

    # --- Fetch Data Function ---
    def fetch_video_data():
        url = st.session_state.diag_url_input
        if not url:
            st.error("Please enter a YouTube URL.")
            return
        
        video_id = parse_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL.")
            return
        
        try:
            with st.spinner("Fetching data from YouTube API..."):
                video_data = get_video_details(YOUTUBE_API_KEY, video_id)
                if video_data:
                    st.session_state.diag_video_data = video_data
                    st.session_state.diag_show_form = True
                    st.session_state.diag_final_report = None
                    # Reset form values for new video
                    st.session_state.avd_value = 0
                    st.session_state.hook_value = 0
                    st.session_state.niche_value = ""
                    st.rerun()
                else:
                    st.error("Could not fetch video details. Check URL or API Key.")
        except Exception as e:
            st.error(f"Error fetching video data: {str(e)}")

    # --- UI Components ---
    if not YOUTUBE_API_KEY:
        st.error("YOUTUBE_API_KEY not found in your .env file. Please add it to use this feature.")
    else:
        # URL Input Section
        st.text_input(
            "Enter the YouTube Short URL you want to diagnose:",
            key="diag_url_input"
        )
        
        # Fetch Button
        if st.button("Fetch Public Data", key="fetch_data_btn"):
            fetch_video_data()

        # Video Data Display
        if st.session_state.diag_show_form and st.session_state.diag_video_data:
            video_data = st.session_state.diag_video_data
            st.markdown("---")
            st.subheader(f"Diagnosing: \"{video_data['snippet']['title']}\"")
            
            # Metrics Display
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Views", f"{int(video_data['statistics'].get('viewCount', 0)):,}")
            col2.metric("Likes", f"{int(video_data['statistics'].get('likeCount', 0)):,}")
            col3.metric("Comments", f"{int(video_data['statistics'].get('commentCount', 0)):,}")
            
            # Published Date
            published_date = video_data['snippet'].get('publishedAt', '')
            if published_date:
                from datetime import datetime
                try:
                    # Parse the ISO 8601 date format from YouTube API
                    date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime("%B %d, %Y")
                    col4.metric("Published", formatted_date)
                except:
                    col4.metric("Published", "Unknown")
            else:
                col4.metric("Published", "Unknown")

            st.info("Go to this video's analytics in your YouTube Studio and provide the following crucial metrics:")

            # Diagnosis Form Fields (Outside of form for better state persistence)
            st.markdown("### 📊 Enter Your Analytics Data")
            
            # Form fields with proper state management
            avd_input = st.text_input(
                "Average View Duration (%)", 
                help="From 'Audience retention' card. Enter a number (e.g., 14 or 0.14)",
                key="avd_field"
            )
            
            # Convert text input to number with validation
            try:
                avd = float(avd_input) if avd_input else 0
            except ValueError:
                avd = 0
                st.error("Please enter a valid number for Average View Duration")
            
            hook_input = st.text_input(
                "Retention at 3-seconds (%)", 
                help="Hover over start of retention graph. Enter a number (e.g., 85, 120, 150). Can exceed 100% for very short videos.",
                key="hook_field"
            )
            
            # Convert text input to number with validation
            try:
                hook = float(hook_input) if hook_input else 0
            except ValueError:
                hook = 0
                st.error("Please enter a valid number for Retention")
            
            niche = st.text_input(
                "Channel's primary niche?", 
                help="e.g., Tech, Comedy, Finance",
                key="niche_field"
            )

            # Diagnose Button
            if st.button("🔬 Diagnose My Short", key="diagnose_btn"):
                # Generate diagnosis
                manual_data = {
                    "avd_percentage": avd, 
                    "hook_retention": hook, 
                    "niche": niche
                }
                
                with st.spinner("Strategist is diagnosing the data..."):
                    model_choice = "gemini-1.5-pro-latest"
                    report = generate_post_upload_diagnosis(video_data, manual_data, model_choice)
                    st.session_state.diag_final_report = report
                    # Keep the form visible after report generation
                    st.session_state.diag_show_form = True
                    st.rerun()

        # Display Final Report
        if st.session_state.diag_final_report:
            st.markdown("---")
            st.subheader("📋 Diagnosis & Strategy Report")
            st.markdown(st.session_state.diag_final_report)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Report", key="clear_report_btn"):
                    del st.session_state.diag_final_report
                    st.rerun()
            with col2:
                if st.button("Generate New Report", key="new_report_btn"):
                    del st.session_state.diag_final_report
                    st.rerun()
