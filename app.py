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
import json
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
st.markdown("Create your content plan in minutes.")

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

def handle_youtube_format_issues(url):
    """Provides specific guidance for common YouTube format issues."""
    st.error("🚫 **YouTube Format Issue Detected**")
    st.markdown("""
    **The Problem:** This video has format restrictions that prevent audio extraction.
    
    **Common Causes:**
    - Video is part of a playlist with restricted formats
    - Video has age restrictions or region blocks
    - YouTube has changed their format availability
    - Video uses newer codecs not supported by yt-dlp
    
    **Solutions to Try:**
    1. **Use File Upload Instead** - Upload the video file directly (recommended)
    2. **Try Different Video** - Test with another YouTube video
    3. **Check Video Settings** - Ensure the video is public and accessible
    4. **Update yt-dlp** - Run: `pip install --upgrade yt-dlp`
    """)
    
    st.info("💡 **Quick Fix:** Switch to 'From File Upload' and upload the video file directly!")

def download_audio_from_youtube(url, output_path="temp_audio"):
    """Downloads the best quality audio from a YouTube URL and saves it as an MP3."""
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'writesubtitles': False,
        'writeautomaticsub': False,
        'ignoreerrors': False,
        'nocheckcertificate': True,
        'geo_bypass': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, try to get available formats to debug
            try:
                info = ydl.extract_info(url, download=False)
                available_formats = info.get('formats', [])
                audio_formats = [f for f in available_formats if f.get('acodec') != 'none']
                
                if not audio_formats:
                    st.warning("No audio formats available for this video. Trying alternative approach...")
                    # Fallback to more permissive format selection
                    ydl_opts['format'] = 'bestaudio/best'
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                        ydl2.download([url])
                else:
                    # Use the original options with available formats
                    ydl.download([url])
                    
            except Exception as format_error:
                st.warning(f"Format detection failed, trying fallback: {format_error}")
                # Ultimate fallback - try any available format
                ydl_opts['format'] = 'best'
                with yt_dlp.YoutubeDL(ydl_opts) as ydl3:
                    ydl3.download([url])
        
        # Check if the file was created
        expected_file = f"{os.path.splitext(output_path)[0]}.mp3"
        if os.path.exists(expected_file):
            return expected_file
        else:
            # Try alternative extensions
            for ext in ['.m4a', '.webm', '.mp3']:
                alt_file = f"{os.path.splitext(output_path)[0]}{ext}"
                if os.path.exists(alt_file):
                    return alt_file
            
            st.error("Download completed but file not found. This may be a format compatibility issue.")
            return None
            
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for specific format errors
        if "requested format is not available" in error_msg or "format" in error_msg:
            handle_youtube_format_issues(url)
        else:
            st.warning(f"Could not download audio from {url}. Error: {e}")
            st.info("💡 **Troubleshooting Tips:**")
            st.markdown("""
            - **Try a different video** - Some videos have restricted formats
            - **Check video availability** - Make sure the video is public and accessible
            - **Use file upload** - If URL doesn't work, try uploading the video file directly
            """)
        
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

        # --- NEW HELPER FUNCTION ---
def generate_context_from_transcript(transcript, model_name):
    """Analyzes a transcript to generate a structured JSON context summary."""
    prompt = f"""
    ROLE: You are a senior YouTube content strategist and analyst.
    TASK: Analyze the provided video transcript and distill its core components into a structured JSON object.

    INSTRUCTIONS:
    1. Read the entire transcript to understand the topic, message, and emotional undercurrent.
    2. Extract the information and format it precisely into the following JSON keys:
        - "core_message": (String) The single most important takeaway or argument of the clip in one sentence.
        - "emotional_tone": (String) Describe the primary emotion of the speaker (e.g., "Hopeful Vindication", "Frustration", "Excitement").
        - "key_topics": (List of strings) A list of the main topics discussed.
        - "target_audience": (String) Who is this video for? (e.g., "Aspiring artists", "Tech enthusiasts").
        - "hook_statement": (String) The single most powerful or attention-grabbing sentence from the transcript.
    3. Your output MUST be ONLY the raw JSON object and nothing else. Do not wrap it in markdown backticks.

    TRANSCRIPT:
    ---
    {transcript}
    ---
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # Clean the response to ensure it's a valid JSON string
        clean_json_str = response.text.strip().replace("```json", "").replace("```", "")
        # Parse the JSON string into a Python dictionary
        context_dict = json.loads(clean_json_str)
        return context_dict
    except Exception as e:
        st.error(f"An error occurred during context generation (JSON parsing failed): {e}")
        st.text_area("Failed AI Response:", response.text if 'response' in locals() else "No response from AI.", height=150)
        return None
def analyze_transcript_for_clips(transcript, model_name, num_clips, chunk_size, progress_bar=None):
    """Analyzes a transcript to find viral clips with smart selection across entire content."""
    
    # Split transcript into chunks for analysis (but not for clip generation)
    chunks, _, _ = split_transcript_into_chunks(transcript, num_clips, chunk_size)
    
    if not chunks:
        st.error("Failed to process transcript chunks.")
        return None
    
    # Show chunking information to user
    st.info(f"📊 Analyzing transcript in {len(chunks)} chunks for optimal coverage. Target: {num_clips} high-quality clips.")
    
    # Step 1: Analyze each chunk to find ALL potential clip moments
    all_potential_clips = []
    
    for i, chunk in enumerate(chunks):
        # Update progress bar
        if progress_bar:
            progress_text = f"Analyzing chunk {i+1}/{len(chunks)} for potential clips..."
            progress_bar.progress((i) / len(chunks), text=progress_text)
        
        # Find potential clips in this chunk (more than needed, for selection)
        potential_clips = find_potential_clips_in_chunk(chunk, model_name, i+1, len(chunks))
        if potential_clips:
            all_potential_clips.extend(potential_clips)
    
    # Step 2: Smart selection of best clips across entire transcript
    if progress_bar:
        progress_bar.progress(0.8, text="Selecting best clips from entire transcript...")
    
    selected_clips = select_best_clips_from_all(all_potential_clips, num_clips, model_name)
    
    # Final progress update
    if progress_bar:
        progress_bar.progress(1.0, text="Analysis complete!")
    
    if not selected_clips:
        st.warning("⚠️ No high-quality clips found. This may indicate:")
        st.markdown("""
        - **Content quality** - The transcript may not contain enough viral-worthy moments
        - **Duration requirements** - Clips must be 35-45 seconds for optimal performance
        - **AI analysis** - Consider using Gemini Pro for more detailed content analysis
        """)
        return None
    
    # Show selection results
    # Removed the success message about number of clips found
    
    if len(selected_clips) < num_clips:
        st.info(f"💡 Only {len(selected_clips)} clips met our quality standards. This is actually better than forcing {num_clips} mediocre clips!")
    
    return selected_clips

def find_potential_clips_in_chunk(chunk, model_name, chunk_num, total_chunks):
    """Finds potential clips in a chunk using the engineered prompt for better distribution."""
    
    prompt = f"""
    The Engineered Prompt
    ROLE:
    You are an expert YouTube Shorts strategist and viral content analyst. Your core skill is identifying high-engagement moments in long-form video transcripts and structuring them into a strategic content plan. You think like both a data analyst and a creative video editor.

    OBJECTIVE:
    Analyze the provided raw video transcript chunk to identify high-potential clips that can be edited into successful YouTube Shorts. This is chunk {chunk_num} of {total_chunks} from a larger transcript.

    INPUT DATA:
    A video transcript chunk with timestamps.

    CRITICAL RULES & CONSTRAINTS:

    Clip Selection: Identify ALL high-potential clips in this chunk. Prioritize moments that contain one or more of the following viral triggers:

    - Strong Emotional Hooks: Stories of struggle, success, vulnerability, or nostalgia.
    - Counter-Intuitive Insights: Ideas that challenge conventional wisdom.
    - High Value & Actionable Advice: Specific formulas, frameworks, or tips that the audience can apply to their own lives (e.g., business strategies, life hacks, investment philosophies).
    - Shock & Surprise: Unexpected revelations, shocking numbers, or bizarre anecdotes.
    - Relatability: Universal experiences, problems, or feelings.

    Duration Mandate: Every single clip identified must have a duration that is strictly between 25 and 45 seconds. This is a non-negotiable rule.

    Accuracy Mandate: All timestamp selections and duration calculations must be mathematically perfect. Before providing the final output, you must double-check every single row to ensure the (End Time) - (Start Time) calculation is precise and fits the 25-45 second window.

    REQUIRED OUTPUT FORMAT:
    Present the final results in a single, comprehensive markdown table. The table must have the following seven columns, arranged in this exact order:

    Shorts No | Timestamp (Start - End) | Duration | Why it Works (Content Focus) | Strategy | Hook for this Short | Success Potential & Justification

    COLUMN DEFINITIONS:

    Shorts No: A sequential number for each clip suggestion (1, 2, 3...).

    Timestamp (Start - End): The precise start and end time from the transcript for the suggested clip.

    Duration: The exact, calculated duration of the clip in seconds. Must be between 25 and 45.

    Why it Works (Content Focus): A concise summary of the clip's core message and why it is compelling or valuable to an audience.

    Strategy: A brief, actionable strategy for how to edit or frame the clip for maximum impact (e.g., "Use bold text overlays for key numbers," "Frame it as a secret strategy," "Build suspense before the reveal").

    Hook for this Short: A powerful, attention-grabbing sentence that can be used as the video's title or opening line. This should be directly derived from the clip's content.

    Success Potential & Justification: A rating (e.g., Medium-High, High, Very High, Extremely High) followed by a single, concise sentence explaining why it has that potential (e.g., "It taps into the universal emotion of a pet's unconditional love, making it highly shareable.").

    EXECUTION:
    Begin your analysis now on the transcript chunk provided below. Find ALL viral-worthy moments in this chunk, ensuring even distribution across the timeline.

    **Transcript Chunk {chunk_num} to Analyze:**
    ---
    {chunk}
    ---
    
    Return ONLY a Markdown table with your findings. If no viral moments exist, return an empty table.
    """
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        if response.text and '|' in response.text:
            return response.text
        return None
        
    except Exception as e:
        st.error(f"Error analyzing chunk {chunk_num}: {e}")
        return None

def select_best_clips_from_all(all_potential_clips, target_clips, model_name):
    """Intelligently selects the best clips from all potential moments across the transcript."""
    
    if not all_potential_clips:
        return None
    
    # Combine all potential clips into one analysis
    combined_potential = "\n\n".join(all_potential_clips)
    
    prompt = f"""
    The Engineered Prompt - Final Selection
    ROLE:
    You are an expert YouTube Shorts strategist and viral content analyst. Your core skill is identifying high-engagement moments in long-form video transcripts and structuring them into a strategic content plan.

    OBJECTIVE:
    Review ALL potential clips discovered from the entire transcript and select the TOP {target_clips} clips that will give the best performance. This is the final selection phase.

    INPUT DATA:
    Multiple markdown tables containing potential viral clips from different parts of the transcript.

    CRITICAL RULES & CONSTRAINTS:

    Selection Criteria: Select EXACTLY {target_clips} clips based on:
    1. **Viral Potential**: Highest success potential ratings
    2. **Content Quality**: Most compelling and valuable content
    3. **Timeline Distribution**: Spread clips evenly across the entire transcript duration
    4. **Duration Compliance**: All clips must be 25-45 seconds
    5. **Diversity**: Mix of different viral trigger types

    Distribution Mandate: 
    - DO NOT cluster all clips in one time period
    - Spread clips evenly across the transcript timeline
    - If transcript is 26 minutes, clips should cover 0:00 to 26:00
    - If transcript is 10 minutes, clips should cover 0:00 to 10:00
    - **CRITICAL**: Ensure clips are distributed across different time periods, not clustered together

    REQUIRED OUTPUT FORMAT:
    Present the final results in a single, comprehensive markdown table. The table must have the following seven columns, arranged in this exact order:

    Shorts No | Timestamp (Start - End) | Duration | Why it Works (Content Focus) | Strategy | Hook for this Short | Success Potential & Justification

    COLUMN DEFINITIONS:

    Shorts No: A sequential number for each clip suggestion (1, 2, 3...).

    Timestamp (Start - End): The precise start and end time from the transcript for the suggested clip.

    Duration: The exact, calculated duration of the clip in seconds. Must be between 25 and 45.

    Why it Works (Content Focus): A concise summary of the clip's core message and why it is compelling or valuable to an audience.

    Strategy: A brief, actionable strategy for how to edit or frame the clip for maximum impact.

    Hook for this Short: A powerful, attention-grabbing sentence that can be used as the video's title or opening line.

    Success Potential & Justification: A rating (e.g., Medium-High, High, Very High, Extremely High) followed by a single, concise sentence explaining why it has that potential.

    FINAL REQUIREMENTS:
    1. Return EXACTLY {target_clips} clips
    2. Sort by timestamp to show chronological distribution
    3. Ensure even distribution across the entire transcript timeline
    4. All clips must be 25-45 seconds
    5. Each clip must have genuine viral potential

    **Potential Clips to Select From:**
    ---
    {combined_potential}
    ---
    """
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        if response.text and '|' in response.text:
            # Validate that we got the right number of clips
            lines = [line for line in response.text.strip().split('\n') if '|' in line and '---' not in line]
            if len(lines) > 0:
                # Removed the AI selection info messages
                if len(lines) < target_clips:
                    st.warning(f"⚠️ AI could only find {len(lines)} viral-worthy clips out of {target_clips} requested")
                return response.text
        return None
        
    except Exception as e:
        st.error(f"Error selecting best clips: {e}")
        return None

def split_transcript_into_chunks(transcript, num_clips, max_chunk_duration_minutes=10):
    """
    Splits transcript into chunks and distributes clip requests evenly across them.
    This ensures clips cover the entire transcript duration.
    """
    if not transcript:
        return [], 0, 0
    
    # Parse transcript to get total duration
    lines = transcript.strip().split('\n')
    timestamps = []
    
    for line in lines:
        if '-->' in line:
            try:
                start_time = line.split('-->')[0].strip()
                end_time = line.split('-->')[1].strip()
                # Convert timestamp to seconds
                start_seconds = timestamp_to_seconds(start_time)
                end_seconds = timestamp_to_seconds(end_time)
                timestamps.append((start_seconds, end_seconds))
            except:
                continue
    
    if not timestamps:
        return [transcript], num_clips, 0  # Fallback if no timestamps found
    
    total_duration = max([end for _, end in timestamps])
    chunk_duration = max_chunk_duration_minutes * 60  # Convert to seconds
    num_chunks = max(1, int(total_duration / chunk_duration) + 1)
    
    # Distribute clips evenly across chunks
    clips_per_chunk = max(1, num_clips // num_chunks)
    remaining_clips = num_clips % num_chunks
    
    chunks = []
    current_chunk = []
    current_chunk_start = 0
    
    for line in lines:
        if '-->' in line:
            try:
                start_time = line.split('-->')[0].strip()
                start_seconds = timestamp_to_seconds(start_time)
                
                # Check if we need to start a new chunk
                if start_seconds >= current_chunk_start + chunk_duration:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_chunk_start = start_seconds
                else:
                    current_chunk.append(line)
            except:
                current_chunk.append(line)
        else:
            current_chunk.append(line)
    
    # Add the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks, clips_per_chunk, remaining_clips

def timestamp_to_seconds(timestamp):
    """Converts SRT timestamp format (HH:MM:SS,ms) to seconds."""
    try:
        # Remove milliseconds
        timestamp = timestamp.split(',')[0]
        parts = timestamp.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            return int(parts[0])
    except:
        return 0

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

# --- THIS IS THE CORRECTED FUNCTION ---

def generate_shorts_titles(generated_context, strategy_params, model_name):
    """Generates Shorts titles using generated context and strategy parameters."""
    # This part is crucial: it unpacks the dictionary from the UI
    audience = strategy_params.get('audience', 'a general audience, yotube audience ')
    takeaway = strategy_params.get('takeaway', 'the main message')
    tone = strategy_params.get('tone', 'an engaging tone')

    prompt = f"""
    ROLE AND GOAL:
    You are an expert viral content strategist based in Noida, specializing in writing high-engagement, "scroll-stopping" titles for YouTube Shorts. Your goal is to generate 15-20 powerful titles based on the provided video context and strategic parameters.

    CONTEXT OF THE VIDEO:
    ---
    {generated_context}
    ---

    STRATEGIC PARAMETERS:
    - **Target Audience:** {audience}
    - **Main Takeaway/Message:** {takeaway}
    - **Desired Tone/Style:** {tone}

    TITLE GENERATION STRATEGIES TO USE:
    (Your full list of 18 strategies goes here)
    1.  **Punchline / Reveal:** ...
    2.  **Controversial Opinion:** ...
    (etc.)

    INSTRUCTIONS:
    Your final output must be ONLY a Markdown table with two columns: "Strategy" and "Suggested Title". Do not include any other text, explanation, or introduction.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return None
def generate_headers(transcript, core_message, peak_moment, audience_tone, model_name):
    """Generates thumbnail headers using the transcript directly."""
    prompt = f"""
    ROLE AND GOAL:
    You are an expert thumbnail text generator. Your sole focus is to create 10-15 short, powerful, and high-impact headers for a YouTube Short thumbnail based on the provided transcript and context. The headers should be 3-7 words max.

    CONTEXT OF THE VIDEO:
    ---
    {transcript}
    ---

    STRATEGIC PARAMETERS:
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

def generate_headers_from_context(generated_context, header_params, model_name):
    """Generates thumbnail headers using the generated context from transcript."""
    core_message = header_params.get('core_message', 'the main message')
    peak_moment = header_params.get('peak_moment', 'the key moment')
    audience_tone = header_params.get('audience_tone', 'general audience')
    
    prompt = f"""
ROLE AND GOAL:
You are an expert YouTube Thumbnail Text strategist. Your goal is to create 10-15 powerful, high-impact headers for a YouTube Short thumbnail. The headers must be concise but also highly specific and intriguing to maximize click-through rates. They should be visually scannable in 1-2 seconds.

CRITICAL CONTEXT OF THE VIDEO (Generated from Transcript):
---
{generated_context}
---

STRATEGIC PARAMETERS:
- **Core Message:** {core_message}
- **Peak Moment / Hook:** {peak_moment}
- **Audience & Tone:** {audience_tone}

GUIDING PRINCIPLES (You MUST follow these):
1.  **BE HYPER-SPECIFIC:** This is the most important rule. You MUST incorporate specific names, numbers, keywords, or unique concepts from the context above.
    -   **BAD (Generic):** "Feeling Overwhelmed?"
    -   **GOOD (Specific):** "Overwhelmed by Finances?"
    -   **BAD (Generic):** "The Best Advice I Ever Got"
    -   **GOOD (Specific):** "Tanmay Bhatt's Best Advice"

2.  **FOCUS ON TRANSFORMATION & OUTCOME:** Frame the headers around a clear "before & after" or a tangible result.
    -   **BAD (Generic):** "Get Organized Now"
    -   **GOOD (Specific):** "From Confused To Clear" or "My Finances In 1 Sheet"

3.  **LEVERAGE AUTHORITY/PERSONALITY:** If a specific person or brand is mentioned in the context (like 'Tanmay Bhatt'), use their name directly in the headers to build credibility and curiosity.

4.  **THINK VISUALLY (LINES OF TEXT):** While the headers should be short, don't be afraid to use 2-3 extra words if it adds crucial context. Imagine how the text would break into lines on a thumbnail.
    -   Example: "TANMAY'S SECRET // TO FIXING FINANCES" is more powerful than "The Financial Secret".

HEADER STRATEGIES TO USE (Apply the principles above to these strategies):
- **Problem/Solution:** (e.g., "Finances A Mess? Try This.")
- **Curiosity Gap:** (e.g., "Tanmay Bhatt's 1-Sheet Secret")
- **Bold Statement:** (e.g., "This Excel Sheet Changed My Life")
- **Result-Oriented:** (e.g., "Clarity In 15 Minutes")
- **Emotional Trigger:** (e.g., "I Was So Lost With Money")
- **Direct Question:** (e.g., "Need Financial Clarity?")
- **Surprising/Unexpected:** (e.g., "My Mentor? Tanmay Bhatt.")
- **Motivational:** (e.g., "Stop Feeling Stuck With Money")
- **Nostalgic/Sentimental:** (e.g., "The Advice That Saved Me")
- **Aspirational:** (e.g., "Your Path To Financial Freedom")
- **Intriguing/Mysterious:** (e.g., "The Secret The Rich Use")
- **Urgent/Timely:** (e.g., "Fix Your Finances NOW")

INSTRUCTIONS:
Your final output must be ONLY a Markdown table with two columns: "Strategy" and "Suggested Header". Do not include any other text, preamble, or explanation.
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
    if not lines: return pd.DataFrame(columns=header)
    header = [h.strip() for h in lines[0].split('|') if h.strip()]
    data = [[r.strip() for r in line.split('|') if r.strip()] for line in lines[1:]]
    data = [row for row in data if len(row) == len(header)]
    return pd.DataFrame(data, columns=header) if data else pd.DataFrame(columns=header)

def validate_clip_durations(clips_text, min_duration=30, max_duration=60):
    """
    Validates that generated clips meet duration requirements and provides feedback.
    """
    if not clips_text:
        return clips_text, []
    
    # Parse the markdown table
    lines = clips_text.strip().split('\n')
    valid_clips = []
    invalid_clips = []
    
    for line in lines:
        if '|' in line and '---' not in line:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 7:  # Ensure we have enough columns
                try:
                    # Extract duration from the second column
                    duration_str = parts[1].replace('s', '').replace(' seconds', '').strip()
                    duration = int(duration_str)
                    
                    if min_duration <= duration <= max_duration:
                        valid_clips.append(line)
                    else:
                        invalid_clips.append({
                            'line': line,
                            'duration': duration,
                            'issue': f"Duration {duration}s is outside required range ({min_duration}-{max_duration}s)"
                        })
                except (ValueError, IndexError):
                    # If we can't parse duration, include it but mark as potentially problematic
                    valid_clips.append(line)
            else:
                valid_clips.append(line)
        else:
            valid_clips.append(line)
    
    return '\n'.join(valid_clips), invalid_clips

# ##############################################################################
# --- STREAMLIT UI SECTION ---
# ##############################################################################

if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""

tab1, tab3 = st.tabs([
    "🎬 Generate Clips", "✍️ Generate Titles & Headers"
])

# --- TAB 1: GENERATE CLIPS ---
with tab1:
    st.header("Generate Clip Ideas from a Transcript")
    input_method = st.radio("Choose transcript source:", ("From YouTube URL", "From File Upload"), horizontal=True, key="clip_gen_source", on_change=lambda: st.session_state.update(raw_analysis_table=None, strong_clips_table=None))

    if input_method == "From YouTube URL":
        video_url = st.text_input("Enter YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=...")
        
        # Add helpful guidance about potential issues
        st.info("💡 **Note:** Some YouTube videos may have format restrictions. If URL processing fails, use 'From File Upload' instead.")
        
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
        col1, col2, col3 = st.columns(3)
        num_clips = col1.number_input("Number of clip ideas to find", 3, 50, 7, 1, key="num_clips_input")
        model_choice_t1 = col2.selectbox("Choose Model", ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"), key="model_choice_t1", help="Flash is faster. Pro provides more detailed analysis.")
        chunk_size = col3.number_input("Chunk size (minutes)", 5, 20, 10, 1, key="chunk_size_input", help="Smaller chunks = better coverage of long transcripts. For 26+ min videos, use 5-8 minutes.")
        
        # Show chunk size recommendations for long transcripts
        if st.session_state.get('transcript'):
            transcript_length = len(st.session_state.transcript.split('\n'))
            if transcript_length > 1000:  # Estimate for long transcripts
                st.info("📏 **For long transcripts (26+ minutes):** Use chunk size 5-8 minutes for optimal coverage and 35-45 second clips.")
        
        # Explain the new smart selection approach
        st.info("🎯 **New Smart Selection System:** The AI will analyze your ENTIRE transcript but only select truly viral-worthy moments. You may get fewer clips than requested if the content quality doesn't meet our standards - this ensures every clip has real viral potential!")
        
        if st.button("✨ Find Clip Ideas", type="primary", key="analyze_btn_t1"):
            with st.spinner("Gemini is analyzing the transcript for clips..."):
                # Show chunking information first
                chunks, _, _ = split_transcript_into_chunks(st.session_state.transcript, num_clips, chunk_size)
                if chunks:
                    st.info(f"📊 Processing transcript in {len(chunks)} chunks. Target: {num_clips} clips total.")
                    
                    # Show progress bar for chunk processing
                    progress_bar = st.progress(0, text="Processing chunks...")
                    
                    st.session_state['raw_analysis_table'] = analyze_transcript_for_clips(st.session_state.transcript, model_choice_t1, num_clips, chunk_size, progress_bar)
                    st.session_state['strong_clips_table'] = None
                    
                    progress_bar.empty()
                else:
                    st.error("Failed to process transcript into chunks. Please check your transcript format.")
        
        if st.session_state.get('raw_analysis_table'):
            # Parse and display the raw analysis results
            df = parse_markdown_table(st.session_state['raw_analysis_table'])
            if not df.empty:
                st.success(f"🎯 Analysis Complete! Found {len(df)} high-quality clips from your entire transcript.")
                
                # Show clip distribution across timeline
                if 'Timestamp (Start - End)' in df.columns:
                    # Extract start times from timestamp column to show distribution
                    start_times = []
                    for timestamp in df['Timestamp (Start - End)']:
                        if ' - ' in str(timestamp):
                            start_time = str(timestamp).split(' - ')[0].strip()
                            start_times.append(start_time)
                    
                    if start_times:
                        # Show timeline coverage
                        st.info(f"📊 **Timeline Coverage:** Clips span from {start_times[0]} to {start_times[-1]}")
                        
                        # Show distribution across time periods
                        if len(start_times) > 1:
                            st.info(f"🎯 **Distribution:** {len(start_times)} clips distributed across the transcript timeline")
                
                # Display all discovered clips
                st.dataframe(df, use_container_width=True)
                st.download_button("Download Full Analysis as CSV", df.to_csv(index=False).encode('utf-8'), "full_clip_analysis.csv", "text/csv")
                
                st.markdown("---")
                st.subheader("Step 2: Select the Strongest Clips for Upload")
                num_top_clips = st.number_input("Number of top clips to select", 1, min(len(df), 20), min(3, len(df)), key="num_top_clips")
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

# --- TAB 2: ANALYZE PATTERNS (COMMENTED OUT) ---
# with tab2:
#     st.subheader("Deconstruct What Makes a Short Go Viral")
#     st.markdown("Add up to 5 successful shorts. The tool will **transcribe each one** and then analyze the content to find common patterns and strategies.")
#     
#     if 'num_url_inputs' not in st.session_state: st.session_state.num_url_inputs = 1
#     
#     def add_url_input():
#         if st.session_state.num_url_inputs < 5: st.session_state.num_url_inputs += 1
#     def remove_url_input():
#         if st.session_state.num_url_inputs > 1: st.session_state.num_url_inputs -= 1
# 
#     for i in range(st.session_state.num_url_inputs):
#         st.text_input(f"URL for Short #{i+1}", key=f"url_{i}")
#     
#     col1, col2, _ = st.columns([1, 1, 4])
#     col1.button("Add URL ➕", on_click=add_url_input, use_container_width=True)
#     col2.button("Remove Last ➖", on_click=remove_url_input, use_container_width=True, disabled=(st.session_state.num_url_inputs <= 1))
# 
#     model_choice_t2 = st.selectbox("Choose Gemini Model", ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"), key="model_choice_tab2")
# 
#     if st.button("Find Patterns from Transcripts", type="primary", key="find_patterns_btn"):
#         valid_urls = [st.session_state.get(f"url_{i}") for i in range(st.session_state.num_url_inputs) if st.session_state.get(f"url_{i}")]
#         if valid_urls:
#             all_transcripts_for_prompt = []
#             progress_bar = st.progress(0, text="Starting transcriptions...")
#             for i, url in enumerate(valid_urls):
#                 progress_text = f"Processing video {i+1}/{len(valid_urls)}..."
#                 progress_bar.progress((i) / len(valid_urls), text=progress_text)
#                 transcript_text = transcribe_audio_with_whisper(url)
#                 if transcript_text:
#                     all_transcripts_for_prompt.append(f"--- TRANSCRIPT FOR VIDEO {i+1} ({url}) ---\n{transcript_text}\n--- END ---\n")
#             
#             progress_bar.progress(1.0, text="Analyzing with Gemini...")
#             if all_transcripts_for_prompt:
#                 final_prompt_content = "\n".join(all_transcripts_for_prompt)
#                 analysis_result = analyze_viral_patterns(final_prompt_content, model_choice_t2)
#                 st.markdown("---")
#                 st.subheader("🔬 Analysis Report")
#                 st.markdown(analysis_result)
#             else:
#                 st.error("Could not generate any transcripts. Please check the URLs.")
#             progress_bar.empty()
#         else:
#             st.warning("Please enter at least one YouTube Short URL.")

# --- TAB 3: GENERATE TITLES & HEADERS ---
# --- MERGED TITLE AND HEADER GENERATOR ---

with tab3:
    st.subheader("✍️ Generate High-Performing Titles & Headers")
    st.markdown("Generate both titles and thumbnail headers using common strategy parameters and transcript-based context.")
    
    # Input method selection
    input_method_t3 = st.radio("Choose input method:", ("Upload Video", "Paste Transcript"), horizontal=True, key="title_header_input_method")
    
    if input_method_t3 == "Upload Video":
        # Video upload section
        uploaded_video_t3 = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'], key="title_header_video_uploader")
        
        if uploaded_video_t3:
            # Two-column layout for video and form
            left_col_t3, right_col_t3 = st.columns([3, 2])
            
            with left_col_t3:
                st.markdown("### 📝 Common Strategy Parameters")
                # Common parameters for both titles and headers
                audience = st.text_input("Target Audience", placeholder="e.g., Beginner freelancers", key="audience_t3_video")
                tone = st.selectbox("Desired Tone", ("Educational", "Bold & Controversial", "Inspirational", "Humorous", "Relatable", "Urgent/Timely", "Intriguing/Mysterious", "Motivational", "Nostalgic/Sentimental", "Aspirational / Luxurious", "Surprising/Unexpected"), key="tone_t3_video")
                takeaway = st.text_input("Main Takeaway or Message", placeholder="e.g., Build a personal brand to earn more", key="takeaway_t3_video")
                core_message = st.text_input("Core Message (for headers)", placeholder="e.g., The importance of networking", key="core_message_t3_video")
                peak_moment = st.text_input("Peak Moment / Hook (for headers)", placeholder="e.g., 'I got my dream job from one conversation.'", key="peak_moment_t3_video")
                model_choice_t3 = st.selectbox("Choose Gemini Model", ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"), key="model_choice_tab3_video")
                
                if st.button("🚀 Generate Titles & Headers from Video", type="primary", key="generate_titles_headers_video_btn"):
                    if audience and takeaway and core_message and peak_moment:
                        # STEP 1: Transcribe the video
                        with st.spinner("Step 1/4: Transcribing video..."):
                            video_bytes = uploaded_video_t3.getvalue()
                            video_transcript = transcribe_uploaded_video(video_bytes)
                        
                        if video_transcript:
                            st.success("Transcription complete!")
                            
                            # STEP 2: Generate Context from the transcript
                            with st.spinner("Step 2/4: AI is analyzing the transcript context..."):
                                generated_context = generate_context_from_transcript(video_transcript, model_choice_t3)

                            if generated_context:
                                st.success("Context analysis complete!")
                                with st.expander("View Generated Context"):
                                    st.write(generated_context)
                                
                                # Bundle strategy parameters for both titles and headers
                                strategy_params = {"audience": audience, "takeaway": takeaway, "tone": tone}
                                header_params = {"core_message": core_message, "peak_moment": peak_moment, "audience_tone": f"{audience}; {tone}"}

                                # STEP 3: Generate Titles using the context
                                with st.spinner("Step 3/4: Gemini is crafting strategic titles..."):
                                    titles = generate_shorts_titles(generated_context, strategy_params, model_choice_t3)
                                
                                # STEP 4: Generate Headers using the context
                                with st.spinner("Step 4/4: Gemini is crafting powerful headers..."):
                                    headers = generate_headers_from_context(generated_context, header_params, model_choice_t3)
                                
                                # Display results
                                if titles and headers:
                                    st.markdown("---")
                                    st.subheader("🔥 Suggested Titles (with Strategy)")
                                    st.markdown(titles)
                                    
                                    st.markdown("---")
                                    st.subheader("🚀 Suggested Thumbnail Headers")
                                    st.markdown(headers)
                                else:
                                    if not titles:
                                        st.error("Title generation failed.")
                                    if not headers:
                                        st.error("Header generation failed.")
                            else:
                                st.error("Context generation failed after transcription.")
                        else:
                            st.error("Could not transcribe the video.")
                    else:
                        st.warning("Please fill in all required fields for the best results.")
            
            with right_col_t3:
                st.markdown("### 📹 Video Preview")
                st.video(uploaded_video_t3)
                st.caption(f"📁 {uploaded_video_t3.name}")
    
    else: # This block handles the "Paste Transcript" option
        transcript_input_t3 = st.text_area("Paste your full Short transcript here", height=200, key="transcript_input_tab3")
        
        st.markdown("### 📝 Common Strategy Parameters")
        col1, col2 = st.columns(2)
        # Common parameters for both titles and headers
        audience = col1.text_input("Target Audience", placeholder="e.g., Beginner freelancers", key="audience_t3_text")
        tone = col1.selectbox("Desired Tone", ("Educational", "Bold & Controversial", "Inspirational", "Humorous", "Relatable", "Urgent/Timely", "Intriguing/Mysterious", "Motivational", "Nostalgic/Sentimental", "Aspirational / Luxurious", "Surprising/Unexpected"), key="tone_t3_text")
        takeaway = col1.text_input("Main Takeaway or Message", placeholder="e.g., Build a personal brand to earn more", key="takeaway_t3_text")
        core_message = col2.text_input("Core Message (for headers)", placeholder="e.g., The importance of networking", key="core_message_t3_text")
        peak_moment = col2.text_input("Peak Moment / Hook (for headers)", placeholder="e.g., 'I got my dream job from one conversation.'", key="peak_moment_t3_text")
        model_choice_t3 = st.selectbox("Choose Gemini Model", ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"), key="model_choice_tab3_text")
        
        if st.button("🚀 Generate Titles & Headers from Transcript", type="primary", key="generate_titles_headers_text_btn"):
            if transcript_input_t3 and audience and takeaway and core_message and peak_moment:
                # STEP 1: Generate Context from the pasted transcript
                with st.spinner("Step 1/3: AI is analyzing the transcript context..."):
                    generated_context = generate_context_from_transcript(transcript_input_t3, model_choice_t3)
                
                if generated_context:
                    st.success("Context analysis complete!")
                    with st.expander("View Generated Context"):
                        st.write(generated_context)
                        
                    # Bundle strategy parameters for both titles and headers
                    strategy_params = {"audience": audience, "takeaway": takeaway, "tone": tone}
                    header_params = {"core_message": core_message, "peak_moment": peak_moment, "audience_tone": f"{audience}; {tone}"}
                    
                    # STEP 2: Generate Titles using the context
                    with st.spinner("Step 2/3: Gemini is crafting strategic titles..."):
                        titles = generate_shorts_titles(generated_context, strategy_params, model_choice_t3)
                    
                    # STEP 3: Generate Headers using the context
                    with st.spinner("Step 3/3: Gemini is crafting powerful headers..."):
                        headers = generate_headers_from_context(generated_context, header_params, model_choice_t3)
                    
                    # Display results
                    if titles and headers:
                        st.markdown("---")
                        st.subheader("🔥 Suggested Titles (with Strategy)")
                        st.markdown(titles)
                        
                        st.markdown("---")
                        st.subheader("🚀 Suggested Thumbnail Headers")
                        st.markdown(headers)
                    else:
                        if not titles:
                            st.error("Title generation failed.")
                        if not headers:
                            st.error("Header generation failed.")
                else:
                    st.error("Context generation failed.")
            else:
                st.warning("Please fill in all required fields for the best results.")

# --- TAB 4: PRE-UPLOAD REVIEW (COMMENTED OUT) ---
# with tab4:
#     st.subheader("🚀 Get a Final Pre-Upload Review")
#     uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'], key="review_video_uploader")
#     
#     if uploaded_video:
#         # Create a two-column layout: Form on left, Video on right
#         left_col, right_col = st.columns([3, 2])
#         
#         with left_col:
#             st.markdown("### 📝 Video Context & Analysis")
#             review_title = st.text_input("Proposed Title", placeholder="e.g., DON'T Make This Freelance Mistake!", key="review_title")
#             review_header = st.text_input("Thumbnail Header Text", placeholder="e.g., THE #1 MISTAKE", key="review_header")
#             review_audience = st.text_input("Target Audience", placeholder="e.g., Aspiring freelancers in India", key="review_audience")
#             review_model_choice = st.selectbox("Choose Gemini Model", ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"), key="review_model_choice")
# 
#             if st.button("🔬 Analyze and Review Short", type="primary", key="get_review_btn"):
#                 if review_title and review_audience:
#                     with st.spinner("Transcribing video..."):
#                         video_bytes = uploaded_video.getvalue()
#                         review_transcript = transcribe_uploaded_video(video_bytes)
#                     if review_transcript:
#                         st.success("Transcription complete! Now getting feedback...")
#                         with st.spinner("Your personal strategist is reviewing the content..."):
#                             feedback = get_pre_upload_feedback(review_transcript, review_title, review_header, review_audience, review_model_choice)
#                         if feedback:
#                             st.markdown("---")
#                             st.subheader("📋 Feedback Report")
#                             st.markdown(feedback)
#                         else:
#                             st.error("The review could not be generated.")
#                     else:
#                         st.error("Could not transcribe the video.")
#                 else:
#                     st.warning("Please fill in the Title and Audience fields.")
#         
#         with right_col:
#             st.markdown("### 📹 Video Preview")
#             st.video(uploaded_video)
#             st.caption(f"📁 {uploaded_video.name}")

# --- TAB 5: POST-UPLOAD DIAGNOSIS (COMMENTED OUT) ---
# Initialize session state variables
# if 'diag_url_input' not in st.session_state:
#     st.session_state.diag_url_input = ""
# if 'diag_uploaded_video' not in st.session_state:
#     st.session_state.diag_uploaded_video = None
# 
# with tab5:
#     st.subheader("📈 Diagnose Why Your Short Isn't Performing")
#     
#     # Input method selection
#     input_method_t5 = st.radio("Choose input method:", ("YouTube URL", "Upload Video"), horizontal=True, key="diagnosis_input_method")
#     
#     if input_method_t5 == "YouTube URL":
#         st.session_state.diag_url_input = st.text_input("Enter YouTube Short URL", placeholder="e.g., https://youtube.com/shorts/...", key="diag_url_input")
#         
#         if st.button("🔍 Diagnose Performance Issues", type="primary", key="diagnose_url_btn"):
#             if st.session_state.diag_url_input:
#                 with st.spinner("Transcribing and analyzing your short..."):
#                     transcript_text = transcribe_audio_with_whisper(st.session_state.diag_url_input)
#                     if transcript_text:
#                         st.success("Transcription complete! Now diagnosing performance issues...")
#                         with st.spinner("Your personal performance analyst is reviewing the content..."):
#                             diagnosis_result = diagnose_short_performance(transcript_text, model_choice_t5)
#                         if diagnosis_result:
#                             st.markdown("---")
#                             st.subheader("🔬 Performance Diagnosis Report")
#                             st.markdown(diagnosis_result)
#                         else:
#                             st.error("The diagnosis could not be generated.")
#                     else:
#                         st.error("Could not transcribe the video from the URL.")
#             else:
#                 st.warning("Please enter a YouTube Short URL.")
#     else:
#         st.session_state.diag_uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'], key="diagnosis_video_uploader")
#         
#         if st.session_state.diag_uploaded_video:
#             # Create a two-column layout: Form on left, Video on right
#             left_col_t5, right_col_t5 = st.columns([3, 2])
#             
#             with left_col_t5:
#                 st.markdown("### 📝 Performance Analysis")
#                 diag_model_choice = st.selectbox("Choose Gemini Model", ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"), key="diag_model_choice")
#                 
#                 if st.button("🔍 Diagnose Performance Issues", type="primary", key="diagnose_upload_btn"):
#                     with st.spinner("Transcribing video..."):
#                         video_bytes = st.session_state.diag_uploaded_video.getvalue()
#                         diag_transcript = transcribe_uploaded_video(video_bytes)
#                     if diag_transcript:
#                         st.success("Transcription complete! Now diagnosing performance issues...")
#                         with st.spinner("Your personal performance analyst is reviewing the content..."):
#                             diagnosis_result = diagnose_short_performance(diag_transcript, diag_model_choice)
#                         if diagnosis_result:
#                             st.markdown("---")
#                             st.subheader("🔬 Performance Diagnosis Report")
#                             st.markdown(diagnosis_result)
#                         else:
#                             st.error("The diagnosis could not be generated.")
#                     else:
#                         st.error("Could not transcribe the video.")
#             
#             with right_col_t5:
#                 st.markdown("### 📹 Video Preview")
#                 st.video(st.session_state.diag_uploaded_video)
#                 st.caption(f"📁 {st.session_state.diag_uploaded_video.name}")
