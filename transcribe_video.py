import whisper

def transcribe_video(file_path):
    model = whisper.load_model("base")  # you can try "tiny" for faster speed
    result = model.transcribe(file_path)
    return result["text"]

if __name__ == "__main__":
    transcript = transcribe_video("video.mp4")
    print("📝 Transcript:\n")
    print(transcript)
