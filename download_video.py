import yt_dlp

def download_youtube_video(url, output_path="video.mp4"):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

if __name__ == "__main__":
    video_url = input("Enter YouTube URL: ")
    file_path = download_youtube_video(video_url)
    print(f"✅ Video downloaded as: {file_path}")
