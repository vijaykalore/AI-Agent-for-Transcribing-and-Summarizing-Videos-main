import subprocess
import whisper
import os

def extract_audio(video_path: str, audio_path: str = "temp_audio.wav") -> str:
    """
    Extracts the audio track from a video using FFmpeg and saves it to `audio_path`.
    
    :param video_path: Path to the input video file
    :param audio_path: Desired path for the extracted audio file (WAV format)
    :return: The path to the extracted audio file
    """
    # Remove old audio file if it exists
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # Construct FFmpeg command
    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",   # High quality
        "-map", "a",   # Select audio track
        audio_path,
        "-y"           # Overwrite without asking
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    return audio_path

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribes audio using OpenAI's Whisper model and returns the transcribed text.
    
    :param audio_path: Path to the audio file (wav, mp3, etc.)
    :param model_size: Whisper model size (tiny, base, small, medium, large)
    :return: Transcribed text
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    transcript = result["text"]
    return transcript
