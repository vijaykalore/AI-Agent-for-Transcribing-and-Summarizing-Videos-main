import os
from transcriber import extract_audio, transcribe_audio
from summarizer import summarize_text
from utils import chunked_summarize

def video_to_summary(
    video_path: str,
    model_size: str = "base",
    summarizer_model_name: str = "facebook/bart-large-cnn",
    use_chunking: bool = False
) -> str:
    """
    High-level function to convert a video to a summarized text.
    
    :param video_path: Path to the input video file
    :param model_size: Whisper model size (tiny, base, small, medium, large)
    :param summarizer_model_name: Name of the Hugging Face summarization model
    :param use_chunking: If True, we'll chunk large transcripts before summarizing
    :return: The final summary as a string
    """
    # 1. Extract audio
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)

    # 2. Transcribe audio
    transcript = transcribe_audio(audio_path, model_size=model_size)

    # 3. Summarize transcript
    if use_chunking:
        # Summarize in multiple chunks and then do a final summary
        final_summary = chunked_summarize(
            text=transcript,
            summarize_func=lambda txt: summarize_text(
                txt, model_name=summarizer_model_name
            ),
            max_chunk_size=2000
        )
    else:
        # Summarize in a single pass (works best for shorter transcripts)
        final_summary = summarize_text(
            transcript,
            model_name=summarizer_model_name
        )

    # (Optional) Clean up temporary audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return final_summary

if __name__ == "__main__":
    # Example usage
    video_file = "example_video.mp4"
    summary_output = video_to_summary(
        video_file,
        model_size="base",
        summarizer_model_name="facebook/bart-large-cnn",
        use_chunking=True
    )
    print("=== FINAL SUMMARY ===")
    print(summary_output)
