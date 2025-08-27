from transformers import pipeline

def summarize_text(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 150,
    min_length: int = 30
) -> str:
    """
    Summarizes the given text using a Transformers summarization pipeline.
    
    :param text: The text to be summarized
    :param model_name: The Hugging Face model name (default is 'facebook/bart-large-cnn')
    :param max_length: The maximum length of the summary
    :param min_length: The minimum length of the summary
    :return: A string containing the summarized text
    """
    # Initialize the summarizer pipeline
    summarizer = pipeline("summarization", model=model_name)

    # For short transcripts, we can summarize in one shot
    # For very long transcripts, consider chunking (we'll add chunk logic in utils.py if needed)
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

    # Return the first (and only) summary result
    return summary[0]["summary_text"]
