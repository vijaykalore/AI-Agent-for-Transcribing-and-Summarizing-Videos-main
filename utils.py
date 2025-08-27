def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list:
    """
    Splits the text into chunks of approximately `chunk_size` characters,
    with an optional overlap to help maintain context continuity.
    
    :param text: The full text to be chunked
    :param chunk_size: Maximum characters per chunk
    :param overlap: Number of characters to overlap between chunks
    :return: A list of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)

        # Move the start pointer, allowing for overlap if any
        start += chunk_size - overlap

        if start < 0:
            start = 0

    return chunks


def chunked_summarize(text: str, summarize_func, max_chunk_size: int = 2000) -> str:
    """
    Summarizes large text by splitting it into chunks, summarizing each chunk,
    and then combining those summaries into a final summary.
    
    :param text: The full text to be summarized
    :param summarize_func: A function that takes a text string and returns a summary
    :param max_chunk_size: Approximate maximum characters per chunk
    :return: A single string containing the combined summary
    """
    # 1. Split the text into chunks
    text_chunks = chunk_text(text, chunk_size=max_chunk_size, overlap=200)
    
    # 2. Summarize each chunk individually
    partial_summaries = [summarize_func(chunk) for chunk in text_chunks]

    # 3. Combine partial summaries
    combined_summary_input = " ".join(partial_summaries)

    # 4. Run a final summarization step on the combined partial summaries
    final_summary = summarize_func(combined_summary_input)
    return final_summary
