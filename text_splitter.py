def split_text(text: str, chunk_size: int = 500) -> list:
    """
    Split the text into chunks for embedding.
    :param text: The text to be split.
    :param chunk_size: The size of each chunk.
    :return: List of text chunks.
    """
    text_chunks = []
    for i in range(0, len(text), chunk_size):
        text_chunks.append({"text": text[i:i + chunk_size]})
    return text_chunks

