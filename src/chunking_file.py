
def split_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    start = 0


    # extract piece of text
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end == len(text):
            break
        
        # preserve text
        
        start = end - overlap

    return chunks


paper_excerpt = """
Brain tumor segmentation is an important task in medical image analysis.
Convolutional neural networks are widely used to identify tumor regions
from MRI scans. U-Net uses an encoder-decoder architecture with skip
connections to combine contextual information with spatial details.

Researchers often evaluate segmentation models using Dice score and
Intersection over Union. However, model performance can vary depending
on the dataset, preprocessing strategy, and tumor characteristics.
"""

chunks = split_into_chunks(paper_excerpt, chunk_size=150, overlap=30)

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)
