# Smart Text Similarity Engine

A text similarity engine that combines TF-IDF, cosine similarity, and Jaccard distance to find semantically similar documents. Useful for plagiarism detection, document deduplication, and content recommendation.

**Author:** Maharshi Soni | **License:** MIT

## Features

- Multi-algorithm similarity scoring (TF-IDF + Cosine + Jaccard)
- REST API for real-time similarity queries
- Batch CSV processing for document comparison
- Configurable similarity thresholds

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Compare two texts
python main.py --text1 "Machine learning is a subset of AI" --text2 "AI includes machine learning"

# Batch comparison
python main.py --batch documents.csv --threshold 0.7
```
