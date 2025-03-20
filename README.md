<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Gradio-Interactive%20UI-green?style=for-the-badge" alt="Gradio">
  <img src="https://img.shields.io/badge/Dataset-Kaggle-orange?style=for-the-badge" alt="Kaggle">
</p>

<h1 align="center">📚 Semantic Book Recommender</h1>

<p align="center">
  A smart book recommendation system built with SBERT, Chroma, and Gradio. Describe your vibe, pick a category and tone, and get tailored book picks with thumbnails!
</p>

<p align="center">
  <a href="http://your-droplet-ip:7860">Live Demo (Coming Soon)</a> • 
  <a href="https://github.com/iamdanielsuresh/book-recommender">Source Code</a> • 
  <a href="https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata">Dataset</a>
</p>

---

## 🌟 Overview

This project transforms raw book metadata into a semantic recommender system. Starting with 7,000+ books from Kaggle, I explored the data, built a vector search engine, generalized categories with zero-shot classification, added sentiment analysis, and wrapped it in a slick Gradio dashboard. It’s fast, free, and runs locally (or soon on DigitalOcean!).

### Key Features
- **Semantic Search**: Find books by description using SBERT embeddings.
- **Category Filters**: Generalized categories (e.g., Fiction, Children’s) via zero-shot LLM.
- **Emotional Tones**: Sort by feelings (Happy, Sad, etc.) with sentiment analysis.
- **Interactive UI**: Gradio dashboard with thumbnails and clickable source code.

---

## 🛠️ How It Works

Here’s the journey from raw data to dope dashboard:

### 1. Data Exploration (`data-exploration.ipynb`)
- **Dataset**: [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) (~5,197 usable entries after cleaning).
- **Steps**:
  - Loaded `books_cleaned.csv` (ISBN13, title, authors, categories, description).
  - Created `tagged_description.txt` (ISBN13 + description per line).
  - Explored `categories`—found messy, varied labels needing generalization.

### 2. Vector Search (`vector-search.ipynb`)
- **Goal**: Search books by description similarity.
- **Tech**:
  - **SBERT**: Swapped slow Ollama (`tinyllama`: 17m, `mistral`: 132m) for `all-MiniLM-L6-v2` (~1-5m build).
  - **Chroma**: Persistent vector store (`./chroma_db`).
- **Process**:
  - Loaded `tagged_description.txt` into Chroma with SBERT embeddings.
  - Tested queries like “A book about to teach children about nature”—way better than Ollama!

### 3. Text Classification (`text-classification.ipynb`)
- **Goal**: Generalize messy `categories` (e.g., “Young Adult Sci-Fi” → “Fiction”).
- **Tech**: Zero-shot classification with Hugging Face’s `bart-large-mnli` (faster than `mistral`).
- **Steps**:
  - Defined broad categories: Fiction, Nonfiction, Children’s, Science, Fantasy, Mystery.
  - Mapped raw `categories` to `simple_categories` (~5-15m).
  - Saved to `books_with_emotions.csv`.

### 4. Sentiment Analysis (`sentiment-analysis.ipynb`)
- **Goal**: Add emotional tones to recommendations.
- **Tech**: Likely used a pre-trained model (e.g., DistilBERT) for emotions.
- **Steps**:
  - Analyzed descriptions for joy, surprise, anger, fear, sadness.
  - Added scores to `books_with_emotions.csv` (e.g., `joy: 0.8`, `sadness: 0.2`).

### 5. Gradio Dashboard (`gradio-dashboard.py`)
- **Goal**: Tie it all together in a user-friendly UI.
- **Tech**: Gradio with SBERT + Chroma backend.
- **Features**:
  - Input: Description query, category dropdown, tone dropdown.
  - Output: Gallery of 16 books with thumbnails, titles, authors, and truncated descriptions.
  - Bonus: “Click here for source code” link to this repo!

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git

