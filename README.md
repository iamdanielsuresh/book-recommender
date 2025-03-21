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
- **Purpose**: Clean and explore the dataset.
- **Details**:
  - Load and preprocess the dataset (`books_cleaned.csv`).
  - Generate a tagged description file (`tagged_description.txt`).
  - Analyze and identify inconsistencies in book categories.
  

### 2. Vector Search (`vector-search.ipynb`)
- **Purpose**: Build a semantic search engine for book descriptions.
- **Details**:
  - Use SBERT (`all-MiniLM-L6-v2`) to encode book descriptions into embeddings.
  - Store and query embeddings using Chroma for fast similarity searches.
  

### 3. Text Classification (`text-classification.ipynb`)
- **Purpose**: Simplify and generalize book categories (`books_with_categories.csv`)
- **Details**:
  - Apply zero-shot classification using Hugging Face’s `bart-large-mnli` to map raw categories to broader ones.
  - Save the updated dataset with simplified categories.
  

### 4. Sentiment Analysis (`sentiment-analysis.ipynb`)
- **Purpose**: Add emotional tone analysis to book descriptions((`books_with_emotions.csv`).
- **Details**:
  - Use Hugging Face’s `DistilBERT` to analyze book descriptions for emotions like joy, sadness, and anger.
  - Append emotion scores to the dataset for personalized recommendations.
  

### 5. Gradio Dashboard (`app.py`)
- **Purpose**: Create an interactive user interface for book recommendations.
- **Details**:
  - Integrate semantic search, category filters, and tone analysis.
  - Display results with thumbnails, titles, and descriptions in a user-friendly format.

---

## 📋 Requirements

- Python 3.9+
- pip (Python package manager)
- Internet connection (for downloading pre-trained models)

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iamdanielsuresh/book-recommender.git
   cd book-recommender
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. Run the Gradio dashboard:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:7860
   ```

3. Enter a description, select a category and tone, and get personalized book recommendations!

---

## 📁 Project Structure

- `data-exploration.ipynb`: Data cleaning and exploration.
- `vector-search.ipynb`: Building the semantic search engine.
- `text-classification.ipynb`: Generalizing book categories.
- `sentiment-analysis.ipynb`: Adding emotional tone analysis.
- `app.py`: Gradio dashboard for user interaction.

---

## 🎯 How to Use

1. Enter a description of the type of book you're looking for.
2. Select a category (Fiction, Non-Fiction, Children's, etc.).
3. Choose an emotional tone (Happy, Surprising, Suspenseful, etc.).
4. Click "Find recommendations."
5. Browse through the gallery of recommended books.

---

## 🤝 Contributing

1. Fork the repository.
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Kaggle Dataset by Dylan Castillo.
- Sentence-Transformers for the SBERT model.
- Gradio for the web interface.
- Hugging Face for transformer models.

<p align="center"> Made with ❤️ by <a href="https://github.com/iamdanielsuresh">Daniel Suresh</a> </p>


