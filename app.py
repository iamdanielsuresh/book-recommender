import pandas as pd
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import shutil

# Custom SBERT embeddings
class SBERTEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, batch_size=32).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Load books dataset
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books["thumbnail"] + "&fife=w800"
books['large_thumbnail'] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

# Initialize embeddings
embeddings = SBERTEmbeddings()

# Load or build vector store with persistence
persist_dir = "./chroma_db"
if not os.path.exists(persist_dir):
    print("Building vector store...")
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    texts = [doc.page_content for doc in documents]
    db_books = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
else:
    print("Loading existing vector store...")
    db_books = Chroma(persist_directory=persist_dir, embedding_function=embeddings)




# Recommendation function
def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [rec.page_content.split()[0].strip('"') for rec in recs]
    book_recs = books[books["isbn13"].astype(str).isin(books_list)].copy()
    
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    
    if tone and tone != "All":
        tone_map = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness"
        }
        if tone in tone_map:
            book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)
    
    return book_recs.head(final_top_k)

# Format recommendations for Gradio
def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = " ".join(row["description"].split()[:30]) + "..."
        authors_split = row["authors"].split(";")
        authors_str = (f"{authors_split[0]} and {authors_split[1]}" if len(authors_split) == 2 else
                       f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}" if len(authors_split) > 2 else
                       row["authors"])
        caption = f"{row['title']} by {authors_str}: {description}"
        results.append((row["large_thumbnail"], caption))
    return results

# Setup UI options
categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


github_repo_link = "https://github.com/iamdanielsuresh/book-recommender"  

# Gradio interface
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")
    gr.Markdown(f"Source code: [Click here for source code]({github_repo_link})")
    with gr.Row():
        user_query = gr.Textbox(label="Enter a book description:", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotional Tone:", value="All")
        submit_button = gr.Button("Find Recommendations")
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns=8, rows=2)
    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=7860)