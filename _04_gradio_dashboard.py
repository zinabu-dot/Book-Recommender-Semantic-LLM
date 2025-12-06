import os
from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
# We use the largest thumbnail for book covers from google for better resolution.
books['large_thumbnail'] = books['thumbnail'] + "&fife=800"
# Now we replace the missing covers with cover not found image (cover-not-found.png)
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), "cover-not-found.png", books['large_thumbnail'])

""" For semantic book recommendations """
raw_docs = TextLoader("tagged_description.txt",  encoding="utf-8").load() # We read/load tagged book description data to text loader
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0) # we create a chunk of the description of each book (separated by new line)
docs = text_splitter.split_documents(raw_docs) # Splitting comes into effect

TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": TOKEN}
)
db_books = Chroma.from_documents(documents=docs, embedding=embeddings) # We convert the chunks into document embeddings and store that to the chroma vector database. use OpenAIEmbeddings() if there is openai subscription

def book_semantic_recommendations(query: str, category:str, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(rec[0].page_content.split(" ")[0].replace('"', "")) for rec in recs]
    book_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["new_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def book_recommender(query: str, category: str, tone: str) -> List[Tuple[Any, str]]:
    recommendations = book_semantic_recommendations(query, category, tone)

    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30])

        # Add the ellipsis if the description was truncated
        if len(truncated_desc_split) > 30:
            truncated_description += "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:- 1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


categories = ["All"] + sorted(books["new_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr. Textbox(label = "Please enter a description of a book:",
                                    placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    recommendation_gallery = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=book_recommender,
                        inputs=[user_query, category_dropdown, tone_dropdown], outputs=recommendation_gallery)


if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Glass())