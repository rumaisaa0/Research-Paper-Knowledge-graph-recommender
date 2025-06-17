import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

import google.generativeai as genai

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA2MUn9IK6VPIE3MVS7sfPOvWt0vRgxbLw"

# Configure Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def ask_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """
    Sends a prompt to the Gemini API and returns the response.
    """
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[ERROR] Gemini API call failed: {e}"

def ask_mistral(prompt: str, model: str = "mistral"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False  # set to True if you want chunked response
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["response"]
# Load datasets
import os
import pandas as pd

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full path to CSV
csv_path = os.path.join(BASE_DIR, "dfg1.csv")

# Read CSV using full path
dfg1 = pd.read_csv(csv_path)
csvpath2=os.path.join(BASE_DIR, "arxiv_metadata.csv")
#dfg1 = pd.read_csv("dfg1.csv")
#arxiv_df = pd.read_csv("arxiv_metadata.csv")
arxiv_df=pd.read_csv(csvpath2)
# Drop missing data
dfg1.dropna(subset=["topic_name"], inplace=True)
arxiv_df.dropna(subset=["title", "abstract"], inplace=True)

# Step 1: Vectorize KG node titles and user query
vectorizer = TfidfVectorizer(stop_words="english")
title_embeddings = vectorizer.fit_transform(dfg1["edge"].astype(str))

# Recommender function: now based on *any* query (not just paper titles)
def recommend_similar_papers(query, top_n=5):
    # Load arxiv metadata
    arxiv_df = pd.read_csv("arxiv_metadata.csv")

# Drop rows with missing title or abstract
    arxiv_df = arxiv_df.dropna(subset=["title", "abstract"])

# Combine title and abstract into a single text column
    arxiv_df["text"] = arxiv_df["title"] + " " + arxiv_df["abstract"]

# Vectorize the combined text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(arxiv_df["text"])
    # Vectorize the input query using the same TF-IDF vectorizer
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity between query and all documents
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get indices of top N similar papers
    top_indices = sim_scores.argsort()[::-1][:top_n]
    
    recommendations = []
    for i in top_indices:
        recommendations.append({
            "title": arxiv_df.iloc[i]["title"],
            "url": arxiv_df.iloc[i].get("pdf_url", "N/A"),
            "abstract": arxiv_df.iloc[i]["abstract"][:300] + "...",  # Truncate abstract
            "similarity_score": round(sim_scores[i], 3)
        })
        
    return recommendations

def get_related_topics_from_query(query, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, title_embeddings).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    return dfg1.iloc[top_indices]["topic_name"].tolist()

# Step 2: Find connected nodes from knowledge graph
def get_connected_nodes(seed_titles):
    connected_nodes = set()
    for _, row in dfg1.iterrows():
        if row["topic_name"] in seed_titles:
            connected_nodes.add(row["node_1"])
            connected_nodes.add(row["node_2"])
    return list(connected_nodes)
# Step 3: Recommend papers from arxiv_df
def recommend_papers_from_topics(topics, top_n=5):
    # Normalize for case-insensitive matching
    arxiv_df["title_lower"] = arxiv_df["title"].str.lower()
    topic_set = set([t.lower() for t in topics])
    
    matched = arxiv_df[arxiv_df["title_lower"].isin(topic_set)]
    matched = matched.drop_duplicates(subset="title")
    
   
    results = []
    for _, row in matched.head(top_n).iterrows():
        results.append({
            "title": row["title"],
            "url": row.get("pdf_url", "N/A"),
            "abstract": row["abstract"][:300] + "...",
        })
    
    return results

def kg_based_recommender_from_query(query, top_n=5):
    print(f"\n Finding topics related to: \"{query}\"")
    
    related_topics = get_related_topics_from_query(query)
    print(f"Matched Topics: {related_topics}")

    connected_nodes = get_connected_nodes(related_topics)
    
    # Now get the corresponding topics for those nodes
    related_topic_names = dfg1[dfg1["node_1"].isin(connected_nodes) | dfg1["node_2"].isin(connected_nodes)]["topic_name"].unique()
    print(f"Related Topics via Connected Nodes: {list(related_topic_names)}")

    recommendations = recommend_papers_from_topics(related_topic_names, top_n=top_n)
    return recommendations


def get_query_embedding(query, vectorizer):
    return vectorizer.transform([query])

def evaluate_recommendations(query, recommended_papers, vectorizer, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = []

    for paper in recommended_papers:
        paper_vec = vectorizer.transform([paper['title']+""+paper['abstract']])
        similarity = cosine_similarity(query_vec, paper_vec)
        similarities.append(similarity.flatten()[0])

    # Attach similarity scores
    scored_papers = []
    for paper, sim in zip(recommended_papers, similarities):
        paper_copy = paper.copy()
        paper_copy['similarity'] = sim
        scored_papers.append(paper_copy)

    sorted_papers = sorted(scored_papers, key=lambda x: x['similarity'], reverse=True)

    for s in sorted_papers[:top_n]:
        print(f"Title: {s['title']}\nURL: {s['url']}\nSimilarity: {s['similarity']:.4f}\nAbstract: {s['abstract']}\n{'-'*80}")

    return sorted_papers


def recommender(query):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(dfg1["topic_name"].astype(str))  # Fit the vectorizer on the dataset

# Step 1: Vectorize KG node titles and user query
    title_embeddings = vectorizer.transform(dfg1["topic_name"].astype(str))
    recommended_paperskg=kg_based_recommender_from_query(query, top_n=5)

    recommended_paperstfidf=recommend_similar_papers(query, top_n=5)
# Assuming `recommend_papers_from_query` provides the list of recommended papers
    print("\n\nknowlegde graph recommender: ")

    evaluate_recommendations(query,recommended_paperskg,vectorizer)    
    print("\n\ntfidf recommender: ")
    for r in recommended_paperstfidf:
      print(f"Title: {r['title']}\nURL: {r['url']}\nSimilarity: {r['similarity_score']}\nAbstract: {r['abstract']}\n{'-'*80}")
    all_recommendations = recommended_paperstfidf+recommended_paperskg 

    # Prepare documents for RAG (each as a dictionary with 'title', 'url', 'abstract')
    documents = []
    for paper in all_recommendations:
        document = {
            'title': paper['title'],
            'url': paper['url'],
            'content': paper['abstract']  # We will treat abstract as the content
        }
        documents.append(document)

    return documents
def retrieve_documents(query, top_n=5):
    # Use your recommender system
    return recommender(query)

def build_prompt(query, retrieved_papers):
    context = "\n\n".join([paper['content'] for paper in retrieved_papers])
    prompt = f"""
    You are a helpful research assistant.
    Given the following paper abstracts, answer the user's question.

    User's Question: {query}

    Paper Abstracts:
    {context}

    Answer:"""
    return prompt


def generate_answer(prompt):
    #return ask_mistral(prompt)
    return ask_gemini(prompt)

# Full RAG pipeline
def rag_pipeline(query):
    papers = retrieve_documents(query)
    prompt = build_prompt(query, papers)
    answer = generate_answer(prompt)
    return papers, answer
