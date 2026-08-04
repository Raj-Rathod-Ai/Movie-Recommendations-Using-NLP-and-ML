import os
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner="Loading movie recommendation dataset...")
def load_data():
    """
    Load dataset and sparse TF-IDF matrix into memory with caching.
    """
    df_path = "df.pkl"
    indices_path = "indices.pkl"
    tfidf_path = "tfidf_matrix.pkl"

    if not (os.path.exists(df_path) and os.path.exists(indices_path) and os.path.exists(tfidf_path)):
        raise FileNotFoundError("Required pickle files (df.pkl, indices.pkl, tfidf_matrix.pkl) not found.")

    df = pd.read_pickle(df_path)
    indices = pd.read_pickle(indices_path)
    tfidf_matrix = pd.read_pickle(tfidf_path)

    # Ensure clean indexing
    df = df.reset_index(drop=True)
    
    return df, indices, tfidf_matrix

def format_movie_title(title: str, year=None) -> str:
    """
    Format movie title with proper capitalization and optional release year.
    """
    if not isinstance(title, str):
        return "Unknown Movie"
    
    clean_title = title.strip().title()
    if year:
        return f"{clean_title} ({year})"
    return clean_title

def truncate_text(text: str, max_chars: int = 110) -> str:
    """
    Truncate text at word boundary without chopping words in half.
    """
    if not text or not isinstance(text, str):
        return ""
    clean_text = text.strip()
    if len(clean_text) <= max_chars:
        return clean_text
    
    # Truncate at nearest word boundary
    truncated = clean_text[:max_chars].rsplit(' ', 1)[0]
    return truncated + "..."

@st.cache_data(show_spinner=False)
def search_movies(query: str, all_titles: list, limit: int = 15) -> list:
    """
    Fast case-insensitive search and autocomplete matching.
    """
    if not query or len(query.strip()) == 0:
        return all_titles[:limit]

    q_clean = query.strip().lower()
    
    # Prefix matches first, then substring matches
    prefix_matches = []
    sub_matches = []

    for t in all_titles:
        t_lower = str(t).lower()
        if t_lower.startswith(q_clean):
            prefix_matches.append(t)
            if len(prefix_matches) >= limit:
                break
        elif q_clean in t_lower:
            sub_matches.append(t)

    results = prefix_matches + sub_matches
    return results[:limit]
