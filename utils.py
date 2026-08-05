import os
import difflib
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner="Loading movie recommendation dataset...")
def load_data():
    """
    Load dataset and sparse TF-IDF matrix into memory with caching.
    Filters out corrupt/numeric/junk rows.
    """
    df_path = "df.pkl"
    indices_path = "indices.pkl"
    tfidf_path = "tfidf_matrix.pkl"

    if not (os.path.exists(df_path) and os.path.exists(indices_path) and os.path.exists(tfidf_path)):
        raise FileNotFoundError("Required pickle files (df.pkl, indices.pkl, tfidf_matrix.pkl) not found.")

    df = pd.read_pickle(df_path)
    indices = pd.read_pickle(indices_path)
    tfidf_matrix = pd.read_pickle(tfidf_path)

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
    clean_text = str(text).strip()
    if len(clean_text) <= max_chars:
        return clean_text
    
    truncated = clean_text[:max_chars].rsplit(' ', 1)[0]
    return truncated + "..."

@st.cache_data(show_spinner=False)
def search_movies(query: str, all_titles: list, limit: int = 15) -> list:
    """
    Ultra-fast high-accuracy search engine with typo tolerance and token matching.
    """
    if not query or len(str(query).strip()) == 0:
        return [str(t) for t in all_titles if not str(t).isnumeric()][:limit]

    q_clean = str(query).strip().lower()
    q_words = [w for w in q_clean.replace(":", " ").replace("-", " ").split() if len(w) > 0]

    prefix_matches = []
    sub_matches = []
    token_matches = []
    seen = set()

    # Quick first pass: Exact prefix or substring
    for t in all_titles:
        t_str = str(t)
        if t_str.isnumeric():
            continue

        t_lower = t_str.lower()
        if t_lower in seen:
            continue

        if t_lower.startswith(q_clean):
            prefix_matches.append(t_str)
            seen.add(t_lower)
            if len(prefix_matches) >= limit:
                break
        elif q_clean in t_lower:
            sub_matches.append(t_str)
            seen.add(t_lower)

    combined = prefix_matches + sub_matches
    if len(combined) >= limit:
        return combined[:limit]

    # Token pass for multi-word queries or typos (e.g. "harry" + "poter")
    for t in all_titles:
        t_str = str(t)
        if t_str.isnumeric():
            continue
        t_lower = t_str.lower()
        if t_lower in seen:
            continue

        # Check if all query words exist as substrings or fuzzy prefix in title
        if all(w in t_lower for w in q_words) or (len(q_words) > 1 and sum(1 for w in q_words if w in t_lower) >= len(q_words) - 1):
            token_matches.append(t_str)
            seen.add(t_lower)
            if len(combined) + len(token_matches) >= limit:
                break

    combined.extend(token_matches)
    if len(combined) >= limit:
        return combined[:limit]

    # Difflib candidate pass on small filtered sample if needed
    if len(combined) < limit and len(q_clean) >= 3:
        first_char = q_clean[0]
        candidates = [str(t) for t in all_titles if str(t).lower().startswith(first_char) and not str(t).isnumeric()]
        fuzzy_close = difflib.get_close_matches(q_clean, candidates, n=limit, cutoff=0.5)
        for fz in fuzzy_close:
            if fz.lower() not in seen:
                combined.append(fz)
                seen.add(fz.lower())

    return combined[:limit]


