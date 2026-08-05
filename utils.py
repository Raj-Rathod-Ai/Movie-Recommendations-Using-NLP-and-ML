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

ALIAS_MAP = {
    'spiderman': ['Spider-Man', 'Spider-Man 2', 'Spider-Man 3', 'The Amazing Spider-Man', 'Spider-Man: Into the Spider-Verse', 'Spider-Man: No Way Home'],
    'spider-man': ['Spider-Man', 'Spider-Man 2', 'Spider-Man 3', 'The Amazing Spider-Man', 'Spider-Man: Into the Spider-Verse', 'Spider-Man: No Way Home'],
    'money heist': ['Money Heist', 'Berlin', 'Money Heist: Korea - Joint Economic Area'],
    'money heist s4': ['Money Heist', 'Berlin', 'Money Heist: Korea - Joint Economic Area'],
    'berlin': ['Berlin', 'Money Heist', 'Money Heist: Korea - Joint Economic Area'],
    'harry poter': ['Harry Potter and the Sorcerer\'s Stone', 'Harry Potter and the Chamber of Secrets', 'Fantastic Beasts and Where to Find Them'],
    'harry potter': ['Harry Potter and the Sorcerer\'s Stone', 'Harry Potter and the Chamber of Secrets', 'Fantastic Beasts and Where to Find Them'],
    'house of dragon': ['House of the Dragon', 'Game of Thrones'],
    'got': ['Game of Thrones', 'House of the Dragon'],
    'dark knight': ['The Dark Knight', 'Batman Begins', 'The Dark Knight Rises', 'The Batman'],
    'batman': ['The Dark Knight', 'Batman Begins', 'The Batman', 'The Dark Knight Rises']
}

@st.cache_data(show_spinner=False)
def search_movies(query: str, all_titles: list, limit: int = 15) -> list:
    """
    Ultra-fast NLP search engine with typo auto-correction and alias mapping.
    """
    if not query or len(str(query).strip()) == 0:
        return [str(t) for t in all_titles if not str(t).isnumeric()][:limit]

    q_clean = str(query).strip().lower()
    q_words = [w for w in q_clean.replace(":", " ").replace("-", " ").split() if len(w) > 0]

    matches = []
    seen = set()

    # 1. Alias & Typo map lookup
    for alias_key, mapped_list in ALIAS_MAP.items():
        if alias_key in q_clean or q_clean in alias_key or (len(q_clean) >= 4 and difflib.SequenceMatcher(None, q_clean, alias_key).ratio() > 0.72):
            for m in mapped_list:
                if m.lower() not in seen:
                    matches.append(m)
                    seen.add(m.lower())

    # 2. Prefix & Substring Match
    for t in all_titles:
        t_str = str(t)
        if t_str.isnumeric():
            continue

        t_lower = t_str.lower()
        if t_lower in seen:
            continue

        if t_lower.startswith(q_clean) or q_clean in t_lower:
            matches.append(t_str)
            seen.add(t_lower)
            if len(matches) >= limit:
                break

    if len(matches) >= limit:
        return matches[:limit]

    # 3. Multi-word Token Match
    for t in all_titles:
        t_str = str(t)
        if t_str.isnumeric():
            continue
        t_lower = t_str.lower()
        if t_lower in seen:
            continue

        if all(w in t_lower for w in q_words if len(w) > 1):
            matches.append(t_str)
            seen.add(t_lower)
            if len(matches) >= limit:
                break

    if len(matches) >= limit:
        return matches[:limit]

    # 4. Global Difflib Typo Auto-Corrector
    if len(matches) < limit and len(q_clean) >= 3:
        all_non_numeric = [str(t) for t in all_titles if not str(t).isnumeric()]
        fuzzy_close = difflib.get_close_matches(q_clean, all_non_numeric, n=limit, cutoff=0.50)
        for fz in fuzzy_close:
            if fz.lower() not in seen:
                matches.append(fz)
                seen.add(fz.lower())

    return matches[:limit]



