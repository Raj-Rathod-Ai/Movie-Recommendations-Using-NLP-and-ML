import pandas as pd
import numpy as np
import difflib
from insights_helper import get_smart_fallback_recommendations

def get_recommendations(movie_title: str, df: pd.DataFrame, indices: pd.Series, tfidf_matrix, top_n: int = 10, access_key: str = None) -> tuple:
    """
    Get top N relevant movie & series recommendations based on content similarity.
    Filters out obscure, numeric, or corrupt rows and falls back cleanly to smart AI recommendations.
    """
    clean_title = movie_title.strip()
    if not clean_title:
        clean_title = "Harry Potter and the Philosopher's Stone"
    
    # 1. Look up movie index using exact match, case-insensitive match, or strict fuzzy match
    idx = None
    if clean_title in indices:
        idx = indices[clean_title]
        if isinstance(idx, (pd.Series, np.ndarray)):
            idx = idx.iloc[0] if len(idx) > 0 else None
    else:
        # Case-insensitive title lookup
        matching_rows = df[df['title'].str.lower() == clean_title.lower()]
        if not matching_rows.empty:
            idx = matching_rows.index[0]
        else:
            # Strict fuzzy match lookup in local dataset (cutoff=0.72 to avoid matching unrelated 'House of...' titles)
            non_numeric_titles = [str(t) for t in df['title'].tolist() if not str(t).isnumeric()]
            close_matches = difflib.get_close_matches(clean_title, non_numeric_titles, n=1, cutoff=0.72)
            if close_matches:
                match_title = close_matches[0]
                m_rows = df[df['title'] == match_title]
                if not m_rows.empty:
                    idx = m_rows.index[0]

    # 2. High-Accuracy Content Similarity Ranking
    if idx is not None and idx < len(df):
        try:
            target_vec = tfidf_matrix[idx]
            sim_scores = tfidf_matrix.dot(target_vec.T).toarray().flatten()
            
            popularity = pd.to_numeric(df['popularity'], errors='coerce').fillna(0).values
            votes = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0).values
            is_numeric_title = df['title'].astype(str).str.isnumeric().values
            
            log_pop = np.log1p(popularity)
            norm_pop = (log_pop - log_pop.min()) / (log_pop.max() - log_pop.min() + 1e-5)
            
            # Prioritize similarity (85%) over popularity (15%) for exact thematic match
            hybrid_scores = (0.85 * sim_scores) + (0.15 * norm_pop)
            
            # Penalize numeric titles, unrated titles, or self match
            valid_mask = (votes > 3.0) & (~is_numeric_title)
            hybrid_scores[~valid_mask] *= 0.05
            
            sorted_indices = np.argsort(hybrid_scores)[::-1]
            top_indices = [i for i in sorted_indices if i != idx and not str(df.iloc[i]['title']).isnumeric()][:top_n]
            
            rec_df = df.iloc[top_indices].copy()
            rec_df['similarity_score'] = sim_scores[top_indices]
            
            # Ensure top recommendations have strong similarity
            if len(rec_df) > 0 and sim_scores[top_indices[0]] > 0.08:
                return rec_df, 'local'
        except Exception as e:
            print(f"Error computing similarity: {e}")

    # 3. Smart Fallback recommendations for TV Shows or out-of-catalog items
    smart_movies = get_smart_fallback_recommendations(clean_title, top_n=top_n, access_key=access_key)
    
    if smart_movies:
        smart_df = pd.DataFrame(smart_movies)
        for col in ['title', 'overview', 'genres', 'vote_average', 'popularity', 'release_year']:
            if col not in smart_df.columns:
                smart_df[col] = "N/A"
        smart_df['similarity_score'] = 0.95
        smart_df = smart_df[~smart_df['title'].astype(str).str.isnumeric()]
        if not smart_df.empty:
            return smart_df, 'smart_fallback'

    # Filtered popular fallback as last resort (Guaranteed NO numeric titles)
    clean_df = df[(~df['title'].astype(str).str.isnumeric()) & (pd.to_numeric(df['vote_average'], errors='coerce') > 5.0)].copy()
    pop_df = clean_df.sort_values(by='popularity', ascending=False).head(top_n).copy()
    pop_df['similarity_score'] = 0.50
    return pop_df, 'popular_fallback'



