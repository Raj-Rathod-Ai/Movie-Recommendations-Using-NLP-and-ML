import pandas as pd
import numpy as np
from gemini_helper import get_gemini_fallback_recommendations

def get_recommendations(movie_title: str, df: pd.DataFrame, indices: pd.Series, tfidf_matrix, top_n: int = 10, api_key: str = None) -> tuple:
    """
    Get top N relevant movie recommendations combining TF-IDF cosine similarity with popularity weighting.
    Filters out obscure zero-rating movies and falls back to Gemini API if results are poor.
    """
    clean_title = movie_title.strip()
    
    # 1. Look up movie index
    idx = None
    if clean_title in indices:
        idx = indices[clean_title]
        if isinstance(idx, (pd.Series, np.ndarray)):
            idx = idx.iloc[0] if len(idx) > 0 else None
    else:
        matching_rows = df[df['title'].str.lower() == clean_title.lower()]
        if not matching_rows.empty:
            idx = matching_rows.index[0]

    # 2. Hybrid TF-IDF + Popularity Ranking
    if idx is not None and idx < len(df):
        try:
            target_vec = tfidf_matrix[idx]
            sim_scores = tfidf_matrix.dot(target_vec.T).toarray().flatten()
            
            # Numeric popularity and rating conversion
            popularity = pd.to_numeric(df['popularity'], errors='coerce').fillna(0).values
            votes = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0).values
            
            # Normalize popularity (log scale to avoid single outlier dominance)
            log_pop = np.log1p(popularity)
            norm_pop = (log_pop - log_pop.min()) / (log_pop.max() - log_pop.min() + 1e-5)
            
            # Hybrid score = 55% Content Similarity + 45% Popularity
            hybrid_scores = (0.55 * sim_scores) + (0.45 * norm_pop)
            
            # Filter out 0-vote obscure movies unless similarity is exceptionally high (>0.6)
            valid_mask = (votes > 3.0) | (sim_scores > 0.4)
            hybrid_scores[~valid_mask] *= 0.1
            
            # Rank candidates, excluding the query movie itself
            sorted_indices = np.argsort(hybrid_scores)[::-1]
            top_indices = [i for i in sorted_indices if i != idx][:top_n]
            
            rec_df = df.iloc[top_indices].copy()
            rec_df['similarity_score'] = sim_scores[top_indices]
            
            # If recommendations are still empty or invalid, fallback to Gemini
            if len(rec_df) > 0:
                return rec_df, 'local'
        except Exception as e:
            print(f"Error computing hybrid similarity: {e}")

    # 3. Fallback to Gemini AI if movie is not found or local ML results are poor
    print(f"Triggering Gemini AI recommendation engine for query: '{clean_title}'...")
    ai_movies = get_gemini_fallback_recommendations(clean_title, top_n=top_n, api_key=api_key)
    
    if ai_movies:
        ai_df = pd.DataFrame(ai_movies)
        for col in ['title', 'overview', 'genres', 'vote_average', 'popularity', 'release_year']:
            if col not in ai_df.columns:
                ai_df[col] = "N/A"
        ai_df['similarity_score'] = 0.95
        return ai_df, 'gemini'

    # Fallback to top popular films from dataset
    pop_df = df.sort_values(by='popularity', ascending=False).head(top_n).copy()
    pop_df['similarity_score'] = 0.50
    return pop_df, 'popular_fallback'
