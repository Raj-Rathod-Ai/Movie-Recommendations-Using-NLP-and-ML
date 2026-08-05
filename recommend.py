import pandas as pd
import numpy as np
import difflib
import re
from insights_helper import get_smart_fallback_recommendations

# Canonical Title Alias Mapping to convert queries to official dataset titles
TITLE_CANONICAL_MAP = {
    'avenger': 'The Avengers',
    'avengers': 'The Avengers',
    'spiderman': 'Spider-Man',
    'spider-man': 'Spider-Man',
    'ironman': 'Iron Man',
    'iron man': 'Iron Man',
    'thor': 'Thor',
    'batman': 'The Dark Knight',
    'dark knight': 'The Dark Knight',
    'harry potter': "Harry Potter and the Sorcerer's Stone",
    'harry poter': "Harry Potter and the Sorcerer's Stone",
    'money heist': 'Money Heist',
    'berlin': 'Berlin',
    'a aa': 'A Aa',
    'avatar': 'Avatar',
    'got': 'Game of Thrones',
    'house of dragon': 'House of the Dragon'
}

def clean_title_string(title: str) -> str:
    """
    Remove language/year suffix tags like (IN Hindi), (IN Telugu), (2016) from query title.
    e.g. 'A Aa (IN Hindi)' -> 'A Aa'
    """
    if not title or not isinstance(title, str):
        return ""
    cleaned = re.sub(r'\s*\((?:IN\s*)?[A-Za-z0-9\s]+\)\s*$', '', title, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(\d{4}\)\s*$', '', cleaned)
    t_str = cleaned.strip() or title.strip()
    
    # Map alias if available
    t_lower = t_str.lower()
    if t_lower in TITLE_CANONICAL_MAP:
        return TITLE_CANONICAL_MAP[t_lower]
    return t_str

def is_valid_title_match(query_title: str, match_title: str) -> bool:
    """
    Validate that candidate match title actually shares core words with user query.
    """
    clean_q = clean_title_string(query_title)
    q_words = [w for w in clean_q.lower().replace(':', ' ').replace('-', ' ').split() if len(w) > 1]
    m_words = [w for w in match_title.lower().replace(':', ' ').replace('-', ' ').split() if len(w) > 1]
    if not q_words:
        return clean_q.lower() == match_title.lower()
    matched_count = sum(1 for qw in q_words if any(qw == mw or difflib.SequenceMatcher(None, qw, mw).ratio() > 0.82 for mw in m_words))
    return (matched_count / len(q_words)) >= 0.75

def get_recommendations(movie_title: str, df: pd.DataFrame, indices: pd.Series, tfidf_matrix, top_n: int = 10, lang: str = "all", access_key: str = None) -> tuple:
    """
    Get top N relevant movie & series recommendations based on content similarity and selected language.
    Filters out obscure, numeric, or corrupt rows and falls back cleanly to smart AI recommendations.
    """
    raw_title = movie_title.strip() if movie_title else "The Avengers"
    clean_title = clean_title_string(raw_title)
    
    # 1. Look up movie index using clean title
    idx = None
    known_modern_shows = ["money heist", "berlin", "house of dragon", "house of the dragon", "game of thrones", "breaking bad", "stranger things", "peaky blinders", "lupin", "prison break", "narcos", "squid game"]

    if clean_title.lower() in known_modern_shows:
        idx = None
    elif clean_title in indices:
        matched_idx = indices[clean_title]
        if isinstance(matched_idx, (pd.Series, np.ndarray)):
            matched_idx = matched_idx.iloc[0] if len(matched_idx) > 0 else None
        
        if matched_idx is not None and matched_idx < len(df):
            row_pop = pd.to_numeric(df.iloc[matched_idx].get('popularity', 0), errors='coerce') or 0
            row_votes = pd.to_numeric(df.iloc[matched_idx].get('vote_average', 0), errors='coerce') or 0
            if row_pop < 4.0 and row_votes < 4.0:
                idx = None
            else:
                idx = matched_idx
    else:
        # Case-insensitive title lookup
        matching_rows = df[df['title'].str.lower() == clean_title.lower()]
        if not matching_rows.empty:
            matched_idx = matching_rows.index[0]
            row_pop = pd.to_numeric(df.iloc[matched_idx].get('popularity', 0), errors='coerce') or 0
            if row_pop < 4.0:
                idx = None
            else:
                idx = matched_idx
        else:
            # Strict validated fuzzy match lookup
            non_numeric_titles = [str(t) for t in df['title'].tolist() if not str(t).isnumeric()]
            close_matches = difflib.get_close_matches(clean_title, non_numeric_titles, n=3, cutoff=0.65)
            for cm in close_matches:
                if is_valid_title_match(clean_title, cm):
                    m_rows = df[df['title'] == cm]
                    if not m_rows.empty:
                        idx = m_rows.index[0]
                        break

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
            
            # Hybrid Score = 88% Content Similarity + 12% Popularity
            hybrid_scores = (0.88 * sim_scores) + (0.12 * norm_pop)
            
            # Franchise Collection Boosting
            target_title = str(df.iloc[idx].get('title', '')).lower()
            core_words = [w for w in target_title.replace(':', ' ').replace('-', ' ').split() if len(w) > 1 and w not in ['the', 'movie', 'part', 'and']]
            if core_words:
                primary_kw = core_words[0]
                franchise_mask = df['title'].astype(str).str.lower().str.contains(primary_kw).fillna(False).values
                hybrid_scores += (0.40 * franchise_mask)

            target_genres = str(df.iloc[idx].get('genres', '')).lower()
            is_target_animation = 'animation' in target_genres

            valid_mask = (votes > 4.0) & (~is_numeric_title)

            # Apply language filter if specific language selected
            if lang and lang.lower() != 'all':
                lang_mask = (df['original_language'].astype(str).str.lower() == lang.lower())
                valid_mask = valid_mask & lang_mask

            hybrid_scores[~valid_mask] *= 0.05
            
            sorted_indices = np.argsort(hybrid_scores)[::-1]
            
            top_indices = []
            for i in sorted_indices:
                if i == idx:
                    continue
                cand_title = str(df.iloc[i]['title'])
                if cand_title.isnumeric() or cand_title.lower().startswith('list of'):
                    continue
                cand_genres = str(df.iloc[i].get('genres', '')).lower()
                if not is_target_animation and 'animation' in cand_genres and sim_scores[i] < 0.2:
                    continue
                top_indices.append(i)
                if len(top_indices) >= top_n:
                    break

            rec_df = df.iloc[top_indices].copy()
            rec_df['similarity_score'] = sim_scores[top_indices]
            
            # Require solid similarity score (> 0.16) or direct title prefix match
            top_sim = sim_scores[top_indices[0]] if top_indices else 0.0
            is_franchise_hit = df.iloc[top_indices[0]]['title'].lower().startswith(core_words[0]) if (top_indices and core_words) else False
            
            if len(rec_df) > 0 and (top_sim >= 0.16 or is_franchise_hit):
                return rec_df, 'local'

        except Exception as e:
            print(f"Error computing similarity: {e}")

    # 3. Smart Fallback recommendations for TV Shows, low-similarity items, or out-of-catalog titles
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

    # Filtered popular fallback as last resort
    clean_df = df[(~df['title'].astype(str).str.isnumeric()) & (pd.to_numeric(df['vote_average'], errors='coerce') > 5.0)].copy()
    pop_df = clean_df.sort_values(by='popularity', ascending=False).head(top_n).copy()
    pop_df['similarity_score'] = 0.50
    return pop_df, 'popular_fallback'
