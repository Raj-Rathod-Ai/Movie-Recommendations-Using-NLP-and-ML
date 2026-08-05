import pandas as pd
import numpy as np
import difflib
from insights_helper import get_smart_fallback_recommendations

def is_valid_title_match(query_title: str, match_title: str) -> bool:
    """
    Validate that candidate match title actually shares core words with user query.
    Prevents matching unrelated movies (e.g. Coin Heist for Money Heist, Merlin for Berlin).
    """
    q_words = [w for w in query_title.lower().replace(':', ' ').replace('-', ' ').split() if len(w) > 2]
    m_words = [w for w in match_title.lower().replace(':', ' ').replace('-', ' ').split() if len(w) > 2]
    if not q_words:
        return query_title.lower() == match_title.lower()
    matched_count = sum(1 for qw in q_words if any(qw == mw or difflib.SequenceMatcher(None, qw, mw).ratio() > 0.82 for mw in m_words))
    return (matched_count / len(q_words)) >= 0.75

def get_recommendations(movie_title: str, df: pd.DataFrame, indices: pd.Series, tfidf_matrix, top_n: int = 10, lang: str = "all", access_key: str = None) -> tuple:
    """
    Get top N relevant movie & series recommendations based on content similarity and selected language.
    Filters out obscure, numeric, or corrupt rows and falls back cleanly to smart AI recommendations.
    """
    clean_title = movie_title.strip()
    if not clean_title:
        clean_title = "Harry Potter and the Philosopher's Stone"
    
    # 1. Look up movie index using exact match, case-insensitive match, or strict validated fuzzy match
    idx = None
    known_modern_shows = ["money heist", "berlin", "house of dragon", "house of the dragon", "game of thrones", "breaking bad", "stranger things", "peaky blinders", "lupin", "prison break", "narcos", "squid game"]

    # Skip local movie dataset for famous modern TV shows
    if clean_title.lower() in known_modern_shows:
        idx = None
    elif clean_title in indices:
        matched_idx = indices[clean_title]
        if isinstance(matched_idx, (pd.Series, np.ndarray)):
            matched_idx = matched_idx.iloc[0] if len(matched_idx) > 0 else None
        
        # Verify that matched local movie is not an obscure zero-popularity entry overriding a famous modern show
        if matched_idx is not None and matched_idx < len(df):
            row_pop = pd.to_numeric(df.iloc[matched_idx].get('popularity', 0), errors='coerce') or 0
            row_votes = pd.to_numeric(df.iloc[matched_idx].get('vote_average', 0), errors='coerce') or 0
            if row_pop < 5.0 and row_votes < 5.0:
                idx = None
            else:
                idx = matched_idx
    else:
        # Case-insensitive title lookup
        matching_rows = df[df['title'].str.lower() == clean_title.lower()]
        if not matching_rows.empty:
            matched_idx = matching_rows.index[0]
            row_pop = pd.to_numeric(df.iloc[matched_idx].get('popularity', 0), errors='coerce') or 0
            if row_pop < 5.0:
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
            
            # Prioritize similarity (85%) over popularity (15%) for exact thematic match
            hybrid_scores = (0.85 * sim_scores) + (0.15 * norm_pop)
            
            # Franchise / Franchise Collection Boosting (e.g. Spider-Man -> Spider-Man 2, Spider-Man 3)
            target_title = str(df.iloc[idx].get('title', '')).lower()
            core_words = [w for w in target_title.replace(':', ' ').replace('-', ' ').split() if len(w) > 3 and w not in ['the', 'movie', 'part']]
            if core_words:
                primary_kw = core_words[0]
                franchise_mask = df['title'].astype(str).str.lower().str.contains(primary_kw).fillna(False).values
                hybrid_scores += (0.45 * franchise_mask)

            target_genres = str(df.iloc[idx].get('genres', '')).lower()
            is_target_animation = 'animation' in target_genres

            # Penalize numeric titles, unrated titles, or mismatched animation movies
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
                if cand_title.isnumeric():
                    continue
                cand_genres = str(df.iloc[i].get('genres', '')).lower()
                # If target is live action, don't recommend unrelated animated movies
                if not is_target_animation and 'animation' in cand_genres and sim_scores[i] < 0.2:
                    continue
                top_indices.append(i)
                if len(top_indices) >= top_n:
                    break

            rec_df = df.iloc[top_indices].copy()
            rec_df['similarity_score'] = sim_scores[top_indices]
            
            # Require minimum content similarity score (> 0.10) to return local results
            if len(rec_df) > 0 and (sim_scores[top_indices[0]] >= 0.10 or df.iloc[top_indices[0]]['title'].lower().startswith(core_words[0] if core_words else '')):
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



