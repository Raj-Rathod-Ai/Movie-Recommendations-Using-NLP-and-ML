import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import urllib.parse
from dotenv import load_dotenv

# Load environment variables silently
load_dotenv()

from utils import load_data, format_movie_title, search_movies, truncate_text
from recommend import get_recommendations
from poster_helper import fetch_poster_and_details
from gemini_helper import get_gemini_insights

# 1. Page Configuration
st.set_page_config(
    page_title="CinemaVerse | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Inject Custom CSS
def load_css():
    css_file = "style.css"
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# 3. Initialize Session State
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "recently_viewed" not in st.session_state:
    st.session_state.recently_viewed = []
if "selected_movie_detail" not in st.session_state:
    st.session_state.selected_movie_detail = None
if "current_search" not in st.session_state:
    st.session_state.current_search = "Toy Story"

# 4. Load Data (Ultra-fast cached in memory)
try:
    df, indices, tfidf_matrix = load_data()
    all_titles = df['title'].tolist()
except Exception as e:
    st.error(f"Error loading movie dataset: {e}")
    st.stop()

# 5. Clean Sidebar (Saved Favorites & History)
with st.sidebar:
    st.markdown("### ❤️ Saved Movies")
    if st.session_state.favorites:
        for fav in st.session_state.favorites[-6:]:
            st.markdown(f"- **{fav}**")
        if st.button("Clear Favorites", key="clear_favs"):
            st.session_state.favorites = []
            st.rerun()
    else:
        st.caption("No favorites added yet. Click '❤️ Save' on any card.")

    st.markdown("---")
    st.markdown("### 🕒 Recent Searches")
    if st.session_state.recently_viewed:
        for recent in st.session_state.recently_viewed[-6:]:
            st.markdown(f"- {recent}")
    else:
        st.caption("Your search history will appear here.")

# 6. Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎬 CinemaVerse</div>
    <div class="hero-subtitle">Discover your next favorite movie from over 300,000+ films</div>
</div>
""", unsafe_allow_html=True)

# 7. Search & Discovery Controls
col_search, col_count, col_btn1, col_btn2 = st.columns([3, 2, 1, 1])

with col_search:
    user_query = st.text_input("Search for a movie:", value=st.session_state.current_search, placeholder="Type movie title (e.g. Inception, Avatar)...")
    suggestions = search_movies(user_query, all_titles, limit=12)
    selected_from_dropdown = st.selectbox("Select title from catalog:", options=suggestions, index=0 if suggestions else None)

with col_count:
    num_recs = st.select_slider(
        "Recommendations to show:",
        options=[4, 6, 8, 10, 12, 16, 20],
        value=8,
        help="Select how many movie suggestions to generate."
    )

with col_btn1:
    st.markdown("<br>", unsafe_allow_html=True)
    get_rec_btn = st.button("🚀 Recommend", use_container_width=True)

with col_btn2:
    st.markdown("<br>", unsafe_allow_html=True)
    surprise_btn = st.button("🎲 Surprise Me", use_container_width=True)

# Handle Surprise Me
if surprise_btn:
    random_title = random.choice(all_titles)
    st.session_state.current_search = random_title
    selected_movie = random_title
    st.rerun()
else:
    selected_movie = selected_from_dropdown if selected_from_dropdown else user_query

# 8. Recommendation Generation Grid
if get_rec_btn or selected_movie:
    if selected_movie:
        if selected_movie not in st.session_state.recently_viewed:
            st.session_state.recently_viewed.append(selected_movie)

        st.markdown(f"### 🍿 Top {num_recs} Recommendations for **{format_movie_title(selected_movie)}**")
        
        with st.spinner("Generating recommendations..."):
            rec_df, source = get_recommendations(
                selected_movie, df, indices, tfidf_matrix, top_n=num_recs
            )

        # 4-column responsive grid layout
        grid_cols = st.columns(4)

        for idx_pos, (_, row) in enumerate(rec_df.iterrows()):
            if idx_pos >= num_recs:
                break

            m_title = row.get("title", "Unknown Title")
            m_overview = row.get("overview", "")
            m_genres = str(row.get("genres", ""))
            m_rating = str(row.get("vote_average", "N/A"))
            m_pop = str(row.get("popularity", "N/A"))

            # Ultra-fast poster fetch
            meta = fetch_poster_and_details(m_title)
            poster_url = meta["poster_url"]
            year = meta.get("release_year", "")
            formatted_name = format_movie_title(m_title, year)

            col_target = grid_cols[idx_pos % 4]

            with col_target:
                # Unified Card Container with fixed uniform poster size and aligned buttons
                st.markdown(f"""
                <div class="movie-card-container">
                    <div class="movie-poster-wrap">
                        <span class="movie-rating-badge">★ {m_rating[:3] if m_rating != 'N/A' else 'N/A'}</span>
                        <img src="{poster_url}" class="movie-poster-img" alt="{m_title}" />
                    </div>
                    <div class="movie-info">
                        <div class="movie-title">{formatted_name}</div>
                        <div class="movie-genres">{m_genres}</div>
                        <div class="movie-overview">{truncate_text(m_overview, 90)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Card Buttons placed cleanly below card frame
                c_act1, c_act2 = st.columns(2)
                with c_act1:
                    if st.button("Details", key=f"det_{idx_pos}_{m_title}"):
                        st.session_state.selected_movie_detail = {
                            "title": m_title,
                            "formatted_name": formatted_name,
                            "overview": m_overview,
                            "genres": m_genres,
                            "rating": m_rating,
                            "pop": m_pop,
                            "poster_url": poster_url,
                            "meta": meta
                        }
                with c_act2:
                    if st.button("❤️ Save", key=f"fav_{idx_pos}_{m_title}"):
                        if m_title not in st.session_state.favorites:
                            st.session_state.favorites.append(m_title)

# 9. Clean Movie Details Panel
if st.session_state.selected_movie_detail:
    det = st.session_state.selected_movie_detail
    st.markdown("---")
    
    with st.container():
        st.markdown(f'<div class="details-container">', unsafe_allow_html=True)
        col_img, col_info = st.columns([1, 2])

        with col_img:
            st.image(det["poster_url"], use_container_width=True)

        with col_info:
            st.markdown(f'<div class="details-title">{det["formatted_name"]}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span class="badge-tag badge-amber">★ IMDb {det['rating']}</span>
                <span class="badge-tag badge-indigo">Popularity: {det['pop']}</span>
                <span class="badge-tag badge-rose">Language: {det['meta'].get('original_language', 'EN')}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Genres:** `{det['genres']}`")
            st.markdown(f"**Synopsis:**\n{det['overview']}")

            yt_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(det['title'] + ' official trailer')}"
            imdb_url = f"https://www.imdb.com/find?q={urllib.parse.quote(det['title'])}"
            tmdb_url = f"https://www.themoviedb.org/search?query={urllib.parse.quote(det['title'])}"

            st.markdown(f"""
            <div style="margin-top: 1.25rem; display: flex; gap: 10px; flex-wrap: wrap;">
                <a href="{yt_url}" target="_blank" style="background:#f43f5e; color:white; padding:8px 16px; border-radius:8px; text-decoration:none; font-weight:600; font-size:0.9rem;">▶ Watch Trailer</a>
                <a href="{imdb_url}" target="_blank" style="background:#fbbf24; color:#0f172a; padding:8px 16px; border-radius:8px; text-decoration:none; font-weight:700; font-size:0.9rem;">IMDb</a>
                <a href="{tmdb_url}" target="_blank" style="background:#6366f1; color:white; padding:8px 16px; border-radius:8px; text-decoration:none; font-weight:600; font-size:0.9rem;">TMDB</a>
            </div>
            """, unsafe_allow_html=True)

        # Highlights & Insights Panel
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("Loading movie details..."):
            insights = get_gemini_insights(
                det["title"], overview=det["overview"], genres=det["genres"]
            )

        st.markdown(f"""
        <div class="insights-box">
            <div class="insights-header">📌 Movie Highlights & Overview</div>
            <p style="margin-bottom:0.5rem;"><strong>Summary:</strong> {insights.get('summary')}</p>
            <p style="margin-bottom:0.5rem;"><strong>Why You'll Like It:</strong> {insights.get('why_recommended')}</p>
            <p style="margin-bottom:0.5rem;"><strong>Mood:</strong> <span class="badge-tag badge-indigo">{insights.get('mood')}</span></p>
            <p style="margin-bottom:0.5rem;"><strong>Best For:</strong> {insights.get('audience')}</p>
            <p style="margin-bottom:0;"><strong>Trivia:</strong> {insights.get('fun_fact')}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Close Details", key="close_det"):
            st.session_state.selected_movie_detail = None
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# 10. Minimalist Footer
st.markdown("""
<div class="custom-footer">
    <p>🍿 <strong>CinemaVerse</strong> — Premium Movie Recommendation Platform</p>
</div>
""", unsafe_allow_html=True)
