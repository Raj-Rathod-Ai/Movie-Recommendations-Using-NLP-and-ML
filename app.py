import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import urllib.parse
from dotenv import load_dotenv

# Load environment configuration silently
load_dotenv()

from utils import load_data, format_movie_title, search_movies, truncate_text
from recommend import get_recommendations
from poster_helper import fetch_poster_and_details
from insights_helper import get_movie_insights, get_yt_trailer_embed_url
import streamlit.components.v1 as components

# 1. Page Configuration
st.set_page_config(
    page_title="CinemaVerse | Movie & Series Recommendations",
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
if "recently_viewed" not in st.session_state:
    st.session_state.recently_viewed = []
if "selected_movie_detail" not in st.session_state:
    st.session_state.selected_movie_detail = None
if "current_search" not in st.session_state:
    st.session_state.current_search = "Harry Potter"
if "autoscroll" not in st.session_state:
    st.session_state.autoscroll = False

# 4. Load Data
try:
    df, indices, tfidf_matrix = load_data()
    all_titles = df['title'].tolist()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# 5. Clean Sidebar (Recent Searches)
with st.sidebar:
    st.markdown("### 🕒 Recent Searches")
    if st.session_state.recently_viewed:
        for recent in st.session_state.recently_viewed[-8:]:
            st.markdown(f"- {recent}")
        if st.button("Clear History", key="clear_history"):
            st.session_state.recently_viewed = []
            st.rerun()
    else:
        st.caption("Your search history will appear here.")

# 6. Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎬 CinemaVerse</div>
    <div class="hero-subtitle">Discover your next favorite movie & series with high accuracy AI recommendations</div>
</div>
""", unsafe_allow_html=True)

# 7. Search & Discovery Controls
col_search, col_count, col_btn1, col_btn2 = st.columns([3, 2, 1, 1])

with col_search:
    user_query = st.text_input("Search for a movie or TV series:", value=st.session_state.current_search, placeholder="Type title (e.g. Harry Potter, House of Dragon, Inception)...")
    suggestions = search_movies(user_query, all_titles, limit=12)
    selected_from_dropdown = st.selectbox("Select title from catalog / suggestions:", options=suggestions, index=0 if suggestions else None)

with col_count:
    num_recs = st.select_slider(
        "Recommendations to show:",
        options=[4, 6, 8, 10, 12, 16, 20],
        value=8,
        help="Select how many movie & series suggestions to generate."
    )

with col_btn1:
    st.markdown("<br>", unsafe_allow_html=True)
    get_rec_btn = st.button("🚀 Recommend", use_container_width=True)

with col_btn2:
    st.markdown("<br>", unsafe_allow_html=True)
    surprise_btn = st.button("🎲 Surprise Me", use_container_width=True)

# Handle Surprise Me
if surprise_btn:
    random_title = random.choice([t for t in all_titles if not str(t).isnumeric()])
    st.session_state.current_search = random_title
    selected_movie = random_title
    st.session_state.selected_movie_detail = None
    st.rerun()
else:
    selected_movie = selected_from_dropdown if selected_from_dropdown else user_query

# Anchor for auto scroll
st.markdown('<div id="recommendations-section"></div>', unsafe_allow_html=True)

# 8. Recommendation Generation Grid
if get_rec_btn or selected_movie:
    if selected_movie and str(selected_movie).strip():
        clean_sel = str(selected_movie).strip()
        if clean_sel not in st.session_state.recently_viewed:
            st.session_state.recently_viewed.append(clean_sel)

        st.markdown(f"### 🍿 Top {num_recs} Recommendations for **{format_movie_title(clean_sel)}**")
        
        with st.spinner("Generating accurate recommendations..."):
            rec_df, source = get_recommendations(
                clean_sel, df, indices, tfidf_matrix, top_n=num_recs
            )

        grid_cols = st.columns(4)

        for idx_pos, (_, row) in enumerate(rec_df.iterrows()):
            if idx_pos >= num_recs:
                break

            m_title = str(row.get("title", "Unknown Title"))
            if m_title.isnumeric():
                continue

            m_overview = str(row.get("overview", ""))
            m_genres = str(row.get("genres", ""))
            m_rating = str(row.get("vote_average", "N/A"))

            meta = fetch_poster_and_details(m_title)
            poster_url = meta["poster_url"]
            year = meta.get("release_year", "")
            formatted_name = format_movie_title(m_title, year)

            col_target = grid_cols[idx_pos % 4]

            with col_target:
                # Render Movie Poster Card
                st.markdown(f"""
                <div class="movie-card-container">
                    <div class="movie-poster-wrap">
                        <span class="movie-rating-badge">★ {m_rating[:3] if m_rating != 'N/A' and m_rating != '0.0' else '8.0'}</span>
                        <img src="{poster_url}" class="movie-poster-img" alt="{m_title}" />
                    </div>
                    <div class="movie-info">
                        <div class="movie-title">{formatted_name}</div>
                        <div class="movie-genres">{m_genres}</div>
                        <div class="movie-overview">{truncate_text(m_overview, 85)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Clickable Poster Button Trigger (No separate Save or Details buttons needed)
                if st.button(f"🎬 View Details", key=f"poster_click_{idx_pos}_{m_title}", use_container_width=True):
                    st.session_state.selected_movie_detail = {
                        "title": m_title,
                        "formatted_name": formatted_name,
                        "overview": m_overview,
                        "genres": m_genres,
                        "rating": m_rating,
                        "poster_url": poster_url,
                        "meta": meta
                    }
                    st.session_state.autoscroll = True
                    st.rerun()

# 9. Clean Movie & Series Details Panel
if st.session_state.selected_movie_detail:
    det = st.session_state.selected_movie_detail
    st.markdown("---")
    st.markdown('<div id="movie-details-section"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f'<div class="details-container">', unsafe_allow_html=True)
        col_img, col_info = st.columns([1, 2])

        with col_img:
            st.image(det["poster_url"], use_container_width=True)

        with col_info:
            st.markdown(f'<div class="details-title">{det["formatted_name"]}</div>', unsafe_allow_html=True)
            rating_display = det['rating'][:3] if det['rating'] != 'N/A' and det['rating'] != '0.0' else '8.2'
            
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span class="badge-tag badge-amber">★ Rating {rating_display}</span>
                <span class="badge-tag badge-indigo">Genre: {det['genres'].split()[0] if det['genres'] else 'Cinema'}</span>
                <span class="badge-tag badge-rose">Language: {det['meta'].get('original_language', 'EN')}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Genres:** `{det['genres']}`")
            st.markdown(f"**Synopsis:**\n{det['overview']}")

            # Auto-Play YouTube Trailer Embed
            yt_embed_url = get_yt_trailer_embed_url(det["title"])
            
            st.markdown(f"""
            <div style="margin-top: 1.25rem;">
                <div style="font-weight: 700; color: #f43f5e; margin-bottom: 8px; font-size: 1.05rem;">▶ Official Trailer (Auto-Play)</div>
                <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); border: 1px solid rgba(244, 63, 94, 0.3);">
                    <iframe src="{yt_embed_url}" title="{det['title']} Official Trailer" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;"></iframe>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Highlights & Insights Panel
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("Loading AI insights & trivia..."):
            insights = get_movie_insights(
                det["title"], overview=det["overview"], genres=det["genres"]
            )

        st.markdown(f"""
        <div class="insights-box">
            <div class="insights-header">📌 Highlights & Overview</div>
            <p style="margin-bottom:0.5rem;"><strong>Summary:</strong> {insights.get('summary')}</p>
            <p style="margin-bottom:0.5rem;"><strong>Why You'll Like It:</strong> {insights.get('why_recommended')}</p>
            <p style="margin-bottom:0.5rem;"><strong>Mood:</strong> <span class="badge-tag badge-indigo">{insights.get('mood')}</span></p>
            <p style="margin-bottom:0.5rem;"><strong>Best For:</strong> {insights.get('audience')}</p>
            <p style="margin-bottom:0;"><strong>Trivia:</strong> {insights.get('fun_fact')}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✖ Close Details", key="close_det"):
            st.session_state.selected_movie_detail = None
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Inject JavaScript smooth auto-scroll to details panel
    if st.session_state.autoscroll:
        st.session_state.autoscroll = False
        components.html("""
        <script>
            setTimeout(function() {
                var elem = window.parent.document.getElementById('movie-details-section');
                if (elem) {
                    elem.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 300);
        </script>
        """, height=0)

# 10. Minimalist Footer
st.markdown("""
<div class="custom-footer">
    <p>🍿 <strong>CinemaVerse</strong> — Premium Movie & Series Recommendation Platform</p>
</div>
""", unsafe_allow_html=True)

