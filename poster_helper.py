import urllib.parse
import requests
import streamlit as st
import base64
import hashlib
import re

def clean_movie_search_title(title: str) -> str:
    """
    Remove language tags like (IN Hindi), (IN Telugu), (2016), (EN) from movie title for clean poster search.
    e.g. 'A Aa (IN Hindi)' -> 'A Aa'
    """
    if not title or not isinstance(title, str):
        return ""
    # Strip (IN ...), (Hindi), (Telugu), (Tamil), (2020), etc.
    cleaned = re.sub(r'\s*\((?:IN\s*)?[A-Za-z0-9\s]+\)\s*$', '', title, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(\d{4}\)\s*$', '', cleaned)
    return cleaned.strip() or title.strip()

def fetch_real_poster_url(title: str) -> str:
    """
    Fetch real high-resolution poster image for a movie or TV series.
    Tries TVmaze API first (best for TV shows), then OMDb API (best for movies), then iTunes API with clean title.
    """
    clean_title = clean_movie_search_title(title)
    if not clean_title:
        return generate_svg_poster("Cinema")

    encoded_title = urllib.parse.quote(clean_title)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    # 1. Try OMDb Free API Endpoint (Highest accuracy for movie titles: A Aa, Avatar, Spider-Man, Harry Potter)
    try:
        url = f"https://www.omdbapi.com/?t={encoded_title}&apikey=trilogy"
        resp = requests.get(url, headers=headers, timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()
            poster = data.get('Poster')
            if poster and poster.startswith('http') and poster != 'N/A':
                return poster
    except Exception:
        pass

    # 2. Try TVmaze API (Best for TV Series like Berlin, Money Heist, Lupin, House of the Dragon)
    try:
        url = f"https://api.tvmaze.com/singlesearch/shows?q={encoded_title}"
        resp = requests.get(url, headers=headers, timeout=2.5)
        if resp.status_code == 200:
            img = resp.json().get('image')
            if img and isinstance(img, dict):
                if img.get('original'):
                    return img['original']
                elif img.get('medium'):
                    return img['medium']
    except Exception:
        pass

    # 3. Try iTunes Search API (With clean title)
    try:
        url = f"https://itunes.apple.com/search?term={encoded_title}&limit=10"
        resp = requests.get(url, headers=headers, timeout=2.5)
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            q_words = [w for w in clean_title.lower().split() if len(w) > 1 and w not in ['the', 'of', 'and', 'a', 'an', 'in', 'on', 'for', 'to']]
            for item in results:
                track_name = str(item.get('trackName') or item.get('collectionName') or '').lower()
                if q_words and all(w in track_name for w in q_words):
                    art = item.get('artworkUrl100') or item.get('artworkUrl60')
                    if art and isinstance(art, str):
                        high_res = art.replace('100x100bb.jpg', '600x900bb.jpg').replace('100x100bb.png', '600x900bb.jpg').replace('600x600bb', '600x900bb')
                        return high_res
    except Exception:
        pass

    # 4. Fallback SVG Poster
    return generate_svg_poster(clean_title)


def generate_svg_poster(title: str, year: str = "") -> str:
    """
    Generate a full-bleed SVG poster artwork fallback without inset border artifacts.
    """
    clean_t = clean_movie_search_title(title)
    hash_val = int(hashlib.md5(clean_t.encode()).hexdigest(), 16)
    palettes = [
        ("#6366f1", "#0b0f19", "#a5b4fc", "#4f46e5"),
        ("#f43f5e", "#0f172a", "#fda4af", "#e11d48"),
        ("#8b5cf6", "#1e1b4b", "#c084fc", "#7c3aed"),
        ("#10b981", "#064e3b", "#34d399", "#059669"),
        ("#f59e0b", "#1c1917", "#fbbf24", "#d97706")
    ]
    accent, bg_dark, sub_accent, mid_color = palettes[hash_val % len(palettes)]
    display_title = clean_t if len(clean_t) <= 18 else clean_t[:16] + "..."

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 450" width="300" height="450" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="bg_{hash_val % 5}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{accent}" stop-opacity="0.6"/>
      <stop offset="50%" stop-color="{mid_color}" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="{bg_dark}" stop-opacity="0.98"/>
    </linearGradient>
  </defs>
  <rect width="300" height="450" fill="{bg_dark}"/>
  <rect width="300" height="450" fill="url(#bg_{hash_val % 5})"/>
  
  <g transform="translate(110, 130)" fill="{sub_accent}">
    <circle cx="40" cy="40" r="34" fill="none" stroke="{sub_accent}" stroke-width="3"/>
    <circle cx="40" cy="40" r="10" fill="{sub_accent}"/>
    <circle cx="40" cy="18" r="4" fill="{bg_dark}"/>
    <circle cx="40" cy="62" r="4" fill="{bg_dark}"/>
    <circle cx="18" cy="40" r="4" fill="{bg_dark}"/>
    <circle cx="62" cy="40" r="4" fill="{bg_dark}"/>
  </g>
  
  <text x="150" y="275" text-anchor="middle" fill="#ffffff" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-weight="800" font-size="22">
    {display_title}
  </text>
  <text x="150" y="305" text-anchor="middle" fill="{sub_accent}" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-size="14" font-weight="600">
    {year if year else "CINEMA"}
  </text>
</svg>"""

    b64_encoded = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64_encoded}"

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster_and_details(title: str) -> dict:
    """
    Cached poster & metadata fetcher delivering real movie thumbnails.
    """
    clean_title = clean_movie_search_title(title)
    poster_url = fetch_real_poster_url(clean_title)
    
    return {
        "poster_url": poster_url,
        "release_date": "N/A",
        "release_year": "",
        "vote_average": "N/A",
        "popularity": "N/A",
        "overview": "",
        "original_language": "EN"
    }
