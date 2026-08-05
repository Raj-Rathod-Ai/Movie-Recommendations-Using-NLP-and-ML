import os
import json
import re
import urllib.parse
import requests
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

def get_client(access_key: str = None):
    """
    Initialize content generation client.
    """
    key = access_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception as e:
        logger.error(f"Error initializing client: {e}")
        return None

def get_yt_trailer_embed_url(title: str) -> str:
    """
    Fetch YouTube official trailer video embed URL for auto-playing.
    """
    query = f"{title} official trailer"
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        r = requests.get(search_url, headers=headers, timeout=4)
        if r.status_code == 200:
            video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', r.text)
            if video_ids:
                return f"https://www.youtube.com/embed/{video_ids[0]}?autoplay=1&rel=0&enablejsapi=1"
    except Exception as e:
        logger.warning(f"Error fetching YouTube video ID: {e}")
    return f"https://www.youtube.com/embed?listType=search&list={encoded_query}&autoplay=1"

def get_movie_insights(movie_title: str, overview: str = "", genres: str = "", access_key: str = None) -> dict:
    """
    Generate detailed movie summary, rationale, mood, audience, and trivia.
    """
    client = get_client(access_key)
    if not client:
        return {
            "summary": overview or "A fascinating cinematic experience featuring engaging storytelling and memorable performances.",
            "why_recommended": "Recommended based on similar narrative themes, genre alignment, and high audience engagement.",
            "mood": "Captivating & Immersive",
            "audience": "Movie Enthusiasts & Fans of " + (genres or "Great Cinema"),
            "fun_fact": "This title achieved widespread critical acclaim and maintains a dedicated fanbase worldwide."
        }

    prompt = f"""
    You are an expert film and TV critic. Analyze the following title:
    Title: {movie_title}
    Genres: {genres}
    Overview: {overview}

    Return a JSON object with EXACTLY these keys:
    - "summary": A compelling 2-sentence synopsis highlighting key narrative themes.
    - "why_recommended": Why a fan of this genre will enjoy this film/show (1-2 sentences).
    - "mood": 2-3 words describing the emotional vibe (e.g. "Thrilling & Suspenseful", "Heartwarming & Inspiring").
    - "audience": Best target audience (e.g. "Family Movie Night", "Late-Night Thriller Fans", "Couple's Choice").
    - "fun_fact": A fun or surprising trivia fact about production, cast, or box office.

    Return ONLY valid raw JSON with no markdown backticks.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        data = json.loads(response.text)
        return data
    except Exception as e:
        logger.warning(f"Error generating insights for {movie_title}: {e}")
        return {
            "summary": overview or "A captivating film/show with compelling characters and rich storytelling.",
            "why_recommended": "Matches your preference for great storytelling and engaging thematic elements.",
            "mood": "Immersive & Engaging",
            "audience": "Film Lovers & Fans of " + (genres or "Quality Cinema"),
            "fun_fact": "This title stands out in its genre for its distinct direction."
        }

def get_smart_fallback_recommendations(query: str, top_n: int = 10, access_key: str = None) -> list:
    """
    Generate high-accuracy movie/TV show recommendations when local matrix finds no match or for TV Series queries.
    Handles Gemini 429 rate limits gracefully with query keyword matched fallbacks.
    """
    q_lower = query.lower()

    # Pre-defined curated fallbacks by genre/keyword
    fantasy_series_fallback = [
        {"title": "House of the Dragon", "release_year": "2022", "genres": "Action Adventure Drama Fantasy", "vote_average": 8.4, "popularity": 95.0, "overview": "The story of the House Targaryen set 200 years before the events of Game of Thrones."},
        {"title": "Game of Thrones", "release_year": "2011", "genres": "Action Adventure Drama Fantasy", "vote_average": 9.2, "popularity": 98.0, "overview": "Nine noble families fight for control over the lands of Westeros, while an ancient enemy returns after being dormant for millennia."},
        {"title": "The Lord of the Rings: The Rings of Power", "release_year": "2022", "genres": "Action Adventure Drama Fantasy", "vote_average": 7.8, "popularity": 88.0, "overview": "Epic drama set thousands of years before the events of J.R.R. Tolkien's The Hobbit and The Lord of the Rings."},
        {"title": "The Witcher", "release_year": "2019", "genres": "Action Adventure Drama Fantasy", "vote_average": 8.1, "popularity": 90.0, "overview": "Geralt of Rivia, a mutated monster-hunter for hire, journeys toward his destiny in a turbulent world."},
        {"title": "Vikings", "release_year": "2013", "genres": "Action Adventure Drama History", "vote_average": 8.5, "popularity": 87.0, "overview": "Vikings transports us to the brutal and mysterious world of Ragnar Lothbrok, a Viking warrior and farmer."}
    ]

    wizard_fantasy_fallback = [
        {"title": "Harry Potter and the Sorcerer's Stone", "release_year": "2001", "genres": "Adventure Family Fantasy", "vote_average": 7.9, "popularity": 92.0, "overview": "An orphaned boy enrolls in a school of wizardry, where he learns the truth about himself, his family and the terrible evil that haunts the magical world."},
        {"title": "Fantastic Beasts and Where to Find Them", "release_year": "2016", "genres": "Adventure Family Fantasy", "vote_average": 7.3, "popularity": 85.0, "overview": "The adventures of writer Newt Scamander in New York's secret community of witches and wizards seventy years before Harry Potter reads his book in school."},
        {"title": "The Lord of the Rings: The Fellowship of the Ring", "release_year": "2001", "genres": "Action Adventure Drama Fantasy", "vote_average": 8.8, "popularity": 96.0, "overview": "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron."},
        {"title": "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe", "release_year": "2005", "genres": "Adventure Family Fantasy", "vote_average": 6.9, "popularity": 82.0, "overview": "Four kids travel through a wardrobe to the land of Narnia and learn of their destiny to free it with the guidance of a mystical lion."},
        {"title": "Percy Jackson & the Olympians: The Lightning Thief", "release_year": "2010", "genres": "Adventure Family Fantasy", "vote_average": 5.9, "popularity": 78.0, "overview": "A teenager discovers he's the descendant of a Greek god and is set on an adventure to settle an on-going battle between the gods."}
    ]

    general_top_fallback = [
        {"title": "Inception", "release_year": "2010", "genres": "Action Sci-Fi Thriller", "vote_average": 8.8, "popularity": 95.0, "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
        {"title": "Interstellar", "release_year": "2014", "genres": "Adventure Drama Sci-Fi", "vote_average": 8.6, "popularity": 94.0, "overview": "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot is tasked to pilot a spacecraft to find a new planet for humans."},
        {"title": "The Dark Knight", "release_year": "2008", "genres": "Action Crime Drama", "vote_average": 9.0, "popularity": 97.0, "overview": "When the menace known as the Joker wreaks havoc and chaos on Gotham, Batman must accept one of the greatest psychological and physical tests of his ability."},
        {"title": "Avatar", "release_year": "2009", "genres": "Action Adventure Sci-Fi", "vote_average": 7.6, "popularity": 92.0, "overview": "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home."},
        {"title": "Gladiator", "release_year": "2000", "genres": "Action Adventure Drama", "vote_average": 8.5, "popularity": 91.0, "overview": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."}
    ]

    client = get_client(access_key)
    if not client:
        if any(k in q_lower for k in ["dragon", "house", "thrones", "got", "targaryen", "witcher"]):
            return fantasy_series_fallback[:top_n]
        elif any(k in q_lower for k in ["harry", "potter", "wizard", "magic", "beasts", "narnia"]):
            return wizard_fantasy_fallback[:top_n]
        return general_top_fallback[:top_n]

    prompt = f"""
    The user is searching for recommendations related to the movie or TV show: "{query}".
    Provide exactly {top_n} highly relevant, top-rated movie or TV series recommendations.

    Return a JSON array of objects, where each object has these exact keys:
    - "title": Official Movie or TV Series Title
    - "release_year": Year of release (e.g. "2022")
    - "genres": Main genres (e.g. "Fantasy Drama")
    - "vote_average": Rating out of 10 (e.g. 8.5)
    - "popularity": Popularity score (e.g. 85.0)
    - "overview": Concise 2-sentence synopsis.

    Do NOT include generic numbers or numeric titles like "7010", "0", "15".
    Return ONLY valid raw JSON array with no markdown backticks.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        movies = json.loads(response.text)
        if isinstance(movies, list):
            clean_movies = [m for m in movies if isinstance(m, dict) and not str(m.get('title', '')).isnumeric()]
            if clean_movies:
                return clean_movies[:top_n]
    except Exception as e:
        logger.warning(f"Fallback recommendation API error (e.g. rate limit): {e}")

    # Fallback to curated keyword matching list if API fails or hits 429 rate limit
    if any(k in q_lower for k in ["dragon", "house", "thrones", "got", "targaryen", "witcher"]):
        return fantasy_series_fallback[:top_n]
    elif any(k in q_lower for k in ["harry", "potter", "wizard", "magic", "beasts", "narnia"]):
        return wizard_fantasy_fallback[:top_n]
    return general_top_fallback[:top_n]


