import os
import json
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

def get_client(api_key: str = None):
    """
    Get initialized Gemini Client.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception as e:
        logger.error(f"Error initializing Gemini client: {e}")
        return None

def get_gemini_insights(movie_title: str, overview: str = "", genres: str = "", api_key: str = None) -> dict:
    """
    Generate AI movie summary, why recommended, mood, best audience, and fun fact.
    """
    client = get_client(api_key)
    if not client:
        return {
            "summary": overview or "A fascinating cinematic experience featuring engaging storytelling and memorable performances.",
            "why_recommended": "Recommended based on similar narrative themes, genre alignment, and high audience engagement.",
            "mood": "Captivating & Immersive",
            "audience": "Movie Enthusiasts & Fans of " + (genres or "Great Cinema"),
            "fun_fact": "This movie achieved widespread critical acclaim and maintains a dedicated fanbase worldwide."
        }

    prompt = f"""
    You are an expert film critic and cinema enthusiast. Analyze the following movie:
    Title: {movie_title}
    Genres: {genres}
    Overview: {overview}

    Return a JSON object with EXACTLY these keys:
    - "summary": A compelling 2-sentence AI synopsis highlighting key narrative themes.
    - "why_recommended": Why a fan of this genre will enjoy this film (1-2 sentences).
    - "mood": 2-3 words describing the emotional vibe (e.g. "Thrilling & Suspenseful", "Heartwarming & Inspiring").
    - "audience": Best target audience (e.g. "Family Movie Night", "Late-Night Thriller Fans", "Couple's Choice").
    - "fun_fact": A fun or surprising trivia fact about the movie's production, cast, or box office.

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
        logger.warning(f"Gemini API call failed for insights on {movie_title}: {e}")
        return {
            "summary": overview or "A captivating film with compelling characters and rich storytelling.",
            "why_recommended": "Matches your preference for great storytelling and engaging thematic elements.",
            "mood": "Immersive & Engaging",
            "audience": "Film Lovers & Fans of " + (genres or "Quality Cinema"),
            "fun_fact": "This film stands out in its genre for its distinct directorial direction."
        }

def get_gemini_fallback_recommendations(query: str, top_n: int = 10, api_key: str = None) -> list:
    """
    Generate movie recommendations when local matrix finds no match or movie is not in dataset.
    """
    client = get_client(api_key)
    if not client:
        # Fallback hardcoded defaults if key is missing
        return [
            {
                "title": "Inception",
                "release_year": "2010",
                "genres": "Action Sci-Fi Thriller",
                "vote_average": 8.8,
                "popularity": 45.2,
                "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."
            },
            {
                "title": "Interstellar",
                "release_year": "2014",
                "genres": "Adventure Drama Sci-Fi",
                "vote_average": 8.6,
                "popularity": 42.1,
                "overview": "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot, Joseph Cooper, is tasked to pilot a spacecraft, along with a team of researchers, to find a new planet for humans."
            },
            {
                "title": "The Dark Knight",
                "release_year": "2008",
                "genres": "Action Crime Drama",
                "vote_average": 9.0,
                "popularity": 50.0,
                "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
            }
        ][:top_n]

    prompt = f"""
    The user is looking for movie recommendations related to or similar to: "{query}".
    Provide exactly {top_n} recommended movies.

    Return a JSON array of objects, where each object has these exact keys:
    - "title": Official Movie Title
    - "release_year": Year of release (e.g. "2010")
    - "genres": Space-separated or comma-separated main genres (e.g. "Sci-Fi Action")
    - "vote_average": Estimated IMDb score out of 10 (e.g. 8.5)
    - "popularity": Popularity score (e.g. 35.0)
    - "overview": Concise 2-sentence synopsis.

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
            return movies[:top_n]
        return []
    except Exception as e:
        logger.warning(f"Gemini fallback recommendation failed: {e}")
        return []
