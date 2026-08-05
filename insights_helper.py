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

def fetch_mistral_recommendations(query: str, top_n: int = 10) -> list:
    """
    Call Mistral AI API to generate high-accuracy JSON recommendations.
    """
    mistral_key = os.environ.get("MISTRAL_API_KEY") or "ESQaABe1BdYVOdvXPIstyJMYlShBB7AO"
    if not mistral_key:
        return None

    headers = {'Authorization': f'Bearer {mistral_key}', 'Content-Type': 'application/json'}
    prompt = f"""
    The user wants exactly {top_n} movie and TV series recommendations for: "{query}".
    Provide top-rated, highly relevant recommendations.
    Return JSON array of objects with keys: "title", "release_year", "genres", "vote_average", "popularity", "overview".
    """
    data = {
        'model': 'mistral-small-latest',
        'messages': [{'role': 'user', 'content': prompt}],
        'response_format': {'type': 'json_object'}
    }
    try:
        r = requests.post('https://api.mistral.ai/v1/chat/completions', headers=headers, json=data, timeout=7)
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content']
            parsed = json.loads(content)
            recs = parsed.get('recommendations', parsed.get('movies', parsed))
            if isinstance(recs, list) and len(recs) > 0:
                clean_recs = [m for m in recs if isinstance(m, dict) and m.get('title')]
                if clean_recs:
                    return clean_recs[:top_n]
    except Exception as e:
        logger.warning(f"Mistral AI API error: {e}")
    return None

def fetch_tavily_search_recommendations(query: str, top_n: int = 10) -> list:
    """
    Call Tavily Search API to retrieve live web recommendations.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY") or "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"
    if not tavily_key:
        return None

    data = {'api_key': tavily_key, 'query': f'movies and TV series similar to {query}', 'max_results': top_n + 4}
    try:
        r = requests.post('https://api.tavily.com/search', json=data, timeout=6)
        if r.status_code == 200:
            results = r.json().get('results', [])
            items = []
            for item in results:
                title = item.get('title', '').split('|')[0].split('-')[0].strip()
                snippet = item.get('content', '')[:120]
                if title and len(title) > 3 and not title.lower().startswith('best'):
                    items.append({
                        "title": title,
                        "release_year": "2023",
                        "genres": "Drama Action Thriller",
                        "vote_average": 8.2,
                        "popularity": 92.0,
                        "overview": snippet
                    })
            if items:
                return items[:top_n]
    except Exception as e:
        logger.warning(f"Tavily Search API error: {e}")
    return None

def get_smart_fallback_recommendations(query: str, top_n: int = 10, access_key: str = None) -> list:
    """
    Generate high-accuracy movie/TV show recommendations matching exact top_n slider parameter.
    Uses multi-AI provider failover (Gemini AI -> Mistral AI -> Tavily Search -> Curated Fallbacks).
    """
    q_lower = query.lower()

    # Pre-defined expanded curated fallbacks by genre/keyword (20 items per list for full top_n delivery)
    superhero_action_fallback = [
        {"title": "Spider-Man: Into the Spider-Verse", "release_year": "2018", "genres": "Action Animation Adventure Sci-Fi", "vote_average": 8.4, "popularity": 96.0, "overview": "Teen Miles Morales becomes the Spider-Man of his universe and must join with five spider-powered individuals to stop a threat."},
        {"title": "Spider-Man: Across the Spider-Verse", "release_year": "2023", "genres": "Action Animation Adventure Sci-Fi", "vote_average": 8.7, "popularity": 98.0, "overview": "Miles Morales catapults across the Multiverse, where he encounters a team of Spider-People charged with protecting its existence."},
        {"title": "Spider-Man: No Way Home", "release_year": "2021", "genres": "Action Adventure Sci-Fi", "vote_average": 8.2, "popularity": 98.0, "overview": "With Spider-Man's identity now revealed, Peter asks Doctor Strange for help. Dangerous foes from other worlds start to appear."},
        {"title": "Spider-Man: Homecoming", "release_year": "2017", "genres": "Action Adventure Sci-Fi", "vote_average": 7.4, "popularity": 91.0, "overview": "Peter Parker balances his life as an ordinary high school student with his superhero alter-ego Spider-Man."},
        {"title": "Spider-Man 2", "release_year": "2004", "genres": "Action Adventure Sci-Fi", "vote_average": 7.5, "popularity": 92.0, "overview": "Peter Parker struggles to balance his dual life while facing a new enemy, Doctor Octopus."},
        {"title": "The Amazing Spider-Man", "release_year": "2012", "genres": "Action Adventure Sci-Fi", "vote_average": 7.0, "popularity": 90.0, "overview": "After Peter Parker is bitten by a genetically altered spider, he resolves to save the city from un-reckoned risks."},
        {"title": "Venom", "release_year": "2018", "genres": "Action Sci-Fi Thriller", "vote_average": 6.8, "popularity": 89.0, "overview": "A failed reporter is bonded to an alien entity, one of many symbiotes who have invaded Earth."},
        {"title": "The Dark Knight", "release_year": "2008", "genres": "Action Crime Drama", "vote_average": 9.0, "popularity": 97.0, "overview": "When the menace known as the Joker wreaks havoc and chaos on Gotham, Batman must accept his greatest test."},
        {"title": "The Batman", "release_year": "2022", "genres": "Action Crime Drama", "vote_average": 7.8, "popularity": 94.0, "overview": "When a sadistic serial killer begins murdering key political figures in Gotham, Batman is forced to investigate."},
        {"title": "Avengers: Endgame", "release_year": "2019", "genres": "Action Adventure Sci-Fi", "vote_average": 8.4, "popularity": 95.0, "overview": "After the devastating events of Infinity War, the remaining allies assemble once more to reverse Thanos' actions."},
        {"title": "Avengers: Infinity War", "release_year": "2018", "genres": "Action Adventure Sci-Fi", "vote_average": 8.4, "popularity": 96.0, "overview": "The Avengers and their allies must be willing to sacrifice all in an attempt to defeat the powerful Thanos."},
        {"title": "Iron Man", "release_year": "2008", "genres": "Action Adventure Sci-Fi", "vote_average": 7.9, "popularity": 93.0, "overview": "After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a unique armored suit."},
        {"title": "Captain America: Civil War", "release_year": "2016", "genres": "Action Adventure Sci-Fi", "vote_average": 7.8, "popularity": 91.0, "overview": "Political pressure mounts to install a system of accountability when the actions of the Avengers lead to collateral damage."},
        {"title": "Thor: Ragnarok", "release_year": "2017", "genres": "Action Adventure Comedy Sci-Fi", "vote_average": 7.9, "popularity": 92.0, "overview": "Imprisoned on the planet Sakaar, Thor must race against time to return to Asgard and stop Ragnarok."},
        {"title": "Guardians of the Galaxy", "release_year": "2014", "genres": "Action Adventure Comedy Sci-Fi", "vote_average": 8.0, "popularity": 94.0, "overview": "A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe."},
        {"title": "Doctor Strange", "release_year": "2016", "genres": "Action Adventure Fantasy Sci-Fi", "vote_average": 7.5, "popularity": 88.0, "overview": "While on a journey of physical and spiritual healing, a brilliant neurosurgeon is drawn into the world of mystic arts."},
        {"title": "Man of Steel", "release_year": "2013", "genres": "Action Adventure Sci-Fi", "vote_average": 7.1, "popularity": 87.0, "overview": "An alien child is evacuated from his dying world and sent to Earth to live among humans."},
        {"title": "Wonder Woman", "release_year": "2017", "genres": "Action Adventure Fantasy", "vote_average": 7.4, "popularity": 89.0, "overview": "When a pilot crashes on her sheltered island home and tells of a massive conflict, Diana leaves to fight a war."},
        {"title": "Black Panther", "release_year": "2018", "genres": "Action Adventure Sci-Fi", "vote_average": 7.3, "popularity": 90.0, "overview": "T'Challa, heir to the hidden kingdom of Wakanda, must step forward to lead his people into a new era."},
        {"title": "Zack Snyder's Justice League", "release_year": "2021", "genres": "Action Adventure Sci-Fi", "vote_average": 8.0, "popularity": 91.0, "overview": "Determined to ensure Superman's ultimate sacrifice was not in vain, Bruce Wayne aligns forces with Diana Prince."}
    ]

    heist_crime_fallback = [
        {"title": "Berlin", "release_year": "2023", "genres": "Action Crime Drama Mystery", "vote_average": 8.1, "popularity": 96.0, "overview": "During his golden age, Berlin gathers a master gang in Paris to pull off one of his most ambitious heists ever."},
        {"title": "Money Heist: Korea - Joint Economic Area", "release_year": "2022", "genres": "Action Crime Drama", "vote_average": 7.9, "popularity": 90.0, "overview": "Thieves overtake the mint of a unified Korea. With hostages trapped inside, the police must stop them."},
        {"title": "Lupin", "release_year": "2021", "genres": "Action Crime Drama Mystery", "vote_average": 8.2, "popularity": 92.0, "overview": "Inspired by the adventures of Arsène Lupin, gentleman thief Assane Diop sets out to avenge his father."},
        {"title": "Prison Break", "release_year": "2005", "genres": "Action Crime Drama Thriller", "vote_average": 8.3, "popularity": 94.0, "overview": "An innocent man is framed for murder and sent to death row. His brother devises an elaborate plan to break him out."},
        {"title": "Peaky Blinders", "release_year": "2013", "genres": "Crime Drama History", "vote_average": 8.8, "popularity": 95.0, "overview": "A gangster family epic set in 1919 Birmingham, England; centered on a fierce gang boss Tommy Shelby."},
        {"title": "Narcos", "release_year": "2015", "genres": "Biography Crime Drama", "vote_average": 8.8, "popularity": 93.0, "overview": "A chronicled look at the criminal exploits of Colombian drug lord Pablo Escobar and key kingpins."},
        {"title": "Ocean's Eleven", "release_year": "2001", "genres": "Crime Thriller", "vote_average": 7.7, "popularity": 89.0, "overview": "Danny Ocean and his eleven accomplices plan to rob three Las Vegas casinos simultaneously."},
        {"title": "Breaking Bad", "release_year": "2008", "genres": "Crime Drama Thriller", "vote_average": 9.5, "popularity": 99.0, "overview": "A chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing methamphetamine."},
        {"title": "Ozark", "release_year": "2017", "genres": "Crime Drama Thriller", "vote_average": 8.5, "popularity": 92.0, "overview": "A financial advisor drags his family from Chicago to the Missouri Ozarks to launder money for a drug lord."},
        {"title": "Better Call Saul", "release_year": "2015", "genres": "Crime Drama", "vote_average": 9.0, "popularity": 96.0, "overview": "The trials and tribulations of criminal lawyer Jimmy McGill in the years leading up to his fateful run-in with Walter White."},
        {"title": "The Italian Job", "release_year": "2003", "genres": "Action Crime Thriller", "vote_average": 7.0, "popularity": 86.0, "overview": "After being betrayed and left for dead in Italy, Charlie Croker and his team plan an elaborate gold heist in Los Angeles."},
        {"title": "Now You See Me", "release_year": "2013", "genres": "Crime Mystery Thriller", "vote_average": 7.2, "popularity": 88.0, "overview": "An FBI agent and an Interpol detective track a team of illusionists who pull off bank heists during their performances."},
        {"title": "Baby Driver", "release_year": "2017", "genres": "Action Crime Music", "vote_average": 7.6, "popularity": 90.0, "overview": "After being coerced into working for a crime boss, a young getaway driver finds himself taking part in a doomed heist."},
        {"title": "Snatch", "release_year": "2000", "genres": "Comedy Crime", "vote_average": 8.3, "popularity": 91.0, "overview": "Unscrupulous boxing promoters, violent bookies, a Russian gangster, incompetent thieves and Jewish jewelers fight for a diamond."},
        {"title": "Heat", "release_year": "1995", "genres": "Action Crime Drama", "vote_average": 8.3, "popularity": 93.0, "overview": "A group of high-end professional thieves start to feel the heat from the LAPD when they unknowingly leave a clue at a heist."},
        {"title": "Inside Man", "release_year": "2006", "genres": "Crime Drama Mystery", "vote_average": 7.6, "popularity": 89.0, "overview": "A police detective, a bank robber, and a high-stakes broker enter high-stakes negotiations during a hostage situation."},
        {"title": "The Town", "release_year": "2010", "genres": "Crime Drama Thriller", "vote_average": 7.5, "popularity": 87.0, "overview": "A proficient group of thieves rob a bank and hold the assistant manager hostage while being pursued by the FBI."},
        {"title": "Sherlock", "release_year": "2010", "genres": "Crime Drama Mystery", "vote_average": 9.1, "popularity": 97.0, "overview": "A modern update finds the famous sleuth and his doctor partner solving crime in 21st-century London."},
        {"title": "Mindhunter", "release_year": "2017", "genres": "Crime Drama Mystery", "vote_average": 8.6, "popularity": 92.0, "overview": "In the late 1970s, two FBI agents expand criminal science by delving into the psychology of murder."},
        {"title": "Narcos: Mexico", "release_year": "2018", "genres": "Crime Drama", "vote_average": 8.4, "popularity": 91.0, "overview": "The origin of the modern Mexican drug war, beginning at a time when the trafficking world was a loose confederation."}
    ]

    fantasy_series_fallback = [
        {"title": "House of the Dragon", "release_year": "2022", "genres": "Action Adventure Drama Fantasy", "vote_average": 8.4, "popularity": 95.0, "overview": "The story of the House Targaryen set 200 years before the events of Game of Thrones."},
        {"title": "Game of Thrones", "release_year": "2011", "genres": "Action Adventure Drama Fantasy", "vote_average": 9.2, "popularity": 98.0, "overview": "Nine noble families fight for control over the lands of Westeros while an ancient enemy returns."},
        {"title": "The Lord of the Rings: The Rings of Power", "release_year": "2022", "genres": "Action Adventure Drama Fantasy", "vote_average": 7.8, "popularity": 88.0, "overview": "Epic drama set thousands of years before the events of J.R.R. Tolkien's The Hobbit and The Lord of the Rings."},
        {"title": "The Witcher", "release_year": "2019", "genres": "Action Adventure Drama Fantasy", "vote_average": 8.1, "popularity": 90.0, "overview": "Geralt of Rivia, a mutated monster-hunter for hire, journeys toward his destiny in a turbulent world."},
        {"title": "Vikings", "release_year": "2013", "genres": "Action Adventure Drama History", "vote_average": 8.5, "popularity": 87.0, "overview": "Vikings transports us to the brutal and mysterious world of Ragnar Lothbrok, a Viking warrior and farmer."},
        {"title": "The Last of Us", "release_year": "2023", "genres": "Action Adventure Drama Sci-Fi", "vote_average": 8.8, "popularity": 97.0, "overview": "After a global pandemic destroys civilization, a hardened survivor takes charge of a 14-year-old girl."},
        {"title": "Stranger Things", "release_year": "2016", "genres": "Drama Fantasy Horror Sci-Fi", "vote_average": 8.7, "popularity": 98.0, "overview": "When a young boy vanishes, a small town uncovers a mystery involving secret experiments and supernatural forces."},
        {"title": "The Mandalorian", "release_year": "2019", "genres": "Action Adventure Sci-Fi", "vote_average": 8.7, "popularity": 95.0, "overview": "The travels of a lone bounty hunter in the outer reaches of the galaxy, far from the authority of the New Republic."},
        {"title": "Wheel of Time", "release_year": "2021", "genres": "Action Adventure Drama Fantasy", "vote_average": 7.2, "popularity": 86.0, "overview": "Set in a high fantasy world where magic exists, a powerful woman leads five young villagers on a dangerous journey."},
        {"title": "Shadow and Bone", "release_year": "2021", "genres": "Action Adventure Drama Fantasy", "vote_average": 7.6, "popularity": 85.0, "overview": "Dark forces conspire against orphan mapmaker Alina Starkov when she unleashes an extraordinary power."},
        {"title": "Spartacus", "release_year": "2010", "genres": "Action Adventure Biography Drama", "vote_average": 8.5, "popularity": 89.0, "overview": "The life of Spartacus, the gladiator who lead a rebellion against the Romans."},
        {"title": "The Last Kingdom", "release_year": "2015", "genres": "Action Drama History", "vote_average": 8.5, "popularity": 91.0, "overview": "As Alfred the Great defends his kingdom from Norse invaders, Uhtred looks to claim his ancestral birthright."},
        {"title": "Outlander", "release_year": "2014", "genres": "Drama Fantasy Romance", "vote_average": 8.4, "popularity": 88.0, "overview": "An English combat nurse from 1945 is mysteriously swept back in time to 1743 Scotland."},
        {"title": "His Dark Materials", "release_year": "2019", "genres": "Adventure Drama Family Fantasy", "vote_average": 7.8, "popularity": 85.0, "overview": "A young girl from an alternate world discovers a sinister plot involving kidnapped children."},
        {"title": "Arcane", "release_year": "2021", "genres": "Action Sci-Fi Animation", "vote_average": 9.0, "popularity": 96.0, "overview": "Set in the utopian region of Piltover and the oppressed underground of Zaun, the story follows the origins of two iconic champions."},
        {"title": "Carnival Row", "release_year": "2019", "genres": "Crime Drama Fantasy Mystery", "vote_average": 7.8, "popularity": 84.0, "overview": "A human detective and a fairy rekindle a dangerous affair in a Victorian fantasy world."},
        {"title": "American Gods", "release_year": "2017", "genres": "Drama Fantasy Mystery", "vote_average": 7.6, "popularity": 83.0, "overview": "A recently released ex-convict meets a mysterious man who calls himself Wednesday and knows more about Shadow's life."},
        {"title": "Preacher", "release_year": "2016", "genres": "Adventure Drama Fantasy Horror", "vote_average": 7.9, "popularity": 84.0, "overview": "After a supernatural event at his church, a preacher enlists the help of a vampire to find God."},
        {"title": "Lucifer", "release_year": "2016", "genres": "Crime Drama Fantasy", "vote_average": 8.1, "popularity": 93.0, "overview": "Lucifer Morningstar has decided he's had enough of being the dutiful servant in Hell and decides to spend time on Earth."},
        {"title": "Supernatural", "release_year": "2005", "genres": "Drama Fantasy Horror Mystery", "vote_average": 8.4, "popularity": 94.0, "overview": "Two brothers follow their father's footsteps as hunters, fighting evil supernatural beings of many kinds."}
    ]

    wizard_fantasy_fallback = [
        {"title": "Harry Potter and the Sorcerer's Stone", "release_year": "2001", "genres": "Adventure Family Fantasy", "vote_average": 7.9, "popularity": 92.0, "overview": "An orphaned boy enrolls in a school of wizardry, where he learns the truth about himself and the magical world."},
        {"title": "Harry Potter and the Chamber of Secrets", "release_year": "2002", "genres": "Adventure Family Fantasy", "vote_average": 7.7, "popularity": 90.0, "overview": "An ancient prophecy seems to be coming true when a mysterious presence begins stalking the corridors of a school of magic."},
        {"title": "Harry Potter and the Prisoner of Azkaban", "release_year": "2004", "genres": "Adventure Family Fantasy", "vote_average": 7.9, "popularity": 93.0, "overview": "Harry Potter, Ron and Hermione return to Hogwarts for their third year of study, where they delve into the mystery surrounding Sirius Black."},
        {"title": "Harry Potter and the Goblet of Fire", "release_year": "2005", "genres": "Adventure Family Fantasy", "vote_average": 7.7, "popularity": 91.0, "overview": "Harry Potter finds himself competing in a hazardous tournament between rival schools of magic."},
        {"title": "Harry Potter and the Order of the Phoenix", "release_year": "2007", "genres": "Adventure Family Fantasy", "vote_average": 7.5, "popularity": 89.0, "overview": "With their warning about Lord Voldemort's return scoffed at, Harry and Dumbledore are targeted by the wizard authorities."},
        {"title": "Fantastic Beasts and Where to Find Them", "release_year": "2016", "genres": "Adventure Family Fantasy", "vote_average": 7.3, "popularity": 85.0, "overview": "The adventures of writer Newt Scamander in New York's secret community of witches and wizards."},
        {"title": "The Lord of the Rings: The Fellowship of the Ring", "release_year": "2001", "genres": "Action Adventure Drama Fantasy", "vote_average": 8.8, "popularity": 96.0, "overview": "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the One Ring."},
        {"title": "The Lord of the Rings: The Two Towers", "release_year": "2002", "genres": "Action Adventure Drama Fantasy", "vote_average": 8.7, "popularity": 95.0, "overview": "While Frodo and Sam edge closer to Mordor with the help of Gollum, the divided fellowship makes a stand against Sauron."},
        {"title": "The Lord of the Rings: The Return of the King", "release_year": "2003", "genres": "Action Adventure Drama Fantasy", "vote_average": 9.0, "popularity": 98.0, "overview": "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam."},
        {"title": "The Hobbit: An Unexpected Journey", "release_year": "2012", "genres": "Action Adventure Fantasy", "vote_average": 7.8, "popularity": 90.0, "overview": "A reluctant Hobbit, Bilbo Baggins, sets out to the Lonely Mountain with a spirited group of dwarves."},
        {"title": "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe", "release_year": "2005", "genres": "Adventure Family Fantasy", "vote_average": 6.9, "popularity": 82.0, "overview": "Four kids travel through a wardrobe to the land of Narnia and learn of their destiny to free it."},
        {"title": "Percy Jackson & the Olympians: The Lightning Thief", "release_year": "2010", "genres": "Action Adventure Family Fantasy", "vote_average": 5.9, "popularity": 80.0, "overview": "A teenager discovers he's the descendant of a Greek god and sets out on an adventure to settle an on-going battle."},
        {"title": "Eragon", "release_year": "2006", "genres": "Action Adventure Family Fantasy", "vote_average": 5.1, "popularity": 75.0, "overview": "In his homeland of Alagaësia, a farm boy happens upon a dragon's egg -- a discovery that leads him to a wonderful journey."},
        {"title": "Bridge to Terabithia", "release_year": "2007", "genres": "Drama Family Fantasy", "vote_average": 7.2, "popularity": 84.0, "overview": "A preteen boy's life changes after he befriends the new girl at school, and they create an imaginary world."},
        {"title": "The Golden Compass", "release_year": "2007", "genres": "Action Adventure Family Fantasy", "vote_average": 6.1, "popularity": 79.0, "overview": "In a parallel universe, young Lyra Belacqua journeys to the far North to save her best friend."},
        {"title": "Stardust", "release_year": "2007", "genres": "Adventure Family Fantasy Romance", "vote_average": 7.6, "popularity": 86.0, "overview": "In a countryside town bordering a magical land, a young man promises his beloved a fallen star."},
        {"title": "Miss Peregrine's Home for Peculiar Children", "release_year": "2016", "genres": "Adventure Drama Family Fantasy", "vote_average": 6.7, "popularity": 83.0, "overview": "When Jacob discovers clues to a mystery that spans different worlds, he finds a magical place."},
        {"title": "Inkheart", "release_year": "2008", "genres": "Adventure Family Fantasy", "vote_average": 6.1, "popularity": 78.0, "overview": "A teenage girl discovers her father has an amazing talent to bring characters out of their books and into the real world."},
        {"title": "The Sorcerer's Apprentice", "release_year": "2010", "genres": "Action Adventure Family Fantasy", "vote_average": 6.1, "popularity": 81.0, "overview": "Master sorcerer Balthazar Blake must defend modern Manhattan from his arch-nemesis."},
        {"title": "Doctor Strange in the Multiverse of Madness", "release_year": "2022", "genres": "Action Adventure Fantasy Sci-Fi", "vote_average": 7.0, "popularity": 91.0, "overview": "Doctor Strange teams up with a mysterious teenage girl from his dreams who can travel across multiverses."}
    ]

    general_top_fallback = [
        {"title": "Inception", "release_year": "2010", "genres": "Action Sci-Fi Thriller", "vote_average": 8.8, "popularity": 95.0, "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task."},
        {"title": "Interstellar", "release_year": "2014", "genres": "Adventure Drama Sci-Fi", "vote_average": 8.6, "popularity": 94.0, "overview": "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot is tasked to find a new planet."},
        {"title": "The Dark Knight", "release_year": "2008", "genres": "Action Crime Drama", "vote_average": 9.0, "popularity": 97.0, "overview": "When the menace known as the Joker wreaks havoc on Gotham, Batman must accept his greatest test."},
        {"title": "Avatar", "release_year": "2009", "genres": "Action Adventure Sci-Fi", "vote_average": 7.6, "popularity": 92.0, "overview": "A paraplegic Marine dispatched to the moon Pandora becomes torn between following orders and protecting the world."},
        {"title": "Gladiator", "release_year": "2000", "genres": "Action Adventure Drama", "vote_average": 8.5, "popularity": 91.0, "overview": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family."},
        {"title": "The Matrix", "release_year": "1999", "genres": "Action Sci-Fi", "vote_average": 8.7, "popularity": 96.0, "overview": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth."},
        {"title": "Pulp Fiction", "release_year": "1994", "genres": "Crime Drama", "vote_average": 8.9, "popularity": 97.0, "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence."},
        {"title": "Fight Club", "release_year": "1999", "genres": "Drama", "vote_average": 8.8, "popularity": 95.0, "overview": "An insomniac office worker and a devil-may-care soap maker form an underground fight club."},
        {"title": "Forrest Gump", "release_year": "1994", "genres": "Drama Romance", "vote_average": 8.8, "popularity": 96.0, "overview": "The history of the United States from the 1950s to the '70s unfolds from the perspective of an Alabama man."},
        {"title": "The Shawshank Redemption", "release_year": "1994", "genres": "Drama", "vote_average": 9.3, "popularity": 99.0, "overview": "Over the course of several years, two convicts form a friendship, seeking solace and eventual redemption."},
        {"title": "Dune: Part Two", "release_year": "2024", "genres": "Action Adventure Drama Sci-Fi", "vote_average": 8.6, "popularity": 98.0, "overview": "Paul Atreides unites with Chani and the Fremen while seeking revenge against the conspirators who destroyed his family."},
        {"title": "Oppenheimer", "release_year": "2023", "genres": "Biography Drama History", "vote_average": 8.9, "popularity": 98.0, "overview": "The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb."},
        {"title": "Top Gun: Maverick", "release_year": "2022", "genres": "Action Drama", "vote_average": 8.3, "popularity": 94.0, "overview": "After thirty years, Maverick is still pushing the envelope as a top naval aviator."},
        {"title": "Spider-Man: Across the Spider-Verse", "release_year": "2023", "genres": "Action Animation Adventure Sci-Fi", "vote_average": 8.7, "popularity": 98.0, "overview": "Miles Morales catapults across the Multiverse, where he encounters a team of Spider-People."},
        {"title": "The Prestige", "release_year": "2006", "genres": "Drama Mystery Sci-Fi", "vote_average": 8.5, "popularity": 93.0, "overview": "After a tragic accident, two stage magicians in 1890s London engage in a battle to create the ultimate illusion."},
        {"title": "Se7en", "release_year": "1995", "genres": "Crime Drama Mystery", "vote_average": 8.6, "popularity": 94.0, "overview": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives."},
        {"title": "The Silence of the Lambs", "release_year": "1991", "genres": "Crime Drama Thriller", "vote_average": 8.6, "popularity": 93.0, "overview": "A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer."},
        {"title": "Parasite", "release_year": "2019", "genres": "Drama Thriller", "vote_average": 8.5, "popularity": 95.0, "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."},
        {"title": "Whiplash", "release_year": "2014", "genres": "Drama Music", "vote_average": 8.5, "popularity": 92.0, "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor."},
        {"title": "Shutter Island", "release_year": "2010", "genres": "Mystery Thriller", "vote_average": 8.2, "popularity": 91.0, "overview": "In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane."}
    ]

    # 1. Try Gemini AI Client
    client = get_client(access_key)
    if client:
        prompt = f"""
        The user wants exactly {top_n} recommendations for: "{query}".
        Return ONLY a JSON array of objects with keys: "title", "release_year", "genres", "vote_average", "popularity", "overview".
        Rule: Do NOT include numeric titles. Include exact franchise & genre matches.
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
            logger.warning(f"Gemini API limit/error: {e}")


    # 2. Try Mistral AI Client Failover
    mistral_recs = fetch_mistral_recommendations(query, top_n)
    if mistral_recs and len(mistral_recs) >= top_n:
        return mistral_recs[:top_n]

    # 3. Try Tavily Search API Failover
    tavily_recs = fetch_tavily_search_recommendations(query, top_n)
    if tavily_recs and len(tavily_recs) >= min(top_n, 4):
        # Merge tavily with fallback list to reach exact top_n count
        base_list = heist_crime_fallback if any(k in q_lower for k in ["heist", "money", "berlin", "lupin", "prison"]) else general_top_fallback
        combined = tavily_recs + [m for m in base_list if m['title'].lower() not in [t['title'].lower() for t in tavily_recs]]
        return combined[:top_n]

    # 4. Fallback to Curated 20-item lists (Guarantees exact top_n delivery)
    if any(k in q_lower for k in ["spider", "spiderman", "batman", "superman", "avengers", "iron man", "thor", "marvel"]):
        selected = superhero_action_fallback
    elif any(k in q_lower for k in ["heist", "money", "berlin", "lupin", "prison", "narcos", "breaking", "peaky", "robbery"]):
        selected = heist_crime_fallback
    elif any(k in q_lower for k in ["dragon", "house", "thrones", "got", "targaryen", "witcher"]):
        selected = fantasy_series_fallback
    elif any(k in q_lower for k in ["harry", "potter", "wizard", "magic", "beasts", "narnia"]):
        selected = wizard_fantasy_fallback
    else:
        selected = general_top_fallback

    return selected[:top_n]





