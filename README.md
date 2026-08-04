# 🎬 CinemaVerse AI - Professional Movie Recommendation System

A modern, Netflix-inspired **AI Movie Recommendation Application** built with 100% pure **Python** and **Streamlit**. Powered by a pre-calculated TF-IDF cosine similarity matrix over 300,000+ movies, TMDB/OMDb poster integration, and **Google Gemini AI** for intelligent recommendations, summaries, and fallback insights.

---

## ✨ Key Features

- **🎨 Dark Netflix/IMDb UI/UX**: Sleek midnight theme with glowing accents, glassmorphic cards, rating badges, and smooth hover effects.
- **🔍 Search with Live Autocomplete**: Case-insensitive instant movie search with suggestions as you type across 303,000+ dataset entries.
- **⚡ Sparse Similarity Engine**: Fast sub-second recommendation calculation using sparse TF-IDF matrix dot products.
- **🤖 Gemini AI Integration**:
  - AI-generated movie synopses & "Why Watch This" breakdowns.
  - Vibe/Mood classifications, ideal audience tags, and fun facts.
  - **Gemini Fallback Recommender**: Automatically steps in if a movie is not found in the local matrix dataset.
- **🖼️ TMDB & OMDb Poster Fetcher**: High-resolution poster images with dynamic base64 SVG poster artwork fallbacks.
- **📱 Detailed Movie View**: Complete movie specs, YouTube trailer links, IMDb & TMDB links.
- **❤️ Favorites & Recent History**: Bookmarks list stored in session state.
- **🎲 Surprise Me Button**: Random movie discovery feature.

---

## 🛠️ Project Structure

```
Movie_Recommendation/
├── app.py                  # Main Streamlit application
├── main.py                 # Launcher script
├── style.css               # Custom dark movie-themed CSS styling
├── recommend.py            # Cosine similarity calculation & Gemini fallback logic
├── utils.py                # Cached dataset loading & search autocomplete
├── poster_helper.py        # TMDB/OMDb poster fetcher & dynamic SVG artwork generator
├── gemini_helper.py        # Google Gemini AI insights & fallback recommender
├── requirements.txt        # Required Python packages
├── df.pkl                  # Movie dataset (Pandas DataFrame)
├── indices.pkl             # Movie title index mapping
├── tfidf_matrix.pkl        # TF-IDF Sparse matrix pickle
└── README.md               # User guide & documentation
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
*(Or run `python main.py`)*

### 3. API Keys (Optional)
- **Google Gemini API Key**: Enter your key in the app sidebar to enable AI synopses and intelligent fallback recommendations. You can also export `GEMINI_API_KEY` in your environment or `.env`.
- **TMDB API Key**: Enter your key in the sidebar for official high-resolution posters.

---

## 🌐 Deploying to Streamlit Community Cloud (share.streamlit.io)

Follow these step-by-step instructions to deploy your app online for free:

### Step 1: Push Code to GitHub
1. Initialize git (if not already done) and commit your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of CinemaVerse AI Movie Recommender"
   ```
2. Create a new repository on [GitHub](https://github.com/new).
3. Connect your local repository and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/Movie_Recommendation.git
   git branch -M main
   git push -u origin main
   ```
   > **Note on Large Pickle Files**: `tfidf_matrix.pkl` is ~107MB. Use **Git LFS** (`git lfs track "*.pkl"`) or host large pickle files on GitHub Releases / Google Drive if GitHub warns about file size limits.

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click the **"New app"** button.
3. Select your GitHub repository (`Movie_Recommendation`), branch (`main`), and set Main file path to **`app.py`**.

### Step 3: Add API Keys in App Secrets
1. Before clicking Deploy, click **"Advanced settings..."** → **"Secrets"**.
2. Paste your secret keys in TOML format:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   TMDB_API_KEY = "your_tmdb_api_key_here"
   ```
3. Click **"Save"**, then click **"Deploy!"**. Your app will be live on a public `.streamlit.app` link within minutes!

---

## 💻 Tech Stack
- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy, SciPy
- **AI Integration**: Google Gemini API (`google-genai`)
- **Styling**: Vanilla CSS (Injected via `st.markdown`)
- **API Fetching**: Requests (TMDB & OMDb APIs)

