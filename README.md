# 🎬 CinemaVerse - Movie Recommendation System

A modern, Netflix-inspired **Movie Recommendation Application** built with **Python** and **Streamlit**. Powered by a content similarity matrix over 300,000+ movies, interactive search autocomplete, movie posters, and detailed movie synopses.

---

## ✨ Key Features

- **🎨 Dark Cinema UI/UX**: Modern midnight dark theme with glowing accents, glassmorphic cards, rating badges, and smooth hover effects.
- **🔍 Search with Live Autocomplete**: Case-insensitive instant movie search with suggestions as you type across 303,000+ dataset entries.
- **⚡ Fast Recommendation Engine**: Sub-second recommendation calculation using similarity matrix dot products.
- **📌 Movie Highlights & Overview**:
  - Detailed movie synopses & "Why You'll Like It" breakdowns.
  - Vibe/Mood classifications, ideal audience tags, and movie trivia.
- **🖼️ High-Resolution Poster Artwork**: Sleek movie poster visuals and dynamic artwork cards.
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
├── recommend.py            # Cosine similarity calculation & recommendation logic
├── utils.py                # Cached dataset loading & search autocomplete
├── poster_helper.py        # Poster rendering helper
├── insights_helper.py      # Detailed movie synopses & fallback recommender
├── requirements.txt        # Required Python packages
├── df.pkl                  # Movie dataset
├── indices.pkl             # Movie title index mapping
├── tfidf_matrix.pkl        # Sparse similarity matrix pickle
└── README.md               # User guide & documentation
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```
*(Or run `python main.py`)*

---

## 🌐 Deploying to Streamlit Community Cloud (share.streamlit.io)

Follow these step-by-step instructions to deploy your app online for free:

### Step 1: Push Code to GitHub
1. Initialize git and commit your code:
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .
   git commit -m "Deploy CinemaVerse Movie Recommender"
   ```
2. Connect your remote repository and push:
   ```bash
   git branch -M main
   git push -u origin main --force
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click the **"New app"** button.
3. Select your GitHub repository (`Movie-Recommendations-Using-NLP-and-ML`), branch (`main`), and set Main file path to **`app.py`**.
4. Click **"Advanced settings..."** → **"Secrets"** and paste your configuration key:
   ```toml
   GEMINI_API_KEY = "your_access_key_here"
   ```
5. Click **"Save"**, then click **"Deploy!"**.

---

## 💻 Tech Stack
- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy, SciPy
- **Styling**: Vanilla CSS (Injected via `st.markdown`)
- **API Fetching**: Requests (TMDB & OMDb)
