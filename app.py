import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests

def fetch_poster(movie_id):
 
    api_key = "YOUR_TMDB_API_KEY"
    if api_key == "YOUR_TMDB_API_KEY":
        print("Please replace 'YOUR_TMDB_API_KEY' with your actual TMDB API key.")
        return None

    # Construct the API request URL for movie details
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            # Construct the full URL for the poster image
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for movie ID {movie_id}: {e}")
    return "https://via.placeholder.com/150" # Placeholder image if fetching fails
selected_movie_name = 'Avatar'
print(f"Recommendations for: {selected_movie_name}")
names, ids = recommend_tuned(selected_movie_name)

if names and ids:
    for i in range(len(names)):
        movie_name = names[i]
        movie_id = ids[i]
        poster_url = fetch_poster(movie_id)
        print(f"  - {movie_name} (ID: {movie_id})")
        print(f"    Poster URL: {poster_url}\n")
else:
    print("No recommendations found or an error occurred.")
    


st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: #0b0b0f;
    color: #f5f5f5;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.main-title {
    font-size: 52px;
    font-weight: 900;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 0px;
    color: #ffffff;
}

.main-title span {
    color: #e50914;
}

.sub-title {
    text-align: center;
    color: #9b9ba3;
    font-size: 18px;
    margin-bottom: 40px;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(135deg, #17171d, #101014);
    border: 1px solid #292930;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero-text {
    color: #a8a8b0;
    font-size: 16px;
    line-height: 1.6;
}

.movie-card {
    background: #17171d;
    border: 1px solid #292930;
    border-radius: 18px;
    padding: 22px;
    min-height: 190px;
    margin-bottom: 20px;
    transition: 0.3s;
}

.movie-card:hover {
    border-color: #e50914;
    transform: translateY(-3px);
}

.rank {
    color: #e50914;
    font-size: 14px;
    font-weight: 800;
    text-transform: uppercase;
}

.movie-title {
    color: #ffffff;
    font-size: 21px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 15px;
}

.score {
    color: #00d4aa;
    font-size: 14px;
    font-weight: 700;
}

.score-bar {
    height: 6px;
    background: #303038;
    border-radius: 10px;
    margin-top: 8px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 20px;
}

[data-testid="stSidebar"] {
    background: #111116;
    border-right: 1px solid #292930;
}

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    height: 48px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    with open("movie_list.pkl", "rb") as f:
        movies_df = pickle.load(f)

    with open("cv_tuned.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return movies_df, vectorizer

try:
    movies, cv = load_ml_assets()

    movies = movies.reset_index(drop=True)
    movies["tags"] = movies["tags"].fillna("").astype(str)

    movie_vectors = cv.transform(movies["tags"])

    st.sidebar.success("🟢 AI Engine Online")

except Exception as e:
    st.error(
        f"""
        ### ❌ Model Loading Error

        {e}

        Make sure these files are in the same folder as `app.py`:

        - `movie_list.pkl`
        - `cv_tuned.pkl`
        """
    )
    st.stop()

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("AI Movie Recommendation Engine")

    st.divider()

    st.markdown("### ⚙️ Recommendation Settings")

    num_recs = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=12,
        value=6
    )

    st.divider()

    st.markdown("### 🧠 Model Information")

    st.write(f"🎞️ Movies: **{len(movies):,}**")
    st.write(f"🔢 Features: **{movie_vectors.shape[1]:,}**")
    st.write("🤖 Model: **CountVectorizer + Cosine Similarity**")

    st.divider()

    st.caption(
        "CineMatch recommends movies based on similarity between "
        "movie metadata/tags."
    )

st.markdown(
    '<div class="main-title">🎬 Cine<span>Match</span> AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Discover movies similar to the ones you already love.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <div class="hero-title">
        🍿 Find your next favorite movie
    </div>

    <div class="hero-text">
        Select a movie you enjoyed and let CineMatch analyze
        thousands of movie profiles to find the closest matches.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">🔎 What did you enjoy?</div>',
    unsafe_allow_html=True
)

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].tolist(),
    index=0,
    label_visibility="collapsed"
)

generate = st.button(
    "✨ Find Similar Movies",
    type="primary",
    use_container_width=True
)

if generate:
    with st.spinner("🧠 Analyzing movie similarity..."):
        try:
            movie_index = movies.index[
                movies["title"] == selected_movie
            ][0]

            selected_vector = movie_vectors[movie_index]

            similarity_scores = cosine_similarity(
                selected_vector,
                movie_vectors
            ).flatten()

            similarity_scores[movie_index] = -1

            top_indices = similarity_scores.argsort()[::-1][:num_recs]

            recommendations = []

            for idx in top_indices:
                recommendations.append({
                    "title": movies.iloc[idx]["title"],
                    "score": similarity_scores[idx]
                })

            st.markdown(
                f'<div class="section-title">'
                f'🎯 Movies similar to "{selected_movie}"'
                f'</div>',
                unsafe_allow_html=True
            )

            columns = st.columns(3)

            for rank, movie in enumerate(recommendations):
                score_percentage = movie["score"] * 100

                with columns[rank % 3]:
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="rank">
                                #{rank + 1} Recommendation
                            </div>

                            <div class="movie-title">
                                {movie["title"]}
                            </div>

                            <div class="score">
                                Match Score: {score_percentage:.1f}%
                            </div>

                            <div class="score-bar">
                                <div style="
                                    width:{min(score_percentage, 100)}%;
                                    height:6px;
                                    border-radius:10px;
                                    background:#00d4aa;
                                "></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '📊 Recommendation Analytics'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Movies Analyzed",
                    f"{len(movies):,}"
                )

            with col2:
                st.metric(
                    "Features",
                    f"{movie_vectors.shape[1]:,}"
                )

            with col3:
                st.metric(
                    "Recommendations",
                    num_recs
                )

            with col4:
                avg_score = sum(
                    movie["score"]
                    for movie in recommendations
                ) / len(recommendations)

                st.metric(
                    "Average Match",
                    f"{avg_score * 100:.1f}%"
                )

        except Exception as e:
            st.error(f"❌ Recommendation engine error: {e}")

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        text-align:center;
        color:#666670;
        padding:20px;
        border-top:1px solid #222229;
    ">
        🎬 CineMatch AI &nbsp; • &nbsp;
        Content-Based Movie Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)
