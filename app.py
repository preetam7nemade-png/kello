import pickle
import textwrap
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/12161f/e8a33d?text=No+Poster"


# ──────────────────────────────────────────────────────────────────────────
# DATA / MODEL LOADING  (cached — runs once per session)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_data():
    with open("movie_list.pkl", "rb") as f:
        movies = pickle.load(f)
    with open("cv_tuned.pkl", "rb") as f:
        cv = pickle.load(f)

    movies = movies.reset_index(drop=True)
    vectors = cv.transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return movies, cv, vectors, similarity


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster(movie_id: int) -> str:
    api_key = st.secrets.get("TMDB_API_KEY", "")
    if not api_key:
        return PLACEHOLDER_POSTER
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        response = requests.get(url, timeout=6)
        response.raise_for_status()
        poster_path = response.json().get("poster_path")
        if poster_path:
            return f"{TMDB_IMG_BASE}{poster_path}"
    except Exception:
        pass
    return PLACEHOLDER_POSTER


def recommend(movie_title: str, movies: pd.DataFrame, similarity, num_recs: int):
    matches = movies[movies["title"] == movie_title]
    if matches.empty:
        return []
    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:num_recs]

    results = []
    for i, score in scores:
        results.append(
            {
                "title": movies.iloc[i]["title"],
                "movie_id": int(movies.iloc[i]["movie_id"]),
                "score": float(score),
            }
        )
    return results


# ──────────────────────────────────────────────────────────────────────────
# STYLE — Cinema Marquee identity
#   bg:      #0B0E14  near-black, blue-charcoal
#   card:    #12161F  ticket stub
#   marquee: #E8A33D  bulb gold
#   velvet:  #C1121F  curtain crimson
#   ink:     #F2EDE4  warm off-white
#   muted:   #8B92A6
#   display: 'Bebas Neue' (marquee poster type)
#   body:    'Inter'
#   mono:    'IBM Plex Mono' (scores / data)
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse 900px 500px at 50% -10%, rgba(232,163,61,0.10), transparent),
            #0B0E14;
        color: #F2EDE4;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* ---------- Marquee hero ---------- */
    .marquee-wrap {
        text-align: center;
        padding: 2.2rem 0 0.8rem 0;
    }
    .marquee-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.35em;
        font-size: 0.72rem;
        color: #8B92A6;
        text-transform: uppercase;
    }
    .marquee-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 5rem;
        line-height: 1;
        letter-spacing: 0.04em;
        color: #F2EDE4;
        text-shadow:
            0 0 14px rgba(232,163,61,0.55),
            0 0 2px rgba(232,163,61,0.9);
        margin: 0.2rem 0 0.3rem 0;
    }
    .marquee-title span { color: #E8A33D; }
    .marquee-sub {
        color: #8B92A6;
        font-size: 0.95rem;
        max-width: 480px;
        margin: 0 auto;
    }

    /* ---------- Film-strip sprocket divider (signature element) ---------- */
    .filmstrip {
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 1.6rem 0 1.8rem 0;
        opacity: 0.9;
    }
    .filmstrip .rail {
        flex: 1;
        height: 14px;
        background: repeating-linear-gradient(
            90deg,
            #1B2030 0px, #1B2030 10px,
            transparent 10px, transparent 22px
        );
        border-top: 1px solid #2A3040;
        border-bottom: 1px solid #2A3040;
        position: relative;
    }
    .filmstrip .rail::before {
        content: "";
    }

    /* ---------- Section titles ---------- */
    .section-title {
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 0.06em;
        font-size: 1.5rem;
        color: #E8A33D;
        margin: 0.4rem 0 0.9rem 0;
        border-left: 3px solid #C1121F;
        padding-left: 0.6rem;
    }

    /* ---------- Picker card ---------- */
    .picker-card {
        background: #12161F;
        border: 1px solid #202636;
        border-radius: 6px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }

    .stSelectbox label, .stSlider label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8B92A6 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0B0E14 !important;
        border-color: #2A3040 !important;
        border-radius: 4px !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(180deg, #D6862F 0%, #C1121F 100%);
        color: #0B0E14;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.15rem;
        letter-spacing: 0.06em;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 0;
        margin-top: 0.6rem;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
        box-shadow: 0 4px 14px rgba(193,18,31,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(232,163,61,0.35);
        color: #0B0E14;
    }

    /* ---------- Ticket-stub recommendation card ---------- */
    .ticket {
        position: relative;
        background: #12161F;
        border: 1px solid #202636;
        border-radius: 8px;
        margin-top: 0.7rem;
        overflow: hidden;
    }
    .ticket-notch {
        position: absolute;
        top: -9px; left: 50%;
        transform: translateX(-50%);
        width: 18px; height: 18px;
        background: #0B0E14;
        border-radius: 50%;
        border: 1px solid #202636;
    }
    .ticket-body {
        padding: 0.9rem 1rem 1rem 1rem;
        border-top: 1px dashed #2A3040;
    }
    .ticket-rank {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        color: #0B0E14;
        background: #E8A33D;
        padding: 2px 8px;
        border-radius: 3px;
        margin-bottom: 0.5rem;
    }
    .ticket-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.35rem;
        letter-spacing: 0.02em;
        color: #F2EDE4;
        line-height: 1.15;
        min-height: 2.4rem;
    }
    .ticket-score-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-top: 0.6rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #8B92A6;
    }
    .ticket-score-val {
        color: #E8A33D;
        font-size: 0.85rem;
    }
    .ticket-bar-track {
        width: 100%;
        height: 4px;
        background: #202636;
        border-radius: 4px;
        margin-top: 0.35rem;
        overflow: hidden;
    }
    .ticket-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #C1121F, #E8A33D);
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: #12161F;
        border: 1px solid #202636;
        border-radius: 6px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8B92A6 !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Bebas Neue', sans-serif;
        color: #E8A33D !important;
    }

    .footer-strip {
        text-align: center;
        color: #5A6070;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        padding: 1.8rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def filmstrip_divider():
    st.markdown('<div class="filmstrip"><div class="rail"></div></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="marquee-wrap">
        <div class="marquee-eyebrow">NOW SHOWING · CONTENT-BASED ENGINE</div>
        <div class="marquee-title">CINE<span>MATCH</span></div>
        <div class="marquee-sub">
            Pick a title you love. We'll read its tags and find the films
            that share its DNA.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
filmstrip_divider()

# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
try:
    movies, cv, movie_vectors, similarity = load_data()
except FileNotFoundError as e:
    st.error(
        f"❌ Could not find a required file: {e}. "
        "Make sure `movie_list.pkl` and `cv_tuned.pkl` sit next to `app.py`."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# PICKER
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🎬 Choose a Movie</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="picker-card">', unsafe_allow_html=True)

    pick_col, slider_col = st.columns([2.2, 1])

    with pick_col:
        selected_movie = st.selectbox(
            "Search titles",
            options=movies["title"].values,
            index=0,
            placeholder="Type a movie title…",
        )

    with slider_col:
        num_recs = st.slider("Number of picks", min_value=3, max_value=15, value=6)

    find_clicked = st.button("✨ Find Similar Movies")

    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────
if find_clicked:
    try:
        recommendations = recommend(selected_movie, movies, similarity, num_recs)

        if not recommendations:
            st.warning("No matches found for that title — try another one.")
        else:
            filmstrip_divider()
            st.markdown(
                f'<div class="section-title">🍿 Because you liked "{selected_movie}"</div>',
                unsafe_allow_html=True,
            )

            columns = st.columns(3)

            for rank, movie in enumerate(recommendations):
                score_percentage = movie["score"] * 100

                with columns[rank % 3]:
                    poster = fetch_poster(movie["movie_id"])
                    st.image(poster, use_container_width=True)

                    ticket_html = textwrap.dedent(f"""
                        <div class="ticket">
                            <div class="ticket-notch"></div>
                            <div class="ticket-body">
                                <span class="ticket-rank">PICK #{rank + 1}</span>
                                <div class="ticket-title">{movie["title"]}</div>
                                <div class="ticket-score-row">
                                    <span>MATCH</span>
                                    <span class="ticket-score-val">{score_percentage:.1f}%</span>
                                </div>
                                <div class="ticket-bar-track">
                                    <div class="ticket-bar-fill" style="width:{min(score_percentage, 100)}%;"></div>
                                </div>
                            </div>
                        </div>
                    """)
                    st.markdown(ticket_html, unsafe_allow_html=True)

            filmstrip_divider()

            st.markdown(
                '<div class="section-title">📊 Recommendation Analytics</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Movies Analyzed", f"{len(movies):,}")

            with col2:
                st.metric("Features", f"{movie_vectors.shape[1]:,}")

            with col3:
                st.metric("Recommendations", num_recs)

            with col4:
                avg_score = sum(m["score"] for m in recommendations) / len(recommendations)
                st.metric("Average Similarity", f"{avg_score * 100:.1f}%")

    except Exception as e:
        st.error(f"❌ Recommendation engine error: {e}")

# ──────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-strip">
        🎞️ CINEMATCH &nbsp;·&nbsp; CONTENT-BASED MOVIE RECOMMENDATION SYSTEM
    </div>
    """,
    unsafe_allow_html=True,
)

