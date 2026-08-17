import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="CineMatch AI Engine", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 44px !important; font-weight: 800; color: #E50914; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px !important; text-align: center; color: #8c8c8c; margin-bottom: 35px; }
    .card-title { font-size: 20px !important; font-weight: 700; color: #f5f5f7; margin-bottom: 10px; }
    .card-score { font-size: 14px !important; color: #00adb5; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🎬 CineMatch AI Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Hyperparameter-tuned semantic item vector matching pipeline</p>', unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    with open('movie_list.pkl', 'rb') as f:
        movies_df = pickle.load(f)
    with open('similarity_tuned.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)
    return movies_df, similarity_matrix

try:
    movies, similarity = load_ml_assets()
    st.sidebar.success("🚀 AI Core Matrix Active")
except Exception as e:
    st.error(f"Asset extraction boundary alert: {e}. Check if .pkl files exist in this folder.")

st.write("### 🎛️ Configure Inference Controls")
input_col1, input_col2 = st.columns(2)

with input_col1:
    selected_movie = st.selectbox(
        "🍿 What movie did you recently enjoy?", 
        options=movies['title'].values,
        help="Type or select a title from the dataset matrix"
    )

with input_col2:
    num_recs = st.slider(
        "Target output results:", 
        min_value=3, 
        max_value=12, 
        value=6
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button("✨ Generate My Recommendations Matrix", type="primary", use_container_width=True):
    tab1, tab2 = st.tabs(["🎯 Top Matches Grid", "📊 Optimization Analytics"])
    
    with tab1:
        try:
            movie_index = movies[movies['title'] == selected_movie].index[0]
            similarity_scores = list(enumerate(similarity[movie_index]))
            sorted_recommendations = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recs+1]
            
            st.write("#### Optimized Results Matrix:")
            grid_cols = st.columns(3)
            
            for rank, (index, score) in enumerate(sorted_recommendations):
                recommended_title = movies.iloc[index]['title']
                with grid_cols[rank % 3]:
                    st.info(f"**Rank #{rank+1} Match**\n\n### {recommended_title}\n\nMatch Confidence: `{score:.2%}`")
                    
        except Exception as err:
            st.error(f"Prediction structural fault: {err}")
            
    with tab2:
        st.markdown("### 🛠️ Architecture Verification Profile")
        st.write(f"**Target Analyzed Subject:** {selected_movie}")
        st.write(f"**Vector Reference Length Evaluated:** {len(similarity)} records analyzed")
        st.caption("Information Grounding Note: These recommendations are surfaced by querying an optimized cosine distance matrix topology compiled from your tuned hyperparameter settings.")
