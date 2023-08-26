from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from Levenshtein import distance


@st.cache_data
def get_best_matches(query, top_n, embeddings: np.array, embeddings_decoding):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentences = model.encode(query, convert_to_tensor=True)
    scores = embeddings.dot(sentences.T).ravel()
    best = np.argpartition(scores, -top_n)[-top_n:]
    embeddings_decoding["scores"] = scores
    # split into words and measure levenstein
    embeddings_decoding.loc[list(best), "lev_distance"] = embeddings_decoding.loc[
        list(best), "full_address"
    ].apply(lambda x: distance(query, x))
    return (
        embeddings_decoding[["id_building", "full_address", "scores", "lev_distance"]]
        .loc[list(best)]
        .sort_values(by="lev_distance")
        .reset_index(drop=True)
    )
