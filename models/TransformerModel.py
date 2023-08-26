from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from Levenshtein import distance as lev_distance


@st.cache_data
def get_best_matches(query, top_n, embeddings: np.array, sentences, model):
    if model != "Advanced":
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = embeddings.dot(query_embedding.T).ravel()
        best = np.argpartition(scores, -top_n)[-top_n:]
        sentences['scores'] = scores
        sentences.loc[list(best), 'lev_distance'] = sentences.loc[list(best), 'address'].apply(
            lambda x: lev_distance(query, x))
        best_matches = sentences[['id_building', 'address', 'scores', 'lev_distance']].loc[list(best)]
        best_matches_within_id_indices = best_matches.groupby(['id_building']).scores.transform(max) == best_matches.scores
        best_matches = best_matches.loc[best_matches_within_id_indices].sort_values(by='scores',
                                                                                    ascending=False).reset_index(drop=True)
        best_matches['scores'] = best_matches['scores'].apply(lambda x: round(x, 3))
        best_matches['lev_distance'] = best_matches['lev_distance'].astype('int')
        if len(best_matches[best_matches['scores'] >= 0.98]):
            return best_matches[best_matches['scores'] >= 0.98]
        elif len(best_matches[best_matches['scores'] >= 0.90]):
            return best_matches[best_matches['scores'] >= 0.90]
        else:
            return best_matches
    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = embeddings.dot(query_embedding.T).ravel()
        best = np.argpartition(scores, -top_n)[-top_n:]
        sentences['scores'] = scores
        sentences.loc[list(best), 'lev_distance'] = sentences.loc[list(best), 'address'].apply(
            lambda x: lev_distance(query, x))
        best_matches = sentences[['id_building', 'address', 'scores', 'lev_distance']].loc[list(best)]
        add_data = pd.read_pickle("data/processed/main_data.pkl")
        best_matches = best_matches.merge(add_data[['id_building', 'liter_building', 'name_district_full', 'name_town']])
        best_matches_within_id_indices = best_matches.groupby(['id_building']).scores.transform(max) == best_matches.scores
        best_matches = best_matches.loc[best_matches_within_id_indices].sort_values(by='scores',
                                                                                    ascending=False).reset_index(drop=True)

        # add logic to topn selection
        # add catboost here to assess probabilities
        if len(best_matches[best_matches['scores'] >= 0.98]):
            return best_matches[best_matches['scores'] >= 0.98]
        elif len(best_matches[best_matches['scores'] >= 0.90]):
            return best_matches[best_matches['scores'] >= 0.90]
        else:
            return best_matches