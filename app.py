from sklearn.datasets import make_classification
import streamlit as st

# from utils import *
# import lightning as L
import pickle
import pandas as pd
import numpy as np

from models.TransformerModel import get_best_matches

st.set_page_config(
    page_title="Find the best Matching Addresses",
    # page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
)


def main():
    query = st.text_input("Enter some text")
    calculate = st.button("Fetch Matches")
    # add file upload
    with open("notebooks/embeddings_list.pkl", "rb") as f:
        embeddings_list = pickle.load(f)

    train_emb_only = pd.read_pickle("notebooks/names_embeddings_list.pkl")
    if calculate is True:
        st.spinner()
        with st.spinner(text="In progress"):
            prediction = get_best_matches(
                query=query,
                top_n=5,
                embeddings=np.array(embeddings_list),
                embeddings_decoding=train_emb_only,
            )
            st.table(prediction)
            st.success("Done")


if __name__ == "__main__":
    main()
