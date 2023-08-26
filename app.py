import streamlit as st
import pickle
import pandas as pd
import numpy as np
from models.TransformerModel import get_best_matches, multiple_best_matches

st.set_page_config(
    page_title="Find the best Matching Addresses",
    # page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
)


def main():
    query = st.text_input("Enter some text")
    file_uploaded = st.file_uploader("File uploader", type=["csv"])
    calculate = st.button("Fetch Matches")
    model_chosen = st.selectbox("Select Option Model", ["Basic", "Advanced"])
    with open("data/processed/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    for_embeddings = pd.read_pickle("data/processed/for_embeddings_with_names.pkl")
    if calculate is True:
        st.spinner()
        with st.spinner(text="In progress"):
            if file_uploaded:
                prediction = multiple_best_matches(
                    file=pd.read_csv(file_uploaded, index_col=0).values,
                    top_n=1,
                    embeddings=np.array(embeddings),
                    sentences=for_embeddings,
                    model=model_chosen,
                )
            else:
                prediction = get_best_matches(
                    query=query,
                    top_n=10,
                    embeddings=np.array(embeddings),
                    sentences=for_embeddings,
                    model=model_chosen,
                )
            st.table(prediction)
            st.success("Done")


if __name__ == "__main__":
    main()
