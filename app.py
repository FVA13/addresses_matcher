from sklearn.datasets import make_classification
import streamlit as st

# from utils import *
# import lightning as L
import pickle
import pandas as pd
import numpy as np

from models.TransformerModel import get_best_matches

# from PIL import Image

# icon = Image.open("icon.png")

st.set_page_config(
    page_title="Find the best Matching Addresses",
    # page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
)


# # @st.cache_data
# def predict(file, model_chosen):
#     # df = pd.read_csv()
#     # X, y = df[:-2], df[-1:]
#     X, y = make_classification(n_features=20, n_samples=1000, shuffle=True, random_state=21)
#     train, test = get_train_test_for_torch(X, y)
#     # if model_chosen == "Neural Network":
#     model = LitClassifier.load_from_checkpoint(
#         "notebooks/lightning_logs/version_7/checkpoints/epoch=1-step=1400.ckpt",
#         feautures_num=X.shape[1],
#     )
#     trainer = L.Trainer(max_epochs=2)
#     trainer.fit(model, data.DataLoader(train))
#     prediction = trainer.predict(
#         model,
#         data.DataLoader(torch.from_numpy(X).to(torch.float32)) #, num_workers=12)
#     )
#     return prediction


# def main():
#     file_uploaded = st.file_uploader("File uploader", type=["csv"])
#     model_chosen = st.selectbox("Select Option", ["Neural Network", "CatBoost"])
#     calculate = st.button('Calculate')
#     if calculate is True:
#         st.spinner()
#         with st.spinner(text='In progress'):
#             prediction = predict(file_uploaded, model_chosen)
#             st.table(response_to_numpy(prediction))
#         st.success('Done')

def main():
    query = st.text_input("Enter some text")
    calculate = st.button("Fetch Matches")
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
