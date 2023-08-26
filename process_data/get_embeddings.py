import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle


def get_embeddings(data, save_to_path="../data/processed/embeddings.pkl"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        data["address"].astype("str"),
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=128,
    )

    with open(save_to_path, "wb") as fp:
        pickle.dump(embeddings, fp)


if __name__ == "__main__":
    get_embeddings(pd.read_pickle("../data/processed/for_embeddings.pkl"))
