import pandas as pd

from pullenti_wrapper.langs import set_langs, RU
from pullenti_wrapper.processor import Processor, GEO, ADDRESS
from pullenti_wrapper.referent import Referent
from catboost import CatBoostClassifier, Pool, metrics
from sklearn.model_selection import train_test_split

from TransformerModel import *

set_langs([RU])


processor = Processor([GEO, ADDRESS])


def coalesce(s: pd.Series, *series):
    """coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


# recursive function
def get_ner_elements(referent, level=0):
    tmp = {}
    a = ""
    b = ""
    for key in referent.__shortcuts__:
        value = getattr(referent, key)
        if value in (None, 0, -1):
            continue
        if isinstance(value, Referent):
            get_ner_elements(value, level + 1)
        else:
            if key == "type":
                a = value
            if key == "name":
                b = value
                # print('ok', value)
            if key == "house":
                a = "дом"
                b = value
                tmp[a] = b
            if key == "flat":
                a = "квартира"
                b = value
                # print('ok', value)
                tmp[a] = b
            if key == "building":
                a = "Литера"
                b = value
                tmp[a] = b
        if key == "corpus":
            a = "корпус"
            b = value
            tmp[a] = b
    tmp[a] = b
    addr.append(tmp)
    return addr


def get_address_ner_objects(address: str):
    global addr  # we need to declare global variable to change it
    processor = Processor([GEO, ADDRESS])
    addr = []
    try:
        result = processor(address)
        referent = result.matches[0].referent
        ner_model_res = get_ner_elements(referent)
        merged_data = {k: v for d in ner_model_res for k, v in d.items()}
        data = pd.DataFrame([merged_data])
        return merged_data
    except:
        return []


def input_transformation(user_input):
    user_input = pd.DataFrame(
        columns=[
            "город",
            "улица",
            "дом",
            "поселок",
            "проспект",
            "корпус",
            "Литера",
            "муниципальный район",
            "набережная",
            "шоссе",
            "парк",
            "переулок",
            "площадь",
            "аллея",
            "",
            "линия",
            "автодорога",
            "микрорайон",
            "деревня",
            "проезд",
            "квартал",
            "бульвар",
            "станция",
            "район",
            "территория",
            "муниципальный округ",
            "мост",
            "тупик",
            "область",
            "село",
            "поселок городского типа",
            "округ",
            "волость",
        ]
    )  # rewrite
    user_res = pd.concat(
        [user_input, pd.DataFrame(get_address_ner_objects(QUERY), index=[0])]
    )
    user_res["ner_city"] = coalesce(
        user_res["город"],
        user_res["поселок"],
        user_res["деревня"],
        user_res["село"],
        user_res["поселок городского типа"],
    )
    user_res["ner_street"] = coalesce(
        user_res["улица"],
        user_res["проспект"],
        user_res["набережная"],
        user_res["шоссе"],
        user_res["парк"],
        user_res["переулок"],
        user_res["площадь"],
        user_res["аллея"],
        user_res["линия"],
        user_res["автодорога"],
        user_res["проезд"],
        user_res["бульвар"],
    )
    user_res["ner_house"] = user_res["дом"]
    user_res["ner_corpus"] = user_res["корпус"]
    user_res["ner_liter"] = user_res["Литера"]
    user_res["ner_district"] = coalesce(
        user_res["муниципальный район"],
        user_res["микрорайон"],
        user_res["квартал"],
        user_res["район"],
        user_res["муниципальный округ"],
    )
    user_res["ner_area"] = coalesce(
        user_res["станция"], user_res["территория"], user_res["мост"], user_res["тупик"]
    )
    user_res["ner_subject"] = coalesce(
        user_res["область"], user_res["округ"], user_res["волость"]
    )
    user_res = user_res[[col for col in user_res if col.startswith("ner")]]
    return user_res


def get_advanced_matching_scores(input_query, embeddings: np.array, sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    raw_ranking = get_best_matches(input_query, 3, embeddings, sentences)
    raw_ranking_ids = raw_ranking.id_building
    query = input_transformation(input_query).dropna(axis=1)
    mapping = pd.read_pickle("../data/processed/mapping_id_building_index.pkl")
    raw_ranking_indices = mapping.query("id_building in @raw_ranking_ids").index
    for col in query.columns:
        col_value = query[col].values[0]
        query_col_embedding = model.encode(
            col_value, convert_to_tensor=True, normalize_embeddings=True
        )
        with open("../data/processed/embeddings_{}.pkl".format(col), "rb") as f:
            db_col_embedding = np.array(pickle.load(f)[list(raw_ranking_indices)])
        raw_ranking.loc[:, "score_" + col] = db_col_embedding.dot(
            query_col_embedding.T
        ).ravel()
    raw_ranking.loc[:, "original_query"] = input_query
    return raw_ranking


def multiple_advanced_matching_scores(df, embeddings: np.array, sentences):
    res = pd.DataFrame()
    for index, row in df.iterrows():
        i_address = row.address.replace("(", "").replace(")", "")
        i_df = get_advanced_matching_scores(i_address, embeddings, sentences)
        i_df.loc[:, "y_correct"] = i_df.id_building == row.target_building_id
        res = pd.concat([res, i_df])
    return res.reset_index(drop=True)


def get_advanced_pred(data, embeddings, for_embeddings):
    model = CatBoostClassifier()
    model.load_model('../models/CatBoostClassifier')
    data = multiple_advanced_matching_scores(
        data, np.array(embeddings), for_embeddings
    ).drop(columns=["address", "id_building"])
    result = model.predict(data)
    return result
