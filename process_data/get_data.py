import pandas as pd
import numpy as np


def join_non_null_values(l):
    res = [str(x) for x in l if str(x) != "nan"]
    return ", ".join(res)


main_data = (
    pd.read_csv("../data/raw/additional_data/building_20230808.csv")
    .add_suffix("_building")
    .merge(
        pd.read_csv("../data/raw/additional_data/district_20230808.csv").add_suffix(
            "_district"
        ),
        left_on="district_id_building",
        right_on="id_district",
        how="left",
    )
    .merge(
        pd.read_csv("../data/raw/additional_data/prefix_20230808.csv").add_suffix(
            "_prefix"
        ),
        left_on="prefix_id_building",
        right_on="id_prefix",
        how="left",
    )
    .merge(
        pd.read_csv("../data/raw/additional_data/town_20230808.csv").add_suffix(
            "_town"
        ),
        left_on="town_id_prefix",
        right_on="id_town",
        how="left",
    )
    .merge(
        pd.read_csv("../data/raw/additional_data/geonim_20230808.csv").add_suffix(
            "_geonim"
        ),
        left_on="geonim_id_prefix",
        right_on="id_geonim",
        how="left",
    )
    .merge(
        pd.read_csv("../data/raw/additional_data/geonimtype_20230808.csv").add_suffix(
            "_geonimtype"
        ),
        left_on="type_id_geonim",
        right_on="id_geonimtype",
        how="left",
    )
    .merge(
        pd.read_csv("../data/raw/additional_data/area_20230808.csv").add_suffix(
            "_area"
        ),
        left_on="area_id_prefix",
        right_on="id_area",
        how="left",
    )
    .merge(
        pd.read_csv("../data/raw/additional_data/areatype_20230808.csv").add_suffix(
            "_areatype"
        ),
        left_on="type_id_area",
        right_on="id_areatype",
        how="left",
    )
    .assign(name_district_full=lambda df_: df_.name_district + " район")
    .assign(
        all_in_field=lambda df_: df_[
            [
                "full_address_building",
                "type_building",
                "name_district_full",
                "name_area",
                "name_areatype",
            ]
        ].apply(lambda x: join_non_null_values(x), axis=1)
    )
)

main_data.to_pickle("../data/processed/main_data.pkl")

for_embeddings = (
    pd.DataFrame(
        np.concatenate([
            main_data[['id_building', 'full_address_building']].values,
            main_data[['id_building', 'all_in_field']].values,
            # train[['id_building', 'short_address_building']].values,
            # train[['id_building', 'name_prefix']].values,
            # train[['id_building', 'short_name_prefix']].values,
            # train[['id_building', 'name_town']].values,
            # train[['id_building', 'short_name_town']].values,
            # train[['id_building', 'name_geonim']].values,
            # train[['id_building', 'short_name_geonim']].values,
            # train[['id_building', 'only_name_geonim']].values,
            # train[['id_building', 'name_prefix']].values,
            # train[['id_building', 'short_name_prefix']].values,
            # train[['id_building', 'name_geonimtype']].values,
            # train[['id_building', 'only_name_area']].values,
            # train[['id_building', 'name_areatype']].values,
            # train[['id_building', 'short_name_areatype']].values,
            # train[['id_building', 'name_district']].values,
            # train[['id_building', 'name_district_full']].values,
        ], axis=0),
        columns=['id_building', 'address']
    )
    .dropna()
    .reset_index(drop=True)
)

for_embeddings.to_pickle('../data/processed/for_embeddings_with_names.pkl')
