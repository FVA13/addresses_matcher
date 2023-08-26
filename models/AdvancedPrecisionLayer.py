import pandas as pd

from pullenti_wrapper.langs import set_langs, RU
from pullenti_wrapper.processor import Processor, GEO, ADDRESS
from pullenti_wrapper.referent import Referent

set_langs([RU])


processor = Processor([GEO, ADDRESS])


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
