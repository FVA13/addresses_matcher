from flask import Flask, jsonify, request
from models.TransformerModel import get_best_matches_json

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def result():
    request_json = request.json
    prediction = get_best_matches_json(request_json)
    return jsonify({"prediction": str(prediction)})


if __name__ == "__main__":
    app.run()


# {
#     "success": true,
# “query:”:[
#     "address": "аптерский 18 спб"
# ],
# "result": [
#     {
# “target_building_id”: 209676 ,
# “target_address”:  “г.Санкт-Петербург, Аптекарский проспект, дом 18, литера А” }
# ]
# }
