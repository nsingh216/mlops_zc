import pandas as pd
import pickle
import os

from flask import Flask, request, jsonify

print("here4")
path = os.path.dirname(__file__)
print(path)
with open(f'{path}/model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)
    print("found model")



def predict(features):
    print("here5")
    print(features)
    preds = model.predict(features)
    print(preds)
    return float(preds[0])


app = Flask('bikeshare-pred')

@app.route('/', methods=['POST', 'GET'])
def index():
    return "hello world"


@app.route('/predict', methods=['POST', 'GET'])
def predict_endpoint():
    print("here3")

    print(request)


    ride = request.get_json()
    # print("hi")
    df=pd.DataFrame.from_records(ride)
    # print(df)
    # print(type(df))
    pred = predict(df)

    result = {
        'duration': pred
    }
    print(result)

    return jsonify(result)


if __name__ == "__main__":
    print("here1")
    app.run(debug=True, host='0.0.0.0', port=9696)
    print("here2")