
import os
import pandas as pd
import predict
import requests


def get_sample_data(path):

    test_df = pd.read_csv(f"{path}/data/bike_share_cln_test.csv", low_memory=False)
    test_df.drop(columns='duration_minutes', inplace=True)
    ride  = test_df.loc[:0]

    return ride



def test1(path):

    ride = get_sample_data(path)
    result = predict.predict(ride)
    print(result)


def test2(path):
    ride = get_sample_data(path).to_dict(orient='records')
    print(ride)
    url = 'http://localhost:9696/predict'
    response = requests.post(url, json=ride)


    if response.status_code == 200:
        prediction = response.text
        # print(dict(prediction)["duration"])
        return prediction
    else:
        print("Encountered error")
        print(response.status_code)
        print(response.text)
        
    return response



def main():
    path = os.path.dirname(os.path.dirname(__file__))
    test2(path)


if __name__ == "__main__":
    main()