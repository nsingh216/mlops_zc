import os
import pandas as pd
import pickle
import preprocess as pp
import mlflow
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression



def train_lin_reg(X, y):
    """
    
    """
    model = LinearRegression()

    model.fit(X, y)

    model.predict(X)

    score = model.score(X, y)

    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_metric("score", score)

    print(score)

    return score, model


def test_model_w_mse(sklearn_model, test_x, test_y):
    """
    
    """
    pred_y = sklearn_model.predict(test_x)
    mse = mean_squared_error(test_y, pred_y)

    mlflow.log_metric("mse", mse)

    print(mse)

    return mse


def train_random_forest(X, y):
    """
    
    """
    model = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 5, random_state = 18)

    model.fit(X, y)

    model.predict(X)

    score = model.score(X, y)

    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_metric("score", score)

    print(score)

    return score, model


if __name__ == "__main__":


    # setup experiment tracking
    print(mlflow.get_tracking_uri())
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("austin-bikeshare")
    print(mlflow.get_tracking_uri())

    model_output_col = ['duration_minutes']


    with mlflow.start_run():

        # load train & test data
        train_x, train_y, test_x, test_y, tr_file, test_file = pp.get_preprocessed_data()
        

        # tests
        model_input_cols = train_x.columns
        # model_input_cols =  ['subscriber_type_is_local', 'subscriber_type_is_student', 'subscriber_type_is_youth', 'subscriber_type_is_single_trip', 'bike_type_enc', 'start_day_of_week', 'start_month', 'start_hour', 'start_minute', 'start_day_is_holiday', 'start_station_id', 'council_district', 'bike_id_enc']

        # save off initial training params
        mlflow.log_param("train_data", tr_file)
        mlflow.log_param("input_params", model_input_cols)

        # train model
        score, model = train_random_forest(train_x[model_input_cols], train_y)

        mse = test_model_w_mse(model, test_x[model_input_cols], test_y)

        mlflow.sklearn.log_model(model, artifact_path="models")

        with open('./model.pkl', 'wb') as f_out: 
            pickle.dump(model, f_out)


    # train_proc_df.corr().to_csv('corrplot.csv')
    # corr_plt = sns.heatmap(train_proc_df.corr())
    # plt.savefig('output.png')

    # train_lr(X, y)