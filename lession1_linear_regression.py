# lession1_linear_regression.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def show_data_summary(input_data):
    print("Describe Data:")
    print(input_data.describe())

    print("\nFirst 10 rows:")
    print(input_data.head(10))
    print("....")

def data_hist(input_data):
    input_data.hist(bins=100, figsize=(20, 12))
    plt.show()

def data_scatter(input_data):
    input_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()

def permutation_split(data, ratio):
    permutation = np.random.permutation(len(data))
    train_size = int(len(data) * (1 - ratio))
    train_index = permutation[:train_size]
    test_index = permutation[train_size:]
    return data.iloc[train_index], data.iloc[test_index]

def encode_label(data):
    encoder = LabelEncoder()
    data["ocean_proximity"] = encoder.fit_transform(data["ocean_proximity"])

def imputer_by_median(data):
    imputer = Imputer(strategy="median")
    X = imputer.fit_transform(data)
    return pd.DataFrame(X, columns=data.columns)

def scale_data(data):
    scalar = MinMaxScaler(feature_range=(0, 100), copy=False)
    scalar.fit_transform(data)

def compare_scale_data(origin, scaled):
    plt.subplot(2, 1, 1)
    plt.scatter(x=origin["longitude"], y=origin["latitude"],
        c=origin["median_house_value"], cmap="viridis", alpha=0.1)
    plt.subplot(2, 1, 2)
    plt.scatter(x=scaled["longitude"], y=scaled["latitude"],
        c=origin["median_house_value"], cmap="viridis", alpha=0.1)
    plt.show()

def show_predict_result(test_data, test_value, predict_value):
    ax = plt.subplot(221)
    plt.scatter(x=test_data["longitude"], y=test_data["latitude"],
        s=test_value, c="dodgerblue", alpha=0.5)
    plt.subplot(222)
    plt.hist(test_value, color="dodgerblue")

    plt.subplot(223)
    plt.scatter(x=test_data["longitude"], y=test_data["latitude"],
        s=predict_value, c="lightseagreen", alpha=0.5)
    plt.subplot(224)
    plt.hist(predict_value, color="lightseagreen")

    plt.show()

def split_house_value(data):
    value = data["median_house_value"].copy()
    return data.drop(["median_house_value"], axis=1), value

def MES_evaluation(test_value, predict_value):
    mse = mean_squared_error(test_value, predict_value)
    return np.sqrt(mse)

if __name__ == "__main__":
    input_data = pd.read_csv("./data/housing.csv")
    # show_data_summary(input_data)
    # data_scatter(input_data)

    encode_label(input_data)
    input_data = imputer_by_median(input_data)
    # show_data_summary(input_data)

    scale_data(input_data)
    # compare_scale_data(pd.read_csv("./data/housing.csv"), input_data)

    train_set, test_set = train_test_split(input_data,
        test_size=0.1, random_state=59)
    train_data, train_value = split_house_value(train_set)
    test_data, test_value = split_house_value(test_set)
    #show_data_summary(train_data)

    linear_reg = LinearRegression()
    linear_reg.fit(train_data, train_value)

    predict_value = linear_reg.predict(test_data)
    # print("Diff: {}".format(MES_evaluation(test_value, predict_value)))

    scores = cross_val_score(linear_reg, train_data, train_value, cv=10)
    print("cross_val_score: {}".format(scores))

