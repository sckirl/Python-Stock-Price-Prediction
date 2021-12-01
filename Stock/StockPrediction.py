# import important packages
import pandas_datareader as pdr
import pandas as pd
import keras
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime

recent = None
scaler = MinMaxScaler((0, 1))

def addStock(stock: str, 
            start: str = None, end=datetime.today(), 
            source="av-daily",
            api_key=None,
            save = False):
    
    print("Adding {}...".format(stock))
    # try to get the data from source
    try:
        # get the data of the stock from the web, via pandas_datareader
        dataset = pdr.DataReader(stock, source, start, end, api_key=api_key)
        global recent
        recent = stock
    
    # sometimes source is unreachable, notify to user
    except Exception as e: return print(e)

    if save: dataset.to_csv("StockList/{}.csv".format(stock))
    return dataset


def visualize(inputs: list[tuple((pd.DataFrame, str))]):
    # inputs should a list be filled with pd.Dataframe and name of the data as a tuple
    # inputs = [(Data, "name"), (...), ...]
    # visualize the price of a stock.
    plt.figure(figsize=(16, 8))

    for input in inputs:
        plt.plot(input[0], label=input[1])
    plt.legend()

    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Price History", fontsize=20)
    plt.show()

def normalize(stock: str or pd.DataFrame):
    # get the name of the stock
    global recent
    recent = stock 
    
    if type(stock) == str:
        # check if the file exists within the StockList folder
        if not os.path.isfile('StockList/{}.csv'.format(stock)):
            data = pd.read_csv('StockList/{}.csv'.format(stock))
        else: 
            # if the file doesn't exists, put a confirmation of user to add the stock
            addStock(input("Type NASDAQ stock to add: "))

    else: data = stock
    
    # make the price to be in the range of 0 and 1
    scaledData = scaler.fit_transform(data.filter(["Close"])) 
    trainData = scaledData[:len(data), :]

    trainX = []
    trainY = []

    # send the values of arrays into trainX and trainY arrays, making 2 dimensional array.
    # trainX will have the previous 60 days of stock price, while trainY will have the
    # next day's price. This trainY will be the target/answer of the ML model.
    for i in range(60, len(trainData)):
        trainX.append(scaledData[i-60:i,0])
        trainY.append(scaledData[i, 0])

    # reshape it into numpy array for training
    trainX, trainY = np.array(trainX), np.array(trainY)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    return trainX, trainY

def addModel():
    # make the keras model
    print("Making Model...")

    model = keras.Sequential(
        [
            LSTM(50, activation="relu", return_sequences=True),
            LSTM(50, activation="relu"),
            Dense(3),
            Dense(1)
        ]
    )

    model.compile(optimizer='adam', loss='mse')
    return model

def train(trainX: np.array, trainY: np.array, batch=1, epoch=10):
    # train the model from the input/trainX and output/trainY data
    file = "ModelList/{}".format(recent)

    # if the model has been trained before, train on the same file
    if os.path.exists(file): 
        model = keras.models.load_model("ModelList/{}".format(recent))
        model.fit(trainX, trainY, batch, epoch)

    else: 
        # otherwise, make the model and save it. This statement is to prevent unecessary process
        # of retraining the model from the beginning, instead of continue training a trained model.
        model = addModel()
        model.fit(trainX, trainY, batch, epoch)

    model.save("ModelList/{}".format(recent))
    return model

# test the prediction accuracy, if evaluate is set to True it will show a comparison graph.
def test(model: keras.engine.sequential.Sequential, stock: pd.DataFrame, evaluate=False):
    # make the data to be in the scale of 0 to 1
    scaledData = scaler.fit_transform(stock.filter(["Close"]))
    scaledData = np.concatenate((scaledData[:60][::-1], scaledData))

    # make the data to be numpy array, just like the train process.
    testData = np.array([scaledData[i-60:i,0] 
                            for i in range(60, len(scaledData))])

    testData = np.reshape(testData, 
                            (testData.shape[0], 
                            testData.shape[1], 1))
    
    # predict and make the prediction to be a part of data/stock. This is to make 
    # predicted data to be at the same date/location as the actual data.
    prediction = model.predict(testData)
    prediction = scaler.inverse_transform(prediction)
    stock["Prediction"] = prediction

    if evaluate: visualize(inputs=[(stock["Close"], "Real"), 
                          (stock["Prediction"], "Prediction")])
    return stock["Prediction"]