import StockPrediction
import datetime
import pandas as pd
from dateutil import parser
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pytz

class Prediction:
    def __init__(self, stock: pd.DataFrame, model: keras.engine.sequential.Sequential):
        self.stock = stock
        self.model = model
        self.scaler = MinMaxScaler((0, 1))

        self.mostRecentStock = self.stock.iloc[-1].name
        if type(self.mostRecentStock) == str: 
                self.mostRecentStock = parser.parse(self.mostRecentStock)

    def nextDay(self):
        scaledValues = self.scaler.fit_transform(self.stock.filter(["Close"]).values)

        prediction = self.scaledNextDay(scaledValues)
        # return prediction as real price
        res = self.scaler.inverse_transform(prediction)[0][0]
        
        return res, pd.Series(res, 
                                index=pd.date_range(start=self.mostRecentStock+datetime.timedelta(days=1),
                                periods=1, freq="D"), 
                                name="Close")

    # returns the next day prediction in scaled form
    def scaledNextDay(self, data):
        # get the last 60 days of stock price, reshape to numpyData (just like in train process).
        numpyData = np.array([data[-60:]])
        numpyData = np.reshape(numpyData, (numpyData.shape[0], numpyData.shape[1], 1))
        # return prediction as scaled values (0 to 1)
        prediction = self.model.predict(numpyData)

        return prediction

    # prediction of stock price in a given day (may not work as expected)
    def onDate(self, date: datetime.datetime): 
        # count the duration of mostRecentStock to date
        # ex: 2021-1-1 to 2021-1-7 takes 6 days. 6 is the duration
        days = (date-self.mostRecentStock).days 
        if days <= 0: raise Exception("Can't predict past prices")

        scaledValues = self.scaler.fit_transform(self.stock.filter(["Close"]).values)
        prices = []

        # loop until given date
        for _ in range(days):
                pred = self.scaledNextDay(scaledValues)
                # pred returns scaled values (from 0 to 1), so real prices should be inverse_transform'd.
                # and scaledValues will get updated each loop, means keep adding the new prices.
                prices.append(self.scaler.inverse_transform(pred)[0][0])
                scaledValues = np.concatenate((scaledValues, pred))
        
        start = self.mostRecentStock + datetime.timedelta(days=1)
        # return the serial as pd.Series, so its more manageable
        return prices[-1], pd.Series(prices, 
                                        index=pd.date_range(start=start, periods=len(prices), 
                                        freq="D"), name="Close")

    # prediction of day on when the stock reaches given price (may not work as expected)
    def onPrice(self, price):
        # count the duration of mostRecentStock to date
        scaledPrice = self.scaler.fit_transform(self.stock.filter(["Close"]).values)

        # the price should have the most recent stock's price
        prices = [self.stock.filter(["Close"]).values[-1]]

        # loop until the most recent price is at the price target
        while prices[-1] < price:
                pred = self.scaledNextDay(scaledPrice)

                # pred returns scaled values (from 0 to 1), so real prices should be inverse_transform'd.
                # and scaledValues will get updated each loop, means keep adding the new prices.
                prices.append(self.scaler.inverse_transform(pred)[0][0])
                scaledPrice = np.concatenate((scaledPrice, pred))

                # tell price estimation to the user every 30 days
                if len(prices) % 30 == 0 :
                        print("Checking the next {} days - ${}".format(len(prices), prices[-1]))
                
        dates = pd.date_range(self.mostRecentStock, periods=len(prices), freq="D")
        return dates[-1], pd.Series(prices, 
                                        index=dates, 
                                        name="Close")