from matplotlib import pyplot as plt
import pandas as pd
import mplfinance as mpf

# moving averages uses pandas dataframe's methods; rolling and ewm. They return the copy of Pandas Dataframe
# so the original data doesn't get affected.
def SMA(stock: pd.DataFrame, period=5):
    return stock["Close"].rolling(period, 1).mean()

def EMA(stock: pd.DataFrame, period=5):
    return stock["Close"].ewm(com=period).mean()

def MACD(stock: pd.DataFrame):
    # formula according to https://www.investopedia.com/terms/m/macd.asp
    return EMA(stock, 12), EMA(stock, 26)

def RSI(stock: pd.DataFrame, periods=14):
    # https://www.investopedia.com/terms/r/rsi.asp
    # reverse the stock data so the first priority will be the most recent price (because it is the most impactful)
    close = stock.filter(["Close"]).values[::-1]
    gains, losses = 0, 0
    res = {}

    # run the RSI with one iteration(O(n)), it can be calculated with double for-loop
    # but it will be expensive at O(n^2)
    for i in range(len(close)-1):
        if close[i] >= close[i+1]:
            gains +=  close[i+1] - close[i]
        else: losses += close[i] - close[i+1]

        if i % periods == 0 and i != 0:

            averageGain = gains/periods
            averageLoss = losses/periods
            rs = averageGain/averageLoss

            rsi = 100 - (100 / (1 + rs))
            gains, losses = 0, 0

            res.update({stock.iloc[-(i-13)].name : rsi[0]})

    return pd.Series(res, name="RSI")[::-1]

def OBV(stock: pd.DataFrame):
    # https://www.investopedia.com/terms/o/onbalancevolume.asp
    close = stock.filter(["Close"]).values

    res = {stock.iloc[0].name : 0}
    for idx in range(1, len(close)):
        diff = 0
        if close[idx] > close[idx-1]:
            diff = stock["Volume"][idx]
        elif close[idx] < close[idx-1]:
            diff = -stock["Volume"][idx]

        obv = res[list(res)[-1]] + diff
        res.update({stock.iloc[idx].name : obv})

    return pd.Series(res, name="OBV")
        
def analyze(stock: pd.DataFrame):
    # SMA Crossover analysis
    SMA0, SMA1 = SMA(stock, period=200), SMA(stock, period=50)
    yield ("SMA Cross", None) if round(SMA0[-1]) == round(SMA1[-1]) else ("SMA Cross", SMA1[-1] > SMA0[-1])

    # MACD analysis
    MACD0, MACD1 = MACD(stock)
    yield ("MACD", None) if round(MACD0[-1]) == round(MACD1[-1]) else ("MACD", round(MACD0[-1]) > round(MACD1[-1]))

    # RSI analysis
    rsi = RSI(stock)
    yield ("RSI", None) if round(rsi[-1]) in range(30, 70) else ("RSI", round(rsi[-1]) < 30)

    # CandleStick window analysis
    # candle stick window means the next open price is higher/lower than the previous high price, indicating continuation
    yield ("Window", None) if stock["Open"][-1] - stock["High"][-2] < -.2 else ("Window",  stock["Open"][-1] - stock["High"][-2] < -.5)

def candleStick(stock: pd.DataFrame, inputs=list[tuple((pd.DataFrame, str))]):
    mpf.figure(figsize=(16, 8))

    ax = plt.subplot()
    stock.index.name = "Date"
    mpf.plot(stock, type="candle", ax=ax, returnfig=True)
    
    if inputs:
        for input in inputs:
            copy = stock.copy()
            copy["Close"] = input[0]
            mpf.plot(copy, type="line", ax=ax)
            
    mpf.show()