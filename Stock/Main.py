import GeneralPrediction
import schedule
import StockPrediction
import TechnicalAnalysis
from keras.models import load_model
from time import sleep
import telebot
from threading import Thread

name = "AAPL"

TOKEN = "1718794376:AAEcGyO2aHcMdjPy6Rbz4HbZdMFJpowPHwc"
bot = telebot.TeleBot(TOKEN)
savedStock = None

def telegramMessage(current, predicted):
    profit = predicted - current
    
    message = ("="*18 +
               "\n%s" % (name).center(30) +
               "\n Potential Profit: {:.2f} \n".format(profit) + 
                 " Prediction:       {:.2f} ".format(predicted) +
               "\n Current:           {:.2f} \n".format(current) +
               "="*18)
    
    bot.send_message(-580725848, message)

# message_handlers for user responses
@bot.message_handler(commands=["analyze"])
# response analysis to user after user type /analyze
def analyzeResponse(message, name=name):
    stock = StockPrediction.addStock(name, start="2010-1-1", source='stooq')[-120:]
    stock = stock.iloc[::-1]
    responses = TechnicalAnalysis.analyze(stock)

    msg = "".join([response[0] + " : " + str(response[1]) + "\n" for response in list(responses)])
    bot.reply_to(message, msg)

@bot.message_handler(commands=["today"])
def getToday(message):
    main()

def main():
  stock = StockPrediction.addStock(name, start="2010-1-1", source='stooq')
  stock = stock.iloc[::-1]
  stock = stock[-120:]
  
  model = load_model("Stock/ModelList/{}".format(name))

  prediction = GeneralPrediction.Prediction(stock, model)
  test = StockPrediction.test(model, stock, evaluate=False)
  
  nextDay, data = prediction.nextDay()
  telegramMessage(test[-1], nextDay)

  def visualize():
    SMA = TechnicalAnalysis.SMA(stock)
    EMA = TechnicalAnalysis.EMA(stock)

    # in candlestick form
    TechnicalAnalysis.candleStick(stock=stock, 
                                    inputs=[(SMA, "SMA"),
                                            (EMA, "EMA"),
                                            (stock["Prediction"], "Prediction")])
    # in graph form
    StockPrediction.visualize([(stock["Close"], "Real"),
                            (SMA, "SMA"),
                            (EMA, "EMA"),
                            (stock["Prediction"], "Prediction")])

schedule.every().day.at("22:48").do(main)

def onTime():
  while True:
    schedule.run_pending()
    sleep(60)

Thread(target=bot.polling).start() # read telegram message
Thread(target=onTime).start()