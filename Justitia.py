import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import linear_model



class Trade():
    def __init__(self):
        self.entryTime = 0
        self.entryTimeStamp = 0
        self.entryPrice = 0
        self.exitTimeStamp = 0
        self.exitPrice = 0
        self.position = 0

    def getProfit(self):
        if self.exitTime != 0:
            return self.position * (self.exitPrice - self.entryPrice)
        else:
            return 0






class Justitia():
    def __init__(self):
        self.allTrades = ""
        self.trainTrades = []
        self.trainProfits = []
        self.trainTimeStamps = []
        self.testTrades = []
        self.testProfits = []
        self.testTimeStamps = []
        self.statistics = ""
        self.textXY = []

        mpl.style.use("seaborn")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Train/Test plot')

    def parseMCReport(self, reportFile):
        listOfTrade = pd.read_excel(io=path, sheetname='List of Trades', skiprows=2)
        trades = []
        for index, row in listOfTrade.iterrows():
            tradeTime = pd.Timestamp.combine(row['Date'], row['Time'])
            tradeTimeStamp = tradeTime.timestamp()
            tradeType = row['Type']
            if 'Entry' in tradeType:
                # New trade
                trade = Trade()
                trades.append(trade)
                trade.entryTime = tradeTime
                trade.entryTimeStamp = tradeTimeStamp
                trade.entryPrice = row['Price']
                if 'Long' in tradeType:
                    trade.position = 1 * int(row['Contracts'])
                else:
                    trade.position = -1 * int(row['Contracts'])
            else:
                # Closing trade
                trade.exitTime = tradeTime
                trade.exitTimeStamp = tradeTimeStamp
                trade.exitPrice = row['Price']
        self.allTrades = trades

    # This function splits the report by time into train and test
    def splitTrades(self, timeStamp):
        for trade in self.allTrades:
            if trade.exitTimeStamp < timeStamp:
                self.trainTrades.append(trade)
            else:
                self.testTrades.append(trade)
        self.trainProfits = self.getProfits(self.trainTrades)
        self.trainAccount = self.cumsum(self.trainProfits)
        self.trainTimeStamps = self.getTimeStamps(self.trainTrades)
        self.testProfits = self.getProfits(self.testTrades)
        self.testAccount = self.cumsum(self.testProfits, self.trainAccount[-1])
        self.testTimeStamps = self.getTimeStamps(self.testTrades)

    def getProfits(self, trades):
        profits = []
        for trade in trades:
            profits.append(trade.getProfit())
        return profits

    def getTimeStamps(self, trades):
        timeStamp = []
        for trade in trades:
            timeStamp.append(trade.exitTimeStamp)
        return timeStamp

    def cumsum(self, nums, offset=0):
        ret = [offset]
        for num in nums:
            ret.append(num + ret[-1])
        return ret[1:]

    def plotAllTrade(self):
        profits = self.cumsum(self.getProfits(self.allTrades))
        timeStamps = self.getTimeStamps(self.allTrades)

        plt.plot(timeStamps, profits)

    def plotTrainTestTrades(self):
        if self.trainTrades == []:
            print("No training data found, use splitTrades() first")
            return


        plt.plot(self.trainTimeStamps, self.trainAccount)
        plt.plot(self.testTimeStamps, self.testAccount)


    def linearAnalysis(self):
        if self.trainTrades == []:
            print("No training data found, use splitTrades() first")
            return
        from sklearn import linear_model
        from sklearn import datasets, linear_model
        from sklearn.metrics import mean_squared_error, r2_score

        trainTimeStamps = pd.np.array(self.trainTimeStamps).reshape((-1, 1))
        testTimeStamps = pd.np.array(self.testTimeStamps).reshape((-1, 1))

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the train data
        regr.fit(trainTimeStamps, self.trainAccount)
        trainCoef = float(regr.coef_[0]) * 100000
        # Make predictions using the testing set
        trainPredict = regr.predict(trainTimeStamps)

        # Train the test data
        regr.fit(testTimeStamps, self.testAccount)
        testCoef = float(regr.coef_[0]) * 100000
        # Make predictions using the testing set
        testPredict = regr.predict(testTimeStamps)

        trainMSE = mean_squared_error(self.trainAccount, trainPredict)
        testMSE = mean_squared_error(self.testAccount, testPredict)
        trainR2 = r2_score(self.trainAccount, trainPredict)
        testR2 = r2_score(self.testAccount, testPredict)

        plt.plot(trainTimeStamps, trainPredict)
        plt.plot(testTimeStamps, testPredict)

        self.statistics += "Mean square error: train: {:.4f}  test: {:.4f} \n".format(trainMSE, testMSE)
        self.statistics += "Variance score: train: {:.6f}  test: {:.6f} \n".format(trainR2, testR2)
        self.statistics += "Slope: train: {:.6f}  test: {:.6f} \n".format(trainCoef, testCoef)

        plt.text(self.trainTimeStamps[1], self.testAccount[0], self.statistics, fontsize=12)

    def savePlot(self, path):
        plt.show()
        pass

    def getBeta(self, data):
        pass

    def getSharp(self, data):
        pass

    def pfAnalysis(self):
        trainPF = self.getPF(self.trainProfits)
        testPF = self.getPF(self.testProfits)
        print(trainPF)
        print(testPF)

    def getPF(self, data):
        grossProfit = 0
        grossLost = 0
        for profit in data:
            if profit > 0:
                grossProfit += profit
            else:
                grossLost += profit
        grossLost = grossLost * (-1)
        return grossProfit/grossLost

    def getWinningRate(self, data):
        pass

    def getExpectation(self, data):
        pass

if __name__ == "__main__":
    path = r'./reports/TXF1  MH_Bollinger Back-Testing Strategy Performance Report.xlsx'
    path = r'./reports/TXF1  MH_VOLEX Back-Testing Strategy Performance Report.xlsx'
    splitTime = '2017-01-01'
    justitia = Justitia()
    justitia.parseMCReport(path)

    justitia.splitTrades(pd.to_datetime(splitTime).timestamp())
    #justitia.plotTrainTestTrades()
    #justitia.linearAnalysis()
    justitia.pfAnalysis()
    justitia.savePlot("")
