import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import linear_model
import os


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
    def __init__(self, name):
        self.allTrades = ""
        self.trainTrades = []
        self.trainProfits = []
        self.trainTimeStamps = []
        self.testTrades = []
        self.testProfits = []
        self.testTimeStamps = []
        self.statistics = []
        self.textXY = []
        self.name = name

        mpl.style.use("seaborn")
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title(self.name)

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

        self.statistics.append(["Mean square error", trainMSE, testMSE])
        self.statistics.append(["Variance", trainR2, testR2])
        self.statistics.append(["Slope", trainCoef, testCoef])

    def plotStatistics(self):
        info = ""
        for stat in self.statistics:
            info += stat[0] + "   train:" + str(stat[1]) + " test: " + str(stat[2]) + "\n"
        plt.text(self.trainTimeStamps[1], self.testAccount[0], info, fontsize=12)

    def savePlot(self):
        #plt.show()
        plt.savefig("../out/" + self.name + ".png", dpi=600, format="png")

    def getBeta(self, data):
        pass

    # Input the risk free return
    def sharpAnalysis(self, Rf):
        trainSharp = self.getSharp(self.trainProfits, Rf)
        testSharp = self.getSharp(self.testProfits, Rf)
        self.statistics.append(["Sharp", trainSharp, testSharp])

    def getSharp(self, data, Rf):
        # [E(Rp)－Rf]/σp
        return 0

    def pfAnalysis(self):
        trainPF = self.getPF(self.trainProfits)
        testPF = self.getPF(self.testProfits)
        self.statistics.append(["PF", trainPF, testPF])

    def getPF(self, data):
        grossProfit = 0
        grossLost = 0
        for profit in data:
            if profit > 0:
                grossProfit += profit
            else:
                grossLost += profit
        grossLost = grossLost * (-1)
        if grossLost == 0:
          return float("inf")
        return grossProfit/grossLost

    def winningRateAnalysis(self):
        trainWiningRate = self.getWinningRate(self.trainProfits)
        testWiningRate = self.getWinningRate(self.testProfits)
        self.statistics.append(["Winning Rate", trainWiningRate, testWiningRate])

    def getWinningRate(self, data):
        if len(data) == 0:
          return float("inf")
        winTrade = 0.0
        for profit in data:
            if profit > 0:
                winTrade += 1.0

        return winTrade/len(data)

    def expectationAnalysis(self):
        trainE = self.getExpectation(self.trainProfits)
        testE = self.getExpectation(self.testProfits)
        self.statistics.append(["Expectation", trainE, testE])

    def netProfitAnalysis(self, bigPointValue):
        trainP = self.getNetProfit(self.trainProfits, bigPointValue)
        testP = self.getNetProfit(self.testProfits, bigPointValue)
        self.statistics.append(["netProfit", trainP, testP])

    def getExpectation(self, data):
        if len(data) == 0:
          return float("inf")
        netProfit = self.cumsum(data)[-1]
        return netProfit / len(data)

    def MDDAnalysis(self):
        trainMDD = self.getMDD(self.trainAccount)
        testMDD = self.getMDD(self.testAccount)
        self.statistics.append(["MDD", trainMDD, testMDD])

    def getMDD(self, data):
        recordHeigh = float('-inf')
        MDD = float('-inf')
        for balance in data:
            recordHeigh = max(recordHeigh, balance)
            MDD = max(MDD, recordHeigh - balance)
        return MDD

    def getNetProfit(self, data, bigPointValue, fee=0):
        ticks = sum(data) - fee * len(data)
        grossProfit = ticks*bigPointValue
        return grossProfit - fee*bigPointValue


if __name__ == "__main__":
    splitTime = '2019-04-01'

    os.chdir('./report')
    paths = os.listdir('.')

    result = []
    profit = []
    pathCount = 1
    for path in paths:
        print("Working on: " + str(pathCount) + "/" + str(len(paths)) + " " + path)
        pathCount += 1
        name = path.split(" ")[2]

        justitia = Justitia(name)
        justitia.parseMCReport(path)
        justitia.splitTrades(pd.to_datetime(splitTime).timestamp())

        justitia.linearAnalysis()
        justitia.pfAnalysis()
        justitia.winningRateAnalysis()
        #justitia.sharpAnalysis(0.02)
        justitia.expectationAnalysis()
        justitia.MDDAnalysis()
        justitia.netProfitAnalysis(50)

        justitia.plotTrainTestTrades()
        justitia.plotStatistics()
        justitia.savePlot()
        result.append([name, justitia.statistics])

    # Dump in CVS format
    with open('../out/reports.csv', 'w+') as fd:
        # Write columns
        index = 'name, '
        for columnName in result[0][1]:
          index += columnName[0] + " train," + columnName[0] + " test, "
        index += '\n'
        fd.write(index)

        for name, stats in result:
            statString = name + ', '
            for stat in stats:
                statString = statString + str(stat[1]) + ", " + str(stat[2]) + ", "
            statString += '\n'
            fd.write(statString)





    '''
    with open('../out/reports.txt','w+') as fd:
        for stat in statistics:
            info = stat[0] + "\n"
            for data in stat[1]:
                info += '\t' + str(data[0])
                info += '\t\t' + str(data[1])
                info += '\t\t' + str(data[2])
                info += '\n'
            fd.write(info)
      '''
