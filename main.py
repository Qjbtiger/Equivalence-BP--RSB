from matplotlib.pyplot import flag
import numpy as np
from BinaryPerceptron import *
from QLog import qLog

def writeLists(file, list):
    for l in list:
        for item in l:
            file.write('{} '.format(item))
        file.write('\n')

def taskBP():
    qLog.pushDeer('BP program start!')
    NList = [10000]
    # NList = [400, 800, 1000, 1600, 2000, 4000]
    tryTimes = 2
    thresholdTryTimes = 1
    numPoint = 100
    eps = 0.0001

    alphacLists = []
    for N in NList:
        perceptron = BinaryPerceptron(N)

        # use bisection method derive alphac
        alphacList = []
        for n in range(numPoint):
            leftAlphac, rightAlphac = 0.8, 1.2
            while ((rightAlphac - leftAlphac) >= eps):
                middleAlphac = (leftAlphac + rightAlphac) / 2
                perceptron.generateData(middleAlphac)
                count = 0
                for _ in range(tryTimes):
                    flag, iteration = perceptron.messagePassingGPU()
                    qLog.info('N: {} ({}/{}), alpha: {:.4f}, success: {}, iteration: {}                           '.format(N, n, numPoint, middleAlphac, flag, iteration))

                    if flag:
                        count += 1
                    if count >= thresholdTryTimes:
                        break
                if count >= thresholdTryTimes:
                    leftAlphac = middleAlphac
                else:
                    rightAlphac = middleAlphac
                        
            alphacList.append(middleAlphac)

        qLog.pushDeer('N={} finish!'.format(N))
        with open('./data/BP-{}.dat'.format(N), 'w') as file:
            writeLists(file, [alphacList])
        alphacLists.append(alphacList)
            
    with open('./data/BP.dat', 'w') as file:
        writeLists(file, [NList])
        writeLists(file, alphacLists)

def taskAlphaVSRate():
    qLog.pushDeer('Alpha VS Rate program start!')
    NList = [1000, 2000]
    # NList = [400, 1000, 2000]
    alphaLists = [i/1000 for i in np.arange(850, 1150, 2)]
    rateList = []
    tryTimes = 100

    for N in NList:
        perceptron = BinaryPerceptron(N)

        rate = []
        for alpha in alphaLists:
            count = 0
            for _ in range(tryTimes):
                perceptron.generateData(alpha)
                flag, iteration = perceptron.messagePassingGPU()
                qLog.info('N: {}, alpha: {:.4f}, success: {}, iteration: {}                           '.format(N, alpha, flag, iteration))

                if flag:
                    count += 1
                
            rate.append(count/tryTimes)
        
        with open('./data/alphaVSRate-{}.dat'.format(N), 'w') as file:
            writeLists(file, [rate])
        rateList.append(rate)
        qLog.pushDeer('N={} finish!'.format(N))

    with open('./data/alphaVSRate.dat', 'w') as file:
        writeLists(file, [alphaLists])
        writeLists(file, rateList)

                


def taskRS():
    N = 200

    perceptron = BinaryPerceptron(N)
    perceptron.replicaSymmetric()

def main():
    qLog.set(stream=True, file=True, fileLogName='taskAlphaVSRate', fileMode='w')
    qLog.registerPushDeer('PDU1549TDgJhw6oNVKBSzvBeVGOSJkrRn8ynXfA4')
    # temperary disable pushDeer debug
    # qLog.disablePushDeer()

    # taskBP()
    # taskRS()
    taskAlphaVSRate()

if __name__=='__main__':
    main()