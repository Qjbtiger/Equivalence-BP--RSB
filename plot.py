from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

def plotBP():
    alphacLists = []
    with open('./data/BP.dat', 'r') as file:
        NList = file.readline().strip().split(' ')
        for _ in range(len(NList)):
            alphacLists.append(file.readline().strip().split(' '))

    NList = [int(it) for it in NList]
    alphacLists = np.asarray([np.asarray([float(it) for it in alphacList]) for alphacList in alphacLists])
    alphacMeanList = np.mean(alphacLists, axis=1)
    alphacStdList = np.std(alphacLists, axis=1)

    plt.errorbar(NList, alphacMeanList, yerr=alphacStdList, fmt='.-', markersize=10.0, capsize=2.0, label='BP')
    plt.hlines(y=1.0138, xmin=0, xmax=NList[-1], linestyles='--', colors='r', label='RS')
    plt.hlines(y=1.015, xmin=0, xmax=NList[-1], linestyles='--', colors='grey', label='previous result')
    plt.xlabel('$N$')
    plt.ylabel('$\\alpha_c$')
    plt.legend()
    plt.savefig('./image/BP.png', dpi=300)
    plt.show()
    plt.close()

def plotAlph():
    with open('./data/alphaVSRate.dat', 'r') as file:
        alphaLists = file.readline().strip().split(' ')
        N400List = file.readline().strip().split(' ')
        N1000Lists = file.readline().strip().split(' ')
        N2000Lists = file.readline().strip().split(' ')

    alphaLists = [float(it) for it in alphaLists]
    N400List = [float(it) for it in N400List]
    N1000Lists = [float(it) for it in N1000Lists]
    N2000Lists = [float(it) for it in N2000Lists]
    plt.plot(alphaLists, N400List, label='N-400')
    plt.plot(alphaLists, N1000Lists, label='N-1000')
    plt.plot(alphaLists, N2000Lists, label='N-2000')
    plt.legend()
    plt.savefig('./image/alphaVSRate.png', dpi=300)
    plt.show()
    plt.close()


if __name__=='__main__':
    # plotBP()
    plotAlph()