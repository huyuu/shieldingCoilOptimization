import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
from scipy.integrate import quadrature
from numpy import cos, sin, pi, abs, sqrt
import multiprocessing as mp
import math as ma
import datetime as dt

from calculateML import MutalInductance


# Constants

mu0 = 4*nu.pi*1e-7


# Model

def _fBz(phi, r, x, y, z, _z):
    return ( -r*sin(phi)*(y-r*sin(phi)) - r*cos(phi)*(x-r*cos(phi)) ) / ( (x-r*cos(phi))**2 + (y-r*sin(phi))**2 + (z-_z)**2 )**1.5


def Cz(r, x, y, z, coilZPosition):
    return mu0/(4*pi) * quadrature(_fBz, 0, 2*pi, args=(r, x, y, z, coilZPosition), tol=1e-12, maxiter=100000)[0]


def calculateBzFromCoil(r, l, N, I, x, y, z):
    coilZPositions = nu.linspace(-l/2, l/2, N)
    return sum((Cz(r, x, y, z, coilZPosition) for coilZPosition in coilZPositions)) * I


def calculateBzFromLoop(r, coilZPosition, I, x, y, z):
    return Cz(r, x, y, z, coilZPosition) * I


def lossFunction(coil, points=20):
    # get L2
    L2 = coil.calculateL()
    # get M
    M = 0
    for r2, z2 in coil.distributionInRealCoordinates:
        for z1 in nu.linspace(-l1/2, l1/2, N1):
            M += MutalInductance(r1, r2, d=abs(z2-z1))
    # get a, b at specific position
    loss = 0
    zs = nu.linspace(0, coil.Z0, points)
    for zIndex, z in enumerate(zs):
        a = calculateBzFromCoil(r1, l1, N1, I1, 0, 0, z)
        b = sum( (calculateBzFromLoop(r2, z2, I1, 0, 0, z) for r2, z2 in coil.distributionInRealCoordinates) )
        loss += (a - b/sqrt(1+(R2/L2)**2)*M/L2)**2
    print(coil.distribution[-2:, :])
    print(f'L2: {L2}, M: {M}, loss: {loss}')
    assert loss >= 0
    # add to generationQueue
    coil.loss = loss
    return coil
    # queue.put(coil)


class Coil():
    def __init__(self, baseCoil=None):
        self.length = 10e-2
        self.Z0 = self.length/2
        self.minRadius = 1.5e-2
        self.scWidth = 4e-3
        self.scThickness = 100e-6
        self.columnAmount = int(self.length/self.scWidth)
        self.rowAmount = 5  # max turns
        #
        if baseCoil == None:
            self.distribution = nu.zeros((self.rowAmount, self.columnAmount), dtype=nu.int)
            self.distribution[-1, :] = 1
            #
            self.distributionInRealCoordinates = self.__calculateDistributionInRealCoordinates()
        else:
            self.distribution = baseCoil.distribution.copy()
            self.distributionInRealCoordinates = self.__calculateDistributionInRealCoordinates()
        #
        self.loss = None


    # @property
    # def distributionInRealCoordinates(self):
    #     rs = nu.linspace(-self.Z0, self.Z0, self.columnAmount).reshape(1, -1) * self.distribution
    #     zs = nu.linspace(self.minRadius, self.minRadius+self.rowAmount*self.scThickness, self.rowAmount).reshape(-1, 1) * self.distribution
    #     indices = [ (r, z) for r, z in zip(rs.ravel(), zs.ravel()) ]
    #     return indices
    def __calculateDistributionInRealCoordinates(self):
        rs = nu.linspace(-self.Z0, self.Z0, self.columnAmount).reshape(1, -1) * self.distribution
        zs = nu.linspace(self.minRadius, self.minRadius+self.rowAmount*self.scThickness, self.rowAmount).reshape(-1, 1) * self.distribution
        indices = [ (r, z) for r, z in zip(rs[rs!=0].ravel(), zs[zs!=0].ravel()) ]
        assert len(rs) == len(zs)
        return indices


    def makeDescendant(self, row, column, shouldIncrease):
        coil = Coil(baseCoil=self)
        if shouldIncrease:
            coil.distribution[row-1, column] = 1
        else:
            coil.distribution[row, column] = 0
        # print(coil.distribution[-2:, :])
        # print(' ')
        return coil


    def makeDescendants(self, amount):
        descendants = []
        count = 0
        candidates = nu.random.permutation(self.columnAmount).tolist()
        increasedColumns = set()
        # add increased descendants
        while count <= amount//2 and len(candidates) > 0:
            chosenColumn = candidates.pop()
            rows = self.distribution[:, chosenColumn]
            if rows[0] == 1:# can't be increased
                continue
            else:# can be increased
                row = nu.where(rows==1)[0][0]
                descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=True))
                increasedColumns.add(chosenColumn)
                count += 1
        # add decreased descendants
        count = 0
        candidates = nu.random.permutation(list(set(nu.arange(self.columnAmount).tolist()) - increasedColumns)).tolist()
        decreasedColumns = set()
        while count <= amount//2 and len(candidates) > 0:
            chosenColumn = candidates.pop()
            rows = self.distribution[:, chosenColumn]
            if rows[-1] == 0:# can't be decreased
                continue
            else:# can be decreased
                row = nu.where(rows==1)[0][0]
                descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=False))
                decreasedColumns.add(chosenColumn)
                count += 1

        return descendants


    # def __lt__(self, other):
    #     assert self.loss != None
    #     assert other.loss != None
    #     return self.loss < other.loss
    #
    #
    # def __gt__(self, other):
    #     assert self.loss != None
    #     assert other.loss != None
    #     return self.loss > other.loss


    def calculateL(self):
        # get Ms between all turns
        Ms = nu.zeros((len(self.distributionInRealCoordinates), len(self.distributionInRealCoordinates)))
        for i, (r, z) in enumerate(self.distributionInRealCoordinates):
            for j in range(i, len(self.distributionInRealCoordinates)):
                r_, z_ = self.distributionInRealCoordinates[j]
                Ms[i, j] = MutalInductance(r_, r, d=abs(z-z_+1e-8))
        Ms += nu.triu(Ms, k=1).T
        return Ms.sum()


class GeneticAgent():
    def __init__(self):
        self.generation = []
        self.survivalPerGeneration = 10
        self.descendantsPerLife = 10
        # init generation
        coil = Coil()
        for _ in range(self.survivalPerGeneration):
            self.generation.append(coil)


    # http://ja.pymotw.com/2/multiprocessing/communication.html
    def run(self, loopAmount=100):
        for _ in range(loopAmount):
            _start = dt.datetime.now()
            # calculate loss function for this generation and store in self.generationQueue
            self.__calculateLossFunctionInCurrentGeneration()
            print('loss function calculated.')
            # boom next generation
            self.__breedNextGeneration()
            print('next generation made.')
            # check if should end
            _end = dt.datetime.now()
            print('(time cost: {:.3g}[min])'.format((_end-_start).total_seconds()/60))


    def __calculateLossFunctionInCurrentGeneration(self):
        print(f'start cal. loss of {len(self.generation)} coils.')
        with mp.Pool() as pool:
            self.generation = pool.map(lossFunction, self.generation)

        # processTank = []
        # generation = []
        # # get generation individuals
        # while not self.generationQueue.empty():
        #     generation.append(self.generationQueue.get())
        #     self.generationQueue.task_done()
        # # calculate loss
        # print(f'start cal. loss of {len(self.generation)} coils.')
        # for life in generation:
        #     # lossFunction(life, self.generationQueue)
        #     process = mp.Process(target=lossFunction, args=(life, self.generationQueue,))
        #     process.start()
        #     processTank.append(process)
        # # wait until finish
        # for process in processTank:
        #     process.join()


    def __breedNextGeneration(self):

        survived = sorted(self.generation, key=lambda coil: coil.loss)[:self.survivalPerGeneration]
        self.generation = []
        for life in survived:
            descendants = life.makeDescendants(amount=self.descendantsPerLife)
            self.generation.append(life)
            self.generation.extend(descendants)
        # print error
        print(f'{len(self.generation)} individuals put.')
        print('loss: {:.4g}'.format(survived[0].loss), end=' ')

        # # get survived
        # generation = []
        # while not self.generationQueue.empty():
        #     generation.append(self.generationQueue.get())
        #     self.generationQueue.task_done()
        # # self.generationQueue.join()
        # print([ coil.loss for coil in generation ])
        # survived = sorted(generation, key=lambda coil: coil.loss)[:self.survivalPerGeneration]
        # # boom the next generation
        # generation = []
        # for life in survived:
        #     descendants = life.makeDescendants(amount=self.descendantsPerLife)
        #     generation.append(life)
        #     generation.extend(descendants)
        # # write to the next generationQueue
        # for life in generation:
        #     self.generationQueue.put(life)
        # # print error
        # print(f'{len(generation)} individuals put.')
        # print('loss: {:.4g}'.format(survived[0].loss), end=' ')


# Main

# Ouer Coil
r1 = 6.1e-2
N1 = 27
l1 = 17.8e-2  # 20cm
I1 = 1
R2 = 1e-7


if __name__ == '__main__':
    agent = GeneticAgent()
    agent.run()
