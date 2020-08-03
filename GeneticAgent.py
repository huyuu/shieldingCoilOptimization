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
import os

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
    # if loss already calculated, return
    if coil.loss != None:
        return coil
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
            self.distributionInRealCoordinates = self.calculateDistributionInRealCoordinates()
        else:
            self.distribution = baseCoil.distribution.copy()
            self.distributionInRealCoordinates = self.calculateDistributionInRealCoordinates()
        #
        self.loss = None


    def calculateDistributionInRealCoordinates(self):
        rs = nu.linspace(-self.Z0, self.Z0, self.columnAmount).reshape(1, -1) * self.distribution
        zs = nu.linspace(self.minRadius, self.minRadius+self.rowAmount*self.scThickness, self.rowAmount).reshape(-1, 1) * self.distribution
        indices = [ (r, z) for r, z in zip(rs[rs!=0].ravel(), zs[zs!=0].ravel()) ]
        assert len(rs) == len(zs)
        return indices


    def makeDescendant(self, row, column, shouldIncrease):
        coil = Coil(baseCoil=self)
        if shouldIncrease:
            coil.distribution[row, column] = 1
            coil.distribution[row, -1-column] = 1
        else:
            coil.distribution[row, column] = 0
            coil.distribution[row, -1-column] = 0
        # print(coil.distribution[-2:, :])
        # print(' ')
        return coil


    def makeDescendants(self, amount):
        descendants = []
        count = 0
        amount = amount // 2
        candidates = []
        # set candidates
        if self.columnAmount % 2 == 1:#odd
            candidates = nu.random.permutation((self.columnAmount+1)//2).tolist()
        else:#even
            candidates = nu.random.permutation(self.columnAmount//2).tolist()
        increasedColumns = set()
        # add increased descendants
        while count <= amount and len(candidates) > 0:
            chosenColumn = candidates.pop()
            rows = self.distribution[:, chosenColumn]
            if rows[0] == 1:# can't be increased
                continue
            else:# can be increased
                row = nu.where(rows==0)[0][-1]
                descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=True))
                increasedColumns.add(chosenColumn)
                count += 1
        # add decreased descendants
        count = 0
        if self.columnAmount % 2 == 1:#odd
            candidates = nu.random.permutation(list(set(nu.arange((self.columnAmount+1)//2).tolist()) - increasedColumns)).tolist()
        else:#even
            candidates = nu.random.permutation(list(set(nu.arange(self.columnAmount//2).tolist()) - increasedColumns)).tolist()
        decreasedColumns = set()
        while count <= amount and len(candidates) > 0:
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
        self.survivalPerGeneration = 20
        self.descendantsPerLife = 8
        # init generation
        coil = Coil()
        if os.path.exists('bestCoil.npy'):
            distribution = nu.load('bestCoil.npy')
            coil.distribution = distribution
            coil.distributionInRealCoordinates = coil.calculateDistributionInRealCoordinates()
        for _ in range(self.survivalPerGeneration):
            self.generation.append(coil)


    # http://ja.pymotw.com/2/multiprocessing/communication.html
    # https://qiita.com/uesseu/items/791d918c5a076a5b7265#ネットワーク越しの並列化
    def run(self, loopAmount=100):
        minLoss = []
        loopCount = []
        for count in range(loopAmount):
            _start = dt.datetime.now()
            # calculate loss function for this generation and store in self.generationQueue
            # https://github.com/psf/black/issues/564
            with mp.Pool(processes=min(mp.cpu_count()-1, 60)) as pool:
                self.generation = pool.map(lossFunction, self.generation)
            print('loss function calculated.')
            # boom next generation
            survived = sorted(self.generation, key=lambda coil: coil.loss)[:self.survivalPerGeneration]
            self.generation = []
            for life in survived:
                descendants = life.makeDescendants(amount=self.descendantsPerLife)
                self.generation.append(life)
                self.generation.extend(descendants)
            print('next generation made.')
            # check if should end
            _end = dt.datetime.now()
            print('minLoss: {:.4g} (time cost: {:.3g}[min])'.format(survived[0].loss, (_end-_start).total_seconds()/60))
            # plot
            minLoss.append(survived[0].loss)
            loopCount.append(count+1)
            fig = pl.figure()
            pl.title('Training Result', fontsize=22)
            pl.xlabel('loop count', fontsize=18)
            pl.ylabel('min loss', fontsize=18)
            pl.yscale('log')
            pl.plot(loopCount, minLoss)
            pl.tick_params(labelsize=12)
            fig.savefig('trainingResult.png')
            pl.close(fig)
            # save coil, https://deepage.net/features/numpy-loadsave.html
            nu.save('bestCoil.npy', survived[0].distribution)


# Main

# Ouer Coil
r1 = 6.1e-2
N1 = 27
l1 = 17.8e-2  # 20cm
I1 = 1
R2 = 1e-7


if __name__ == '__main__':
    mp.freeze_support()
    agent = GeneticAgent()
    agent.run()
