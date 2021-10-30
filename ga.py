import numpy as np
from itertools import zip_longest
import random
import matplotlib.pyplot as plt
import copy



class KnickKnack:
    def __init__(self, value=None, weight=None):
        if value is None:
            self.value = np.random.randint(1, 100)
        else:
            self.value = value
        if value is None:
            self.weight = np.random.randint(1, 10)
        else:
            self.weight = weight

    def get_value(self):
        return self.value

    def get_weight(self):
        return self.weight


class Individual:
    # Unique index for every Individual
    ID = 0

    def __init__(self, knickKnacksList, maxWeight, phenotype=None):
        # pointer to list of knick knacks
        self.knickKnacksList = knickKnacksList

        # phenotype length
        self.length = len(self.knickKnacksList)

        # border value of weight
        self.maxWeight = maxWeight

        # phenotype determine which knick knacks are carried by this individual
        if phenotype is None:                               # for first generation
            self.phenotype = []
            for i in range(self.length):
                self.phenotype.append(random.randint(0, 1))
        else:                                               # for offspring
            self.phenotype = phenotype

        # value of adaptation function
        self.totalValue = 0

        # sum of weights of all knick knacks
        self.totalWeight = 0

        self.id = Individual.ID
        Individual.ID = Individual.ID + 1

        self.computeAdaptation()

    def get_phenotype(self):
        return self.phenotype

    def computeAdaptation(self):
        while(1):
            self.totalWeight = 0
            for i in range(self.length):
                # weighted sum of carried knick knacks weights
                #print(self.knickKnacksList[i].get_weight())
                self.totalWeight += self.phenotype[i] * self.knickKnacksList[i].get_weight()

            #print(self.totalWeight,self.maxWeight)
            if self.totalWeight <= self.maxWeight:
                break
            else:
                # list of indexes of carried knick knacks
                owned = []
                for i in range(len(self.phenotype)):
                    if self.phenotype[i] == 1:
                        owned.append(i)
                index = random.randint(0, len(owned)-1)
                #print(len(self.phenotype))
                self.phenotype[owned[index]] = 0

        self.totalValue = 0
        for i in range(self.length):
            # weighted sum of carried knick knacks values
            self.totalValue += self.knickKnacksList[i].get_value() * self.phenotype[i]

        return self.totalValue


    def crossover(self, phenotype2):
        offspringPhenotype = self.phenotype.copy()
        offspringPhenotype[0:int(len(self.phenotype)/3)] = phenotype2[0:int(len(self.phenotype)/3)]

        return Individual(self.knickKnacksList, self.maxWeight, offspringPhenotype)

    def mutation(self):
        mutationProbability = 1/len(self.phenotype)
        for i in range(len(self.phenotype)):
            mutate = random.randint(0, 1000) / 1000
            if mutate < mutationProbability:
                self.phenotype[i] = (1 - self.phenotype[i])

    def printData(self):
        print("Individual with id: ", self.id, " and phenotype: ", self.phenotype, "\n")

    def printFullData(self):
        print("Individual with id: ", self.id, " and total weight: ", self.totalValue, "\n")

    def printPhenotype(self):
        for i in self.phenotype:
            print(i)
        print('\n')



def createPopulation(knickKnacksList, maxWeight, populationSize):
    population = []
    for i in range(populationSize):
        population.append(Individual(knickKnacksList, maxWeight))
    return population


def tournamentSelection(population):
    parentPopulation = []
    populationCopy = population.copy()

    for i in range(len(populationCopy)):
        if len(populationCopy) > 1:
            r = np.arange(0, len(populationCopy), 1)
            random.shuffle(r)
            competitors = []
            for j in r:
                competitors.append(populationCopy[j])
            parentPopulation.append(findTheBestIndividual(competitors))
            #populationCopy.remove(parentPopulation[-1])
        else:
            parentPopulation.append(parentPopulation[-1])
            #populationCopy.clear()

    return parentPopulation

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def crossover(parentPopulation):
    np.random.shuffle(parentPopulation)
    pairs = grouper(2, parentPopulation)
    #pairs = parentPopulation.shuffle.each_slice(2).to_a
    offSpringPopulation = []

    copyOrOffspringProbability = 30
    # create two offSprings for each pair
    for pair in pairs:
        copyOrOffspring = np.random.randint(0, 100)
        if copyOrOffspring < copyOrOffspringProbability:
            offSpringPopulation.append(pair[0])
            offSpringPopulation.append(pair[1])
        else:
            offSpringPopulation.append(pair[0].crossover(pair[1].get_phenotype()))
            offSpringPopulation.append(pair[1].crossover(pair[0].get_phenotype()))

    return offSpringPopulation


def mutation(offSpringPopulation):
    for individual in offSpringPopulation:
        individual.mutation()


def findTheBestIndividual(population):
    theBestIndividual = population[0]
    for i in range(1, len(population)):
        if theBestIndividual.totalValue < population[i].totalValue:
            theBestIndividual = population[i]
    return theBestIndividual


def theBestResult(theBestGlobals, theBestLocal, avgLocalList, iteration, show=False):
    maxValuesGlobalHistory = []
    maxValuesLocalHistory = []
    weightsHistory = []
    maxWeight = []
    for individual in theBestGlobals:
        maxValuesGlobalHistory.append(individual.totalValue)
        weightsHistory.append(individual.totalWeight)
        maxWeight.append(individual.maxWeight)

    for individual in theBestLocal:
        maxValuesLocalHistory.append(individual.totalValue)

    theBestIndividual = theBestGlobals[-1]  # findTheBestIndividual(theBestGlobals, True)
    print("max value: ", theBestIndividual.totalValue)
    print("total weight: ", theBestIndividual.totalWeight)
    print("knick knacks: ", theBestIndividual.phenotype)

    if show is True:
        # Create two subplots sharing y axis
        # fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

        plt.figure(1)
        plt.plot(list(range(iteration)), maxValuesLocalHistory, 'g-')
        plt.title('history of local max values')
        plt.ylabel('local max value')
        plt.xlabel('iteration')
        plt.grid()
        plt.ylim(min(maxValuesLocalHistory)*0.98, max(maxValuesLocalHistory)*1.02)


        plt.figure(2)
        plt.plot(list(range(iteration)), maxValuesGlobalHistory, 'b-')
        plt.title('historia łącznych wartości przedmiotów')
        plt.ylabel('łączna wartość przedmiotów')
        plt.xlabel('iteracja')
        plt.grid()
        plt.ylim(min(maxValuesLocalHistory)*0.98, max(maxValuesGlobalHistory)*1.02)


        plt.figure(3)
        plt.plot(list(range(iteration)), maxWeight, 'k-')
        plt.plot(list(range(iteration)), weightsHistory, 'r-')
        plt.title('historia wag')
        plt.ylabel('waga')
        plt.xlabel('iteracja')
        plt.grid()

        plt.figure(4)
        plt.plot(list(range(iteration)), avgLocalList, 'm-')
        plt.title('history of average local value')
        plt.ylabel('average value')
        plt.xlabel('iteration')
        plt.grid()
        plt.show()

    #input()
    return theBestIndividual.phenotype


def avgLocal(population):
    avg = 0
    for individual in population:
        avg += individual.totalValue

    return avg/len(population)


def searchTheBest(population, theBestIndividuals, avgLocalList):
    try:
        # winner = findTheBestIndividual(population.copy())
        winnerLocal = findTheBestIndividual(population.copy())
        winnerGlobal = findTheBestIndividual([winnerLocal] + [theBestIndividuals[-1]])
    except:
        winnerGlobal = findTheBestIndividual(population.copy())
        winnerLocal = winnerGlobal

    avgLocalList.append(avgLocal(population))
    return copy.copy(winnerGlobal), copy.copy(winnerLocal), avgLocalList


def geneticAlgorithm(knickKnacksList, maxWeight, populationSize, repeats=20):
    population = createPopulation(knickKnacksList, maxWeight, populationSize)
    iteration = 0
    algorithm = True

    theBestGlobals = []
    theBestLocals = []
    avgLocalList = []

    while algorithm:
        print("iteration: ", iteration)
        parentPopulation = tournamentSelection(population)
        offSpringPopulation = crossover(parentPopulation)
        mutation(offSpringPopulation)
        population = offSpringPopulation

        for individual in population:
            individual.computeAdaptation()

        theBestGlobal, theBestLocal, avgLocalList = searchTheBest(population, theBestGlobals, avgLocalList)
        theBestGlobals.append(theBestGlobal)
        theBestLocals.append(theBestLocal)

        iteration += 1
        if iteration == repeats:
            break

    return theBestResult(theBestGlobals, theBestLocals, avgLocalList, iteration, True)

def printKnickKnacksList(knickKnacksList):
    # print weights and values of each knick knack
    print("Knick knacks")
    print("number\tweight\tvalue")
    for i in range(len(knickKnacksList)):
        print(i, "\t", knickKnacksList[i].get_weight(), "\t", knickKnacksList[i].get_value())






def test():
    for i in range(7, 9):

        # numberOfSet = str(3)
        numberOfSet = str(i)
        linesC = [line.rstrip('\n') for line in open('tests\P0'+numberOfSet+'\p0'+numberOfSet+'_c.txt')]
        B = int(linesC[0])

        linesW = [line.rstrip('\n') for line in open('tests\P0'+numberOfSet+'\p0'+numberOfSet+'_w.txt')]
        linesP = [line.rstrip('\n') for line in open('tests\P0'+numberOfSet+'\p0'+numberOfSet+'_p.txt')]

        N = len(linesP)

        knickKnacksList = []
        for i in range(N):
            knickKnacksList.append(KnickKnack(int(linesP[i]), int(linesW[i])))

        linesS = [line.rstrip('\n') for line in open('tests\P0'+numberOfSet+'\p0'+numberOfSet+'_s.txt')]
        correctOutput = []
        for i in range(N):
            correctOutput.append(int(linesS[i]))

        # Oxxx<=>xxx<=>xxx<=>xxx<=>xxx<=>xxxO #
        #         editable parameters         #
        # Oxxx<=>xxx<=>xxx<=>xxx<=>xxx<=>xxxO #

        # N e {100, 250, 500}
        # N = 100  # amount of knick knacks
        # M = 300  # population
        # B = 200  # capacity of bag
        M = 50
        repeatings = N*50



        # printKnickKnacksList(knickKnacksList)

        result = geneticAlgorithm(knickKnacksList, B, M, repeatings)

        print(result)
        print(correctOutput)
        if result == correctOutput:
            print("TEST " + numberOfSet + " PASSED")
        else:
            print("TEST " + numberOfSet + " FAILED")


def myAlgorithm():
    # Oxxx<=>xxx<=>xxx<=>xxx<=>xxx<=>xxxO #
    #         editable parameters         #
    # Oxxx<=>xxx<=>xxx<=>xxx<=>xxx<=>xxxO #

    # N e {100, 250, 500}
    N = 300  # amount of knick knacks
    M = 300  # population
    B = 200  # capacity of bag
    repeatings = 3000

    knickKnacksList = []
    for i in range(N):
        knickKnacksList.append(KnickKnack())

    # printKnickKnacksList(knickKnacksList)

    result = geneticAlgorithm(knickKnacksList, B, M, repeatings)

test()
# myAlgorithm()