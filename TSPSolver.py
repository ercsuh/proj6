#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import math
import operator
import pandas as pd
import copy


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		bssf = None
		foundTour = False
		count = 0
		start_time = time.time()
		for city in cities:
			currentCity = city
			route = [currentCity]
			for i in range(ncities - 1):
				currentCity = self.findMinEdge(route, currentCity)
				if currentCity is None:
					break
				route.append(currentCity)
			if currentCity is not None:
				bssf = TSPSolution(route)
				count += 1
				if bssf.cost < np.inf:
					foundTour = True
					break
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	def findMinEdge(self, route, city):
		cities = self._scenario.getCities()
		minCost = np.inf
		minOtherCity = None
		cost = np.inf
		for otherCity in cities:
			if otherCity in route:
				continue
			cost = city.costTo(otherCity)
			if cost < minCost:
				minCost = cost
				minOtherCity = otherCity
		return minOtherCity
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		pass

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		results = {}

		start_time = time.time()
		bssf = self.geneticAlgorithm(100, 20, 0.01, 500)
		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = None
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	def geneticAlgorithm(self, popSize, eliteSize, mutationRate, generations):
		population = self.initialPopulation(popSize)

		for i in range(0, generations):
			population = self.nextGeneration(population, eliteSize, mutationRate)

		bestRouteIndex = self.rankRoutes(population)[0][0]
		return population[bestRouteIndex]


	def nextGeneration(self, currentGen, eliteSize, mutationRate):
		popRanked = self.rankRoutes(currentGen)
		selectionResults = self.selection(popRanked, eliteSize)
		matingPool = self.matingPool(currentGen, selectionResults)
		children = self.breedPopulation(matingPool, eliteSize)
		nextGeneration = self.mutatePopulation(children, mutationRate)
		return nextGeneration

	def initialPopulation(self, popSize):
		population = []

		for i in range(popSize):
			results = self.defaultRandomTour()
			population.append(results['soln'])
		return population

	def getFitness(self, individual):
		return 1000.0 / float(individual.cost)

	def rankRoutes(self, population):
		fitnessResults = {}
		for i in range(len(population)):
			fitnessResults[i] = self.getFitness(population[i])
		return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

	def selection(self, popRanked, eliteSize):
		selectionResults = []
		# I have doubts about the way they do this. It seems like, despite the complicated
		# stuff they do, that the non-elite routes end up being selected completely randomly
		df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
		df['cum_sum'] = df.Fitness.cumsum()
		df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

		for i in range(0, eliteSize):
			selectionResults.append(popRanked[i][0])
		for i in range(0, len(popRanked) - eliteSize):
			pick = 100 * random.random()
			for i in range(0, len(popRanked)):
				if pick <= df.iat[i, 3]:
					selectionResults.append(popRanked[i][0])
					break
		return selectionResults

	def matingPool(self, population, selectionResults):
		matingPool = []
		for i in range(len(selectionResults)):
			index = selectionResults[i]
			matingPool.append(population[index])
		return matingPool

	def breed(self, parent1, parent2):
		child = []
		childP1 = []
		childP2 = []

		geneA = int(random.random() * len(parent1.route))
		geneB = int(random.random() * len(parent2.route))

		startGene = min(geneA, geneB)
		endGene = max(geneA, geneB)

		for i in range(startGene, endGene):
			childP1.append(parent1.route[i])

		childP2 = [item for item in parent2.route if item not in childP1]

		child = childP1 + childP2
		return TSPSolution(child)

	def breedPopulation(self, matingpool, eliteSize):
		children = []
		length = len(matingpool) - eliteSize
		pool = random.sample(matingpool, len(matingpool))

		for i in range(0, eliteSize):
			children.append(matingpool[i])

		for i in range(0, length):
			child = self.breed(pool[i], pool[len(matingpool) - i - 1])
			children.append(child)
		return children

	def mutate(self, individual, mutationRate):
		for swapped in range(len(individual.route)):
			if (random.random() < mutationRate):
				swapWith = int(random.random() * len(individual.route))

				city1 = individual.route[swapped]
				city2 = individual.route[swapWith]

				individual.route[swapped] = city2
				individual.route[swapped] = city1
				individual.cost = individual._costOfRoute()
		return individual

	def mutatePopulation(self, population, mutationRate):
		mutatedPop = []

		for ind in range(len(population)):
			mutatedInd = self.mutate(population[ind], mutationRate)
			mutatedPop.append(mutatedInd)
		return mutatedPop