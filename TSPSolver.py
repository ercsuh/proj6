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
import math
from TSPClasses import *


class TSPSolver:
	def __init__( self, gui_view ):
		self.populationSize = 100
		self.eliteSize = 20
		self.mutationRate = 0.02
		self.generations = 5000
		self.population = []
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
		"""
		Implements a genetic algorithm to find a TSP solution.
		"""
		results = {}
		bssf = None
		genCount = 0

		start_time = time.time()
		self.population = self.initialPopulation()

		for i in range(self.generations):
			self.population = self.nextGeneration()
			genCount += 1
			if time.time() - start_time > time_allowance:
				break

		end_time = time.time()

		self.sortPopulation()
		bssf = self.population[0]

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = genCount
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	def initialPopulation(self):
		"""
		Initializes the population using the greedy and random algorithms.
		:return: the population (a list of possible routes)
		"""
		population = []
		greedyResults = self.greedy()
		population.append(greedyResults['soln'])
		for i in range(self.populationSize - 1):
			population.append(self.getRandomRoute())
		return population

	def getRandomRoute(self):
		"""
		Gets a random route faster than the default solver.
		:return: a non-infinite cost random solution
		"""
		cities = self._scenario.getCities()
		route = []
		for startCity in cities:
			route = [startCity]
			for i in range(len(cities) - 1):
				nextCityIndex = random.randint(1, len(cities) - 1)
				firstNextCityIndex = nextCityIndex
				nextCity = cities[nextCityIndex]
				foundNextCity = True
				while True:
					if nextCity not in route and route[i].costTo(nextCity) != np.inf:
						route.append(nextCity)
						break
					else:
						nextCityIndex = (nextCityIndex + 1) % len(cities)
						if nextCityIndex == firstNextCityIndex:
							foundNextCity = False
							break
						nextCity = cities[nextCityIndex]
				if not foundNextCity:
					break
			if len(route) < len(cities):
				continue
			solution = TSPSolution(route)
			if solution.cost != np.inf:
				return solution

	def nextGeneration(self):
		"""
		Produces a new generation of population.
		:return: a new population
		"""
		self.sortPopulation()
		selectionResults = self.selection()
		children = self.breedPopulation(selectionResults)
		nextGeneration = self.mutatePopulation(children)
		return nextGeneration

	def sortPopulation(self):
		"""
		Sorts population in place based on fitness.
		:return: None
		"""
		self.population.sort(key=self.getFitness, reverse=True)

	def getFitness(self, individual):
		"""
		Key function for sorting individuals based on fitness
		:return: fitness of the individual
		"""
		return individual.fitness

	def selection(self):
		"""
		Selects parents for mating pool.
		:return: list of selected parents for mating
		"""
		selectionResults = []
		fitnessSum = sum(route.fitness for route in self.population)
		interSums = [0]
		for i in range(len(self.population)):
			interSums.append(interSums[i] + self.population[i].fitness)

		for i in range(self.eliteSize):
			selectionResults.append(self.population[i])
		for i in range(len(self.population) - self.eliteSize):
			pick = fitnessSum * random.random()
			for j in range(len(self.population)):
				if interSums[j] < pick <= interSums[j + 1]:
					selectionResults.append(self.population[j])
					break
		return selectionResults

	def breedPopulation(self, matingpool):
		"""
		Create offspring population.
		:return: new population containing new offspring
		"""
		children = []
		length = len(matingpool) - self.eliteSize
		pool = random.sample(matingpool, len(matingpool))

		for i in range(self.eliteSize):
			children.append(matingpool[i])

		for i in range(length):
			child = self.breed(pool[i], pool[len(matingpool) - i - 1])
			children.append(child)
		return children

	def breed(self, parent1, parent2):
		"""
		Uses ordered crossover to create offspring.
		:return: offspring that resulted from breeding
		"""
		for x in range(100):
			child = []

			geneA = int(random.random() * len(parent1.route))
			geneB = int(random.random() * len(parent2.route))

			startGene = min(geneA, geneB)
			endGene = max(geneA, geneB)

			for i in range(startGene, endGene):
				child.append(parent1.route[i])

			routeFound = True
			while len(child) < len(parent1.route):
				childFound = False
				for item in parent2.route:
					if item not in child:
						# check if there is a path from item to last element in childp1
						if len(child) == 0:
							child.append(item)
							childFound = True
						elif child[-1].costTo(item) != np.inf:
							child.append(item)
							childFound = True
				if not childFound:
					routeFound = False
					break
			if routeFound is False:
				continue
			else:
				childSolution = TSPSolution(child)
				return childSolution
		return parent1


	def mutatePopulation(self, children):
		"""
		Mutates throughout new population.
		:return: mutated population
		"""
		# Don't mutate the best route
		mutatedPop = [children[0]]

		for ind in range(1, len(children)):
			mutatedInd = self.mutate(children[ind])
			mutatedPop.append(mutatedInd)
		return mutatedPop

	def mutate(self, individual):
		"""
		Uses swap mutation by swapping two cities in a route.
		:return: a new route with mutation
		"""
		for swapped in range(len(individual.route)):
			if random.random() < self.mutationRate:
				swapWith = int(random.random() * len(individual.route))

				city1 = individual.route[swapped]
				city2 = individual.route[swapWith]

				individual.route[swapped] = city2
				individual.route[swapped] = city1
				individual.calculateCost()
		return individual
