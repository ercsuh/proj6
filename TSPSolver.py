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


# min heap
class Queue:
	def __init__(self):
		self.heap = []
		self.size = 0

	# add item to queue
	def put(self, node):
		# append to end
		self.heap.append(node)
		self.size += 1
		# set i to index of appended node
		i = len(self.heap) - 1
		if i == 0:
			return
		# bubble down until parent is less than node
		j = math.floor(i / 2) if i % 2 != 0 else math.floor(i / 2 - 1)
		while self.heap[j].priority > self.heap[i].priority:
			self.switch(i, j)
			i = j
			j = math.floor(i / 2) if i % 2 != 0 else math.floor(i / 2 - 1)
			if j < 0:
				return

	# get min item from queue
	def get(self):
		# switch first and last items
		self.switch(0, len(self.heap) - 1)
		# remove and return the last item
		min_node = self.heap.pop()
		self.size -= 1
		# bubble up first node until descendants are greater
		i = 0
		while True:
			left = i * 2 + 1
			right = i * 2 + 2
			# account for the case where one or no descendants exist
			if left >= len(self.heap):
				break
			if right >= len(self.heap):
				if self.heap[left].priority < self.heap[i].priority:
					self.switch(i, left)
				break
			# otherwise, pick the smallest descendant and switch
			priority = self.heap[i].priority
			left_priority = self.heap[left].priority
			right_priority = self.heap[right].priority
			min_child = i
			if left_priority < priority and left_priority < right_priority:
				min_child = left
			elif right_priority < priority and right_priority <= left_priority:
				min_child = right
			else:
				break
			self.switch(i, min_child)
			i = min_child
		return min_node

	# returns true if queue is empty
	def empty(self):
		return self.size == 0

	# switches two nodes
	def switch(self, i, j):
		temp = self.heap[i]
		self.heap[i] = self.heap[j]
		self.heap[j] = temp


# TSP node
class Node:
	def __init__(self, bound, path, rcm, ncities):
		self.bound = bound
		self.path = path
		self.rcm = rcm
		self.priority = np.inf
		self.ncities = ncities

	# copies node, appends city to new node, and calculates new lower bound
	def expand(self, city):
		# if there's no edge from last city in path to city, abort
		source = self.path[-1]._index
		dest = city._index
		cost = self.rcm[source][dest]
		if cost == np.inf:
			return None
		# copy node
		node = Node(self.bound, self.path.copy(), np.copy(self.rcm), self.ncities)
		# append city and calculate bound
		node.append(city, source, dest, cost)
		return node

	# computes the queue priority of the node
	def computePriority(self):
		self.priority = self.bound - ((len(self.path)) * (len(self.path)) * 50)

	# appends city to path and calculates new lower bound
	def append(self, city, source, dest, cost):
		# append new city
		self.bound += cost
		self.path.append(city)
		# mark appropriate slots in rcm as infinity
		for i in range(self.ncities):
			self.rcm[i, dest] = np.inf
		for j in range(self.ncities):
			self.rcm[source, j] = np.inf
		self.rcm[dest, source] = np.inf
		# recalculate rcm and lower bound
		self.computeBound()

	# calculates rcm and adjusts the lower bound
	def computeBound(self):
		# take min of each row and subtract it across the row
		minRows = np.min(self.rcm, axis=1)
		for i in range(self.ncities):
			if minRows[i] == np.inf:
				continue
			if minRows[i] == 0:
				continue
			for j in range(self.ncities):
				self.rcm[i, j] -= minRows[i]
			self.bound += minRows[i]
		# take min of each column and subtract it across the column
		minCols = np.min(self.rcm, axis=0)
		for j in range(self.ncities):
			if minCols[j] == np.inf:
				continue
			if minCols[j] == 0:
				continue
			for i in range(self.ncities):
				self.rcm[i, j] -= minCols[j]
			self.bound += minCols[j]
		# compute queue priority with updated bound
		self.computePriority()


class TSPSolver:
	def __init__( self, gui_view ):
		self.populationSize = 50
		self.eliteSize = 10
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
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		max_q = 1
		total = 0
		pruned = 0
		q = Queue()
		bssf = None
		minCost = np.inf

		start_time = time.time()
		# create initial cost matrix using graph
		graphMatrix = np.ndarray((ncities, ncities), dtype=float)
		for cityA in cities:
			for cityB in cities:
				graphMatrix[cityA._index, cityB._index] = cityA.costTo(cityB)

		# create initial node with path of one city and initial rcm
		initialPath = [cities[0]]
		root = Node(0, initialPath, graphMatrix, ncities)
		root.computeBound()
		total += 1

		# find a cycle using the greedy method for our bssf
		greedyResults = self.greedy()
		bssf = greedyResults['soln']
		minCost = greedyResults['cost']

		# if lower bound on initial node is worse than greedy, no use trying
		if root.bound < minCost:
			q.put(root)

		# go until there are no nodes left or time runs out
		while not q.empty():
			if time.time() - start_time > time_allowance:
				break
			node = q.get()
			# reject node if lower bound is worse than bssf
			if node.bound >= minCost:
				pruned += 1
				continue
			# if this is the last city in the cycle, do things differently
			if len(node.path) == ncities - 1:
				for city in cities:
					if time.time() - start_time > time_allowance:
						break
					# don't try to add city if it's already in the path
					if city in node.path:
						continue
					# expand node with city
					newNode = node.expand(city)
					# if there's no edge, abandon ship
					if newNode is None:
						continue
					total += 1
					# try solution and see if it's better than bssf
					newSolution = TSPSolution(newNode.path)
					if newSolution.cost < minCost:
						bssf = newSolution
						minCost = newSolution.cost
						count += 1
					else:
						pruned += 1
			# if this isn't the last city in the cycle, do the normal
			else:
				for city in cities:
					if time.time() - start_time > time_allowance:
						break
					# don't try to add city if it's already in the path
					if city in node.path:
						continue
					# expand node with city
					newNode = node.expand(city)
					# if there's no edge, abandon ship
					if newNode is None:
						continue
					total += 1
					# if new node's bound is less than bssf, add to queue
					if newNode.bound < minCost:
						q.put(newNode)
						if q.size > max_q:
							max_q = q.size
					else:
						pruned += 1
		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = max_q
		results['total'] = total
		results['pruned'] = pruned
		return results

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

			if not routeFound:
				continue
			else:
				childSolution = TSPSolution(child)
				if childSolution.cost != np.inf:
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
