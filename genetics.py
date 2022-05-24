import numpy as np
import car
import math
import sys

'''
evolution of species

two parents create a new entire population
get the pair and swap some genes from these two and create a new pair with some mutations
'''

def mutateGenotype(genes, mutationGenotypeProbability = 1.0):

	'''
	do we need to mutate this genotype?
	'''

	if (np.random.random() < mutationGenotypeProbability):
		mutateGenes(genes)

def mutateGenes(genes,  mutationGeneProbability = 0.3, mutationGeneAmount = 2.0):

	'''
	apply some randomness to each gene from the genotype
	'''

	for i in range(len(genes)):
		if (np.random.random() < mutationGeneProbability):
			genes[i] += (np.random.random() * mutationGeneAmount * 2) - mutationGeneAmount



def crossOver(genes1, genes2, crossOverProbability = 0.6):

	'''
	swap i-gene from both genotypes
	'''

	if len(genes1) != len(genes2):
		print("genes dimension must match")
		sys.exit(-1)

	for i in range(len(genes1)):
		if (np.random.random() < crossOverProbability):
			q = genes1[i]
			genes1[i] = genes2[i]
			genes2[i] = q



def crossOverAndMutation(agent1, agent2, numchildren):

	''' 
	create a new population based on agents 1 and 2
	'''

	children = []

	if (agent1 is None) or (agent2 is None):
		print("expecting both agents for the new generation")
		sys.exit(-1)

	while (numchildren > 0):

		# get data from both parents
		w1 = agent1.getGenotype()
		w2 = agent2.getGenotype()

		# how will its children look?
		crossOver(w1, w2)
		mutateGenotype(w1)
		mutateGenotype(w2)

		# create
		children.append(w1)
		numchildren -= 1

		if (numchildren > 0):
			children.append(w2)
			numchildren -= 1



	return children

def randomRecombination(genotypes, genotype1, genotype2, numchildren):
	
	children = []

	if (genotype1 is None) or (genotype2 is None):
		print("expecting both agents for the new generation")
		sys.exit(-1)

	if (numchildren < 2):
		print("expecting a new generation with more than 2 individuals")
		sys.exit(-1)

	children.append(genotype1.getGenotype())
	children.append(genotype2.getGenotype())

	numchildren -= 2

	while (numchildren > 0):

		i1 = np.random.randint(0, len(genotypes))
		i2 = i1

		while i2 == i1:
			i2 = np.random.randint(0, len(genotypes))

		w1 = genotypes[i1].getGenotype()
		w2 = genotypes[i2].getGenotype()

		crossOver(w1, w2)
		mutateGenotype(w1)
		mutateGenotype(w2)

		children.append(w1)
		numchildren -= 1

		if numchildren > 0:
			children.append(w2)
			numchildren -= 1


	return children




def createCars(numgenotypes = 10, oldgenotypes = None, agent1 = None, agent2 = None):
	
	genotypes = []

	if (numgenotypes < 2):
		print("new generation must contain at least 2")
		sys.exit(-1)

	if (agent1 is None) and (agent2 is None) and (oldgenotypes is None):
		
		# create cars random

		for i in range(numgenotypes):
			newcar = car.Car(steer = -np.random.random() * math.pi)	
			genotypes.append(newcar)

	else:

		w = crossOverAndMutation(agent1, agent2, numchildren = numgenotypes)

		#w = randomRecombination(oldgenotypes, agent1, agent2, numgenotypes)

		for i in range(numgenotypes):
			newcar = car.Car(steer = -np.random.random() * math.pi)	
			newcar.setGenotype(w[i])
			genotypes.append(newcar)


	return genotypes