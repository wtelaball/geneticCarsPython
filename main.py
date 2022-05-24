#!/usr/bin/python

import math
import random
import numpy as np
import cv2 as cv
import copy
import car
import sys
import tracks
import tools
import time

WINDOW_HEIGHT = 600
WINDOW_WIDTH = 800


def mutateGenotype(genes, mutationGenotypeProbability = 1.0):

	if (np.random.random() < mutationGenotypeProbability):
		mutateGenes(genes)

def mutateGenes(genes,  mutationGeneProbability = 0.3, mutationGeneAmount = 2.0):

	for i in range(len(genes)):
		if (np.random.random() < mutationGeneProbability):
			genes[i] += (np.random.random() * mutationGeneAmount * 2) - mutationGeneAmount



def crossOver(genes1, genes2, crossOverProbability = 0.6):

	if len(genes1) != len(genes2):
		print("genes dimension must match")
		sys.exit(-1)

	for i in range(len(genes1)):
		if (np.random.random() < crossOverProbability):
			q = genes1[i]
			genes1[i] = genes2[i]
			genes2[i] = q



def crossOverAndMutation(agent1, agent2, numchildren):

	children = []

	while (numchildren > 0):

		w1 = agent1.getGenotype()
		w2 = agent2.getGenotype()

		crossOver(w1, w2)
		mutateGenotype(w1)
		mutateGenotype(w2)

		# create

		children.append(w1)
		children.append(w2)

		numchildren -= 2

	return children

def randomRecombination(genotypes, genotype1, genotype2, numchildren):
	
	children = []

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

	if (numgenotypes <=2):
		print("new generation must contain at least 2")
		sys.exit(-1)

	if ((numgenotypes % 2) == 1):
		print("genotypes created must be 2-fold")
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

def sleep(timer):

	timeout = time.time() + timer

	while (time.time() < timeout):
		if (cv.waitKey(1) != -1): break

def printStats(img, x, y, trackManager, cars, best, secondbest):

	k = 0

	if best is not None:
		trackManager.printData(img, best, x, y + k * 100)
		k +=1

	if secondbest is not None:
		trackManager.printData(img, secondbest, x, y + k * 100)
		k +=1

	for car in cars:
		if car.isAlive():
			if car != best and car != secondbest:
				trackManager.printData(img, car, x, y + k * 100)
				k += 1

def showNeuronWeights(best, secondBest):

	imgNeuron = np.full((500, 500, 3), (255, 255, 255), dtype = np.uint8)				

	if best is not None:
		best.nn.graph(imgNeuron, 0)

	if secondBest is not None:
		secondBest.nn.graph(imgNeuron, 250)

	return imgNeuron

def zoom(img, x, y, zoomPercent = 2.0):
	
	ZOOM_AREA_WIDTH = 100
	ZOOM_AREA_HEIGHT = 100

	imgWidth = img.shape[1]
	imgHeight = img.shape[0]

	imgZoom = np.full((200, 200, 3), (255, 255, 255), dtype = np.uint8)

	xmin = int(tools.max(0, x - ZOOM_AREA_WIDTH / 2))
	xmax = int(tools.min(imgWidth, x + ZOOM_AREA_WIDTH / 2))
	ymin = int(tools.max(0, y - ZOOM_AREA_HEIGHT / 2))
	ymax = int(tools.min(imgHeight, y + ZOOM_AREA_HEIGHT / 2))

	imgCrop = copy.copy(img[ymin:ymax, xmin:xmax])

	imgCrop = cv.resize(imgCrop, (int(zoomPercent * ZOOM_AREA_HEIGHT), int(zoomPercent * ZOOM_AREA_WIDTH)), interpolation = cv.INTER_CUBIC)

	return imgCrop


def main():

	cv.namedWindow('carSim')
	cv.namedWindow('nn')
	cv.namedWindow('zoom')

	cv.moveWindow('carSim', 0, 0)
	cv.moveWindow('nn', 700, 0)
	cv.moveWindow('zoom', 700, 600)

	trackManager = tracks.TrackManager()
	trackManager.load("tracks/track1_wp.png")


	imgNeuron = np.full((500, 500, 3), (255, 255, 255), dtype = np.uint8)
	cv.imshow('nn', imgNeuron)

	generations = 0
	done = False
	best = None
	secondBest = None
	cars = None
	paused = False

	while generations < 100 and not done:

		cars = createCars(10, cars, best, secondBest)

		for car in cars:
			car.setPos(trackManager.getStart())
			car.setSensorBounds(trackManager.getBounds())



		exit = False

		trackImg = copy.copy(trackManager.getImage())


		while not exit:
			q = cv.waitKey(1) & 0xff
			key = chr(q).upper()
			exit = (key == 'Q')
			done = (key == 'Q')

			if (key == 'E'):
				cars[0].turn_ratio = -math.pi / 180
			elif (key == 'R'):
				cars[0].turn_ratio = +math.pi / 180
			else:
				cars[0].turn_ratio = 0

			if (key == 'W'):
				cars[0].throttle = 1.0
			else:
				cars[0].throttle = 0.0

			if (key == 'P'):
				paused = not paused
				if paused:
					for car in cars:
						car.pause()
				else:
					for car in cars:
						car.resume()


			trackRes = copy.copy(trackImg)

			for car in cars:
				car.autopilot()
				car.apply()
				car.update()
				car.checkForStuck()
				car.draw_sensor_lines(trackImg, trackRes)
				car.draw(trackRes)

				trackManager.updateDistanceToNextWaypoint(car)

			best, secondBest = trackManager.bestCar(cars)

			zoomImg = zoom(trackRes, best.getPos()[0], best.getPos()[1])

			printStats(trackRes, 10, 20, trackManager, cars, best, secondBest)


			cv.imshow('carSim', trackRes)
			cv.imshow('zoom', zoomImg)

			if trackManager.allDone(cars):
				print("finished generation %d" %(generations))
				exit = True


		imgNeuron = showNeuronWeights(best, secondBest)

		cv.imshow('nn', imgNeuron)

		generations += 1

		if not done:
			sleep(1)



if __name__ == '__main__':
	main()