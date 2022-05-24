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
import genetics



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


def createWindows():
	cv.namedWindow('carSim')
	cv.namedWindow('nn')
	cv.namedWindow('zoom')

	cv.moveWindow('carSim', 0, 0)
	cv.moveWindow('nn', 700, 0)
	cv.moveWindow('zoom', 700, 600)


def main():

	createWindows()

	trackManager = tracks.TrackManager()
	trackManager.load("tracks/track1_wp.png")


	generations = 0
	done = False
	best = None
	secondBest = None
	cars = None
	paused = False

	while generations < 100 and not done:

		cars = genetics.createCars(10, cars, best, secondBest)

		# let the new born learn some basics from the track

		for car in cars:
			car.setPos(trackManager.getStart())
			car.setSensorBounds(trackManager.getBounds())



		exit = False

		trackImg = copy.copy(trackManager.getImage())


		while not exit and not done:
			q = cv.waitKey(1) & 0xff
			key = chr(q).upper()
			exit = (key == '1')
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
				exit = True


		print("finished generation %d" %(generations))

		imgNeuron = showNeuronWeights(best, secondBest)
		cv.imshow('nn', imgNeuron)

		generations += 1

		if not done:
			sleep(1)



if __name__ == '__main__':
	main()