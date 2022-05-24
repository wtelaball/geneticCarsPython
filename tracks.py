import math
import cv2 as cv
import numpy as np
import tools
import sys
import copy
import time

class Waypoint:

	RADIUS = 25

	def __init__(self, x = 0, y = 0):
		self.x = x
		self.y = y
		self.visible = False

	def getPos(self):
		return self.x, self.y

	def setPos(self, x, y):
		self.x = x
		self.y = y


	def __str__(self):
		return "(%d,%d)" %(self.x, self.y)


class TrackManager:


	FONT_SCALE = 0.5
	FONT_COLOR = (0, 0, 0)

	def __init__(self, start_x = 0, start_y = 0):
		self.waypoints = []
		self.perWayPointCompletion = 0

		self.startPosition = Waypoint(start_x, start_y)

		# font calcs
		self.font = cv.FONT_HERSHEY_SIMPLEX
		dummy, self.fontSize = cv.getTextSize("TEST", self.font, self.FONT_SCALE, 1)[0]
		self.fontSize += 2

		if tools.is_cv2():
			self.fontAA = cv.CV_AA
		elif tools.is_cv3() or tools.is_cv4():
			self.fontAA = cv.LINE_AA
		else:
			print 'not compatible opencv version'
			sys.exit(1)

	def setStart(self, sx, sy):
		self.startPosition.setPos(sx, sy)

	def getStart(self):
		return self.startPosition.getPos()


	def sortWaypoints(self):
		# sort each time one wp is added
		# let the first one be the highest 'y' and sort them y-decrementally

		for i in range(len(self.waypoints)):
			for j in range(i, len(self.waypoints)):
				if self.waypoints[j].getPos()[1] > self.waypoints[i].getPos()[1]:
					aux = self.waypoints[i].getPos()
					self.waypoints[i].setPos(self.waypoints[j].getPos())
					self.waypoints[j].setPos(aux)



	def addWaypoint(self, wpx, wpy):
		self.waypoints.append(Waypoint(wpx, wpy))
		self.perWayPointCompletion = 1.0 / len(self.waypoints)
		self.sortWaypoints()


	def distanceBetweenWaypoints(self, wpindex0, wpindex1):

		# special case when is the first one
		if (wpindex0 == 0) and (wpindex1 == 0):
			x0, y0 = self.startPosition.getPos()
			x1, y1 = self.waypoints[0].getPos()
		else:
			x0, y0 = self.waypoints[wpindex0].getPos()
			x1, y1 = self.waypoints[wpindex1].getPos()

		return tools.distance(x0, y0, x1, y1)


	def getWayPointCompletion(self, car):

		# if no waypoints then track is finished
		if self.numWaypoints() == 0: return 1

		wpindex = car.waypointIndex

		# is the first one or finished?
		if (wpindex == 0):
			totalWPDist = self.distanceBetweenWaypoints(0, 0)
		elif (wpindex == self.numWaypoints()):
			wpCompletion = 1
			return wpCompletion
		else:
			totalWPDist = self.distanceBetweenWaypoints(wpindex - 1, wpindex)

		# calc waypoint completion based on distance between car and next wp
		carx, cary = car.getPos()
		wpx, wpy = self.waypoints[wpindex].getPos()

		currDistToWp = tools.distance(carx, cary, wpx, wpy)

		#print currDistToWp

		# close enough?
		if (currDistToWp < Waypoint.RADIUS):
			wpCompletion = 1
		else:
			wpCompletion = (totalWPDist - currDistToWp) / totalWPDist

			# just to make sure
			if (wpCompletion < 0):
				wpCompletion = 0

			if (wpCompletion > 1):
				wpCompletion = 1

		return wpCompletion


	def updateDistanceToNextWaypoint(self, car):

		completion = self.getWayPointCompletion(car)

		if (completion == 1):
			# waypoint completed
			car.waypointIndex += 1

			# is track finished?
			if (car.waypointIndex < len(self.waypoints)):
				car.currentWayPointCompletion = 0
				car.trackCompletion += self.perWayPointCompletion
			else:
				car.waypointIndex = len(self.waypoints)
				car.currentWayPointCompletion = 0
				car.trackCompletion = 1
		else:
			# waypoint not completed
			car.waypointIndex += 0
			car.currentWayPointCompletion = completion * self.perWayPointCompletion
			car.trackCompletion += 0



	def numWaypoints(self):
		return len(self.waypoints)


	def removeTransparency(self, source, background_color):
		source_img = cv.cvtColor(source[:,:,:3], cv.COLOR_BGR2GRAY)
		source_mask = source[:,:,3]  * (1 / 255.0)

		background_mask = 1.0 - source_mask

		bg_part = (background_color * (1 / 255.0)) * (background_mask)
		source_part = (source_img * (1 / 255.0)) * (source_mask)

		return cv.cvtColor(np.uint8(cv.addWeighted(bg_part, 255.0, source_part, 255.0, 0.0)), cv.COLOR_GRAY2BGR)

	def load(self, filename):
		self.originalImage = cv.imread(filename, cv.IMREAD_UNCHANGED)

		rf = self.detectStart()
		gf = self.detectWaypoints()

		self.detectTrack(rf, gf)

	def detectTrack(self, redfilter, greenfilter):
		blackFilter = copy.copy(self.originalImage)[:,:,:3]
		blackFilter = cv.cvtColor(blackFilter, cv.COLOR_BGR2HSV)
		mask = cv.inRange(blackFilter, (0, 0, 0), (255, 55, 150))
		mask = cv.bitwise_not(mask)
		color = copy.copy(self.originalImage)[:,:,:3]
		white = np.full((color.shape[0], color.shape[1], 3), (255, 255, 255), dtype = np.uint8)
		blackFilter = cv.bitwise_and(white, white, mask = mask)


		cv.imshow('carSim', mask)
		cv.waitKey(0)

		self.trackImage = blackFilter


	def detectStart(self):
		redFilter = copy.copy(self.originalImage)
		redFilter = cv.cvtColor(redFilter[:,:,:3], cv.COLOR_BGR2HSV)
		maskRed1 = cv.inRange(redFilter, (0, 70, 50), (10, 255, 255))
		maskRed2 = cv.inRange(redFilter, (170, 70, 50), (180, 255, 255))

		redFilter = maskRed1 + maskRed2

		#redFilter = cv.bitwise_and(redFilter, redFilter, mask = (maskRed1 + maskRed2)) #maskRed1 + maskRed2

		#redFilter = cv.cvtColor(redFilter, cv.COLOR_BGR2GRAY)

		contorno = cv.findContours(redFilter, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		contorno = contorno[0]

		if (len(contorno) != 1):
			print("no start detected")
			sys.exit(-1)

		for c in contorno:
			#print cv.boundingRect(c)
			x, y, w, h = cv.boundingRect(c)
			self.setStart(x + w/2, y + h/2)


		return redFilter

	def detectWaypoints(self):
		hsv = copy.copy(self.originalImage)
		hsv = cv.cvtColor(hsv[:,:,:3], cv.COLOR_BGR2HSV)

		mask_green = cv.inRange(hsv, (36, 25, 25), (70, 255,255))

		greenFilter = mask_green


		contorno = cv.findContours(greenFilter, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		contorno = contorno[0]

		if (len(contorno) == 0):
			print("no waypoints detected")
			sys.exit(-1)

		for c in contorno:
			#print cv.boundingRect(c)
			x, y, w, h = cv.boundingRect(c)
			self.addWaypoint(x + w/2, y + h/2)

		return greenFilter

	def printWaypoints(self):
		k = 0

		for i in self.waypoints:
			print("wp=%d (%d,%d)" %(k, i.x, i.y))
			k += 1


	def getImage(self):
		return self.trackImage


	def getBounds(self):
		return self.trackImage.shape[1], self.trackImage.shape[0]

	def printData(self, img, car, x, y):
		posy = y

		text = "POS=(%d, %d)" %(car.cx, car.cy) + " WPI=%d WPC=%d%% TC=%d%% C=%.1f%% M=%d" %(car.waypointIndex, 100 * car.currentWayPointCompletion, 100 * car.trackCompletion, 100.0 * car.completion(), car.getTimer())
		cv.putText(img, text, (x, posy), self.font, self.FONT_SCALE, self.FONT_COLOR, 1, self.fontAA)
		posy += self.fontSize

		text = "T=%d%%" %(int(car.throttle * 100)) + " ODO=%d " %(int(car.odometer)) + "SPD=%.2f TR=%.2f ST=%d" %(car.speed, car.turn_ratio, car.steer * 180 / math.pi)
		cv.putText(img, text, (x, posy), self.font, self.FONT_SCALE, self.FONT_COLOR, 1, self.fontAA)
		posy += self.fontSize

		text = "S=[ "

		for i in range(car.SENSOR_NUM):
			text += "%.2f " %(car.sensors[i])
		
		text += "] C=%d" %(car.collision())
		cv.putText(img, text, (x, posy), self.font, self.FONT_SCALE, self.FONT_COLOR, 1, self.fontAA)
		posy += self.fontSize

		text = "O=[%.2f %.2f]"  %(car.output[0], car.output[1])
		cv.putText(img, text, (x, posy), self.font, self.FONT_SCALE, self.FONT_COLOR, 1, self.fontAA)		
		posy += self.fontSize


	def bestCar(self, cars):

		best = None
		secondBest = None

		# select the first one

		for car in cars:
			
			# put all in normal

			car.setNormalColor()

			if best is None:
				best = car
			else:
				if (car.completion() > best.completion()):
					best = car

		# once best selected, select the next one

		if best is not None:

			for car in cars:

				if secondBest is None:
					if car != best:
						secondBest = car
				else:
					if car != best:
						if (car.completion() > secondBest.completion()) and (car.completion() <= best.completion()):
							secondBest = car


		if best is not None:		
			best.setBestColor()

		if secondBest is not None:
			secondBest.setSecondBestColor()

		return best, secondBest

	def allDone(self, cars):
		for car in cars:
			if car.isAlive(): 
				return False

		return True
