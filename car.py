#!/usr/bin/python

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import neuralnetwork
import tools
import time


class Car:

	'''
	car object, holds the location and neural network controller
	'''

	CAR_THICKNESS = 2
	CAR_WIDTH = 10
	CAR_LENGTH = 20

	SENSOR_DISTANCE = 300
	SENSOR_RADIUS = 2
	SENSOR_COLOR = (0, 255, 0)
	SENSOR_NUM = 5 			# num of sensors in the sensor array
	SENSOR_APERTURE = 100 	# aperture of the sensor array in grades

	CAR_THROTTLE_MAX = 1.0
	CAR_ENGINE_BRAKE = 2
	CAR_ACCELERATION = 8.0
	CAR_TURN_ACC = 4.0

	CAR_NORMAL_COLOR = (255, 0, 0)
	CAR_BEST_COLOR = (255, 255, 0)
	CAR_SECONDBEST_COLOR = (0, 255, 0)

	CAR_SPEED_MAX = 2 		# max speed
	STUCK_TIMEOUT = 20		# time to get to the next waypoint

	CAR_COLLISION_DISTANCE = 0.02 # collision detected if any sensor measure if less than



	def __init__(self, x = 0, y = 0, steer = 0, color = CAR_NORMAL_COLOR, width = CAR_WIDTH, length = CAR_LENGTH):

		self.startx = x
		self.starty = y
		self.cx = x
		self.cy = y

		self.steer = steer
		self.turn_ratio = 0
		self.speed = 0
		self.throttle = 0
		self.odometer = 0
		self.movingTimeout = time.time() + self.STUCK_TIMEOUT
		self.alive = True

		self.car_color = color
		self.car_thickness = self.CAR_THICKNESS


		# define car corners around (0,0)
		self.x1 = -width / 2
		self.y1 = length / 2
		self.x2 = width / 2
		self.y2 = length / 2
		self.x3 = width / 2
		self.y3 = -length / 2
		self.x4 = -width / 2
		self.y4 = -length / 2

		# init distance sensors
		self.sensorx = 0
		self.sensory = length / 4
		self.sensors = np.zeros([self.SENSOR_NUM])

		for i in range(self.SENSOR_NUM):
			self.sensors[i] = self.SENSOR_DISTANCE / self.SENSOR_DISTANCE

		self.sensorBounds = (0, 0)

		# init neuralnetwork
		self.nn = neuralnetwork.NeuralNetwork()
		self.nn.addLayer(self.SENSOR_NUM, 4)
		self.nn.addLayer(4, 3)
		self.nn.addLayer(3, 2)

		self.nn.randomWeights()

		self.output = np.zeros([2])

		# waypoints
		self.trackCompletion = 0 			# % of track completed from the beginning to the last crossed waypoint
		self.currentWayPointCompletion = 0 	# % completed of the current waypoint (% depends on num of waypoints)
		self.waypointIndex = 0 				# index of next waypoint
		self.bestTrackCompletion = 0 		# to measure distance between two checks, to check if stuck


		self.lastUpdateTime = 0


		self.paused = False
		self.pausedWhen = 0


	def pause(self):
		self.paused = True
		self.pausedWhen = time.time()

	def resume(self):
		self.paused = False
		#self.lastUpdateTime = self.lastUpdateTime + (time.time() - self.pausedWhen)
		self.movingTimeout = self.movingTimeout + (time.time() - self.pausedWhen)

	def update(self):

		'''
		updates next position based on controller inputs
		'''

		# time from last update

		if self.lastUpdateTime > 0:
			deltaTime = time.time() - self.lastUpdateTime
		else:
			deltaTime = 0

		self.lastUpdateTime = time.time()

		# have I collide?

		if self.collision(): 
			self.alive = False
			return

		if self.paused:
			return

		# calc acceleration

		if self.throttle > self.CAR_THROTTLE_MAX:
			self.throttle = self.CAR_THROTTLE_MAX

		if self.throttle < -self.CAR_THROTTLE_MAX:
			self.throttle = -self.CAR_THROTTLE_MAX


		if (self.throttle != 0):
			self.speed += self.throttle * self.CAR_ACCELERATION * deltaTime
		else:
			if self.speed > 0:
				self.speed -= self.CAR_ENGINE_BRAKE * deltaTime
				if self.speed < 0:
					self.speed = 0
			else:
				self.speed += self.CAR_ENGINE_BRAKE * deltaTime
				if self.speed > 0:
					self.speed = 0

		
		# and limit max speed

		if self.speed < 0:
			self.speed = 0

		if (self.speed > self.CAR_SPEED_MAX):
			self.speed = self.CAR_SPEED_MAX

		# calc steering angle

		self.steer = self.steer + self.turn_ratio * self.CAR_TURN_ACC * deltaTime

		# calc new position

		oldcx = self.cx
		oldcy = self.cy

		self.cx += math.cos(self.steer + math.pi/2) * self.speed
		self.cy += math.sin(self.steer + math.pi/2) * self.speed

		# distance from last point

		self.odometer += tools.distance(self.cx, self.cy, oldcx, oldcy)



	def checkForStuck(self):

		'''
		check if track completion has increased 
		if yes top life
		if not then reduce life
		'''

		if self.alive and not self.paused:

			if self.trackCompletion > self.bestTrackCompletion:
				self.movingTimeout = time.time() + self.STUCK_TIMEOUT
				self.bestTrackCompletion = self.trackCompletion
			else:
				if time.time() > self.movingTimeout:
					self.alive = False

	
	def reset(self):
		self.alive = True
		self.moving = time.time() + self.STUCK_TIMEOUT
		self.odometer = 0
		self.trackCompletion = 0
		self.waypointIndex = 0
		self.currentWayPointCompletion = 0
		self.bestTrackCompletion = 0
		self.paused = False


	def setNormalColor(self):
		self.car_color = self.CAR_NORMAL_COLOR
		self.car_thickness = self.CAR_THICKNESS

	def setBestColor(self):
		self.car_color = self.CAR_BEST_COLOR
		self.car_thickness = self.CAR_THICKNESS * 2

	def setSecondBestColor(self):
		self.car_color = self.CAR_SECONDBEST_COLOR
		self.car_thickness = self.CAR_THICKNESS * 2

	def draw_sensor_line(self, imgTrack, imgRes, angle):
		'''
		draw an imaginary straight line from the sensor start location to a point given by SENSOR_DISTANCE
		if any oclusion found the stop and return collision point and modular distance [0, 1]
		'''

		# start point
		sx, sy = tools.rotate(self.sensorx, self.sensory, self.steer)
		sx += self.cx
		sy += self.cy
		sx = int(sx)
		sy = int(sy)

		# theorical end point
		ex = self.sensorx + self.SENSOR_DISTANCE * math.cos(angle)
		ey = self.sensory + self.SENSOR_DISTANCE * math.sin(angle)

		# rotate 
		ex, ey = tools.rotate(ex, ey, self.steer)
		ex += self.cx
		ey += self.cy
		ex = int(ex)
		ey = int(ey)

		# check if collision from sx,sy to ex,ey
		px, py = self.detect_collision(imgTrack, sx, sy, ex, ey)

		# paint rays
		cv.line(imgRes, (sx, sy), (px, py), (0, 0, 255), 1)

		# return collision point of ray plus normalized distance

		return px, py, tools.min(tools.distance(sx, sy, px, py), self.SENSOR_DISTANCE) / self.SENSOR_DISTANCE


	def detect_collision(self, imgTrack, x1, y1, x2, y2):
	
		'''
		just a simple line tracing from (x1,y1) to (x2,y2)
		returns when the next point in line is ocuppied yet
		'''

		dx = abs(x2 - x1)
		if (x1 < x2): 
			sx = 1
		else: 
			sx = -1

		dy = abs(y2 - y1)
		if (y1 < y2):
			sy = 1
		else:
			sy = -1

		if (dx > dy):
			err = dx / 2
		else:
			err = -dy / 2


		while 1:

			if (x1 == x2) and (y1 == y2): break

			e2 = err

			if (e2 > -dx):
				err -= dy
				x1 += sx
			if (e2 < dy):
				err += dx
				y1 += sy

			# check for collision
			if not self.checkColor(imgTrack, x1, y1):
				return x1, y1

		# assume x2, y2 to be the end

		return x2, y2

	def checkColor(self, img, x, y):
		
		# detect if track margins on (x,y)

		# check for outside the field, outside view is considered also track margin

		if (x >= self.sensorBounds[0]) or (x<0):
			return False

		if (y >= self.sensorBounds[1]) or (y<0):
			return False

		# is pixel color = track color?

		q = img[y, x]

		if q[0] != 255 and q[1] != 255 and q[2] != 255:
			return False

		# no collision

		return True




	def draw_sensor_lines(self, imgTrack, imgResult):

		'''
		paint sensor lines and update sensor measurements
		'''


		if self.alive:

			# calc angle and angle increment for a given aperture

			sensor_initial_angle = (math.pi/2) - ((self.SENSOR_APERTURE / 2) * math.pi / 180)
			sensor_incremental_angle = (self.SENSOR_APERTURE * math.pi / 180) / (self.SENSOR_NUM - 1)

			# calc all sensor readings

			for i in range(self.SENSOR_NUM):
				x, y, self.sensors[i] = self.draw_sensor_line(imgTrack, imgResult, sensor_initial_angle + sensor_incremental_angle * i)
				cv.circle(imgResult, (x, y), self.SENSOR_RADIUS, self.SENSOR_COLOR, 1)

	def draw(self, img):

		'''
		draw car contour
		'''

		p1x, p1y = tools.rotate(self.x1, self.y1, self.steer)
		p2x, p2y = tools.rotate(self.x2, self.y2, self.steer)
		p3x, p3y = tools.rotate(self.x3, self.y3, self.steer)
		p4x, p4y = tools.rotate(self.x4, self.y4, self.steer)

		p1x += self.cx
		p1y += self.cy
		p2x += self.cx
		p2y += self.cy
		p3x += self.cx
		p3y += self.cy
		p4x += self.cx
		p4y += self.cy

		p1x = int(p1x)
		p1y = int(p1y)
		p2x = int(p2x)
		p2y = int(p2y)
		p3x = int(p3x)
		p3y = int(p3y)
		p4x = int(p4x)
		p4y = int(p4y)

		cv.line(img, (p1x, p1y), (p2x, p2y), self.car_color, self.car_thickness)
		cv.line(img, (p2x, p2y), (p3x, p3y), self.car_color, self.car_thickness)
		cv.line(img, (p3x, p3y), (p4x, p4y), self.car_color, self.car_thickness)
		cv.line(img, (p4x, p4y), (p1x, p1y), self.car_color, self.car_thickness)

		return img

	
	def autopilot(self):

		'''
		calc outputs based on sensor readings
		'''

		if self.alive:
			self.output = self.nn.processInputs(self.sensors)
			self.output[0] = self.output[0]
			self.output[1] = (self.output[1] - 0.5)

	def apply(self):

		'''
		apply outputs from NN to throttle and turn
		'''

		self.throttle = self.output[0]
		self.turn_ratio = self.output[1]

	def getPos(self):
		return self.cx, self.cy


	def setPos(self, pos):
		self.cx = pos[0]
		self.cy = pos[1]

	def completion(self):
		return self.currentWayPointCompletion + self.trackCompletion

	def completed(self):
		return (self.trackCompletion >= 1)

	def setSensorBounds(self, bounds):
		self.sensorBounds = bounds

	def collision(self):

		# check for any sensor readings less than this

		for i in range(len(self.sensors)):
			if self.sensors[i] < self.CAR_COLLISION_DISTANCE: return True

		return False

	def isAlive(self):
		return self.alive

	def getTimer(self):
		if self.alive:
			if not self.paused:
				return int(self.movingTimeout - time.time())
			else:
				return int(self.movingTimeout - self.pausedWhen)
		else:
			return 0

	def getGenotype(self):
		'''
		get features of this individual
		'''

		return self.nn.getWeights()

	def setGenotype(self, w):
		'''
		set features of this individual
		'''
		self.nn.setWeights(w)
