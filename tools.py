import cv2 as cv
import math
import numpy as np

def is_cv2():
	# if we are using OpenCV 2, then our cv2.__version__ will start
	# with '2.'
	return check_opencv_version("2.")

def is_cv3():
	# if we are using OpenCV 3.X, then our cv2.__version__ will start
	# with '3.'
	return check_opencv_version("3.")

def is_cv4():
	# if we are using OpenCV 3.X, then our cv2.__version__ will start
	# with '4.'
	return check_opencv_version("4.")


def check_opencv_version(major, lib=None):
	return cv.__version__.startswith(major)


def distance(x1, y1, x2, y2):
	d = (x1 - x2) ** 2
	d += (y1 - y2) ** 2
	return math.sqrt(d)


def resize(image, ratio):
	width = int(image.shape[1] * ratio / 100)
	height = int(image.shape[0] * ratio / 100)
	dim = (width, height)

	resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)

	return resized


def rotate(x, y, angle):
	x1 = x * math.cos(angle) - y * math.sin(angle)
	y1 = x * math.sin(angle) + y * math.cos(angle)

	return x1, y1

def min(x, y):
	if (x < y): 
		return x 
	else: 
		return y

def max(x, y):
	if (x > y):
		return x
	else:
		return y

def maxFromList(q):
	maximo = q[0]
	for i in q:
		if i > maximo:
			maximo = i

	return maximo

def minFromList(q):
	minimo = q[0]
	for i in q:
		if i < minimo:
			minimo = i

	return minimo

def sigmoid(x):
	return (1 / (1 + np.exp(-x)))
