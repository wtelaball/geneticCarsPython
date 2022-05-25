import numpy as np
import copy
import sys
import cv2 as cv
import tools

class NeuralLayer:
	'''
	object containing a neural layer
	'''

	def __init__(self, nodeCount, outputCount):

		if (nodeCount < 1):
			print("one node at least for the layer")
			sys.exit(-1)

		if (outputCount < 1):
			print("one output at least for the layer")
			sys.exit(-1)

		self.nodeCount = nodeCount
		self.outputCount = outputCount
		# 2 outputs 3 nodes
		# w11 w12 w13 w14		o = w11 * i1 + w12 * i2 + w13 * i3 + b1, being b1 = w14
		# w21 w22 w23 w24
		self.weights = np.zeros([outputCount, nodeCount + 1])


	def setWeights(self, weights):
		
		dimension = self.outputCount * (self.nodeCount + 1)

		# weights have weights plus biases
		if len(weights) != dimension:
			print("weights not enough for this layer")
			sys.exit(-1)

		k = 0

		for j in range(self.outputCount):
			for i in range(self.nodeCount + 1):
				self.weights[j, i] = weights[k]
				k += 1

	def getWeights(self):
		w = []

		for j in range(self.outputCount):
			for i in range(self.nodeCount + 1):
				w.append(self.weights[j, i])

		return w

	def getTopology(self):
		return self.outputCount, self.nodeCount + 1

	def getDimension(self):
		return self.outputCount * (self.nodeCount + 1)

	def activation(self, x):
		return tools.sigmoid(x)

	def processInputs(self, inputs):
		'''
		calc the output of the layer for a given input
		'''

		if len(inputs) != self.nodeCount:
			print("inputs does not match node count")
			sys.exit(-1)


		# o = w11 * i1 + w12 * i2 + w13 * i3 + w14, w14 is the bias

		sum = np.zeros([self.outputCount])

		for j in range(self.outputCount):

			for i in range(self.nodeCount):
				sum[j] += inputs[i] * self.weights[j, i]

			# add bias
			sum[j] += self.weights[j, self.nodeCount]
			
			sum[j] = self.activation(sum[j])

		return sum

	def randomWeights(self, min, max):
		'''
		set random weights to the layer
		'''
		
		rango = abs(min - max)

		for j in range(self.outputCount):
			for i in range(self.nodeCount + 1):
				self.weights[j, i] = min + (np.random.rand() * rango)



	def show(self):
		print("weights")
		print(self.weights)

class NeuralNetwork:

	'''
	object containing the neural layers
	'''

	def __init__(self):
		self.layers = []

	def addLayer(self, neuronCount, outputCount):
		'''
		add a layer to the neural network
		'''

		layer = NeuralLayer(neuronCount, outputCount)
		self.layers.append(layer)

	def processInputs(self, inputs):

		'''
		calc the output of the neural network based on the inputs given
		'''
		if len(self.layers) == 0:
			print("add at least one layer to the nn")
			sys.exit(-1)

		if len(inputs) != self.layers[0].nodeCount:
			print("inputs does not match node count on first layer")
			sys.exit(-1)

		outputs = copy.copy(inputs)
		k = 1

		for layer in self.layers:
			#print("layer %d input" %(k))
			#print(outputs)

			outputs = layer.processInputs(outputs)

			#print("layer %d output" %(k))
			#print(outputs)

			k += 1

		

		return outputs

	def randomWeights(self, min = -1.0, max = 1.0):
		'''
		apply random weights to all layers
		'''
		for layer in self.layers:
			layer.randomWeights(min, max)

	def show(self):
		k = 1

		for layer in self.layers:
			print("layer %d nc %d oc %d" %(k, layer.nodeCount, layer.outputCount))
			layer.show()
			k += 1

	def getWeights(self):
		'''
		create a weight list with the weights coming from layer to layer
		'''

		weights = []

		for layer in self.layers:
			wl = layer.getWeights()
			#print('layer len=', layer.getDimension(), "wl len=", len(wl))
			for w in wl:
				weights.append(w) 

		#print('nn len=', len(weights))

		return weights

	def setWeights(self, weights):
		'''
		apply weights in a layer to layer basis
		get chunks of weights and apply to the corresponding layer
		'''

		k = 0

		for layer in self.layers:
			
			dimension = layer.getDimension()
			wl = []
			
			for i in range(dimension):
				wl.append(weights[k])
				k += 1


			layer.setWeights(wl)



	NEURON_COLOR = (255, 0, 0)
	NEURON_RADIUS = 10
	NEURON_THICKNESS = 2
	NEURON_YSTART = 50
	NEURON_XSTART = 50
	NEURON_YSTEP = 50
	LAYER_XSTEP = 100
	WEIGHT_COLOR_POSITIVE = (0, 255, 0)
	WEIGHT_COLOR_NEGATIVE = (0, 0, 255)
	WEIGHT_THICKNESS = 10

	def graph(self, img, ystart = 0):

		'''
		shows a visual representation of the neural network on given img
		'''

		if len(self.layers) == 0:
			print("add at least one layer to the nn")
			sys.exit(-1)
			

		# draw the dots

		x = self.NEURON_XSTART

		for layer in self.layers:
			y = self.NEURON_YSTART + ystart
			for n in range(layer.nodeCount):
				cv.circle(img, (x, y), self.NEURON_RADIUS, self.NEURON_COLOR, self.NEURON_THICKNESS)
				y += self.NEURON_YSTEP
			x += self.LAYER_XSTEP

		y = self.NEURON_YSTART + ystart

		for o in range(self.layers[len(self.layers) - 1].outputCount):
			cv.circle(img, (x, y), self.NEURON_RADIUS, self.NEURON_COLOR, self.NEURON_THICKNESS)
			y += self.NEURON_YSTEP



		# connect the dots

		for i in range(len(self.layers)):

			# get the abs max weight to scale thickness
			weights = self.layers[i].getWeights()

			maxw = tools.maxFromList(weights)
			minw = tools.minFromList(weights)

			if abs(maxw) > abs(minw):
				maxw = abs(maxw)
			else:
				maxw = abs(minw)

			for n in range(self.layers[i].nodeCount):

				x1 = self.NEURON_XSTART + i * self.LAYER_XSTEP
				y1 = self.NEURON_YSTART + n * self.NEURON_YSTEP	+ ystart
				x2 = x1 + self.LAYER_XSTEP

				for o in range(self.layers[i].outputCount):

					y2 = self.NEURON_YSTART + o * self.NEURON_YSTEP + ystart
					w = self.layers[i].weights[o, n]
					thickness = abs(int(self.WEIGHT_THICKNESS * w / maxw)) + 1

					if (w >= 0):
						cv.line(img, (x1, y1), (x2, y2), self.WEIGHT_COLOR_POSITIVE, thickness)
					else:
						cv.line(img, (x1, y1), (x2, y2), self.WEIGHT_COLOR_NEGATIVE, thickness)



