# geneticCarsPython

Just a simple mix of cars, neuralnetworks and genetics.
Do not take this work so seriously, it's just a test bench for my personal ideas.

Requirements:
python, opencv, numpy


Based on:

First Read

https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3

Unity Project I was amazed of

https://github.com/ArztSamuel/Applying_EANNs


After all the reading I decided to go on myself based on the project by Samuel Arzt but using Python and OpenCV instead.
I included the LICENSE file from the original as conceptually this works is based on his, my way to give credit.


## simulation

For a given number of iterations, N cars are created.

Cars can move freely through the track with just one purpose: get to the next waypoint.

After all cars die (due to timeout or collision), the best two (cyan and green) are taken to create a new population based on their features. Hopefully these special features, which led them to complete more track than the others, with some mutations can make the new born population complete the 100% of the track or so ;)

## car

Just a car. Steer and throttle is what you can control.
Physics included to make it feel more realistic.

Every car includes a five sensor array in the front measuring the straight distance to the margin of the track.

Data coming from this sensor will be used as an input for the neural network to drive.

## neuralnetwork

3 layers: x5 inputs - x4 outputs, x4 inputs - x3 outputs, x3 inputs - x2 outputs.

Randomly assigned weights and bias at the start. With x5 inputs (distance sensors) and x2 outputs (turn ration and throttle), the neural network should learn how to drive on its own.

You will normally expect training this neuralnetwork with good data about how to drive. However, the approach here is test a new bunch of cars and pick the best. Use the best as a seed for the new generation of cars.


## track

Load the track from a png. You can test your own.
Use green square to point the start location and red squares for the waypoints.

Waypoints must be sorted y-decrementally to solve which one is the next. You know, put one in any x-location but the y-location must be less than the past one.

## user interface

Three windows: one for the track, another one for a neural network representation of the best two and the third one just a zoom for the current best one.
Just the first window is the one needed, the other two are just for fun.

Car progress data is shown on the first window. The best and second-best at the top, the rest are not sorted in any way.

Use 'Q' key to stop and exit the simulation.
Use 'P' key to pause and resume simulation.


