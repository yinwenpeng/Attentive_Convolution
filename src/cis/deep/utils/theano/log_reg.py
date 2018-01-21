# -*- coding: utf-8 -*-
"""
This file contains a basic logistic regression test. 
http://deeplearning.net/software/theano/tutorial/examples.html?highlight=tanh#a-real-example-logistic-regression
(2013-12-18)
"""

import os.path

import numpy
from theano import config
import theano

from cis.deep.utils import save_object_to_file, load_object_from_file
import theano.tensor as T


rng = numpy.random
 
floatX = config.floatX
 
class LogisticRegression():
 
    def __init__(self):
        self.N = 400
        self.feats = 784
        self.D = (numpy.asarray(rng.randn(self.N, self.feats), floatX),
                numpy.asarray(rng.randint(size=self.N, low=0, high=2), floatX))
        self.training_steps = 10
 
        # Declare Theano symbolic variables
        self.x = T.matrix("x", floatX)
        self.y = T.vector("y", floatX)
        self.w = theano.shared(numpy.asarray(numpy.zeros(self.feats), floatX),
                name="w")
        self.b = theano.shared(numpy.cast[floatX](0.), name="b")
#         print "Initial model:"
#         print self.w.get_value(), self.b.get_value()
 
    def create_graph(self):
 
        # Construct Theano expression graph
#         self.p_1 = theano.printing.Print('p_1')(1 / (1 + T.exp(-T.dot(self.x, self.w) - self.b)))  # Probability that target = 1
#         self.prediction = theano.printing.Print('prediction')(self.p_1 > 0.5)  # The prediction thresholded
#         y = theano.printing.Print('y')(self.y)
#         left = theano.printing.Print('left')(-y * T.log(self.p_1))
#         self.xent = theano.printing.Print('xent')(left - (1 - self.y) * T.log(1 - self.p_1))  # Cross-entropy loss function
#         self.cost = theano.printing.Print('cost')(self.xent.mean() + 0.01 * (self.w ** 2).sum())  # The cost to minimize
#         self.gw, self.gb = T.grad(self.cost, [self.w, self.b])  # Compute the gradient of the cost
        self.p_1 = 1 / (1 + T.exp(-T.dot(self.x, self.w) - self.b))  # Probability that target = 1
        self.prediction = self.p_1 > 0.5  # The prediction thresholded
        self.xent = -self.y * T.log(self.p_1) - (1 - self.y) * T.log(1 - self.p_1)  # Cross-entropy loss function
        self.cost = self.xent.mean() + 0.01 * (self.w ** 2).sum()  # The cost to minimize
        self.gw, self.gb = T.grad(self.cost, [self.w, self.b])  # Compute the gradient of the cost
 
 
        # Compile
        self.train = theano.function(
                  inputs=[self.x, self.y],
                  outputs=[self.prediction, self.xent],
                  updates=((self.w, self.w - 0.1 * self.gw), (self.b, self.b - 0.1 * self.gb)))
        self.predict = theano.function(inputs=[self.x], outputs=self.prediction)
 
    def do_train(self):
 
        # Train
        for i in range(self.training_steps):
            print self.b.get_value(), type(self.b.get_value())
            pred, err = self.train(self.D[0], self.D[1])
 
print floatX


if not os.path.exists('test'):
    print 'create new model'
    model = LogisticRegression()
    model.create_graph()
    model.do_train()
    save_object_to_file(model, 'test')
else:
    print 'load model'
    model = load_object_from_file('test')

print "Final model:"
print model.w.get_value(), model.b.get_value()
print "target values for D:", model.D[1]
print "prediction on D:", model.predict(model.D[0])
 

# import numpy
# import theano
# import theano.tensor as T
# from theano import config
# rng = numpy.random
# floatX = config.floatX
# 
# N = 400
# feats = 784
# D = (numpy.asarray(rng.randn(N, feats), floatX), numpy.asarray(rng.randint(size=N,low=0, high=2), floatX))
# training_steps = 10000
# 
# # Declare Theano symbolic variables
# x = T.matrix("x", floatX)
# y = T.vector("y", floatX)
# w = theano.shared(numpy.asarray(rng.randn(feats), floatX), name="w")
# b = theano.shared(numpy.cast[floatX](0.), name="b")
# print "Initial model:"
# print w.get_value(), b.get_value()
# 
# # Construct Theano expression graph
# p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
# prediction = p_1 > 0.5                    # The prediction thresholded
# xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
# cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
# gw,gb = T.grad(cost, [w, b])              # Compute the gradient of the cost
# 
# # Compile
# train = theano.function(
#           inputs=[x,y],
#           outputs=[prediction, xent],
#           updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
# predict = theano.function(inputs=[x], outputs=prediction)
# 
# # Train
# for i in range(training_steps):
#     pred, err = train(D[0], D[1])
# 
# print "Final model:"
# print w.get_value(), b.get_value()
# print "target values for D:", D[1]
# print "prediction on D:", predict(D[0])