# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:19:59 2017

@author: Chengch
"""

import gzip
import cPickle
import numpy as np
import random
import matplotlib
import matplotlib.pylab as plt

eta = 0.3
epcoes = 30

def vectorized_result(y):
    z = np.zeros((10,1))
    z[y] = 1.0 
    return z

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z))
    
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
def cost_derivative(output_activations,y):
    return (output_activations - y )

f = gzip.open('E:/mnist.pkl.gz','rb')
tr_d, va_d, tr_d = cPickle.load(f)
f.close()

training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
traing_data = zip(training_inputs, training_results)
validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
validation_data = zip(validation_inputs, va_d[1])
test_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
test_data = zip(test_inputs, tr_d[1])


network = ([784,30,10])
num_layers = len(network)
biases = [np.random.randn(y,1) for y in network[1:]]
weights = [np.random.randn(y,x) for x,y in zip(network[:-1],network[1:])]
           
nabla_b = [np.zeros(b.shape) for b in biases]
nabla_w = [np.zeros(w.shape) for w in weights]

sizes = 30
x_labels = []
y_labels = []
n = len(traing_data)
for j in xrange(epcoes):
    random.shuffle(traing_data)
    mnist_batchs =[traing_data[k : k + sizes] for k in xrange(0,n,sizes)] 
    for mnist_batch in mnist_batchs:
        for x, y in mnist_batch:
            activation = x
            y = y
            activations = [x] 
            zs = []
            for b, w in zip(biases, weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            for l in xrange(2,num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            delta_nabla_b, delta_nabla_w = nabla_b,nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            weights = [w-(eta)*nw for w, nw in zip(weights, nabla_w)]
            biases = [b-(eta)*nb for b, nb in zip(biases, nabla_b)]
    i = 0  
    for x,y in test_data:
        activation = x
        activations = [x]
        for b ,w in zip(biases,weights):
            z = np.dot(w,activation) + b
            activation = sigmoid(z)
            y_ = np.argmax(activation)
        if y == y_:
            i += 1
    x_labels.append(j)
    y_labels.append(float(i) / float(len(test_data)) * 100)
    print 'Epoch{0} had complete, and the accuracy rate is {1}%'.format(j + 1,float(i)/ float(len(test_data)) * 100)

fig = plt.figure(figsize=(9,6))

matplotlib.rc('xtick',labelsize=14)
matplotlib.rc('ytick',labelsize=14)

ax1 = plt.subplot(111)
ax1.plot(x_labels,y_labels,'b-',linewidth=2.5)
plt.ylabel('Accuracy Rate / % ',size=18)
plt.xlabel('Epoch',size=18)
plt.savefig('2.png',dpi=300)
plt.show()