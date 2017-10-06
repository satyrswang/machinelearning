#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:58:18 2017

@author: wyq
"""

from numpy import *
#import matplotlib
import pandas as pd
#from sklearn import cross_validation


def run():
    file_dir = "/Users/wyq/Downloads/data.csv"
    delimiter=","
   # data = getfromtxt(file_dir,delimiter)
    data =pd.read_csv(file_dir, delimiter, encoding='us-ascii',header = None)
    input = data[0]
    output = data[1]
    initial_b = 0 # initial y-intercept guess
    initial_w = 0 # initial slope guess
    learning_rate = 0.0001
    num_iterations = 1000
    
    
    [b, w] = gradient_descent_runner(input,output, initial_b, initial_w, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, w = {2}, error = {3}".format(num_iterations, b, w, error(input,output,b, w))


 
#def corss_vali(data):    
#    training = 
#    testing = 
        

def error(input,output, b,w):
    totalError = 0
    for i in range(0, len(input)):
        x = input[i]
        y = output[i]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(input))
    
#def deri_func(loss_func):
    
def step_gradient(input,output,current_b,current_w,learn_rate,num_it):
    b_gradient = 0
    w_gradient = 0
    N = len(input)
    for i in range(0, len(input)):
        x = input[i]
        y = output[i]
        b_gradient += -(2/N) * (y - ((current_w * x) + current_b))
        w_gradient += -(2/N) * x * (y - ((current_w * x) + current_b))
    new_b = current_b - (learn_rate * b_gradient)
    new_w = current_w - (learn_rate * w_gradient)
    return [new_b, new_w]
    
def gradient_descent_runner(input,output, starting_b, starting_w, learn_rate, num_it):
    b = starting_b
    w = starting_w
    for i in range(num_it):
        b, w = step_gradient(input,output,b, w,learn_rate,num_it)
    return [b, w]

if __name__ == '__main__':
    run()
        
