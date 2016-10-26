## Restricted Boltzmann Machine ##
## Shiyu Dong
## shiyud@andrew.cmu.edu

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import argparse
import random


class RBM():

	def Initialization(self, args):

		self.NumOfv  = args.visible													# num of input
		self.NumOfh  = args.hidden													# num of hidden layer
		self.rate        = args.rate          										# learning rate
		self.NumOfEpoch  = args.epoch												# num of epoch	
		c = np.sqrt(6)/np.sqrt(self.NumOfv + self.NumOfh)
		self.w = np.random.uniform(-c, c, [self.NumOfv, self.NumOfh])
		self.hbias = np.zeros(self.NumOfh)
		self.vbias = np.zeros(self.NumOfv)
		self.sum_error = 0
		
	def Sigmoid(self, z):
		a = 1/(1 + np.exp(-z))
		return a


	def Sample_h_given_v(self, v):

		pre_activation_h = np.dot(v, self.w) + self.hbias
		activation_h = self.Sigmoid(pre_activation_h)
		p = np.random.uniform(0,1, self.NumOfh) 
		h_sample =  p < activation_h
		h_sample = h_sample.astype(int)
		
		return h_sample


	def Sample_v_given_h(self, h):
		
		pre_activation_v = np.dot(self.w, h) + self.vbias
		activation_v = self.Sigmoid(pre_activation_v)
		
		
		p = np.random.uniform(0,1,self.NumOfv) 
		v_sample =  p <= activation_v
		v_sample = v_sample.astype(int)
		
		
		return activation_v, v_sample


	def CDK(self, v0, k=1):
		
		h0_sample = self.Sample_h_given_v(v0)
		h_sample = h0_sample
		while (k>0):
			activation_v, v_sample = self.Sample_v_given_h(h_sample)
			h_sample = self.Sample_h_given_v(v_sample)
			k -= 1

		error = v0*np.log(activation_v)+(1-v0)*np.log(1-activation_v)

		self.sum_error += -sum(error.transpose())
		a = np.dot(v0.reshape(self.NumOfv,1), h0_sample.reshape(1,self.NumOfh))
		b = np.dot(v_sample.reshape(self.NumOfv,1), h_sample.reshape(1, self.NumOfh))
		self.w += self.rate*(a - b)
		self.hbias += self.rate*(h0_sample - h_sample)
		self.vbias += self.rate*(v0 - v_sample)


	def Train(self, train_list):	
			
		for l in train_list:
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			inputs = np.array(map(float, line))
			inputs = inputs>0.5
			inputs = inputs.astype(int)

			self.CDK(inputs, k=2)

		

	def Main(self, args):
	
		self.Initialization(args)

		train_list = open(args.filename[0]).readlines()
		valid_list = open(args.filename[1]).readlines()
		test_list  = open(args.filename[2]).readlines()

		self.NumOfTrain = len(train_list)
		self.NumOfValid = len(valid_list)
		self.NumOfTest  = len(test_list)
		
		n = 1
		
		while(n <= self.NumOfEpoch):
			self.sum_error = 0
			self.Train(train_list)	
			print "epoch--", n, "error: ", self.sum_error/float(3000)
			n += 1 
		
		self.PlotWeight()
		print "training finished!"

		
		
	def Plot(self):
		t = np.arange(0, self.NumOfEpoch, 1)
		plt.plot(t, self.train_loss, 'r--', t, self.valid_loss, 'b--', t, self.test_loss, 'g--')
		plt.show()
		plt.plot(t, self.train_err, 'r--', t, self.valid_err, 'b--', t, self.test_err, 'g--')
		plt.show()

		
	def PlotWeight(self):

		for i in range(self.NumOfh):
			fig = plt.subplot(10,10,i)
			fig.axes.get_xaxis().set_visible(False)
			fig.axes.get_yaxis().set_visible(False)
			fig.imshow(self.w[:,i].reshape(sqrt(self.NumOfv),sqrt(self.NumOfv)), cmap='gray')
	
		
		plt.show()
				

if __name__ == "__main__":

	start_time = time.time()
	parser = argparse.ArgumentParser(description='script for testing')
	parser.add_argument('filename', nargs='+')
	parser.add_argument('--rate', '-r', type=float, default=0.01, help='The learning rate')
	parser.add_argument('--epoch', '-e', type=int, default=50, help='the number of epoch')
	parser.add_argument('--visible', '-v', type=int, default=784, help='the number of visible units')
	parser.add_argument('--hidden', type=int, default=100, help='the number of hidden units')
	args = parser.parse_args()
	RBMachine = RBM()
	RBMachine.Main(args)
	print("--- %s seconds ---" % (time.time() - start_time))

	

