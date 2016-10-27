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
import pickle


class RBM():

	def Initialization(self, args):

		self.NumOfv  = args.visible													# num of input
		self.NumOfh  = args.hidden													# num of hidden layer
		self.rate        = args.rate          										# learning rate
		self.NumOfEpoch  = args.epoch												# num of epoch	
		self.k 				= args.ksteps
		c = np.sqrt(6)/np.sqrt(self.NumOfv + self.NumOfh)
		self.w = np.random.uniform(-c, c, [self.NumOfv, self.NumOfh])
		self.hbias = np.zeros(self.NumOfh)
		self.vbias = np.zeros(self.NumOfv)
		self.train_loss = []
		self.valid_loss = []
		self.sample = []
		
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


	def CDK(self, v0, k):
		
		h0_sample = self.Sample_h_given_v(v0)
		h_sample = h0_sample
		while (k>0):
			activation_v, v_sample = self.Sample_v_given_h(h_sample)
			h_sample = self.Sample_h_given_v(v_sample)
			k -= 1

		error = v0*np.log(activation_v)+(1-v0)*np.log(1-activation_v)
		error = -sum(error.transpose())

		return v0, h0_sample, v_sample, h_sample, error
	
	
	def Update(self, v0, h0_sample, v_sample, h_sample):
		a = np.dot(v0.reshape(self.NumOfv,1), h0_sample.reshape(1,self.NumOfh))
		b = np.dot(v_sample.reshape(self.NumOfv,1), h_sample.reshape(1, self.NumOfh))
		self.w += self.rate*(a - b)
		self.hbias += self.rate*(h0_sample - h_sample)
		self.vbias += self.rate*(v0 - v_sample)

		
	def Train(self, train_list):	
		sum_error = 0	
		for l in train_list:
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			inputs = np.array(map(float, line))
			inputs = inputs>0.5
			inputs = inputs.astype(int)

			v0, h0_sample, v_sample, h_sample, error = self.CDK(inputs, self.k)
			self.Update(v0, h0_sample, v_sample, h_sample)
			sum_error += error

		return sum_error
	
	
	def Valid(self, valid_list):
		sum_error = 0	
		for l in valid_list:
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			inputs = np.array(map(float, line))
			inputs = inputs>0.5
			inputs = inputs.astype(int)

			v0, h0_sample, v_sample, h_sample, error = self.CDK(inputs, self.k)
			sum_error += error

		return sum_error
		
		
		
	def Sample(self, NumOfSamples):
		
		for i in range(NumOfSamples):
			sample_img = np.random.uniform(0, 1, 784)
			sample_img = sample_img > 0.5
			sample_img = sample_img.astype(int)
			
			v0, h0_sample, v_sample, h_sample, error = self.CDK(sample_img, 1000)
			
			self.sample.append(v_sample)
		
		
		
		
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
			
			train_error = self.Train(train_list)/self.NumOfTrain	
			valid_error = self.Valid(valid_list)/self.NumOfValid
			self.train_loss.append(train_error)
			self.valid_loss.append(valid_error)
			print "epoch--", n, "train_error: ", train_error, "valid_error: ", valid_error
			n += 1
			
		print "training finished!"
		weight = self.w
		f = open('rbm.pickle', 'wb')
		pickle.dump(weight, f)
		print "start sampling"
		self.Sample(100)
		
		
	def Plot(self):
		t = np.arange(0, self.NumOfEpoch, 1)
		plt.plot(t, self.train_loss, 'r--', t, self.valid_loss, 'b--')
		plt.show()
		

		
	def PlotWeight(self):

		for i in range(self.NumOfh):
			fig = plt.subplot(10,10,i)
			fig.axes.get_xaxis().set_visible(False)
			fig.axes.get_yaxis().set_visible(False)
			a = self.w[:,i]
			mina = min(a)
			maxa = max(a)
			a = 255*(a - mina)/(maxa-mina)
			fig.imshow(a.reshape(28,28), cmap='gray')
	
		plt.show()
		
		
		
	def PlotSample(self):
		for i in range(100):
			fig = plt.subplot(10,10,i)
			fig.axes.get_xaxis().set_visible(False)
			fig.axes.get_yaxis().set_visible(False)
			fig.imshow(self.sample[i].reshape(28,28), cmap='gray')
	
		
		plt.show()
		
		

if __name__ == "__main__":

	start_time = time.time()
	parser = argparse.ArgumentParser(description='script for testing')
	parser.add_argument('filename', nargs='+')
	parser.add_argument('--rate', '-r', type=float, default=0.01, help='The learning rate')
	parser.add_argument('--epoch', '-e', type=int, default=50, help='the number of epoch')
	parser.add_argument('--visible', '-v', type=int, default=784, help='the number of visible units')
	parser.add_argument('--hidden', type=int, default=100, help='the number of hidden units')
	parser.add_argument('--ksteps','-k', type=int, default=1, help='the number of k steps')
	args = parser.parse_args()
	RBMachine = RBM()
	RBMachine.Main(args)
	print("--- %s seconds ---" % (time.time() - start_time))
	RBMachine.Plot()
	RBMachine.PlotWeight()
	RBMachine.PlotSample()

	

