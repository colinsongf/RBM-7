## Backpropagation for Neural Network ##
## Shiyu Dong
## shiyud@andrew.cmu.edu

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import argparse
import random
class NeuralNetwork():

	def Initialization(self, args):

		self.NumOfLayer  = len(args.layer)						# num of layers
		self.NumOfUnits  = args.layer									# units for each layer, e.g:[784, 100, 10]
		self.rate        = args.rate          				# learning rate
		self.NumOfEpoch  = args.epoch									# num of epoch
		self.train_loss  = []													# cross entropy for training set
		self.valid_loss  = []													# cross entropy for validation set
		self.train_err   = []													# classification error for training set
		self.valid_err   = []													# classification error for validation set
		self.test_loss   = []
		self.test_err    = []
		self.p           = args.dropout								# dropout probability
		self.alpha       = args.momentum							# momentum
		self.w = []																		# w
		self.dw = []																	# dw
		self.b = []																		# bias
		self.db = []
		for i in range(self.NumOfLayer-1):
				c = np.sqrt(6)/np.sqrt(self.NumOfUnits[i+1] + self.NumOfUnits[i])
				self.w.append(np.random.uniform(-c, c, [self.NumOfUnits[i+1], self.NumOfUnits[i]]))
				self.dw.append(np.zeros(self.w[i].shape))
				self.b.append(np.random.uniform(-c, c, self.NumOfUnits[i+1]))
				self.db.append(np.zeros(self.b[i].shape))	
				
	def TrainFeedForward(self, inputs):
		activation = []				
		for i in range(self.NumOfLayer-1):							
			x = inputs if (i==0) else a
			z = np.dot(self.w[i], x)+ self.b[i]
			mask = np.random.rand(*z.shape) < self.p 
			a = self.Softmax(z) if (i==self.NumOfLayer-2) else self.Sigmoid(z)*mask			
			activation.append(a)	
		return activation
	
	def ValidFeedForward(self, inputs):
		activation = []				
		for i in range(self.NumOfLayer-1):							
			x = inputs if (i==0) else a
			z = np.dot(self.w[i], x)+ self.b[i]
			a = self.Softmax(z) if (i==self.NumOfLayer-2) else self.Sigmoid(z)*self.p			
			activation.append(a)	
		return activation		
		
	def BackProp(self, inputs, activation, target):
		
#		for i in range(self.NumOfLayer-1):
#			n = self.NumOfLayer -2 -i 											# the layer that is being propagated
#			delta = np.dot(delta, self.w[n+1])*activation[n]*(1-activation[n]) if (i!=0) else target - activation[-1]
#			x = activation[n-1] if (n!=0) else inputs
#			#dw = np.tile(delta, (self.NumOfUnits[n],1)).transpose()*np.tile(x, (self.NumOfUnits[n+1],1))*self.rate + self.alpha*self.dw[n]
#			self.dw[n] += np.tile(delta, (self.NumOfUnits[n],1)).transpose()*np.tile(x, (self.NumOfUnits[n+1],1))*self.rate + self.alpha*self.dw[n]
#			#self.w[n] += dw
#			#self.dw[n] += dw
#			self.db[n] += delta
#		
#		self.w += self.dw
#		self.b += self.db
		delta2 = target - activation[-1]
			
		## input -> hidden
		delta1 = np.dot(delta2, self.w[1])*activation[0]*(1-activation[0])

		## update w1 & w2
		# for x2 in range(self.w2.shape[0]):
		# 	for y2 in range(self.w2.shape[1]): 
		# 		self.w2[x2][y2] += deltak[x2]*a1[y2]*self.rate 

		self.w[1] += np.tile(delta2, (self.NumOfUnits[1],1)).transpose()*np.tile(activation[0], (self.NumOfUnits[2], 1))*self.rate + self.alpha*self.dw[1]
		self.dw[1] = np.tile(delta2, (self.NumOfUnits[1],1)).transpose()*np.tile(activation[0], (self.NumOfUnits[2], 1))*self.rate + self.alpha*self.dw[1]


		# for x1 in range(self.w1.shape[0]): 
		# 	for y1 in range(self.w1.shape[1]):
		# 		#print deltaj[x1]*inputs[y1]*self.rate
		# 		self.w1[x1][y1] += deltaj[x1]*inputs[y1]*self.rate

		self.w[0] += np.tile(delta1, (self.NumOfUnits[0],1)).transpose()*np.tile(inputs, (self.NumOfUnits[1], 1))*self.rate + self.alpha*self.dw[0]
		self.dw[0] = np.tile(delta1, (self.NumOfUnits[0],1)).transpose()*np.tile(inputs, (self.NumOfUnits[1], 1))*self.rate + self.alpha*self.dw[0]

		self.b[1] += delta2
		self.b[0] += delta1


	def Train(self, train_list):
		loss = 0
		error = 0	
			
		for l in train_list:
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			
			inputs = np.array(map(float, line))
			a = self.TrainFeedForward(inputs)
			if (a[-1].argmax()!=target):
				error += 1
				
			t = np.zeros(self.NumOfUnits[-1])
			t[target] = 1
			
			loss += -self.Loss(t, a[-1])	
			self.BackProp(inputs, a, t)
					
		return loss/self.NumOfTrain, error/float(self.NumOfTrain)
	
	def Valid(self, valid_list, length):
	
		loss = 0
		error = 0
		for l in valid_list:
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			
			inputs = np.array(map(float, line))
			a = self.ValidFeedForward(inputs)
			if (a[-1].argmax()!=target):
				error += 1
			
			t = np.zeros(self.NumOfUnits[-1])
			t[target] = 1
			loss += -self.Loss(t, a[-1])
		
		return loss/length, error/float(length)
		

	def Main(self, args):
	
		self.Initialization(args)
		
		train_list = open(args.filename[0]).readlines()
		valid_list = open(args.filename[1]).readlines()
		test_list  = open(args.filename[2]).readlines()
		
		self.NumOfTrain = len(train_list)
		self.NumOfValid = len(valid_list)
		self.NumOfTest  = len(test_list)
		
		n = 0
		
		while(n < self.NumOfEpoch):
				
			loss_v, error_v = self.Valid(valid_list, self.NumOfValid)
			print "validation error", loss_v, error_v
			self.valid_loss.append(loss_v)
			self.valid_err.append(error_v)
			
			loss_t, error_t = self.Valid(test_list, self.NumOfTest)
			print "test error", loss_t, error_t
			self.test_loss.append(loss_t)
			self.test_err.append(error_t)
			
			random.shuffle(train_list)
			loss, error = self.Train(train_list)
			print "training error", loss, error
			self.train_loss.append(loss)
			self.train_err.append(error)
				
			n += 1 
		
		print "training finished!"


	def Sigmoid(self, z):
		a = 1/(1 + np.exp(-z))
		return a


	def Softmax(self, z):
		a = np.exp(z)
		return a/sum(a)


	def Loss(self, t, a):		
		return sum(t*np.log(a))
		
		
	def Plot(self):
		t = np.arange(0, self.NumOfEpoch, 1)
#		plt.plot(t, self.train_loss, 'r--', t, self.valid_loss, 'b--')
#		plt.show()
#		plt.plot(t, self.train_err, 'r--', t, self.valid_err, 'b--')
#		plt.show()
		plt.plot(t, self.train_loss, 'r--', t, self.valid_loss, 'b--', t, self.test_loss, 'g--')
		plt.show()
		plt.plot(t, self.train_err, 'r--', t, self.valid_err, 'b--', t, self.test_err, 'g--')
		plt.show()

		
	def PlotWeight(self):
	
		a = int(np.sqrt(self.NumOfUnits[1]))	
		b = int(np.sqrt(self.NumOfUnits[0]))
		plotimg = np.zeros([a*b, a*b])
		for x in range(plotimg.shape[0]):
			for y in range(plotimg.shape[1]):
				plotimg[x][y] = self.w[0][a*(x/b) + y/b][b*(x%b) + y%b]
				
		plt.imshow(plotimg, cmap='gray')
		plt.show()
				

if __name__ == "__main__":

	start_time = time.time()
	parser = argparse.ArgumentParser(description='script for testing')
	parser.add_argument('filename', nargs='+')
	parser.add_argument('--dropout', '-d', type=float, default=1, help='the dropout vallues')
	parser.add_argument('--rate', '-r', type=float, default=0.1, help='The learning rate')
	parser.add_argument('--epoch', '-e', type=int, default=200, help='the number of epoch')
	parser.add_argument('--momentum', '-m', type=float, default=0, help='momentum parameter')
	parser.add_argument('--layer', '-l', type=int, nargs='+', default =(784, 100, 10), help='the number of units for each layer')
	args = parser.parse_args()
	NN = NeuralNetwork()
	NN.Main(args)
	NN.Plot()
	NN.PlotWeight()
	print("--- %s seconds ---" % (time.time() - start_time))

	
