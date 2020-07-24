# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import random
import codecs
import numpy as np
import cv2
from SamplePreprocessor import preprocess


class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:
	"loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

	def __init__(self, filePath, batchSize, imgSize, maxTextLen):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'  # should end with '/'

		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []
		# use your own path and trick to read the images with the txt file
		with codecs.open(filePath+'full.txt', "r", encoding='utf-8') as f:
			lines = f.readlines()
		lines = [x.strip() for x in lines] # removing newline
		chars = set()
		print(lines[2])
		for line in lines:
			# ignore comment line
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split(' ')

			if lineSplit[0] == '\ufeff':  # since reading first line give '\ufeff'
				continue

			fileName = filePath + lineSplit[0]

			# GT text are columns starting at 1
			gtText = self.truncateLabel(' '.join(lineSplit[1]), maxTextLen)
			chars = chars.union(set(list(gtText)))

			# check if image is not empty
			if not os.path.getsize(fileName):
				#bad_samples.append(lineSplit[0] + '.jpg')
				continue

			# put sample into list
			self.samples.append(Sample(gtText, fileName))



		# split into training and validation set: 95% - 5%
		splitIdx = int(0.95 * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		# put words into lists
		self.trainWords = [x.gtText for x in self.trainSamples]
		self.validationWords = [x.gtText for x in self.validationSamples]

		# number of randomly chosen samples per epoch for training 
		self.numTrainSamplesPerEpoch = 25000 
		
		# start with train set
		self.trainSet()

		# list of all chars in dataset
		self.charList = sorted(list(chars))
		# print("TrainSet: {}, TestSet: {}".format(len(self.trainSamples), len(self.validationSamples)))
		# print("Total chars: ",len(self.charList))  # length: 109
		# charList = ''.join([str(x) for x in self.charList])
		# print(charList)
		# codecs.open('../model/charList.txt','w', encoding='utf-8').write(charList)

	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text

	def trainSet(self):
		"switch to randomly chosen subset of training set"
		self.dataAugmentation = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

	def validationSet(self):
		"switch to validation set"
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples

	def getIteratorInfo(self):
		"current batch index and overall number of batches"
		return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

	def hasNext(self):
		"iterator"
		return self.currIdx + self.batchSize <= len(self.samples)
		
	def getNext(self):
		"iterator"
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		gtTexts = [self.samples[i].gtText for i in batchRange]
		imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTexts, imgs)


#DataLoader('../data/', 50, (128,32), 32)
