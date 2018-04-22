from helpers.file import textFile
from helpers.wordlist import Wordlist
from sklearn.decomposition import TruncatedSVD
from helpers.logger import logger
from scipy.sparse import lil_matrix
from helpers.cache import Cache
import time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import spatial

class SVD:

	def __init__(self, dataFolder, sessionName):
		self.dataFolder = dataFolder
		self.textFileReader = textFile.TextFile()
		self.wordList = Wordlist()
		self.sessionName = sessionName
		self.logger = logger.Logger()
		self.cache = Cache()
		self.logger.log("Starting a new SVD Session: " + sessionName)

	def getOneHotMatrix(self, data, wordlist):
		nr_cols = len(wordlist)
		self.logger.startLongRunningLog(len(data), 10000)
		counter = 0
		ts = time.time()
		matrix = lil_matrix((len(data), len(wordlist)))
		for session in data:
			row = np.zeros(nr_cols)
			for device in session:
				row[wordlist[device]] += 1

			matrix[counter] = row
			counter += 1
			self.logger.log("")
		self.logger.endLongRunningLog()
		print("matrix: {} r1: {} c1: {}".format(matrix.shape, matrix[0], matrix[0,0]))
		return matrix

	def getOneHotSquareMatrixSession(self, data, wordlist):
		nr_cols = len(wordlist)
		matrix = lil_matrix((len(wordlist), len(wordlist)))

		for dataRow in data:
			prev = None
			row = np.zeros(nr_cols)
			sessionItems = []
			for item in dataRow:
				for sessionItem in sessionItems:
					matrix[wordlist[sessionItem], wordlist[item]] += 1
					matrix[wordlist[item], wordlist[sessionItem]] += 1

				sessionItems.append(item)
				
		return matrix

	def getOneHotMatrixFromGraph(self, data, keys):
		matrix = []
		size = len(keys)
		matrix = lil_matrix((size, size))
		self.logger.startLongRunningLog(size, 100)

		counter = 0
		for i in range(size):
			row = np.zeros(size)
			for j in range(size):
				if keys[i] in data and keys[j] in data[keys[i]]:
					row[j] = data[keys[i]][keys[j]]['weight']

			## Instead of zero add avg as default value ##
			'''
			rowWithMean = np.full(size, np.mean(row))
			for j in range(size):
				if keys[i] in data and keys[j] in data[keys[i]]:
					rowWithMean[j] = data[keys[i]][keys[j]]['weight'] 

			matrix[counter] = rowWithMean
			'''
			matrix[counter] = row
			counter += 1
			self.logger.log("Building graph matrix...")

		self.logger.endLongRunningLog()
		return matrix

	def createOneHotFromName(self, items):
		row = np.zeros(len(self.wl))
		for item in items:
			row[self.wl[item]] += 1
		return row

	def getConceptFromOneHot(self, vector):
		print(self.vt.shape)
		print(self.v.shape)
		print(vector.reshape((1, -1)).shape)

		return np.inner(vector.reshape((1, -1)), self.vt)
		#return np.dot(vector.reshape((1, -1)), self.v)

	def cosineSimilarity(self, v1, v2, n_cols=None, applySigma=True):
		if applySigma:
			return cosine_similarity([np.multiply(v1, self.sigma)], [np.multiply(v2, self.sigma)])[0][0]
		else:
			return cosine_similarity([v1], [v2])[0][0]

	def exportSimilarityUnidirectionalSquareMatrix(self, filepath):
		size = len(self.u)

		self.logger("Exporting nodes...")
		with open(filepath + "/nodes.csv", "w") as text_file:
			text_file.write("Id,Label\n")
			for i in range(size):
					text_file.write("{},{}\n".format(i, self.reversedWl[i]))

		self.logger.startLongRunningLog((size * (size -1) / 2), 10000)
		with open(filepath + "/edges.csv", "w") as text_file:
			text_file.write("Source,Target,Weight\n")
			for i in range(size):
				concepts_i = self.u[i]
				for j in range(i+1, size):
					concepts_j = self.u[j]
					sim = self.cosineSimilarity(concepts_i, concepts_j)

					if(sim > 0.5):
						text_file.write("{},{},{}\n".format(i,j,sim))
					self.logger("Saving edges...")


	def getMostSimilarInU(self, concepts, threshold=0.9):
		best = 0
		bestRow = None
		session = -1
		counter = 0
		items = []

		uWithSigma = np.matmul(self.u, np.diag(self.sigma))
		similarities = cosine_similarity([np.multiply(concepts, self.sigma)], uWithSigma)[0]
		for similarity in similarities:
			if similarity > best:
				best = similarity
				bestRow = -1
				session = self.reversedWl[counter]

			if similarity > threshold:
				#print("{} SIM:{}".format(self.reversedWl[counter], similarity))
				
				items.append({
					'item': self.reversedWl[counter],
					'score': similarity
				})
				
			counter += 1
		return session, bestRow, best, items

	def getHighestIndex(self, vector, nr):
		highest = []
		for i in range(len(vector)):
			highest.append({'index': i, 'val':(vector[i] * self.sigma[i])})

		newlist = sorted(highest, key=lambda k: k['val'], reverse=True) 
		indexes = []
		for i in range(nr):
			indexes.append(newlist[i]['index'])

		return indexes

	def selectMaxFromVFromColumns(self, columns):
		best = -1
		bestItem = None
		counter = 0

		for item in self.v:
			score = 0
			for c in columns:
				score += item[c] * self.sigma[c]

			if score > best:
				best = score
				bestItem = self.reversedWl[counter]

			counter += 1

		return bestItem, score

	def getWordlist(self):
		return self.wl

	def getSigma(self):
		return self.sigma.tolist()

	def visualize(self):

		x1 = self.u[:,0]
		y1 = self.u[:,1]
		z1 = self.u[:,2]

		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(x1, y1, z1)
		plt.show()

		'''
		x1 = np.multiply(self.u[:,0], self.sigma[0])
		y1 = np.multiply(self.u[:,1], self.sigma[1])
		z1 = np.multiply(self.u[:,2], self.sigma[2])

		x2 = np.multiply(self.v[:,0], self.sigma[0])
		y2 = np.multiply(self.v[:,1], self.sigma[1])
		z2 = np.multiply(self.v[:,2], self.sigma[2])
		'''
		'''
		x1 = self.u[:,0]
		y1 = self.u[:,1]
		z1 = self.u[:,2]

		
		x2 = self.v[:,0]
		y2 = self.v[:,1]
		z2 = self.v[:,2]
		
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(x1, y1, z1)
		ax.scatter(x2, y2, z2)
		plt.show()
		'''

		'''
		y = self.sigma
		x = range(len(self.sigma))
		plt.plot(x, y)
		plt.show()
		'''

	def getItemFromV(self, itemKey):
		if itemKey in self.wl:
			index = self.wl[itemKey]
			return np.multiply(self.v[index], self.sigma).tolist(), True
		else:
			return None, False

	def runSvdOnJsonGraph(self, n_components, n_iter):
		self.logger.log("Loading data...")
		self.data = self.textFileReader.read_json(self.dataFolder)
		self.logger.log("Generating wordlist...")
		items, wl, reversedWl = self.wordList.getWordlistFromGraph(self.data)
		self.wl = wl
		self.reversedWl = reversedWl
		matrix, wasCached = self.cache.lazyCache(self.sessionName + "device_device_adjecency.matrix", self.getOneHotMatrixFromGraph, {'data':self.data, 'keys': items})
		if wasCached:
			self.logger.log("Loaded matrix from cache")

		self._runSvd(matrix, n_components, n_iter)
		
	def _runSvd(self, matrix, n_components, n_iter):

		#Load the factorization from disk
		self.u, uWasCached = self.cache.loadNPIfExists(self.sessionName + "u.bin")
		self.sigma, sigmaWasCached = self.cache.loadNPIfExists(self.sessionName + "sigma.bin")
		self.vt, vtWasCached = self.cache.loadNPIfExists(self.sessionName + "vt.bin")
		self.v, vWasCached = self.cache.loadNPIfExists(self.sessionName + "v.bin")

		#If any of the cached matrices was missing
		if not uWasCached or not sigmaWasCached or not vtWasCached and not vWasCached:
			self.logger("Factorization was not present, calculating... (Might take a while)")
			self.logger("Fitting the randomized_svd with {} iterations and {} components".format(n_iter, n_components))
			U, Sigma, VT = randomized_svd(matrix,  n_components=n_components, n_iter=n_iter)

			self.sigma = Sigma
			self.u = U
			self.vt = VT
			self.v = np.transpose(self.vt)

			self.u.dump(self.sessionName + "u.bin")
			self.sigma.dump(self.sessionName + "sigma.bin")
			self.vt.dump(self.sessionName + "vt.bin")
			self.v.dump(self.sessionName + "v.bin")
		else:
			self.logger("Loaded factorization from disk")

		self.logger(matrix.shape)
		self.logger(self.vt.shape)
		self.logger(self.sigma.shape)

		self.logger(self.u.shape)
		self.logger(self.v.shape)


	def runSvdOnCsv(self, n_components, n_iter):
		self.logger.log("Loading data...")
		self.data = self.textFileReader.read_folder(self.dataFolder)
		self.logger.log("Generating wordlist...")
		items, wl, reversedWl = self.wordList.getWordlist(self.data)
		self.wl = wl
		self.reversedWl = reversedWl
		matrix, wasCached = self.cache.lazyCache(self.sessionName + "session_device_adjecency.matrix", self.getOneHotSquareMatrixSession, {'data':self.data, 'wordlist': wl})
		if wasCached:
			self.logger.log("Loaded matrix from cache")

		self._runSvd(matrix, n_components, n_iter)


