import pickle
import os.path
import numpy as np

class Cache: 

	def pickleIt(self, variable, name):
		pickle.dump(variable, open( name, "wb" ))

	def loadPickle(self, name):
		return pickle.load(open( name, "rb" ))

	def loadIfExists(self, name):
		if os.path.isfile(name):
			return self.loadPickle(name), True
		else:
			return None, False

	def loadNPIfExists(self, name):
		if os.path.isfile(name):
			return np.load(name), True
		else:
			return None, False

	def lazyCache(self, name, callable, args=None):
		if os.path.isfile(name):
			return self.loadPickle(name), True
		else:
			data = callable(**args)
			self.pickleIt(data, name)
			return data, False
