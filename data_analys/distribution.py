from helpers.file import textFile
from helpers.cache import Cache

class Distribution:
	dataFolder = None
	textFileReader = None
	cache = None
	wordlist = None

	def __init__(self, dataFolder):
		self.dataFolder = dataFolder
		self.textFileReader = textFile.TextFile()
		self.cache = Cache()
		self.calcDistribution()

	def _readDataFromDisk(self):
		return self.textFileReader.read_folder(self.dataFolder)

	def _readSortedList(self, data):
		items = {}
		for row in data:
			for col in row:
				if col not in items:
					items[col] = 0
				items[col] += 1


		itemList = []
		for item in items.keys():
			itemList.append({
				'item': item,
				'val': items[item]
			})
			
		return sorted(itemList, key=lambda k: k['val'], reverse=True)

	def _readWordList(self):
		wordlist = {}
		for i in range(len(self.sortedList)):
			item = self.sortedList[i]
			wordlist[item['item']] = i

		return wordlist

	def calcDistribution(self):
		data, wasCached = self.cache.lazyCache("distribution.pkl", self._readDataFromDisk, {})
		if wasCached:
			print("Loaded distribution from cache")

		items = {}
		for row in data:
			for col in row:
				if col not in items:
					items[col] = 0
				items[col] += 1


		itemList = []
		for item in items.keys():
			itemList.append({
				'item': item,
				'val': items[item]
			})
			
		self.sortedList, sortedWasCached = self.cache.lazyCache("sortedDist.pkl", self._readSortedList, {'data': data})
		if sortedWasCached:
			print("Loaded sortedList from cache")
		self.wordlist, wordListWasCached = self.cache.lazyCache("wordListDist.pkl", self._readWordList, {})
		if wordListWasCached:
			print("Loaded wordlist from cache")

	def getDistribution(self):
		return self.sortedList, self.wordlist
