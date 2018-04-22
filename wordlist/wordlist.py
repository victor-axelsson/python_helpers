import numpy as np

class Wordlist:
	def getWordlist(self, data):
		items = set()
		for row in data:
			for col in row:
				items.add(col)

		wordlist = dict()
		reversedList = []

		counter = 0
		for item in items:
			wordlist[item] = counter
			reversedList.append(item)
			counter += 1

		return items, wordlist, reversedList

	def getWordlistFromGraph(self, data):
		allItems = {}
		for k in data:
			allItems[k] = 1
			for innerKey in data[k]:
				allItems[innerKey] = 1 

		items = np.array(list(allItems.keys()))

		reversedList = []
		itemsKeys = {}
		for i in range(len(items)):
			itemsKeys[items[i]] = i
			reversedList.append(items[i])

		return items, itemsKeys, reversedList