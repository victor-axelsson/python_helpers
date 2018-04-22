import matplotlib.pyplot as plt
import seaborn as sns
from helpers.file import textFile
import numpy as np

class Deviation:

	reader = None

	def __init__(self):
		self.reader = textFile.TextFile()

	def plotAndShowFromJson(self, filePath):
		data = self.reader.read_json(filePath)
		x = []
		for key in data:
			x.append(data[key])

		arr = np.array(x)
		print('%f' % np.std(arr, axis=0))
		#print(str(float()).format('.8f'))
		sns.distplot(x, bins=1000, color='red');
		plt.show()
