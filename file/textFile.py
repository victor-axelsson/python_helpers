import numpy as np
import glob
import json

class TextFile:

	def read_folder(self, folder):
		content = []
		for filename in glob.iglob(folder + '**/*.txt', recursive=True):
			content += self.read_data(filename)

		return np.array(content)

	def read_data(self, fname):
		with open(fname) as f:
			lines = f.readlines()

		content = [line.strip() for line in lines]

		tmp = []
		for i in range(len(content)):
			pts = content[i].split(",")
			tmp += [pts]

		content = tmp

		return content

	def read_json(self, path):
		with open(path) as data_file:    
		    return json.load(data_file)