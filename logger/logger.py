import time
import datetime
import psutil


class Logger: 

	longRunning = False

	def __call__(self, message):
		self.log(message)

	def startLongRunningLog(self, totalIterations, tick):
		self.totalIterations = totalIterations
		self.tick = tick
		self.longRunning = True
		self.counter = 0
		self.ts = time.time()

	def endLongRunningLog(self):
		self.totalIterations = 0
		self.tick = 0
		self.longRunning = False
		self.counter = 0
		self.ts = time.time()

	def _logLongRunning(self, message):
		if(self.counter % self.tick == 0):
			diff = time.time() - self.ts 
			msPerTick = diff / self.tick
			remaining = msPerTick * (self.totalIterations - self.counter)
			self.ts = time.time()
			mem = psutil.virtual_memory().percent
			cpu = psutil.cpu_percent()

			print("{}, [CPU]=>{} [MEM]=>{} [ETA]=>{}, {}/{}=>{:.2} {}".format(datetime.datetime.fromtimestamp(self.ts).strftime('%Y-%m-%d %H:%M:%S'), cpu, mem, datetime.datetime.fromtimestamp(time.time() + remaining).strftime('%Y-%m-%d %H:%M:%S'), self.counter, self.totalIterations, self.counter / self.totalIterations, message))
		self.counter += 1

	def log(self, message):
		if(self.longRunning):
			self._logLongRunning(message)
		else:
			print("{} {}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), message))
