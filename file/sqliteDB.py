class SqliteDB:
	db = None

	def __init__(self, db_name='similarities.db'):
		self.db = sqlite3.connect(db_name)

	def close(self):
		self.db.close()

	def execute(self, query):
		cursor = self.db.cursor()
		cursor.execute(query)

	def commit(self):
		self.db.commit()

	def createTables(self):
		query = '''
			CREATE TABLE similarities(
				id INTEGER PRIMARY KEY, 
				from_device TEXT not null,
				to_device TEXT not null,
				similarity REAL not null
			)
			'''
		self.execute(query)
		self.commit()

	def insertSimilarity(self, sim):
		cursor = self.db.cursor()
		query = '''
			INSERT INTO similarities (from_device, to_device, similarity) 
			VALUES (:from_device, :to_device, :similarity)
		'''
		'''
		v1 = ",".join(str(v) for v in sim[3]) 
		v2 = ",".join(str(v) for v in sim[4])
		'''
		
		if math.isnan(sim[2]):
			sim[2] = 0

		cursor.execute(query, {'from_device': str(sim[0]), 'to_device': sim[1], 'similarity': sim[2]})