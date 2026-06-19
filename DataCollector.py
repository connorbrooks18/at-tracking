import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class DataCollector:
	
	def __init__(self):
		self.rows = []
		# row.append({ 'milli': time, 'name': 'apple', 'x': x, 'y': y, 'z': z, qx, qy, qz, qw})

	def dump(self, filename="output.parquet"):
		df = pd.DataFrame(self.rows)
		table = pa.Table.from_pandas(df)
		pq.write_table(table, filename)

	def read(self, filename="output.parquet"):
		return pq.read_table(filename)
	
	def print(self):
		print(self.rows)
		
	def update(self, milli, name, x, y, z, qx, qy, qz, qw):
		self.rows.append([milli, name, x, y, z, qx, qy, qz, qw])

def main():
	d = DataCollector()
	print(d.read())
	


if __name__ == "__main__":
	main()
