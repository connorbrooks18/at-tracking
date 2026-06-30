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
    table = d.read()
    millis = table.column(0)
    names = table.column(1)
    xs = table.column(2)
    ys = table.column(3)
    zs = table.column(4)
    qxs = table.column(5)
    qys = table.column(6)
    qzs = table.column(7)
    qws = table.column(8)

    blank_dict = {"all": [0, 0],}

    for i in range(table.num_rows):
        print(millis[i])
        pose = (float(xs[i]), float(ys[i]), float(zs[i]))
        if names[i] not in blank_dict:
            blank_dict[names[i]] = [0, 0]
        if pose == (0, 0, 0):
            blank_dict["all"][0] += 1
            blank_dict[names[i]][0] += 1
        blank_dict[names[i]][1] += 1
        blank_dict["all"][1] += 1


    for key in blank_dict:
        print(f"{key} has {blank_dict[key][0]} +  blank out of  + {blank_dict[key][1]} or {blank_dict[key][0]/blank_dict[key][1]}")
    length = -(float(millis[0])-float(millis[-1]))
    frames = len(set(millis))
    fps = frames/length
    print(f"length was {length:.2f} seconds with {frames} frames or {fps:.2f} fps")
	


if __name__ == "__main__":
	main()
