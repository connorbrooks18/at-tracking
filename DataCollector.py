import json
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TRACKING_COLUMNS = ["timestamp", "name", "x", "y", "z", "qx", "qy", "qz", "qw"]


class DataCollector:
	
	def __init__(self, metadata=None):
		self.rows = []
		self.metadata = dict(metadata or {})

	def dump(self, filename="output.parquet", metadata=None):
		"""Write long-form tracker poses with JSON metadata in the Parquet footer."""
		df = pd.DataFrame(self.rows, columns=TRACKING_COLUMNS)
		table = pa.Table.from_pandas(df, preserve_index=False)
		file_metadata = dict(self.metadata)
		file_metadata.update(metadata or {})
		file_metadata.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
		file_metadata.setdefault("timestamp_clock", "Unix wall clock from time.time()")
		file_metadata.setdefault("timestamp_unit", "seconds")
		file_metadata.setdefault("schema_name", "real_apple_tracking_raw")
		file_metadata.setdefault("schema_version", "1.0.0")
		schema_metadata = dict(table.schema.metadata or {})
		schema_metadata[b"dataset_metadata"] = json.dumps(
			file_metadata, sort_keys=True, default=str
		).encode("utf-8")
		table = table.replace_schema_metadata(schema_metadata)
		pq.write_table(table, filename)

	def read(self, filename="output.parquet"):
		return pq.read_table(filename)
	
	def print(self):
		print(self.rows)
		
	def update(self, timestamp, name, x, y, z, qx, qy, qz, qw):
		self.rows.append([timestamp, name, x, y, z, qx, qy, qz, qw])

def main():
    d = DataCollector()
    table = d.read()
    timestamps = table.column(0)
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

    length = -(float(timestamps[0])-float(timestamps[-1]))
    frames = blank_dict[names[0]][1]
    fps = frames/length
    print(f"length was {length:.2f} seconds with {frames} frames or {fps:.2f} fps")
	


if __name__ == "__main__":
	main()
