import pandas as pd

class modelParser():
	def __init__(self, model_path):
		self.model_path = model_path

	def parse(self):
		model_buckets_df = pd.read_json(self.model_path)
		return model_buckets_df
