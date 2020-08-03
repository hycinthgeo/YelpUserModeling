import sys
sys.path.append('src/python')
#from preprocessing import PreProcessing
import preprocessing as prep
import split_train_test_data as split
import modeling as train
import prediction as predict
import logging
import pandas as pd
import similar_user_expansion as sim


def main():
	App = myapp()

	# initialization
	io_config = App.init_IO()
	model_path = io_config["transformer path"]
	logger = App.init_logging(io_config["log path"], "info.log")
	
	
	# run selected part of application
	if len(sys.argv) == 1:
		logger.info("="*5 + "APPLICATION MODE = PRE-PROCESSING (STARTED)" + "="*5 )
		app_mode = "full"
	else:
	    app_mode = App.parsing_arguments(logger)#, io_config)

	if app_mode in ["pre-processing", "full"]:
		logger.info("="*5 + "APPLICATION MODE = PRE-PROCESSING (STARTED)" + "="*5 )
		App.run_preprocessing(logger, io_config)

	if app_mode in ["split", "full"]:
		logger.info("="*5 + "APPLICATION MODE = SPLITTING-TRAIN-TEST (STARTED)" + "="*5 )
		App.run_train_test_split(logger, io_config)

	if app_mode in ["training", "full"]:
		logger.info("="*5 + "APPLICATION MODE = TRAINING (STARTED)" + "="*5 )
		App.run_model_training(logger, io_config)

	if app_mode in ["prediction", "full"]:
		logger.info("="*5 + "APPLICATION MODE = PREDICTION (STARTED)" + "="*5 )
		App.run_model_prediction(logger, io_config)

	if app_mode in ["similar-user", "full"]:
		logger.info("="*5 + "APPLICATION MODE = PREDICTION (STARTED)" + "="*5 )
		App.run_similar_user_expansion(logger, io_config)

	# print out summary with file path and purpose of output files


	
class myapp():
	def init_IO(self):
		# initialize IO from IO_Config.json
		io_config = pd.read_json("configs/data-pipeline.json", typ='series')
		#HOME_DIR = io_config['home directory']
		user_path = io_config["data path for user table"]
		model_path = io_config["transformer path"]
		result_path = io_config["result path"]
		log_path = io_config["log path"]

		return io_config.to_dict()#user_path, model_path, result_path, log_path

	def init_logging(self, log_path, logname):
		# initialize logging
		logger=logging.getLogger("myapp")
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s %(message)s', 
			handlers=[
				logging.FileHandler(log_path + logname, mode = 'w'),
				logging.StreamHandler()
			]
		)
		return logger

	def parsing_arguments(self, logger):
		# parsing arguments to determine what parts of the full application to run
		mode = sys.argv[1].split("=")[1].strip()
		if mode != "full":
			logger.info("Application started: run %s ONLY"%mode)
		else: 
			logger.info("Application started: run the full application")
		return mode

	def parsing_model_file(self, logger, model_path):
		logger.info("parsing model file from %s"%model_path)

	def run_preprocessing(self, logger, io_config):
		Prep = prep.PreProcessing(logger, io_config)
		Prep.process_user_table()

	def run_train_test_split(self, logger, io_config):
		Split = split.Splitter(logger, io_config)
		Split.split_and_write()

	def run_model_training(self, logger, io_config):
		Train = train.modelTraining(logger, io_config)
		Train.train()

	def run_model_prediction(self, logger, io_config):
		Predict = predict.modelPrediction(logger, io_config)
		Predict.prediction()

	def run_similar_user_expansion(self, logger, io_config):
		Sim = sim.similarUserExpansion(logger, io_config)
		Sim.filter_and_expand()









if __name__ == "__main__":
	main() 







	


