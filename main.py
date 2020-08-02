import sys
sys.path.append('src/python')
#from preprocessing import PreProcessing
import preprocessing as prep
import modeling as train
import logging
import pandas as pd


def main():
	App = myapp()

	# initialization
	io_config = App.init_IO()
	print(io_config)
	model_path = io_config["transformer path"]
	logger = App.init_logging(io_config["log path"])
	
	
	# run selected part of application
	if len(sys.argv) == 1:
		logger.info("="*5 + "APPLICATION MODE = PRE-PROCESSING (STARTED)" + "="*5 )
		app_mode = "full"
	else:
	    app_mode = App.parsing_arguments(logger)#, io_config)
	if app_mode in ["pre-processing", "full"]:
		logger.info("="*5 + "APPLICATION MODE = PRE-PROCESSING (STARTED)" + "="*5 )
		App.run_preprocessing(logger, io_config)

	if app_mode in ["training", "full"]:
		logger.info("="*5 + "APPLICATION MODE = TRAINING (STARTED)" + "="*5 )
		App.run_model_training(logger, io_config)

	if app_mode in ["prediction", "full"]:
		logger.info("="*5 + "APPLICATION MODE = PREDICTION (STARTED)" + "="*5 )

	# print out summary with file path and purpose of output files


	
class myapp():
	def init_IO(self):
		# initialize IO from IO_Config.json
		io_config = pd.read_json("IO_Config.json", typ='series')
		#HOME_DIR = io_config['home directory']
		user_path = io_config["data path for user table"]
		model_path = io_config["transformer path"]
		result_path = io_config["result path"]
		log_path = io_config["log path"]

		return io_config.to_dict()#user_path, model_path, result_path, log_path

	def init_logging(self, log_path):
		# initialize logging
		logger=logging.getLogger("myapp")
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s %(message)s', 
			handlers=[
				logging.FileHandler(log_path + "info.log", mode = 'w'),
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

	def run_model_training(self, logger, io_config):
		Train = train.modelTraining(logger, io_config)
		Train.train()








if __name__ == "__main__":
	main() 







	


