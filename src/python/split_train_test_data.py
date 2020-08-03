import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Splitter:
    def __init__(self, logger, io_config):
        self.logger = logger
        self.io_config = io_config
        #self.model_config_path = io_config["modelConfig path"]

    def split_and_write(self):
        self.logger.info("Splitting train and test data started")

        #def optional further apply transformer
        input_path = self.io_config["all transformed data"]
        output_path_train = self.io_config["train data"]
        output_path_test = self.io_config["test data"]
        #user_path = self.io_config["data path for user table"]

        # get label from user table (outdated - now we won't specify features vs. labels until the modeling step)
        """
        model_config = pd.read_json(self.model_config_path, typ = "series")
        raw_label_name = model_config["label"]
        label_eval = model_config["label-eval"]
        y = get_labels(user_path, raw_label_name, label_eval)
        """

        # get features from input path
        df = pd.read_csv(input_path)
        X = df.values
        y = [[0] for i in range(len(df))] #dummy to use sklearn train_test_split_scheme

        #train test split
        test_size = float(self.io_config["train-test split"])
        train_test_split_random_state = int(self.io_config["train-test split random state"])

        self.logger.info("Started train-test split, using test_size=%2.1f, random_state=%d"%(test_size, train_test_split_random_state))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=train_test_split_random_state)
        #output_path_train = input_path[:input_path.find("/")+1] + "train" + input_path[input_path.find("Data"):]
        #output_path_test = input_path[:input_path.find("/")+1] + "test" + input_path[input_path.find("Data"):]
        #output_colnames = list(df.columns) + [label_eval] if len(label_eval) > 0 else list(df.columns) + [raw_label_name]
        output_colnames = list(df.columns)
        print(output_colnames)
        self.logger.info("Started writing train data to %s"%output_path_train)
        train = pd.DataFrame(X_train, columns = df.columns)
        #train = pd.DataFrame(np.concatenate((X_train, [[cur] for cur in y_train]), axis = 1), columns = output_colnames)
        train.to_csv(output_path_train, index=False)
        self.logger.info("Started writing test data to %s"%output_path_test)
        test = pd.DataFrame(X_test, columns = df.columns)
        #test = pd.DataFrame(np.concatenate((X_test, [[cur] for cur in y_test]), axis = 1), columns = output_colnames)
        test.to_csv(output_path_test, index=False)

        self.logger.info("Splitting train and test data completed")

"""
def get_labels(user_path, raw_label_name, label_eval):
    df_label = pd.read_json(user_path, lines = True)[[raw_label_name]]
    if label_eval == "":
        y = [x[0] for x in df_label.values]
    else:
        y = [1 if x is True else 0 for x in df_label.eval(label_eval)]
    return y
"""



