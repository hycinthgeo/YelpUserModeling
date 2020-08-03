from joblib import dump, load
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class modelPrediction:
    def __init__(self, logger, io_config):
        self.logger = logger
        self.io_config = io_config
        self.model_config_path = io_config["modelConfig path"]
        self.model_config = pd.read_json(self.model_config_path, typ='Series')
        self.test_path = io_config["test data"]
        self.result_path = io_config["result path"]

    def prediction(self):
        self.logger.info("model prediction started")
        input_path_model_artefact = self.io_config["modelArtefact path"]
        clf = load(input_path_model_artefact)
        self.logger.info("Loaded model artefacts = %s"%input_path_model_artefact)

        # Loading test data
        df = pd.read_csv(self.test_path)
        label = self.model_config["label"]
        X_test = df.drop([label], axis = 1).values
        y_test = [x[0] for x in df[[c for c in [label]]].values]
        feature_names = df.drop([label], axis = 1).columns
        self.logger.info("Loaded test data from = %s"%self.test_path)

        # Predict and score
        scoring = self.model_config["scoring"]
        score = clf.score(X_test, y_test)
        y_test_pred = clf.predict(X_test)

        self.logger.info(
            "Summary \n precision score = %4.3f\n recall score = %4.3f\n f1-score = %4.3f\n roc_auc_score = %4.3f"%(precision_score(y_test, y_test_pred),
            recall_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred),
            roc_auc_score(y_test, y_test_pred)
        ))


