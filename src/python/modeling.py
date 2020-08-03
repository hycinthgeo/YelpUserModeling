import pandas as pd
from parse_model import modelParser
import numpy as np
from constants import *
from utility_functions import *
import pandas.core.frame as pcf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

#from sklearn.model_selection import train_test_split

class modelTraining:
    def __init__(self, logger, io_config):
        self.logger = logger
        self.io_config = io_config
        self.train_path =  io_config["train data"]
        self.result_path = io_config["result path"]
        self.model_config_path = io_config["modelConfig path"]
        self.model_config = pd.read_json(self.model_config_path, typ='Series')

    def train(self):
        self.logger.info("model training started")
        df, cols_to_scale =load_CSV_file_for_sanity_check(self.logger, self.train_path)

        label = self.model_config["label"]

        X_train = df.drop([label], axis = 1).values
        y_train = [x[0] for x in df[[c for c in [label]]].values]
        feature_names = df.drop([label], axis = 1).columns
        with open('results/feature_names.txt', 'w') as filehandle:
            for listitem in feature_names:
                filehandle.write('%s\n' % listitem)

        algorithm = self.model_config["algorithm"]
        if_tuning = self.model_config["if_tuning"]
        n_cv = self.model_config["K-fold cross-validation"]
        scoring = self.model_config["scoring"]
        logger = self.logger
        #output_path_model_artefact = "artefacts/"+self.model_config_path[self.model_config_path.find("/") + 1:self.model_config_path.find(".json")]
        output_path_model_artefact = self.io_config["modelArtefact path"]

        if algorithm == "regression:logsticRegression":
            est = LogisticRegression(max_iter=10000)
            if if_tuning == "True":
                Cs = [0.001, 0.01, 0.1, 1]
                clf = GridSearchCV(estimator=est, param_grid=dict(C=Cs),n_jobs=-1, return_train_score=True,
                               cv = n_cv, scoring=scoring)
                clf.fit(X_train, y_train)
                best_C = clf.best_estimator_.C
                logger.info("Best hyperparameter C = %f, best score (%s) = %f"%(best_C, scoring, clf.best_score_))
                coef = clf.best_estimator_.coef_[0]

                #plot tuning results
                model_config_path = self.model_config_path[self.model_config_path.find("-"):self.model_config_path.find(".json")]
                fig_path_tuning = self.result_path + "tuning-C%s.png"%model_config_path
                plot_tuning_scores(clf, n_cv, scoring, fig_path_tuning)
                dump(clf.best_estimator_, output_path_model_artefact)

            else:
                best_C = 0.1
                clf = LogisticRegression(max_iter=10000, C=best_C)
                clf.fit(X_train, y_train)
                scores = cross_validate(clf, X_train, y_train, cv=n_cv, scoring=[scoring])
                best_score = max(scores["test_"+scoring])
                coef = clf.coef_[0]
                #pred_score = clf.score(X_train, y_train, scoring = scoring)
                logger.info("Use the pre-tuned hyperparameter C=%f, validation score = %f"%(best_C, best_score))
                dump(clf, output_path_model_artefact)

        coef_topK = 50
        model_config_path = self.model_config_path[self.model_config_path.find("-"):self.model_config_path.find(".json")]
        fig_path_coef = self.result_path + "coef%s.png"%model_config_path
        plot_coefficients(feature_names, coef, coef_topK, fig_path_coef, True)






        # split data into train, validation, test





def get_labels(user_path, raw_label_name, label_eval):
    df_label = pd.read_json(user_path, lines = True)[[raw_label_name]]
    if label_eval == "":
        y = [x[0] for x in df_label.values]
    else:
        y = [1 if x is True else 0 for x in df_label.eval(label_eval)]
    return y

def plot_coefficients(feature_names, coef, coef_topK, fig_path_coef, save_fig):
    pref_id_map = {}
    for i, col in enumerate(feature_names):
        if "bucket" in col:
            col_prefix, col_bid = col.split("_bucket_")
        else:
            col_prefix, col_bid = col.split("-")
        pref_id_map.setdefault(col_prefix, [])
        pref_id_map[col_prefix].append(i)
    plt.figure(figsize=(6, 10))
    plt.plot(coef, [i for i in range(len(coef))])
    plt.gca().invert_yaxis()

    coef_df = pd.DataFrame(coef, columns=['coef'])
    coef_df['feature'] = feature_names
    coef_df['abs(coef)'] = coef_df['coef'].apply(lambda x: abs(x))

    coef_topK_df = coef_df.sort_values(['abs(coef)'], ascending=False).reset_index().loc[:coef_topK-1]
    plt.scatter(coef_topK_df['coef'], coef_topK_df['index'])
    xlim = [-2.0, 3.0]
    i = 0
    for pref, ids in pref_id_map.items():
        if i < len(pref_id_map):
            plt.plot(xlim, [max(ids) +0.5, max(ids)+0.5], color="gray")
            #plt.text( 2.3, (min(ids) + max(ids))/2.0, i)
            plt.text( 1.5, (min(ids) + max(ids))/2.0, "%2d-%s"%(i, pref))
            i += 1
    plt.legend(['coef', "feature_type_boundary"], loc="lower left")
    plt.xlim(xlim)
    plt.ylim([120, -10])
    plt.xlabel("coef")
    plt.ylabel("Feature ID (index)")
    if save_fig:
        plt.savefig(fig_path_coef)
    #plt.show()

def plot_tuning_scores(clf, n_cv, scoring, fig_path_tuning):
    temp = pd.DataFrame(clf.cv_results_)
    new_score = pd.DataFrame(temp['param_C'])
    new_score['train'] = temp['split0_train_score']
    new_score['validation'] = temp['split0_test_score']

    fig = plt.figure(figsize = (10, 6))
    for i in range(1, n_cv):
        cur_score = pd.DataFrame(temp['param_C'])
        cur_score['train'] = temp['split'+str(i)+'_train_score']
        cur_score['validation'] = temp['split'+str(i)+'_test_score']
        new_score = pd.concat([new_score, cur_score], axis = 0)
    axes = new_score.boxplot(['train', 'validation'], by = "param_C",layout=(1,2),
                      figsize = (10, 6), grid = False)

    axes[0].set_ylabel(scoring, fontweight="bold", fontsize=12)
    plt.subplots_adjust(wspace = 0.2)
    plt.suptitle(x = 0.5, y = 1, t="Score boxplot from %d-fold cross-validation (Score = %s)"%(n_cv, scoring),
                 fontsize = 14, fontweight="bold")
    plt.savefig(fig_path_tuning)
    #plt.show()