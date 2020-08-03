import pandas as pd
from parse_model import modelParser
import numpy as np
from constants import *
from utility_functions import *
import pandas.core.frame as pcf
from sklearn import preprocessing

class PreProcessing:
    def __init__(self, logger, io_config):
        self.logger = logger
        self.model_path = io_config["transformer path"]
        self.output_path = io_config["all transformed data"]
        self.user_path = io_config["data path for user table"]
        self.result_path = io_config["result path"]


    def process_user_table(self):
        self.logger.info("Processing user table")
        output_path_all_data = self.output_path#self.result_path + "allData.csv"

        #parse model file
        mParser = modelParser(self.model_path)
        model_buckets_df = mParser.parse()
        data_source_id = set(model_buckets_df.loc['source'].values[2:])
        self.logger.info("Prepare to load %d data sources"%len(data_source_id))

        # mandatory processing of user table (main table)
        df = read_raw_user_json_and__print_basic_info(self.logger, self.user_path) #mandatory reading of user table
        joined = pd.DataFrame([i for i in range(len(df))], columns = ["temp_index"])
        self.logger.info("Loading data from the following file, data source = %s"%self.user_path)

        # get statistics of input columns to drive user's decision on buckets
        self.logger.info("Scanning raw columns to output quantile stats")
        output_path_quantile_stats = self.result_path + "QuantileOfDiscreteFields_users.csv"
        convert_op_val_to_int = True
        quantile_df_votes = get_stats_df(df, RAW_COLNAME_VOTES, convert_op_val_to_int)
        quantile_df_compliments = get_stats_df(df, RAW_COLNAME_COMPLIMENTS, convert_op_val_to_int)
        quantile_df_review_count = get_stats_df(df, RAW_COLNAME_REVIEW_COUNT, convert_op_val_to_int)
        quantile_df_fans = get_stats_df(df, RAW_COLNAME_FANS, convert_op_val_to_int)
        quantile_df_merged = pd.concat(
            [quantile_df_review_count, quantile_df_fans,quantile_df_votes, quantile_df_compliments])
        quantile_df_merged.to_csv(output_path_quantile_stats, index=False)

        # apply evaluation
        cur_evals = model_buckets_df.loc[:, model_buckets_df.loc["source"] == "user"].loc["eval"].dropna()
        cur_cols = list(cur_evals.keys())

        for col in cur_cols:
            expression = cur_evals[col]
            new_col = cur_evals[col]
            cur_df = pd.DataFrame(df[[col]].eval(expression), columns=[new_col])
            cur_df[new_col] = cur_df[new_col].astype(int)
            joined =pd.concat([joined, cur_df], axis = 1)

        # apply bukcetization
        cur_buckets = model_buckets_df.loc["buckets", model_buckets_df.loc["source"] == "user"].dropna()
        cur_cols = list(cur_buckets.keys())
        processed_prefix = set()
        dict_cols = {}
        for col in cur_cols:
            if "-" in col:
                col_prefix, col_suffix = col.split("-")
                dict_cols.setdefault(col_prefix, [])
                dict_cols[col_prefix].append(col_suffix)
        for col_prefix, col_suffix_list in dict_cols.items():
            cur_df = flatten_and_bucketize(self.logger, df, cur_buckets, col_prefix, col_suffix_list)
            joined =pd.concat([joined, cur_df], axis = 1)
        nrow, ncol = joined.shape
        self.logger.info("Present output dataframe size =(%d x %d)"%(nrow, ncol))

        for col in cur_cols:
            if "-" not in col:
                if "_derive" in col:
                    col_true, operator = col.split("_derive_")[0], col.split("_derive_")[1]
                    bucket_list = cur_buckets[col]
                    def get_bucket_id(inval, bucket_list):
                            for i in range(len(bucket_list) - 1):
                                if bucket_list[i] < inval <= bucket_list[i+1]:
                                    return i + 1
                    df_friends_count_bucket_id = pd.DataFrame(
                        df[col_true].str.len().apply(lambda x: get_bucket_id(x, bucket_list)),
                    )
                    df_friends_count_bucket_id.columns = ["friends_count"]
                    cur_df = pd.get_dummies(df_friends_count_bucket_id["friends_count"], prefix="friends_count_bucket")
                    print_and_log_sparsity(cur_df, self.logger) #0-2917 friends
                else: #list
                    col_type = type(df[col].loc[0])
                    if col_type == list:
                        col_to_bucketize = col
                        cur_df= get_bucketized_df_list_col(self.logger, df, cur_buckets, col_to_bucketize)
                        print_and_log_sparsity(cur_df, self.logger)
                    elif col_type in [np.int64, np.int32, np.float64, np.float32]:
                        cur_df = get_bucket_df(df, col, cur_buckets[col])
                        print_and_log_sparsity(cur_df, self.logger)
                    else:
                        self.logger.info("ERROR, NOT CAPTURED, COL = ", col)

                joined =pd.concat([joined, cur_df], axis = 1)
        self.logger.info("Pre-Processing Completed - writing output started, output path = %s"%output_path_all_data)
        self.logger.info(joined.columns)
        joined.drop(["temp_index"], axis =1, inplace=True)
        #joined.to_csv(output_path_all_data, index=False)
        #load_CSV_file_for_sanity_check(self.logger, output_path_all_data)
        cols_to_normalize = sanity_check_dataframe(self.logger, joined)

        # optional: apply scaler
        apply_scaler = model_buckets_df.loc["apply_scaler", "models"]
        if len(cols_to_normalize) > 0:
            optional_further_apply_transformer(self.logger, joined, output_path_all_data, apply_scaler)
        else:
            joined.to_csv(output_path_all_data, index=False)

    #if "user" in data_source_id:







def read_raw_user_json_and__print_basic_info(logger, user_path):

    logger.info("==========Step 1: Loading Data & Get Basic Info (Started) ==========")
    users = pd.read_json(user_path, lines=True)
    df = users
    logger.info("Scanning raw inputs: %d rows and %d columns " %(len(df), len(df.loc[0])))
    logger.info("Columns are " + ", ".join(df.columns))

    raw_columns = df.columns
    logger.info("%30s | %20s | %20s"%("INFO: raw column name","#non-null value", "%non-null value"))
    logger.info("".join("-" for i in range(100)))
    for cur_col in raw_columns:
        cur_type = type(df[cur_col].loc[0])
        if cur_type in [np.float64, str]:
            count_valid = len(df[df[cur_col].notnull()])
        elif cur_type in [list, dict]:
            count_valid = len(df[df[cur_col].str.len() > 0])
        logger.info("%30s | %20d | %19.1f"%(cur_col, count_valid, 100.0 * count_valid/len(df)))

    num_elite_users = len(df[df[RAW_COLNAME_ELITE].str.len() > 0]) #elite is the years being an elite
    num_users_w_friends = len(df[df[RAW_COLNAME_FRIENDS].str.len() > 0])
    num_users_w_compliements = len(df[df[RAW_COLNAME_COMPLIMENTS].str.len() > 0])
    logger.info("SUMMARY")
    logger.info("%50s = %d"%("num of elite users", num_elite_users))
    logger.info("%50s = %d"%("number of users with friends", num_users_w_friends))
    logger.info("%50s = %d"%("number of users with compliments", num_users_w_compliements))

    logger.info("==========Step 1: Loading Data & Get Basic Info (Completed)==========")
    return df

def get_stats_df(df: pcf.DataFrame, raw_col:str, convert_op_val_to_int: bool) -> pcf.DataFrame:
    metrics = []
    list_fixed_quantiles = [0, 0.01] + list(np.arange(0.1, 1.0, 0.1)) + [0.99, 1.0]
    columns = ['field'] + ['q_'+ "{:.2f}".format(cur_quantile) for cur_quantile in list_fixed_quantiles]

    if type(df.loc[0][raw_col]) == dict:
        comp_dict = dict()
        for cur_dict in df[raw_col]:
            if len(cur_dict) > 0:
                for key, val in cur_dict.items():
                    comp_dict.setdefault(key, [])
                    comp_dict[key].append(val)

        for col in comp_dict.keys():
            vals = comp_dict[col]
            if convert_op_val_to_int:
                list_quantile_vals = [raw_col + " - " + col] + [int(np.quantile(vals, cur_quantile)) for cur_quantile in list_fixed_quantiles]
            else:
                list_quantile_vals = [raw_col + " - " + col] + [np.quantile(vals, cur_quantile) for cur_quantile in list_fixed_quantiles]
            metrics.append(list_quantile_vals)

    elif type(df.loc[0][raw_col]) == np.int64:
        vals = df[raw_col]
        list_quantile_vals = [raw_col] + [int(np.quantile(vals, cur_quantile)) for cur_quantile in list_fixed_quantiles]
        metrics.append(list_quantile_vals)
    elif type(df.loc[0][raw_col]) == np.float64:
        vals = df[raw_col]
        list_quantile_vals = [raw_col] + [np.quantile(vals, cur_quantile) for cur_quantile in list_fixed_quantiles]
        metrics.append(list_quantile_vals)
    quantile_df = pd.DataFrame(metrics, columns=columns)
    return quantile_df

# bucketize list of integer by their counts
def get_bucketized_df_list_col(logger, df, model_buckets_df, col_to_bucketize):
    bucket_list = model_buckets_df[col_to_bucketize]
    bucket_list_str = "[" + ", ".join([str(cur) for cur in bucket_list]) + "]"
    unique_years = len(bucket_list) - 2
    min_elite_year = bucket_list[1]
    data = []
    count = 0
    for cur in df[col_to_bucketize]:
        count += 1
        cur_num_years = len(cur)
        cur_elite_years = [0 for i in range(unique_years)]
        for year in cur:
            offset = int(year) - min_elite_year
            cur_elite_years[offset] = 1
        data.append(cur_elite_years + [cur_num_years])
    elite_year_df = pd.DataFrame(data,
        columns= [col_to_bucketize+"_bucket_"+str(year-min_elite_year) for year in bucket_list[1:-1]] + [col_to_bucketize + "-total"])
        #columns= [col_to_bucketize+"-"+str(year) for year in bucket_list[1:-1]] + [col_to_bucketize + "-total"])

    return elite_year_df
#df_elite_year = get_bucketized_df_list_col(model_buckets_df, col_to_bucketize)

def print_and_log_sparsity(elite_year_df, logger):
    logger.info("%30s | %20s | %20s | %20s"%(
        "bucketized feature", "total records", "non-zero records", "% non-zero records"))
    logger.info("".join(["-"*100]))
    for col in elite_year_df.columns:
        nrecords = len(elite_year_df)
        nonzero = len(elite_year_df[elite_year_df[col] != 0])
        perc_nonzero = 100.0 * nonzero / nrecords
        logger.info("%30s | %20d | %20d | %17.3f"%(col, nrecords, nonzero, perc_nonzero))

def get_bucket_df(df, col_to_bucketize, bucket_list):
    bucket_tuple =  pd.cut(df[col_to_bucketize], bucket_list,
        labels=False, retbins=True, right=False, duplicates='drop')

    col_bucketized = col_to_bucketize + "_bucket"
    df_bucket_id = pd.DataFrame({
        col_bucketized: bucket_tuple[0]
    }).fillna(0).astype("int32")

    df_output = pd.get_dummies(df_bucket_id[col_bucketized],prefix=col_bucketized,drop_first=False)

    return df_output

def flatten_and_bucketize(logger, df, cur_buckets, col_prefix, col_suffix_list):

    def get_bucket_of_a_dictionary_column(df_votes, cur_buckets):
        col_to_bucketize = df_votes.columns[0]
        bucket_list = cur_buckets[col_to_bucketize]
        ret_df = get_bucket_df(df_votes, col_to_bucketize, bucket_list)

        for i, col in enumerate(df_votes.columns[1:]):
            cur_df = get_bucket_df(df_votes, col, bucket_list)
            ret_df = pd.concat([ret_df, cur_df], axis = 1)
        return ret_df

    df_votes_flatten = pd.DataFrame(df[col_prefix].values.tolist(), index=df.index).fillna(0).astype('int32')
    df_votes_flatten.columns = [col_prefix + "-" + cur for cur in df_votes_flatten.columns]
    df_votes_flatten = df_votes_flatten[[col_prefix + "-" + suffix for suffix in col_suffix_list]]
    df_votes = get_bucket_of_a_dictionary_column(df_votes_flatten, cur_buckets)
    print_and_log_sparsity(df_votes, logger)
    return df_votes


def optional_further_apply_transformer(logger, df, input_path, apply_scaler):
    if apply_scaler != "":
        x = df.values
        #output_path_all_data = input_path[:input_path.find(".csv")]+"_" + apply_scaler +".csv"
        output_path_all_data = input_path
        if apply_scaler == "MinMaxScaler":
            scaler = preprocessing.MinMaxScaler()
        elif apply_scaler == "StandardScaler":
            scaler = preprocessing.StandardScaler()
        x_scaled = scaler.fit_transform(x)
        logger.info("write further transformed all data using transformer (started) = %s"%apply_scaler)
        df = pd.DataFrame(x_scaled, columns=df.columns)
        df.to_csv(output_path_all_data, index=False)
        logger.info("write further transformed all data using transformer (done) = %s"%apply_scaler)
    else:
        logger.info("No transformer being applied to input data")

