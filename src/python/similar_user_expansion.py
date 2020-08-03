import pandas as pd
from sklearn.neighbors import NearestNeighbors

class similarUserExpansion:
    def __init__(self, logger, io_config):
        self.logger = logger
        self.io_config = io_config
        self.model_config_path = io_config["modelConfig path"]
        self.model_config = pd.read_json(self.model_config_path, typ='Series')
        self.test_path = io_config["test data"]
        self.result_path = io_config["result path"]

    def filter_and_expand(self):

        self.logger.info("Filtering to get seed users")

        io_config = self.io_config
        model_config = pd.read_json(io_config["modelConfig-case2 path"])
        input_path_raw_data_subset = io_config["all transformed data-case2"]
        user_path_subset = io_config["data path for user table-case2"]
        output_user_list = io_config["output path for user list-case2"]
        target_total_audience = model_config['target_num_users'][0]

        raw = pd.read_csv(user_path_subset)
        alldata = pd.read_csv(input_path_raw_data_subset)

        temp = raw.copy()
        temp['yelping_since_year'] = temp['yelping_since'].apply(
		    lambda x: int(x.split("-")[0]))
        temp['elite_year_count'] = temp['elite'].apply(
		    lambda x: 0 if x == "[]" else len(x.split(",")))
        temp['friends_count'] = temp['friends'].apply(
		    lambda x: len(x.split(",")))

        print("Total users = %d" % (len(temp)))
        myfilter = [True for x in range(len(temp))]
        for col in model_config.columns:
        	if col != "target_num_users":
        		min_val = model_config[col][0]
        		max_val = model_config[col][1]
        		myfilter = myfilter & (temp[col] >= min_val) & (temp[col] <= max_val)
        		print("Filter field = %20s,   range = [%5s, %5s],   post-filter records = %d" %
				 (col, str(min_val), str(max_val), len(temp[myfilter])))

        temp = temp[myfilter]
        num_exact_match = len(temp)
        print("Exact-match users = %d" % (num_exact_match))

        if num_exact_match >= target_total_audience:
            out_df.to_csv(output_user_list, index=False)

        self.logger.info("Expansion started")

        seed_user_index = list(temp.index)
        est_K = (target_total_audience // num_exact_match + 1) * \
            2  # to be more conservative
        X = alldata.values
        print("Estimated K-nearest neighbors, K = %d" % est_K)
        nbrs = NearestNeighbors(n_neighbors=est_K, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        final_user_index_dist = {}
        for idx in seed_user_index:
        	final_user_index_dist[idx] = [0, "exact_match"]

        for idx in seed_user_index:
        	nns = indices[idx]
        	for j, nn in enumerate(nns):
        		if nn not in final_user_index_dist.keys():
        			final_user_index_dist[nn] = [distances[idx][j], "expanded-" + str(idx)]

        out = []
        for key, vals in final_user_index_dist.items():
            out.append((key, vals[0], vals[1]))
        out_df = pd.DataFrame(
            out, columns=["user_idx", "dist_from_exact_match", "user_source"])
        out_df.to_csv(output_user_list, index=False)

        self.logger.info(
            "Expansion completed, output path = %s" % output_user_list)
