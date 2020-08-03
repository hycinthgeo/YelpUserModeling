import pandas as pd

def sanity_check_dataframe(logger, raw):

    term_bucket_count = {}
    logger.info("%30s %20s"%("term name", "number of buckets"))

    cols_to_normalize = []
    for col in raw.columns:
        if "bucket" in col:
            term, bucket_id = col.split("_bucket_")
            term_bucket_count.setdefault(term, 0)
            term_bucket_count[term] += 1
        else:
            term_bucket_count[col.split("-")[0]] = 1
        if (abs(raw[col].min() - 0) > 0.01 or abs(raw[col].max() - 1) > 0.01):
            cols_to_normalize.append(col)


    for term, count in term_bucket_count.items():
        logger.info("%30s, %20d"%(term, count))

    if len(cols_to_normalize) > 0:
        logger.info("Sanity check: Before moving to the modeling state, If your algorithm is sensitive to values")
        logger.info("we recommend to apply scaler() for those columns with range beyond [0,1]")
        for col in cols_to_normalize:
            logger.info("%30s %20s %20s"%(col, raw[col].min(), raw[col].max()))

    return cols_to_normalize




def load_CSV_file_for_sanity_check(logger, output_path_all_data):
    raw = pd.read_csv(output_path_all_data)
    nrow, ncol = raw.shape
    logger.info("Sanity check started for %s, #rows = %d, #columns = %d"%(output_path_all_data, nrow, ncol))
    raw = pd.read_csv(output_path_all_data)
    cols_to_normalize = sanity_check_dataframe(logger, raw)

    return raw, cols_to_normalize