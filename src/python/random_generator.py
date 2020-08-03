import pandas as pd

def generate_random_case2_inputs(num_rand_members):
    input_path_all_data = "results/allData.csv"
    user_path = "yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json"

    suffix = "-subset2"
    loc = user_path.find(".json")
    user_path_subset = user_path[:loc] + suffix + ".csv"
    loc = input_path_all_data.find(".csv")
    input_path_raw_data_subset = input_path_all_data[:loc] + suffix + input_path_all_data[loc:]
    raw = pd.read_json(user_path, lines=True)

    # randomly generate 1000 users to demo
    def generate_random_member_subset(num_rand_members, raw):
        from random import randint
        rand_members = set()



        for i in range(num_rand_members * 2):
            rand_members.add(randint(0, len(raw)))
            if (len(rand_members) == num_rand_members):
                break

        return rand_members
    rand_members = generate_random_member_subset(num_rand_members, raw)
    raw.loc[rand_members].to_csv(user_path_subset, index = False)
    alldata = pd.read_csv(input_path_all_data)
    alldata.loc[rand_members].to_csv(input_path_raw_data_subset, index = False)
