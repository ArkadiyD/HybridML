from learn_multi_class import *
import numpy as np
import pandas as pd
import pickle
import time
import sys

'''
Main file for training random forest:

RUN CODE:
python main_learn.py <N number of networks> <bool (0/1) for balanced data> <maxL>
maxL: in the data set, maxL is the maximum number of leaves of an instance group.

EXAMPLE: 
python LearningCherries/main_learn.py 1000 1 100
'''

if __name__ == "__main__":
    N = int(sys.argv[1])
    balanced = int(sys.argv[2])
    max_l = int(sys.argv[3])

    # PREPARE METADATA OF ML MODEL
    metadata = pd.Series(index=["N", "maxL", "num_data", "data_non_cher", "data_cher",
                                "data_ret_cher", "data_no_ret_cher", "acc_all", "acc_non_cher", "acc_cher",
                                "acc_ret_cher", "acc_no_ret_cher", "train_dur", "balanced"], dtype=float)
    metadata["balanced"] = balanced
    metadata["maxL"] = max_l

    # rf model name
    data_set_string = f"N{N}_maxL{max_l}_random"
    if balanced:
        data_set_string += "_balanced"

    print(f"Start {data_set_string}")
    output_list = []
    data_set = f"../ALL_ML_data_N{N}_maxL{max_l}_random.pickle"
    with open(data_set, "rb") as handle:
        output_list.append(pickle.load(handle))

    # Clean data by deleting bad data points
    X_new = output_list[0]["X"]
    Y_new = output_list[0]["Y"]
    for output in output_list[1:]:
        X_new = pd.concat([X_new, output["X"]])
        Y_new = pd.concat([Y_new, output["Y"]])

    # make new index for X and Y first
    X_new.index = np.arange(len(X_new))
    Y_new.index = X_new.index
    X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_new.dropna(inplace=True)
    Y_new = Y_new.loc[X_new.index]

    # Balance data such that each class has equal size
    if balanced:
        X_new["class"] = Y_new.dot(np.array([0, 1, 2, 3]).transpose())
        g = X_new.groupby('class')
        g = pd.DataFrame(g.apply(lambda x: x.sample(int(g.size().min()*1.5), replace=True).reset_index(drop=False)))
        X_new = g
        X_new.index = g["index"]
        X_new.drop(["index", "class"], axis=1, inplace=True)
        Y_new = Y_new.loc[X_new.index]

    output_info = Y_new.describe()
    print(output_info)
    num_data = len(Y_new)
    metadata["N"] = N
    metadata["num_data"] = num_data
    metadata[["data_non_cher", "data_cher", "data_ret_cher", "data_no_ret_cher"]] = list(output_info.loc["mean"])
    print(f"Num data points = {num_data}")

    # TRAIN RF MODEL
    start_time = time.time()
    print("\nRF MODEL")
    metadata[["acc_all", "acc_non_cher", "acc_cher", "acc_ret_cher", "acc_no_ret_cher"]] = train_cherries_rf(X_new, Y_new, name=data_set_string)
    print(f"RF train duration (normal): {np.round(time.time() - start_time, 3)}s")
    metadata["train_dur"] = time.time() - start_time
    metadata.to_pickle(f"RFModels/model_metadata_{data_set_string}.pickle")
