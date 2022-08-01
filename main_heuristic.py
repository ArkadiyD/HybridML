from datetime import datetime
import pandas as pd
import numpy as np
import copy
import sys
import csv
import warnings

import CPH

warnings.simplefilter(action='ignore', category=FutureWarning)

'''
Code consisting of main run file and two functions:
- run_heuristic:            1. open tree set and make CPH.PhT environment for each tree
                            2. run cherry picking heuristic (CPH)
                            3. return results
- run_main:                 run CPH with four "PickNextCherry" methods:
                                1. ML
                                2. TrivialML
                                3. Rand
                                4. TrivialRand

RUN in terminal:
python main_heuristic.py <instance num.> <ML model name> <leaf number> <bool (0/1) for exact input> <option>
option: 
if exact input = 0:
    option = reticulation number
else:
    option = forest size
EXAMPLE: 
python main_heuristic.py 0 N10_maxL100_random_balanced 20 0 50
'''


def run_heuristic(tree_set=None, tree_set_newick=None, inst_num=0, repeats=1, time_limit=None,
                  progress=False,  reduce_trivial=False, pick_ml=False, pick_ml_triv=False,
                  pick_random=False, model_name=None, relabel=False, relabel_cher_triv=False, problem_type="",
                  full_leaf_set=True, ml_thresh=None):
    # READ TREE SET
    now = datetime.now().time()
    if progress:
        print(f"Instance {inst_num} {problem_type}: Start at {now}")

    if tree_set is None and tree_set_newick is not None:
        # Empty set of inputs
        inputs = []

        # Read each line of the input file with name set by "option_file_argument"
        f = open(tree_set_newick, "rt")
        reader = csv.reader(f, delimiter='~', quotechar='|')
        for row in reader:
            inputs.append(str(row[0]))
        f.close()

        # Make the set of inputs usable for all algorithms: use the CPH class
        tree_set = CPH.Input_Set(newick_strings=inputs, instance=inst_num, full_leaf_set=full_leaf_set)

    # RUN HEURISTIC CHERRY PICKING SEQUENCE
    # Run the heuristic to find a cherry-picking sequence `seq' for the set of input trees.
    # Arguments are set as given by the terminal arguments
    seq_dist, seq, df_pred = tree_set.CPSBound(repeats=repeats,
                                               progress=progress,
                                               time_limit=time_limit,
                                               reduce_trivial=reduce_trivial,
                                               pick_ml=pick_ml,
                                               pick_ml_triv=pick_ml_triv,
                                               pick_random=pick_random,
                                               relabel=relabel,
                                               relabel_cher_triv=relabel_cher_triv,
                                               model_name=model_name,
                                               ml_thresh=ml_thresh,
                                               problem_type=problem_type)

    # Output the computation time for the heuristic
    now = datetime.now().time()
    if progress:
        print(f"Instance {inst_num} {problem_type}: Finish at {now}")
        print(f"Instance {inst_num} {problem_type}: Computation time heuristic: {tree_set.CPS_Compute_Time}")
        print(f"Instance {inst_num} {problem_type}: Reticulation number = {min(tree_set.RetPerTrial.values())}")
    if pick_ml:
        return tree_set.RetPerTrial, tree_set.DurationPerTrial, seq, df_pred
    else:
        return tree_set.RetPerTrial, tree_set.DurationPerTrial, seq


def run_main(i, l, exact, ret=None, forest_size=None,
             repeats=1000, time_limit=None, ml_name=None, full_leaf_set=True, ml_thresh=None, progress=False):
    if exact:
        test_info = f"L{l}_R{ret}_exact_all"
        file_info = f"L{l}_R{ret}_exact"
    else:
        test_info = f"L{l}_T{forest_size}_all"
        file_info = f"L{l}_T{forest_size}"

    # ML MODEL
    model_name = f"LearningCherries/RFModels/rf_cherries_{ml_name}.joblib"
    # save results
    score = pd.DataFrame(
        index=pd.MultiIndex.from_product([[i], ["RetNum", "Time"], np.arange(repeats)]),
        columns=["ML", "TrivialML", "Rand", "TrivialRand", "UB"], dtype=float)
    df_seq = pd.DataFrame()
    env_info_file = f"Data/Test/inst_results/tree_data_{file_info}_{i}.pickle"
    # INSTANCE
    tree_set_newick = f"Data/Test/TreeSetsNewick/tree_set_newick_{file_info}_{i}_LGT.txt"

    # ML HEURISTIC
    ret_score, time_score, seq_ml, df_pred = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=1,
        time_limit=time_limit,
        pick_ml=True,
        relabel=True,
        model_name=model_name,
        problem_type="ML",
        full_leaf_set=full_leaf_set,
        ml_thresh=ml_thresh,
        progress=progress)

    score.loc[i, "RetNum", 0]["ML"] = copy.copy(ret_score[0])
    score.loc[i, "Time", 0]["ML"] = copy.copy(time_score[0])
    ml_time = score.loc[i, "Time", 0]["ML"]
    ml_ret = int(score.loc[i, "RetNum"]["ML"][0])
    df_seq = pd.concat([df_seq, pd.Series(seq_ml)], axis=1)

    # ML Trivial HEURISTIC
    ret_score, time_score, seq_ml_triv = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=1,
        time_limit=time_limit,
        pick_ml_triv=True,
        relabel=True,
        model_name=model_name,
        problem_type="TrivialML",
        full_leaf_set=full_leaf_set,
        ml_thresh=ml_thresh,
        progress=progress)

    score.loc[i, "RetNum", 0]["TrivialML"] = copy.copy(ret_score[0])
    score.loc[i, "Time", 0]["TrivialML"] = copy.copy(time_score[0])
    ml_triv_ret = int(score.loc[i, "RetNum"]["TrivialML"][0])
    df_seq = pd.concat([df_seq, pd.Series(seq_ml_triv)], axis=1)

    # RANDOM HEURISTIC
    ret_score, time_score, seq_ra = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=repeats,
        time_limit=ml_time,
        problem_type="Rand",
        pick_random=True,
        relabel=False,
        full_leaf_set=full_leaf_set,
        progress=progress)

    for r, ret in ret_score.items():
        score.loc[i, "RetNum", r]["Rand"] = copy.copy(ret)
        score.loc[i, "Time", r]["Rand"] = copy.copy(time_score[r])
    ra_ret = int(min(score.loc[i, "RetNum"]["Rand"]))
    df_seq = pd.concat([df_seq, pd.Series(seq_ra)], axis=1)

    # TRIVIAL RANDOM
    ret_score, time_score, seq_tr = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=repeats,
        time_limit=ml_time,
        reduce_trivial=True,
        relabel=True,
        problem_type="TrivialRand",
        full_leaf_set=full_leaf_set,
        progress=progress)

    for r, ret in ret_score.items():
        score.loc[i, "RetNum", r]["TrivialRand"] = copy.copy(ret)
        score.loc[i, "Time", r]["TrivialRand"] = copy.copy(time_score[r])
    tr_ret = int(min(score.loc[i, "RetNum"]["TrivialRand"]))
    df_seq = pd.concat([df_seq, pd.Series(seq_tr)], axis=1)

    # upper bound of ret
    env_info = pd.read_pickle(env_info_file)
    ub_ret = int(env_info["metadata"]["rets"])
    score.loc[i, "RetNum"]["UB"] = np.ones(repeats)*ub_ret
    # print results
    if progress:
        print()
        print("FINAL RESULTS\n"
              f"Instance = {i} \n"
              f"RETICULATIONS\n"
              f"ML            = {ml_ret}\n"
              f"TrivialML     = {ml_triv_ret}\n"
              f"Rand          = {ra_ret}\n"
              f"TrivialRand   = {tr_ret}\n"
              f"Reference     = {ub_ret}\n"
              f"ML time       = {np.round(ml_time)}s")
    else:
        print(i, ml_ret, ml_triv_ret, ra_ret, tr_ret, ub_ret, ml_time)

    # save dataframes
    # scores
    score.dropna(axis=0, how="all").to_pickle(f"Data/Results/inst_results/heuristic_scores_lgt_ML[{ml_name}]_"
                                              f"TEST[{test_info}]_"
                                              f"{i}.pickle")
    # ml predictions
    df_pred.to_pickle(f"Data/Results/inst_results/cherry_prediction_info_ML[{ml_name}]_"
                      f"TEST[{test_info}]_"
                      f"{i}.pickle")
    # best sequences
    df_seq.columns = score.columns[:-1]
    df_seq.index = pd.MultiIndex.from_product([[i], df_seq.index])
    df_seq.to_pickle(f"Data/Results/inst_results/cherry_seq_ML[{ml_name}]_"
                     f"TEST[{test_info}]_"
                     f"{i}.pickle")


if __name__ == "__main__":
    i = int(sys.argv[1])
    ml_name = str(sys.argv[2])
    l = int(sys.argv[3])
    exact_input = int(sys.argv[4])

    if exact_input:
        exact = True
        ret = int(sys.argv[5])
        forest_size = None
    else:
        exact = False
        ret = None
        forest_size = int(sys.argv[5])

    if len(sys.argv) == 7:
        ml_thresh = int(sys.argv[6])
    else:
        ml_thresh = None

    run_main(i, l, exact, ret, forest_size, ml_name=ml_name, full_leaf_set=True, ml_thresh=ml_thresh, progress=True)
