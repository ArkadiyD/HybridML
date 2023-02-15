from NetworkGen.NetworkToTree import *
from NetworkGen.LGT_network import *
from NetworkGen.tree_to_newick import *

from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import time
import sys

from NetworkGen.network_containtment import *
'''
Code used for generating test instances
'''
from TreeWidthTreeContainment import BOTCH

def make_data_fun(net_num, l=20, exact=False, ret=None, num_trees=None):
    # PARAMS OF LGT GENERATOR
    beta = 1
    distances = True

    if exact:
        tree_info = f"_L{l}_R{ret}_exact"
    else:
        tree_info = f"_L{l}_T{num_trees}"

    now = datetime.now().time()
    st = time.time()

    # make network
    network_gen_st = time.time()
    if exact:
        n = l - 2 + ret
        print('exact')
        print(f"JOB {net_num} ({now}): Start creating NETWORK (In-Sample, L = {l}, R = {ret}, n = {n})")
        while True:
            if l <= 20:
                alpha = np.random.uniform(0.1, 0.5)
            elif l <= 50:
                alpha = np.random.uniform(0.1, 0.3)
            else:
                alpha = np.random.uniform(0.1, 0.2)
            net, ret_num = simulation(n, alpha, 1, beta)
            num_leaves = len(leaves(net))
            if num_leaves == l and ret_num == ret:
                break
            else:
                print(f"JOB {net_num} ({now}): NETWORK GEN FAILED, again (In-Sample, L = {l}, R = {ret}, n = {n})")

    else:
        # randomize reticulation!
        min_ret = int(np.ceil(np.log2(num_trees)))
        max_ret = int(min([5*np.ceil(np.log2(num_trees)), 60]))
        ret = np.random.randint(min_ret, max_ret)
        n = l - 2 + ret     # preferably a reticulation number of at least 3 + minimum
        #print(min_ret, ret, max_ret)
        trials_per_n = 20
        print(f"JOB {net_num} ({now}): Start creating NETWORK (Out-of-Sample, L = {l}, T = {num_trees}, n = {n})")
        while True:
            alpha = np.random.uniform(0.3, 0.5)
            net, ret_num = simulation(n, alpha, 1, beta, ret)
            num_leaves = len(leaves(net))
            #print(ret, ret_num, num_leaves, alpha)
            if num_leaves == l:
                break
            else:
                if trials_per_n:
                    trials_per_n -= 1
                else:
                    trials_per_n = 10
                    n -= 1
                    print(f"JOB {net_num} ({now}): Start creating NETWORK (Out-of-Sample, L = {l}, T = {num_trees}, n = {n})")

            if time.time() - network_gen_st > 60*1:
                print(f"JOB {net_num} ({now}): FAILED (Out-of-Sample, L = {l}, T = {num_trees})")
                return None

    net_nodes = int(len(net.nodes))
    now = datetime.now().time()
    if exact:
        print(f"JOB {net_num} ({now}): Start creating TREE SET (L = {num_leaves}, T = {2**ret_num}, R = {ret_num})")
    else:
        print(f"JOB {net_num} ({now}): Start creating TREE SET (L = {num_leaves}, T = {num_trees}, R = {ret_num})")

    for x, y in net.edges:
        print(x, y)
    net_copy = deepcopy(net)
    #print(net_copy.edges, net.nodes)
    tree_set, tree_lvs = net_to_tree(net, num_trees, distances=distances, net_lvs=num_leaves)
    #print(net.edges, net.nodes)
    #exit(0)
    for ind in tree_set:
        for n in tree_set[ind].nodes():
            if tree_set[ind].in_degree[n] == 0:
                print(tree_set[ind].out_degree[n])
                tree_set[ind].add_edge(1000,n)
                break
        print(tree_set[ind].nodes)
    for n in net_copy.nodes():
        if net_copy.in_degree[n] == 0:
            print(net_copy.out_degree[n])
            net_copy.add_edge(1000,n)
            break
    print(net_copy.nodes, net_copy.edges)

    for tree in tree_set.values():
        #print(tree.edges, net_copy.edges, tree.nodes, net_copy.nodes)
        T = deepcopy(tree)
        N = deepcopy(net_copy)
        print('nodes',T.nodes, T.edges)
        #T.add_edge(-1,0)
        #N.add_edge(-1,0)
        # res = BOTCH.tc_brute_force(T, N)
        # print(res)

    # print('tree lvs', tree_lvs)
    # for tree_index in tree_set:
    #     net_ = deepcopy(net_copy)
    #     tree = deepcopy(tree_set[tree_index])
    #     network_containtment(net_, net_)
    #     print('-'*100)
    #     TCNContains(net_, net_)
    
    if num_trees is None:
        num_trees = 2 ** ret_num

    tree_to_newick_fun(tree_set, net_num, tree_info=tree_info)

    tree_child = is_tree_child(net)
    metadata_index = ["exact", "rets", "nodes", "net_leaves", "tree_child", "chers", "ret_chers", "trees", "n", "alpha",
                      "beta", "runtime"]

    net_cher, net_ret_cher = network_cherries(net)
    metadata = pd.Series([exact, ret_num, net_nodes, num_leaves, tree_child, len(net_cher)/2, len(net_ret_cher),
                          len(tree_set), n, alpha, beta, time.time() - st],
                         index=metadata_index,
                         dtype=float)
    print('-'*100)
    output = {"net": net, "forest": tree_set, "metadata": metadata}
    with open(
            f"Data/Test/inst_results/tree_data{tree_info}_{net_num}.pickle", "wb") as handle:
        pickle.dump(output, handle)
    now = datetime.now().time()
    if exact:
        print(f"JOB {net_num} ({now}): FINISHED in {np.round(time.time() - st, 3)}s (In-Sample, L = {num_leaves}, "
              f"R = {ret_num}, n = {n})")
    else:
        print(f"JOB {net_num} ({now}): FINISHED in {np.round(time.time() - st, 3)}s (Out-of-Sample, L = {num_leaves}, "
              f"T = {num_trees}, n = {n})")
    return output


def is_tree_child(net):
    for n in net.nodes:
        if net.out_degree(n) == 2:
            two_rets = []
            for c in net.successors(n):
                if net.out_degree(c) == 1:
                    two_rets.append(True)
                else:
                    two_rets.append(False)
            if all(two_rets):
                return False
        elif net.out_degree(n) == 1:
            for c in net.successors(n):
                if net.out_degree(c) == 1:
                    return False
    return True


if __name__ == "__main__":
    net_num = int(sys.argv[1])
    l = int(sys.argv[2])
    exact_input = int(sys.argv[3])

    if exact_input:
        exact = True
        ret = int(sys.argv[4])
        num_trees = None
    else:
        exact = False
        ret = None
        num_trees = int(sys.argv[4])

    make_data_fun(net_num, l, exact, ret, num_trees)
