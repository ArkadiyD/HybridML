from Features import Features
from CPH import *
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import copy


# return reticulation nodes
def reticulations(G):
    return [v for v in G.nodes() if G.in_degree(v) == 2]


# for non-binary give ret number per reticulation node
def reticulations_non_binary(G):
    return [G.in_degree(i)-1 for i in G.nodes if G.in_degree(i) >= 2]


# return leaves from network
def leaves(net):
    return {u for u in net.nodes() if net.out_degree(u) == 0}


# MAKE TREES FROM NETWORK
def net_to_tree(net, num_trees=None, distances=True, partial=False, net_lvs=None):
    # we only consider binary networks here
    tree_set = dict()
    rets = reticulations(net)
    ret_num = len(rets)
    if net_lvs is not None:
        tree_lvs = []
    if ret_num == 0:
        return False

    if num_trees is None:
        ret_dels_tmp = itertools.product(*[np.arange(2)]*ret_num)
        ret_dels = None
        for opt in ret_dels_tmp:
            opt = np.array(opt).reshape([1, -1])
            try:
                ret_dels = np.vstack([ret_dels, opt])
            except:
                ret_dels = opt
    else:
        ret_dels_set = set()
        its = 0
        while len(ret_dels_set) < num_trees:
            ret_dels_set.add(tuple(np.random.randint(0, 2, ret_num)))
            its +=1
        ret_dels = np.array([list(opt) for opt in ret_dels_set])

    t = 0
    for opt in ret_dels:
        if opt[0] is None:
            continue
        tree = copy.deepcopy(net)
        for i in np.arange(ret_num):
            if opt[i] is None:
                continue
            ret = rets[i]
            # check if reticulation still has indegree 2!
            if tree.in_degree(ret) < 2:
                continue
            ret_pre_both = list(tree.pred[ret]._atlas.keys())
            ret_pre_del = ret_pre_both[opt[i]]
            ret_pre_other = ret_pre_both[1 - opt[i]]  # if binary
            # delete reticulation edge
            tree.remove_edge(ret_pre_del, ret)
            # delete node in and out degree = 1
            if tree.in_degree(ret_pre_del) == 1 and tree.out_degree(ret_pre_del) == 1:
                node_pre = list(tree.pred[ret_pre_del]._atlas.keys())[0]

                try:
                    pre_len = tree.edges[(node_pre, ret_pre_del)]["length"]
                except KeyError:
                    pre_len = tree.edges[(node_pre, ret_pre_del)]["lenght"]

                node_succ = list(tree.succ[ret_pre_del]._atlas.keys())[0]

                try:
                    succ_len = tree.edges[(ret_pre_del, node_succ)]["length"]
                except KeyError:
                    succ_len = tree.edges[(ret_pre_del, node_succ)]["lenght"]

                # remove parent of reticulation edge
                tree.remove_node(ret_pre_del)
                # add edge from parent to child
                tree.add_edge(node_pre, node_succ, lenght=pre_len + succ_len)

            # regraft remaining edge
            try:
                pre_len = tree.edges[(ret_pre_other, ret)]["length"]
            except KeyError:
                pre_len = tree.edges[(ret_pre_other, ret)]["lenght"]

            node_succ = list(tree.succ[ret]._atlas.keys())[0]

            try:
                succ_len = tree.edges[(ret, node_succ)]["length"]
            except KeyError:
                succ_len = tree.edges[(ret, node_succ)]["lenght"]

            # remove reticulation node
            tree.remove_node(ret)
            # add edge from parent to child of reticulation
            tree.add_edge(ret_pre_other, node_succ, length=pre_len + succ_len)

            # check if out_degree of root == 1
            if tree.out_degree(0) == 1:
                for c in tree.successors(0):
                    child = c
                tree.remove_node(0)
                tree = nx.relabel_nodes(tree, {child: 0})
        # if partial, delete random leaves from tree
        if max([tree.in_degree(l) for l in tree.nodes]) > 1:
            print("Bad tree before partial")
            continue
        elif net_lvs is not None:
            tree_lvs.append(net_lvs)

        if partial:
            lvs = leaves(tree)
            lower_bound = 0
            upper_bound = int(np.floor(0.9*len(lvs)))
            mean = 0
            var = 0.25*(upper_bound - mean)
            num_del_tmp = int(np.floor(np.random.normal(mean, var)))
            if num_del_tmp < lower_bound:
                num_del = lower_bound
            elif num_del_tmp > upper_bound:
                num_del = upper_bound
            else:
                num_del = num_del_tmp
            # num_del = np.random.randint(lower_bound, upper_bound + 1)
            if net_lvs is not None:
                tree_lvs[-1] -= num_del
            if num_del > 0:
                # select leaves to delete
                lvs_del = list(np.random.choice(lvs, num_del, replace=False))
                # delete leaves
                for l in lvs_del:
                    tree.remove_node(l)
                while True:
                    # delete nodes with outdegree zero and not part of leave
                    out_zero_nodes = [n for n in tree.nodes if (tree.out_degree(n) == 0 and n not in lvs)]
                    for n in out_zero_nodes:
                        tree.remove_node(n)

                    # delete nodes with in and outdegree 1
                    in_out_one_nodes = [n for n in tree.nodes if (tree.in_degree(n) == 1 and tree.out_degree(n) == 1)]
                    if not out_zero_nodes and not in_out_one_nodes:
                        break
                    for n in in_out_one_nodes:
                        for p in tree.predecessors(n):
                            pn = p
                        for c in tree.successors(n):
                            cn = c
                        # add new edge
                        try:
                            len_1 = tree.edges[(pn, n)]["length"]
                        except KeyError:
                            len_1 = tree.edges[(pn, n)]["lenght"]

                        try:
                            len_2 = tree.edges[(n, cn)]["length"]
                        except KeyError:
                            len_2 = tree.edges[(n, cn)]["lenght"]

                        tree.add_edge(pn, cn, length=len_1 + len_2)
                        tree.remove_node(n)

        add_node_attributes(tree, distances=distances, root=0)
        if max([tree.in_degree(l) for l in tree.nodes]) == 1:
            tree_set[t] = tree
            t += 1
            if net_lvs is not None:
                tree_lvs[-1] = tree_lvs[-1]/net_lvs
        else:
            print("Bad tree after partial")
            if net_lvs is not None:
                tree_lvs.pop(-1)
    if net_lvs is not None:
        return tree_set, np.array(tree_lvs)
    else:
        return tree_set


# REDUCE TREES FROM NETWORK
def net_to_reduced_trees(net, num_red=1, num_rets=0, num_net=0, distances=True, comb_measure=True, net_lvs=None):
    # extract trees from network
    if num_rets == 0:
        tree = deepcopy(net)
        add_node_attributes(tree, distances=distances, root=0)
        tree_set = {0: tree}
    else:
        tree_set, _ = net_to_tree(net, distances=distances, net_lvs=net_lvs)

    # make network and forest environments
    net_env = deepcopy(PhN(net))
    init_forest_env = Input_Set(tree_set=tree_set, leaves=net_env.leaves)
    forest_env = deepcopy(init_forest_env)
    # get cherries from network and forest
    net_cher, net_ret_cher = network_cherries(net_env.nw)
    reducible_pairs = forest_env.find_all_pairs()

    # output information
    num_cher = [len(net_cher)]
    num_ret_cher = [len(net_ret_cher)]
    tree_set_num = [len(forest_env.trees)]

    # features
    features = Features(reducible_pairs, forest_env.trees, root=0)

    # create input X and output Y data
    X = deepcopy(features.data)
    # change index of X
    X_index = [f"{c}_{num_net}" for c in X.index]
    X.index = X_index
    CPS = []
    Y = cherry_labels(net_cher, net_ret_cher, list(reducible_pairs), X.index, num_net)
    # now, reduce tree_set and net at the same time to get labelled data!
    # ret_happened = False
    tree_child = True
    for r in np.arange(num_red):
        triv_picked = False
        # pick random cherry
        cherry_in_all = {c for c, trees in reducible_pairs.items() if len(trees) == len(tree_set)}.intersection(net_cher)
        pickable_chers = net_ret_cher.intersection(set(reducible_pairs))
        if cherry_in_all:    # if any cherry in all, reduce that one first
            chosen_cherry = list(cherry_in_all)[np.random.choice(len(cherry_in_all))]
        elif net_cher:  # otherwise, pick trivial cherry, with relabelling
            chosen_cherry = list(net_cher)[np.random.choice(len(net_cher))]
            # check if we need to relabel
            try:
                triv_check = any([chosen_cherry[0] in tree.leaves for t, tree in forest_env.trees.items() if t not in reducible_pairs[chosen_cherry]])
            except:
                print("CHERRY IN NETWORK NOT A CHERRY IN THE FOREST")
                tree_child = False
                break
            if triv_check:
                triv_picked = True
            else:
                triv_picked = False
        else:                # reticulate cherries
            try:
                chosen_cherry = list(pickable_chers)[np.random.choice(len(pickable_chers))]
            except ValueError:
                print("NO PICKABLE CHERRIES")
                tree_child = False
                break
        CPS.append(chosen_cherry)

        if triv_picked:
            reducible_pairs, merged_cherries = forest_env.relabel_trivial(*chosen_cherry, reducible_pairs)
            features.relabel_trivial_features(*chosen_cherry, reducible_pairs, merged_cherries, forest_env.trees)

        # update some features before picking
        features.update_cherry_features_before(chosen_cherry, reducible_pairs, forest_env.trees)
        # reduce trees with chosen cherry
        new_reduced = forest_env.reduce_pair_in_all(chosen_cherry, reducible_pairs=reducible_pairs)
        forest_env.update_node_comb_length(*chosen_cherry, new_reduced)

        if any([any([trees.nw.in_degree(n) == 2 for n in trees.nw.nodes]) for t, trees in
                forest_env.trees.items()]):
            print("RET HAPPENED")
            tree_child = False
            break

        reducible_pairs = forest_env.update_reducible_pairs(reducible_pairs, new_reduced)
        if len(forest_env.trees) == 0:
            break
        # update features after picking
        features.update_cherry_features_after(chosen_cherry, reducible_pairs, forest_env.trees, new_reduced)
        net_env.reduce_pair(*chosen_cherry)
        net_cher, net_ret_cher = network_cherries(net_env.nw)

        # output information
        num_cher += [len(net_cher)/2]
        num_ret_cher += [len(net_ret_cher)]
        tree_set_num += [len(forest_env.trees)]

        # in and output cherries
        X_new = deepcopy(features.data)
        # change index of X
        X_index = [f"{c}_{num_net}" for c in X_new.index]
        X_new.index = X_index
        Y_new = cherry_labels(net_cher, net_ret_cher, list(reducible_pairs), X_new.index, num_net)

        X = pd.concat([X, X_new])
        Y = pd.concat([Y, Y_new])

    return X, Y, num_cher, num_ret_cher, tree_set_num, tree_child


def test_trivial(net, num_red=1, num_rets=0, num_net=0, distances=True, comb_measure=True, net_lvs=None):
    # extract trees from network
    if num_rets == 0:
        tree = deepcopy(net)
        add_node_attributes(tree, distances=distances, root=0)
        tree_set = {0: tree}
    else:
        tree_set, _ = net_to_tree(net, distances=distances, net_lvs=net_lvs)

    # make network and forest environments
    net_env = deepcopy(PhN(net))
    init_forest_env = Input_Set(tree_set=tree_set, leaves=net_env.leaves)
    forest_env = deepcopy(init_forest_env)
    # get cherries from network and forest
    net_cher, net_ret_cher = network_cherries(net_env.nw)
    reducible_pairs = forest_env.find_all_pairs()

    # output information
    num_cher = [len(net_cher)]
    num_ret_cher = [len(net_ret_cher)]
    tree_set_num = [len(forest_env.trees)]

    CPS = []
    shite = False
    relabelled = False
    for r in np.arange(num_red):
        triv_picked = False
        # pick random cherry
        cherry_in_all = {c for c, trees in reducible_pairs.items() if len(trees) == len(tree_set)}.intersection(net_cher)
        pickable_chers = net_ret_cher.intersection(set(reducible_pairs))
        if cherry_in_all:    # if any cherry in all, reduce that one first
            chosen_cherry = list(cherry_in_all)[np.random.choice(len(cherry_in_all))]
        elif net_cher:  # otherwise, pick trivial cherry, with relabelling
            chosen_cherry = list(net_cher)[np.random.choice(len(net_cher))]
            # check if we need to relabel
            try:
                triv_check = any([chosen_cherry[0] in tree.leaves for t, tree in forest_env.trees.items() if t not in reducible_pairs[chosen_cherry]])
            except:
                print("CHERRY IN NETWORK NOT A TREE IN THE FOREST")
                break
            if triv_check:
                triv_picked = True
            else:
                triv_picked = False
        else:                # reticulate cherries
            try:
                chosen_cherry = list(pickable_chers)[np.random.choice(len(pickable_chers))]
            except ValueError:
                print("NO PICKABLE CHERRIES")
                break
        CPS.append(chosen_cherry)

        if triv_picked:
            reducible_pairs, merged_cherries = forest_env.relabel_trivial(*chosen_cherry, reducible_pairs)
            relabelled = True
            # print(f"relabel {chosen_cherry[0]} to {chosen_cherry[1]}")

        # reduce trees with chosen cherry
        new_reduced = forest_env.reduce_pair_in_all(chosen_cherry, reducible_pairs=reducible_pairs)

        if any([any([trees.nw.in_degree(n) == 2 for n in trees.nw.nodes]) for t, trees in
                forest_env.trees.items()]):
            print(f"CH{chosen_cherry}. RET HAPPENED")
            break

        reducible_pairs = forest_env.update_reducible_pairs(reducible_pairs, new_reduced)
        if len(forest_env.trees) == 0:
            break
        prev_net_edges = copy.deepcopy(net_env.nw.edges)
        net_env.reduce_pair(*chosen_cherry)
        # print("\nnet")
        # for x, y in net_env.nw.edges:
        #     print(x, y)
        net_cher, net_ret_cher = network_cherries(net_env.nw)

        # for all net_cher, check if triv, also for net_ret_cher
        # print(f"\nChosen cherry {chosen_cherry}")
        for c, trees in reducible_pairs.items():
            cherry_in_all = len(trees) == len(tree_set)
            if cherry_in_all:
                if c in net_cher:
                    continue
                else:
                    print(f"CH{chosen_cherry}. {c}: IN ALL BUT NOT CHER IN NETWORK")
                    shite = True
            triv_check = all([set(c).issubset(tree.leaves) == False for t, tree in forest_env.trees.items() if t not in trees])
            if triv_check and c in net_cher:
                # print(f"CH{chosen_cherry}. {c}: TRIV AND CHER")
                pass
            elif (triv_check and c in net_ret_cher) or (triv_check and c[::-1] in net_ret_cher):
                print(f"CH{chosen_cherry}. {c}: TRIV AND RETCHER")
                print(f"CPS UP TO NOW = {CPS}")
                shite = True
            elif triv_check:
                print(f"CH{chosen_cherry}. {c}: TRIV BUT NOT IN NETWORK")
                shite = True
        if shite:
            break

        # output information
        num_cher += [len(net_cher)/2]
        num_ret_cher += [len(net_ret_cher)]
        tree_set_num += [len(forest_env.trees)]
    if shite:
        if relabelled:
            print("SHITE and RELABELLED")
        else:
            print("SHITE")
        print(f"reducible = {set(reducible_pairs)}")
        print(f"chers = {net_cher}")
        print(f"ret_chers = {net_ret_cher}")
        print(f"\nCHOSEN CHERRY = {chosen_cherry}")
        print("\nprev network")
        for x, y in prev_net_edges:
            print(x, y)
        print("\nnetwork")
        for x, y in net_env.nw.edges:
            print(x, y)
        print("\nstarting network")
        for x, y in net.edges:
            print(x, y)
        print("\nforest")
        for t, tree in tree_set.items():
            print(t)
            for x, y in tree.edges:
                print(x, y)
    return num_cher, num_ret_cher, tree_set_num


# FIND CHERRIES AND RETICULATED CHERRIES
def network_cherries(net):
    cherries = set()
    retic_cherries = set()
    lvs = leaves(net)

    for l in lvs:
        for p in net.pred[l]:
            if net.out_degree(p) > 1:
                for cp in net.succ[p]:
                    if cp == l:
                        continue
                    if cp in lvs:
                        cherries.add((l, cp))
                        cherries.add((cp, l))
                    elif net.in_degree(cp) > 1:
                        for ccp in net.succ[cp]:
                            if ccp in lvs:
                                retic_cherries.add((ccp, l))

    return cherries, retic_cherries


def tree_cherries(tree_set):
    cherries = set()
    reducible_pairs = dict()
    t = 0
    for tree in tree_set.values():
        lvs = leaves(tree)

        for l in lvs:
            for p in tree.pred[l]:
                if tree.out_degree(p) > 1:
                    for cp in tree.succ[p]:
                        if cp == l:
                            continue
                        if cp in lvs:
                            cherry = (l, cp)
                            cherries.add(cherry)
                            cherries.add(cherry[::-1])

                            # add tree to cherry
                            if cherry not in reducible_pairs:
                                reducible_pairs[cherry] = {t}
                                reducible_pairs[cherry[::-1]] = {t}
                            else:
                                reducible_pairs[cherry].add(t)
                                reducible_pairs[cherry[::-1]].add(t)
        t += 1
    return cherries, reducible_pairs


# check if cherry is reducible
def is_cherry(tree, x, y):
    lvs = leaves(tree)
    if (x not in lvs) or (y not in lvs):
        return False
    # tree, so no reticulations
    px = tree.pred[x]._atlas.keys()
    py = tree.pred[y]._atlas.keys()
    return px == py


def is_ret_cherry(net, x, y):
    for p in net.pred[y]:
        if net.out_degree(p) > 1:
            for cp in net.succ[p]:
                if cp == y:
                    continue
                if net.in_degree(cp) > 1:
                    for ccp in net.succ[cp]:
                        if ccp == x:
                            return True
    return False


# CHERRY LABELS
def cherry_labels(net_cher, net_ret_cher, tree_cher, index, num_net=0):
    # LABELS
    df_labels = pd.DataFrame(0, index=index, columns=np.arange(4), dtype=np.int8)
    for c in tree_cher:
        # cherry in network
        if c in net_cher:
            df_labels.loc[f"{c}_{num_net}", 1] = 1
        elif c in net_ret_cher:
            df_labels.loc[f"{c}_{num_net}", 2] = 1
        elif c[::-1] in net_ret_cher:
            df_labels.loc[f"{c}_{num_net}", 3] = 1
        else:
            df_labels.loc[f"{c}_{num_net}", 0] = 1
    return df_labels


def add_node_attributes(tree, distances=True, root=0):
    attrs = dict()
    for x in tree.nodes:
            if distances:
                try:
                    attrs[x] = {"node_length": nx.algorithms.shortest_paths.generic.shortest_path_length(tree, root, x, weight="length"),
                                "node_comb": nx.algorithms.shortest_paths.generic.shortest_path_length(tree, root, x)}
                except nx.exception.NetworkXNoPath:
                    attrs[x] = {"node_length": None, "node_comb": None}
            else:
                try:
                    attrs[x] = {"node_comb": nx.algorithms.shortest_paths.generic.shortest_path_length(tree, root, x)}
                except nx.exception.NetworkXNoPath:
                    attrs[x] = {"node_comb": None}

    nx.set_node_attributes(tree, attrs)


# FINAL DATA GENERATION
def data_gen(net, tree_set, str_features=None, num_net=0, distances=True, comb_measure=True):
    # cherries of network
    net_cher, net_ret_cher = network_cherries(net)
    # cherries of trees
    tree_cher, reducible_pairs = tree_cherries(tree_set)

    # in and output cherries
    # old method
    features = Features(reducible_pairs, tree_set, str_features, root=0, distances=distances, comb_measure=comb_measure)
    X = features.data

    # change index of X
    X_index = [f"{c}_{num_net}" for c in X.index]
    X.index = X_index

    Y = cherry_labels(net_cher, net_ret_cher, tree_cher, X.index, num_net)
    return X, Y, len(net_cher), len(net_ret_cher)
