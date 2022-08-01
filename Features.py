import networkx as nx
import pandas as pd
import numpy as np
import copy

'''
CODE FOR FEATURES:
Features CLASS: The parent class of all other child feature classes. 
                Mainly used in CPH.py and train_data_gen.py. This class calls the child classes for the following tasks:
                - init_fun:                         Initialize all features for the input tree set
                - new_cherries_fun:                 Get features of new found cherries
                - update_cherry_features_before:    Update the cherry features before reducing the chosen cherry 
                                                    in the tree set
                - update_cherry_features_after:     Update some cherry features after reducing the chosen cherry
                - chosen_cherry_cleaning:           After reducing the chosen cherry, delete attributes attached to this 
                                                    cherry
                - update_data:                      After updating the features, update the total input data
                - relabel_trivial_features          If a trivial feature is chosen, and relabelling needs to happen,
                                                    this function calls all child classes to relabel the features accordingly
                                                    
ALL FEATURE CHILD CLASSES: 
TreeHeight, LeafPair, Trivial, CherryHeight, RedAfterPick, LeafDist, LeafHeight.

They all have a similar structure: 
                - init_fun:                         Initialize all features for the input tree set based on this feature
                - new_cherries:                     Get feature values of new found cherries
                - update:                           Update the current cherries' feature values
                - chosen_cherry_cleaning:           After reducing the chosen cherry, delete attributes attached to it
                - relabel                           Relabel certain cherries

'''


# FEATURE CLASS
class Features:
    def __init__(self, reducible_pairs, tree_set, root=2):
        # PARAMETERS
        self.num_trees = len(tree_set)
        # FEATURE CLASSES
        # independent of height
        self.name = []
        self.data = pd.DataFrame(dtype=float)

        self.leaf_pair = LeafPair(self.num_trees)
        self.name += self.leaf_pair.name

        self.trivial = Trivial()
        self.name += self.trivial.name

        self.red_after_pick = RedAfterPick(root)
        self.name += self.red_after_pick.name

        # tree height
        self.tree_height = TreeHeight(root)
        self.name += self.tree_height.name

        # dependent of height
        self.cherry_height = CherryHeight()
        self.name += self.cherry_height.name

        self.leaf_dist = LeafDist()
        self.name += self.leaf_dist.name

        self.leaf_height = LeafHeight()
        self.name += self.leaf_height.name

        # RUN INIT FUN
        self.init_fun(reducible_pairs, tree_set)

    def init_fun(self, reducible_pairs, tree_set):
        self.data = pd.DataFrame(index=reducible_pairs, columns=self.name, dtype=float)
        unique_cherries = set([tuple(sorted(c)) for c in reducible_pairs])
        # LEAF PAIRS
        new_data_column = self.leaf_pair.init_fun(reducible_pairs, tree_set)
        self.data[self.leaf_pair.name] = new_data_column

        # TRIVIAL
        new_data_column = self.trivial.init_fun(reducible_pairs, tree_set, unique_cherries, self.num_trees,
                                                self.leaf_pair.n)
        self.data[self.trivial.name] = new_data_column

        # TREE HEIGHT
        new_data_column = self.tree_height.init_fun(tree_set, reducible_pairs, unique_cherries, self.trivial.n)
        self.data[self.tree_height.name] = new_data_column

        # REDUCTION AFTER PICK
        new_data_column = self.red_after_pick.init_fun(reducible_pairs, tree_set, unique_cherries, self.trivial.n)
        self.data[self.red_after_pick.name] = new_data_column

        # HEIGHT DEPENDENT FEATURES
        # cherry height
        new_data_column = self.cherry_height.init_fun(reducible_pairs, tree_set, unique_cherries,
                                                      self.tree_height.dist, self.tree_height.comb,
                                                      self.trivial.n)
        self.data[self.cherry_height.name] = new_data_column

        # leaf distance
        new_data_column = self.leaf_dist.init_fun(reducible_pairs, tree_set, unique_cherries,
                                                  self.tree_height.dist, self.tree_height.comb,
                                                  self.leaf_pair.leaf_pair_cont_tree, self.leaf_pair.n)
        self.data[self.leaf_dist.name] = new_data_column

        # leaf height
        new_data_column = self.leaf_height.init_fun(reducible_pairs, tree_set, unique_cherries,
                                                    self.tree_height.dist, self.tree_height.comb,
                                                    self.leaf_pair.leaf_pair_cont_tree, self.leaf_pair.n)
        self.data[self.leaf_height.name] = new_data_column

        return None

    def new_cherries_fun(self, reducible_pairs, tree_set, new_cherries):
        if len(new_cherries) == 0:
            return new_cherries

        new_reducible_pairs = {c: reducible_pairs[c] for c in new_cherries}
        unique_new_cherries = set([tuple(sorted(c)) for c in new_cherries])

        # LEAF PAIRS
        self.leaf_pair.new_cherries(new_reducible_pairs, tree_set)

        # trivial
        self.trivial.new_cherries(new_reducible_pairs, unique_new_cherries, self.leaf_pair.leaf_pair_cont_tree,
                                  self.leaf_pair.n)

        # TREE HEIGHT
        self.tree_height.new_cherries(new_reducible_pairs, unique_new_cherries, self.trivial.n)

        # HEIGHT INDEPENDENT FEATURES
        # reduction after pick
        self.red_after_pick.new_cherries(new_reducible_pairs, tree_set, unique_new_cherries, self.tree_height.d)

        # HEIGHT DEPENDENT FEATURES
        # cherry height
        self.cherry_height.new_cherries(new_reducible_pairs, tree_set, unique_new_cherries,
                                        self.tree_height.dist, self.tree_height.comb, self.trivial.n)

        # leaf distance
        self.leaf_dist.new_cherries(new_reducible_pairs, tree_set, unique_new_cherries,
                                    self.tree_height.dist, self.tree_height.comb,
                                    self.leaf_pair.leaf_pair_cont_tree, self.leaf_pair.n)

        # leaf height
        self.leaf_height.new_cherries(new_reducible_pairs, tree_set, unique_new_cherries,
                                      self.tree_height.dist, self.tree_height.comb,
                                      self.leaf_pair.leaf_pair_cont_tree, self.leaf_pair.n)

    # UPDATE FUNCTIONS
    def update_cherry_features_before(self, chosen_cherry, reducible_pairs, tree_set):
        # leaf distance
        self.leaf_dist.update_before(chosen_cherry, tree_set, reducible_pairs)
        # leaf height
        self.leaf_height.update_before(chosen_cherry, tree_set, reducible_pairs)

    def update_cherry_features_after(self, chosen_cherry, reducible_pairs, tree_set, new_reduced):
        new_cherries = copy.deepcopy(set(reducible_pairs) - set(self.data.index))
        unique_reducible_pairs = {tuple(sorted(c)): trees for c, trees in reducible_pairs.items()}
        num_trees_changed = False
        for t in new_reduced:
            if t not in tree_set:
                self.num_trees -= 1
                num_trees_changed = True
        # update leaf pair
        change_leaf_pair = self.leaf_pair.update(chosen_cherry, new_cherries, new_reduced, reducible_pairs,
                                                 self.num_trees, num_trees_changed)

        # update trivial
        change_cherry_in_forest = self.trivial.update(new_reduced, reducible_pairs, tree_set, new_cherries,
                                                      self.leaf_pair.leaf_pair_cont_tree, change_leaf_pair)

        # update tree height
        change_height_dist, change_height_comb = self.tree_height.update(chosen_cherry, new_cherries, new_reduced,
                                                                         tree_set,
                                                                         self.cherry_height.comb,
                                                                         self.cherry_height.dist,
                                                                         change_cherry_in_forest,
                                                                         reducible_pairs)

        # update cherry height
        self.cherry_height.update(reducible_pairs, tree_set, new_cherries,
                                  self.tree_height.dist, self.tree_height.comb,
                                  change_cherry_in_forest, change_height_dist, change_height_comb)

        # update red after pick
        self.red_after_pick.update(chosen_cherry, new_reduced, unique_reducible_pairs, tree_set, new_cherries,
                                   change_cherry_in_forest)

        # new cherries
        self.new_cherries_fun(reducible_pairs, tree_set, new_cherries)

        # update_data of the features already updated
        self.leaf_dist.update_data(new_cherries, unique_reducible_pairs, self.tree_height.dist, self.tree_height.comb,
                                   change_height_dist, change_height_comb, change_leaf_pair, tree_set)

        self.leaf_height.update_data(new_cherries, unique_reducible_pairs, self.tree_height.dist, self.tree_height.comb,
                                     change_height_dist, change_height_comb,
                                     change_leaf_pair, tree_set)

        self.chosen_cherry_cleaning(chosen_cherry)

        self.update_data(new_cherries)
        return None

    def chosen_cherry_cleaning(self, chosen_cherry):
        self.data.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        # LEAF PAIRS
        self.leaf_pair.chosen_cherry_cleaning(chosen_cherry)

        # TRIVIAL
        self.trivial.chosen_cherry_cleaning(chosen_cherry)

        # TREE HEIGHT
        self.tree_height.chosen_cherry_cleaning(chosen_cherry)

        # REDUCTION AFTER PICK
        self.red_after_pick.chosen_cherry_cleaning(chosen_cherry)

        # HEIGHT DEPENDENT FEATURES
        # cherry height
        self.cherry_height.chosen_cherry_cleaning(chosen_cherry)

        # leaf distance
        self.leaf_dist.chosen_cherry_cleaning(chosen_cherry)

        # leaf height
        self.leaf_height.chosen_cherry_cleaning(chosen_cherry)

    def update_data(self, new_cherries):
        # self.data = self.data.append(pd.DataFrame(index=new_cherries))
        self.data = pd.concat([self.data, pd.DataFrame(index=new_cherries, dtype=float)])
        # LEAF PAIRS
        self.data[self.leaf_pair.name] = self.leaf_pair.data_column
        # TRIVIAL
        self.data[self.trivial.name] = self.trivial.data_column
        # TREE HEIGHT
        self.data[self.tree_height.name] = self.tree_height.data_column
        # REDUCTION AFTER PICK
        self.data[self.red_after_pick.name] = self.red_after_pick.data_column

        # HEIGHT DEPENDENT FEATURES
        # cherry height
        self.data[self.cherry_height.name] = self.cherry_height.data_column
        # leaf distance
        self.data[self.leaf_dist.name] = self.leaf_dist.data_column
        # leaf height
        self.data[self.leaf_height.name] = self.leaf_height.data_column

    # RELABEL
    def relabel_trivial_features(self, x, y, reducible_pairs, merged_cherries, tree_set):
        # LEAF PAIRS
        any_relabelled, leaf_pair_in_new_trees = self.leaf_pair.relabel(x, y, reducible_pairs, merged_cherries, tree_set)

        if not any_relabelled:
            return None

        # initialize data
        self.data = pd.DataFrame(index=reducible_pairs, columns=self.name)
        # leaf pair data
        self.data[self.leaf_pair.name] = self.leaf_pair.data_column
        # TRIVIAL
        self.trivial.relabel(x, y, reducible_pairs, merged_cherries, leaf_pair_in_new_trees)
        self.data[self.trivial.name] = self.trivial.data_column

        # TREE HEIGHT
        self.tree_height.relabel(x, y, reducible_pairs, merged_cherries)
        self.data[self.tree_height.name] = self.tree_height.data_column

        # REDUCTION AFTER PICK
        self.red_after_pick.relabel(x, y, reducible_pairs, merged_cherries)
        self.data[self.red_after_pick.name] = self.red_after_pick.data_column

        # HEIGHT DEPENDENT FEATURES
        # cherry height
        self.cherry_height.relabel(x, y, reducible_pairs, merged_cherries)
        self.data[self.cherry_height.name] = self.cherry_height.data_column

        # leaf distance
        self.leaf_dist.relabel(x, y, reducible_pairs, merged_cherries, tree_set, leaf_pair_in_new_trees)
        self.data[self.leaf_dist.name] = self.leaf_dist.data_column

        # leaf height
        self.leaf_height.relabel(x, y, reducible_pairs, merged_cherries, tree_set, leaf_pair_in_new_trees)
        self.data[self.leaf_height.name] = self.leaf_height.data_column

        return None


# TREE HEIGHT
class TreeHeight:
    def __init__(self, root=2):
        # PARAMETERS
        self.name = ["tree_height", "tree_height_comb"]
        self.data_column = pd.DataFrame(dtype=float)
        self.root = root
        # distances
        self.tree_level_width_dist = {}
        self.dist = pd.Series(dtype=float)
        self.prev_dist = pd.Series(dtype=float)
        self.dist_n = pd.Series(dtype=float)
        self.max_dist = 0

        # combinatorial
        self.tree_level_width_comb = {}
        self.comb = pd.Series(dtype=float)
        self.prev_comb = pd.Series(dtype=float)
        self.comb_n = pd.Series(dtype=float)
        self.max_comb = 0

        self.d = pd.Series(dtype=float)

    def init_fun(self, tree_set, reducible_pairs, unique_cherries, cherry_in_forest):
        self.dist = pd.Series(index=tree_set, dtype=float)
        self.comb = pd.Series(index=tree_set, dtype=float)

        for t, tree in tree_set.items():
            try:
                tree_input = tree.nw
                tree_leaves = tree.leaves
            except AttributeError:
                tree_input = tree
                tree_leaves = [u for u in tree.nodes() if tree.out_degree(u) == 0]

            self.tree_level_width_comb[t] = {}
            self.tree_level_width_dist[t] = {}
            for n in tree_input.nodes:
                # skip leaves!
                if n in tree_leaves or (self.root == 2 and n == 1):
                    continue
                # COMBINATORIAL
                if tree_input.nodes[n]["node_comb"] in self.tree_level_width_comb[t]:
                    self.tree_level_width_comb[t][tree_input.nodes[n]["node_comb"]] += 1
                else:
                    self.tree_level_width_comb[t][tree_input.nodes[n]["node_comb"]] = 1
                # DISTANCES
                height = np.round(tree_input.nodes[n]["node_length"], 3)
                if height in self.tree_level_width_dist[t]:
                    self.tree_level_width_dist[t][height] += 1
                else:
                    self.tree_level_width_dist[t][height] = 1
            # find max
            # COMBINATORIAL
            self.tree_level_width_comb[t] = dict(sorted(self.tree_level_width_comb[t].items()))
            self.comb[t] = list(self.tree_level_width_comb[t])[-1]
            # DISTANCES
            self.tree_level_width_dist[t] = dict(sorted(self.tree_level_width_dist[t].items()))
            self.dist[t] = list(self.tree_level_width_dist[t])[-1]

        self.prev_dist = copy.copy(self.dist)
        self.prev_comb = copy.copy(self.comb)
        self.max_dist = self.dist.max()
        self.max_comb = self.comb.max()
        self.init_data(reducible_pairs, unique_cherries, cherry_in_forest, new_cherries=False)
        return self.data_column

    def init_data(self, reducible_pairs, unique_cherries, cherry_in_forest, new_cherries=True):
        if not new_cherries:
            self.data_column = pd.DataFrame(index=reducible_pairs, columns=self.name, dtype=float)
            self.dist_n = pd.Series(index=reducible_pairs, dtype=float)
            self.comb_n = pd.Series(index=reducible_pairs, dtype=float)
            self.d = cherry_in_forest

        for c in unique_cherries:
            trees = list(reducible_pairs[c])

            self.dist_n[c] = self.dist[trees].sum()
            self.dist_n[c[::-1]] = self.dist_n[c]

            if self.max_dist:
                tree_dist_val = self.dist_n[c] / self.max_dist / self.d[c]
            else:
                tree_dist_val = 0
            self.data_column.loc[c, "tree_height"] = tree_dist_val
            self.data_column.loc[c[::-1], "tree_height"] = tree_dist_val

            self.comb_n[c] = self.comb[trees].sum()
            self.comb_n[c[::-1]] = self.comb_n[c]

            if self.max_comb:
                tree_comb_val = self.comb_n[c] / self.max_comb / self.d[c]
            else:
                tree_comb_val = 0
            self.data_column.loc[c, "tree_height_comb"] = tree_comb_val
            self.data_column.loc[c[::-1], "tree_height_comb"] = tree_comb_val
        return None

    def new_cherries(self, reducible_pairs, unique_cherries, cherry_in_forest):
        self.init_data(reducible_pairs, unique_cherries, cherry_in_forest, new_cherries=True)

    def update(self, chosen_cherry, new_cherries, new_reduced, tree_set, cherry_height_comb, cherry_height_dist,
               change_cherry_in_forest, reducible_pairs):
        change_height_dist = pd.Series(0, index=tree_set, dtype=float)
        change_height_comb = pd.Series(0, index=tree_set, dtype=float)
        change_height_dist_bool = False
        change_height_comb_bool = False
        for t in new_reduced:
            if t not in tree_set:
                continue
            # COMBINATORIAL
            height = cherry_height_comb.loc[chosen_cherry][t]
            if self.tree_level_width_comb[t][height] == 1:
                change_height_comb[t] = 1
                change_height_comb_bool = True
                del self.tree_level_width_comb[t][height]
                if self.tree_level_width_comb[t]:
                    self.comb[t] = list(self.tree_level_width_comb[t])[-1]
                else:
                    self.comb[t] = 0
            else:
                self.tree_level_width_comb[t][height] -= 1
            # DISTANCES
            height = np.round(cherry_height_dist.loc[chosen_cherry][t], 3)
            if self.tree_level_width_dist[t][height] == 1:
                change_height_dist[t] = 1
                change_height_dist_bool = True
                del self.tree_level_width_dist[t][height]
                if self.tree_level_width_dist[t]:
                    self.dist[t] = list(self.tree_level_width_dist[t])[-1]
                else:
                    self.dist[t] = 0
            else:
                self.tree_level_width_dist[t][height] -= 1

        # update max height level!
        change_max_dist = False
        change_max_comb = False
        if change_height_dist_bool:
            new_max_dist = self.dist.max()
            if abs(new_max_dist - self.max_dist) > 10e-3:
                self.max_dist = new_max_dist
                change_max_dist = True
        if change_height_comb_bool:
            new_max_comb = self.comb.max()
            if abs(new_max_comb - self.max_comb) > 10e-3:
                self.max_comb = new_max_comb
                change_max_comb = True
            self.max_comb = self.comb.max()
        self.update_data(reducible_pairs, new_cherries, change_height_dist, change_height_comb, change_max_dist,
                         change_max_comb, change_cherry_in_forest, tree_set)
        return change_height_dist, change_height_comb

    def update_data(self, reducible_pairs, new_cherries, change_height_dist, change_height_comb, change_max_dist,
                    change_max_comb, change_cherry_in_forest, tree_set):
        # UPDATE DATA
        change_dist = pd.Series(0, index=reducible_pairs, dtype=float)
        change_comb = pd.Series(0, index=reducible_pairs, dtype=float)
        for c, trees in reducible_pairs.items():
            if c in new_cherries:
                continue
            for t in trees:
                if t in change_cherry_in_forest["trees_in"][c]:
                    # self.d[c] += 1
                    self.dist_n[c] += self.dist[t]
                    self.comb_n[c] += self.comb[t]
                    continue
                if change_height_dist[t]:
                    self.dist_n[c] -= self.prev_dist[t]
                    self.dist_n[c] += self.dist[t]
                    change_dist[c] = 1
                if change_height_comb[t]:
                    self.comb_n[c] -= self.prev_comb[t]
                    self.comb_n[c] += self.comb[t]
                    change_comb[c] = 1

            if change_max_dist and change_dist[c]:
                self.data_column.loc[c, "tree_height"] = self.dist_n[c] / self.max_dist / self.d[c]
            if change_max_comb and change_comb[c]:
                self.data_column.loc[c, "tree_height_comb"] = self.comb_n[c] / self.max_comb / self.d[c]

        for t in tree_set:
            if change_height_dist[t]:
                self.prev_dist[t] = copy.copy(self.dist[t])
            if change_height_comb[t]:
                self.prev_comb[t] = copy.copy(self.comb[t])

    def chosen_cherry_cleaning(self, chosen_cherry):
        self.data_column.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.dist_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.comb_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

    # RELABEL
    def relabel(self, x, y, reducible_pairs, merged_cherries):
        relabelled = pd.Series(0, index=reducible_pairs, dtype=float)
        for c_x, c_y in self.data_column.index:
            if (c_x, c_y) in [(x, y), (y, x)]:
                continue
            if x != c_x:
                continue
            relabelled[(y, c_y)] = True
            relabelled[(c_y, y)] = True
            if (c_x, c_y) in merged_cherries:
                # change dist n
                self.dist_n[(y, c_y)] += self.dist_n[(x, c_y)]
                self.dist_n[(c_y, y)] += self.dist_n[(c_y, x)]
                self.dist_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change comb n
                self.comb_n[(y, c_y)] += self.comb_n[(x, c_y)]
                self.comb_n[(c_y, y)] += self.comb_n[(c_y, x)]
                self.comb_n.drop([(x, c_y), (c_y, x)], inplace=True)
            else:
                # rename dist n
                self.dist_n.index = list(self.dist_n.index)
                self.dist_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.dist_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.dist_n.index = pd.MultiIndex.from_tuples(self.dist_n.index)
                # rename comb n
                self.comb_n.index = list(self.comb_n.index)
                self.comb_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.comb_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.comb_n.index = pd.MultiIndex.from_tuples(self.comb_n.index)
            self.data_column.drop([(x, c_y), (c_y, x)], inplace=True)

        for c in reducible_pairs:
            if relabelled[c]:
                # update data
                self.data_column.loc[c, "tree_height"] = self.dist_n[c] / self.d[c]
                self.data_column.loc[c, "tree_height_comb"] = self.comb_n[c] / self.d[c]


# LEAF PAIR
class LeafPair:
    def __init__(self, num_trees):
        self.name = ["leaf_pair_in_tree"]
        self.data_column = pd.DataFrame(columns=self.name)
        self.n = pd.Series(dtype=float)
        self.d = num_trees
        self.leaf_pair_cont_tree = pd.DataFrame(dtype=float)

    def init_fun(self, reducible_pairs, tree_set):
        self.leaf_pair_cont_tree = pd.DataFrame(True, index=reducible_pairs, columns=tree_set)
        self.n = pd.Series(0, index=reducible_pairs, dtype=float)
        for t, tree in tree_set.items():
            for c in reducible_pairs:
                if set(c).issubset(tree.leaves):
                    self.leaf_pair_cont_tree.loc[c, t] = True
                    self.n[c] += 1
                else:
                    self.leaf_pair_cont_tree.loc[c, t] = False
        self.data_column["leaf_pair_in_tree"] = self.n / self.d
        return self.data_column

    def new_cherries(self, reducible_pairs, tree_set):
        for c in reducible_pairs:
            for t, tree in tree_set.items():
                self.leaf_pair_cont_tree.loc[c, t] = set(c).issubset(tree.leaves)
            self.n[c] = self.leaf_pair_cont_tree.loc[c].sum()
            self.data_column.loc[c, "leaf_pair_in_tree"] = self.n[c] / self.d

    def update(self, chosen_cherry, new_cherries, new_reduced, reducible_pairs, num_trees, num_trees_changed):
        # delete, only change! The x in the cherry only disappears in the reduced
        change_leaf_pair = {"any_diff": pd.Series(0, index=reducible_pairs, dtype=float),
                            "trees_out": {c: set() for c in reducible_pairs}}
        change_leaf_pair_bool = False
        for c in reducible_pairs:
            num_new_reduced = 0
            if c in [chosen_cherry, chosen_cherry[::-1]]:
                continue
            if chosen_cherry[0] not in c:
                continue
            for t in new_reduced:
                if self.leaf_pair_cont_tree.loc[c, t]:
                    num_new_reduced += 1
                    self.leaf_pair_cont_tree.loc[c, t] = False
                    change_leaf_pair["trees_out"][c].add(t)
            self.n[c] -= num_new_reduced
            change_leaf_pair["any_diff"][c] = 1
            change_leaf_pair_bool = True

        # UPDATE DATA
        if num_trees_changed:
            self.d = num_trees
        if not num_trees_changed and not change_leaf_pair_bool:
            return change_leaf_pair

        for c in reducible_pairs:
            if c in new_cherries:
                continue
            if num_trees_changed or change_leaf_pair["any_diff"][c]:
                self.data_column.loc[c, "leaf_pair_in_tree"] = self.n[c] / self.d
        return change_leaf_pair

    def chosen_cherry_cleaning(self, chosen_cherry):
        self.data_column.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.leaf_pair_cont_tree.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

    def relabel(self, x, y, reducible_pairs, merged_cherries, tree_set):
        anything_relabelled = False
        relabelled = pd.Series(0, index=reducible_pairs, dtype=float)
        leaf_pair_in_new_trees = {c: set() for c in reducible_pairs}
        for c_x, c_y in self.data_column.index:
            if (c_x, c_y) in [(x, y), (y, x)]:
                continue
            if x == c_x:
                anything_relabelled = True
                relabelled[(y, c_y)] = True
                relabelled[(c_y, y)] = True
                if (c_x, c_y) in merged_cherries:
                    self.leaf_pair_cont_tree.loc[(y, c_y)] = self.leaf_pair_cont_tree.loc[[(x, c_y), (y, c_y)]].max()
                    self.leaf_pair_cont_tree.loc[(c_y, y)] = self.leaf_pair_cont_tree.loc[[(c_y, x), (c_y, y)]].max()
                    self.leaf_pair_cont_tree.drop([(x, c_y), (c_y, x)], inplace=True)
                    # change n
                    self.n[(y, c_y)] += self.n[(x, c_y)]
                    self.n[(c_y, y)] += self.n[(c_y, x)]
                    self.n.drop([(x, c_y), (c_y, x)], inplace=True)
                else:
                    self.leaf_pair_cont_tree.index = list(self.leaf_pair_cont_tree.index)
                    self.leaf_pair_cont_tree.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                    self.leaf_pair_cont_tree.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                    self.leaf_pair_cont_tree.index = pd.MultiIndex.from_tuples(self.leaf_pair_cont_tree.index)
                    # rename n
                    self.n.index = list(self.n.index)
                    self.n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                    self.n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                    self.n.index = pd.MultiIndex.from_tuples(self.n.index)
                self.data_column.drop([(x, c_y), (c_y, x)], inplace=True)

        for c in reducible_pairs:
            c_x, c_y = c
            if c in [(x, y), (y, x)]:
                continue
            if y == c_x:
                for t in tree_set:
                    if not self.leaf_pair_cont_tree.loc[c][t] and {c_x, c_y}.issubset(tree_set[t].leaves):
                        anything_relabelled = True
                        relabelled[c] = True
                        relabelled[c[::-1]] = True
                        leaf_pair_in_new_trees[c].add(t)
                        leaf_pair_in_new_trees[c[::-1]].add(t)
                        self.leaf_pair_cont_tree.loc[c, t] = True
                        self.leaf_pair_cont_tree.loc[c[::-1], t] = True
                        self.n[c] += 1
                        self.n[c[::-1]] += 1

            if relabelled[c]:
                # update data
                self.data_column.loc[c, "leaf_pair_in_tree"] = self.n[c] / self.d

        return anything_relabelled, leaf_pair_in_new_trees


# TRIVIAL
class Trivial:
    def __init__(self):
        # PARAMETERS
        self.name = ["trivial", "cherry_in_tree"]
        self.data_column = pd.DataFrame(columns=self.name)

        self.trivial = pd.DataFrame()
        self.n = pd.Series(dtype=float)

        self.triv_d = pd.Series(dtype=float)
        self.cher_d = 1

    def init_fun(self, reducible_pairs, tree_set, unique_cherries, num_trees, leaf_pair_forest):
        # ONLY WORKS FOR FULL LEAF SET
        self.trivial = pd.DataFrame(index=reducible_pairs, columns=tree_set)
        for c in unique_cherries:
            trees = list(reducible_pairs[c])
            self.trivial.loc[c, trees] = True
            self.trivial.loc[c[::-1], trees] = True
        self.trivial.fillna(False, inplace=True)

        # trivial data
        self.n = self.trivial.sum(axis=1)
        self.triv_d = leaf_pair_forest
        self.data_column["trivial"] = self.n / self.triv_d

        # cherry in tree data
        self.cher_d = num_trees
        self.data_column["cherry_in_tree"] = self.n / self.cher_d

        return self.data_column

    def new_cherries(self, reducible_pairs, unique_cherries, leaf_pair_in_tree, leaf_pair_forest):
        for c in unique_cherries:
            trees = reducible_pairs[c]
            # first fill with false where this leaf pair exist, then fill with true?
            for t in leaf_pair_in_tree.columns:
                if leaf_pair_in_tree.loc[c, t]:
                    self.trivial.loc[c, t] = False
                    self.trivial.loc[c[::-1], t] = False
            # now check where it should be
            for t in trees:
                self.trivial.loc[c, t] = True
                self.trivial.loc[c[::-1], t] = True

        # trivial data
        for c in unique_cherries:
            nom = self.trivial.loc[c].sum()
            self.n.loc[c] = nom
            self.n.loc[c[::-1]] = nom

            triv_val = self.n[c] / self.triv_d[c]
            self.data_column.loc[c, "trivial"] = triv_val
            self.data_column.loc[c[::-1], "trivial"] = triv_val

            # cherry in tree data
            cher_val = self.n[c] / self.cher_d
            self.data_column.loc[c, "cherry_in_tree"] = cher_val
            self.data_column.loc[c[::-1], "cherry_in_tree"] = cher_val

    def update(self, new_reduced, reducible_pairs, tree_set, new_cherries, leaf_pair_cont_tree,
               change_leaf_pair):
        change_cher_denom = 0
        change_cherry_in = {"any_diff": pd.Series(0, index=reducible_pairs, dtype=int),
                            "trees_in": {c: set() for c in reducible_pairs}}
        for t in new_reduced:
            if t not in tree_set:
                # cherry denom is only changed if a tree is fully reduced
                change_cher_denom = 1
                self.cher_d -= 1
                continue
            for c, trees in reducible_pairs.items():
                if c in new_cherries:
                    continue
                # if triv is False, could become nan or True
                # if triv is True, nothing changes
                # if triv is nan, nothing changes
                if not self.trivial.loc[c, t]:
                    if not leaf_pair_cont_tree.loc[c, t]:
                        self.trivial.loc[c, t] = np.nan
                        # self.triv_d[c] -= 1
                    elif t in trees:
                        self.trivial.loc[c, t] = True
                        self.n[c] += 1
                        change_cherry_in["any_diff"][c] = 1
                        change_cherry_in["trees_in"][c].add(t)

        for c in reducible_pairs:
            if c in new_cherries:
                continue
            if change_cherry_in["any_diff"][c]:
                self.data_column.loc[c, "trivial"] = self.n[c] / self.triv_d[c]
                self.data_column.loc[c, "cherry_in_tree"] = self.n[c] / self.cher_d
                continue

            if change_leaf_pair["any_diff"][c]:
                self.data_column.loc[c, "trivial"] = self.n[c] / self.triv_d[c]

            if change_cher_denom:
                self.data_column.loc[c, "cherry_in_tree"] = self.n[c] / self.cher_d
        return change_cherry_in

    def chosen_cherry_cleaning(self, chosen_cherry):
        self.data_column.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.trivial.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        # self.triv_d.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

    # RELABEL
    def relabel(self, x, y, reducible_pairs, merged_cherries, leaf_pair_in_new_trees):
        relabelled = pd.Series(0, index=reducible_pairs, dtype=int)
        for c_x, c_y in self.data_column.index:
            if (c_x, c_y) in [(x, y), (y, x)]:
                continue
            if x != c_x:
                continue
            relabelled[(y, c_y)] = True
            relabelled[(c_y, y)] = True
            if (c_x, c_y) in merged_cherries:
                self.trivial.loc[(y, c_y)] = self.trivial.loc[[(x, c_y), (y, c_y)]].max().fillna(value=np.nan)
                self.trivial.loc[(c_y, y)] = self.trivial.loc[[(c_y, x), (c_y, y)]].max().fillna(value=np.nan)
                self.trivial = self.trivial.drop([(x, c_y), (c_y, x)])
                # change n
                self.n[(y, c_y)] += self.n[(x, c_y)]
                self.n[(c_y, y)] += self.n[(c_y, x)]
                self.n.drop([(x, c_y), (c_y, x)], inplace=True)
            else:
                self.trivial.index = list(self.trivial.index)
                self.trivial.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.trivial.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.trivial.index = pd.MultiIndex.from_tuples(self.trivial.index)
                # rename n
                self.n.index = list(self.n.index)
                self.n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.n.index = pd.MultiIndex.from_tuples(self.n.index)
            self.data_column.drop([(x, c_y), (c_y, x)], inplace=True)

        for c in reducible_pairs:
            for t in leaf_pair_in_new_trees[c]:
                relabelled[c] = True
                self.triv_d[c] += 1
                self.trivial.loc[c, t] = False

            if relabelled[c]:
                # update data
                self.data_column.loc[c, "trivial"] = self.n[c] / self.triv_d[c]
                self.data_column.loc[c, "cherry_in_tree"] = self.n[c] / self.cher_d


# CHERRY HEIGHT
class CherryHeight:
    def __init__(self):
        # PARAMETERS
        self.name = ["cherry_height", "cherry_height_comb"]
        self.data_column = pd.DataFrame(columns=self.name)

        # distances
        self.dist_n = pd.Series(dtype=float)
        self.dist = pd.DataFrame(dtype=float)

        # combinatorial
        self.comb_n = pd.Series(dtype=float)
        self.comb = pd.DataFrame(dtype=float)

        self.d = pd.Series(dtype=float)
        # tree height parameters
        self.tree_dist_prev = pd.Series(dtype=float)
        self.tree_comb_prev = pd.Series(dtype=float)

    def init_fun(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb, cherry_in_forest,
                 new_cherries=False):
        if not new_cherries:
            self.tree_dist_prev = copy.deepcopy(tree_height_dist)
            self.tree_comb_prev = copy.deepcopy(tree_height_comb)

            self.dist = pd.DataFrame(index=reducible_pairs, columns=tree_set)
            self.comb = pd.DataFrame(index=reducible_pairs, columns=tree_set)
            self.d = cherry_in_forest

        for c in unique_cherries:
            for t in reducible_pairs[c]:
                tree = tree_set[t]
                dist, comb = self.get_cherry_height(tree, *c)
                self.dist.loc[c, t] = dist
                self.dist.loc[c[::-1], t] = dist
                self.comb.loc[c, t] = comb
                self.comb.loc[c[::-1], t] = comb

        # DATA
        if not new_cherries:
            self.dist_n = (self.dist / self.tree_dist_prev).sum(axis=1)
            self.comb_n = (self.comb / self.tree_comb_prev).sum(axis=1)

            self.data_column["cherry_height"] = self.dist_n / self.d
            self.data_column["cherry_height_comb"] = self.comb_n / self.d
            return self.data_column

        else:
            for c in unique_cherries:
                self.dist_n[c] = 0
                self.dist_n[c[::-1]] = 0
                self.comb_n[c] = 0
                self.comb_n[c[::-1]] = 0
                for t in reducible_pairs[c]:
                    if not tree_height_dist[t]:
                        dist_n_val = 0
                    else:
                        dist_n_val = self.dist.loc[c, t] / tree_height_dist[t]
                    self.dist_n[c] += dist_n_val
                    self.dist_n[c[::-1]] += dist_n_val

                    if not tree_height_comb[t]:
                        comb_n_val = 0
                    else:
                        comb_n_val = self.comb.loc[c, t] / tree_height_comb[t]
                    self.comb_n[c] += comb_n_val
                    self.comb_n[c[::-1]] += comb_n_val

                dist_val = self.dist_n[c] / self.d[c]
                self.data_column.loc[c, "cherry_height"] = dist_val
                self.data_column.loc[c[::-1], "cherry_height"] = dist_val

                comb_val = self.comb_n[c] / self.d[c]
                self.data_column.loc[c, "cherry_height_comb"] = comb_val
                self.data_column.loc[c[::-1], "cherry_height_comb"] = comb_val

    def get_cherry_height(self, tree, x, y):
        for p in tree.nw.predecessors(x):
            p_cherry = p
        height = tree.nw.nodes[p_cherry]["node_length"]
        height_comb = tree.nw.nodes[p_cherry]["node_comb"]

        return height, height_comb

    def new_cherries(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                     cherry_in_forest):
        self.init_fun(reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb, cherry_in_forest,
                      new_cherries=True)

    def update(self, reducible_pairs, tree_set, new_cherries,
               new_height_dist, new_height_comb,
               change_cherry_in_forest, change_height_dist, change_height_comb):
        changed_dist = pd.Series(0, index=reducible_pairs, dtype=int)
        changed_comb = pd.Series(0, index=reducible_pairs, dtype=int)
        for c in reducible_pairs:
            if c in new_cherries:
                continue
            for t in change_cherry_in_forest["trees_in"][c]:
                self.dist.loc[c][t], self.comb.loc[c][t] = self.get_cherry_height(tree_set[t], *c)
                if self.tree_comb_prev[t]:
                    self.comb_n[c] += self.comb.loc[c][t] / self.tree_comb_prev[t]
                if self.tree_dist_prev[t]:
                    self.dist_n[c] += self.dist.loc[c][t] / self.tree_dist_prev[t]
                changed_dist[c] = 1
                changed_comb[c] = 1

        # CHANGE tree height and denominator at once
        for c, trees in reducible_pairs.items():
            if c in new_cherries:
                continue
            # numerator
            for t in trees:
                if change_height_dist[t]:
                    changed_dist[c] = 1
                    if self.tree_dist_prev[t]:
                        self.dist_n[c] -= self.dist.loc[c][t] / self.tree_dist_prev[t]
                    if new_height_dist[t]:
                        self.dist_n[c] += self.dist.loc[c][t] / new_height_dist[t]
                if change_height_comb[t]:
                    changed_comb[c] = 1
                    if self.tree_comb_prev[t]:
                        self.comb_n[c] -= self.comb.loc[c][t] / self.tree_comb_prev[t]
                    if new_height_comb[t]:
                        self.comb_n[c] += self.comb.loc[c][t] / new_height_comb[t]
            # denominator
            if change_cherry_in_forest["any_diff"][c]:
                self.data_column.loc[c, "cherry_height"] = self.dist_n[c] / self.d[c]
                self.data_column.loc[c, "cherry_height_comb"] = self.comb_n[c] / self.d[c]
                continue
            if changed_dist[c]:
                self.data_column.loc[c, "cherry_height"] = self.dist_n[c] / self.d[c]
            if changed_comb[c]:
                self.data_column.loc[c, "cherry_height_comb"] = self.comb_n[c] / self.d[c]

        # and finally, update previous tree heights
        self.tree_dist_prev = copy.deepcopy(new_height_dist)
        self.tree_comb_prev = copy.deepcopy(new_height_comb)

    def chosen_cherry_cleaning(self, chosen_cherry):
        self.data_column.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.dist_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.dist.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.comb_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.comb.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        # self.d.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

    # RELABEL
    def relabel(self, x, y, reducible_pairs, merged_cherries):
        relabelled = pd.Series(0, index=reducible_pairs, dtype=int)
        for c_x, c_y in self.data_column.index:
            if (c_x, c_y) in [(x, y), (y, x)]:
                continue
            if x != c_x:
                continue
            relabelled[(y, c_y)] = True
            relabelled[(c_y, y)] = True
            if (c_x, c_y) in merged_cherries:
                # change dist n
                self.dist_n[(y, c_y)] += self.dist_n[(x, c_y)]
                self.dist_n[(c_y, y)] += self.dist_n[(c_y, x)]
                self.dist_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change comb n
                self.comb_n[(y, c_y)] += self.comb_n[(x, c_y)]
                self.comb_n[(c_y, y)] += self.comb_n[(c_y, x)]
                self.comb_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change comb
                self.comb.loc[(y, c_y)] = self.comb.loc[[(x, c_y), (y, c_y)]].max()
                self.comb.loc[(c_y, y)] = self.comb.loc[[(c_y, x), (c_y, y)]].max()
                self.comb.drop([(x, c_y), (c_y, x)], inplace=True)
                # change dist
                self.dist.loc[(y, c_y)] = self.dist.loc[[(x, c_y), (y, c_y)]].max()
                self.dist.loc[(c_y, y)] = self.dist.loc[[(c_y, x), (c_y, y)]].max()
                self.dist.drop([(x, c_y), (c_y, x)], inplace=True)
            else:
                # rename dist n
                self.dist_n.index = list(self.dist_n.index)
                self.dist_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.dist_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.dist_n.index = pd.MultiIndex.from_tuples(self.dist_n.index)
                # rename comb n
                self.comb_n.index = list(self.comb_n.index)
                self.comb_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.comb_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.comb_n.index = pd.MultiIndex.from_tuples(self.comb_n.index)
                # rename comb
                self.comb.index = list(self.comb.index)
                self.comb.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.comb.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.comb.index = pd.MultiIndex.from_tuples(self.comb.index)
                # rename dist
                self.dist.index = list(self.dist.index)
                self.dist.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.dist.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.dist.index = pd.MultiIndex.from_tuples(self.dist.index)
            # DATA
            self.data_column.drop([(x, c_y), (c_y, x)], inplace=True)

        for c in reducible_pairs:
            if relabelled[c]:
                # update data
                self.data_column.loc[c, "cherry_height"] = self.dist_n[c] / self.d[c]
                self.data_column.loc[c, "cherry_height_comb"] = self.comb_n[c] / self.d[c]


# REDUCTION AFTER PICKING A CHERRY
class RedAfterPick:
    def __init__(self, root=2):
        # PARAMETERS
        self.name = ["red_after_pick", "new_diff_cherries"]
        self.root = root

        self.data_column = pd.DataFrame(columns=self.name)

        self.red_n = pd.Series(dtype=float)
        self.red_d = pd.Series(dtype=float)
        self.red_after_pick = pd.DataFrame()

        self.cher_red_n = pd.Series(dtype=float)
        self.cher_red_d = 1
        self.prev_cher_red_d = 1
        self.cherry_red_set = dict()

    def init_fun(self, reducible_pairs, tree_set, unique_cherries, cherry_in_forest, new_cherries=False):
        if not new_cherries:
            self.red_after_pick = pd.DataFrame(index=reducible_pairs, columns=tree_set)
            # new cherries/old cherries, so doesn't have to be scaled anymore
            # network cherries works with just one tree, tree cherries with a set
            num_unique_cherries = len(unique_cherries)
            self.cher_red_n = pd.Series(num_unique_cherries - 1, index=reducible_pairs)
            self.cher_red_d = num_unique_cherries
            self.prev_cher_red_d = num_unique_cherries
        if new_cherries:
            for c in reducible_pairs:
                self.cher_red_n[c] = self.cher_red_d - 1

        for c in unique_cherries:
            self.cherry_red_set[c] = set()
            self.cherry_red_set[c[::-1]] = set()
            for t in reducible_pairs[c]:
                num_after_pick, new_cat_cherry_a, new_cat_cherry_b = self.get_num_red_after_pick(tree_set[t], *c)
                self.red_after_pick.loc[c, t] = num_after_pick
                self.red_after_pick.loc[c[::-1], t] = num_after_pick
                if new_cat_cherry_a is not None:
                    self.cherry_red_set[c].add(new_cat_cherry_a)
                    self.cherry_red_set[c[::-1]].add(new_cat_cherry_b)

            for red_cher in self.cherry_red_set[c]:
                if red_cher not in unique_cherries:
                    self.cher_red_n[c] += 1
            for red_cher in self.cherry_red_set[c[::-1]]:
                if red_cher not in unique_cherries:
                    self.cher_red_n[c[::-1]] += 1

        # DATA
        if not new_cherries:
            self.data_column["new_diff_cherries"] = self.cher_red_n/self.cher_red_d
            self.red_n = self.red_after_pick.sum(axis=1)
            self.red_d = copy.deepcopy(cherry_in_forest)
            self.data_column["red_after_pick"] = self.red_n / self.red_d
            return self.data_column
        else:
            for c in unique_cherries:
                cher_val = self.cher_red_n[c]/self.cher_red_d
                self.data_column.loc[c, "new_diff_cherries"] = cher_val
                self.data_column.loc[c[::-1], "new_diff_cherries"] = cher_val

                red_n_val = self.red_after_pick.loc[c].sum()
                self.red_n[c] = red_n_val
                self.red_n[c[::-1]] = red_n_val
                self.red_d[c] = cherry_in_forest[c]
                self.red_d[c[::-1]] = cherry_in_forest[c[::-1]]

                red_val = self.red_n[c] / self.red_d[c]
                self.data_column.loc[c, "red_after_pick"] = red_val
                self.data_column.loc[c[::-1], "red_after_pick"] = red_val

    def new_cherries(self, reducible_pairs, tree_set, unique_cherries, denom):
        self.init_fun(reducible_pairs, tree_set, unique_cherries, denom, new_cherries=True)

    def get_num_red_after_pick(self, tree, x, y):
        # new cherries/old cherries, so doesn't have to be scaled anymore
        # network cherries works with just one tree, tree cherries with a set
        # ALL ABOUT CATERPILLARS
        for p in tree.nw.predecessors(x):
            p_cherry = p
        if p_cherry == self.root:
            return 0, None, None
        for p in tree.nw.predecessors(p_cherry):
            if tree.nw.out_degree(p) != 2:
                if self.root == 2 and p in [1, 2]:
                    return 0, None, None
                print(f"Something went wrong")
            for ch in tree.nw.successors(p):
                if ch == p_cherry:
                    continue
                if tree.nw.out_degree(ch) == 0:
                    return 1, tuple(sorted([ch, y])), tuple(sorted([ch, x]))
                else:
                    return 0, None, None

    def update(self, chosen_cherry, new_reduced, unique_reducible_pairs, tree_set, new_cherries,
               change_cherry_in_forest):

        x, y = chosen_cherry
        unique_cherries = set(unique_reducible_pairs)

        # UPDATE self.cher_red_d FOR ALL CHERRIES!
        num_new_cherries = len(new_cherries) / 2
        if num_new_cherries == 0 or num_new_cherries > 1:
            update_cher_num = True
            self.cher_red_d += num_new_cherries - 1
            self.cher_red_n += num_new_cherries - 1
        else:
            update_cher_num = False

        for c, trees in unique_reducible_pairs.items():
            cher_updated = False
            if c in new_cherries:
                continue
            for t in trees:
                if t not in new_reduced:
                    continue
                # EITHER, NEW CHERRY IN TREE
                if t in change_cherry_in_forest["trees_in"][c]:
                    num_after_pick, new_cat_cherry_a, new_cat_cherry_b = self.get_num_red_after_pick(tree_set[t], *c)
                    self.red_after_pick.loc[c, t] = num_after_pick
                    self.red_after_pick.loc[c[::-1], t] = num_after_pick
                    self.red_d[c] += 1
                    self.red_d[c[::-1]] += 1
                    if num_after_pick:
                        self.red_n[c] += 1
                        self.red_n[c[::-1]] += 1
                    if new_cat_cherry_a is not None:
                        if new_cat_cherry_a:
                            if new_cat_cherry_a not in self.cherry_red_set[c]:
                                self.cherry_red_set[c].add(new_cat_cherry_a)
                                if new_cat_cherry_a not in unique_cherries:
                                    self.cher_red_n[c] += 1
                            if new_cat_cherry_b not in self.cherry_red_set[c[::-1]]:
                                self.cherry_red_set[c[::-1]].add(new_cat_cherry_b)
                                if new_cat_cherry_b not in unique_cherries:
                                    self.cher_red_n[c[::-1]] += 1
                    cher_updated = True
                    continue
                if self.red_after_pick.loc[c, t] == 1:
                    continue
                # OR, ALREADY IN TREE, BUT BECAUSE OF CHERRY PICKED, OTHER SCENARIO NOW
                tree = tree_set[t]
                # test if child of grand parent is y
                # first: find grand parent of c
                for p in tree.nw.predecessors(c[0]):
                    p_c = p
                for p in tree.nw.predecessors(p_c):
                    gp_c = p
                # second: find child of grand parent
                for ch in tree.nw.successors(gp_c):
                    if ch == y:
                        cher_updated = True
                        # red_after_pick becomes 1
                        self.red_after_pick.loc[c, t] = 1
                        self.red_after_pick.loc[c[::-1], t] = 1
                        self.red_n[c] += 1
                        self.red_n[c[::-1]] += 1

                        # add (y, c[0]) and (y, c[1]) to cherry_red_set
                        new_cat_cherry_a = tuple(sorted([c[0], y]))
                        new_cat_cherry_b = tuple(sorted([c[1], y]))
                        if new_cat_cherry_a not in self.cherry_red_set[c]:
                            self.cherry_red_set[c].add(new_cat_cherry_a)
                            if new_cat_cherry_a not in unique_cherries:
                                self.cher_red_n[c] += 1
                        if new_cat_cherry_b not in self.cherry_red_set[c[::-1]]:
                            self.cherry_red_set[c[::-1]].add(new_cat_cherry_b)
                            if new_cat_cherry_b not in unique_cherries:
                                self.cher_red_n[c[::-1]] += 1
            # UPDATE DATA
            if cher_updated:
                self.data_column.loc[c, "new_diff_cherries"] = self.cher_red_n[c] / self.cher_red_d
                self.data_column.loc[c[::-1], "new_diff_cherries"] = self.cher_red_n[c[::-1]] / self.cher_red_d

                red_val = self.red_n[c] / self.red_d[c]
                self.data_column.loc[c, "red_after_pick"] = red_val
                self.data_column.loc[c[::-1], "red_after_pick"] = red_val
            elif update_cher_num:
                self.data_column.loc[c, "new_diff_cherries"] = self.cher_red_n[c] / self.cher_red_d
                self.data_column.loc[c[::-1], "new_diff_cherries"] = self.cher_red_n[c[::-1]] / self.cher_red_d

    def chosen_cherry_cleaning(self, chosen_cherry):
        self.data_column.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.red_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.red_d.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.red_after_pick.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.cher_red_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        del self.cherry_red_set[chosen_cherry], self.cherry_red_set[chosen_cherry[::-1]]

    # RELABEL
    def relabel(self, x, y, reducible_pairs, merged_cherries):
        relabelled = pd.Series(0, index=reducible_pairs, dtype=int)
        for c_x, c_y in self.data_column.index:
            if (c_x, c_y) in [(x, y), (y, x)]:
                continue
            if x != c_x:
                continue
            relabelled[(y, c_y)] = True
            relabelled[(c_y, y)] = True
            if (c_x, c_y) in merged_cherries:
                # change red n
                self.red_n[(y, c_y)] += self.red_n[(x, c_y)]
                self.red_n[(c_y, y)] += self.red_n[(c_y, x)]
                self.red_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change red d
                self.red_d[(y, c_y)] += self.red_d[(x, c_y)]
                self.red_d[(c_y, y)] += self.red_d[(c_y, x)]
                self.red_d.drop([(x, c_y), (c_y, x)], inplace=True)
                # change cher red n
                self.cher_red_n[(y, c_y)] += self.cher_red_n[(x, c_y)]
                self.cher_red_n[(c_y, y)] += self.cher_red_n[(c_y, x)]
                self.cher_red_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change red after pick
                self.red_after_pick.loc[(y, c_y)] = self.red_after_pick.loc[[(x, c_y), (y, c_y)]].max()
                self.red_after_pick.loc[(c_y, y)] = self.red_after_pick.loc[[(c_y, x), (c_y, y)]].max()
                self.red_after_pick.drop([(x, c_y), (c_y, x)], inplace=True)
                # cher red set
                for cher_red in self.cherry_red_set[(x, c_y)]:
                    self.cherry_red_set[(y, c_y)].add(cher_red)
                for cher_red in self.cherry_red_set[(c_y, x)]:
                    self.cherry_red_set[(c_y, y)].add(cher_red)
                del self.cherry_red_set[(x, c_y)], self.cherry_red_set[(c_y, x)]
            else:
                # rename red n
                self.red_n.index = list(self.red_n.index)
                self.red_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.red_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.red_n.index = pd.MultiIndex.from_tuples(self.red_n.index)
                # rename red d
                self.red_d.index = list(self.red_d.index)
                self.red_d.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.red_d.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.red_d.index = pd.MultiIndex.from_tuples(self.red_d.index)
                # rename cher red n
                self.cher_red_n.index = list(self.cher_red_n.index)
                self.cher_red_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.cher_red_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.cher_red_n.index = pd.MultiIndex.from_tuples(self.cher_red_n.index)
                # rename red after pick
                self.red_after_pick.index = list(self.red_after_pick.index)
                self.red_after_pick = self.red_after_pick.rename(index={(x, c_y): (y, c_y)}, inplace=False)
                self.red_after_pick = self.red_after_pick.rename(index={(c_y, x): (c_y, y)}, inplace=False)
                self.red_after_pick.index = pd.MultiIndex.from_tuples(self.red_after_pick.index)
                # cher red set
                self.cherry_red_set[(y, c_y)] = self.cherry_red_set[(x, c_y)]
                self.cherry_red_set[(c_y, y)] = self.cherry_red_set[(c_y, x)]
                del self.cherry_red_set[(x, c_y)], self.cherry_red_set[(c_y, x)]
            # DATA
            self.data_column.drop([(x, c_y), (c_y, x)], inplace=True)

        for c in reducible_pairs:
            if relabelled[c]:
                # update data
                self.data_column.loc[c, "red_after_pick"] = self.red_n[c] / self.red_d[c]
                self.data_column.loc[c, "new_diff_cherries"] = self.cher_red_n[c] / self.cher_red_d


# FEATURES WITH UPDATE BEFORE
# LEAF DISTANCE
class LeafDist:
    def __init__(self):
        # PARAMETERS
        self.name = ["leaf_dist", "leaf_dist_comb", "leaf_dist_frac", "leaf_dist_comb_frac"]
        self.data_column = pd.DataFrame(columns=self.name, dtype=float)

        # distances
        self.dist_n = pd.Series(dtype=float)
        self.dist = pd.DataFrame(dtype=float)

        self.dist_frac_n = pd.Series(dtype=float)
        self.dist_frac = pd.DataFrame(dtype=float)

        # combinatorial
        self.comb_n = pd.Series(dtype=float)
        self.comb = pd.DataFrame(dtype=float)

        self.comb_frac_n = pd.Series(dtype=float)
        self.comb_frac = pd.DataFrame(dtype=float)

        self.d = pd.Series(dtype=float)
        # tree height parameters
        self.tree_dist_prev = pd.Series(dtype=float)
        self.tree_comb_prev = pd.Series(dtype=float)

        self.changed_before = pd.Series(dtype=int)

    def init_fun(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb, leaf_pair_in_forest,
                 leaf_pair_in, new_cherries=False):
        if not new_cherries:
            self.tree_dist_prev = copy.deepcopy(tree_height_dist)
            self.tree_comb_prev = copy.deepcopy(tree_height_comb)

            self.dist = pd.DataFrame(index=reducible_pairs, columns=tree_set)
            self.dist_frac = pd.DataFrame(index=reducible_pairs, columns=tree_set)
            self.comb = pd.DataFrame(index=reducible_pairs, columns=tree_set)
            self.comb_frac = pd.DataFrame(index=reducible_pairs, columns=tree_set)

        for c in unique_cherries:
            for t, tree in tree_set.items():
                if not leaf_pair_in_forest.loc[c, t]:
                    continue
                leaf_dist_comb, x_comb, leaf_dist, x_dist = self.get_leaf_dist(tree, *c)
                # data stuff
                self.comb.loc[c, t] = leaf_dist_comb
                self.comb.loc[c[::-1], t] = leaf_dist_comb
                self.comb_frac.loc[c, t] = x_comb
                self.comb_frac.loc[c[::-1], t] = 1 - x_comb

                self.dist.loc[c, t] = leaf_dist
                self.dist.loc[c[::-1], t] = leaf_dist
                self.dist_frac.loc[c, t] = x_dist
                self.dist_frac.loc[c[::-1], t] = 1 - x_dist

        # DATA
        if not new_cherries:
            self.d = leaf_pair_in

            # leaf distance
            self.dist_n = (self.dist / self.tree_dist_prev).sum(axis=1)
            self.comb_n = (self.comb / self.tree_comb_prev).sum(axis=1)

            self.data_column["leaf_dist_comb"] = self.comb_n / self.d
            self.data_column["leaf_dist"] = self.dist_n / self.d

            # leaf distance frac
            self.dist_frac_n = self.dist_frac.sum(axis=1)
            self.comb_frac_n = self.comb_frac.sum(axis=1)

            self.data_column["leaf_dist_comb_frac"] = self.comb_frac_n / self.d
            self.data_column["leaf_dist_frac"] = self.dist_frac_n / self.d
            return self.data_column
        else:
            for c in unique_cherries:
                self.dist_n[c] = 0
                self.dist_n[c[::-1]] = 0
                self.comb_n[c] = 0
                self.comb_n[c[::-1]] = 0
                for t in tree_set:
                    if np.isnan(self.dist.loc[c, t]):
                        continue
                    # leaf distance
                    if not tree_height_dist[t]:
                        dist_n_val = 0
                    else:
                        dist_n_val = self.dist.loc[c, t] / tree_height_dist[t]
                    self.dist_n[c] += dist_n_val
                    self.dist_n[c[::-1]] += dist_n_val

                    if not tree_height_comb[t]:
                        comb_n_val = 0
                    else:
                        comb_n_val = self.comb.loc[c, t] / tree_height_comb[t]
                    self.comb_n[c] += comb_n_val
                    self.comb_n[c[::-1]] += comb_n_val

                leaf_dist_val = self.dist_n[c] / self.d[c]
                self.data_column.loc[c, "leaf_dist"] = leaf_dist_val
                self.data_column.loc[c[::-1], "leaf_dist"] = leaf_dist_val

                leaf_comb_val = self.comb_n[c] / self.d[c]
                self.data_column.loc[c, "leaf_dist_comb"] = leaf_comb_val
                self.data_column.loc[c[::-1], "leaf_dist_comb"] = leaf_comb_val

                # leaf distance frac
                self.dist_frac_n[c] = self.dist_frac.loc[c].sum()
                self.dist_frac_n[c[::-1]] = self.dist_frac.loc[c[::-1]].sum()

                self.comb_frac_n[c] = self.comb_frac.loc[c].sum()
                self.comb_frac_n[c[::-1]] = self.comb_frac.loc[c[::-1]].sum()

                self.data_column.loc[c, "leaf_dist_frac"] = self.dist_frac_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "leaf_dist_frac"] = self.dist_frac_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "leaf_dist_comb_frac"] = self.comb_frac_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "leaf_dist_comb_frac"] = self.comb_frac_n[c[::-1]] / self.d[c[::-1]]
            return self.data_column.loc[list(reducible_pairs)]

    def get_leaf_dist(self, tree, x, y):
        # so-called up down distance. So find first common ancestor,
        # and then compute distance from this node to both leaves
        lca = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(tree.nw, x, y)

        # LENGTH TO X
        for p in tree.nw.predecessors(x):
            p_x = p
        # LENGTH TO Y
        for p in tree.nw.predecessors(y):
            p_y = p
        # ignore length of thingy itself
        dist_length = tree.nw.nodes[p_x]["node_length"] + tree.nw.nodes[p_y]["node_length"] - 2 * tree.nw.nodes[lca][
            "node_length"]
        x_dist = (tree.nw.nodes[x]["node_length"] - tree.nw.nodes[lca]["node_length"]) / (tree.nw.nodes[x]["node_length"]
                                                                                          + tree.nw.nodes[y]["node_length"]
                                                                                          - 2 * tree.nw.nodes[lca][
            "node_length"])
        comb_length = tree.nw.nodes[p_x]["node_comb"] + tree.nw.nodes[p_y]["node_comb"] - 2 * tree.nw.nodes[lca][
            "node_comb"]
        x_comb = (tree.nw.nodes[x]["node_comb"] - tree.nw.nodes[lca]["node_comb"]) / (tree.nw.nodes[x]["node_comb"] + \
                 tree.nw.nodes[y]["node_comb"] - 2 * tree.nw.nodes[lca]["node_comb"])

        return comb_length, x_comb, dist_length, x_dist

    def new_cherries(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                     leaf_pair_in_forest, leaf_pair_in):
        return self.init_fun(reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                             leaf_pair_in_forest, leaf_pair_in, new_cherries=True)

    def update_before(self, chosen_cherry, tree_set, reducible_pairs):
        self.changed_before = pd.Series(0, index=reducible_pairs, dtype=int)
        x, y = chosen_cherry
        for t in reducible_pairs[chosen_cherry]:
            tree = tree_set[t]
            for w in tree.leaves:
                if w == x:
                    continue
                # Don't delete following: UPDATE leaf distance for cherry (w, y)
                try:
                    check = reducible_pairs[(w, y)]
                except KeyError:
                    continue
                # if (w, y) in reducible_pairs:
                self.changed_before[(w, y)] = 1
                self.changed_before[(y, w)] = 1
                # COMBINATORIAL
                self.comb.loc[(w, y), t] -= 1
                self.comb_n[(w, y)] -= 1 / self.tree_comb_prev[t]
                self.comb.loc[(y, w), t] -= 1
                self.comb_n[(y, w)] -= 1 / self.tree_comb_prev[t]

                lca = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(tree.nw, x, w)
                comb_frac = (tree.nw.nodes[x]["node_comb"] - tree.nw.nodes[lca]["node_comb"] - 1) / (tree.nw.nodes[x]["node_comb"] + \
                                                                               tree.nw.nodes[w]["node_comb"] - 2 *
                                                                               tree.nw.nodes[lca]["node_comb"] - 1)
                prev_comb_frac_a = self.comb_frac.loc[(y, w), t]
                prev_comb_frac_b = self.comb_frac.loc[(w, y), t]
                if abs(prev_comb_frac_a - comb_frac) > 10e-3:
                    self.comb_frac.loc[(y, w), t] = comb_frac
                    self.comb_frac_n[(y, w)] -= prev_comb_frac_a
                    self.comb_frac_n[(y, w)] += comb_frac
                if abs(prev_comb_frac_b - (1 - comb_frac)) > 10e-3:
                    self.comb_frac.loc[(w, y), t] = 1 - comb_frac
                    self.comb_frac_n[(w, y)] -= prev_comb_frac_a
                    self.comb_frac_n[(w, y)] += 1 - comb_frac

    def update_data(self, new_cherries, unique_reducible_pairs, new_height, new_height_comb,
                    change_height_dist, change_height_comb, change_leaf_pair, tree_set):

        for c, trees in unique_reducible_pairs.items():
            change_dist = False
            change_comb = False
            if c in new_cherries:
                continue
            for t in trees:
                if t in change_leaf_pair["trees_out"]:
                    if self.tree_comb_prev[t]:
                        self.comb_n[c] -= self.comb.loc[c, t] / self.tree_comb_prev[t]
                        self.comb_n[c[::-1]] -= self.comb.loc[c[::-1], t] / self.tree_comb_prev[t]

                    self.comb.loc[c, t] = np.nan
                    self.comb.loc[c[::-1], t] = np.nan

                    self.comb_frac_n[c] -= self.comb_frac.loc[c, t]
                    self.comb_frac_n[c[::-1]] -= self.comb_frac.loc[c[::-1], t]

                    self.comb_frac.loc[c, t] = np.nan
                    self.comb_frac.loc[c[::-1], t] = np.nan

                    if self.tree_dist_prev[t]:
                        self.dist_n[c] -= self.dist.loc[c, t] / self.tree_dist_prev[t]
                        self.dist_n[c[::-1]] -= self.dist.loc[c[::-1], t] / self.tree_dist_prev[t]

                    self.dist.loc[c, t] = np.nan
                    self.dist.loc[c[::-1], t] = np.nan

                    self.dist_frac_n[c] -= self.dist_frac.loc[c, t]
                    self.dist_frac_n[c[::-1]] -= self.dist_frac.loc[c[::-1], t]

                    self.dist_frac.loc[c, t] = np.nan
                    self.dist_frac.loc[c[::-1], t] = np.nan
                    continue
                if change_height_dist[t]:
                    if self.tree_dist_prev[t]:
                        self.dist_n[c] -= self.dist.loc[c, t] / self.tree_dist_prev[t]
                        self.dist_n[c[::-1]] -= self.dist.loc[c[::-1], t] / self.tree_dist_prev[t]
                    if new_height[t]:
                        self.dist_n[c] += self.dist.loc[c, t] / new_height[t]
                        self.dist_n[c[::-1]] += self.dist.loc[c[::-1], t] / new_height[t]
                    change_dist = True

                if change_height_comb[t]:
                    if self.tree_comb_prev[t]:
                        self.comb_n[c] -= self.comb.loc[c, t] / self.tree_comb_prev[t]
                        self.comb_n[c[::-1]] -= self.comb.loc[c[::-1], t] / self.tree_comb_prev[t]
                    if new_height_comb[t]:
                        self.comb_n[c] += self.comb.loc[c, t] / new_height_comb[t]
                        self.comb_n[c[::-1]] += self.comb.loc[c[::-1], t] / new_height_comb[t]
                    change_comb = True

            if change_leaf_pair["any_diff"][c]:
                leaf_dist_val = self.dist_n[c] / self.d[c]
                self.data_column.loc[c, "leaf_dist"] = leaf_dist_val
                self.data_column.loc[c[::-1], "leaf_dist"] = leaf_dist_val

                leaf_comb_val = self.comb_n[c] / self.d[c]
                self.data_column.loc[c, "leaf_dist_comb"] = leaf_comb_val
                self.data_column.loc[c[::-1], "leaf_dist_comb"] = leaf_comb_val

                self.data_column.loc[c, "leaf_dist_frac"] = self.dist_frac_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "leaf_dist_frac"] = self.dist_frac_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "leaf_dist_comb_frac"] = self.comb_frac_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "leaf_dist_comb_frac"] = self.comb_frac_n[c[::-1]] / self.d[c[::-1]]
                continue
            if change_dist:
                leaf_dist_val = self.dist_n[c] / self.d[c]
                self.data_column.loc[c, "leaf_dist"] = leaf_dist_val
                self.data_column.loc[c[::-1], "leaf_dist"] = leaf_dist_val
            if self.changed_before[c]:
                self.data_column.loc[c, "leaf_dist_frac"] = self.dist_frac_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "leaf_dist_frac"] = self.dist_frac_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "leaf_dist_comb_frac"] = self.comb_frac_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "leaf_dist_comb_frac"] = self.comb_frac_n[c[::-1]] / self.d[c[::-1]]
                continue
            if change_comb:
                leaf_comb_val = self.comb_n[c] / self.d[c]
                self.data_column.loc[c, "leaf_dist_comb"] = leaf_comb_val
                self.data_column.loc[c[::-1], "leaf_dist_comb"] = leaf_comb_val

        for t in tree_set:
            if change_height_dist[t]:
                self.tree_dist_prev[t] = new_height[t]
            if change_height_comb[t]:
                self.tree_comb_prev[t] = new_height_comb[t]

        return None

    def chosen_cherry_cleaning(self, chosen_cherry):
        self.data_column.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        # distances
        self.dist_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.dist.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        self.dist_frac_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.dist_frac.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        # combinatorial
        self.comb_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.comb.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        self.comb_frac_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.comb_frac.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        self.changed_before = pd.Series(dtype=int)

    # RELABEL
    def relabel(self, x, y, reducible_pairs, merged_cherries, tree_set, leaf_pair_in_new_trees):
        relabelled = pd.Series(0, index=reducible_pairs, dtype=int)
        unique_reducible_pairs = {tuple(sorted(c)): trees for c, trees in reducible_pairs.items()}
        for c_x, c_y in self.data_column.index:
            if (c_x, c_y) in [(x, y), (y, x)]:
                continue
            if x != c_x:
                continue
            relabelled[(y, c_y)] = True
            relabelled[(c_y, y)] = True
            if (c_x, c_y) in merged_cherries:
                # change dist
                self.dist.loc[(y, c_y)] = self.dist.loc[[(x, c_y), (y, c_y)]].max()
                self.dist.loc[(c_y, y)] = self.dist.loc[[(c_y, x), (c_y, y)]].max()
                self.dist = self.dist.drop([(x, c_y), (c_y, x)])
                # n
                self.dist_n[(y, c_y)] += self.dist_n[(x, c_y)]
                self.dist_n[(c_y, y)] += self.dist_n[(c_y, x)]
                self.dist_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change dist frac
                self.dist_frac.loc[(y, c_y)] = self.dist_frac.loc[[(x, c_y), (y, c_y)]].max()
                self.dist_frac.loc[(c_y, y)] = self.dist_frac.loc[[(c_y, x), (c_y, y)]].max()
                self.dist_frac = self.dist_frac.drop([(x, c_y), (c_y, x)])
                # n
                self.dist_frac_n[(y, c_y)] += self.dist_frac_n[(x, c_y)]
                self.dist_frac_n[(c_y, y)] += self.dist_frac_n[(c_y, x)]
                self.dist_frac_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change comb
                self.comb.loc[(y, c_y)] = self.comb.loc[[(x, c_y), (y, c_y)]].max()
                self.comb.loc[(c_y, y)] = self.comb.loc[[(c_y, x), (c_y, y)]].max()
                self.comb = self.comb.drop([(x, c_y), (c_y, x)])
                # n
                self.comb_n[(y, c_y)] += self.comb_n[(x, c_y)]
                self.comb_n[(c_y, y)] += self.comb_n[(c_y, x)]
                self.comb_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change comb frac
                self.comb_frac.loc[(y, c_y)] = self.comb_frac.loc[[(x, c_y), (y, c_y)]].max()
                self.comb_frac.loc[(c_y, y)] = self.comb_frac.loc[[(c_y, x), (c_y, y)]].max()
                self.comb_frac = self.comb_frac.drop([(x, c_y), (c_y, x)])
                # n
                self.comb_frac_n[(y, c_y)] += self.comb_frac_n[(x, c_y)]
                self.comb_frac_n[(c_y, y)] += self.comb_frac_n[(c_y, x)]
                self.comb_frac_n.drop([(x, c_y), (c_y, x)], inplace=True)
            else:
                # rename dist
                self.dist.index = list(self.dist.index)
                self.dist.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.dist.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.dist.index = pd.MultiIndex.from_tuples(self.dist.index)
                # rename n
                self.dist_n.index = list(self.dist_n.index)
                self.dist_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.dist_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.dist_n.index = pd.MultiIndex.from_tuples(self.dist_n.index)
                # rename dist frac
                self.dist_frac.index = list(self.dist_frac.index)
                self.dist_frac.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.dist_frac.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.dist_frac.index = pd.MultiIndex.from_tuples(self.dist_frac.index)
                # rename n
                self.dist_frac_n.index = list(self.dist_frac_n.index)
                self.dist_frac_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.dist_frac_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.dist_frac_n.index = pd.MultiIndex.from_tuples(self.dist_frac_n.index)
                # rename comb
                self.comb.index = list(self.comb.index)
                self.comb.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.comb.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.comb.index = pd.MultiIndex.from_tuples(self.comb.index)
                # rename n
                self.comb_n.index = list(self.comb_n.index)
                self.comb_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.comb_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.comb_n.index = pd.MultiIndex.from_tuples(self.comb_n.index)
                # rename comb frac
                self.comb_frac.index = list(self.comb_frac.index)
                self.comb_frac.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.comb_frac.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.comb_frac.index = pd.MultiIndex.from_tuples(self.comb_frac.index)
                # rename n
                self.comb_frac_n.index = list(self.comb_frac_n.index)
                self.comb_frac_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.comb_frac_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.comb_frac_n.index = pd.MultiIndex.from_tuples(self.comb_frac_n.index)
            # DATA
            self.data_column.drop([(x, c_y), (c_y, x)], inplace=True)

        for c in unique_reducible_pairs:
            for t in leaf_pair_in_new_trees[c]:
                relabelled[c] = True
                relabelled[c[::-1]] = True
                # self.d[c] += 1
                # self.d[c[::-1]] += 1
                leaf_dist_comb, x_comb, leaf_dist, x_dist = self.get_leaf_dist(tree_set[t], *c)
                # data stuff
                self.comb.loc[c, t] = leaf_dist_comb
                self.comb.loc[c[::-1], t] = leaf_dist_comb
                if self.tree_comb_prev[t]:
                    self.comb_n[c] += leaf_dist_comb / self.tree_comb_prev[t]
                    self.comb_n[c[::-1]] += leaf_dist_comb / self.tree_comb_prev[t]
                self.comb_frac.loc[c, t] = x_comb
                self.comb_frac.loc[c[::-1], t] = 1 - x_comb
                self.comb_frac_n[c] += x_comb
                self.comb_frac_n[c[::-1]] += 1 - x_comb

                self.dist.loc[c, t] = leaf_dist
                self.dist.loc[c[::-1], t] = leaf_dist
                if self.tree_dist_prev[t]:
                    self.dist_n[c] += leaf_dist / self.tree_dist_prev[t]
                    self.dist_n[c[::-1]] += leaf_dist / self.tree_dist_prev[t]
                self.dist_frac.loc[c, t] = x_dist
                self.dist_frac.loc[c[::-1], t] = 1 - x_dist
                self.dist_frac_n[c] += x_dist
                self.dist_frac_n[c[::-1]] += 1 - x_dist

        for c in reducible_pairs:
            if relabelled[c]:
                # update data
                self.data_column.loc[c, "leaf_dist"] = self.dist_n[c] / self.d[c]

                self.data_column.loc[c, "leaf_dist_comb"] = self.comb_n[c] / self.d[c]

                self.data_column.loc[c, "leaf_dist_frac"] = self.dist_frac_n[c] / self.d[c]

                self.data_column.loc[c, "leaf_dist_comb_frac"] = self.comb_frac_n[c] / self.d[c]


# LEAF HEIGHT
class LeafHeight:
    def __init__(self):
        # PARAMETERS
        self.name = ["x_height", "y_height", "x_vs_y_height", "x_height_comb", "y_height_comb", "x_vs_y_height_comb"]
        self.data_column = pd.DataFrame(columns=self.name)

        # distances
        self.x_dist_n = pd.Series(dtype=float)
        self.x_dist = pd.DataFrame(dtype=float)

        self.y_dist_n = pd.Series(dtype=float)
        self.y_dist = pd.DataFrame(dtype=float)

        self.xy_dist_n = pd.Series(dtype=float)
        self.x_vs_y = pd.DataFrame(dtype=float)

        # combinatorial
        self.x_comb_n = pd.Series(dtype=float)
        self.x_comb = pd.DataFrame(dtype=float)

        self.y_comb_n = pd.Series(dtype=float)
        self.y_comb = pd.DataFrame(dtype=float)

        self.xy_comb_n = pd.Series(dtype=float)
        self.x_vs_y_comb = pd.DataFrame(dtype=float)

        self.d = pd.Series(dtype=float)
        # height parameters
        self.tree_dist_prev = pd.Series(dtype=float)
        self.tree_comb_prev = pd.Series(dtype=float)

        self.changed_before = pd.Series(dtype=float)

    def init_fun(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                 leaf_pair_in_forest, leaf_pair_in, new_cherries=False):
        if not new_cherries:
            self.tree_dist_prev = copy.deepcopy(tree_height_dist)
            self.tree_comb_prev = copy.deepcopy(tree_height_comb)

            self.x_dist = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
            self.y_dist = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
            self.x_comb = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
            self.y_comb = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
            self.x_vs_y = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
            self.x_vs_y_comb = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)

            self.d = leaf_pair_in

        for c in unique_cherries:
            for t, tree in tree_set.items():
                if not leaf_pair_in_forest.loc[c, t]:
                    continue
                x_height, x_height_comb, y_height, y_height_comb, x_vs_y, x_vs_y_comb = self.get_leaf_height(tree, *c)
                # data stuff
                # distances
                self.x_dist.loc[c, t] = x_height
                self.x_dist.loc[c[::-1], t] = y_height

                self.y_dist.loc[c, t] = y_height
                self.y_dist.loc[c[::-1], t] = x_height

                self.x_vs_y.loc[c, t] = x_vs_y
                self.x_vs_y.loc[c[::-1], t] = 1 / x_vs_y
                # combinatorial
                self.x_comb.loc[c, t] = x_height_comb
                self.x_comb.loc[c[::-1], t] = y_height_comb

                self.y_comb.loc[c, t] = y_height_comb
                self.y_comb.loc[c[::-1], t] = x_height_comb

                self.x_vs_y_comb.loc[c, t] = x_vs_y_comb
                self.x_vs_y_comb.loc[c[::-1], t] = 1 / x_vs_y_comb

        # DATA
        if not new_cherries:
            # leaf heights
            self.x_dist_n = (self.x_dist / self.tree_dist_prev).sum(axis=1)
            self.y_dist_n = (self.y_dist / self.tree_dist_prev).sum(axis=1)

            self.x_comb_n = (self.x_comb / self.tree_comb_prev).sum(axis=1)
            self.y_comb_n = (self.y_comb / self.tree_comb_prev).sum(axis=1)

            self.data_column["x_height"] = self.x_dist_n / self.d
            self.data_column["y_height"] = self.y_dist_n / self.d

            self.data_column["x_height_comb"] = self.x_comb_n / self.d
            self.data_column["y_height_comb"] = self.y_comb_n / self.d

            # leaf distance frac
            self.xy_dist_n = self.x_vs_y.sum(axis=1)
            self.xy_comb_n = self.x_vs_y_comb.sum(axis=1)

            self.data_column["x_vs_y_height"] = self.xy_dist_n / self.d
            self.data_column["x_vs_y_height_comb"] = self.xy_comb_n / self.d

            return self.data_column
        else:
            for c in unique_cherries:
                self.x_dist_n[c] = 0
                self.x_dist_n[c[::-1]] = 0
                self.y_dist_n[c] = 0
                self.y_dist_n[c[::-1]] = 0

                self.x_comb_n[c] = 0
                self.x_comb_n[c[::-1]] = 0
                self.y_comb_n[c] = 0
                self.y_comb_n[c[::-1]] = 0
                for t in tree_set:
                    if np.isnan(self.x_dist.loc[c, t]):
                        continue
                    if not tree_height_dist[t]:
                        x_dist_n_val = 0
                        y_dist_n_val = 0
                    else:
                        # leaf heights
                        x_dist_n_val = self.x_dist.loc[c, t] / tree_height_dist[t]
                        y_dist_n_val = self.y_dist.loc[c, t] / tree_height_dist[t]

                    self.x_dist_n[c] += x_dist_n_val
                    self.x_dist_n[c[::-1]] += y_dist_n_val

                    self.y_dist_n[c] += y_dist_n_val
                    self.y_dist_n[c[::-1]] += x_dist_n_val

                    if not tree_height_comb[t]:
                        x_comb_n_val = 0
                        y_comb_n_val = 0
                    else:
                        x_comb_n_val = self.x_comb.loc[c, t] / tree_height_comb[t]
                        y_comb_n_val = self.y_comb.loc[c, t] / tree_height_comb[t]

                    self.x_comb_n[c] += x_comb_n_val
                    self.x_comb_n[c[::-1]] += y_comb_n_val

                    self.y_comb_n[c] += y_comb_n_val
                    self.y_comb_n[c[::-1]] += x_comb_n_val

                self.data_column.loc[c, "x_height"] = self.x_dist_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_height"] = self.x_dist_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "y_height"] = self.y_dist_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "y_height"] = self.y_dist_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "x_height_comb"] = self.x_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_height_comb"] = self.x_comb_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "y_height_comb"] = self.y_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "y_height_comb"] = self.y_comb_n[c[::-1]] / self.d[c[::-1]]

                # leaf distance frac
                self.xy_dist_n[c] = self.x_vs_y.loc[c].sum()
                self.xy_dist_n[c[::-1]] = self.x_vs_y.loc[c[::-1]].sum()

                self.xy_comb_n[c] = self.x_vs_y_comb.loc[c].sum()
                self.xy_comb_n[c[::-1]] = self.x_vs_y_comb.loc[c[::-1]].sum()

                self.data_column.loc[c, "x_vs_y_height"] = self.xy_dist_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_vs_y_height"] = self.xy_dist_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "x_vs_y_height_comb"] = self.xy_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_vs_y_height_comb"] = self.xy_comb_n[c[::-1]] / self.d[c[::-1]]

    def get_leaf_height(self, tree, x, y):
        for p in tree.nw.predecessors(x):
            p_x = p
        for p in tree.nw.predecessors(y):
            p_y = p

        x_height = tree.nw.nodes[p_x]["node_length"]
        y_height = tree.nw.nodes[p_y]["node_length"]

        x_vs_y = tree.nw.nodes[x]["node_length"]/tree.nw.nodes[y]["node_length"]

        x_height_comb = tree.nw.nodes[p_x]["node_comb"]
        y_height_comb = tree.nw.nodes[p_y]["node_comb"]

        x_vs_y_comb = tree.nw.nodes[x]["node_comb"] / tree.nw.nodes[y]["node_comb"]

        return x_height, x_height_comb, y_height, y_height_comb, x_vs_y, x_vs_y_comb

    def new_cherries(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                     leaf_pair_in_forest, leaf_pair_in):
        return self.init_fun(reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                             leaf_pair_in_forest, leaf_pair_in, new_cherries=True)

    def update_before(self, chosen_cherry, tree_set, reducible_pairs):
        self.changed_before = pd.Series(0, index=reducible_pairs, dtype=int)
        x, y = chosen_cherry
        for t in reducible_pairs[chosen_cherry]:
            tree = tree_set[t]
            for w in tree.leaves:
                if w == x:
                    continue
                # UPDATE leaf distance for cherry (w, y)
                try:
                    check = reducible_pairs[(w, y)]
                except KeyError:
                    continue
                # if (w, y) in reducible_pairs:
                self.changed_before[(w, y)] = 1
                self.changed_before[(y, w)] = 1

                self.y_comb.loc[(w, y), t] -= 1
                if self.tree_comb_prev[t]:
                    self.y_comb_n[(w, y)] -= 1 / self.tree_comb_prev[t]

                self.x_comb.loc[(y, w), t] -= 1
                if self.tree_comb_prev[t]:
                    self.x_comb_n[(y, w)] -= 1 / self.tree_comb_prev[t]

                y_vs_w_comb = (tree.nw.nodes[y]["node_comb"] - 1) / tree.nw.nodes[w]["node_comb"]
                prev_xy_a = self.x_vs_y_comb.loc[(y, w), t]
                prev_xy_b = self.x_vs_y_comb.loc[(w, y), t]
                self.x_vs_y_comb.loc[(y, w), t] = y_vs_w_comb
                self.xy_comb_n[(y, w)] -= prev_xy_a
                self.xy_comb_n[(y, w)] += y_vs_w_comb

                self.x_vs_y_comb.loc[(w, y), t] = 1 / y_vs_w_comb
                self.xy_comb_n[(w, y)] -= prev_xy_b
                self.xy_comb_n[(w, y)] += 1 / y_vs_w_comb
        return None

    def update_data(self, new_cherries, unique_reducible_pairs, new_height, new_height_comb,
                    change_height_dist, change_height_comb, change_leaf_pair, tree_set):

        for c, trees in unique_reducible_pairs.items():
            change_dist = False
            change_comb = False
            if c in new_cherries:
                continue
            for t in trees:
                if t in change_leaf_pair["trees_out"]:
                    # distances
                    if self.tree_dist_prev[t]:
                        self.x_dist_n[c] -= self.x_dist.loc[c, t] / self.tree_dist_prev[t]
                        self.x_dist_n[c[::-1]] -= self.x_dist.loc[c[::-1], t] / self.tree_dist_prev[t]

                    self.x_dist.loc[c, t] = np.nan
                    self.x_dist.loc[c[::-1], t] = np.nan

                    if self.tree_dist_prev[t]:
                        self.y_dist_n[c] -= self.y_dist.loc[c, t] / self.tree_dist_prev[t]
                        self.y_dist_n[c[::-1]] -= self.y_dist.loc[c[::-1], t] / self.tree_dist_prev[t]

                    self.y_dist.loc[c, t] = np.nan
                    self.y_dist.loc[c[::-1], t] = np.nan

                    self.xy_dist_n[c] -= self.x_vs_y.loc[c, t]
                    self.xy_dist_n[c[::-1]] -= self.x_vs_y.loc[c[::-1], t]

                    self.x_vs_y.loc[c, t] = np.nan
                    self.x_vs_y.loc[c[::-1], t] = np.nan

                    # combinatorial
                    if self.tree_comb_prev[t]:
                        self.x_comb_n[c] -= self.x_comb.loc[c, t] / self.tree_comb_prev[t]
                        self.x_comb_n[c[::-1]] -= self.x_comb.loc[c[::-1], t] / self.tree_comb_prev[t]

                    self.x_comb.loc[c, t] = np.nan
                    self.x_comb.loc[c[::-1], t] = np.nan

                    if self.tree_comb_prev[t]:
                        self.y_comb_n[c] -= self.y_comb.loc[c, t] / self.tree_comb_prev[t]
                        self.y_comb_n[c[::-1]] -= self.y_comb.loc[c[::-1], t] / self.tree_comb_prev[t]

                    self.y_comb.loc[c, t] = np.nan
                    self.y_comb.loc[c[::-1], t] = np.nan

                    self.xy_comb_n[c] -= self.x_vs_y_comb.loc[c, t]
                    self.xy_comb_n[c[::-1]] -= self.x_vs_y_comb.loc[c[::-1], t]

                    self.x_vs_y_comb.loc[c, t] = np.nan
                    self.x_vs_y_comb.loc[c[::-1], t] = np.nan

                    continue
                if change_height_dist[t]:
                    if self.tree_dist_prev[t]:
                        self.x_dist_n[c] -= self.x_dist.loc[c, t] / self.tree_dist_prev[t]
                        self.x_dist_n[c[::-1]] -= self.x_dist.loc[c[::-1], t] / self.tree_dist_prev[t]
                    if new_height[t]:
                        self.x_dist_n[c] += self.x_dist.loc[c, t] / new_height[t]
                        self.x_dist_n[c[::-1]] += self.x_dist.loc[c[::-1], t] / new_height[t]
                    if self.tree_dist_prev[t]:
                        self.y_dist_n[c] -= self.y_dist.loc[c, t] / self.tree_dist_prev[t]
                        self.y_dist_n[c[::-1]] -= self.y_dist.loc[c[::-1], t] / self.tree_dist_prev[t]
                    if new_height[t]:
                        self.y_dist_n[c] += self.y_dist.loc[c, t] / new_height[t]
                        self.y_dist_n[c[::-1]] += self.y_dist.loc[c[::-1], t] / new_height[t]
                    change_dist = True

                if change_height_comb[t]:
                    if self.tree_comb_prev[t]:
                        self.x_comb_n[c] -= self.x_comb.loc[c, t] / self.tree_comb_prev[t]
                        self.x_comb_n[c[::-1]] -= self.x_comb.loc[c[::-1], t] / self.tree_comb_prev[t]
                    if new_height_comb[t]:
                        self.x_comb_n[c] += self.x_comb.loc[c, t] / new_height_comb[t]
                        self.x_comb_n[c[::-1]] += self.x_comb.loc[c[::-1], t] / new_height_comb[t]
                    if self.tree_comb_prev[t]:
                        self.y_comb_n[c] -= self.y_comb.loc[c, t] / self.tree_comb_prev[t]
                        self.y_comb_n[c[::-1]] -= self.y_comb.loc[c[::-1], t] / self.tree_comb_prev[t]
                    if new_height_comb[t]:
                        self.y_comb_n[c] += self.y_comb.loc[c, t] / new_height_comb[t]
                        self.y_comb_n[c[::-1]] += self.y_comb.loc[c[::-1], t] / new_height_comb[t]
                    change_comb = True

            if change_leaf_pair["any_diff"][c]:
                # leaf heights
                self.data_column.loc[c, "x_height"] = self.x_dist_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_height"] = self.x_dist_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "y_height"] = self.y_dist_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "y_height"] = self.y_dist_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "x_height_comb"] = self.x_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_height_comb"] = self.x_comb_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "y_height_comb"] = self.y_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "y_height_comb"] = self.y_comb_n[c[::-1]] / self.d[c[::-1]]

                # leaf distance frac
                self.data_column.loc[c, "x_vs_y_height"] = self.xy_dist_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_vs_y_height"] = self.xy_dist_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "x_vs_y_height_comb"] = self.xy_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_vs_y_height_comb"] = self.xy_comb_n[c[::-1]] / self.d[c[::-1]]
                continue
            if change_dist:
                self.data_column.loc[c, "x_height"] = self.x_dist_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_height"] = self.x_dist_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "y_height"] = self.y_dist_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "y_height"] = self.y_dist_n[c[::-1]] / self.d[c[::-1]]
            if self.changed_before[c]:
                self.data_column.loc[c, "x_height_comb"] = self.x_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_height_comb"] = self.x_comb_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "y_height_comb"] = self.y_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "y_height_comb"] = self.y_comb_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "x_vs_y_height_comb"] = self.xy_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_vs_y_height_comb"] = self.xy_comb_n[c[::-1]] / self.d[c[::-1]]
                continue
            if change_comb:
                self.data_column.loc[c, "x_height_comb"] = self.x_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "x_height_comb"] = self.x_comb_n[c[::-1]] / self.d[c[::-1]]

                self.data_column.loc[c, "y_height_comb"] = self.y_comb_n[c] / self.d[c]
                self.data_column.loc[c[::-1], "y_height_comb"] = self.y_comb_n[c[::-1]] / self.d[c[::-1]]

        for t in tree_set:
            if change_height_dist[t]:
                self.tree_dist_prev[t] = copy.copy(new_height[t])
            if change_height_comb[t]:
                self.tree_comb_prev[t] = copy.copy(new_height_comb[t])

        return None

    def chosen_cherry_cleaning(self, chosen_cherry):
        self.data_column.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.x_dist_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.x_dist.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        self.y_dist_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.y_dist.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        self.xy_dist_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.x_vs_y.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        # combinatorial
        self.x_comb_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.x_comb.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        self.y_comb_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.y_comb.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        self.xy_comb_n.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)
        self.x_vs_y_comb.drop([chosen_cherry, chosen_cherry[::-1]], inplace=True)

        self.changed_before = pd.Series(dtype=float)

    # RELABEL
    def relabel(self, x, y, reducible_pairs, merged_cherries, tree_set, leaf_pair_in_new_trees):
        relabelled = pd.Series(0, index=reducible_pairs, dtype=int)
        unique_reducible_pairs = {tuple(sorted(c)): trees for c, trees in reducible_pairs.items()}
        for c_x, c_y in self.data_column.index:
            if (c_x, c_y) in [(x, y), (y, x)]:
                continue
            if x != c_x:
                continue
            relabelled[(y, c_y)] = True
            relabelled[(c_y, y)] = True
            if (c_x, c_y) in merged_cherries:
                # DIST
                # change x dist
                self.x_dist.loc[(y, c_y)] = self.x_dist.loc[[(x, c_y), (y, c_y)]].max()
                self.x_dist.loc[(c_y, y)] = self.x_dist.loc[[(c_y, x), (c_y, y)]].max()
                self.x_dist = self.x_dist.drop([(x, c_y), (c_y, x)])
                # n
                self.x_dist_n[(y, c_y)] += self.x_dist_n[(x, c_y)]
                self.x_dist_n[(c_y, y)] += self.x_dist_n[(c_y, x)]
                self.x_dist_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change y dist
                self.y_dist.loc[(y, c_y)] = self.y_dist.loc[[(x, c_y), (y, c_y)]].max()
                self.y_dist.loc[(c_y, y)] = self.y_dist.loc[[(c_y, x), (c_y, y)]].max()
                self.y_dist.drop([(x, c_y), (c_y, x)], inplace=True)
                # n
                self.y_dist_n[(y, c_y)] += self.y_dist_n[(x, c_y)]
                self.y_dist_n[(c_y, y)] += self.y_dist_n[(c_y, x)]
                self.y_dist_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change xy dist
                self.x_vs_y.loc[(y, c_y)] = self.x_vs_y.loc[[(x, c_y), (y, c_y)]].max()
                self.x_vs_y.loc[(c_y, y)] = self.x_vs_y.loc[[(c_y, x), (c_y, y)]].max()
                self.x_vs_y.drop([(x, c_y), (c_y, x)], inplace=True)
                # n
                self.xy_dist_n[(y, c_y)] += self.xy_dist_n[(x, c_y)]
                self.xy_dist_n[(c_y, y)] += self.xy_dist_n[(c_y, x)]
                self.xy_dist_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # COMB
                # change x comb
                self.x_comb.loc[(y, c_y)] = self.x_comb.loc[[(x, c_y), (y, c_y)]].max()
                self.x_comb.loc[(c_y, y)] = self.x_comb.loc[[(c_y, x), (c_y, y)]].max()
                self.x_comb.drop([(x, c_y), (c_y, x)], inplace=True)
                # n
                self.x_comb_n[(y, c_y)] += self.x_comb_n[(x, c_y)]
                self.x_comb_n[(c_y, y)] += self.x_comb_n[(c_y, x)]
                self.x_comb_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change y comb
                self.y_comb.loc[(y, c_y)] = self.y_comb.loc[[(x, c_y), (y, c_y)]].max()
                self.y_comb.loc[(c_y, y)] = self.y_comb.loc[[(c_y, x), (c_y, y)]].max()
                self.y_comb.drop([(x, c_y), (c_y, x)], inplace=True)
                # n
                self.y_comb_n[(y, c_y)] += self.y_comb_n[(x, c_y)]
                self.y_comb_n[(c_y, y)] += self.y_comb_n[(c_y, x)]
                self.y_comb_n.drop([(x, c_y), (c_y, x)], inplace=True)
                # change xy comb
                self.x_vs_y_comb.loc[(y, c_y)] = self.x_vs_y_comb.loc[[(x, c_y), (y, c_y)]].max()
                self.x_vs_y_comb.loc[(c_y, y)] = self.x_vs_y_comb.loc[[(c_y, x), (c_y, y)]].max()
                self.x_vs_y_comb.drop([(x, c_y), (c_y, x)], inplace=True)
                # n
                self.xy_comb_n[(y, c_y)] += self.xy_comb_n[(x, c_y)]
                self.xy_comb_n[(c_y, y)] += self.xy_comb_n[(c_y, x)]
                self.xy_comb_n.drop([(x, c_y), (c_y, x)], inplace=True)
            else:
                # DIST
                # change x dist
                self.x_dist.index = list(self.x_dist.index)
                self.x_dist.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.x_dist.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.x_dist.index = pd.MultiIndex.from_tuples(self.x_dist.index)
                # n
                self.x_dist_n.index = list(self.x_dist_n.index)
                self.x_dist_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.x_dist_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.x_dist_n.index = pd.MultiIndex.from_tuples(self.x_dist_n.index)
                # change y dist
                self.y_dist.index = list(self.y_dist.index)
                self.y_dist.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.y_dist.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.y_dist.index = pd.MultiIndex.from_tuples(self.y_dist.index)
                # n
                self.y_dist_n.index = list(self.y_dist_n.index)
                self.y_dist_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.y_dist_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.y_dist_n.index = pd.MultiIndex.from_tuples(self.y_dist_n.index)
                # change xy dist
                self.x_vs_y.index = list(self.x_vs_y.index)
                self.x_vs_y.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.x_vs_y.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.x_vs_y.index = pd.MultiIndex.from_tuples(self.x_vs_y.index)
                # n
                self.xy_dist_n.index = list(self.xy_dist_n.index)
                self.xy_dist_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.xy_dist_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.xy_dist_n.index = pd.MultiIndex.from_tuples(self.xy_dist_n.index)
                # COMB
                # change x comb
                self.x_comb.index = list(self.x_comb.index)
                self.x_comb.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.x_comb.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.x_comb.index = pd.MultiIndex.from_tuples(self.x_comb.index)
                # n
                self.x_comb_n.index = list(self.x_comb_n.index)
                self.x_comb_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.x_comb_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.x_comb_n.index = pd.MultiIndex.from_tuples(self.x_comb_n.index)
                # change y comb
                self.y_comb.index = list(self.y_comb.index)
                self.y_comb.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.y_comb.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.y_comb.index = pd.MultiIndex.from_tuples(self.y_comb.index)
                # n
                self.y_comb_n.index = list(self.y_comb_n.index)
                self.y_comb_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.y_comb_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.y_comb_n.index = pd.MultiIndex.from_tuples(self.y_comb_n.index)
                # change xy comb
                self.x_vs_y_comb.index = list(self.x_vs_y_comb.index)
                self.x_vs_y_comb.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.x_vs_y_comb.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.x_vs_y_comb.index = pd.MultiIndex.from_tuples(self.x_vs_y_comb.index)
                # n
                self.xy_comb_n.index = list(self.xy_comb_n.index)
                self.xy_comb_n.rename(index={(x, c_y): (y, c_y)}, inplace=True)
                self.xy_comb_n.rename(index={(c_y, x): (c_y, y)}, inplace=True)
                self.xy_comb_n.index = pd.MultiIndex.from_tuples(self.xy_comb_n.index)
            # DATA
            self.data_column.drop([(x, c_y), (c_y, x)], inplace=True)

        for c in unique_reducible_pairs:
            for t in leaf_pair_in_new_trees[c]:
                relabelled[c] = True
                relabelled[c[::-1]] = True

                x_height, x_height_comb, y_height, y_height_comb, x_vs_y, x_vs_y_comb = \
                    self.get_leaf_height(tree_set[t], *c)
                # data stuff
                # distances
                self.x_dist.loc[c, t] = x_height
                self.x_dist.loc[c[::-1], t] = y_height
                self.y_dist.loc[c, t] = y_height
                self.y_dist.loc[c[::-1], t] = x_height
                if self.tree_dist_prev[t]:
                    self.x_dist_n[c] += x_height / self.tree_dist_prev[t]
                    self.x_dist_n[c[::-1]] += y_height / self.tree_dist_prev[t]
                    self.y_dist_n[c] += y_height / self.tree_dist_prev[t]
                    self.y_dist_n[c[::-1]] += x_height / self.tree_dist_prev[t]

                self.x_vs_y.loc[c, t] = x_vs_y
                self.x_vs_y.loc[c[::-1], t] = 1 / x_vs_y
                self.xy_dist_n[c] += x_vs_y
                self.xy_dist_n[c[::-1]] += 1 / x_vs_y
                # combinatorial
                self.x_comb.loc[c, t] = x_height_comb
                self.x_comb.loc[c[::-1], t] = y_height_comb
                self.y_comb.loc[c, t] = y_height_comb
                self.y_comb.loc[c[::-1], t] = x_height_comb
                if self.tree_comb_prev[t]:
                    self.x_comb_n[c] += x_height_comb / self.tree_dist_prev[t]
                    self.x_comb_n[c[::-1]] += y_height_comb / self.tree_dist_prev[t]
                    self.y_comb_n[c] += y_height_comb / self.tree_dist_prev[t]
                    self.y_comb_n[c[::-1]] += x_height_comb / self.tree_dist_prev[t]

                self.x_vs_y_comb.loc[c, t] = x_vs_y_comb
                self.x_vs_y_comb.loc[c[::-1], t] = 1 / x_vs_y_comb
                self.xy_comb_n[c] += x_vs_y_comb
                self.xy_comb_n[c[::-1]] += 1 / x_vs_y_comb

        for c in reducible_pairs:
            if relabelled[c]:
                # update data
                self.data_column.loc[c, "x_height"] = self.x_dist_n[c] / self.d[c]

                self.data_column.loc[c, "y_height"] = self.y_dist_n[c] / self.d[c]

                self.data_column.loc[c, "x_vs_y_height"] = self.xy_dist_n[c] / self.d[c]

                self.data_column.loc[c, "x_height_comb"] = self.x_comb_n[c] / self.d[c]

                self.data_column.loc[c, "y_height_comb"] = self.y_comb_n[c] / self.d[c]

                self.data_column.loc[c, "x_vs_y_height_comb"] = self.xy_comb_n[c] / self.d[c]
