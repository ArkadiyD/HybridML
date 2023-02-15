import networkx as nx
import random
import ast
import re
import time
import joblib
import numpy as np
from copy import deepcopy
from copy import copy
from TreeWidthTreeContainment import BOTCH

import pandas as pd

from Features import Features
memodict = {}


'''
This script consists of 4 parts: 
- Help functions (mainly for handling the input)
- Input_Set CLASS: code for opening the input and running the Cherry Picking Heuristic (CPH)
- PhT CLASS: environment for a phylogenetic tree
- PhN CLASS: environment for a phylogenetic network
'''

#######################################################################
########                   HELP FUNCTIONS                      ########
#######################################################################

# Write length newick: convert ":" to "," and then evaluate as list of lists using ast.literal_eval
# Then, in each list, the node is followed by the length of the incoming arc.
# This only works as long as each branch has length and all internal nodes are labeled.
def newick_to_tree(newick, current_labels=dict()):
	# newick = newick[:-1]
	distances = False
	# presence of : indicates the use of lengths in the trees
	if ":" in newick:
		distances = True
		# taxon names may already be enclosed by " or ', otherwise, we add these now
		if "'" not in newick and '"' not in newick:
			newick = re.sub(r"([,\(])([a-zA-Z\d]+)", r"\1'\2", newick)
			newick = re.sub(r"([a-zA-Z\d]):", r"\1':", newick)
		newick = newick.replace(":", ",")
	else:
		# taxon names may already be enclosed by " or ', otherwise, we add these now
		if not "'" in newick and not '"' in newick:
			newick = re.sub(r"([,\(])([a-zA-Z\d]+)", r"\1'\2", newick)
			newick = re.sub(r"([a-zA-Z\d])([,\(\)])", r"\1'\2", newick)
	# turn the string into a pyhton nested list using [ instead of (
	newick = newick.replace("(", "[")
	newick = newick.replace(")", "]")
	nestedtree = ast.literal_eval(newick)
	# parse the nested list into a list of edges with some additional information about the leaves
	# we start with the root 2, so that we can append a root edge (1,2)
	edges, leaves, current_labels, current_node = nested_list_to_tree(nestedtree, 2, current_labels, distances=distances)
	# put all this information into a networkx DiGraph with or without distances/lengths
	tree = nx.DiGraph()
	if distances:
		edges.append((1, 2, 0))
		tree.add_weighted_edges_from(edges, weight='length')

	else:
		edges.append((1, 2, 0))
		# tree.add_edges_from(edges)
		tree.add_weighted_edges_from(edges, weight='length')
	add_node_attributes(tree, distances=distances, root=2)
	print(tree, leaves, current_labels)
	return tree, leaves, current_labels, distances


# Auxiliary function to convert list of lists to tree (graph)
# Works recursively, where we keep track of the nodes we have already used
# Leaves are nodes with negative integer as ID, and already existing taxa are coupled to node IDs by current_labels.
def nested_list_to_tree(nestedList, next_node, current_labels, distances=False):
	edges = []
	leaves = set()
	top_node = next_node
	current_node = next_node + 1
	if distances:
		# each element in the sublist has 2 properties, the subtree, and the length, which are adjacent in nestedList
		for i in range(0, len(nestedList), 2):
			t = nestedList[i]
			length = nestedList[i + 1]
			if type(t) == list:  # Not a leaf
				edges.append((top_node, current_node, length))
				extra_edges, extra_leaves, current_labels, current_node = nested_list_to_tree(t, current_node,
																							 current_labels,
																							 distances=distances)
			else:  # A leaf
				if str(t) not in current_labels:
					current_labels[str(t)] = -len(current_labels)
				edges.append((top_node, current_labels[str(t)], length))
				extra_edges = []
				extra_leaves = {current_labels[str(t)]}
			edges = edges + extra_edges
			leaves = leaves.union(extra_leaves)
	else:
		# no lengths/distances, so each subtree is simply an element of nestedList
		for t in nestedList:
			length = 1
			if type(t) == list:
				edges.append((top_node, current_node, length))
				extra_edges, extra_leaves, current_labels, current_node = nested_list_to_tree(t, current_node,
																							 current_labels)
			else:
				if str(t) not in current_labels:
					current_labels[str(t)] = -len(current_labels)
				edges.append((top_node, current_labels[str(t)], length))
				extra_edges = []
				extra_leaves = {current_labels[str(t)]}
			edges = edges + extra_edges
			leaves = leaves.union(extra_leaves)
	return edges, leaves, current_labels, current_node


# per node, add the edge based and comb height of the node as an attribute.
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
					node_comb = nx.algorithms.shortest_paths.generic.shortest_path_length(tree, root, x)
					attrs[x] = {"node_length": node_comb,
								"node_comb": node_comb}
				except nx.exception.NetworkXNoPath:
					attrs[x] = {"node_length": None, "node_comb": None}

	nx.set_node_attributes(tree, attrs)


# Modifies a cherry-picking sequence so that it represents a network with exactly one root.
# A sequence may be such that reconstructing a network from the sequence results in multiple roots
# This function adds some pairs to the sequence so that the network has a single root.
# returns the new sequence, and also modifies the sets of trees reduced by each pair in the sequence,
# so that the new pairs are also represented (they reduce no trees)
def sequence_add_roots(seq, red_trees):
	leaves_encountered = set()
	roots = set()
	# The roots can be found by going back through the sequence and finding pairs where the second element has not been
	# encountered in the sequence yet
	for pair in reversed(seq):
		if pair[1] not in leaves_encountered:
			roots.add(pair[1])
		leaves_encountered.add(pair[0])
		leaves_encountered.add(pair[1])
	i = 0
	roots = list(roots)
	# Now add some pairs to make sure each second element is already part of some pair in the sequence read backwards,
	# except for the last pair in the sequence
	for i in range(len(roots) - 1):
		seq.append((roots[i], roots[i + 1]))
		# none of the trees are reduced by the new pairs.
		red_trees.append(set())
		i += 1
	return seq, red_trees


# return leaves from network
def get_leaves(net):
	return {u for u in net.nodes() if net.out_degree(u) == 0}

#######################################################################
########            INPUT SET CLASS with CPH method            ########
#######################################################################


# Methods for sets of phylogenetic trees
class Input_Set:
	def __init__(self, newick_strings=[], tree_set=None, instance=0, leaves=None, full_leaf_set=True):
		# The dictionary of trees
		self.trees = dict()
		# the set of leaf labels of the trees
		self.labels = dict()
		self.labels_reversed = dict()
		self.leaves = set()
		self.instance = instance

		# the current best sequence we have found for this set of trees
		self.best_seq = None
		# the list of reduced trees for each of the pairs in the best sequence
		self.best_red_trees = None

		# the best sequence for the algorithm using lengths as input as well
		self.best_seq_with_lengths = None
		# the sets of reduced trees for each pair in this sequence
		self.best_seq_with_lengths_red_trees = None
		# the height of each pair in this sequence
		self.best_seq_with_lengths_heights = None

		# true if distances are used
		self.distances = True
		# computation times
		self.CPS_Compute_Time = 0
		self.CPS_Compute_Reps = 0
		self.DurationPerTrial = []
		self.RetPerTrial = []

		if tree_set is None:
			# read the input trees in 'newick_strings'
			for n in newick_strings:
				tree = PhT()
				self.trees[len(self.trees)] = tree
				self.labels, distances_in_tree = tree.tree_from_newick(newick=n, current_labels=self.labels)
				self.distances = self.distances and distances_in_tree
			# ONLY INTERSECTION
			self.leaves = set(self.labels.keys())
		else:
			self.trees = {t: PhT(tree) for t, tree in tree_set.items()}
			self.leaves = leaves
			self.labels = {l: l for l in self.leaves}
		#(len(self.trees))
		if full_leaf_set:
			self.unique_leaves = self.leaves
		else:
			self.unique_leaves = self.trees[0].leaves
			for t, tree in self.trees.items():
				if t == 0:
					continue
				self.unique_leaves = tree.leaves.intersection(self.unique_leaves)
		self.num_leaves = len(self.unique_leaves)
		print(f'{self.num_leaves=}')
		# make a reverse dictionary for the leaf labels, to look up the label of a given node
		for l, i in self.labels.items():
			self.labels_reversed[i] = l

	# Make a deepcopy of an instance
	def __deepcopy__(self, memodict={}):
		copy_inputs = Input_Set()
		copy_inputs.trees = deepcopy(self.trees, memodict)
		copy_inputs.labels = deepcopy(self.labels, memodict)
		copy_inputs.labels_reversed = deepcopy(self.labels_reversed, memodict)
		copy_inputs.leaves = deepcopy(self.leaves, memodict)
		return copy_inputs

	# Find new cherry-picking sequences for the trees and update the best found
	def CPSBound(self, repeats=1,
				 progress=False,
				 time_limit=None,
				 reduce_trivial=False,
				 pick_ml=False,
				 pick_ml_triv=False,
				 pick_random=False,
				 model_name=None,
				 relabel=False,
				 relabel_cher_triv=False,
				 ml_thresh=None,
				 problem_type=None):
		# Set the specific heuristic that we use, based on the user input and whether the trees have lengths
		Heuristic = self.CPHeuristic
		# Initialize the recorded best sequences and corresponding data
		best = None
		red_trees_best = []
		starting_time = time.time()
		self.DurationPerTrial = dict()
		self.RetPerTrial = dict()
		# Try as many times as required by the integer 'repeats'
		df_pred = None
		for i in np.arange(repeats):
			start_trial = time.time()
			if not (pick_ml or pick_ml_triv) and progress:
				print(f"\nInstance {self.instance} {problem_type}: Trial {i}\n")
			# RUN HEURISTIC
			new, reduced_trees, df_pred = Heuristic(progress=progress,
													reduce_trivial=reduce_trivial,
													pick_ml=pick_ml,
													pick_ml_triv=pick_ml_triv,
													pick_random=pick_random,
													model_name=model_name,
													relabel=relabel,
													relabel_cher_triv=relabel_cher_triv,
													ml_thresh=ml_thresh,
													problem_type=problem_type)
			if progress:
				print(f"Instance {self.instance} {problem_type}: found sequence of length: {len(new)}")
			# COMPLETE PARTIAL SEQUENCE
			new, reduced_trees = sequence_add_roots(new, reduced_trees)
			print(new)
			if progress:
				print(f"Instance {self.instance} {problem_type}: length after completing sequence: {len(new)}")

			self.CPS_Compute_Reps += 1
			self.DurationPerTrial[i] = time.time() - start_trial

			# FIND RETICULATION NUMBER
			seq_length = len(new)
			self.RetPerTrial[i] = seq_length - self.num_leaves + 1
			# store best sequence
			if best is None or seq_length < best_length:
				best = new
				best_length = copy(seq_length)
				red_trees_best = reduced_trees
			if progress:
				print(f"Instance {self.instance} {problem_type}: best sequence has length: {best_length}")
			if time_limit and time.time() - starting_time > time_limit:
				break

		# storing stuff of heuristic
		self.CPS_Compute_Time += time.time() - starting_time
		# storing best network
		new_seq = best
		if not self.best_seq_with_lengths or len(new_seq) < len(self.best_seq_with_lengths):
			converted_new_seq = []
			for pair in new_seq:
				converted_new_seq += [(self.labels_reversed[pair[0]], self.labels_reversed[pair[1]])]
			self.best_seq_with_lengths = converted_new_seq
			self.best_seq_with_lengths_red_trees = red_trees_best
		seq_return = [(self.labels_reversed[x], self.labels_reversed[y]) for x, y in new_seq]
		return self.best_seq_with_lengths, seq_return, df_pred

	def CPHeuristic(self, progress=False, reduce_trivial=True, pick_ml=False,
					pick_ml_triv=False, model_name=None, pick_random=False, relabel=False, relabel_cher_triv=False,
					ml_thresh=None, problem_type=None):
		# Works in a copy of the input trees, copy_of_inputs, because trees have to be reduced somewhere.
		copy_of_inputs = deepcopy(self)
		CPS = []
		reduced_trees = []
		# Make dict of reducible pairs
		reducible_pairs = self.find_all_pairs()
		triv_picked = False
		if pick_ml or pick_ml_triv:
			# create initial features
			start_time_init = time.time()
			features = Features(reducible_pairs, copy_of_inputs.trees, root=2)
			df_pred = pd.DataFrame(columns=["x", "y", "no_cher_pred", "cher_pred", "ret_cher_pred", "no_ret_cher_pred", "trees_reduced"])
			if progress:
				print(
					f"Instance {self.instance} {problem_type}: Initial features found in {np.round(time.time() - start_time_init, 3)}s")
			# open prediction model
			if model_name is None:
				model_name = "LearningCherries/RFModels/rf_cherries_N1000_maxL20_random_balanced.joblib"
			rf_model = joblib.load(model_name)
		else:
			df_pred = None
		while copy_of_inputs.trees:
			if progress and (pick_ml or pick_ml_triv):
				print(f"Instance {self.instance} {problem_type}: Sequence has length {len(CPS)}")
				print(f"Instance {self.instance} {problem_type}: {len(copy_of_inputs.trees)} trees left.\n")
			if reduce_trivial:
				chosen_cherry, triv_picked = copy_of_inputs.pick_trivial(reducible_pairs)
				if chosen_cherry is None:
					pick_random = True
				else:
					pick_random = False

			if pick_ml_triv:
				trivial_slice = features.data["trivial"] == 1
				if trivial_slice.any():       # find trivial cherry
					triv_cherries = list(features.data.loc[trivial_slice].index)
					if relabel:
						triv_cherry_num = np.random.choice(np.arange(len(triv_cherries)))
						chosen_cherry = triv_cherries[triv_cherry_num]
					else:
						chosen_cherry = copy_of_inputs.pick_cherry(triv_cherries, reducible_pairs)
					if features.data.loc[chosen_cherry]["cherry_in_tree"] < 1:
						triv_picked = True
					else:
						triv_picked = False
					pick_ml = False
				else:
					pick_ml = True

			if pick_ml:
				# predict if cherry
				prediction = pd.DataFrame(np.array([p[:, 1] for p in rf_model.predict_proba(
					features.data)]).transpose(), index=features.data.index)

				max_cherry = (prediction[1] + prediction[2]).argmax()
				chosen_cherry = prediction.index[max_cherry]
				chosen_cherry_prob = prediction.loc[chosen_cherry, 1] + prediction.loc[chosen_cherry, 2]

				if ml_thresh is not None and chosen_cherry_prob < ml_thresh:
					random_cherry_num = np.random.choice(len(reducible_pairs))
					chosen_cherry = list(reducible_pairs)[random_cherry_num]

				df_pred.loc[len(df_pred)] = [*chosen_cherry, *prediction.loc[chosen_cherry], np.nan]

				if relabel and features.data.loc[chosen_cherry]["trivial"] == 1 and features.data.loc[chosen_cherry]["cherry_in_tree"] < 1:
					triv_picked = True
				elif relabel_cher_triv and features.data.loc[chosen_cherry]["trivial"] == 1 and \
						features.data.loc[chosen_cherry]["cherry_in_tree"] < 1 and \
						prediction.loc[chosen_cherry][1] > prediction.loc[chosen_cherry][[0, 2, 3]].sum():
					triv_picked = True
				else:
					triv_picked = False

				if progress and (pick_ml or pick_ml_triv):
					print(f"Instance {self.instance} {problem_type}: chosen cherry = {chosen_cherry}, "
						  f"ML prediction = {list(prediction.loc[chosen_cherry])}")

			if pick_random:
				random_cherry_num = np.random.choice(len(reducible_pairs))
				chosen_cherry = list(reducible_pairs)[random_cherry_num]
				if reduce_trivial:
					triv_picked = False
				elif relabel and copy_of_inputs.trivial_check(chosen_cherry, reducible_pairs[chosen_cherry]):
					triv_picked = True
				else:
					triv_picked = False

			CPS += [chosen_cherry]

			# RELABEL
			if triv_picked and relabel:
				reducible_pairs, merged_cherries = copy_of_inputs.relabel_trivial(*chosen_cherry, reducible_pairs)
				if pick_ml or pick_ml_triv:
					features.relabel_trivial_features(*chosen_cherry, reducible_pairs, merged_cherries, copy_of_inputs.trees)

			# UPDATE SOME FEATURES BEFORE
			if pick_ml or pick_ml_triv:
				features.update_cherry_features_before(chosen_cherry, reducible_pairs, copy_of_inputs.trees)

			# REDUCE CHOSEN CHERRY FROM FOREST
			new_reduced = copy_of_inputs.reduce_pair_in_all(chosen_cherry, reducible_pairs=reducible_pairs)
			reducible_pairs = copy_of_inputs.update_reducible_pairs(reducible_pairs, new_reduced)
			if progress and (pick_ml or pick_ml_triv):
				print(f"Instance {self.instance} {problem_type}: {len(reducible_pairs)} reducible pairs left")
			reduced_trees += [new_reduced]

			if len(copy_of_inputs.trees) == 0:
				break

			# UPDATE FEATURES AFTER REDUCTION
			if pick_ml or pick_ml_triv:
				copy_of_inputs.update_node_comb_length(*chosen_cherry, new_reduced)
				features.update_cherry_features_after(chosen_cherry, reducible_pairs, copy_of_inputs.trees, new_reduced)

		return CPS, reduced_trees, df_pred

	# select order of chosen cherry
	def pick_order(self, x, y, new_reduced, return_cherry=True):
		leaf_x_left = 0
		leaf_y_left = 0
		for t, tree in self.trees.items():
			if t in new_reduced:
				continue
			if x in tree.leaves:
				leaf_x_left += 1
			if y in tree.leaves:
				leaf_y_left += 1
		if return_cherry:
			# FAVOR X, Y OVER Y, X
			if leaf_x_left <= leaf_y_left:
				return x, y
			else:
				return y, x
		else:
			return leaf_x_left, leaf_y_left

	def pick_cherry(self, triv_cherries, reducible_pairs):
		leaf_left = dict()
		for x, y in triv_cherries:
			leaf_x_left, leaf_y_left = self.pick_order(x, y, reducible_pairs[x, y], return_cherry=False)
			leaf_left[(x, y)] = leaf_x_left
			leaf_left[(y, x)] = leaf_y_left
		best_cherry_id = np.argmin(list(leaf_left.values()))
		return list(leaf_left)[best_cherry_id]

	# when using machine learning, update the topological/combinatorial length of nodes
	def update_node_comb_length(self, x, y, reduced_trees):
		for t in reduced_trees:
			try:
				self.trees[t].nw.nodes[y]["node_comb"] -= 1
			except KeyError:
				continue

	# Finds the set of reducible pairs in all trees
	# Returns a dictionary with reducible pairs as keys, and the trees they reduce as values.
	def find_all_pairs(self):
		reducible_pairs = dict()
		for i, t in self.trees.items():
			red_pairs_t = t.find_all_reducible_pairs()
			for pair in red_pairs_t:
				if pair in reducible_pairs:
					reducible_pairs[pair].add(i)
				else:
					reducible_pairs[pair] = {i}
		return reducible_pairs

	# Returns the updated dictionary of reducible pairs in all trees after a reduction (with the trees they reduce as values)
	# we only need to update for the trees that got reduced: 'new_red_treed'
	def update_reducible_pairs(self, reducible_pairs, new_red_trees):
		# Remove trees to update from all pairs
		pair_del = []
		for pair, trees in reducible_pairs.items():
			trees.difference_update(new_red_trees)
			if len(trees) == 0:
				pair_del.append(pair)
		for pair in pair_del:
			del reducible_pairs[pair]
		# Add the trees to the right pairs again
		for index in new_red_trees:
			if index in self.trees:
				t = self.trees[index]
				red_pairs_t = t.find_all_reducible_pairs()
				for pair in red_pairs_t:
					if pair in reducible_pairs:
						reducible_pairs[pair].add(index)
					else:
						reducible_pairs[pair] = {index}
		return reducible_pairs

	# reduces the given pair in all trees
	# Returns the set of trees that were reduced
	# CHANGES THE SET OF TREES, ONLY PERFORM IN A COPY OF THE CLASS INSTANCE
	def reduce_pair_in_all(self, pair, reducible_pairs=None):
		if not len(reducible_pairs):
			print("no reducible pairs")
		if reducible_pairs is None:
			reducible_pairs = dict()
		reduced_trees_for_pair = []
		if pair in reducible_pairs:
			trees_to_reduce = reducible_pairs[pair]
		else:
			trees_to_reduce = deepcopy(self.trees)
		for t in trees_to_reduce:
			if t in self.trees:
				tree = self.trees[t]
				# print(t, tree.leaves)
				if tree.reduce_pair(*pair):
					reduced_trees_for_pair += [t]
					if (self.trees[t].root == 0 and len(tree.nw.edges()) <= 1) or \
							(self.trees[t].root == 2 and len(tree.nw.edges()) <= 2):
						# print(t, pair, tree.leaves)
						del self.trees[t]
		return set(reduced_trees_for_pair)

	def pick_trivial(self, reducible_pairs):
		trivial_cherries = []
		trivial_in_all_cherries = []
		for c, trees in reducible_pairs.items():
			if len(trees) == len(self.trees):
				trivial_in_all_cherries.append(c)
				continue
			trivial_check = self.trivial_check(c, trees)
			if trivial_check:
				trivial_cherries.append(c)

		if trivial_in_all_cherries:
			chosen_cherry = trivial_in_all_cherries[np.random.choice(len(trivial_in_all_cherries))]
			triv_picked = False
		elif trivial_cherries:
			chosen_cherry = trivial_cherries[np.random.choice(len(trivial_cherries))]
			triv_picked = True
		else:
			chosen_cherry = None
			triv_picked = False

		return chosen_cherry, triv_picked

	def trivial_check(self, c, trees):
		if len(trees) == len(self.trees):
			return False
		return len([t for t, tree in self.trees.items() if (set(c).issubset(tree.leaves) and t not in trees)]) == 0

	# Returns all trivial pairs involving the leaf l
	def trivial_pair_with(self, l):
		pairs = set()
		# Go through all trees t with index i.
		for i, t in self.trees.items():
			# If the leaf occurs in t
			if l in t.leaves:
				# Compute reducible pairs of t with the leaf as first coordinate
				pairs_in_t = t.find_pairs_with_first(l)
				# If we did not have a set of candidate pairs yet, use pairs_in_t
				if not pairs:
					pairs = pairs_in_t
				# Else, the candidate pairs must also be in t, so take intersection
				else:
					pairs = pairs & pairs_in_t
				# If we do not have any candidate pairs after checking a tree with l as leaf, we stop.
				if not pairs:
					break
		return pairs

	def relabel_trivial(self, x, y, reducible_pairs):
		# print(f"Cherry = {(x, y)}: RELABEL X = {x} to Y = {y}")
		merged_cherries = set()
		new_cherries = set()
		for t, tree in self.trees.items():
			if t in reducible_pairs[(x, y)]:
				continue
			if x in tree.leaves:
				# change leaf set
				tree.leaves.remove(x)
				tree.leaves.add(y)

				# relabel x to y
				tree.nw = nx.relabel_nodes(tree.nw, {x: y})

				# check if we have a new cherry now
				for p in tree.nw.predecessors(y):
					for c in tree.nw.successors(p):
						if c == y:
							continue
						if c not in tree.leaves:
							continue
						if (c, y) in reducible_pairs:
							reducible_pairs[(c, y)].add(t)
							reducible_pairs[(y, c)].add(t)
							try:
								del reducible_pairs[(c, x)], reducible_pairs[(x, c)]
								merged_cherries.add((x, c))
							except KeyError:
								pass
						else:
							# add to reducible_pairs?
							reducible_pairs[(c, y)] = {t}
							reducible_pairs[(y, c)] = {t}
							new_cherries.add((c, y))
							try:
								del reducible_pairs[(c, x)], reducible_pairs[(x, c)]
							except KeyError:
								pass
		return reducible_pairs, merged_cherries


#######################################################################
########              PHYLOGENETIC TREE CLASS                  ########
#######################################################################

# A class representing a phylogenetic tree
# Contains methods to reduce trees
class PhT:
	def __init__(self, tree=None):
		if tree is None:
			# the actual graph
			self.nw = nx.DiGraph()
			# the set of leaf labels of the network
			self.leaves = set()
			self.root = 2
		else:
			self.nw = tree
			self.root = 0
			self.leaves = get_leaves(self.nw)

	# Builds a tree from a newick string
	def tree_from_newick(self, newick=None, current_labels=dict()):
		self.nw, self.leaves, current_labels, distances = newick_to_tree(newick, current_labels)
		return current_labels, distances

	# Checks whether the pair (x,y) forms a cherry in the tree
	def is_cherry(self, x, y):
		if (x not in self.leaves) or (y not in self.leaves):
			return False
		px = -1
		py = -1
		for p in self.nw.predecessors(x):
			px = p
		for p in self.nw.predecessors(y):
			py = p
		return px == py

	# Returns the height of (x,y) if it is a cherry:
	#     i.e.: length(p,x)+length(p,y)/2
	# Returns false otherwise
	def height_of_cherry(self, x, y):
		if (x not in self.leaves) or (y not in self.leaves):
			return False
		px = -1
		py = -1
		for p in self.nw.predecessors(x):
			px = p
		for p in self.nw.predecessors(y):
			py = p
		if px == py:
			height = [float(self.nw[px][x]['length']), float(self.nw[py][y]['length'])]
			return height
		return False

		# suppresses a degree-2 node v and returns true if successful

	# the new arc has length length(p,v)+length(v,c)
	# returns false if v is not a degree-2 node
	def clean_node(self, v):
		if self.nw.out_degree(v) == 1 and self.nw.in_degree(v) == 1:
			pv = -1
			for p in self.nw.predecessors(v):
				pv = p
			cv = -1
			for c in self.nw.successors(v):
				cv = c
			self.nw.add_edges_from([(pv, cv, self.nw[pv][v])])
			if 'length' in self.nw[pv][v] and 'length' in self.nw[v][cv]:
				self.nw[pv][cv]['length'] = self.nw[pv][v]['length'] + self.nw[v][cv]['length']
			self.nw.remove_node(v)
			return True
		return False

	# reduces the pair (x,y) in the tree if it is present as cherry
	# i.e., removes the leaf x and its incoming arc, and then cleans up its parent node.
	# note that if px, and py have different lengths, the length of px is lost in the new network.
	# returns true if successful and false otherwise
	def reduce_pair(self, x, y):
		if x not in self.leaves or y not in self.leaves:
			return False
		px = - 1
		py = - 1
		for p in self.nw.predecessors(x):
			px = p
		for p in self.nw.predecessors(y):
			py = p
		if self.is_cherry(x, y):
			self.nw.remove_node(x)
			self.leaves.remove(x)
			self.clean_node(py)
			return True
		return False

	# Returns all reducible pairs in the tree involving x, where x is the first element
	def find_pairs_with_first(self, x):
		pairs = set()
		px = -1
		for p in self.nw.predecessors(x):
			px = p
		if self.nw.out_degree(px) > 1:
			for cpx in self.nw.successors(px):
				if cpx in self.leaves:
					if cpx == x:
						continue
					pairs.add((x, cpx))
		return pairs - {x, x}

	# Returns all reducible pairs in the tree
	def find_all_reducible_pairs(self):
		red_pairs = set()
		for l in self.leaves:
			red_pairs = red_pairs.union(self.find_pairs_with_first(l))
		return red_pairs


#######################################################################
########              PHYLOGENETIC NETWORK CLASS               ########
#######################################################################


# A class for phylogenetic networks
class PhN:
	def __init__(self, net=None, seq=set(), newick=None, best_tree_from_network=None, reduced_trees=None, heights=None):
		# the actual graph
		self.nw = nx.DiGraph()
		# the set of leaf labels of the network
		self.leaves = set()
		# a dictionary giving the node for a given leaf label
		self.labels = dict()
		# the number of nodes in the graph
		self.no_nodes = 0
		self.leaf_nodes = dict()
		self.TCS = seq
		self.CPS = seq
		self.newick = newick
		self.reducible_pairs = set()
		self.reticulated_cherries = set()
		self.cherries = set()
		self.level = None
		self.no_embedded_trees = 0
		# if a cherry-picking sequence is given, build the network from this sequence
		if seq:
			total_len = len(seq)
			current_trees_embedded = set()
			# Creates a phylogenetic network from a cherry picking sequence:
			if reduced_trees:
				for i, pair in enumerate(reversed(seq)):
					if heights:
						self.add_pair(*pair, red_trees=reduced_trees[total_len - 1 - i],
									  current_trees=current_trees_embedded, height=heights[total_len - 1 - i])
						current_trees_embedded = current_trees_embedded | reduced_trees[total_len - 1 - i]
					else:
						self.add_pair(*pair, red_trees=reduced_trees[total_len - 1 - i],
									  current_trees=current_trees_embedded)
				self.no_embedded_trees = len(current_trees_embedded)
			else:
				for pair in reversed(seq):
					self.add_pair(*pair)
		# if a newick string is given, build the network from the newick string
		elif newick:
			self.newick = newick
			network, self.leaves, self.labels = self.newick_to_network(newick)
			self.nw = network
			self.no_nodes = len(list(self.nw))
			self.compute_leaf_nodes()
		# if a network 'best_tree_from_network' is given, extract the best tree from this network and use this tree
		# as the network
		elif best_tree_from_network:
			self.nw.add_edges_from(best_tree_from_network.Best_Tree())
			self.labels = best_tree_from_network.labels
			self.leaf_nodes = best_tree_from_network.leaf_nodes
			self.leaves = best_tree_from_network.leaves
			self.no_nodes = best_tree_from_network.no_nodes
			# self.Clean_Up()

		elif net:
			self.nw = net
			self.leaves = get_leaves(self.nw)
			self.labels = {l: l for l in self.leaves}

	def is_cherry(self, x, y):
		if (x not in self.leaves) or (y not in self.leaves):
			return False
		px = -1
		py = -1
		for p in self.nw.predecessors(x):
			px = p
		for p in self.nw.predecessors(y):
			py = p
		return px == py

	# Returns true if (x_label,y_label) forms a reticulate cherry in the network, false otherwise
	def is_ret_cherry(self, x_label, y_label):
		if not x_label in self.leaves or not x_label in self.leaves:
			return False
		x = self.labels[x_label]
		y = self.labels[y_label]
		px = -1
		py = -1
		for p in self.nw.predecessors(x):
			px = p
		for p in self.nw.predecessors(y):
			py = p
		return (self.nw.in_degree(px) > 1) and self.nw.out_degree(px) == 1 and (py in self.nw.predecessors(px))

	# Returns the leaf nodes of the network
	def compute_leaf_nodes(self):
		self.leaf_nodes = dict()
		for v in self.labels:
			self.leaf_nodes[self.labels[v]] = v

	def reticulations_non_binary(self):
		return [self.nw.in_degree(v)-1 for v in self.nw.nodes() if self.nw.in_degree(v) >= 2]

	def ret_number(self):
		return np.sum(self.reticulations_non_binary())
	
	def num_trees_contained(self, trees):
		trees_contained = 0
		for tree in trees.values():
			try:
				res = BOTCH.tc_brute_force(deepcopy(tree.nw), deepcopy(self.nw))
				trees_contained += int(res)
			except Exception as e:
				pass
		return trees_contained

	# Adds a pair to the network, using the construction from a cherry-picking sequence
	# returns false if y is not yet in the network and the network is not empty
	def add_pair(self, x, y, red_trees=set(), current_trees=set(), height=[1, 1]):
		# if the network is empty, create a cherry (x,y)
		if len(self.leaves) == 0:
			self.nw.add_edge(0, 1, no_of_trees=len(red_trees), length=0)
			self.nw.add_edge(1, 2, no_of_trees=len(red_trees), length=height[0])
			self.nw.add_edge(1, 3, no_of_trees=len(red_trees), length=height[1])
			self.leaves = {x, y}
			self.labels[x] = 2
			self.labels[y] = 3
			self.leaf_nodes[2] = x
			self.leaf_nodes[3] = y
			self.no_nodes = 4
			return True
		# if y is not in the network return false, as there is no way to add the pair and get a phylogenetic network
		if y not in self.leaves:
			return False
		# add the pair to the existing network
		node_y = self.labels[y]
		parent_node_y = -1
		for p in self.nw.predecessors(node_y):
			parent_node_y = p

		# first add all edges around y
		length_incoming_y = self.nw[parent_node_y][node_y]['length']
		no_of_trees_incoming_y = self.nw[parent_node_y][node_y]['no_of_trees']
		height_goal_x = height[0]
		if height[1] < length_incoming_y:
			height_pair_y_real = height[1]
		else:
			height_pair_y_real = length_incoming_y
			height_goal_x += height[1] - height_pair_y_real

		self.nw.add_edge(node_y, self.no_nodes, no_of_trees=no_of_trees_incoming_y + len(red_trees - current_trees),
						 length=height_pair_y_real)
		self.nw[parent_node_y][node_y]['length'] = length_incoming_y - height_pair_y_real
		self.leaf_nodes.pop(self.labels[y], False)
		self.labels[y] = self.no_nodes
		self.leaf_nodes[self.no_nodes] = y

		# Now also add edges around x
		# x is not yet in the network, so make a cherry (x,y)
		if x not in self.leaves:
			self.nw.add_edge(node_y, self.no_nodes + 1, no_of_trees=len(red_trees), length=height_goal_x)
			self.leaves.add(x)
			self.labels[x] = self.no_nodes + 1
			self.leaf_nodes[self.no_nodes + 1] = x
			self.no_nodes += 2
		# x is already in the network, so create a reticulate cherry (x,y)
		else:
			node_x = self.labels[x]
			for parent in self.nw.predecessors(node_x):
				px = parent
			length_incoming_x = self.nw[px][node_x]['length']
			no_of_trees_incoming_x = self.nw[px][node_x]['no_of_trees']
			# if x is below a reticulation, and the height of the new pair is above the height of this reticulation,
			# add the new hybrid arc to the existing reticulation
			if self.nw.in_degree(px) > 1 and length_incoming_x <= height_goal_x:
				self.nw.add_edge(node_y, px, no_of_trees=len(red_trees), length=height_goal_x - length_incoming_x)
				self.nw[px][node_x]['no_of_trees'] += len(red_trees)
				self.no_nodes += 1
			# create a new reticulation vertex above x to attach the hybrid arc to
			else:
				height_pair_x = min(height_goal_x, length_incoming_x)
				self.nw.add_edge(node_y, node_x, no_of_trees=len(red_trees), length=height_goal_x - height_pair_x)
				self.nw.add_edge(node_x, self.no_nodes + 1, no_of_trees=no_of_trees_incoming_x + len(red_trees),
								 length=height_pair_x)
				self.nw[px][node_x]['length'] = length_incoming_x - height_pair_x
				self.leaf_nodes.pop(self.labels[x], False)
				self.labels[x] = self.no_nodes + 1
				self.leaf_nodes[self.no_nodes + 1] = x
				self.no_nodes += 2
		return True

	# suppresses v if it is a degree-2 node
	def Clean_Node(self, v):
		if self.nw.out_degree(v) == 1 and self.nw.in_degree(v) == 1:
			pv = -1
			for p in self.nw.predecessors(v):
				pv = p
			cv = -1
			for c in self.nw.successors(v):
				cv = c
			self.nw.add_edges_from([(pv, cv, self.nw[v][cv])])
			if self.nw[pv][v]['length'] and self.nw[v][cv]['length']:
				self.nw[pv][cv]['length'] = self.nw[pv][v]['length'] + self.nw[v][cv]['length']
			self.nw.remove_node(v)
			return True
		return False

	def Suppresss_Node(self, v):
		if self.nw.out_degree(v) == 1 and self.nw.in_degree(v) == 1:
			pv = -1
			for p in self.nw.predecessors(v):
				pv = p
			cv = -1
			for c in self.nw.successors(v):
				cv = c
			self.nw.add_edges_from([(pv, cv, self.nw[v][cv])])
			#if self.nw[pv][v]['length'] and self.nw[v][cv]['length']:
			if 'length' in self.nw[pv][v] and 'length' in self.nw[v][cv]:

				self.nw[pv][cv]['length'] = self.nw[pv][v]['length'] + self.nw[v][cv]['length']
			return True
		return False

	def Suppresss_Node_GivenNet(self, net, v):
		if net.out_degree(v) == 1 and net.in_degree(v) == 1:
			pv = -1
			for p in net.predecessors(v):
				pv = p
			cv = -1
			for c in net.successors(v):
				cv = c
			net.add_edges_from([(pv, cv, net[v][cv])])
			if 'length' in net[pv][v] and 'length' in net[v][cv]:

				net[pv][cv]['length'] = net[pv][v]['length'] + net[v][cv]['length']
			return True
		return False

	# reduces the pair (x_label,y_label) if it is reducible in the network
	# returns a new set reducible pairs that involve the leaves x_label and y_label
	def reduce_pair(self, x_label, y_label):
		if x_label not in self.leaves or not y_label in self.leaves:
			return set()
		x = self.labels[x_label]
		y = self.labels[y_label]
		px = -1
		py = -1
		for p in self.nw.predecessors(x):
			px = p
		for p in self.nw.predecessors(y):
			py = p
		if self.is_cherry(x_label, y_label):
			self.reducible_pairs.difference_update({(x_label, y_label), (y_label, x_label)})
			self.nw.remove_node(x)
			self.leaves.remove(x_label)
			self.labels.pop(x_label, False)
			self.Clean_Node(py)
			# AddCherriesInvolving y
			new_pairs = {("no_leaf", "no_leaf")} | self.Find_Pairs_With_First(y_label) | self.Find_Pairs_With_Second(
				y_label)
			self.reducible_pairs = self.reducible_pairs.union(new_pairs - {("no_leaf", "no_leaf")})
			return new_pairs
		if self.is_ret_cherry(x_label, y_label):
			# print(f"{(x_label, y_label)} is a reticulated cherry")
			self.reducible_pairs.difference_update({(x_label, y_label), (y_label, x_label)})
			self.nw.remove_edge(py, px)
			self.Clean_Node(px)
			self.Clean_Node(py)
			# AddCherriesInvolving x and y
			new_pairs = {("no_leaf", "no_leaf")} | self.Find_Pairs_With_First(x_label) | self.Find_Pairs_With_Second(
				x_label) | self.Find_Pairs_With_First(y_label) | self.Find_Pairs_With_Second(y_label)
			self.reducible_pairs = self.reducible_pairs.union(new_pairs - {("no_leaf", "no_leaf")})
			return new_pairs
		return set()

	# Returns all reducible pairs in the network where x_label is the first element of the pair
	def Find_Pairs_With_First(self, x_label):
		pairs = set()
		x = self.labels[x_label]
		px = -1
		for p in self.nw.predecessors(x):
			px = p
		if self.nw.in_degree(px) > 1:
			for ppx in self.nw.predecessors(px):
				for cppx in self.nw.successors(ppx):
					if cppx in self.leaf_nodes:
						pairs.add((x_label, self.leaf_nodes[cppx]))
		if self.nw.out_degree(px) > 1:
			for cpx in self.nw.successors(px):
				if cpx in self.leaf_nodes:
					pairs.add((x_label, self.leaf_nodes[cpx]))
		return pairs - {(x_label, x_label)}

	# Returns all reducible pairs in the network where x_label is the second element of the pair
	def Find_Pairs_With_Second(self, x_label):
		pairs = set()
		x = self.labels[x_label]
		px = -1
		for p in self.nw.predecessors(x):
			px = p
		if self.nw.out_degree(px) > 1:
			for cpx in self.nw.successors(px):
				if cpx in self.leaf_nodes:
					pairs.add((self.leaf_nodes[cpx], x_label))
				if self.nw.in_degree(cpx) > 1:
					for ccpx in self.nw.successors(cpx):
						if ccpx in self.leaf_nodes:
							pairs.add((self.leaf_nodes[ccpx], x_label))
		return pairs - {(x_label, x_label)}

	def Check_Is_Suppressible(self, v):
		#print( self.nw.out_degree(v),  self.nw.in_degree(v))
		if self.nw.out_degree(v) == 1 and self.nw.in_degree(v) == 1:
			return True
		return False

	def Check_Is_Suppressible_GivenNet(self, net, v):
		#print( self.nw.out_degree(v),  self.nw.in_degree(v))
		if net.out_degree(v) == 1 and net.in_degree(v) == 1:
			return True
		return False


	def get_legal_moves(self):
		legal_moves = []
		n = len(self.nw.nodes)
		for e1 in self.nw.edges:
			for e2 in self.nw.edges:
				u,v = e1
				x,y = e2		
				if self.check_if_move_is_legal(u, v, x, y):
					legal_moves.append((u,v,x,y))
		print(len(self.nw.edges), len(legal_moves))
		return legal_moves

	def check_if_move_is_legal(self, u,v, s, t):
		net = deepcopy(self.nw)
		#print(u,v,s,t)
		if len(set([u,v,s,t])) != 4:
			return False
		if (u,v) not in net.edges or  (s,t) not in net.edges:
			return False
		#print('passed1')
		try:
			net.remove_edge(u, v) #delete (u,v)
			if not self.Check_Is_Suppressible_GivenNet(net, u):
				return False
		except Exception as e:
			print(e)
			return False
		
		try:
			supress = self.Suppresss_Node_GivenNet(net, u) #suppress u
			net.add_edge(s, u) #adding u' on the edge (s,t)
			net.add_edge(u, t) #adding u' on the edge (s,t)
			net.add_edge(u, v) #adding u',v
		except Exception as e:
			print(e)
			return False

		return True


	def tail_move(self, u,v, s,t):
		if len(set([u,v,s,t])) != 4:
			return False

		check1 = (u,v) in self.nw.edges # (u,v) is an edge
		check2 = (s,t) in self.nw.edges # (s,t) is an edge
		
		if not (check1 and check2):
			return False

		self.nw.remove_edge(u, v) #delete (u,v)
		check3 = self.Check_Is_Suppressible(u)        
		
		if not check3:
			return False

		self.Suppresss_Node(u) #suppress u
		self.nw.add_edge(s, u) #adding u' on the edge (s,t)
		self.nw.add_edge(u, t) #adding u' on the edge (s,t)
		self.nw.add_edge(u, v) #adding u',v
		
		return True