from Features import Features
from CPH import *
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import copy
import sys
import os
sys.path.append(os.path.abspath('../CPH'))

from NetworkGen.NetworkToTree import is_cherry

def find_cherry(net, x):
	cherries = []
	for p in net.pred[x]:
		if net.in_degree(p) == 1:
			for pc in net.succ[p]:
				if pc == x:
					continue
				t = net.out_degree(pc)
				if t == 0:
					cherries.append((pc, x))
				elif t == 1:
					for pcc in net.succ[pc]:
						if net.out_degree(pcc) == 0:
							cherries.append((pcc, x))

	#print('cherries', cherries)
	return cherries

def find_ret_cherry(net, x):
	ret_cherries = []
	for p in net.pred[x]:
		if net.out_degree(p) == 1:
			for pp in net.pred[p]:
				for ppc in net.succ[pp]:
					if ppc == p:
						continue
					if net.out_degree(ppc) == 0:
						ret_cherries.append((x, ppc))

	return ret_cherries

def check_cherry(net, x,y):
	print(net.pred[x], net.pred[x]._atlas)
	leaves = [v for v, d in net.out_degree() if d == 0]
	#print(leaves)
	if x in net.nodes and y in net.nodes:
		for px in net.pred[x]:
			for py in net.pred[y]:
				if px == py:
					return 1
				if net.out_degree[px] == 1:
					if px in net.succ[py]:
						return 2
	return False

def reduce_pair(net, x,y):
	k = check_cherry(net, x,y)
	if k == 1:
		for px in net.pred[x]:
			net.remove_node(x)
			for ppx in net.pred[px]:
				net.remove_node(px)
				net.add_edge(ppx, y)
			return True
	if k == 2:
		for px in net.pred[x]:
			for py in net.pred[y]:
				net.remove_edge(py, px)
				if net.in_degree[px] == 1:
					for ppx in net.pred[px]:
						net.add_edge(ppx,x)
						net.remove_node(px)
				for ppy in net.pred[py]:
					net.add_edge(ppy,y)
					net.remove_node(py)
				return True
	return False

def find_tcs_old(net):
	seq_todo = []
	for x in net.nodes:
		#print(x,net.out_degree[x])
		if net.out_degree(x) == 0:
			cherry = find_cherry(net, x)
			seq_todo += cherry
	seq_tcs = []
	while seq_todo:
		cherry = seq_todo.pop()
		k = check_cherry(net, cherry[0], cherry[1])
		if k == 1:
			seq_tcs.append(cherry)
			reduce_pair(net, cherry[0], cherry[1])
			if (cherry[1], cherry[0]) in seq_todo:
				seq_todo.remove((cherry[1], cherry[0]))
			seq_todo += find_cherry(net, cherry[1])
			seq_todo += find_ret_cherry(net, cherry[1])
		if k ==2:
			seq_tcs.append(cherry)
			reduce_pair(net, cherry[0], cherry[1])
			seq_todo += find_ret_cherry(net, cherry[0])
			seq_todo += find_cherry(net, cherry[1])
			seq_todo += find_ret_cherry(net, cherry[1])

	return seq_tcs

def find_tcs(net):
	seq_todo = []
	for x in net.nodes:
		#print(x,net.out_degree[x])
		if net.out_degree(x) == 0:
			cherry = find_cherry(net, x)
			seq_todo += deepcopy(cherry)
	seq_tcs = []
	#print(seq_todo)
	while seq_todo:
		cherry = seq_todo.pop()
		#print('cherry',cherry)
		k = check_cherry(net, cherry[0], cherry[1])
		if k == 1 or k == 2:
			reduce_pair(net, cherry[0], cherry[1])
			seq_tcs.append(cherry)
			seq_todo += find_cherry(net, cherry[1])
			seq_todo += find_ret_cherry(net, cherry[1])
		
	return seq_tcs

def cps_reduces_network(net, seq):
	print('cps reduces network')
	print('net',net)
	for cherry in seq:
		#print(cherry)
		reduce_pair(net, cherry[0], cherry[1])
	print('after reduction',net, net.edges)
	if len(net.edges) == 1:
		return True
	return False

def network_containtment(net, tree):
	print('start network containtment')
	print('net:',net)
	print('tree:',tree)
	tcs_seq = find_tcs_old(deepcopy(net))
	print('tcs seq',tcs_seq)
	cps_reduces_network(deepcopy(net), tcs_seq)

	result =  cps_reduces_network(deepcopy(tree), tcs_seq)
	print('result:',result)


#Algorithm 1
def FindRP2nd(N, x):
	lst = list()
	for p in N.predecessors(x):
		if N.in_degree(p) == 1:
			for cp in N.successors(p):
				if cp != x:
					t = N.out_degree(cp)
					if t == 0:
						lst.append((cp, x))
					if t == 1:
						for ccp in N.successors(cp):
							if N.out_degree(ccp) == 0:
								lst.append((ccp,x))
	return lst

#algorithm 2
def FindRP1st(N, x):
	lst = list()
	for p in N.predecessors(x):
		if N.out_degree(p) == 1:
			for g in N.predecessors(p):
				for cg in N.successors(g):
					if cg != p:
						if N.out_degree(cg) == 0:
							lst.append((x, cg))
	return lst


#Checks if two nodes form a cherry (1) or reticulated cherry (2), returns False otherwise
#Not in the paper
def CheckCherry(N, x, y):
	if N.has_node(x) and N.has_node(y):
		px = None
		py = None
		for parent in N.predecessors(x):
			px = parent
		for parent in N.predecessors(y):
			py = parent
		if px == py:
			return 1
		if N.out_degree(px) == 1 and px in N.successors(py):
			return 2
	return False


#Algorithm 3
def ReducePair(N, x, y):
	k = CheckCherry(N, x, y)
	if k == 1:
		for px in N.predecessors(x):
			N.remove_node(x)
			for ppx in N.predecessors(px):
				N.remove_node(px)
				N.add_edge(ppx,y)
			return True
	if k == 2:
		for px in N.predecessors(x):
			for py in N.predecessors(y):
				N.remove_edge(py,px)
				if N.in_degree(px) == 1:
					for ppx in N.predecessors(px):
						N.add_edge(ppx, x)
						N.remove_node(px)
				#if N.out_degree(py) == 1:
				for ppy in N.predecessors(py):
					N.add_edge(ppy, y)
					N.remove_node(py)
				return True
	return False


#Algorithm 4
def FindTCS(N):
	lst1 = list()
	for x in N.nodes():
		if N.out_degree(x) == 0:
			cherry1 = FindRP2nd(N,x)
			lst1.extend(cherry1)
	lst2 = list()
	while lst1:
		cherry = lst1.pop()
		k = CheckCherry(N, *cherry)
		if (k == 1) or (k == 2):
			ReducePair(N, *cherry)
			lst2.append(cherry)
			lst1.extend(FindRP2nd(N,cherry[1]))
			lst1.extend(FindRP1st(N,cherry[1]))
	return lst2


#Algorithm 5
def CPSReducesNetwork(N, lst):
	for cherry in lst:
		ReducePair(N, *cherry)
	print('afte rreduction:',N)
	if N.size() == 1:
		return True
	return False


#Algorithm 6
def TCNContains(N, M):
	print('start network containtment')
	print('net:',N)
	print('tree:',M)
	seq = FindTCS(N)
	print('seq:',seq)
	result =  CPSReducesNetwork(M,FindTCS(N))
	print('result:',result)

