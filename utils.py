from copy import deepcopy
import networkx as nx
import numpy as np

def construct_network_from_adj_matrix(matrix):
    net = nx.DiGraph()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if matrix[i,j] == 1:
                net.add_edge(i,j)
    return net

def Check_Is_Suppressible_GivenNet(net, v):
    if net.out_degree(v) == 1 and net.in_degree(v) == 1:
        return True
    return False

def Suppresss_Node_GivenNet(net, v):
    if net.out_degree(v) == 1 and net.in_degree(v) == 1:
        pv = -1
        for p in net.predecessors(v):
            pv = p
        cv = -1
        for c in net.successors(v):
            cv = c
        net.add_edges_from([(pv, cv, net[v][cv])])
        if 'length' in net[pv][v] and 'length' in net[v][cv]:

            net[pv][cv]['length'] = net[pv][v]['length'] + \
                net[v][cv]['length']
        return True
    return False

def get_legal_moves(net):
    legal_moves = []
    n = len(net.nodes)
    for e1 in net.edges:
        for e2 in net.edges:
            u, v = e1
            x, y = e2
            if check_if_move_is_legal(net, [u, v, x, y]):
                legal_moves.append((u, v, x, y))
    print(len(net.nodes), len(net.nodes) **
            4, len(net.edges), len(legal_moves))
    return legal_moves

def get_next_legal_nodes(net, current_nodes_):
    current_nodes = [x for x in current_nodes_ if x != -1]
    #print(current_nodes)
    legal_nodes_mask = [1 for x in net.nodes]
    if len(current_nodes) >= 1:
        for i, x in enumerate(net.nodes):
            picked_nodes = deepcopy(current_nodes)
            picked_nodes.append(x)
            #print(picked_nodes)
            legal_nodes_mask[i] = int(check_if_move_is_legal(net, picked_nodes))
    return np.array(legal_nodes_mask)

def check_if_move_is_legal(net_, picked_nodes):
    net = deepcopy(net_)
    if len(picked_nodes) == 0:
        return True
    if len(picked_nodes) > 4:
        return False        
    if len(set(picked_nodes)) != len(picked_nodes):
        return False
    
    in_degrees = [net.in_degree(x) for x in picked_nodes if x >= 0]    
    out_degrees = [net.out_degree(x) for x in picked_nodes if x >= 0]    
    if np.min(in_degrees) == 0:
        return False

    if len(picked_nodes) == 1:
        if net.out_degree(picked_nodes[0]) == 0:
            return False
        else:
            return True

    u,v = picked_nodes[0], picked_nodes[1]

    if (u, v) not in net.edges:
        #print('not in edges')
        return False

    try:
        net.remove_edge(u, v)  # delete (u,v)
        if not Check_Is_Suppressible_GivenNet(net, u):
            return False
    except Exception as e:
        print(e)
        return False

    if len(picked_nodes) <= 2:
        return True
    #print('3')
    if len(picked_nodes) == 3:
        if net.out_degree(picked_nodes[2]) == 0:
            return False
        else:
            return True

    s,t = picked_nodes[2], picked_nodes[3]

    if (s, t) not in net.edges:
        return False

    try:
        supress = Suppresss_Node_GivenNet(net, u)  # suppress u
        net.add_edge(s, u)  # adding u' on the edge (s,t)
        net.add_edge(u, t)  # adding u' on the edge (s,t)
        net.add_edge(u, v)  # adding u',v
    except Exception as e:
        print(e)
        return False

    return True