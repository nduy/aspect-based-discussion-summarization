#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Sep 18, 2017
    PERFORM COMMUNITY DETECTION FROM A GRAPH
"""

import networkx as nx
from networkx.algorithms import community
from utils import *

sample_community_names = [u'α',u'β',u'γ',u'δ',u'ε',u'ζ',u'η',u'θ',u'ι',u'κ',u'λ',u'μ',
                           u'ν',u'ξ',u'ο',u'π',u'ρ',u'σ',u'τ',u'υ',u'φ',u'χ',u'ψ',u'ω']


def detect_communities(g=None, comm_opt=None):
    maybe_print("Detecting communities.", 2, 'i')
    ENABLE_DETECTION = False
    ALGORITHM = 'fluid'
    graph = g
    if not graph:
        maybe_print("   Can't detect community because the graph is undefined (value is None).\n "
                    "      Trying to load from tmp/pruned_graph.gpickle",1,'E')
        try:
            graph = nx.read_gpickle("tmp/pruned_graph.gpickle")
        except Exception:
            raise RuntimeError("Unable to detect communities. Invalid input graph.")

    if not comm_opt:
        raise ValueError("Invalid community detection options.")
    else:
        ENABLE_DETECTION = comm_opt['enable_community_detection'] if 'enable_community_detection' in comm_opt else False
        ALGORITHM = comm_opt['method']['algorithm'] if 'algorithm' in comm_opt else 'fluid_communities'
    # Convert it to undirected graph
    undir_graph = graph.to_undirected()

    if not undir_graph:
        raise ValueError("Unable to perform community detection! Perhaps due to the malformed graph.")
    if ENABLE_DETECTION:
        try:
            if ALGORITHM == "fluid_communities":
                # get the largest messy graph
                # Get number of communities to be detected
                n_com = comm_opt['method']['n_communities'] if 'n_communities' in comm_opt['method'] else 4
                gc = max(nx.connected_component_subgraphs(undir_graph), key=len)
                # list of list. Each sublist contain ID of nodes in the same community
                communities = list(community.asyn_fluidc(gc, n_com))
                com_index = -1
                for com in communities:
                    com_index += 1
                    for node_id in com:
                        graph.node[node_id]['cluster_id'] = sample_community_names[com_index]
                return graph
        except Exception as inst:
            maybe_print(" Error while running algorithm {0} to detect communities. Error name: {1}. "
                        "Perhaps incorrect algorithm name of parameters. Community detection is skipped and community label"
                        " for all nodes is set to be \'unknown\'.".format(ALGORITHM,inst), 2, 'E')
            return g
    else:
        return g
'''
import argparse

graph = None

parser = argparse.ArgumentParser(description='Load a file, perform community detection.')
parser.add_argument('-f', action='store', dest='graph', help='Input graph')

if __name__ == "__main__":
    print parser.parse_args()


pruned_graph = nx.read_gpickle("tmp/pruned_graph.gpickle")
dir_graph = pruned_graph.to_undirected()

############## girvan_newman method
communities_generator = community.girvan_newman(dir_graph)
communities_generator = community.girvan_newman(Gc)
top_level_communities = next(communities_generator)

next_level_communities = next(communities_generator)
sorted(map(sorted, next_level_communities))


############## Kernighan–Lin algorithm
section1,section2 = community.kernighan_lin_bisection(dir_graph)
section1,section2 = community.kernighan_lin_bisection(Gc)
subgraph1 = dir_graph.subgraph(list(section1))
subgraph2 = dir_graph.subgraph(list(section2))
community.kernighan_lin_bisection(subgraph1)
community.kernighan_lin_bisection(subgraph2)

#############33 Fluid Communities
Gc = max(nx.connected_component_subgraphs(dir_graph), key=len)
list(community.asyn_fluidc(Gc,6))
list(community.asyn_fluidc(Gc,4))

############   K-Clique
list(community.k_clique_communities(Gc, 6))
'''
