#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Sep 18, 2017
    PERFORM COMMUNITY DETECTION FROM A GRAPH
"""

import networkx as nx
from networkx.algorithms import community

pruned_graph = nx.read_gpickle("tmp/pruned_graph.gpickle")
dir_graph = pruned_graph.to_undirected()

communities_generator = community.girvan_newman(pruned_graph)

top_level_communities = next(communities_generator)

next_level_communities = next(communities_generator)