#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
'''
    Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    MAIN SUMMARIZATION BASELINE
'''

options = {
    'enable_pruning': True,
    'min_word_length': 2,
    'remove_isolated_node': True,
    'unify_matched_keywords': {
        'enable': True,
        'intra_cluster_unify': True,
        'inter_cluster_unify': False,
        'unification_mode': 'contract'  # modes: link, contract
    },
    'node_freq_min': 1,
    'edge_freq_min': 1,
    'node_degree_min': 2
}
from functions import *

if __name__ == "__main__":
    dataset = read_comment_file("data/comments_article0.txt");
    maybe_print("Loaded data set! Number of conversation thread: {0}".format(len(dataset)), 0)
    asp_graph = build_sum_graph(0,dataset) # Build sum keyraph at mode 0
    #print asp_graph.edges()
    # pruning
    pruned_graph = prun_graph(asp_graph, options)
    #print pruned_graph.edges()
    if pruned_graph:
        json_g = generate_json_from_graph(pruned_graph)
    with open('dump.json', 'w') as outfile:
        json.dump(json_g, outfile, sort_keys=True, indent=4, separators=(',', ': '))


    #print json.dumps(json_g, sort_keys=True, indent=4, separators=(',', ': '))
