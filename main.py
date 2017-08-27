#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
'''
    Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    MAIN SUMMARIZATION BASELINE
'''

from functions import *
from decoration import *
import datetime

build_options = {
    'build_mode': 0,
    'sentiment_ana_mode': 'global'  # 'global', 'local'
}

prune_options = {
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

dep_opt = {
    'preferred_pos': ['NNP', 'NN', 'NNS', 'VB', 'VBD', 'RB'], # preferred part-of-speech tags
    'preferred_rel': 'all',  # ['nsubk','nsubkpass','obj','iobj'] list of relation to remains
    'compound_merge': True

}


if __name__ == "__main__":

    comments = read_comment_file("data/comments_article1.txt")
    title, article = read_article_file("data/article1.txt")
    g = build_directed_graph_from_text(txt=title)
    print 'Nodes:', g.nodes(data=True), '\n Edges:', g.edges(data=True)

    '''
    dataset = {'title': title,
               'article': article,
               'comments': comments}
    #  print title
    # print article
    print 'SSS', texttiling_tokenize(article)

    maybe_print("Loaded data set! Number of conversation thread: {0}".format(len(dataset['title'])), 0)

    asp_graph = build_sum_graph(0,dataset,build_options) # Build sum keyraph at mode 0
    #print asp_graph.edges()
    # pruning
    pruned_graph = prune_graph(asp_graph, prune_options)
    #print pruned_graph.edges()
    '''
    pruned_graph = g #############################
    sen_graph = compute_sentiment_score(pruned_graph)
    colored_graph = coloring_nodes(sen_graph)
    #for n in colored_graph.nodes():
    #  print colored_graph.node[n]
    json_g = None
    if colored_graph.nodes():
        json_g = generate_json_from_graph(colored_graph)
        # Add build options
        json_g['options'] = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
            'build_option': build_options,
            'pruning_option': prune_options
        }
    with open('dump.json', 'w') as outfile:
        json.dump(json_g, outfile, sort_keys=True, indent=4, separators=(',', ': '))


    #print json.dumps(json_g, sort_keys=True, indent=4, separators=(',', ': '))


