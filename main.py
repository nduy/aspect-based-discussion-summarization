#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    MAIN SUMMARIZATION BASELINE
"""
from functions import *
from decoration import *
import datetime
import pickle

# ------ Time recording
import time
start_time = time.time()
#############################

build_options = {
    'build_mode': 1,
    'sentiment_ana_mode': 'global',  # 'global', 'local'
    'use_thread_structure': False   # if yes, the thread structure of comments will be used. Otherwise just treat them
                                    # equally
}

prune_options = {
    'enable_pruning': True,
    'min_word_length': 2,
    'remove_isolated_node': True,
    'node_freq_min': 1,
    'edge_freq_min': 1,
    'node_degree_min': 2
}

uni_options = {
    'unify_matched_keywords': {
        'enable': True,
        'intra_cluster_unify': True,
        'inter_cluster_unify': True,
        'unification_mode': 'contract'  # modes: link, contract
    }
}

dep_opt = {
    # preferred_pos': ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'preferred_pos': ['NN', 'NNS', 'NNP', 'NNPS'],
    'preferred_rel': 'all',  # ['nsubk','nsubkpass','obj','iobj'] list of relation to remains
    'compound_merge': True

}

# Verbality: to print or not to print ################################################################################
script_verbality = 2     # 0: silent, 1: print main info, 2: print some techs info, 3. print debugging info


if __name__ == "__main__":
    # comments = read_comment_file("data/comments_article0_clipped.txt", read_as_threads=False)
    # title, article = read_article_file("data/article0_clipped.txt")
    comments = read_comment_file("data/comments_article0.txt", read_as_threads=False)
    title, article = read_article_file("data/article0.txt")

    # g = build_directed_graph_from_text(txt=title.lower(), threadid='title')
    # print 'Nodes:', g.nodes(data=True), '\n Edges:', g.edges(data=True)

    dataset = {'title': title,
               'article': article,
               'comments': comments}
    #  print title
    # print article
    maybe_print("Loaded data set! Number of conversation thread: {0}".format(len(dataset['title'])), 0)

    # Build aspect graph, then serialize
    asp_graph = build_sum_graph(dataset)  # Build sum keygraph at mode 1
    with open('tmp/asp_graph.adjlist', 'wb+') as handle:
        # pickle.dump(asp_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        nx.write_adjlist(asp_graph, handle)
    # asp_graph = nx.read_adjlist('tmp/asp_graph.adjlist', create_using=nx.DiGraph())

    # Prune the graph, then serialize
    pruned_graph = prune_graph(asp_graph, prune_options)
    with open('tmp/pruned_graph.adjlist', 'wb+') as handle:
        # pickle.dump(pruned_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        nx.write_adjlist(pruned_graph, handle)
    # pruned_graph = nx.read_adjlist('tmp/pruned_graph.adjlist', create_using=nx.DiGraph())

    # Compute sentiment scores, then serialize
    sen_graph = compute_sentiment_score(pruned_graph)
    with open('tmp/sen_graph.adjlist', 'wb+') as handle:
        # pickle.dump(sen_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        nx.write_adjlist(sen_graph, handle)
    # sen_graph = nx.read_adjlist('tmp/sen_graph.adjlist', create_using=nx.DiGraph())

    # Coloring the graph by sentiment, then serialize
    colored_graph = coloring_nodes(sen_graph)
    with open('tmp/colored_graph.adjlist', 'wb+') as handle:
        # pickle.dump(colored_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        nx.write_adjlist(colored_graph, handle)
    # colored_graph = nx.read_adjlist('tmp/colored_graph.adjlist', create_using=nx.DiGraph())

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

    print("Execution time:  %s seconds" % (time.time() - start_time))
