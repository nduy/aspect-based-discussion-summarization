#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    MAIN SUMMARIZATION BASELINE
"""

from functions import *
from decoration import *
from datetime import datetime as dt
from datetime import timedelta as td
from config import *
from community_detect import *
# ------ Time recording
import time

start_time = time.time()
if __name__ == "__main__":
    comments = read_comment_file("data/comments_article0.txt", read_as_threads=False)
    title, article = read_article_file("data/article0.txt")

    dataset = {'title': title,
               'article': article,
               'comments': comments}

    maybe_print("Loaded data set! Number of conversation thread: {0}".format(len(dataset['title'])), 0)

    # Build aspect graph, then serialize
    asp_graph = build_sum_graph(dataset)  # Build sum keygraph at mode 1
    nx.write_gpickle(asp_graph, "tmp/asp_graph.gpickle")
    print("[i] Elapsed time:  {0} seconds".format(str(td(seconds=(time.time() - start_time)))))

    # Compute sentiment scores, then serialize
    sen_graph = compute_sentiment_score(asp_graph)
    nx.write_gpickle(sen_graph, "tmp/sen_graph.gpickle")
    del asp_graph
    print("[i] Elapsed time:  {0} seconds".format(str(td(seconds=(time.time() - start_time)))))

    # Coloring the graph by sentiment, then serialize
    colored_graph = coloring_nodes(sen_graph)
    nx.write_gpickle(sen_graph, "tmp/colored_graph.gpickle")
    del sen_graph
    print("[i] Elapsed time:  {0} seconds".format(str(td(seconds=(time.time() - start_time)))))

    # Prune the graph, then serialize
    # colored_graph = nx.read_gpickle("tmp/colored_graph.gpickle")
    pruned_graph = prune_graph(colored_graph)
    nx.write_gpickle(pruned_graph, "tmp/pruned_graph.gpickle")
    del colored_graph
    print("[i] Elapsed time:  {0} seconds".format(str(td(seconds=(time.time() - start_time)))))
    

    # pruned_graph = nx.read_gpickle("tmp/pruned_graph.gpickle")

    # Compute sentiment scores, then serialize
    com_graph = detect_communities(pruned_graph, community_detect_options)
    nx.write_gpickle(com_graph, "tmp/com_graph.gpickle")
    del pruned_graph
    print("[i] Elapsed time:  {0} seconds".format(str(td(seconds=(time.time() - start_time)))))

    json_g = None
    if com_graph.nodes():
        json_g = generate_json_from_graph(com_graph)
        # Add build options
        json_g['options'] = {
            'timestamp': dt.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
            'build_option': build_options,
            'pruning_option': prune_options,
            'unification_option': uni_options,
            'community_detection_option': community_detect_options
        }
    with open('result.json', 'w') as outfile:
        json.dump(json_g, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    print("Execution time:  {0} seconds".format(str(td(seconds=(time.time() - start_time)))))
