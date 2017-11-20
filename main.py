#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    MAIN SUMMARIZATION BASELINE
"""

from functions import *
from decoration import *
from datetime import timedelta as td
from datetime import datetime as dt
from config import *
from community_detect import *
import AGmodel
import argparse
from nltk.corpus import wordnet as wn

# ------ Time recording
import time

build_scenario = 'unknown'  # three scenario: "article only", "comment only" and "combine"

def add_args(parser):
    """Command-line arguments to extract a summarization dataset from the
    NYT Annotated Corpus.
    """
    # Paths to training data
    # for building AG
    parser.add_argument('--art_path', action='store',
                        help='path to the ARTICLE file',
                        default=None)
    parser.add_argument('--com_path', action='store',
                        help='path to the COMMENT file',
                        default=None)
    # Path to JSON graph visualization file
    parser.add_argument('--json_path', action='store',
                        help='path to store JSON file',
                        default='./art23.json')
    parser.add_argument('--evaluate_conversation', action='store_true',
                        help='Enable evaluation for CONVERSATION. Test set will be picked up dirrectly from data. '
                             'Split the dataset to training and testing dataset, You need to specify path to the gold '
                             'standard as well',
                        default=False)
    parser.add_argument('--truth_conversation_path', action='store',
                        help='path to the ground truth file for CONVERSATION. It comprise 2 columns: first is the id, s'
                        'second is the truth label.',
                        default=None)
    parser.add_argument('--centrality', action='store',
                        choices=('eigenvector', 'pagerank', 'degree', 'closeness'),
                        help='Centrality measure method for building probabilistic model',
                        default='eigenvector')
    parser.add_argument('--normalization', action='store',
                        choices=('sum', 'softmax'),
                        help='Normaization method apply to centrality score, in order to sum to 1 (probability)',
                        default='sum')
    # Filters for NYT corpus based on descriptors and summary type
    '''
    parser.add_argument('--summary_type', action='store',
                        choices=('abstract', 'lead', 'online_lead'),
                        help='type of NYT summary to consider',
                        default='online_lead')
    parser.add_argument('--descriptors', action='store', nargs='+',
                        help='topics of docs to extract',
                        default=None)
    parser.add_argument('--descriptor_types', action='store', nargs='+',
                        choices=('indexing', 'online', 'online_general',
                                 'taxonomic', 'type'),
                        help='topic categories considered for --descriptors',
                        default=('online_general',))
    parser.add_argument('--exclude', action='store_true',
                        help='drop docs with --descriptors',
                        default=False)

    # Filters for extracted docs based on the summary size
    parser.add_argument('--limit', action='store', type=int,
                        help='number of docs to consider',
                        default=None)
    parser.add_argument('--cost_type', action='store',
                        choices=('char', 'word', 'sent'),
                        help='type of cost per unit', default='char')
    parser.add_argument('--min_ref_cost', action='store', type=int,
                        help='minimum cost of a reference summary',
                        default=1)
    parser.add_argument('--max_ref_cost', action='store', type=int,
                        help='maximum cost of a reference summary',
                        default=int(1e9))
    parser.add_argument('--min_ref_sents', action='store', type=int,
                        help='minimum number of reference summary sentences',
                        default=1)
    parser.add_argument('--max_ref_sents', action='store', type=int,
                        help='maximum number of reference summary sentences',
                        default=int(1e9))

    # Mutually-exclusive filters for extractive and near-extractive summaries.
    # Multiple filters are treated as disjunctive, i.e., combinations of
    # the single-filter datasets
    parser.add_argument('--extractive', action='store_true',
                        help='every summary sentence comes from the doc')
    parser.add_argument('--semi_extractive', action='store_true',
                        help='one or more summary sentence is a contiguous '
                             'substring in a doc sentence; rest extractive')
    parser.add_argument('--sub_extractive', action='store_true',
                        help='one or more summary sentence is a '
                             'non-contiguous subsequence in a doc sentence; '
                             'rest semi-extractive')

    # Dataset partitioning by date
    parser.add_argument('--partition', action='store',
                        choices=('train', 'dev', 'test'),
                        help='dataset partition to extract',
                        default=None)
    parser.add_argument('--id_split', action='store', nargs=2,
                        help='any two prefixes of YYYY/MM/DD/DOCID to divide '
                             'the train/dev/test partition',
                        default=['2005/', '2006/'])
    '''


start_time = time.time()
if __name__ == "__main__":
    #  ################# Init #####################
    # print wn.__class__            # <class 'nltk.corpus.util.LazyCorpusLoader'>
    wn.ensure_loaded()            # first access to wn transforms it
    # print wn.__class__
    parser = argparse.ArgumentParser(description="Run Aspect graph construction and model building")
    add_args(parser)
    args = parser.parse_args()
    # Load (or create) a cached corpus of NYT docs
    print(args)

    #  ################# PRE-BUILD #####################
    if args.art_path and args.com_path is None:  # Article only mode
        build_scenario = "article_only"
        title, article = read_article_file(args.art_path)
        comments, comment_js, comment_des = None, None, None

    elif args.art_path is None and args.com_path:  # Article only mode
        build_scenario = "comment_only"
        comments, comment_js, comment_des = read_comment_file(args.com_path,
                                                              read_as_threads=False)
        title = ""
        article = ""
    elif args.com_path and args.art_path:
        build_scenario = "combine"
        title, article = read_article_file(args.art_path)
        comments, comment_js, comment_des = read_comment_file(args.com_path,
                                                              read_as_threads=False)
    else:
        raise ValueError("Please specify article and/or comment path. The build mode will be decided automatically.")
    # evaluation for conversation -> spit the data, then write them to three separate files for further processing
    if args.evaluate_conversation and args.truth_conversation_path:
        if article:
            raise UserWarning("You are running evaluation on conversation, but article was appointed. Thus, the "
                              "program will run in combined mode. Perhaps you want to run it in comment only mode.")
        # load truth file
        all_ground_truth = [line.rstrip('\n').split('\t') for line in open(args.truth_conversation_path)]
        # now do the splitting
        assert len(comments) == len(all_ground_truth), "Number of comments {0} an {1} ground truth is mismatched!"\
            .format(len(comments), len(all_ground_truth))
        n_comment = int(len(comments)*3/10)
        test_ids_f = open('./tmp/test_ids.txt', 'w+')
        test_text_f = open('./tmp/test_text.txt', 'w+')
        test_truth_f = open('./tmp/test_truth.txt', 'w+')
        for i in xrange(int(len(all_ground_truth)*0.7),len(all_ground_truth)):
            assert str(comments[i]['no']) == str(all_ground_truth[i][0]), 'Mismatch id between ground truth and comment, ' \
                                                                'cmn num {0}'.format(i)
            test_ids_f.write('{0}\n'.format(all_ground_truth[i][0]))
            test_text_f.write('{0}\n'.format(comments[i]['content']))
            test_truth_f.write('{0}\n'.format(all_ground_truth[i][1]))
        test_ids_f.close()
        test_text_f.close()
        test_truth_f.close()
        dataset = {'title': title,
                   'article': article,
                   'comments': comments[:int(len(all_ground_truth)*0.7)]} # get top 70% as training file
    else:
        dataset = {'title': title,
                   'article': article,
                   'comments': comments}
    config.model_build_options['centrality_method'] = args.centrality
    config.model_build_options['normalization_method'] = args.normalization

    maybe_print("Loaded data set! Data mode: {0}".format(build_scenario), 0)
    if build_scenario == 'comment_only' or build_scenario == 'combine':
        maybe_print("Loaded data set! Number of comments: {0}".format(len(dataset['comments'])), 0)

    #  ################### BUILD #######################
    # Build aspect graph, then serialize
    asp_graph = build_sum_graph(dataset)  # Build sum keygraph at mode 1
    nx.write_gpickle(asp_graph, "tmp/asp_graph.gpickle")
    print("[i] Raw graph build completed. Elapsed time:  {0} seconds"
          .format(str(td(seconds=(time.time() - start_time)))))
    del dataset

    # Compute sentiment scores, then serialize
    sen_graph = compute_sentiment_score(asp_graph)
    nx.write_gpickle(sen_graph, "tmp/sen_graph.gpickle")
    del asp_graph
    print("[i] Sentiment computation completed. Elapsed time:  {0} seconds"
          .format(str(td(seconds=(time.time() - start_time)))))

    # Coloring the graph by sentiment, then serialize
    colored_graph = coloring_nodes(sen_graph)
    nx.write_gpickle(sen_graph, "tmp/colored_graph.gpickle")
    del sen_graph
    print("[i] Coloring graph completed. "
          "Elapsed time:  {0} seconds".format(str(td(seconds=(time.time() - start_time)))))

    # Prune the graph, then serialize
    # colored_graph = nx.read_gpickle("tmp/colored_graph.gpickle")
    pruned_graph = prune_graph(colored_graph)
    nx.write_gpickle(pruned_graph, "tmp/pruned_graph.gpickle")
    del colored_graph
    print("[i] Graph prunning completed. Elapsed time:  {0} seconds"
          .format(str(td(seconds=(time.time() - start_time)))))
    
    # pruned_graph = nx.read_gpickle("tmp/pruned_graph.gpickle")

    # Print top 20 words by centrality
    print_top_keyphrases(g=pruned_graph, ntop=100, out_path='./tmp/top20.csv')

    # Compute sentiment scores, then serialize
    com_graph = detect_communities(pruned_graph, community_detect_options)
    nx.write_gpickle(com_graph, "tmp/com_graph.gpickle")
    del pruned_graph
    print("Community detection completed. Elapsed time:  {0} seconds"
          .format(str(td(seconds=(time.time() - start_time)))))

    #  ################### PREDICTIVE MODEL #######################
    if args.evaluate_conversation:
        # Now evaluate the model for CONVERSATION ----> Comment only!
        model = AGmodel.AGmodel(asp_graph=com_graph)  # Build the model bases on the communities
        # del com_graph
        # Load the test file:  doc_ids,  multi_docs, ground_truth
        doc_ids = [line.rstrip() for line in open('./tmp/test_ids.txt')]
        multi_docs = [line.rstrip() for line in open('./tmp/test_text.txt')]
        ground_truth = [line.rstrip() for line in open('./tmp/test_truth.txt')]
        evaluation_result = model.evaluate_model(doc_ids=doc_ids,
                                                 multi_docs=multi_docs,
                                                 ground_truth=ground_truth)
        maybe_print("Model evaluation result: {0}".format(evaluation_result),2,'i')

        '''        
        model = AGmodel.AGmodel(asp_graph=com_graph)  # Build the model bases on the communities
        del com_graph
        # Load the test file:  doc_ids,  multi_docs, ground_truth
        doc_ids = [line.rstrip('\n') for line in open('data/nyt.online_lead_top10.shelf_test_ids_lead_only.txt')]
        multi_docs = [line.rstrip('\n') for line in open('data/nyt.online_lead_top10.shelf_test_lead_only.txt')]
        ground_truth = [line.rstrip('\n') for line in open('data/nyt.online_lead_top10.shelf_test_truth_lead_only.txt')]
        evaluation_result = model.evaluate_model(doc_ids=doc_ids,
                                                 multi_docs=multi_docs,
                                                 ground_truth=ground_truth)
        maybe_print("Model evaluation result: {0}".format(evaluation_result),2,'i')
        '''

    #################### DUMP JSON FILE #######################
    json_g = None
    if com_graph.nodes():
        json_g = generate_json_from_graph(com_graph)
        # Add build options
        json_g['options'] = {
            'timestamp': dt.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
            'build_option': build_options,
            'pruning_option': prune_options,
            'unification_option': uni_options,
            'community_detection_option': community_detect_options,
            'title': 'fluid with pagerank'
        }
        json_g['summary'] = {
            'n_comments': len(comments),
            'n_nodes': nx.number_of_nodes(com_graph),
            'n_edges': nx.number_of_edges(com_graph),
            'elapsed_time_sec': str(td(seconds=(time.time() - start_time)))
        }

        # Add edges from nodes to comments
        comment_edges,comment_mean_sentiment = extract_comment_relations(com_graph)
        # print(comment_mean_sentiment)
        json_g['edges'].extend(comment_edges)
        # add comments
        json_g['comments'] = comment_js  # add comment descriptions

        # Combine comment node and its sentiment
        for i in xrange(0,len(comment_des)):
            if comment_des[i]['id'] in comment_mean_sentiment:
                comment_des[i]['sen_score'] = comment_mean_sentiment[comment_des[i]['id']]
            else:
                comment_des[i]['sen_score'] = 0.0
        # add comment nodes to graph
        json_g['nodes'].extend(comment_des)

    with open(args.json_path, 'w') as outfile:
        json.dump(json_g, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    print("Execution time:  {0} seconds".format(str(td(seconds=(time.time() - start_time)))))
