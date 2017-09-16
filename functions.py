#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    FUNCTIONS FOR TEXT SUMMARIZATION
"""

######################################
# IMPORT LIBRARY

import en
import codecs
import csv
import itertools
import threading
import time
import networkx as nx
from textblob import TextBlob
from nltk.corpus import stopwords
from itertools import combinations
from polyglot.text import Text
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils import *
from config import prune_options
from config import uni_options
from config import build_options
from config import dep_opt

######################################
# EXTRA DECLARATIONS
# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Stanford dependency tagger
# dep_parser = StanfordDependencyParser(model_path="./models/lexparser/englishPCFG.caseless.ser.gz")

# dependency parsing server
# dep_server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))
# Sentiment analyzer
sid = SentimentIntensityAnalyzer()
# List of stopword
stopwords = set(stopwords.words('english'))  # Add more stopword to ignore her
# POS to keep (for build mode 1)
preferredTags = {'NOUN', 'PROPN'}  # use in build mode 1 only
# Graph building mode
BUILD_MODE = 0  # 0: Do nothing

# Sentiment analysis mode:
#   'global': use TextBlob sentiment analysis to analyse the whole comment. 'local': perform at the sentence level
SENTIMENT_ANALYSIS_MODE = 'local'  # m

# Read number of computational threads will be used during the extraction
N_THREADS = 2

# a lock for controlling server request on Multi threading
threadLock = threading.Lock()

######################################
# FUNCTIONS


# Build a aspect graph from inpData.
# @params: threads - discussion threads. Structure array of {id: "id", :"content", supports:[]}.
#           merging_mode:
#               - 0: NO merging. Just put all graphs together
#               - 1: Keyword match. unify keywords that are exactly the same into one node
#               - 2: Semantic similarity
# @return: One summarization aspect graph.
def build_sum_graph(dataset):
    # Read options
    MERGE_MODE = build_options['build_mode'] if build_options['build_mode'] else 0
    global N_THREADS
    N_THREADS = build_options['n_thread'] if build_options['n_thread'] else 1
    maybe_print("Start building sum graph in mode {0}".format(MERGE_MODE), 1)
    global SENTIMENT_ANALYSIS_MODE
    SENTIMENT_ANALYSIS_MODE = build_options['sentiment_ana_mode'] if build_options['sentiment_ana_mode'] else 'global'

    if MERGE_MODE == 0:
        threads = dataset['comments']
        g = build_mode_0(threads)
        #  print g.edges()
        maybe_print("--> Graph BUILD in mode 0 completed.\n    Number of nodes: {0}\n    Number of edges: {1}"
                    .format(len(g.nodes()), len(g.edges())), 1)
        #  print "---", g.nodes()
        return g
    if MERGE_MODE == 1:
        g = build_mode_1(dataset['title'], dataset['article'], dataset['comments'])
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # maybe_print("--> Graph BUILD in mode 1 completed.\n    Number of nodes: {0}\n    Number of edges: {1}"
        #            .format(len(g.nodes()), len(g.edges())), 1)
        # print "---", g.nodes()
        return g


# Merging mode 0: Do nothing. Indeed, it just copy exactly all nodes and edges from the extracted keygraph.
# @param: Target graph G and data threads to be merge thrds
# @output: The result graph
def build_mode_0(threads):
    rs = nx.Graph()

    for thrd in threads:
        maybe_print(":: Building aspect graph for text {0}".format(thrd), 3)
        # Extract the graph
        cen_gr, sup_grs = build_thread_graph(thrd)  # Extract central and supports
        maybe_print("---- Found {0} centroid and {1} valid support(s).\n".format(1 if cen_gr else 0, len(sup_grs)),
                    3)
        # Add the nodes
        if cen_gr:
            # print "XXXXXX", len(cen_gr.nodes(data=True))
            rs.add_nodes_from(cen_gr.nodes(data=True))
        if sup_grs:
            rs.add_nodes_from(flatten_list([sup_gr.nodes(data=True) for sup_gr in sup_grs]))
        # Add the edges
        if cen_gr:
            # print "XXXXXX", len(cen_gr.edges(data=True))
            rs.add_edges_from(cen_gr.edges(data=True))
            # print cen_gr.edges()
        if sup_grs:
            rs.add_edges_from(flatten_list([sup_gr.edges(data=True) for sup_gr in sup_grs]))
            # print ooo
    return rs


# ================== UNDER CONSTRUCTION =================
# Merging mode 1: In build mode 1, we perform the several tasks:
# 1. Extract the graph representation of the article (title and content)
# 2. Attach discussion comments to the built graph
# 3. Extract the graph for comments
# 4. Attach these extracted to the built graph
# @param: Target graph G and data threads to be merge thrds
# @output: The result graph
def build_mode_1(title, article, comments):
    rs = nx.DiGraph()
    # First add the title as central node
    rs.add_node('Central~title~_~{0}'.format(gen_mcs_only()), {"label": title[:9] + "...",
                                                               "weight": 1,
                                                               "group_id": ['central.group'],
                                                               "sentiment": {'pos_count': 0,
                                                                             'neu_count': 1,
                                                                             'neg_count': 0}
                                                               })

    # Second work on the article
    maybe_print("\nStart building aspect graph for the ARTICLE.", 2)
    article_graph = nx.DiGraph()
    article_group_count = 0
    for segment in texttiling_tokenize(article):  # Run texttiling, then go to each segment
        maybe_print(" - Building graph for segment {0}".format(article_group_count), 2)
        segment_graph = build_directed_graph_from_text(txt=segment,
                                                       group_id="art."+str(article_group_count))
        article_graph = nx.compose(article_graph, segment_graph)
        article_group_count += 1

    # Unify the article
    rs = nx.compose(rs, article_graph)
    rs = graph_unify(rs, uni_options)

    # threadLock.release()  # Release the lock that may be used while extracting graph for article
    global server
    server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(), jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))

    maybe_print("\nStart building aspect graph for COMMENTS, add up to the article graph.",2)
    # Third work on the comments.
    # Check if we use thread structure or not
    if build_options['use_thread_structure']:
        raise RuntimeError("Thread structure usage in build mode 1 is NOT supported!")
    else:
        count = 0
        data_chunks = [[] for _ in xrange(0,N_THREADS)]
        while count < len(comments):
            data_chunks[count % N_THREADS].append(comments[count])
            count += 1
        # initialize th threads
        threads = []
        results = [[] for _ in xrange(0,N_THREADS)]
        for i in xrange(0,N_THREADS):
            thread = DirGraphExtractor("Thread "+ str(i), data_chunks[i], results[i])
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        for i in xrange(0,N_THREADS):
            for th_rs in results[i]:
                if th_rs:
                    rs = nx.compose(rs, th_rs)
                    # print 'Thread ', i, th_rs.edges(data=True)

        #for comment in comments:
        #    comment_id = comment['member_id']
        #    group_id = comment['group_id']
        #    content = comment['content']
        #    comment_graph = build_directed_graph_from_text(txt=content, group_id=group_id, member_id=comment_id)
        #    if comment_graph:
        #        rs = nx.compose(rs, comment_graph)
    # print 'Composed: ', rs.edges(data=True)
    rs = graph_unify(rs, uni_options)
    # print 'Unified: ', rs.edges(data=True)
    return rs


# Build the graph with text THREAD as input. Each thread has a structure as defined in the next procedure
# @param: text thread, each has a ID, a central and some supports
# @return: central_graph, support_graphs(as an array) and dictionary for looking up
def build_thread_graph(thrd):
    maybe_print("- Building keygraph for thread " + thrd['id'], 1)
    group_id = thrd['id']
    central = thrd['central']
    supports = thrd['supports']
    maybe_print("-------\nAnalyzing thread {0} with {1} central and {2} support(s)"
                .format(group_id, "ONE" if central else "NO", len(supports)), 2)
    # Build graph for central text
    central_gr = None
    if central:
        central_gr = build_graph_from_text(central, group_id, '0')

    # Build graphs for support texts
    supports_gr = [build_graph_from_text(supports[i], group_id, i) for i in xrange(0, len(supports))]
    # print supports_gr[0].edges()
    return central_gr, [sup for sup in supports_gr if sup]


# Build a graph from a text ---- Initial implementation
def build_graph_from_text(txt, group_id='_', member_id='_'):
    maybe_print(u"Generating graph for text: {0}".format(txt), 3)
    sentences = sent_tokenize(txt.strip())  # sentence segmentation
    g = nx.Graph()
    # Get nodes and edges from sentence
    sen_count = 0
    sen_scores = []
    # Perform sentiment analysis for GLOBAL mode
    if SENTIMENT_ANALYSIS_MODE == 'global':
        sen_scores = [sid.polarity_scores(txt)['compound'] for _ in xrange(0, len(sentences))]
    for sen in sentences:
        #  print sen
        # Extract named-entities
        named_entities = []
        n_ent = 0
        pg_text = Text(sen)  # convert to pyglot
        try:
            n_ent = pg_text.entities
        except Exception:
            maybe_print("No entity found!", 3)
            pass

        if n_ent > 0:
            named_entities = [' '.join(list(item)) for item in pg_text.entities if len(item) > 1]

        tb_text = TextBlob(sen)  # Convert to textblob. # find noun phrases in text.noun_phrases
        noun_phrases = tb_text.noun_phrases

        raw_text = sen
        for item in (named_entities+noun_phrases):  # group the words in noun phrase / NE into one big word
            raw_text = raw_text.replace(item, item.replace(' ', '_'))
        # Convert to polygot format
        text = Text(raw_text, hint_language_code='en')
        # Filter tokens by POS tag
        preferred_words = [lemmatizer.lemmatize(w.lower()) for w, t in text.pos_tags if t in preferredTags]
        # Filter stopwords
        #  filtered_words = list(set([w for w in preferred_words if w not in stopwords]))
        filtered_words = [w for w in preferred_words if w not in stopwords]
        # Perform sentiment analysis for LOCAL mode
        sen_score = 0
        if SENTIMENT_ANALYSIS_MODE == 'local':
            sen_score = sid.polarity_scores(raw_text)['compound']
            sen_scores.append(sen_score)

        # Assign id and label to the nodes before adding the graph
        assigned_nodes = [('{0}~{1}~{2}~{3}'.format(filtered_words[i], group_id, member_id, gen_mcs_only()),
                           {'label': filtered_words[i]}) for i in xrange(0, len(filtered_words))]
        #  print '---____----',assigned_nodes
        g.add_nodes_from(assigned_nodes)  # Add nodes from filtered words
        # Update nodes's weight
        for node in assigned_nodes:
            try:  # Node has already existed
                g.node[node[0]]['weight'] += 1
                # Update sentiment score
                if sen_score > 0:
                    g.node[node[0]]['sentiment']['pos_count'] += 1
                elif sen_score < 0:
                    g.node[node[0]]['sentiment']['neg_count'] += 1
                else:
                    g.node[node[0]]['sentiment']['neu_count'] += 1

            except KeyError:  # New node
                g.node[node[0]]['weight'] = 1
                g.node[node[0]]['group_id'] = group_id
                g.node[node[0]]['sentiment'] = {'pos_count': 1 if sen_score > 0 else 0,  # Add sentiment score
                                                'neg_count': 1 if sen_score < 0 else 0,
                                                'neu_count': 1 if sen_score == 0 else 0}

        maybe_print('Sentence no ' + str(sen_count) + '\nNodes ' + str(g.nodes()), 3)
        sen_count += 1
        edges = combinations([i[0] for i in assigned_nodes], 2)
        filtered_edges = [(n, m) for n, m in edges if n.split('~')[0] != m.split('~')[0]]
        #  print list(edges)
        #  print '-----',filtered_edges
        if filtered_edges:
            g.add_edges_from(filtered_edges)  # Add edges from the combination of words co-occurred in the same sentence
            # Update edges's weight
            for u, v in filtered_edges:
                try:
                    g.edge[u][v]['weight'] += 1
                except KeyError:
                    g.edge[u][v]['weight'] = 1
            maybe_print('Edges ' + str(g.edges()) + '\n', 3)
        sen_count += 1  # Increase the sentence count index
    if len(g.nodes()) == 0:
        return None

    maybe_print('Nodes ' + str(g.nodes()), 3)
    maybe_print('Edges ' + str(g.edges()) + '\n', 3)
    return g


# This function build a directed graph from a
# Build a graph from a text ---- Directional implementation
# This function build a directed graph fom a piece of text
def build_directed_graph_from_text(txt, group_id='', member_id=''):
    maybe_print(u"Generating directed graph for text: {0}".format(txt), 3)
    # First do the co-reference refine
    corefered_txt = coreference_refine(txt)
    sentences = sent_tokenize(corefered_txt.strip())  # sentence segmentation
    g = nx.DiGraph()
    # Get nodes and edges from sentence
    sen_count = 0
    sen_scores = []
    # Perform sentiment analysis for GLOBAL mode
    if SENTIMENT_ANALYSIS_MODE == 'global':
        sen_scores = [sid.polarity_scores(txt)['compound'] for _ in xrange(0, len(sentences))]
    for sen in sentences:
        dependencies, keys, new_sen = dep_extract_from_sent(sen, dep_opt)
        # Perform sentiment analysis for LOCAL mode
        sen_score = 0
        if SENTIMENT_ANALYSIS_MODE == 'local':
            sen_score = sid.polarity_scores(new_sen)['compound']
            sen_scores.append(sen_score)

        # Assign id and label to the nodes before adding the graph
        assigned_nodes = [('{0}~{1}~{2}~{3}'.format(keys[i], group_id, member_id, gen_mcs_only()),
                           {'label': keys[i], 'weight':0}) for i in xrange(0, len(keys))]
        # print '---____----',assigned_nodes
        g.add_nodes_from(assigned_nodes)  # Add nodes from filtered words
        # Update nodes's weight
        for node in assigned_nodes:
            # print g.node[node[0]]
            try:  # Node has already existed
                g.node[node[0]]['weight'] += 1
                # print g.node[node[0]]
                # Update sentiment score
                if sen_score > 0:
                    g.node[node[0]]['sentiment']['pos_count'] += 1
                elif sen_score < 0:
                    g.node[node[0]]['sentiment']['neg_count'] += 1
                else:
                    g.node[node[0]]['sentiment']['neu_count'] += 1

            except KeyError:  # New node
                g.node[node[0]]['weight'] = 1
                g.node[node[0]]['group_id'] = {group_id}  # it's a set
                g.node[node[0]]['sentiment'] = {'pos_count': 1 if sen_score > 0 else 0,  # Add sentiment score
                                                'neg_count': 1 if sen_score < 0 else 0,
                                                'neu_count': 1 if sen_score == 0 else 0}
        maybe_print('Sentence no ' + str(sen_count) + '\nNodes ' + str(g.nodes()), 3)
        sen_count += 1
        word2id = dict(zip(keys, assigned_nodes))
        filtered_edges = [(word2id[s][0], word2id[t][0], {'label': r,
                                                          'weight': 1}) for (s, _), r, (t, _) in dependencies if s != t and r]
        #  print list(edges)
        #  print '-----',filtered_edges
        if filtered_edges:
            g.add_edges_from(filtered_edges)  # Add edges from the combination of words co-occurred in the same sentence
            # Update edges's weight
            for u, v, r in filtered_edges:
                try:
                    g.edge[u][v]['weight'] += 1
                    g.edge[u][v]['label'] = u'{0},{1}'.format(g.get_edge_data(u,v)['label'], r['label'])
                except KeyError:
                    g.edge[u][v]['weight'] = 1
                    g.edge[u][v]['label'] = r['label']
            maybe_print('Edges ' + str(g.edges()) + '\n', 3)
        sen_count += 1  # Increase the sentence count index
    if len(g.nodes()) == 0:
        # raise ValueError("Generated graph is empty")
        return None

    maybe_print('   + Graph for group: {0:5s} \t member: {1:15s} \t has {2:3d} nodes and {3:3d} edges '
                .format(group_id,member_id,len(g.nodes()),len(g.edges())), 2)
    maybe_print('Nodes ' + str(g.nodes()), 3)
    maybe_print('Edges ' + str(g.edges()) + '\n', 3)
    return g


# Declaration for graph extractor in order to run mulithreading
class DirGraphExtractor (threading.Thread):
    name = 'unnamed'
    data = []
    result = []
    dep_parser = None

    def __init__(self, name, data, result):
        threading.Thread.__init__(self)
        self.name = name
        self.data = data
        self.result = result

    def run(self):
        print "Starting extractor {0} with {1} data records".format(self.name, len(self.data))
        for record in self.data:
            txt = record['content'] if record['content'] else ""
            group = record['group_id'] if record['group_id'] else ""
            member = record['member_id'] if record['member_id'] else ""
            g = build_directed_graph_from_text(txt=txt, group_id=group, member_id=member)
            if g:
                self.result.append(g)
        print "Exiting extractor {0}".format(self.name)


# Pruning the graph according to restrictions in options
def prune_graph(graph):
    maybe_print("Start pruning aspect graph.", 1)
    ori_nnode = graph.number_of_nodes()
    ori_nedge = graph.number_of_edges()
    options = prune_options
    g = graph
    # Load options
    if not g or not options:
        return None
    # Set default values
    # Now read the options
    ENABLE_PRUNING = options['enable_prunning']
    if not ENABLE_PRUNING:
        maybe_print("--> Pruning skipped.", 1)
        return graph  # Skip the pruning, return original graph

    NUM_MIN_NODES = 20      # minimum number of node. If total number of nodes is < this number, pruning will be skipped
    NUM_MIN_EDGES = 30      # minimum number of edge. The value could not more than (NUM_MIN_NODES)*(NUM_MIN_NODES-1).
    #                       If total number of edge is  < this number, pruning will be skipped
    REMOVE_ISOLATED_NODE = True #

    NUM_MAX_NODES = 200     # maximum number of nodes to keep
    NUM_MAX_EDGES = 300     # maximum number of edge to keep.
    #                       The value could not more than (NUM_MAX_NODES)*(NUM_MAX_NODES-1)
    EDGE_FREQ_MIN = 1       # minimum frequency that an edge is required to be. Being smaller, it will be eliminated.
    NODE_FREQ_MIN = 1       # minimum frequency that a node is required to be. Being smaller, it will be eliminated.
    NODE_DEGREE_MIN = 1     # minimum degree that a node is required to have. Being smaller, it will be eliminated.
    MIN_WORD_LENGTH = 3     # Minimum number of character of a word, accepted to enter the graph
    RE_PATTERN = '.+'       # Regular expression pattern to filter out the node label
    WHITE_NODES_LIST = []   # While list of words to be kept
    BLACK_NODE_LIST = []    # Black list of words to be destroyed
    BLACK_DEPENDENCIES = [] # Black list of dependency to be destroyed

    if 'num_min_nodes' in options:
        NUM_MIN_NODES = options['num_min_nodes']
    if 'num_min_edges' in options:
        NUM_MIN_EDGES = options['num_min_edges']

    if 'min_word_length' in options:
        MIN_WORD_LENGTH = options['min_word_length']

    if 'remove_isolated_node' in options:
        REMOVE_ISOLATED_NODE = options['remove_isolated_node']

    if 'num_max_nodes' in options:
        NUM_MAX_NODES = options['num_max_nodes']
    if 'num_max_edges' in options:
        NUM_MAX_EDGES = options['num_max_edges']

    if 'edge_freq_min' in options:
        EDGE_FREQ_MIN = options['edge_freq_min']
    if 'node_freq_min' in options:
        NODE_FREQ_MIN = options['node_freq_min']
    if 'node_degree_min' in options:
        NODE_DEGREE_MIN = options['node_degree_min']

    if 'regex_pattern' in options:
        RE_PATTERN = options['regex_pattern']

    if 'white_node_labels' in options:
        WHITE_NODES_LIST = set(options['white_node_labels'])

    if 'black_node_labels' in options:
        BLACK_NODE_LIST = set(options['black_node_labels'])

    if 'black_dependencies' in options:
        BLACK_DEPENDENCIES = set(options['black_dependencies'])


    maybe_print("Start pruning the graph.", 1)
    # :: Perform pruning
    # Decide whether to skip the pruning because of the tiny size of graph
    if len(g.nodes()) < NUM_MIN_NODES or len(g.edges()) < NUM_MIN_EDGES:
        return None # Skip the pruning

    # Remove short words
    to_remove_nodes = []
    to_remove_edges = []
    for node in g.nodes():
        node_label = g.node[node]['label']
        if node_label in WHITE_NODES_LIST:
            continue
        elif node_label in BLACK_NODE_LIST:
            to_remove_nodes.append(node)
        elif len(node_label) < MIN_WORD_LENGTH or not re.match(RE_PATTERN,g.node[node]['label']):
            to_remove_nodes.append(node)
    for edge in g.edges():
        if (edge[0] in to_remove_nodes) or (edge[1] in to_remove_nodes):
            to_remove_edges.append(edge)
    g.remove_nodes_from(to_remove_nodes)
    g.remove_edges_from(to_remove_edges)

    # Filter nodes by frequency
    if NODE_FREQ_MIN > 1:
        to_remove_nodes = [n for n in g.nodes()
                           if g.node[n]['weight'] < NODE_FREQ_MIN and g.node[n]['label'] not in WHITE_NODES_LIST]
        g.remove_nodes_from(to_remove_nodes)

    if EDGE_FREQ_MIN > 1:
        to_remove_edges = [(s, t) for (s, t) in g.edges() if g.edge[s][t]['weight'] < EDGE_FREQ_MIN]
        g.remove_edges_from(to_remove_edges)

    # Filter nodes by degree
    if NODE_DEGREE_MIN > 1:
        to_remove_nodes = [n for n in g.nodes()
                           if g.degree(n) < NODE_DEGREE_MIN and g.node[n]['label'] not in WHITE_NODES_LIST]
        g.remove_nodes_from(to_remove_nodes)

    # Remove isolated nodes
    to_remove_edges = []
    if REMOVE_ISOLATED_NODE:
        degrees = nx.degree(g)
        to_remove_nodes = [i for i in degrees if degrees[i] == 0 and g.node[i]['label'] not in WHITE_NODES_LIST]
        for edge in g.edges(data=True):
            if (edge[0] in to_remove_nodes) or (edge[1] in to_remove_nodes) or (edge[2]['label'] in BLACK_DEPENDENCIES):
                to_remove_edges.append(edge)
        g.remove_nodes_from(to_remove_nodes)
        g.remove_edges_from(to_remove_edges)

    # :: Done pruning
    maybe_print("--> Graph PRUNNING completed.\n    Number of nodes: {0}\n    Number of edges: {1}"
                .format(len(g.nodes()), len(g.edges())),2)
    maybe_print("--> {0} nodes removed. {1} edges removed.".format(ori_nnode - g.number_of_nodes(),
                                                                   ori_nedge - g.number_of_edges()),2)
    return g


# Compute polarity of sentiment of the graph G. This score is compute as (positive_score-negative_score)/max_freq
# where max_freq is the normalization constant, and max_freq is the maximum frequency of all nodes on the graph
# @param: graph g
# @return: new graph with sentiment_score attached
def compute_sentiment_score(g):
    # get the normalization constant
    max_val, _ = get_max_value_attribute(g, 'weight')
    # print max_val
    tg_g = g  # copy the graph
    for n in g.nodes():
        # print g.node[n]['sentiment']['pos_count'],g.node[n]['sentiment']['pos_count']
        # print g.node[n],"\n"
        tg_g.node[n]['sentiment_score'] = float((g.node[n]['sentiment']['pos_count']
                                                 - g.node[n]['sentiment']['neg_count'])) / max_val
    return tg_g


# Extract a graph from a sentence
# @param: a sentence, and filtering options
# @output: 1. a list of dependencies 2. a list of keys, 3. the sentence after grouped compounds/entities
def dep_extract_from_sent(sent, filter_opt):
    sentence = sent
    blob = TextBlob(sentence)
    # print blob.noun_phrases
    for phrase in blob.noun_phrases:
        sentence = sentence.replace(phrase,phrase.replace(u' ',u'_'))
    # print sentence
    '''
    result = dep_parser.raw_parse(sentence)
    dependencies = result.next()
    raw_results = [((lemmatizer.lemmatize(s.lower()), s_tag), r, (lemmatizer.lemmatize(t.lower()), t_tag))
                   for (s, s_tag), r, (t, t_tag) in list(dependencies.triples())]
    '''
    parse_result = None
    try:
        threadLock.acquire()
        r = server.parse(sentence)
        threadLock.release()
        parse_result = loads(r)
    except Exception as detail:
        cut = min(30,len(sentence))
        maybe_print('[W] Unable to parse sentence for dependencies: {0}.'.format(sentence[:cut]),2)
        maybe_print('    --> Error: {0}'.format(detail), 2)
        threadLock.release()
        return [],[],u""
    pos_dict = dict()
    # build look up word-postag dictionary
    for e in re.findall('\(([A-Z]{1,4}\s[\w\d\'_-]+)\)', parse_result[u'sentences'][0][u'parsetree']):
        v = e.split()
        pos_dict[v[1]] = v[0]

    # Get the parsing result and save them a tripples
    raw_results = []
    # print parse_result[u'sentences'][0][u'dependencies']
    for e in parse_result[u'sentences'][0][u'dependencies']:
        r = e[0].split(u":")[0]
        s = e[1]
        t = e[2]
        if r not in [u'root',u'punct'] and s in pos_dict and t in pos_dict and len(s)>1 and len(t)>1:

            raw_results.append(((lemmatizer.lemmatize(s.lower()), pos_dict[s]),
                                r,
                                (lemmatizer.lemmatize(t.lower()), pos_dict[t])))

    #  Filter by relationship
    preferred_rel = filter_opt['preferred_rel']
    if type(preferred_rel) != list:  # take all POS
        filter_rel_result = raw_results
    else:
        # filter tripples that has relation in the list
        filter_rel_result = [trip for trip in raw_results if (trip[1] in preferred_rel)]

    # Custom node contract
    if dep_opt['custom_nodes_contract'] and dep_opt['custom_nodes_contract']['enable']:
        to_replace_keys = dict()
        for (s, s_tag), r, (t, t_tag) in filter_rel_result:
            for rule in dep_opt['custom_nodes_contract']['rule_set']:
                if rule['from_pos'] == s_tag and rule['to_pos'] == t_tag and rule['rel_name'] == r:
                    keyword = t + u'_' + s if rule['rs_direction'] == u'1-2' else s + u'_' + t
                    pos = rule['rs_pos']
                    to_replace_keys[s] = {'key': keyword, 'tag': pos}
                    to_replace_keys[t] = {'key': keyword, 'tag': pos}
                    break
        contracted_nodes_result = set()
        key_set = set()   # Save the set of words, so we can use it for further
        edge_set = set()  # Save all the edges
        for (s, s_tag), r, (t, t_tag) in filter_rel_result:
            item = (to_replace_keys[s]['key'] if s in to_replace_keys else s,
                    to_replace_keys[s]['tag'] if s in to_replace_keys else s_tag), \
                   r, \
                   (to_replace_keys[t]['key'] if t in to_replace_keys else t,
                    to_replace_keys[t]['tag'] if t in to_replace_keys else t_tag)
            if item[0][0] != item[2][0]:
                key_set.add((item[0][0], s_tag))
                key_set.add((item[2][0], t_tag))
                edge_set.add((item[0][0],item[2][0]))
                contracted_nodes_result.add(item)

        del to_replace_keys
        # print contracted_nodes_result
        contracted_nodes_result = list(contracted_nodes_result)
    else:
        contracted_nodes_result = filter_rel_result

    # Custom edge contract
    # Extract all existing tuples
    contracted_edges_result = None
    if dep_opt['custom_edges_contract'] and dep_opt['custom_edges_contract']['enable']:
        to_add_rels = set()
        to_remove_rels = set()
        for node,tag in key_set:
            # get a the list of nodes that it connected to
            # print node
            neighbors = [((s, s_tag), r, (t, t_tag), u'out' if s == node else u'in')
                         for (s, s_tag), r, (t, t_tag) in contracted_nodes_result
                         if s == node or t == node]
            permutations = list(itertools.permutations(neighbors,2))
            # print permutations
            if permutations:
                for rel1,rel2 in permutations:
                    # print "@@@@@",rel1, rel2
                    # Now search in the rule set to see if there is any match to the current node
                    for r in dep_opt['custom_edges_contract']['rule_set']:
                        # print r
                        if r['rel_name1'] == rel1[1] and r['rel_direction1'] == rel1[3] and r['rel_name2'] == rel2[1] \
                           and r['rel_direction2'] == rel2[3] and r['n_pos'] == tag:  # Matched
                            node1 = rel1[2][0] if rel1[3] == u'out' else rel1[0][0]
                            node3 = rel2[2][0] if rel2[3] == u'out' else rel2[0][0]
                            if node1 == node3:  # skip duplicate label
                                continue

                            rs_label = r['rs_label'].replace(u'{n_label}', en.verb.infinitive(node))\
                                                    .replace(u'{l_label}', en.verb.infinitive(node1)) \
                                                    .replace(u'{r_label}', en.verb.infinitive(node3))
                            rs_label = rs_label.upper()
                            # print rel1, rel2
                            # Define new nodes
                            if r['nodes_label']==u'1-2':
                                to_add_rels.add((rel1[2] if rel1[3] == u'out' else rel1[0],
                                                 rs_label,
                                                 (node,tag)
                                                 ))
                            elif r['nodes_label']==u'2-1':
                                to_add_rels.add(((node,tag),
                                                rs_label,
                                                rel1[2] if rel1[3] == u'out' else rel1[0]))
                            elif r['nodes_label']==u'2-3':
                                to_add_rels.add(((node,tag),
                                                rs_label,
                                                rel2[2] if rel2[3] == u'out' else rel2[0]))
                            elif r['nodes_label'] == u'3-2':
                                to_add_rels.add((rel2[2] if rel2[3] == u'out' else rel2[0],
                                                rs_label,
                                                (node, tag)
                                                ))
                            elif r['nodes_label'] == u'1-3':
                                to_add_rels.add((rel1[2] if rel1[3] == u'out' else rel1[0],
                                                rs_label,
                                                rel2[2] if rel2[3] == u'out' else rel2[0]))
                            elif r['nodes_label'] == u'3-1':
                                to_add_rels.add((rel2[2] if rel2[3] == u'out' else rel2[0],
                                                rs_label,
                                                rel2[2] if rel2[3] == u'out' else rel2[0]))
                            else:
                                raise ValueError("[E] Invalid node label definition: ")
                            '''
                            if r['rs_direction'] == u'left-to-right':
                                to_add_rels.add((rel1[2] if rel1[3] == u'out' else rel1[0],
                                                 rs_label,
                                                 rel2[2] if rel2[3] == u'out' else rel2[0]))
                            else:  # right to left
                                to_add_rels.add((rel2[2] if rel2[3] == u'out' else rel2[0],
                                                 rs_label,
                                                 rel1[2] if rel1[3] == u'out' else rel1[0]))
                            '''
                            # print '!@$@!!!!!!!!!!!!!!!!!!!!!!!!'
                            to_remove_rels.add(rel1[:3])  # Mark these relationship to remove
                            to_remove_rels.add(rel2[:3])
                # Now remove the contracted relationships
                # print '[org]',contracted_nodes_result
                # print '[rmv]', to_remove_rels
                # print '[add]',to_add_rels
                contracted_edges_result = list((set(contracted_nodes_result) - to_remove_rels) | to_add_rels)
                # print '[rs]', contracted_edges_result
        if len(to_add_rels) > 0:
            maybe_print(u"   + Contracted {0} ed-n-ed using rules for sentence \"{1}...\": {2}"
                        .format(len(to_add_rels),sentence[:50], str(['{0}--{1}->{2}'.format(s,r,t)
                                                                     for (s,_),r,(t,_) in to_add_rels])),2)
            # print "asdsad", contracted_edges_result  ######################################################

    if not contracted_edges_result:
        contracted_edges_result = contracted_nodes_result
    # print '[rs2]', contracted_edges_result
    # Compound merge
    new_sen = sentence  # new sentence contains grouped item with _ as connector
    # Merge compounds
    compound_merge = filter_opt['compound_merge']
    if compound_merge:
        tokens = [lemmatizer.lemmatize(w) for w in word_tokenize(sentence.lower())]
        compounds = [(t, s) for (s, s_tag), r, (t, t_tag) in contracted_edges_result if r == u'compound']
        # print "!!!!!!!!!", compounds
        replacements = dict()
        for i in xrange(0, len(tokens)-1):
            if (tokens[i], tokens[i+1]) in compounds:
                replacements[tokens[i]] = tokens[i]+"_" + tokens[i+1]
                replacements[tokens[i+1]] = tokens[i] + "_" + tokens[i + 1]
                # Now do the replace
                # print "found compound", tokens[i],tokens[i+1]
        for i in xrange(0, len(contracted_edges_result)):
            (s, s_tag), r, (t, t_tag) = contracted_edges_result[i]
            if s in replacements:
                contracted_edges_result[i] = (replacements[s], s_tag), r, (t, t_tag)
            (s, s_tag), r, (t, t_tag) = contracted_edges_result[i]  # Update in case 1st element changes
            if t in replacements:
                contracted_edges_result[i] = (s, s_tag), r, (replacements[t], t_tag)
        for key in replacements:
            new_sen.replace(key, replacements[key])
        # now remove duplicate
        filter_comp_result = [((s, s_tag), r, (t, t_tag)) for (s, s_tag), r, (t, t_tag) in contracted_edges_result
                              if (s != t)]
    else:
        filter_comp_result = contracted_edges_result

    # Filter out by POS
    preferred_pos = filter_opt['preferred_pos']
    if type(preferred_pos) != list:  # take all POS
        filter_pos_result = contracted_edges_result
    else:
        # filter triples whose beginning and ending tags are inside the list
        # print raw_results
        # print filter_pos_result
        filter_pos_result = [trip for trip in contracted_edges_result
                             if (trip[0][1] in preferred_pos or len(trip[0][0]) > 10) and
                                (trip[2][1] in preferred_pos or len(trip[2][0]) > 10)
                             ]  # keep potential phrases

    # Final refine
    rs = []  # store the refined tuples
    keys = set()  # store the keywords
    for (s, s_tag), r, (t, t_tag) in filter_pos_result:
        s_f = en.verb.infinitive(s) if en.verb.infinitive(s) else s
        t_f = en.verb.infinitive(t) if en.verb.infinitive(t) else t
        rs.append(((s_f, s_tag), r, (t_f, t_tag)))
        keys.add(s_f)
        keys.add(t_f)


    return rs, list(keys), new_sen


# Function to apply several unification methods
# @param: g - the graph to be unified, uni_opi - unification options
# @return: the unified graph
def graph_unify(g=None, uni_opt=None):
    # print g.edges(data=True)
    maybe_print("Start graph Unification", 2)
    # print uni_opt
    if not uni_opt:
        return g
    # print uni_opt
    #  For keyword matching and unification
    ENABLE_UNIFY_MATCHED_KEYWORDS = True
    INTRA_CLUSTER_UNIFY = True
    INTER_CLUSTER_UNIFY = False
    UNIFY_MODE = 'link'  # Modes"# 'link': create a virtual link between the nodes
    # 'contract': Direct contract, sum weight
    if 'unify_matched_keywords' in uni_opt:
        try:
            uop = uni_opt['unify_matched_keywords']
            ENABLE_UNIFY_MATCHED_KEYWORDS = uop['enable'] if uop['enable'] else False
            INTRA_CLUSTER_UNIFY = uop['intra_cluster_unify'] if uop['intra_cluster_unify'] else False
            INTER_CLUSTER_UNIFY = uop['inter_cluster_unify'] if uop['inter_cluster_unify'] else False
            UNIFY_MODE = uop['unification_mode'] if uop['unification_mode'] else 'link'
        except:
            raise ValueError("Error while parsing matched words unification options.")

    # Merge sub graph with similar keyword -> Now just
    if ENABLE_UNIFY_MATCHED_KEYWORDS:
        # print "!!#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$44"
        # for i in xrange(0,len(node_ids)-1):
        #    for j in xrange(i+1,len(node_ids)):
        #        if g.node[node_ids[i]]['label'] == g.node[node_ids[j]]['label']:  # two node have the same label
        #            matches.add((node_ids[i],node_ids[j]))
        # matches = list(matches)
        # Find matches
        match_dict = dict()
        for node in g.nodes(data=True):
            node_id = node[0]
            label = node[1]['label']
            # print "--------->", node_id, label
            if label not in match_dict:
                match_dict[label] = set([node_id])
            else:  # Key already exists
                # match_dict[label] = match_dict[label].add(node_id)
                match_dict[label].add(node_id)
        # print match_dict
        # Establish the match list
        matches = []
        for key in match_dict:
            ids = list(match_dict[key])
            if len(ids) < 2:
                continue  # Skip words appear once
            for i in xrange(1, len(ids)):
                matches.append((ids[0], ids[i]))
        del match_dict
        # print matches
        rs = g
        if matches:
            # Split matches to 2 set:
            # 1. Intra: set of node within the same segment/group
            # 2. InterL set of node between groups
            intra_match = []
            inter_match = []
            for n0, n1 in matches:
                # print n0, n1
                if g.node[n0]['group_id'] & g.node[n1]['group_id']:
                    intra_match.append((n0, n1))
                else:
                    inter_match.append((n0, n1))
            del matches
            # intra_match = list(intra_match)
            # inter_match = list(inter_match)
            # Do the unification. INTRA unification must be carried out before INTER unification. Because the scope of
            # the latter cover the former one

            # Implementation of INTER cluster unification
            if INTER_CLUSTER_UNIFY:
                # print "2....", inter_match
                if UNIFY_MODE == 'link':
                    max_score, _ = get_max_value_attribute(g, 'weight')  # Get the max weight of all nodes in graph
                    print max_score
                    # Now make the links
                    # print inter_match
                    for node0, node1 in inter_match:
                        rs.add_edge(node0, node1, {
                                                    'weight': max_score,
                                                    'id': node0 + '|' + node1,
                                                    'label': '_linked'
                                                    })
                elif UNIFY_MODE == 'contract':
                    while len(inter_match) > 0:
                        node0 = inter_match[0][0]
                        node1 = inter_match[0][1]
                        # Sum up the weight of the two node
                        sum_weight = rs.node[node0]['weight'] + rs.node[node1]['weight']
                        # print g.node[node0]['weight'], g.node[node1]['weight']
                        # Sum up the sentiment of the two nodes
                        pos_count = rs.node[node0]['sentiment']['pos_count'] + rs.node[node1]['sentiment']['pos_count']
                        neg_count = rs.node[node0]['sentiment']['neg_count'] + rs.node[node1]['sentiment']['neg_count']
                        neu_count = g.node[node0]['sentiment']['neu_count'] + rs.node[node1]['sentiment']['neu_count']
                        # Sum up the weight if two nodes has same neighbor
                        share_neighbors = set(nx.all_neighbors(rs, node0)) & set(nx.all_neighbors(rs, node1))
                        add_up_weights = []
                        if share_neighbors:
                            #  print share_neighbors
                            for n in share_neighbors:
                                n0_n_weight = rs.edge[node0][n]['weight'] if rs.has_edge(node0, n) else None
                                n_n0_weight = rs.edge[n][node0]['weight'] if rs.has_edge(n, node0) else None
                                n1_n_weight = rs.edge[node1][n]['weight'] if rs.has_edge(node1, n) else None
                                n_n1_weight = rs.edge[n][node1]['weight'] if rs.has_edge(n, node1) else None
                                if n0_n_weight and n1_n_weight:
                                    # print rs.edge[node0][n]['label'], rs.edge[node1][n]['label']
                                    label = unicode.join(u',',[rs.edge[node0][n]['label'], rs.edge[node1][n]['label']])
                                    #     rs.edge[node0][n]['label'] + u',' + rs.edge[node1][n]['weight']
                                    add_up_weights.append((node0, n, n0_n_weight+n1_n_weight, label))
                                if n_n0_weight and n_n1_weight:
                                    # label = rs.edge[n][node0]['label'] + u"," + rs.edge[n][node1]['weight']
                                    label = unicode.join(u',', [rs.edge[n][node0]['label'], rs.edge[n][node1]['label']])
                                    add_up_weights.append((n, node0, n_n0_weight+n_n1_weight, label))
                        group_id = rs.node[node0]['group_id'] | rs.node[node1]['group_id']
                        rs = nx.contracted_nodes(rs, node0, node1)
                        rs.node[node0]['weight'] = sum_weight
                        rs.node[node0]['label'] = rs.node[node0]['label']
                        rs.node[node0]['sentiment'] = {'pos_count': pos_count,
                                                       'neg_count': neg_count,
                                                       'neu_count': neu_count
                                                       }
                        rs.node[node0]['group_id'] = group_id
                        # Update the weight of edges that has been added
                        if add_up_weights:
                            for s, t, sw, lb in add_up_weights:
                                rs.edge[s][t]['weight'] = sw
                                rs.edge[s][t]['label'] = lb
                        # Update the match lst
                        inter_match.pop(0)  # Remove first element
            # Implementation of INTRA cluster unification
            if INTRA_CLUSTER_UNIFY:  # Unify the same  keywords in one cluster
                # print "1....", intra_match
                if UNIFY_MODE == 'link':
                    max_score, _ = get_max_value_attribute(g, 'weight')  # Get the max weight of all nodes in graph
                    print max_score
                    # Now make the links
                    print matches
                    for node0, node1 in intra_match:
                        rs.add_edge(node0, node1, {
                                                    'weight': max_score,
                                                    'id': node0 + '|' + node1,
                                                    'label': '_linked'
                                                    })
                else:
                    if UNIFY_MODE == 'contract':
                        while len(intra_match) > 0:
                            node0 = intra_match[0][0]
                            node1 = intra_match[0][1]
                            # Sum up the weight of the two node
                            sum_weight = rs.node[node0]['weight'] + rs.node[node1]['weight']
                            # Sum up the sentiment of the two nodes
                            pos_count = rs.node[node0]['sentiment']['pos_count'] \
                                        + rs.node[node1]['sentiment']['pos_count']
                            neg_count = rs.node[node0]['sentiment']['neg_count'] \
                                        + rs.node[node1]['sentiment']['neg_count']
                            neu_count = g.node[node0]['sentiment']['neu_count'] \
                                        + rs.node[node1]['sentiment']['neu_count']

                            # Sum up the weight if two nodes has same neighbor
                            share_neighbors = set(nx.all_neighbors(rs, node0)) & set(nx.all_neighbors(rs, node1))
                            add_up_weights = []
                            if share_neighbors:
                                #  print share_neighbors
                                for n in share_neighbors:
                                    n0_n_weight = rs.edge[node0][n]['weight'] if rs.has_edge(node0, n) else None
                                    n_n0_weight = rs.edge[n][node0]['weight'] if rs.has_edge(n, node0) else None
                                    n1_n_weight = rs.edge[node1][n]['weight'] if rs.has_edge(node1, n) else None
                                    n_n1_weight = rs.edge[n][node1]['weight'] if rs.has_edge(n, node1) else None
                                    if n0_n_weight and n1_n_weight:
                                        # label = rs.edge[node0][n]['label'] + u"," + rs.edge[node1][n]['weight']
                                        label = unicode.join(u',',
                                                             [rs.edge[node0][n]['label'], rs.edge[node1][n]['label']])
                                        add_up_weights.append((node0, n, n0_n_weight + n1_n_weight,label))
                                    if n_n0_weight and n_n1_weight:
                                        # label = rs.edge[n][node0]['label'] + u"," + rs.edge[n][node1]['weight']
                                        label = unicode.join(u',',
                                                             [rs.edge[n][node0]['label'], rs.edge[n][node1]['label']])
                                        add_up_weights.append((n, node0, n_n0_weight + n_n1_weight,label))

                            rs = nx.contracted_nodes(rs, node0, node1)
                            rs.node[node0]['weight'] = sum_weight
                            rs.node[node0]['label'] = rs.node[node0]['label']
                            rs.node[node0]['sentiment'] = {'pos_count': pos_count,
                                                           'neg_count': neg_count,
                                                           'neu_count': neu_count
                                                            }
                            # Update the weight of edges that has been added
                            if add_up_weights:
                                for s, t, sw, lb in add_up_weights:
                                    rs.edge[s][t]['weight'] = sw
                                    rs.edge[s][t]['label'] = lb
                            intra_match.pop(0)  # Remove first element
    else:
        return g
    maybe_print(" -> Unification complete! Number of nodes changed from {0} to {1}".format(len(g.nodes()),
                                                                                           len(rs.nodes())), 2)
    return rs


# Read a data file, cluster threads of discussion
# @param: path to the data file
# @return: data structure: [
#   [{'id':'cluster_id', 'central_comment': 'abc', 'supports':[support comments]},
#                       {}]
def read_comment_file(data_file, read_as_threads=False):
    dataset = None
    if read_as_threads:
        try:
            with codecs.open(data_file, "rb", "utf8") as comment_file:
                reader = unicode_csv_reader(comment_file, delimiter='	', quotechar='"')
                dataset = []
                index = -1
                count = 0
                for entry in reader:
                    #  print entry;
                    if entry[2] == u"":
                        cluster_id = gen_mcs() + "^" + str(index+1)
                        dataset.append({'id': cluster_id, 'central': text_preprocessing(entry[6]), 'supports': []})
                        index += 1
                    else:
                        if count > 0:
                            dataset[index]['supports'].append(text_preprocessing(entry[6]))
                    count += 1
        except IOError:
            print "Unable to open {0}".format(data_file)
        maybe_print(json.dumps(dataset, sort_keys=True, indent=4, separators=(',', ': ')), 3)
    else:
        try:
            with codecs.open(data_file, "rb", "utf8") as comment_file:
                reader = unicode_csv_reader(comment_file, delimiter='	', quotechar='"')
                dataset = []
                count = 0
                for entry in reader:
                    sentence_id = gen_mcs() + "^" + str(count)
                    dataset.append({'member_id': sentence_id,
                                    'group_id': 'com.',
                                    'content': text_preprocessing(entry[6])})
                    count += 1
        except IOError:
            print "Unable to open {0}".format(data_file)
        maybe_print(json.dumps(dataset, sort_keys=True, indent=4, separators=(',', ': ')), 3)
    return dataset


# Read the ARTICLE file
# @param: path to the data file
# @return: a pair of
#  1. title sentence
#  2. a list, each element is a sentence
def read_article_file(data_file):
    title = None
    article = None
    try:
        with codecs.open(data_file, "rb", encoding='utf-8') as article_file:
            count = 0
            article = []
            for line in article_file:
                if count == 0:
                    title = line
                    count += 1
                    continue
                else:
                    if not line.strip():
                        article[-1] = article[-1] + " \t\n"
                        article.append(u'*break_me*')
                    else:
                        sens = [text_preprocessing(s) for s in sent_tokenize(line.replace(u"\r\n", u""))]
                        # print sens
                        if sens:
                            article.extend(sens)

    except IOError:
        print "Unable to open {0}".format(data_file)
    return title, article


#  Read unicode csv file, ignore unencodable characters
def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]


# Refine a sentence by replacing its reference be the referee word/phrase
# @param: a sentence
# @return: refined sentence
def coreference_refine(text):
    if not text.strip():
        return text
    tokens = [[tok for tok in word_tokenize(sen)] for sen in sent_tokenize(text)]
    rs_tks = tokens
    parse_rs = None
    try:
        # parse = server.parse(text)
        threadLock.acquire()
        parse_rs = loads(server.parse(text))
        threadLock.release()
        # parse_rs = loads(parse)
    except Exception as detail:
        maybe_print("[W] Can't parse sentence this sentence for coreference \"{0}...\"\n --> Error: {1}"
                    .format(text[:min(30,len(text))],detail), 2)
        threadLock.release()
    try:
        if not parse_rs or 'coref' not in parse_rs:
            return text
        for group in parse_rs['coref']:
            for s, t in group:
                # print len(rs_tks[s[1]])
                if s[0] and t[0] and len(s[0]) < 50 and len(t[0]) < 50:
                    # calculate size differences:
                    diff = (s[4] - s[3]) - (t[4] - t[3])
                    # Remove the reference
                    # print rs_tks[s[1]], s[3], s[4]
                    for i in xrange(s[3], s[4]):
                        rs_tks[s[1]].pop(s[3])
                    if diff == 0:
                        # Add the referee
                        starting_pos = s[3]
                        for i in xrange(t[3], t[4]):
                            rs_tks[s[1]].insert(starting_pos, tokens[t[1]][i])
                            starting_pos += 1
                    elif diff > 0:  # to-be-replace is greater than to replace
                        # Add the referee
                        starting_pos = s[3]
                        for i in xrange(t[3], t[4]):
                            #if i >= t[4]:
                            #    rs_tks[s[1]].insert(starting_pos, u"")
                            #else:
                            rs_tks[s[1]].insert(starting_pos, tokens[t[1]][i])
                            starting_pos += 1
                        for i in xrange(0,diff):
                            rs_tks[s[1]].insert(starting_pos, u"")
                            starting_pos += 1
                    else:  # to-be-replace is greater than to replace
                        # Add the referee
                        starting_pos = s[3]
                        for i in xrange(t[3], t[4] + diff):
                            if i == t[4] + diff -1:
                                item = u""
                                for j in xrange(i, t[4]):
                                    item = item + u" " + tokens[t[1]][j]
                                item = item[1:]
                                rs_tks[s[1]].insert(starting_pos, item)
                            else:
                                rs_tks[s[1]].insert(starting_pos, tokens[t[1]][i])
                                starting_pos += 1
                # print len(rs_tks[s[1]])
        rs = u" ".join([u" ".join(s_list) for s_list in rs_tks])
        if rs:
            return rs
        else:
            return text
    except Exception as detail:
        maybe_print("[W] --> unable to extract co-reference for sentence: \"{0}...\"\n     --> Error: {1}"
                    .format(text[:min(30,len(text))],detail), 2)
        return text