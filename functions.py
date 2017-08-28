#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    FUNCTIONS FOR TEXT SUMMARIZATION
"""

######################################
# IMPORT LIBRARY
import csv
import codecs
import json
from datetime import datetime
import networkx as nx
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, texttiling, word_tokenize
from nltk.corpus import stopwords
from itertools import combinations
from polyglot.text import Text
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils import *
from main import dep_opt
from nltk.parse.stanford import StanfordDependencyParser


######################################
# EXTRA DECLARATIONS
# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Stanford dependency tagger
dep_parser = StanfordDependencyParser(model_path="./models/lexparser/englishPCFG.caseless.ser.gz")

# Sentiment analyzer
sid = SentimentIntensityAnalyzer()

stopwords = set(stopwords.words('english'))  # Add more stopword to ignore her


preferredTags = {'NOUN', 'PROPN'}




BUILD_MODE = 0  # 0: Do nothing

# Sentiment analysis mode:
#   'global': use TextBlob sentiment analysis to analyse the whole comment. 'local': perform at the sentence level
SENTIMENT_ANALYSIS_MODE = 'local'  # m


######################################
# FUNCTIONS

# Build a aspect graph from inpData.
# @params: threads - discussion threads. Structure array of {id: "id", :"content", supports:[]}.
#           merging_mode:
#               - 0: NO merging. Just put all graphs together
#               - 1: Keyword match. unify keywords that are exactly the same into one node
#               - 2: Semantic similarity
# @return: One summarization aspect graph.
def build_sum_graph(merging_mode, dataset, build_options):
    maybe_print("Start building sum graph in mode {0}".format(merging_mode), 1)
    # Read options
    MERGE_MODE = build_options['build_mode'] if build_options['build_mode'] else 0
    SENTIMENT_ANALYSIS_MODE = build_options['sentiment_ana_mode'] if build_options['sentiment_ana_mode'] else 'global'

    if MERGE_MODE == 0:
        #  print build_mode_0(thrds)
        #  print thrds
        threads = dataset['comments']
        g = build_mode_0(threads)
        #  print g.edges()
        maybe_print("--> Graph BUILD in mode 0 completed.\n    Number of nodes: {0}\n    Number of edges: {1}"
                     .format(len(g.nodes()), len(g.edges())), 1)
        #  print "---", g.nodes()
        return g
    if MERGE_MODE == 1:
        g = build_mode_1(dataset['title'], dataset['article'], dataset['comments'])
        # print g.edges()
        #maybe_print("--> Graph BUILD in mode 1 completed.\n    Number of nodes: {0}\n    Number of edges: {1}"
        #            .format(len(g.nodes()), len(g.edges())), 1)
        # print "---", g.nodes()
        #return g
        print "j"


# Merging mode 0: Do nothing. Indeed, it just copy exactly all nodes and edges from the extracted keygraph.
# @param: Target graph G and data threads to be merge thrds
# @output: The result graph
def build_mode_0(thrds):
    rs = nx.Graph()

    for thrd in thrds:
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
def build_mode_1(title,article,thrds):
    rs = nx.Graph()

    for thrd in thrds:
        maybe_print(":: Building aspect graph for text {0}".format(thrd), 3)
        # Extract the graph
        cen_gr, sup_grs = build_thread_graph(thrd)  # Extract central and supports
        maybe_print("---- Found {0} centroid and {1} valid support(s).\n".format(1 if cen_gr else 0, len(sup_grs)),
                    3)
        # Add the nodes
        if cen_gr:
            # print "--", len(cen_gr.nodes(data=True))
            rs.add_nodes_from(cen_gr.nodes(data=True))
        if sup_grs:
            rs.add_nodes_from(flatten_list([sup_gr.nodes(data=True) for sup_gr in sup_grs]))
        # Add the edges
        if cen_gr:
            # print "--", len(cen_gr.edges(data=True))
            rs.add_edges_from(cen_gr.edges(data=True))
            # print cen_gr.edges()
        if sup_grs:
            rs.add_edges_from(flatten_list([sup_gr.edges(data=True) for sup_gr in sup_grs]))
            # print ooo
    return rs


# Build the graph with text THREAD as input. Each thread has a structure as defined in the next procedure
# @param: text thread, each has a ID, a central and some supports
# @return: central_graph, support_graphs(as an array) and dictionary for looking up
def build_thread_graph(thrd):
    maybe_print("- Building keygraph for thread " + thrd['id'], 1)
    thread_id = thrd['id']
    central = thrd['central']
    supports = thrd['supports']
    maybe_print("-------\nAnalyzing thread {0} with {1} central and {2} support(s)"
                .format(thread_id, "ONE" if central else "NO", len(supports)), 2)
    # Build graph for central text
    central_gr = None
    if central:
        central_gr = build_graph_from_text(central, thread_id, '0')

    # Build graphs for support texts
    supports_gr = [build_graph_from_text(supports[i], thread_id, i) for i in xrange(0, len(supports))]
    # print supports_gr[0].edges()
    return central_gr, [sup for sup in supports_gr if sup]


# Build a graph from a text ---- Initial implementation
def build_graph_from_text(txt, threadid='_', comment_id='_'):
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
        except:
            pass

        if n_ent > 0:
            named_entities = [' '.join(list(item)) for item in pg_text.entities if len(item) > 1]

        tb_text = TextBlob(sen)  # Convert to textblob. # find noun phrases in text.noun_phrases
        noun_phrases = tb_text.noun_phrases

        raw_text = sen
        for item in (named_entities+noun_phrases): # group the words in noun phrase / NE into one big word
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
        assigned_nodes = [('{0}~{1}~{2}~{3}'.format(filtered_words[i], threadid, comment_id, gen_mcs_only()),
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

            except KeyError: # New node
                g.node[node[0]]['weight'] = 1
                g.node[node[0]]['thread_id'] = threadid
                g.node[node[0]]['sentiment'] = {'pos_count': 1 if sen_score > 0 else 0, # Add sentiment score
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

    maybe_print('Nodes ' + str(g.nodes()), 2)
    maybe_print('Edges ' + str(g.edges()) + '\n', 2)
    return g


# Build a graph from a text ---- Directional implementation
# This function build a directed graph fom a piece of text
def build_directed_graph_from_text(txt, threadid='_', comment_id='_'):
    maybe_print(u"Generating directed graph for text: {0}".format(txt), 3)
    sentences = sent_tokenize(txt.strip())  # sentence segmentation
    g = nx.Graph()
    # Get nodes and edges from sentence
    sen_count = 0
    sen_scores = []
    # Perform sentiment analysis for GLOBAL mode
    if SENTIMENT_ANALYSIS_MODE == 'global':
        sen_scores = [sid.polarity_scores(txt)['compound'] for _ in xrange(0, len(sentences))]
    for sen in sentences:
        dependencies,keys,new_sen = dep_extract_from_sent(sen,dep_opt)
        # Perform sentiment analysis for LOCAL mode
        sen_score = 0
        if SENTIMENT_ANALYSIS_MODE == 'local':
            sen_score = sid.polarity_scores(new_sen)['compound']
            sen_scores.append(sen_score)

        # Assign id and label to the nodes before adding the graph
        assigned_nodes = [('{0}~{1}~{2}~{3}'.format(keys[i], threadid, comment_id, gen_mcs_only()),
                           {'label': keys[i]}) for i in xrange(0, len(keys))]
        # print '---____----',assigned_nodes
        g.add_nodes_from(assigned_nodes)  # Add nodes from filtered words
        # print "wewewew ", g.nodes()
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
                g.node[node[0]]['thread_id'] = threadid
                g.node[node[0]]['sentiment'] = {'pos_count': 1 if sen_score > 0 else 0, # Add sentiment score
                                                'neg_count': 1 if sen_score < 0 else 0,
                                                'neu_count': 1 if sen_score == 0 else 0}
        maybe_print('Sentence no ' + str(sen_count) + '\nNodes ' + str(g.nodes()), 3)
        sen_count += 1
        word2id= dict(zip(keys, assigned_nodes))
        filtered_edges = [(word2id[s][0], word2id[t][0], {'label': r}) for (s, _), r, (t, _) in dependencies if s != t]
        #  print list(edges)
        #  print '-----',filtered_edges
        if filtered_edges:
            g.add_edges_from(filtered_edges)  # Add edges from the combination of words co-occurred in the same sentence
            # Update edges's weight
            for u, v, _ in filtered_edges:
                try:
                    g.edge[u][v]['weight'] += 1
                except KeyError:
                    g.edge[u][v]['weight'] = 1
            maybe_print('Edges ' + str(g.edges()) + '\n', 3)
        sen_count += 1  # Increase the sentence count index
    if len(g.nodes()) == 0:
        # raise ValueError("Generated graph is empty")
        return None

    maybe_print('Nodes ' + str(g.nodes()), 2)
    maybe_print('Edges ' + str(g.edges()) + '\n', 2)
    #print g.edges()
    return g


# Read a data file, cluster threads of discussion
# @param: path to the data file
# @return: data structure: [
#   [{'id':'cluster_id', 'central_comment': 'abc', 'supports':[support comments]},
#                       {}]
def read_comment_file(data_file):
    dataset = None
    try:
        with codecs.open(data_file, "rb", "utf8") as comment_file:
            reader = unicode_csv_reader(comment_file, delimiter='	', quotechar='"')
            dataset = []
            index = -1;
            count = 0;
            for entry in reader:
                #  print entry;
                if entry[2] == u"":
                    cluster_id = gen_mcs() + "^" + str(index+1);
                    dataset.append({'id': cluster_id, 'central': text_preprocessing(entry[6]), 'supports': []})
                    index += 1
                else:
                    if count > 0:
                        dataset[index]['supports'].append(text_preprocessing(entry[6]))
                count += 1
    except IOError:
        print "Unable to open {0}".format(data_file)

    maybe_print(json.dumps(dataset, sort_keys=True, indent=4, separators=(',', ': ')), 2)
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
        with codecs.open(data_file, "rb", "utf8") as article_file:
            count = 0
            article = []
            for line in article_file:
                if count != 0:
                    if line:
                        sens = sent_tokenize(line.replace("\r",""))
                        if sens:
                            article.extend(sens)
                else:  # the title line
                    title = line
                count += 1
    except IOError:
        print "Unable to open {0}".format(data_file)
    return title, article


# Segmentize the sentences
# @param: a list, each element is a sentence the document
# @return: a list, each element is group of sentence concatenated to form a paragraph.
def texttiling_tokenize(sentence_list):
    doc = ' '.join(sentence_list)
    tt = texttiling.TextTilingTokenizer()
    segmented_text = tt.tokenize(doc)
    return [simple_normalize(para.strip()) for para in segmented_text if para.strip()]


#  Read unicode csv file, ignore unencodable characters
def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]


# Support function for reading unicode characters
def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8', 'ignore')


# Generate the code representing current time in second-microsecond
def gen_mcs():
    return hex(int(datetime.now().strftime("%s%f")))


# Generate the identical code representing current micro-second
def gen_mcs_only():
    return hex(int(datetime.now().strftime("%s")))


# Generate the JSON code from the networkx graph structure and perform node coloring
def generate_json_from_graph(G):
    result = {'nodes': [],
              'edges': []
              }
    if not G or len(G.nodes()) == 0:
        return None

    for node in G.nodes():
        # print G.node[node]
        # Go one by one
        item = dict()
        item['id'] = node
        if G.node[node]['weight']:
            w = G.node[node]['weight']
            item['value'] = w
            item['title'] = "*Freq: " + str(w) \
                            + " <br> *Sen_Score: " + str(round(G.node[node]['sentiment_score'], 4)) \
                            + " <br> *Sentiment: " + json.dumps(G.node[node]['sentiment'])

        if G.node[node]['label']:
            item['label'] = G.node[node]['label']
        if G.node[node]['color']:
            item['color'] = str(G.node[node]['color'])
        #  print item
        result['nodes'].append(item)

    for edge in G.edges(data=True):
        item = dict()
        item['id'] = edge[0]+'|'+edge[1]
        #  print "SSSSS",type(G.edges())
        if G.edge[edge[0]][edge[1]]['weight']:
            w = G.edge[edge[0]][edge[1]]['weight']
            item['value'] = w
            item['title'] = "freq: " + str(w)
        #  if G.edge[edge[0]][edge[1]]['label']:
        #    item['label'] = G.edge[edge[0]][edge[1]]['label']
        item['from'] = edge[0]
        item['to'] = edge[1]
        item['label'] = edge[2]['label']
        result['edges'].append(item)

    return result


# Pruning the graph according to restrictions in options
def prune_graph(graph, options):
    g = graph
    # Load options
    if not g or not options:
        return None
    # Set default values
    ENABLE_PRUNING = False
    NUM_MIN_NODES = 20  # minimum number of node. If total number of nodes is  < this number, pruning will be skipped
    NUM_MIN_EDGES = 30  # minimum number of edge. The value could not more than (NUM_MIN_NODES)*(NUM_MIN_NODES-1).
    #                     If total number of edge is  < this number, pruning will be skipped
    REMOVE_ISOLATED_NODE = True #

    #  For keyword matching and unification
    ENABLE_UNIFY_MATCHED_KEYWORDS = True
    INTRA_CLUSTER_UNIFY = True
    INTER_CLUSTER_UNIFY = False
    UNIFY_MODE = 'link' # Modes"# 'link': create a virtual link between the nodes
                                # 'contract': Direct contract, sum weight

    NUM_MAX_NODES = 200  # maximum number of nodes to keep
    NUM_MAX_EDGES = 300  # maximum number of edge to keep.
    #                      The value could not more than (NUM_MAX_NODES)*(NUM_MAX_NODES-1)
    EDGE_FREQ_MIN = 1  # minimum frequency that an edge is required to be. Being smaller, it will be eliminated.
    NODE_FREQ_MIN = 1  # minimum frequency that a node is required to be. Being smaller, it will be eliminated.
    NODE_DEGREE_MIN = 1  # minimum degree that a node is required to have. Being smaller, it will be eliminated.
    MIN_WORD_LENGTH = 3  # Minimum number of character of a word, accepted to enter the graph


    # Now read the options
    ENABLE_PRUNING = options['enable_pruning']
    if not ENABLE_PRUNING:
        return None # Skip the pruning

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

    if 'unify_matched_keywords' in options:
        try:
            uop = options['unify_matched_keywords']
            ENABLE_UNIFY_MATCHED_KEYWORDS = uop['enable'] if uop['enable'] else False
            INTRA_CLUSTER_UNIFY = uop['intra_cluster_unify'] if uop['intra_cluster_unify'] else False
            INTER_CLUSTER_UNIFY = uop['inter_cluster_unify'] if uop['inter_cluster_unify'] else False
            UNIFY_MODE = uop['unification_mode'] if uop['unification_mode'] else 'link'
        except:
            raise ValueError, "Error while parsing matched words unification options."

    maybe_print("Start pruning the graph.", 1)
    # :: Perform pruning
    # Decide whether to skip the pruning because of the tiny size of graph
    if len(g.nodes()) < NUM_MIN_NODES or len(g.edges()) < NUM_MIN_EDGES:
        return None # Skip the pruning

    # Remove short words
    to_remove_nodes = []
    to_remove_edges = []
    for node in g.nodes():
        if len(g.node[node]['label']) < MIN_WORD_LENGTH:
            to_remove_nodes.append(node)
    for edge in g.edges():
        if (edge[0] in to_remove_nodes) or (edge[1] in to_remove_nodes):
            to_remove_edges.append(edge)
    g.remove_nodes_from(to_remove_nodes)
    g.remove_edges_from(to_remove_edges)

    # Merge sub graph with similar keyword -> Now just
    if ENABLE_UNIFY_MATCHED_KEYWORDS:
        matches = set([])
        node_ids = g.nodes()
        for i in xrange(0,len(node_ids)-1):
            for j in xrange(i+1,len(node_ids)):
                if g.node[node_ids[i]]['label'] == g.node[node_ids[j]]['label']: # two node have the same label
                    matches.add((node_ids[i],node_ids[j]))
        matches = list(matches)
        if matches:
            # Do the unification. INTRA unification must be carried out before INTER unification. Because the scope of
            # the latter cover the former one
            # Implementation of INTRA cluster unification
            if INTRA_CLUSTER_UNIFY: # Unify the same  keywords in one cluster
                if UNIFY_MODE == 'link':
                    max_score,_ = get_max_value_attribute(g, 'weight') # Get the max weight of all nodes in graph
                    print max_score
                    # Now make the links
                    print matches
                    for node0,node1 in matches:
                        if g.node[node0]['thread_id'] == g.node[node1]['thread_id']: # unify if in the same thread
                            g.add_edge(node0, node1, {'weight': max_score, 'id': node0+'|'+node1})
                elif UNIFY_MODE == 'contract':
                    t_g = g
                    while len(matches)> 0:
                        node0 = matches[0][0]
                        node1 = matches[0][1]
                        if g.node[node0]['thread_id'] == g.node[node1]['thread_id']:
                            # Sum up the weight of the two node
                            sum_weight = g.node[node0]['weight'] + g.node[node1]['weight']
                            t_g = nx.contracted_nodes(t_g,node0,node1)
                            t_g.node[node0]['weight'] = sum_weight
                            t_g.node[node0]['label'] = g.node[node0]['label']
                            # Sum up the sentiment of the two nodes
                            pos_count = g.node[node0]['sentiment']['pos_count'] \
                                        + g.node[node1]['sentiment']['pos_count']
                            neg_count = g.node[node0]['sentiment']['neg_count'] \
                                        + g.node[node1]['sentiment']['neg_count']
                            neu_count = g.node[node0]['sentiment']['neu_count'] \
                                        + g.node[node1]['sentiment']['neu_count']
                            t_g.node[node0]['sentiment'] = {'pos_count': pos_count,
                                                            'neg_count': neg_count,
                                                            'neu_count': neu_count
                                                            }
                            # Update the match lst
                            for i in xrange(0, len(matches)): # now node1 disappear, the unified node holds node0 id
                                m = node0 if matches[i][0] == node1 else matches[i][0]
                                n = node0 if matches[i][1] == node1 else matches[i][1]
                                matches[i] = (m, n)
                        matches.pop(0)  # Remove first element
                    g = t_g

            # Implementation of INTER cluster unification
            if INTER_CLUSTER_UNIFY:
                if UNIFY_MODE == 'link':
                    max_score,_ = get_max_value_attribute(g, 'weight') # Get the max weight of all nodes in graph
                    print max_score
                    # Now make the links
                    print matches
                    for node0,node1 in matches:
                        g.add_edge(node0, node1, {'weight': max_score, 'id': node0+'|'+node1})
                elif UNIFY_MODE == 'contract':
                    t_g = g
                    while len(matches) > 0:
                        node0 = matches[0][0]
                        node1 = matches[0][1]
                        sum_weight = g.node[node0]['weight'] + g.node[node1]['weight']
                        t_g = nx.contracted_nodes(t_g, node0, node1)
                        t_g.node[node0]['weight'] = sum_weight
                        t_g.node[node0]['label'] = g.node[node0]['label']
                        # Sum up the weight of the two node
                        sum_weight = g.node[node0]['weight'] + g.node[node1]['weight']
                        t_g = nx.contracted_nodes(t_g, node0, node1)
                        t_g.node[node0]['weight'] = sum_weight
                        t_g.node[node0]['label'] = g.node[node0]['label']
                        # Sum up the sentiment of the two nodes
                        pos_count = g.node[node0]['sentiment']['pos_count'] \
                                    + g.node[node1]['sentiment']['pos_count']
                        neg_count = g.node[node0]['sentiment']['neg_count'] \
                                    + g.node[node1]['sentiment']['neg_count']
                        neu_count = g.node[node0]['sentiment']['neu_count'] \
                                    + g.node[node1]['sentiment']['neu_count']
                        t_g.node[node0]['sentiment'] = {'pos_count': pos_count,
                                                        'neg_count': neg_count,
                                                        'neu_count': neu_count
                                                        }
                        # Update the match lst
                        matches.pop(0) # Remove first element
                        for i in xrange(0, len(matches)): # since now node1 disappear, the unified node holds node0 id
                            m = node0 if matches[i][0] == node1 else matches[i][0]
                            n = node0 if matches[i][1] == node1 else matches[i][1]
                            matches[i] = (m, n)
                    g = t_g

    # Filter nodes by frequency
    to_remove_nodes = []
    if NODE_FREQ_MIN > 1:
        to_remove_nodes = [n for n in g.nodes() if g.node[n]['weight']< NODE_FREQ_MIN]
        g.remove_nodes_from(to_remove_nodes)

    if EDGE_FREQ_MIN > 1:
        to_remove_edges = [(s, t) for (s, t) in g.edges() if g.edge[s][t]['weight'] < EDGE_FREQ_MIN]
        g.remove_edges_from(to_remove_edges)

    # Filter nodes by degree
    if NODE_DEGREE_MIN > 1:
        to_remove_nodes = [n for n in g.nodes() if g.degree(n) < NODE_DEGREE_MIN]
        g.remove_nodes_from(to_remove_nodes)

    # Remove isolated nodes
    to_remove_nodes = []
    to_remove_edges = []
    if REMOVE_ISOLATED_NODE:
        degrees = nx.degree(g)
        to_remove_nodes = [i for i in degrees if degrees[i] == 0]
        for edge in g.edges():
            if (edge[0] in to_remove_nodes) or (edge[1] in to_remove_nodes):
                to_remove_edges.append(edge)
        g.remove_nodes_from(to_remove_nodes)
        g.remove_edges_from(to_remove_edges)

    # :: Done pruning
    maybe_print("--> Graph PRUNNING completed.\n    Number of nodes: {0}\n    Number of edges: {1}"
                .format(len(g.nodes()), len(g.edges())))

    return g


# Compute polarity of sentiment of the graph G. This score is compute as (positive_score-negative_score)/max_freq
# where max_freq is the normalization constant, and max_freq is the maximum frequency of all nodes on the graph
# @param: graph g
# @return: new graph with sentiment_score attached
def compute_sentiment_score(g):
    # get the normalization constant
    max_val,_ = get_max_value_attribute(g,'weight')
    #print max_val
    tg_g = g # copy the graph
    for n in g.nodes():
        #print g.node[n]['sentiment']['pos_count'],g.node[n]['sentiment']['pos_count']
        tg_g.node[n]['sentiment_score'] = float((g.node[n]['sentiment']['pos_count']
                                                 - g.node[n]['sentiment']['neg_count']))\
                                          /max_val
    return tg_g


# Function to perform simple rule-based normalization the text
# @param: a string, should be in unicode
# @return: the normalized string
def simple_normalize(cnt):
    reps = ('\n', ''), \
           ('            ', ''), \
           ("\n \n", ""), \
           (u"\u201c", "\""), \
           (u"\u201d", "\""), \
           (u"\u2019", "\'"), \
           (u"\u2018", "\'"), \
           (u"\u2013", "-"), \
           (u"\u2014", "-"), \
           (u"\u2011", "-")
    return reduce(lambda a, kv: a.replace(*kv), reps, cnt)


# Extract a graph from a sentence
# @param: a sentence, and filtering options
# @output: 1. a list of dependencies 2. a list of keys, 3. the sentence after grouped compounds/entities
def dep_extract_from_sent(sentence,filter_opt):
    blob = TextBlob(sentence)
    #print blob.noun_phrases
    for phrase in blob.noun_phrases:
        sentence = sentence.replace(phrase,phrase.replace(' ','_'))
    #print sentence
    result = dep_parser.raw_parse(sentence)
    dependencies = result.next()
    raw_results = list(dependencies.triples())
    # print('Options: ',filter_opt)
    # Filter out by POS
    preferred_pos = filter_opt['preferred_pos']
    if type(preferred_pos) != list:  # take all POS
        filter_pos_result = raw_results
    else:
        # filter triples whose beginning and ending tags are inside the list
        #print raw_results
        filter_pos_result = [trip for trip in raw_results
                             if (trip[0][1] in preferred_pos and trip[2][1] in preferred_pos)
                             or len(trip[0][0])>10 or len(trip[2][0])>10]  # keep potential phrases

    # Filter by relationship
    # print filter_pos_result
    prefered_rel = filter_opt['preferred_rel']
    if type(prefered_rel) != list:  # take all POS
        filter_rel_result =  filter_pos_result
    else:
        # filter tripples that has relation in the list
        filter_rel_result = [trip for trip in raw_results if (trip[1] in prefered_rel)]

    # print filter_rel_result
    # Merge compounds
    compound_merge = filter_opt['compound_merge']
    if compound_merge:
        tokens = word_tokenize(sentence)
        compounds = [(t,s) for (s,s_tag),r,(t,t_tag) in filter_rel_result if r == u'compound' and s_tag == t_tag]
        # print "!!!!!!!!!", compounds
        replacements = dict()
        for i in xrange(0, len(tokens)-1):
            if (tokens[i],tokens[i+1]) in compounds:
                replacements[tokens[i]] = tokens[i]+"_" + tokens[i+1]
                replacements[tokens[i+1]] = tokens[i] + "_" + tokens[i + 1]
                # Now do the replace
                # print "found compound", tokens[i],tokens[i+1]
        for i in xrange(0,len(filter_rel_result)):
            (s,s_tag),r,(t,t_tag) = filter_rel_result[i]
            if replacements.has_key(s):
                filter_rel_result[i] = (replacements[s],s_tag),r,(t,t_tag)
            (s, s_tag), r, (t, t_tag) = filter_rel_result[i]  # Update in case 1st element changes
            if replacements.has_key(t):
                filter_rel_result[i] = (s,s_tag),r,(replacements[t],t_tag)
        new_sen = sentence  # new sentence contains grouped item with _ as connector
        for key in replacements:
            new_sen.replace(key,replacements[key])
        # now remove duplicate
        filter_comp_result = [((s,s_tag),r,(t,t_tag)) for (s,s_tag),r,(t,t_tag) in filter_rel_result if
                                (s != t)]
    else:
        filter_comp_result = filter_rel_result
    keys = set()
    for (s,_),_,(t,_) in filter_comp_result:
        keys.add(t)
        keys.add(s)

    return filter_comp_result,list(keys),new_sen
