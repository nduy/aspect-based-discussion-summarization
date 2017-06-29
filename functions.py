#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
 Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    FUNCTIONS FOR TEXT SUMMARIZATION
"""

######################################
# IMPORT LIBRARY
import csv, re, codecs, json
from cucco import Cucco
from datetime import datetime
import networkx as nx
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from itertools import combinations
from polyglot.text import Text
from polyglot.downloader import downloader
from nltk.stem import WordNetLemmatizer

######################################
# EXTRA DECLARATIONS

# Normalization and cleaning engine
cucco = Cucco()

# Lemmatizer
lemmatizer = WordNetLemmatizer()

normalizations = [
    'remove_accent_marks',
    ('replace_urls', {'replacement': ''}),
    ('replace_emails', {'replacement': ''}),
    'remove_extra_whitespaces',
]

stopwords = set(stopwords.words('english')) # Add more stopword to ignore her


# Verbality: to print or not to print ################################################################################
script_verbality= 1     # 0: silent, 1: print sometime, 2: print everything

preredTags = set(['NOUN','PROPN']);

MERGE_MODE = 0  # 0: Do nothing

# Sentiment analysis mode:
#   'global': use TextBlob sentiment analysis to analyse the whole sentence. The result polarity therefore stand for
SENTIMENT_ANALYSIS_MODE = 'global'

######################################
# FUNCTIONS

# Build a aspect graph from inpData.
# @params: thrds - discussion threads. Structure array of {id: "id", :"content", supports:[]}.
#           mergining_mode:
#               - 0: NO merging. Just put all graphs together
#               - 1: Keyword match. unify keywords that are exactly the same into one node
#               - 2: Semantic similarity
# @return: One summarization aspect graph.
def build_sum_graph(merging_mode,thrds,build_options):
    #g = nx.Graph()
    maybe_print("Start building sum graph in mode {0}".format(merging_mode), 1)
    # Read options
    MERGE_MODE = build_options['merge_mode'] if build_options['merge_mode'] else 0
    SENTIMENT_ANALYSIS_MODE = build_options['sentiment_ana_mode'] if build_options['sentiment_ana_mode'] else 'global'

    if MERGE_MODE == 0:
        # print merge_mode_0(thrds)
        #print thrds
        g = merge_mode_0(thrds)
        #print g.edges()
        maybe_print("--> Graph BUILD completed.\n    Number of nodes: {0}\n    Number of edges: {1}"
                     .format(len(g.nodes()), len(g.edges())), 1)
        #print "zzzz", g.nodes()
        return g

# Merging mode 0: Do nothing. Indeed, it just copy exactly all nodes and edges from the extracted keygraph.
# @param: Target graph G and data threads to be merge thrds
# @output: The result graph
def merge_mode_0(thrds):
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



# Build the graph with text THREAD as input. Each thread has a structure as defined in the next procedure
# @param: text thread, each has a ID, a central and some supports
# @return: central_graph, support_graphs(as an array) and dictionary for looking up
def build_thread_graph(thrd):
    maybe_print("- Buliding keygraph for thread "+ thrd['id'],1)
    thread_id = thrd['id']
    central = thrd['central']
    supports = thrd['supports']
    maybe_print("-------\nAnalyzing thread {0} with {1} central and {2} support(s)"
                .format(thread_id, "ONE" if central else "NO", len(supports)), 2)
    # Build graph for central text
    central_gr = None;
    if central:
        central_gr = build_graph_from_text(central, thread_id, '0');

    # Build graphs for support texts
    supports_gr = [build_graph_from_text(supports[i], thread_id,i) for i in xrange(0,len(supports))]
    #print supports_gr[0].edges()
    return central_gr, [sup for sup in supports_gr if sup]


# Build a graph from a text
def build_graph_from_text(txt,threadid='_',comment_id='_'):
    maybe_print(u"Generating graph for text: {0}".format(txt), 3)
    sentences = sent_tokenize(txt.strip())  # sentence segmentation
    G = nx.Graph()
    # Get nodes and edges from sentence
    sen_count = 0;
    for sen in sentences:
        #print sen
        # Extract named-entities
        named_entities = []
        n_ent = 0
        pg_text = Text(sen)# convert to pyglot
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
        preferred_words = [lemmatizer.lemmatize(w.lower()) for w, t in text.pos_tags if t in preredTags]
        # Filter stopwords
        #filtered_words = list(set([w for w in preferred_words if w not in stopwords]))
        filtered_words = [w for w in preferred_words if w not in stopwords]
        # do sentiment analysis
        sen_scores = None;
        if SENTIMENT_ANALYSIS_MODE == 'global':
            blob = TextBlob(raw_text)
            sen_scores = [blob.sentiment.polarity for _ in filtered_words]

        # Assign id and label to the nodes before adding the graph
        assigned_nodes = [('{0}~{1}~{2}~{3}'.format(filtered_words[i], threadid, comment_id, gen_mcs_only()),
                           {'label': filtered_words[i]}) for i in xrange(0, len(filtered_words))]
        #print '---____----',assigned_nodes
        G.add_nodes_from(assigned_nodes)  # Add nodes from filtered words
        # Update nodes's weight
        for node in assigned_nodes:
            try:
                G.node[node[0]]['weight'] += 1
            except KeyError:
                G.node[node[0]]['weight'] = 1
                G.node[node[0]]['thread_id'] = threadid

        maybe_print('Sentence no ' + str(sen_count) + '\nNodes ' + str(G.nodes()), 3)
        sen_count +=1
        edges = combinations([i[0] for i in assigned_nodes], 2)
        filtered_edges = [(n,m) for n,m in edges if n.split('~')[0] != m.split('~')[0]]
        #print list(edges)
        #print 'xxxafsdgfsa',filtered_edges
        if filtered_edges:
            G.add_edges_from(filtered_edges)  # Add edges from the combination of words co-occurred in the same sentence
            # Update edges's weight
            for u, v in filtered_edges:
                try:
                    G.edge[u][v]['weight'] += 1
                except KeyError:
                    G.edge[u][v]['weight'] = 1
            maybe_print('Edges ' + str(G.edges()) + '\n', 3)
        sen_count +=1  # Increase the sentence count index
    if len(G.nodes()) == 0:
        return None

    maybe_print('Nodes ' + str(G.nodes()), 2)
    maybe_print('Edges ' + str(G.edges()) + '\n', 2)
    return G


# Read a data file, cluster threads of discussion
# @param: path to the data file
# @return: data structure: [
#   [{'id':'cluster_id', 'central_comment': 'abc', 'supports':[support comments]},
#                       {}]
def read_comment_file(dataFile):
    dataset=None;
    try:
        with codecs.open(dataFile, "rb", "utf8") as comment_file:
            reader = unicode_csv_reader(comment_file, delimiter='	', quotechar='"')
            dataset = []
            index = -1;
            count = 0;
            for entry in reader:
                #print entry;
                if entry[2] == u"":
                    cluster_id = gen_mcs() + "^" + str(index+1);
                    dataset.append({'id': cluster_id, 'central': text_preprocessing(entry[6]), 'supports': []})
                    index += 1
                else:
                    if count > 0:
                        dataset[index]['supports'].append(text_preprocessing(entry[6]))
                count += 1
    except IOError:
        print "Unable to open {0}".format(dataFile)

    maybe_print(json.dumps(dataset, sort_keys=True, indent=4, separators=(',', ': ')), 2)
    return dataset


# Clean up the text, and get rig of unnecessary characters
def text_preprocessing(rawText):
    txt = cucco.normalize(rawText, normalizations)
    txt = re.sub('<a.*>.*?</a>', '', txt)
    txt = txt.replace('<blockquote>', '').replace('</blockquote>', '')
    return txt;


# Decide to print the print text or not according to the verbality
# command_verbality:
#   0: always print
#   1: sometime print
#   2: rarely print
def maybe_print(text, command_verbality=1):
    if script_verbality > 0:
        if script_verbality >= command_verbality:
            print text;


# Read unicode csv file, ignore unencodable characters
def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]


# Support function for reading unicode charcters
def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8', 'ignore')


# Generate the code representing current time in second-microsecond
def gen_mcs():
    return hex(int(datetime.now().strftime("%s%f")))


# Generate the identical code representing current micro-second
def gen_mcs_only():
    return hex(int(datetime.now().strftime("%s")))


# Generate the JSON code from the networkx graph structure
def generate_json_from_graph(G):
    result = {'nodes': [],
              'edges': []
              }
    if not G or len(G.nodes()) == 0:
        return None

    for node in G.nodes():
        #print node
        # Go one by one
        item = {}
        item['id'] = node
        if G.node[node]['weight']:
            w = G.node[node]['weight']
            item['value'] = w
            item['title'] = "freq: " + str(w)
        if G.node[node]['label']:
            item['label'] = G.node[node]['label']
        #print item
        result["nodes"].append(item)

    for edge in G.edges():
        item = {}
        item['id'] = edge[0]+'|'+edge[1]
        #print "SSSSS",type(G.edges())
        if G.edge[edge[0]][edge[1]]['weight']:
            w = G.edge[edge[0]][edge[1]]['weight']
            item['value'] = w
            item['title'] = "freq: " + str(w)
        #if G.edge[edge[0]][edge[1]]['label']:
        #    item['label'] = G.edge[edge[0]][edge[1]]['label']
        item['from'] = edge[0]
        item['to'] = edge[1]
        result["edges"].append(item)

    return result


# flattening list of sublist into a single list
def flatten_list(l):
    if isinstance(l, list):
        lx = []
        for lz in l:
            lx.extend(map(flatten_list, lz))
        return lx
    else:
        return l


# get maximum node weight of the graph
# @param: the graph g, respect to attribute att
# @output: the integer of maximum weight AND the id of node
def get_max_value_attribute(g,att):
    #print g.nodes()
    #if not g.nodes():
    #    raise ValueError, "Can't get max weight of undefined graph"
    #return None,None

    max = -1
    id = None
    for node in g.nodes():
        try:
            val = g.node[node][att]
            if val > max:
                max = val
                id = node
        except:
            raise ValueError,"Attribute " + att + "  not found for node " + node
    return max,id


# Pruning the graph according to restrictions in options
def prun_graph(graph,options):
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

    #For keyword matching and unification
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
                            sum_weight = g.node[node0]['weight'] + g.node[node1]['weight']
                            #print node0,node1
                            t_g = nx.contracted_nodes(t_g,node0,node1)
                            t_g.node[node0]['weight'] = sum_weight
                            t_g.node[node0]['label'] = g.node[node0]['label']
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
                    while len(matches)> 0:
                        node0 = matches[0][0]
                        node1 = matches[0][1]
                        sum_weight = g.node[node0]['weight'] + g.node[node1]['weight']
                        t_g = nx.contracted_nodes(t_g, node0, node1)
                        t_g.node[node0]['weight'] = sum_weight
                        t_g.node[node0]['label'] = g.node[node0]['label']
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
        to_remove_nodes = [n for n in g.nodes() if g.node[n]['weight']<NODE_FREQ_MIN]
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


