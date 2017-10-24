#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Sep 18, 2017
    PERFORM COMMUNITY DETECTION FROM A GRAPH
"""

import networkx as nx
from utils import *
import functions
import config
from glove import Glove
import numpy as np
import traceback
import random
from nltk.corpus import wordnet as wn
import DbpediaLabeller
from nltk.stem import WordNetLemmatizer
from sklearn.svm import OneClassSVM
import en
from asyn_fluidc import asyn_fluidc

# Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

sample_community_names = [u'α',u'β',u'γ',u'δ',u'ε',u'ζ',u'η',u'θ',u'ι',u'κ',u'λ',u'μ',
                           u'ν',u'ξ',u'ο',u'π',u'ρ',u'σ',u'τ',u'υ',u'φ',u'χ',u'ψ',u'ω']

# Outliner rate: how many nodes do you want to treat as outliner
outliers_fraction = 0.15


# Detect the communities in a graph
def detect_communities(g=None, comm_opt=None):
    maybe_print(" Detecting communities.", 2, 'i')
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
        LABEL_DETECTION_METHOD = comm_opt['community_label_inference']['method'] \
            if 'community_label_inference' in comm_opt and 'method' in comm_opt['community_label_inference']\
            else 'distributed_semantic'

    # Convert directed graph to undirected graph
    undir_graph = graph.to_undirected()

    if not undir_graph:
        raise ValueError("Unable to perform community detection! Perhaps due to the malformed graph.")
    if ENABLE_DETECTION:
        # Load model for  inferring cluster name using Glove
        glove_model = functions.glove_model
        try:
            if not glove_model:
                GLOVE_MODEL_FILE = config.uni_options['unify_semantic_similarity']['glove_model_file']
                maybe_print("   + Glove model is undefined. Trying to load from " + GLOVE_MODEL_FILE, 2, "i")
                glove_model = Glove.load_stanford(GLOVE_MODEL_FILE)
                maybe_print("   + Model loading completed :)", 2)
        except Exception as inst:
            maybe_print("   + Error while detecting group names. Check whether the Glove model was correctly loaded.", 2,
                        "E")
            print(inst)
        # Run algorithm
        try:
            if ALGORITHM == "fluid_communities":
                # get the largest messy graph
                # Get number of communities to be detected
                n_com = comm_opt['method']['params']['n_communities'] \
                    if 'n_communities' in comm_opt['method']['params'] else 4
                enable_pagerank = comm_opt['method']['params']['enable_pagerank_initialization'] \
                    if 'enable_pagerank_initialization' in comm_opt['method']['params'] else 4

                gc = max(nx.connected_component_subgraphs(undir_graph), key=len)
                # list of list. Each sublist contain ID of nodes in the same community
                communities = list(asyn_fluidc(gc, n_com,enable_pr=enable_pagerank))
                com_index = -1
                for com in communities:
                    com_index += 1
                    # SVM One class classifier for outlier detection.
                    clf = OneClassSVM(nu=0.90 * outliers_fraction + 0.05, kernel="poly", gamma=0.03, degree=3)
                    #####################
                    # How this work? the program compute weight sum over the vector of all member of the communities who
                    # DO EXIST in the glove vector space. The scale factor is the ratio between the node's frequency
                    # (under 'weight' attribute) and the sum of weights of all keywords that DO EXIST in the vector spac
                    # -e. Those who many not exist will be disregarded.
                    # It first extract vector representation of each member of the community. Those
                    # keywords whose are successfully extracted (exist in glove vector space) has NonZero vector. Then
                    # weights are computed for the original keywords of these NonZero vectors.
                    # Suggest a label for the community
                    comm_labels = [graph.node[node_id]['label'] for node_id in com]
                    comm_labels_array = np.array(comm_labels)
                    # Now run abstraction by different method
                    suggested_labels = None
                    if LABEL_DETECTION_METHOD == 'distributed_semantic':
                        WINDOW = comm_opt['community_label_inference']['params']['window'] \
                            if 'window' in comm_opt['community_label_inference']['params'] else 3
                        V_WEIGHTS = comm_opt['community_label_inference']['params']['weight_ls'] \
                            if 'weight_ls' in comm_opt['community_label_inference']['params'] else 3
                        COMPOSITION_METHOD = comm_opt['community_label_inference']['params']['composition_method'] \
                            if 'composition_method' in comm_opt['community_label_inference']['params'] else 3
                        words_matrix = extract_vector_from_text_list(comm_labels,
                                                                     model=glove_model,
                                                                     window=WINDOW,
                                                                     vector_weights=V_WEIGHTS)
                        # get indices for rows whose is zero rows
                        zeros_indices = np.where(~words_matrix.any(axis=1))[0]
                        maybe_print(' --> Community ' + str(com_index) + ' has ' + str(len(zeros_indices))
                                    + " zero key(s) out of "+ str(len(com)),2,'i')
                        # remove zero rows from words_matrix
                        words_matrix = np.delete(words_matrix, zeros_indices, axis=0)
                        # remaining labels
                        comm_labels_array = np.delete(comm_labels_array,zeros_indices)
                        maybe_print(' --> Remaining labels: {0}'.format(', '.join(comm_labels_array)))

                        # get all the weight in the community, then convert to float by multiply 1.0
                        # Compute vector weight according to composition method
                        vector_weight = None
                        if COMPOSITION_METHOD == 'weighted_average':
                            vector_weight = np.array([graph.node[n]['weight'] for n in com]) * 1.0
                            vector_weight = np.delete(vector_weight, zeros_indices, axis=0)  # remove zero rows
                            # Compute weights -> this is a kind of weighted sum
                            vector_weight = vector_weight/np.sum(vector_weight)  # compute scale/co-efficient, whatever :D
                            vector_weight = vector_weight.reshape((len(vector_weight), 1))  # Transpose to column vector
                        elif COMPOSITION_METHOD == 'average':
                            n_row = len(com) - len(zeros_indices)
                            vector_weight = np.full((n_row,1),1.0/n_row,dtype=np.float)
                        elif COMPOSITION_METHOD == 'vec_sum':
                            vector_weight = np.ones((len(com) - len(zeros_indices), 1))
                        else:
                            raise ValueError('Invalid vector composition method')
                        # print words_matrix.shape, vector_weight.shape
                        assert words_matrix.shape[0] == vector_weight.shape[0], \
                            'Mismatch size of matrix for community {0}  with {1} members and its weight matrix.\n'\
                            .format(com_index-1, len(com))
                        # Multiple matrices and the sum te vector to be the representative vector for the community
                        # composition_matrix = np.multiply(words_matrix,vector_weight)
                        # Remove outliers
                        #clf.fit(X=composition_matrix)  # fit the model
                        print words_matrix.shape, vector_weight.flatten().shape
                        clf.fit(X=words_matrix,sample_weight=vector_weight.flatten())  # fit the model
                        # predict with the model. The outcome is an array, each element is the predicted value of
                        # the word/row. It can be 1 (inlier) or -1 (outlier)
                        # y_pred = clf.predict(composition_matrix)
                        y_pred = clf.predict(words_matrix)
                        print y_pred

                        # Weighted AVERAGE composition
                        composition_matrix = np.multiply(words_matrix, vector_weight)
                        # Now filter inliner only
                        filtered_composition_vector = composition_matrix[np.where(y_pred == 1)]
                        #filtered_composition_vector = words_matrix[np.where(y_pred == 1)]
                        # Remove predicted outlier
                        comm_labels_array = np.delete(comm_labels_array, np.where(y_pred == -1))
                        maybe_print('  --> Outlier removal discarded {0} words. Remaining words: {1}'
                                    .format(len(np.where(y_pred == -1)[0]), str(comm_labels_array)))
                        # Sum the matrix by row to form one vector
                        composition_vector = np.sum(filtered_composition_vector, axis=0)
                        # print composition_vector
                        # Dig to vector space of Glove to get the label
                        dst = (np.dot(glove_model.word_vectors, composition_vector)
                               / np.linalg.norm(glove_model.word_vectors, axis=1)
                               / np.linalg.norm(composition_vector))
                        word_ids = np.argsort(-dst)
                        # Get 2 most similar words @@@@@
                        raw_suggested_labels = [glove_model.inverse_dictionary[x] for x in word_ids[:50]
                                                if x in glove_model.inverse_dictionary]
                        suggested_labels = []
                        # Filter result by POS
                        for w in raw_suggested_labels:
                            if len(w)>2:
                                related_pos = set([syn.pos() for syn in wn.synsets(w)])
                                if related_pos and len(set([u'v',u'a',u's',u'r']) & related_pos) == 0:  # Filter: exclude some pos
                                    suggested_labels.append(w)
                        # Get 3 most frequent word
                    freqs = [w for w,_ in sorted([(g.node[n]['label'],g.node[n]['weight']) for n in com],
                                                 key=lambda e: int(e[1]),reverse=True)]
                    # suggested_labels = glove_model.most_similar_paragraph(comm_labels)
                    if len(suggested_labels) >5:
                        suggested_labels = suggested_labels[:5]

                    # Apply DBPedia Labeler
                    # top10 =[subword for word in freqs[:5] for subword in word.split('_') if en.is_noun(subword)] + freqs[:5]
                    top10 = freqs[:10]
                    print "---> ",top10
                    # DB_labels = DbpediaLabeller.DBPprocess(top10)
                    DB_labels = DbpediaLabeller.DBPprocess(top10)
                    # print 'ZZZZZZZZZzzzzz',comm_labels_array
                    # DB_labels = DbpediaLabeller.DBPprocess(comm_labels_array)
                    print DB_labels
                    if len(DB_labels) >5:
                        DB_labels = DB_labels[:5]
                    for node_id in com:  # sample_community_names[com_index]
                        graph.node[node_id]['cluster_id'] = u'[{0}] Top: {1} \nV.Comp: {2} \nDbpedia: {3}'\
                                                                .format(sample_community_names[com_index],
                                                                        ', '.join(freqs[:5]),
                                                                        ' - '.join(suggested_labels).upper(),
                                                                        ' - '.join(DB_labels).upper())

                return graph
        except Exception as inst:
            maybe_print(" Error while running algorithm {0} to detect communities. Error name: {1}. \n"
                        "Perhaps incorrect algorithm name of parameters. Community detection is skipped and community "
                        "label for all nodes is set to be \'unknown\'.".format(ALGORITHM,inst.message), 2, 'E')
            traceback.print_exc()
            return g
    else:
        return g


# Extract from a text, form abc_def_xyz to form a vector. This vector is a sum vector in a distributed semantic space
# under variable model. The window define how many words should we use for sum up, starting from the END of sentence
# What is vector weight? Each words in phrase has different importance. For example in "red car", car is more important
# than red (which is a modifier). ATTENTION: the weight's indices are REVERED order of the window. That is, the fist
# weight is for the last word of the inp_str and so on.
def extract_vector_from_text(inp_str,model,window=3,vector_weights = [1.0,.5,.3]):
    PREFERED_POS = set([u'n'])          # Filtering: the word must be able to play this role
    BLACK_POS = set([u'a'])   # Filtering: the word with these tag will be excluded
    if not inp_str:
        raise ValueError('Invalid keyword for vector extraction:' + inp_str)
    if len(vector_weights) != window:
        raise ValueError('While extracting vector from text, size of the weight list {0} '
                         'must be the same as the window {1}.'.format(len(window),window))
    raw_ws = None
    if inp_str in model.dictionary:
        raw_ws = [inp_str]
    else:
        raw_ws = [lemmatizer.lemmatize(w.strip()) for w in re.findall(r"[a-z]+", inp_str)]

    # filtering by part-of-speech
    ws = []
    for w in raw_ws:
        related_pos = set([syn.pos() for syn in wn.synsets(w)])
        # print w, related_pos
        if w in model.dictionary and (related_pos and len(PREFERED_POS & related_pos) > 0 and len(BLACK_POS & related_pos) == 0):  # Filter: must include preferred pos
            ws.append(w)

    if not ws:  # Non word is in semantic space
        maybe_print('None of the word in string "{0}" is in the dictionary or has invalid POS: {1}. Return zero vector.'
                    .format(inp_str,', '.join(related_pos)), 2, 'W')
        random_key = random.choice(model.dictionary.keys())
        random_vector = (model.word_vectors[model.dictionary[random_key]])
        # print random_vector.shape
        return np.zeros(random_vector.shape[0])
    vector = np.array(model.word_vectors[model.dictionary[ws[-1]]])
    if len(ws) > window:
        for i in xrange(1,window):
            vector = vector + np.array(model.word_vectors[model.dictionary[ws[-1*i]]]) * vector_weights[i-1]
        return vector
    else:
        for i in xrange(1,len(ws)):
            vector = vector + np.array(model.word_vectors[model.dictionary[ws[-1*i]]]) * vector_weights[i-1]
        return vector


# Apply sentence extract for each element in a list inp_str_list, then stack them to form a matrix. For parameter detail
# see the above function
def extract_vector_from_text_list(inp_ls,model,window=3,vector_weights= [1.0,.5,.3]):
    rs = None
    if not inp_ls:
        raise ValueError("Nothing to extract vector.")
    if len(vector_weights) != window:
        raise ValueError('While extracting vector from text, size of the weight list {0} '
                         'must be the same as the window {1}.'.format(len(window),window))
    count = 0
    for inp_srt in inp_ls:
        count += 1
        if count == 1:
            rs = np.array(extract_vector_from_text(inp_srt,model,window,vector_weights))
        else:
            rs = np.vstack((rs,np.array(extract_vector_from_text(inp_srt,model,window,vector_weights))))
    return rs
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
