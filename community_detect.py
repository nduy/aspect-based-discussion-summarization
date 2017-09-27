#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Sep 18, 2017
    PERFORM COMMUNITY DETECTION FROM A GRAPH
"""

import networkx as nx
from networkx.algorithms import community
from utils import *
import functions
import config
from glove import Glove
import numpy as np
import traceback
import random
from nltk.corpus import wordnet as wn
import en
from nltk.stem import WordNetLemmatizer

# Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

sample_community_names = [u'α',u'β',u'γ',u'δ',u'ε',u'ζ',u'η',u'θ',u'ι',u'κ',u'λ',u'μ',
                           u'ν',u'ξ',u'ο',u'π',u'ρ',u'σ',u'τ',u'υ',u'φ',u'χ',u'ψ',u'ω']


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
        LABEL_DETECTION_METHOD = comm_opt['community_label_inference'] if 'community_label_inference' in comm_opt \
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
                n_com = comm_opt['method']['n_communities'] if 'n_communities' in comm_opt['method'] else 4
                gc = max(nx.connected_component_subgraphs(undir_graph), key=len)
                # list of list. Each sublist contain ID of nodes in the same community
                communities = list(community.asyn_fluidc(gc, n_com))
                com_index = -1

                for com in communities:
                    com_index += 1
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
                    words_matrix = extract_vector_from_text_list(comm_labels, model=glove_model, window=3)
                    # get indices for rows whose is zero rows
                    zeros_indices = np.where(~words_matrix.any(axis=1))[0]
                    maybe_print(' -->Community' + str(com_index) + ' has ' + str(len(zeros_indices)) + \
                          " zero key(s) out of "+ str(len(com)),2,'i')
                    # remove zero rows from words_matrix
                    words_matrix = np.delete(words_matrix, zeros_indices, axis=0)
                    # get all the weight in the community, then convert to float by multiply 1.0
                    vector_weight = np.array([graph.node[node_id]['weight'] for node_id in com]) * 1.0
                    vector_weight = np.delete(vector_weight, zeros_indices, axis=0)  # remove lines whose vector is zero
                    # Compute weights -> this is a kind of weighted sum
                    vector_weight = vector_weight/np.sum(vector_weight)  # compute ratio/scale/co-efficient, whatever :D
                    vector_weight = vector_weight.reshape((len(vector_weight), 1))  # Transpose to column vector
                    print vector_weight
                    #vector_weight = np.ones((len(com),1))
                    print words_matrix.shape, vector_weight.shape
                    assert words_matrix.shape[0] == vector_weight.shape[0], 'Mismatch size of matrix for community {0}'\
                                                                          ' with {1} members and its weight matrix.\n'\
                        .format(com_index-1, len(com))
                    # Multiple matrices and the sum te vector to be the representative vector for the community
                    composition_matrix = np.multiply(words_matrix,vector_weight)
                    # Remove zero rows and sum
                    # composition_vector = np.sum(composition_matrix[~np.all(composition_matrix == 0, axis=1)],axis=0)
                    composition_vector = np.sum(composition_matrix, axis=0)
                    print composition_vector
                    # Dig to vector space of Glove to get the label
                    dst = (np.dot(glove_model.word_vectors, composition_vector)
                           / np.linalg.norm(glove_model.word_vectors, axis=1)
                           / np.linalg.norm(composition_vector))
                    word_ids = np.argsort(-dst)
                    # Get 2 most similar words @@@@@
                    suggested_labels = [glove_model.inverse_dictionary[x] for x in word_ids[:10]
                                        if x in glove_model.inverse_dictionary]
                    # suggested_labels = glove_model.most_similar_paragraph(comm_labels)
                    for node_id in com: #sample_community_names[com_index]
                        graph.node[node_id]['cluster_id'] = sample_community_names[com_index] + ' ' \
                                                            + (' - '.join(suggested_labels)).upper()

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
def extract_vector_from_text(inp_str,model,window=3):
    PREFERED_POS = set([u'n',u'v'])
    if not inp_str:
        raise ValueError('Invalid keyword for vector extraction:' + inp_str)

    raw_ws = None
    if inp_str in model.dictionary:
        raw_ws = [inp_str]
    else:
        raw_ws = [lemmatizer.lemmatize(w.strip()) for w in re.findall(r"[a-z]+", inp_str)]

    # filtering by part-of-speech
    ws = []
    for w in raw_ws:
        related_pos = set([syn.pos() for syn in wn.synsets(w)])
        print w, related_pos
        if w in model.dictionary and len(PREFERED_POS & related_pos) > 0:
            ws.append(w)

    if not ws:  # Non word is in semantic space
        maybe_print('None of the word in string "{0}" is not in the dictionary. Return zero vector.'
                    .format(inp_str), 2, 'E')
        random_key = random.choice(model.dictionary.keys())
        random_vector = (model.word_vectors[model.dictionary[random_key]])
        # print random_vector.shape
        return np.zeros(random_vector.shape[0])
    vector = np.array(model.word_vectors[model.dictionary[ws[-1]]])
    if len(ws) > window:
        for i in xrange(1,window):
            vector = vector + np.array(model.word_vectors[model.dictionary[ws[-1*i]]])
        return vector
    else:
        for i in xrange(1,len(ws)):
            vector = vector + np.array(model.word_vectors[model.dictionary[ws[-1*i]]])
        return vector


# Apply sentence extract for each element in a list inp_str_list, then stack them to form a matrix. For parameter detail
# see the above function
def extract_vector_from_text_list(inp_ls,model,window=3):
    rs = None
    if not inp_ls:
        raise ValueError("Nothing to extract vector.")
    count = 0
    for inp_srt in inp_ls:
        count += 1
        if count == 1:
            rs = np.array(extract_vector_from_text(inp_srt,model,window))
        else:
            rs = np.vstack((rs,np.array(extract_vector_from_text(inp_srt,model,window))))
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
