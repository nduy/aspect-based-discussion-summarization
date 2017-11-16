#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Mon Jun 19 10:51:42 2017
    CONFIGURATION
"""

# General graph building setup
build_options = {
    'build_mode': 1,
    'sentiment_ana_mode': 'local',  # 'global', 'local'
    'use_thread_structure': False,   # if yes, the thread structure of comments will be used. Otherwise just treat them
                                     # equally
    'n_thread': 10  # number of thread for Multithreading
}

# Options for prunning the graph
prune_options = {
    'enable_prunning': True,            # Enable/Disable prunning
    'min_word_length': 3,               # Minimum length of node label
    'remove_isolated_node': True,       # Remove not whose degree is 0
    'node_freq_min': 2,                 # Minimum frequency of node
    'edge_freq_min': 2,                 # Minimum frequency of edge
    'node_degree_min': 2,               # Minimum degree of node, this override the remove_isolated_node
    'remove_rings': True,               # Remove edges that connect a node to itself
    'regex_pattern': r'^[a-z]{2,}([-_\/]{1}[a-z]{2,})*$',   # Regular expression pattern to keep
    # A while list of words to keep no mater how bad it is
    'white_node_labels': [],
    # A black list of words to be killed no mater how good it is
    'black_node_labels': [u"week",u"month",u"year",u"amount",u"bit",u"wa",u"think",u"look",u"new",u"know",
                          u"a",u"plenty",u"many",u"use",u"able",u"sort",u"little",u"most",u"used",u"do",u"make",
                          u"about",u"above",u"across",u"after",u"afterwards",u"again",u"against",u"all",u"almost",
                          u"alone",u"along",u"already",u"also",u"although",u"always",u"am",u"among",u"amongst",
                          u"amongst",u"an",u"and",u"another",u"any",u"anyhow",u"anyone",u"anything",u"anyway",
                          u"anywhere",u"are",u"around",u"as",u"at",u"be",u"became",u"because",u"been",u"before",
                          u"beforehand",u"behind",u"being",u"below",u"beside",u"besides",u"between",u"beyond",u"both",
                          u"but",u"by",u"can",u"cannot",u"could",u"dare",u"despite",u"did",u"do",u"does",u"done",
                          u"down",u"during",u"each",u"eg",u"either",u"else",u"elsewhere",u"enough",u"etc",u"even",
                          u"ever",u"every",u"everyone",u"everything",u"everywhere",u"except",u"few",u"first",u"for",
                          u"former",u"formerly",u"from",u"further",u"furthermore",u"had",u"has",u"have",u"he",u"hence",
                          u"her",u"here",u"hereabouts",u"hereafter",u"hereby",u"herein",u"hereinafter",u"heretofore",
                          u"hereunder",u"hereupon",u"herewith",u"hers",u"herself",u"him",u"himself",u"his",u"how",
                          u"however",u"i",u"ie",u"if",u"in",u"indeed",u"inside",u"instead",u"into",u"is",u"it",u"its",
                          u"itself",u"last",u"latter",u"latterly",u"least",u"less",u"lot",u"lots",u"many",u"may",u"me",
                          u"meanwhile",u"might",u"mine",u"more",u"moreover",u"most",u"mostly",u"much",u"must",u"my",
                          u"myself",u"namely",u"near",u"need",u"neither",u"never",u"nevertheless",u"next",u"no",
                          u"nobody",u"none",u"noone",u"nor",u"not",u"nothing",u"now",u"nowhere",u"of",u"off",u"often",
                          u"oftentimes",u"on",u"once",u"one",u"only",u"onto",u"or",u"other",u"others",u"otherwise",
                          u"ought",u"our",u"ours",u"ourselves",u"out",u"outside",u"over",u"per",u"perhaps",u"rather",
                          u"re",u"same",u"second",u"several",u"shall",u"she",u"should",u"since",u"so",u"some",
                          u"somehow",u"someone",u"something",u"sometime",u"sometimes",u"somewhat",u"somewhere",
                          u"still",u"such",u"than",u"that",u"the",u"their",u"theirs",u"them",u"themselves",u"then",
                          u"thence",u"there",u"thereabouts",u"thereafter",u"thereby",u"therefore",u"therein",u"thereof",
                          u"thereon",u"thereupon",u"these",u"they",u"third",u"this",u"those",u"though",u"through",
                          u"throughout",u"thru",u"thus",u"to",u"together",u"too",u"top",u"toward",u"towards",u"under",
                          u"until",u"up",u"upon",u"us",u"used",u"very",u"via",u"was",u"we",u"well",u"were",u"what",
                          u"whatever",u"when",u"whence",u"whenever",u"where",u"whereafter",u"whereas",u"whereby",
                          u"wherein",u"whereupon",u"wherever",u"whether",u"which",u"while",u"whither",u"who",u"whoever",
                          u"whole",u"whom",u"whose",u"why",u"whyever",u"will",u"with",u"within",u"without",u"would",
                          u"yes",u"yet",u"you",u"your",u"yours",u"yourself",u"yourselves"],
# All edges with this dependency will be removed
    'black_dependencies': [u'advcl',u'dep',u'parataxis',u'det',u'acl',u'case'],
    'black_pos': [u'VB',u'VBD',u'VBG',u'VBN',u'VBZ',u'VBP',u'JJ'],  # All node with this POS wil be removed
    'min_edge_similarity': 0.15          # Minimum semantic similarity ins distributed semantic vector space
}

community_detect_options = {
    'enable_community_detection': True,             # Enable/Disable community detection,
    'method': {
                    'algorithm': 'fluid_communities',     # other options 'bipartitions', Label propagation
                    'params':{
                        'n_communities': 10,
                        'enable_pagerank_initialization': True
                    }


               },
    'community_label_inference': {   # method for inferring the label
        'method': 'distributed_semantic',
        'params': {
            'window': 3,
            'weight_ls': [1., .7, .4],
            # 'weighted_average', 'average' (special kind of average), 'vec_sum'
            'composition_method': 'weighted_average'
        }
    }
}

# Verbality: to print or not to print ################################################################################
script_verbality = 2     # 0: silent, 1: print main info, 2: print some techs info, 3. print debugging info

uni_options = {
    'unify_matched_keywords': {         # Merely merge keywords that are completely matched together
        'enable': True,
        'intra_cluster_unify': True,
        'inter_cluster_unify': True,
        'unification_mode': 'contract'  # modes: link, contract
    },
    'unify_semantic_similarity': {
        'enable': True,
        'threshold': 0.85,   # Those nodes whose similarity greater than this threshold will be unified
        'glove_model_file': '../models/glove.6B.200d.txt'
    }
}


# options for dependency parsing.
dep_opt = {
    # preferred_pos': ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'preferred_pos': [u'NN', u'NNS', u'NNP', u'NNPS', u'JJ', u'VB', u'VBD', u'VBG', u'VBN', u'VBP', u'VBZ'],
    'preferred_rel': 'all',  # ['nsubk','nsubkpass','obj','iobj'] list of relation to remains
    'compound_merge': True,  # Merge
    # for each SENTENCE, we group those nodes  whose satisfy the pattern: from_pos --rel_name--> to_pos
    # params:
    #  - from_pos: POS of the starting node
    #  - to_pos: POS of the ending node
    #  - rs_pos: designated POS of the result node
    #  - rs_direction: designated direction of the result node '1-2' or '2-1'
    'custom_nodes_contract': {
        'enable': True,
        'rule_set': [
            {'from_pos': u'NN', 'to_pos': u'JJ', 'rel_name': u'amod', 'rs_pos': u'NN', 'rs_direction': u'2-1'},
            {'from_pos': u'NNS', 'to_pos': u'JJ', 'rel_name': u'amod', 'rs_pos': u'NNS', 'rs_direction': u'2-1'},
            {'from_pos': u'NNS', 'to_pos': u'NN', 'rel_name': u'amod', 'rs_pos': u'NNS', 'rs_direction': u'2-1'},
            {'from_pos': u'CD', 'to_pos': u'CD', 'rel_name': u'compound', 'rs_pos': u'CD', 'rs_direction': u'2-1'},
            {'from_pos': u'NNS', 'to_pos': u'CD', 'rel_name': u'nummod', 'rs_pos': u'NNS', 'rs_direction': u'2-1'},
            {'from_pos': u'NN', 'to_pos': u'CD', 'rel_name': u'nummod', 'rs_pos': u'NN', 'rs_direction': u'2-1'},
            {'from_pos': u'NN', 'to_pos': u'NN', 'rel_name': u'compound', 'rs_pos': u'NN', 'rs_direction': u'2-1'},
            {'from_pos': u'NN', 'to_pos': u'NNP', 'rel_name': u'compound', 'rs_pos': u'NNP', 'rs_direction': u'2-1'},
        ],
    },
    # for each SENTENCE, we group those edges whose satisfy the pattern: <--rel_name1--> n_pos <--rel_name2-->
    # A declaration for a rule come up with the following format:
    #       {rel_name1:'', 'rel_direction1':'' ,rel_name2:'','rel_direction1':'', 'n_pos': '', 'label': 's.text'}
    # where rel_name is the name of relations. rel_direction is the direction of the arrow: 'left' or 'right'
    #       "n_pos": POS tag of the middle node
    #       "rs_label" is the name of the relationship. It could be a fixed value, or a pattern. 7 known tokens are
    # {rel_name1}, {rel_direction1}, {rel_name2}, {rel_direction2}, {n_pos}, {n_pos}, {n_label} (label of the node)
    # E.g: {n_label}: (central node label), {l_label}: left node label, {r_label}: right node label
    #       "rs_direction" is the direction of result graph
    #       "nodes_label": what label to be taken. For example: [1]->[2]<-[3]:
    #           - "1-2" result in [1]->[2], "1-3" results in [1]->[3]
    'custom_edges_contract': {
        'enable': True,
        'rule_set': [
            # ------------------ DOBJ family
            # <-nsub-[VBZ]-dobj-> 'rs_direction': u'left-to-right'
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VB', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBD', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBZ]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBG', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBN', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBZ]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBP', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBZ', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # ------------------ ACL family
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'acl', 'rel_direction2': u'in',
             'n_pos': u'VB', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]<-acl-
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'acl', 'rel_direction2': u'in',
             'n_pos': u'VBD', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBZ]<-acl-
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'acl', 'rel_direction2': u'in',
             'n_pos': u'VBG', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]<-acl-
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'acl', 'rel_direction2': u'in',
             'n_pos': u'VBN', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBZ]<-acl-
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'acl', 'rel_direction2': u'in',
             'n_pos': u'VBP', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]<-acl-
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'acl', 'rel_direction2': u'in',
             'n_pos': u'VBZ', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # ------------------ CCOMP family
            # <-nsub-[VBZ]-ccomp-> 'rs_direction': u'left-to-right'
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'ccomp', 'rel_direction2': u'out',
             'n_pos': u'VB', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-ccomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'ccomp', 'rel_direction2': u'out',
             'n_pos': u'VBD', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBZ]-ccomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'ccomp', 'rel_direction2': u'out',
             'n_pos': u'VBG', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-ccomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'ccomp', 'rel_direction2': u'out',
             'n_pos': u'VBN', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBZ]-ccomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'ccomp', 'rel_direction2': u'out',
             'n_pos': u'VBP', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-ccomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'ccomp', 'rel_direction2': u'out',
             'n_pos': u'VBZ', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # ------------------ XCOMP family
            # <-nsub-[VBZ]-xcomp-> 'rs_direction': u'left-to-right'
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'xcomp', 'rel_direction2': u'out',
             'n_pos': u'VB', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-xcomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'xcomp', 'rel_direction2': u'out',
             'n_pos': u'VBD', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBZ]-xcomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'xcomp', 'rel_direction2': u'out',
             'n_pos': u'VBG', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-xcomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'xcomp', 'rel_direction2': u'out',
             'n_pos': u'VBN', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBZ]-xcomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'xcomp', 'rel_direction2': u'out',
             'n_pos': u'VBP', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # <-nsub-[VBD]-xcomp->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'xcomp', 'rel_direction2': u'out',
             'n_pos': u'VBZ', 'rs_label': u'{n_label}', 'nodes_label': u'1-3'},
            # ------------------ COP family
            # <-nsub-[VBZ]-cop-> 'rs_direction': u'left-to-right'
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'cop', 'rel_direction2': u'out',
             'n_pos': u'VB', 'rs_label': u'{r_label}', 'nodes_label': u'1-2'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'cop', 'rel_direction2': u'out',
             'n_pos': u'VBD', 'rs_label': u'{r_label}', 'nodes_label': u'1-2'},
            # <-nsub-[VBZ]-cop->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'cop', 'rel_direction2': u'out',
             'n_pos': u'VBG', 'rs_label': u'{r_label}', 'nodes_label': u'1-2'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'cop', 'rel_direction2': u'out',
             'n_pos': u'VBN', 'rs_label': u'{r_label}', 'nodes_label': u'1-2'},
            # <-nsub-[VBZ]-cop->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'cop', 'rel_direction2': u'out',
             'n_pos': u'VBP', 'rs_label': u'{r_label}', 'nodes_label': u'1-2'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'cop', 'rel_direction2': u'out',
             'n_pos': u'VBZ', 'rs_label': u'{r_label}', 'nodes_label': u'1-2'},
            # ------------------ NSUBJPASS family
            # <-nsub-[VBZ]-cop-> 'rs_direction': u'left-to-right'
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'nmod', 'rel_direction2': u'out',
             'n_pos': u'VB', 'rs_label': u'{n_label}', 'nodes_label': u'3-1'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'nmod', 'rel_direction2': u'out',
             'n_pos': u'VBD', 'rs_label': u'{n_label}', 'nodes_label': u'3-1'},
            # <-nsub-[VBZ]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'nmod', 'rel_direction2': u'out',
             'n_pos': u'VBG', 'rs_label': u'{n_label}', 'nodes_label': u'3-1'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'nmod', 'rel_direction2': u'out',
             'n_pos': u'VBN', 'rs_label': u'{n_label}', 'nodes_label': u'3-1'},
            # <-nsub-[VBZ]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'nmod', 'rel_direction2': u'out',
             'n_pos': u'VBP', 'rs_label': u'{n_label}', 'nodes_label': u'3-1'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'nmod', 'rel_direction2': u'out',
             'n_pos': u'VBZ', 'rs_label': u'{n_label}', 'nodes_label': u'3-1'},
            # <-nsub-[VBZ]-auxpass-> 'rs_direction': u'left-to-right'
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'auxpass', 'rel_direction2': u'out',
             'n_pos': u'VB', 'rs_label': u'{n_label}', 'nodes_label': u'2-1'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'auxpass', 'rel_direction2': u'out',
             'n_pos': u'VBD', 'rs_label': u'{r_label}', 'nodes_label': u'2-1'},
            # <-nsub-[VBZ]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'auxpass', 'rel_direction2': u'out',
             'n_pos': u'VBG', 'rs_label': u'{r_label}', 'nodes_label': u'2-1'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'auxpass', 'rel_direction2': u'out',
             'n_pos': u'VBN', 'rs_label': u'{r_label}', 'nodes_label': u'2-1'},
            # <-nsub-[VBZ]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'auxpass', 'rel_direction2': u'out',
             'n_pos': u'VBP', 'rs_label': u'{r_label}', 'nodes_label': u'2-1'},
            # <-nsub-[VBD]-cop->
            {'rel_name1': u'nsubjpass', 'rel_direction1': u'out', 'rel_name2': u'auxpass', 'rel_direction2': u'out',
             'n_pos': u'VBZ', 'rs_label': u'{r_label}', 'nodes_label': u'2-1'},
        ]
    }

}

# Options for making predictive model
model_build_options = {
    # method for computing centrality of node in graph
    'centrality_method': 'eigenvector',  # options: "pagerank", "degree", "closeness", "eigenvector"
    'normalization_method': 'sum'  # Normalization method: softmax or sum
}


replace_pattern = {u"<blockquote>":u"",u"</blockquote>":u"",u"$":u"DOLLAR",u"¢":u"CENT",u"£":u"POUND",u"¤":u"CURRENCY",u"¥":u"YEN",u"֏":u"ARMENIAN_DRAM",u"؋":u"AFGHANI",u"৳":u"BENGALI_RUPEE",u"૱":u"GUJARATI_RUPEE",u"௹":u"TAMIL_RUPEE",u"฿":u"THAI_CURRENCY_SYMBOL_BAHT",u"៛":u"KHMER_CURRENCY_SYMBOL_RIEL",u"₠":u"EURO-CURRENCY",u"₡":u"COLON",u"₢":u"CRUZEIRO",u"₣":u"FRENCH_FRANC",u"₤":u"LIRA",u"₥":u"MILL",u"₦":u"NAIRA",u"₧":u"PESETA",u"₨":u"RUPEE",u"₩":u"WON",u"₪":u"NEW_SHEQEL",u"₫":u"DONG",u"€":u"EURO",u"₭":u"KIP",u"₮":u"TUGRIK",u"₯":u"DRACHMA",u"₰":u"GERMAN_PENNY",u"₱":u"PESO",u"₲":u"GUARANI",u"₳":u"AUSTRAL",u"₴":u"HRYVNIA",u"₵":u"CEDI",u"₶":u"LIVRE_TOURNOIS",u"₷":u"SPESMILO",u"₸":u"TENGE",u"₹":u"INDIAN_RUPEE",u"₺":u"TURKISH_LIRA",u"₻":u"NORDIC_MARK",u"₼":u"MANAT",u"元":u"CJK_UNIFIED_IDEOGRAPH-5143",u"圓":u"CJK_UNIFIED_IDEOGRAPH-5713",u"﷼":u"RIAL",u"﹩":u"SMALL_DOLLAR",u"＄":u"FULLWIDTH_DOLLAR",u"￠":u"FULLWIDTH_CENT",u"￡":u"FULLWIDTH_POUND",u"￥":u"FULLWIDTH_YEN",u"￦":u"FULLWIDTH_WON"}