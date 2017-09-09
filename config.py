# General graph building setup
build_options = {
    'build_mode': 1,
    'sentiment_ana_mode': 'global',  # 'global', 'local'
    'use_thread_structure': False,   # if yes, the thread structure of comments will be used. Otherwise just treat them
                                     # equally
    'n_thread': 5  # number of thread for Multithreading
}

# Options for prunning the graph
prune_options = {
    'enable_prunning': True,            # Enable/Disable prunning
    'min_word_length': 2,               # Minimum length of node label
    'remove_isolated_node': True,       # Remove not whose degree is 0
    'node_freq_min': 2,                 # Minimum frequency of node
    'edge_freq_min': 1,                 # Minimum frequency of edge
    'node_degree_min': 2,               # Minimum degree of node, this override the remove_isolated_node
    'regex_pattern': '[a-z\/]+[-_]?',   # Regular expression pattern to keep
    # A while list of words to keep no mater how bad it is
    'white_node_labels': [],
    # A black list of words to be killed no mater how good it is
    'black_node_labels': [u"wa",u"a",u"about",u"above",u"across",u"after",u"afterwards",u"again",u"against",u"all",u"almost",u"alone",u"along",u"already",u"also",u"although",u"always",u"am",u"among",u"amongst",u"amoungst",u"an",u"and",u"another",u"any",u"anyhow",u"anyone",u"anything",u"anyway",u"anywhere",u"are",u"around",u"as",u"at",u"be",u"became",u"because",u"been",u"before",u"beforehand",u"behind",u"being",u"below",u"beside",u"besides",u"between",u"beyond",u"both",u"but",u"by",u"can",u"cannot",u"could",u"dare",u"despite",u"did",u"do",u"does",u"done",u"down",u"during",u"each",u"eg",u"either",u"else",u"elsewhere",u"enough",u"etc",u"even",u"ever",u"every",u"everyone",u"everything",u"everywhere",u"except",u"few",u"first",u"for",u"former",u"formerly",u"from",u"further",u"furthermore",u"had",u"has",u"have",u"he",u"hence",u"her",u"here",u"hereabouts",u"hereafter",u"hereby",u"herein",u"hereinafter",u"heretofore",u"hereunder",u"hereupon",u"herewith",u"hers",u"herself",u"him",u"himself",u"his",u"how",u"however",u"i",u"ie",u"if",u"in",u"indeed",u"inside",u"instead",u"into",u"is",u"it",u"its",u"itself",u"last",u"latter",u"latterly",u"least",u"less",u"lot",u"lots",u"many",u"may",u"me",u"meanwhile",u"might",u"mine",u"more",u"moreover",u"most",u"mostly",u"much",u"must",u"my",u"myself",u"namely",u"near",u"need",u"neither",u"never",u"nevertheless",u"next",u"no",u"nobody",u"none",u"noone",u"nor",u"not",u"nothing",u"now",u"nowhere",u"of",u"off",u"often",u"oftentimes",u"on",u"once",u"one",u"only",u"onto",u"or",u"other",u"others",u"otherwise",u"ought",u"our",u"ours",u"ourselves",u"out",u"outside",u"over",u"per",u"perhaps",u"rather",u"re",u"same",u"second",u"several",u"shall",u"she",u"should",u"since",u"so",u"some",u"somehow",u"someone",u"something",u"sometime",u"sometimes",u"somewhat",u"somewhere",u"still",u"such",u"than",u"that",u"the",u"their",u"theirs",u"them",u"themselves",u"then",u"thence",u"there",u"thereabouts",u"thereafter",u"thereby",u"therefore",u"therein",u"thereof",u"thereon",u"thereupon",u"these",u"they",u"third",u"this",u"those",u"though",u"through",u"throughout",u"thru",u"thus",u"to",u"together",u"too",u"top",u"toward",u"towards",u"under",u"until",u"up",u"upon",u"us",u"used",u"very",u"via",u"was",u"we",u"well",u"were",u"what",u"whatever",u"when",u"whence",u"whenever",u"where",u"whereafter",u"whereas",u"whereby",u"wherein",u"whereupon",u"wherever",u"whether",u"which",u"while",u"whither",u"who",u"whoever",u"whole",u"whom",u"whose",u"why",u"whyever",u"will",u"with",u"within",u"without",u"would",u"yes",u"yet",u"you",u"your",u"yours",u"yourself",u"yourselves"]

}

# options for dependency parsing.
dep_opt = {
    # preferred_pos': ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'preferred_pos': [u'NN', u'NNS', u'NNP', u'NNPS', u'JJ', u'VB', u'VBD', u'VBG', u'VBN', u'VBP', u'VBZ'],
    'preferred_rel': 'all',  # ['nsubk','nsubkpass','obj','iobj'] list of relation to remains
    'compound_merge': True,  # Merge
    # for each SENTENCE, we group those nodes  whose satisfy the pattern: from_pos --rel_name--> to_pos
    # IMPORTANT! The contractions is carried out after POS filtering. Hence, defining rules with POS tag outside
    # preferred_pos resulting in disregard
    'custom_nodes_contract': {
        'enable': True,
        'rule_set': [
            {'from_pos': u'NN', 'to_pos': u'JJ', 'rel_name': u'amod'}
        ],
    },
    # for each SENTENCE, we group those edges whose satisfy the pattern: <--rel_name1--> n_pos <--rel_name2-->
    # A declaration for a rule come up with the following format:
    #       {rel_name1:'', 'rel_direction1':'' ,rel_name2:'','rel_direction1':'', 'n_pos': '', 'label': 's.text'}
    # where rel_name is the name of relations. rel_direction is the direction of the arrow: 'left' or 'right'
    #       "n_pos": POS tag of the middle node
    #       "rs_label" is the name of the relationship. It could be a fixed value, or a pattern. 7 known tokens are
    # {rel_name1}, {rel_direction1}, {rel_name2}, {rel_direction2}, {n_pos}, {n_pos}, {n_label} (label of the node)
    # E.g: "{n_label}"
    #       "rs_direction" is the direction of result graph
    'custom_edges_contract': {
        'enable': True,
        'rule_set': [
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VB', 'rs_label': u'{n_label}', 'rs_direction': u'left-to-right'}, # <-nsub-[VBZ]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBD', 'rs_label': u'{n_label}', 'rs_direction': u'left-to-right'}, # <-nsub-[VBD]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBG', 'rs_label': u'{n_label}', 'rs_direction': u'left-to-right'},  # <-nsub-[VBZ]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBN', 'rs_label': u'{n_label}', 'rs_direction': u'left-to-right'},  # <-nsub-[VBD]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBP', 'rs_label': u'{n_label}', 'rs_direction': u'left-to-right'},  # <-nsub-[VBZ]-dobj->
            {'rel_name1': u'nsubj', 'rel_direction1': u'out', 'rel_name2': u'dobj', 'rel_direction2': u'out',
             'n_pos': u'VBZ', 'rs_label': u'{n_label}', 'rs_direction': u'left-to-right'}  # <-nsub-[VBD]-dobj->
        ]
    }

}

# Verbality: to print or not to print ################################################################################
script_verbality = 2     # 0: silent, 1: print main info, 2: print some techs info, 3. print debugging info

uni_options = {
    'unify_matched_keywords': {
        'enable': True,
        'intra_cluster_unify': True,
        'inter_cluster_unify': True,
        'unification_mode': 'contract'  # modes: link, contract
    }
}
