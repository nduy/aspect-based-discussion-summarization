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
    'enable_pruning': True,
    'min_word_length': 2,
    'remove_isolated_node': True,
    'node_freq_min': 1,
    'edge_freq_min': 1,
    'node_degree_min': 2
}

# options for dependency parsing.
dep_opt = {
    # preferred_pos': ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'preferred_pos': [u'NN', u'NNS', u'NNP', u'NNPS', u'JJ'],
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
            {'rel_name1': u'nsubj', 'rel_name1': u'left', 'rel_name2': u'dobj', 'rel_name1': u'right',
             'n_pos': u'BVZ', 'rs_label': u'{n_label}', 'rs_direction': u'right'} # <-nsub-[VBZ]-dobj->
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
