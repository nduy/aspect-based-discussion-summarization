build_options = {
    'build_mode': 1,
    'sentiment_ana_mode': 'global',  # 'global', 'local'
    'use_thread_structure': False,   # if yes, the thread structure of comments will be used. Otherwise just treat them
                                     # equally
    'n_thread': 5  # number of thread for Multithreading
}

prune_options = {
    'enable_pruning': True,
    'min_word_length': 2,
    'remove_isolated_node': True,
    'node_freq_min': 1,
    'edge_freq_min': 1,
    'node_degree_min': 2
}

uni_options = {
    'unify_matched_keywords': {
        'enable': True,
        'intra_cluster_unify': True,
        'inter_cluster_unify': True,
        'unification_mode': 'contract'  # modes: link, contract
    }
}

dep_opt = {
    # preferred_pos': ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'preferred_pos': ['NN', 'NNS', 'NNP', 'NNPS'],
    'preferred_rel': 'all',  # ['nsubk','nsubkpass','obj','iobj'] list of relation to remains
    'compound_merge': True

}

# Verbality: to print or not to print ################################################################################
script_verbality = 2     # 0: silent, 1: print main info, 2: print some techs info, 3. print debugging info

