# -*- coding: utf-8 -*-
#    Copyright (C) 2017
#    All rights reserved.
#    BSD license.
#    Author: Ferran Parés <ferran.pares@bsc.es>
#    Forked from https://github.com/networkx/networkx/blob/master/networkx/algorithms/community/asyn_fluidc.py
"""Asynchronous Fluid Communities algorithm for community detection."""

from collections import Counter
import random
from networkx.exception import NetworkXError
from networkx.algorithms.components import is_connected
from networkx.utils import groups
from networkx.utils.decorators import not_implemented_for
from networkx import pagerank
from utils import maybe_print

__all__ = ['asyn_fluidc']


@not_implemented_for('directed', 'multigraph')
def asyn_fluidc(G, k, max_iter=100, enable_pr=True):
    """Returns communities in `G` as detected by Fluid Communities algorithm.

    The asynchronous fluid communities algorithm is described in
    [1]_. The algorithm is based on the simple idea of fluids interacting
    in an environment, expanding and pushing each other. It's initialization is
    random, so found communities may vary on different executions.

    The algorithm proceeds as follows. First each of the initial k communities
    is initialized in a random vertex in the graph. Then the algorithm iterates
    over all vertices in a random order, updating the community of each vertex
    based on its own community and the communities of its neighbours. This
    process is performed several times until convergence.
    At all times, each community has a total density of 1, which is equally
    distributed among the vertices it contains. If a vertex changes of
    community, vertex densities of affected communities are adjusted
    immediately. When a complete iteration over all vertices is done, such that
    no vertex changes the community it belongs to, the algorithm has converged
    and returns.

    This is the original version of the algorithm described in [1]_.
    Unfortunately, it does not support weighted graphs yet.

    Parameters
    ----------
    G : Graph

    k : integer
        The number of communities to be found.

    max_iter : integer
        The number of maximum iterations allowed. By default 15.

    enable_pr : Enable/disable Pagerank for initialize starting points

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    k variable is not an optional argument.

    References
    ----------
    .. [1] Parés F., Garcia-Gasulla D. et al. "Fluid Communities: A
       Competitive and Highly Scalable Community Detection Algorithm".
       [https://arxiv.org/pdf/1703.09307.pdf].
    """
    # Initial checks
    if not isinstance(k, int):
        raise NetworkXError("k must be an integer.")
    if not k > 0:
        raise NetworkXError("k must be greater than 0.")
    if not is_connected(G):
        raise NetworkXError("Fluid Communities can only be run on connected\
        Graphs.")
    if len(G) < k:
        raise NetworkXError("k must be greater than graph size.")
    # Initialization
    max_density = 1.0
    vertices = list(G)
    random.shuffle(vertices)
    # print "@@@",vertices
    if enable_pr:
        # Run PageRank with alpha of 0.9 the push them to the head of vertices
        #  so that it will be understand as start points
        maybe_print("PageRanks: {0}".format(pagerank(G)), 2, u'i')
        # Find the top k  keys by page rank: run pr, sort the value, then get top k key
        top_keys = [word_id for word_id,_ in list(sorted(pagerank(G).items(), key=lambda x:x[1], reverse=True))]
        # random.shuffle(top_keys[:(len(top_keys))/4])
        random.shuffle(top_keys[:(k*2)])
        top_keys = top_keys[:k]
        maybe_print("Top keys: {0}".format(top_keys), 2, u'i')
        # print "+++", top_keys
        # Remove these top keys from the vertices, then append top_key to the head
        top_keys.extend([v for v in vertices if v not in top_keys])
        # print "XXX", vertices

    communities = {n: i for i, n in enumerate(vertices[:k])}
    density = {}
    com_to_numvertices = {}
    for vertex in communities.keys():
        com_to_numvertices[communities[vertex]] = 1
        density[communities[vertex]] = max_density
    # Set up control variables and start iterating
    iter_count = 0
    cont = True
    while cont:
        cont = False
        iter_count += 1
        # Loop over all vertices in graph in a random order
        vertices = list(G)
        random.shuffle(vertices)
        for vertex in vertices:
            # Updating rule
            com_counter = Counter()
            # Take into account self vertex community
            try:
                com_counter.update({communities[vertex]:
                                    density[communities[vertex]]})
            except KeyError:
                pass
            # Gather neighbour vertex communities
            for v in G[vertex]:
                try:
                    com_counter.update({communities[v]:
                                        density[communities[v]]})
                except KeyError:
                    continue
            # Check which is the community with highest density
            new_com = -1
            if len(com_counter.keys()) > 0:
                max_freq = max(com_counter.values())
                best_communities = [com for com, freq in com_counter.items()
                                    if (max_freq - freq) < 0.0001]
                # If actual vertex com in best communities, it is preserved
                try:
                    if communities[vertex] in best_communities:
                        new_com = communities[vertex]
                except KeyError:
                    pass
                # If vertex community changes...
                if new_com == -1:
                    # Set flag of non-convergence
                    cont = True
                    # Randomly chose a new community from candidates
                    new_com = random.choice(best_communities)
                    # Update previous community status
                    try:
                        com_to_numvertices[communities[vertex]] -= 1
                        density[communities[vertex]] = max_density / \
                            com_to_numvertices[communities[vertex]]
                    except KeyError:
                        pass
                    # Update new community status
                    communities[vertex] = new_com
                    com_to_numvertices[communities[vertex]] += 1
                    density[communities[vertex]] = max_density / \
                        com_to_numvertices[communities[vertex]]
        # If maximum iterations reached --> output actual results
        if iter_count > max_iter:
            break
    # Return results by grouping communities as list of vertices
    return iter(groups(communities).values())
