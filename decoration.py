#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Sat Jul  1 00:26:06 2017
    DECORATION FOR GRAPH
    - Graph coloring before exporting json

"""

import networkx as nx
from utils import *
from colour import Color
import numpy as np


N_COLOR = 101  # number of color in range
# Color graph
def coloring_nodes(g):
    tg_g = g
    # get largest value of freq
    max_sen,_ = get_max_value_attribute(g,'sentiment_score')
    min_sen,_ = get_min_value_attribute(g,'sentiment_score')
    # print max_sen, min_sen
    # colors = list(Color("red").range_to(Color("green"), N_COLOR))
    colors = list(Color("#ff0000").range_to(Color("#0000ff"), N_COLOR))
    # print colors
    #  print colors
    marks = np.linspace(min_sen,max_sen,N_COLOR+1)
    for n in g.nodes():
        sen_score = g.node[n]['sentiment_score']
        # tg_g.node[n]['color'] = Color('blue')
        assigned_flag = False
        for i in xrange(0, N_COLOR):
            if (sen_score >= marks[i]) and (sen_score < marks[i+1]):
                tg_g.node[n]['color'] = colors[i]
                assigned_flag = True
                break
        if not assigned_flag:
            tg_g.node[n]['color'] = colors[int(N_COLOR/2)]
    # print tg_g.nodes(data=True)
    return tg_g
