#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Sat Jul  1 00:31:07 2017
    UTILITIES FUNCTIONS
    - Graph coloring before exporting json

"""


from cucco import Cucco
from datetime import datetime
from nltk.tokenize import texttiling
import re
from config import script_verbality

# Normalization and cleaning engine
cucco = Cucco()
normalizations = [
    'remove_accent_marks',
    ('replace_urls', {'replacement': ''}),
    ('replace_emails', {'replacement': ''}),
    'remove_extra_whitespaces',
]


# Decide to print the print text or not according to the verbality
# command_verbality:
#   0: always print
#   1: sometime print
#   2: rarely print
def maybe_print(text, command_verbality=1):
    if script_verbality > 0:
        if script_verbality >= command_verbality:
            print text


# flattening list of sublist into a single list
def flatten_list(l):
    if isinstance(l, list):
        lx = []
        for lz in l:
            lx.extend(map(flatten_list, lz))
        return lx
    else:
        return l


# get maximum value of nodes regarding to an attribute
# @param: the graph g, respect to attribute att
# @output: the integer of maximum weight AND the id of node
def get_max_value_attribute(g, att):
    max_value = -1
    node_id = None
    for node in g.nodes():
        try:
            val = g.node[node][att]
            if val > max_value:
                max_value = val
                node_id = node
        except:
            raise ValueError, "Attribute " + att + "  not found for node " + node
    return max_value, node_id


# get minimum value of nodes regarding to an attribute
# @param: the graph g, respect to attribute att
# @output: the integer of minimum weight AND the id of node
def get_min_value_attribute(g,att):
    min_value = 99990
    node_id = None
    for node in g.nodes():
        try:
            val = g.node[node][att]
            if val < min_value:
                min_value = val
                node_id = node
        except:
            raise ValueError, "Attribute " + att + "  not found for node " + node
    return min_value, node_id


# get maximum node arrtibute, which is a sub attribute in a grah
# @param: the graph g, respect to attribute att_parent, sub-arrtibute att_child
# @output: the integer of maximum weight AND the id of node
def get_max_value_subattribute(g,att_parent,att_child):
    max_value = -1
    node_id = None
    for node in g.nodes():
        try:
            val = g.node[node][att_parent][att_child]
            if val > max_value:
                max_value = val
                node_id = node
        except:
            raise ValueError,"Attribute " + att_parent + "-> " + att_child + " not found for node " + node
    return max_value,node_id


# Generate the code representing current time in second-microsecond
def gen_mcs():
    return hex(int(datetime.now().strftime("%s%f")))


# Generate the identical code representing current micro-second
def gen_mcs_only():
    return hex(int(datetime.now().strftime("%s")))


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


# Segmentize the sentences
# @param: a list, each element is a sentence the document
# @return: a list, each element is group of sentence concatenated to form a paragraph.
def texttiling_tokenize(sentence_list):
    print "sssssssss", sentence_list
    doc = ' '.join(sentence_list)
    tt = texttiling.TextTilingTokenizer()
    segmented_text = tt.tokenize(doc)
    return [simple_normalize(para.strip()) for para in segmented_text if para.strip()]


# Support function for reading unicode characters
def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8', 'ignore')


# Clean up the text, and get rig of unnecessary characters
def text_preprocessing(rawText):
    txt = cucco.normalize(rawText, normalizations)
    txt = re.sub('<a.*>.*?</a>', '', txt)
    txt = txt.replace('<blockquote>', '').replace('</blockquote>', '')
    return txt;