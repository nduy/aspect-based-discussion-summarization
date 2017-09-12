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
from config import replace_pattern
import jsonrpc
from simplejson import loads
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import json
from collections import Counter

# Normalization and cleaning engine
cucco = Cucco()

# Connect to core nLP server
server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))

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
    # doc = ""
    #for sen in sentence_list:
    #    if sen == '\r\n':
    #        doc = doc + u"\u2029"
    #    else:
    #        doc = doc + u" " + sen
    # print sentence_list
    doc = ' '.join(sentence_list)
    doc = doc.replace(u"*break_me*",u" \t\n")
    # print doc
    tt = texttiling.TextTilingTokenizer()
    segmented_text = tt.tokenize(doc)
    return [simple_normalize(para.strip()) for para in segmented_text if para.strip()]


# Support function for reading unicode characters
def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8', 'ignore')


# Multiple replace in string

def replace_all(repls, str):
    # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], str)
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),
                  lambda k: repls[k.group(0)], str)
# Usage:
#text =  "i like apples, but pears scare me"
#print replace_all({"apple": "pear", "pear": "apple"}, text)

# Clean up the text, and get rig of unnecessary characters
def text_preprocessing(rawText):
    # print  replace_pattern
    txt = cucco.normalize(rawText, normalizations)
    txt = re.sub('<\D*>.*?<\D*>', '', txt)
    m = re.search(r"[a-z]([.,;:<>()+])[A-Z]", txt) # Fix space missing typo
    txt = txt[:m.start()+2] + u" " + txt[m.end()-1:] if m else txt
    txt = replace_all(replace_pattern,txt)
    return txt


# Refine a sentence by replacing its reference be the referee word/phrase
# @param: a sentence
# @return: refined sentence
def coreference_refine(text):
    if not text.strip():
        return text
    tokens = [[tok for tok in word_tokenize(sen)] for sen in sent_tokenize(text)]
    rs_tks = tokens
    parse_rs = None
    try:
        parse_rs = loads(server.parse(text))
    except Exception:
        maybe_print("[W] Can't parse sentence \"{0}...\"".format(text[:30]), 2)
    # print parse_rs
    try:
        if not parse_rs or 'coref' not in parse_rs:
            return text
        for group in parse_rs['coref']:
            for s, t in group:
                # print len(rs_tks[s[1]])
                if s[0] and t[0] and len(s[0]) < 50 and len(t[0]) < 50:
                    # calculate size differences:
                    diff = (s[4] - s[3]) - (t[4] - t[3])
                    # Remove the reference
                    # print rs_tks[s[1]], s[3], s[4]
                    for i in xrange(s[3], s[4]):
                        rs_tks[s[1]].pop(s[3])
                    if diff == 0:
                        # Add the referee
                        starting_pos = s[3]
                        for i in xrange(t[3], t[4]):
                            rs_tks[s[1]].insert(starting_pos, tokens[t[1]][i])
                            starting_pos += 1
                    elif diff > 0:  # to-be-replace is greater than to replace
                        # Add the referee
                        starting_pos = s[3]
                        for i in xrange(t[3], t[4]):
                            #if i >= t[4]:
                            #    rs_tks[s[1]].insert(starting_pos, u"")
                            #else:
                            rs_tks[s[1]].insert(starting_pos, tokens[t[1]][i])
                            starting_pos += 1
                        for i in xrange(0,diff):
                            rs_tks[s[1]].insert(starting_pos, u"")
                            starting_pos += 1
                    else:  # to-be-replace is greater than to replace
                        # Add the referee
                        starting_pos = s[3]
                        for i in xrange(t[3], t[4] + diff):
                            if i == t[4] + diff -1:
                                item = u""
                                for j in xrange(i, t[4]):
                                    item = item + u" " + tokens[t[1]][j]
                                item = item[1:]
                                rs_tks[s[1]].insert(starting_pos, item)
                            else:
                                rs_tks[s[1]].insert(starting_pos, tokens[t[1]][i])
                                starting_pos += 1
                # print len(rs_tks[s[1]])
        rs = u" ".join([u" ".join(s_list) for s_list in rs_tks])
        if rs:
            return rs
        else:
            return text
    except Exception:
        maybe_print("[W] --> unable to extract co-reference for sentence: \"{0}...\"".format(text[:30]), 2)
        return text


# Generate the JSON code from the networkx graph structure and perform node coloring
def generate_json_from_graph(g):
    result = {'nodes': [],
              'edges': []
              }
    if not g or len(g.nodes()) == 0:
        return None

    for node in g.nodes():
        # Go one by one
        item = dict()
        item['id'] = node
        if g.node[node]['weight']:
            w = g.node[node]['weight']
            item['value'] = w
            item['title'] = "*Freq: " + str(w) \
                            + " <br> *Sen_Score: " + str(round(g.node[node]['sentiment_score'], 4)) \
                            + " <br> *Sentiment: " + json.dumps(g.node[node]['sentiment']) \
                            + " <br> *Group_ids: " + ','.join(list(g.node[node]['group_id']))

        if g.node[node]['label']:
            item['label'] = g.node[node]['label']
        if g.node[node]['color']:
            item['color'] = str(g.node[node]['color'])
        if 'central.group' in g.node[node]['group_id']:
            item['group'] = 'central'
        else:
            is_article_node = False
            for group_id in g.node[node]['group_id']:
                is_article_node = is_article_node or group_id[:3] == "art"
            if is_article_node:
                item['group'] = 'article'
            else:
                item['group'] = 'comment'

        #  print item
        result['nodes'].append(item)

    for edge in g.edges(data=True): # edge is a tuple (source,target,data)
        item = dict()
        item['id'] = edge[0]+'|'+edge[1]
        if g.edge[edge[0]][edge[1]]['weight']:
            w = g.edge[edge[0]][edge[1]]['weight']
            item['value'] = w
            item['title'] = "freq: " + str(w)
        #  if G.edge[edge[0]][edge[1]]['label']:
        #    item['label'] = G.edge[edge[0]][edge[1]]['label']
        item['from'] = edge[0]
        item['to'] = edge[1]
        counts = Counter([e.strip() for e in edge[2]['label'].split(',')]).most_common()
        item['label'] = ', '.join([l+'^'+str(c) for l,c in counts])
        result['edges'].append(item)

    return result