#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Sat Jul  1 00:31:07 2017
    UTILITIES FUNCTIONS
    - Graph coloring before exporting json

"""

from __future__ import unicode_literals
from cucco import Cucco
from datetime import datetime
from nltk.tokenize import texttiling
import re
from config import script_verbality
from config import replace_pattern
import jsonrpc
import json
from collections import Counter
from scipy.spatial.distance import cosine
import networkx as nx
import en
import sys
from operator import itemgetter
import csv

reload(sys)
sys.setdefaultencoding('utf-8')

# Normalization and cleaning engine
cucco = Cucco()

# Connect to core nLP server
coref_server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))
dep_server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),jsonrpc.TransportTcpIp(addr=("127.0.0.2", 8080)))

normalizations = [
    'remove_accent_marks',
    ('replace_urls', {'replacement': ''}),
    ('replace_emails', {'replacement': ''}),
    'remove_extra_whitespaces',
]

rep = {u'\n':'' ,
       '            ': '',
       u"\n \n": "",
       u" \t  \t ": "",
       u"\u201c": "\"",
       u"\u201d": "\"",
       u"\u2019": "\'",
       u"\u2018": "\'",
       u"\u2013": "-",
       u"\u2014": "-",
       u"\u2011": "-",
#       u"\u0027": "",
#       u"\u0022": "",
       u"\u201e": "",
       u"\u201d": "",
#       u"\u002c": "",
       u"\u002b": "",
       u"\u07f5": "",
       u"\u07f4": "",
       u"\u0092": ""}

rep = dict((re.escape(k), v) for k, v in rep.iteritems())
pattern = re.compile("|".join(rep.keys()))

# Decide to print the print text or not according to the verbality
# command_verbality:
#   0: always print
#   1: sometime print
#   2: rarely print
def maybe_print(text, command_verbality=1, alias=" "):
    if script_verbality > 0:
        if script_verbality >= command_verbality:
            print u'[{0}] {1}'.format(alias,text.encode('utf-8'))


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
        except Exception:
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
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], cnt)


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
    #print segmented_text
    # print [simple_normalize(para.strip()) for para in segmented_text if para.strip()]
    # exit(0)
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
    txt = cucco.normalize(rawText.decode('utf8', 'ignore'), normalizations)
    tmp = re.sub('<\D*>.*?<\D*>', '', txt)
    if tmp != '':
        txt = tmp
    # m = re.search(r"[a-z]([.,;:<>()+])[A-Z]", txt) # Fix space missing typo
    m = re.search(r"[a-z]([<>()+])[A-Z]", txt) # Fix space missing typo
    txt = txt[:m.start()+2] + u" " + txt[m.end()-1:] if m else txt
    txt = replace_all(replace_pattern,txt)
    return txt


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
        cluster_id = g.node[node]['cluster_id'] if 'cluster_id' in g.node[node] else 'unknown'
        # print 'HISTORY', g.node[node]['history'], ' type: ', type(g.node[node]['history'])
        if g.node[node]['weight']:
            w = g.node[node]['weight']
            # print type(g.node[node]['history']), g.node[node]['history']
            item['value'] = w
            item['title'] = u"✭NodeID: " + node \
                            + u" <br> ✭ Freq: " + str(w) \
                            + u" <br> ✭ Sen_Score: " + str(round(g.node[node]['sentiment_score'], 4)) \
                            + u" <br> ✭ Sentiment: " + json.dumps(g.node[node]['sentiment']) \
                            + u" <br> ✭ POS: " + ','.join(list(g.node[node]['pos'])) \
                            + u" <br> ✭ Group_ids: " + ','.join(list(g.node[node]['group_id'])) \
                            + u" <br> ✭ Cluster_ids: " + cluster_id \
                            + u" <br> ✭ History: {0}".format(g.node[node]['history'])
            item['cid'] = cluster_id
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

    for edge in g.edges(data=True):  # edge is a tuple (source,target,data)
        item = dict()
        item['id'] = edge[0]+'|'+edge[1]
        label_counts = Counter([e.strip() for e in edge[2]['label'].split(',')])

        if g[edge[0]][edge[1]]['weight']:
            w = g[edge[0]][edge[1]]['weight']
            item['value'] = w
            item['title'] = "*Freq: {0} <br>*Labels: <br>{1}".format(w, '<br>  -'.join([l+'^'+str(c)
                                                            for l,c in label_counts.most_common()]))
        #  if G[edge[0]][edge[1]]['label']:
        #    item['label'] = G[edge[0]][edge[1]]['label']
        item['from'] = edge[0]
        item['to'] = edge[1]

        item['label'] = label_counts.most_common(1)[0][0] # Get the label of the mmost common relationship

        result['edges'].append(item)

    return result


# Check if all element of set x is in set y
def all_x_is_in_y(setx=set(),sety=set()):
    if len(setx-sety) > 0:
        return False
    return True


# Implement the cosine similarity calculating using stanford Glove
def cosine_similarity(word1,word2,model):
    if word1 not in model.dictionary:
        ws = [w for w in word1.split(u'_') if w in model.dictionary and en.is_noun(w)]
        if not ws:
            return 0
        elif len(ws) > 1:  # there are at least 2 elements of the list is in dictionary
            # Simple composition by summing the vector
            v1 = model.word_vectors[model.dictionary[ws[-1]]]*0.7 + model.word_vectors[model.dictionary[ws[-2]]]*0.3
        else:  # has 1 element that is in the dictionary
            v1 = model.word_vectors[model.dictionary[ws[0]]]
    else:
        v1 = model.word_vectors[model.dictionary[word1]]

    if word2 not in model.dictionary:
        ws = [w for w in word2.split(u'_') if w in model.dictionary and en.is_noun(w)]
        if not ws:
            return 0
        elif len(ws) > 1:  # there are at least 2 elements of the list is in dictionary
            # Simple composition by summing the vector
            v2 = model.word_vectors[model.dictionary[ws[-1]]]*0.7 + model.word_vectors[model.dictionary[ws[-2]]]*0.3
        else:  # has 1 element that is in the dictionary
            v2 = model.word_vectors[model.dictionary[ws[0]]]
    else:
        v2 = model.word_vectors[model.dictionary[word2]]

    try:
        return 1 - cosine(v1,v2)
    except Exception as ex:  # key does
        return 0.0


# Inspect a history log of a node to detect and count repeated patterns
# E.g: "Initialize X^1 <br> Initialize X^1 <br> Initialize Y^1" to "Initialize X^2"
def repetition_summary(inp_string):
    head_str = inp_string
    rs = u""
    groups = re.findall(r'[\w]+\s([a-z_/\\-][a-z_/\\-]+)\^(\d)+', inp_string)
    tmp = [(key,int(count)) for key,count in groups] # convert count
    full_list=[]
    for key,count in tmp:
        for i in xrange(0,count):
            full_list.append(key)
    count_keys = Counter(full_list)  # convert occurrences of keys
    # Now get those whose occuced more than 1
    for key in count_keys:
        if count_keys[key]>1:
            head_str = re.sub('(<br> - ' + key + '\^[\d]+.+)<br>',u'',head_str)
            rs = rs + u"<br> » " + key + u"^" + str(count_keys[key])
    return head_str + rs


def print_top_keyphrases(g=None, ntop=20, out_path='./top20.csv'):
    """
    :param g: the aspect graph to compute
    :param ntop: number of topwords to extract
    :param out_path: path of output csv file
    :return: None
    """
    assert g is not None, "Can't get keyphrase from a NoneType graph."
    eigenvector_cen = nx.eigenvector_centrality(G=g, max_iter=10000)
    eigenvector_top = sorted(eigenvector_cen.iteritems(), key=itemgetter(1), reverse=True)[:min([ntop,
                                                                                                len(eigenvector_cen)])]

    degree_cen = nx.degree_centrality(G=g)
    degree_top = sorted(degree_cen.iteritems(), key=itemgetter(1), reverse=True)[:min(ntop, len(degree_cen))]

    closeness_cen = nx.closeness_centrality(G=g)
    closeness_top = sorted(closeness_cen.iteritems(), key=itemgetter(1), reverse=True)[:min([ntop, len(closeness_cen)])]

    betweenness_cen = nx.betweenness_centrality(G=g)
    betweenness_top = sorted(betweenness_cen.iteritems(), key=itemgetter(1), reverse=True)[:min([ntop,
                                                                                                len(betweenness_cen)])]

    pagerank_cen = nx.pagerank(G=g)
    pagerank_top = sorted(pagerank_cen.iteritems(), key=itemgetter(1), reverse=True)[:min([ntop, len(pagerank_cen)])]

    # Now write to filec
    with open(out_path, 'wb+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Eigenvector_cen', 'Degree', 'Closeness', 'Betweenness', 'Pagerank'])
        for i in xrange(0,ntop):
            writer.writerow([g.node[eigenvector_top[i][0]]['label'] if i < len(eigenvector_top) else "",
                             g.node[degree_top[i][0]]['label'] if i < len(degree_top) else "",
                             g.node[closeness_top[i][0]]['label'] if i < len(closeness_top) else "",
                             g.node[betweenness_top[i][0]]['label'] if i < len(betweenness_top) else "",
                             g.node[pagerank_top[i][0]]['label']]) if i < len(pagerank_top) else ""

