#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
    Author: Nguyen Duc Duy - UNITN
    Created on Mon August 23 12:09 2017
    Functions to call Stanford NLP library
"""


from nltk.parse.stanford import StanfordDependencyParser

dep_parser=StanfordDependencyParser(model_path="/home/duy/stanford-parser-full-2017-06-09/models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


sentence = 'Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas'
filter_opt = {
    'prefered_pos': 'all', # preferred part-of-speech tags
    'prefered_rel' : 'all' # ['nsubk','nsubkpass','obj','iobj'] list of relation to remains
}




# Extract a graph from a sentence
# @param: a sentence, and filtering options
# @output: a list of dependencies

def dep_extract_from_sent(sentence,filter_opt):
    result = dep_parser.raw_parse(sentence)
    dependencies = result.next()
    raw_results = list(dependencies.triples())
    print('Options: ',filter_opt)
    # Filter out by POS
    prefered_pos = filter_opt['prefered_pos']
    if type(prefered_pos) != list:  # take all POS
        filter_pos_result = raw_results
    else:
        # filter tripples whse beginning and ending tags are inside the list
        filter_pos_result = [trip for trip in raw_results if (trip[0][1] in prefered_pos and trip[2][1] in prefered_pos)]

    # Filter by relationship
    print filter_pos_result
    prefered_rel = filter_opt['prefered_rel']
    if type(prefered_rel) != list:  # take all POS
        return filter_pos_result
    else:
        # filter tripples that has relation in the list
        filter_rel_result = [trip for trip in raw_results if (trip[1] in prefered_rel)]
    return filter_rel_result



#if __name__ == '__main__':
#    dep_extract_from_sent(sentence,filter_opt)

