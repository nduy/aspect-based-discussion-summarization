from utils import maybe_print, text_preprocessing
import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from functions import lemmatizer
from nltk.tokenize import StanfordTokenizer
from collections import Counter
import re

class AGmodel:
    """
    The Aspect Graph model for topic modeling
    """
    node_id_dictionary = {} # NodeID to word dictionary
    dictionary = {}         # LABEL to word dictionary
    dict_size = 0
    inverted_node_id_dictionary = {}
    inverted_dictionary = {}
    n_topics = 0            # number of topic in this model
    matrix = None           # probability matrix. Idealy it has n_topics row and len(dictionary) columns
    topic_ids = None       # ID of the topics
    # Member nodes inside each topic. It is a dict, key is the topic id, value is a list of nodes inside the dict
    topic_nodes = None      # store node ids of member of clusters
    topic_graphs = None     # store the graph that represent each topic. key: topic id, value: the subgraph

    def __init__(self, asp_graph=None, **kwargs):
        """
        Initialize the aspect network model from a given aspect graph (networkx graph)
        :param asp_graph: The aspect graph to be converted to aspect graph model
        :param kwargs:
        """
        # Compute a dictionary for the graph, which is all nodes in graph
        index = 0
        for node,data in asp_graph.nodes(data=True):
            self.dictionary[index] = data['label']
            self.node_id_dictionary[index] = node
            # make inverted label, which is all possible ever instance of this node
            for label in set([item for item in re.findall("([a-z_-]+)~", data['history'])]):
                self.inverted_dictionary[label] = index
            # inverted dictionary for node id
            self.inverted_node_id_dictionary[node] = index
            index += 1

        self.dict_size = len(self.dictionary)

        # get the number of topics and store and the topic ids
        self.topic_ids = nx.get_node_attributes(asp_graph, 'cluster_id').values()
        self.n_topics = len(self.topic_ids)

        # initialize a place holder for storing nodes of clusters
        self.topic_nodes = {key: [] for key in self.topic_ids}
        for node,data in asp_graph.nodes(data=True):
            self.topic_nodes[data['cluster_id']].append(node)
        # Extract subraph from the big graph
        for key,node_list in self.topic_nodes:
            self.topic_graphs[key] = asp_graph.subgraph(self.topic_nodes[key])
        maybe_print("Extracted {0} topics. ".format(self.n_topics), 2, "i")
        for key in self.topic_graphs:
            maybe_print(' -> Topic: {0}, Nodes: {1}'.format(key,list(self.topic_graphs['key'].nodes())), 2, 'i')

        # Compute matrix
        self.matrix = self.compute_probability_matrix(method='eigenvector_centrality', enable_softmax=True)
        print(self.matrix)

    @classmethod
    def get_dictionary(cls):
        """
        Get the dictionary
        :return: the model dictionary in dict format
        """
        return cls.dictionary

    @classmethod
    def get_n_topic(cls):
        """
        Get number of topic
        :return:
        """
        return cls.n_topics

    @classmethod
    def get_matrix(cls):
        """
        Get the predictive matrix
        :return:
        """
        return cls.matrix

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x.
            :param x: numpy array-like structure
            :return: the softmax vector, sum to 1
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @classmethod
    def compute_probability_matrix(cls,method='eigenvector_centrality', enable_softmax=True):
        """
        Compute the probability matrix
        :param method: threee methods:
            - 'count': normalize the term frequency of members of the cluster
            - 'degree': bases on the degree of nodes to judge its importance
            - 'pagerank': bases on result of page rank algorithm
            - 'eigenvector_centrality': Eigenvector centrality computes the centrality for a node based on
            the centrality of its neighbors.
        :param enable_softmax: apply softmax function on computing probability from node characterize statistic
            - True: apply softmax
            - False: apply simple scale by sum
        :return:
        """
        result_matrix = None
        if method=='eigenvector_centrality':
            # for each topic, we compute a vector size len(dictionary)
            for key in cls.topic_ids:
                centrality = nx.eigenvector_centrality(cls.topic_graphs[key]) # it output something like:
                """{0: 0.37174823427120085,
                 1: 0.6015008315175003,
                 2: 0.6015008315175004,
                 3: 0.3717482342712008}
                """
                row = np.zeros(cls.dict_size)
                for node_id in centrality:
                    row[cls.inverted_node_id_dictionary[node_id]] = centrality[node_id]
                if not result_matrix:
                    result_matrix = row
                else:
                    result_matrix = np.vstack((result_matrix, row))
        # Convert characterize statistic to probability
        if not enable_softmax: # divide each element in a row a sum of that row
            # now result_matrix is just a sum matrix. We need to turn it to a normaized matrix
            row_sum = np.sum(result_matrix,axis=1)[np.newaxis] # Sum by row. Result an array size is number of topics
            row_sum = row_sum.T     # convert to matrix for division
            result_matrix = result_matrix/np.repeat(row_sum, cls.dict_size)
        else:  # apply softmax to each row of the matrix
            for i in xrange(0,cls.n_topics):
                result_matrix[i] = cls.softmax(result_matrix[i])
        return result_matrix

    @classmethod
    def predict_document(cls, doc_id = None, text='', vectorize_method='bow'):
        """
        Predict the label of a SINGLE document base on bow representation of the key
        :param doc_id: ID of the document
        :param text:the content of the document
        :param vectorize_method: document vectorization method:
            - 'bow': bag of words: mark 1 if the word exist, otherwise 0
            - 'scaled_bow': scale the count by the highest
        :return: list of tuple (topic_id,probability) in order from highest to lowest
        """

        if not text:
            raise ValueError("Can't predict an empty document!")
        else:
            txt = text_preprocessing(text)
            sentences = sent_tokenize(txt.strip())  # sentence segmentation
            word_counter = None
            for sentence in sentences:
                blob = TextBlob(sentence)
                # print refine sentence with noun phrase
                for phrase in blob.noun_phrases:
                    pos = phrase.rfind(' ')
                    if pos == -1:
                        sentence = sentence.replace(phrase, lemmatizer.lemmatize(phrase))
                    else:
                        new_phrase = phrase[:pos + 1] + lemmatizer.lemmatize(phrase[pos + 1:])
                        sentence = sentence.replace(phrase, new_phrase.replace(u' ', u'_'))
                # now break it to tokens
                word_counter = Counter([lemmatizer.lemmatize(w)
                                        for w in StanfordTokenizer().tokenize(sentence.lower())])
            # initialize a zero vector for
            vec = np.zeros(cls.dict_size,dtype=float)
            # now depend on the vectorize method we compute the vector as follow
            if vectorize_method == 'bow':
                for word in word_counter:
                    if word in cls.inverted_dictionary:
                        vec[cls.inverted_dictionary[word]] = 1.0
            elif vectorize_method == 'scaled_bow': # scale each element by the max value
                max_count = word_counter.most_common(1)[0][1]  # get the max value of counter elements
                for word in word_counter:
                    if word in cls.inverted_dictionary:
                        vec[cls.inverted_dictionary[word]] = word_counter[word]*1.0/max_count
            # now you have the vector that size as the size of dictionary. Now you need
            vec = vec[np.newaxis].T     # transpose to column vector
            print(cls.matrix.shape,vec.shape)
            probabilities = np.reshape(np.matmul(cls.matrix, vec),cls.n_topics)
            result = sorted([(cls.topic_ids[i],probabilities[i]) for i in xrange(0,cls.n_topics)], key=lambda x: x[1])
            assert np.any(probabilities), 'Warning: all predicted probabilities are zero!'
            return result

