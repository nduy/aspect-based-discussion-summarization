from utils import maybe_print, text_preprocessing
import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from functions import lemmatizer
from functions import gen_mcs_only
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import warnings
from itertools import permutations
from sklearn import metrics
from config import model_build_options as options
import sys
from random import shuffle

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
    matrix = None           # probability matrix. Ideally, it has n_topics row and len(dictionary) columns
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
        # print '---', len(asp_graph)
        # Compute a dictionary for the graph, which is all nodes in graph
        index = 0
        for node, data in asp_graph.nodes(data=True):
            raw = [data['label']]
            raw.extend([item for item in re.findall("([a-z_-]+)~", data['history']) if len(item) > 1])
            # print raw
            # print [item for item in re.findall("([a-z_-]+)~", data['history'])]
            labels = list(set(raw))
            for item in raw:
                tokens = item.split('_')
                if len(tokens) > 1:
                    labels.extend(tokens)
                    labels.extend([lemmatizer.lemmatize(t) for t in tokens])
            labels = set(labels)

            for label in labels:
                self.dictionary[index] = label
                self.inverted_dictionary[label] = index
                self.node_id_dictionary[index] = node
                # This is trick. Each node may have multiple words that is in the dictionary, so the value of the
                # inverted_node_id_dictionary is a list, instead of a single value
                if node not in self.inverted_node_id_dictionary: # First time exist
                    self.inverted_node_id_dictionary[node] = set([index])
                else:
                    self.inverted_node_id_dictionary[node].add(index)

                index += 1

            # self.dictionary[index] = data['label']
            #self.node_id_dictionary[index] = node
            # make inverted label, which is all possible ever instance of this node
            #for label in set([item for item in re.findall("([a-z_-]+)~", data['history'])]):
             #   self.inverted_dictionary[label] = index
            #    for lb in label.split('_'):

            # inverted dictionary for node id
            #self.inverted_node_id_dictionary[node] = index
            #index += 1

        self.dict_size = len(self.dictionary)

        # get the number of topics and store and the topic ids
        self.topic_ids = list(set(nx.get_node_attributes(asp_graph, 'cluster_id').values()))
        self.n_topics = len(self.topic_ids)

        # initialize a place holder for storing nodes of clusters
        self.topic_nodes = {key: [] for key in self.topic_ids}
        self.topic_graphs = {key: None for key in self.topic_ids}
        for node,data in asp_graph.nodes(data=True):
            if 'cluster_id' in data:
                self.topic_nodes[data['cluster_id']].append(node)
        # Extract subraph from the big graph
        # print '----',self.topic_nodes

        for cluster_id in self.topic_nodes:
            self.topic_graphs[cluster_id] = asp_graph.subgraph(self.topic_nodes[cluster_id])
        maybe_print("Modeled {0} topics. ".format(self.n_topics), 2, "i")
        for key in self.topic_graphs:
            maybe_print(u' -> Topic: {0}, Nodes: {1}'.format(key,list(self.topic_graphs[key].nodes())), 3, 'i')

        # Compute matrix
        self.matrix = self.compute_probability_matrix(method=options.get('centrality_method'),
                                                      normalize=options.get('normalization_method'))
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

    @classmethod
    def get_topics(cls):
        """
        Get all topics
        :return: Format: "topic name: probability1*word1 + probability2*word2 ..."
        """
        assert cls.matrix, "The predictive matrix is undefined!"
        assert np.any(cls.matrix), "All cells in predictive matrix are ZERO"
        result = u""
        for i in xrange(0,cls.matrix.shape[0]): # get row
            result += u'Topic {0}: '.format(i)
            topics_words = []
            for index in np.where(cls.matrix[i,:] > 0.0)[0]: # for every words whose probability in this topic >0
                topics_words.append((cls.dictionary[index], cls.matrix[i,index]))
            topics_words = sorted(topics_words, key=lambda x: x[1],reverse=True)
            for j in xrange(0,max([20,len(topics_words)])):
                result += topics_words[0] + u'*' + str(topics_words[1])
            result += u'\n'

        return result

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x.
            :param x: numpy array-like structure
            :return: the softmax vector, sum to 1
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def compute_probability_matrix(self, method='eigenvector_centrality', normalize='sum'):
        """
        Compute the probability matrix
        :param method: threee methods:
            - 'count': normalize the term frequency of members of the cluster
            - 'degree': bases on the degree of nodes to judge its importance
            - 'pagerank': bases on result of page rank algorithm
            - 'eigenvector_centrality': Eigenvector centrality computes the centrality for a node based on
            the centrality of its neighbors.
        :param normalize: apply normaliztion on centrality score, so that they sum to 1
            - 'softmax': apply softmax function on computing probability from node characterize statistic
            - 'sum': apply simple scale by sum of all centrality scores
        :return:
        """
        result_matrix = None

        # for each topic, we compute a vector size len(dictionary)
        # print '----->',self.topic_ids
        for key in self.topic_ids:
            # print len(self.topic_graphs[key].edges())
            if method == 'eigenvector':
                centrality = nx.eigenvector_centrality(G=self.topic_graphs[key], max_iter=10000)
            elif method == 'degree':
                centrality = nx.degree_centrality(G=self.topic_graphs[key])
            elif method == 'closeness':
                centrality = nx.closeness_centrality(G=self.topic_graphs[key])
            elif method == 'betweenness':
                centrality = nx.betweenness_centrality(G=self.topic_graphs[key])
            elif method == 'pagerank':
                centrality = nx.pagerank(G=self.topic_graphs[key])
            else:
                raise ValueError('Invalid centrality method: {0}'.format(method))
            # print centrality
            row = np.zeros(self.dict_size)
            for node_id in centrality:
                # row[self.inverted_node_id_dictionary[node_id]] = centrality[node_id]
                for i in self.inverted_node_id_dictionary[node_id]:
                    row[i] = centrality[node_id]
            if result_matrix is None:
                result_matrix = row
            else:
                result_matrix = np.vstack((result_matrix, row))

        # Convert characterize statistic to probability
        # print result_matrix
        if normalize == "softmax": # divide each element in a row a sum of that row
            # now result_matrix is just a sum matrix. We need to turn it to a normaized matrix
            row_sum = np.sum(result_matrix,axis=1)[np.newaxis] # Sum by row. Result an array size is number of topics
            row_sum = row_sum.T     # convert to matrix for division
            result_matrix = result_matrix/np.repeat(row_sum, self.dict_size,axis=1)
        elif normalize == "sum":  # apply softmax to each row of the matrix
            for i in xrange(0, self.n_topics):
                result_matrix[i] = self.softmax(result_matrix[i])
        else:
            raise ValueError('Invalid normalization method: {0}'.format(normalize))
        return result_matrix

    def predict_doc(self, doc_id=None, text='', vectorize_method='bow', top=3):
        """
        Predict the label of a SINGLE document base on bow representation of the key
        :param doc_id: ID of the document
        :param text:the content of the document
        :param vectorize_method: document vectorization method:
            - 'bow': bag of words: mark 1 if the word exist, otherwise 0
            - 'scaled_bow': scale the count by the highest
        :param top: get n top elements only
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
                                        for w in word_tokenize(sentence.lower())])
            # initialize a zero vector for
            vec = np.zeros(self.dict_size, dtype=float)
            # now depend on the vectorize method we compute the vector as follow
            if vectorize_method == 'bow':
                for word in word_counter:
                    if word in self.inverted_dictionary:
                        vec[self.inverted_dictionary[word]] = 1.0
            elif vectorize_method == 'scaled_bow': # scale each element by the max value
                max_count = word_counter.most_common(1)[0][1]  # get the max value of counter elements
                for word in word_counter:
                    if word in self.inverted_dictionary:
                        vec[self.inverted_dictionary[word]] = word_counter[word]*1.0/max_count
            # now you have the vector that size as the size of dictionary. Now you need
            vec = vec[np.newaxis].T     # transpose to column vector
            # print(self.matrix.shape, vec.shape)
            # print self.matrix.shape, vec.shape
            # print vec, self.dictionary
            # print np.matmul(self.matrix, vec)
            probabilities = np.reshape(np.matmul(self.matrix, vec),self.n_topics)
            # print probabilities.shape, vec.shape
            result = sorted([(self.topic_ids[i],probabilities[i]) for i in xrange(0,self.n_topics)],
                            key=lambda x: x[1],
                            reverse=True)
            if not np.any(probabilities):
                warnings.warn("All predicted probabilities are zero! Assigned unknown_topic as default.", UserWarning)
                result = [('unknown_topic',1.0)]
            # print result
            return {'doc_id': doc_id, 'labels': result[:top]}

    def predict_multi_docs(self, doc_ids=None, multi_docs=None, vectorize_method='bow'):
        """
        Process multiple document for predicting which topic it belong to.
        :param self:
        :param doc_ids: list of the id of all docs.
        :param multi_docs: List, each element is a text document to be processed
        :param vectorize_method: document vectorizing method, see predict_doc method for more details
        :return: a list, whose size is equal to the size of multi_docs, each is a label for the corresponsing doc
        """
        assert multi_docs, "Can not predict a NoneType list of document."
        if doc_ids:
            assert len(doc_ids) == len(multi_docs), 'Size of list of document ID and text corpus are different!'

        result = dict()
        for i in xrange(0, len(multi_docs)):
            doc_id = doc_ids[i] if doc_ids else None
            content = multi_docs[i]
            try:
                rs = self.predict_doc(doc_id=doc_id,
                                      text=content,
                                      vectorize_method=vectorize_method,
                                      top=1)
                # print rs
                result[rs['doc_id'] if rs['doc_id'] else str(i)+u'~' + gen_mcs_only()] = rs['labels']
                # print "_+_++", rs
            except Exception as ins:
                print ins
                warnings.warn("Can't predict document {0}.".format(doc_id
                                                                   if doc_id else
                                                                   content[:max(100,len(content))])
                              ,UserWarning)
                # print "Unexpected error:", sys.exc_info()[0]
        return result

    def evaluate_model(self, doc_ids=None,  multi_docs=None, ground_truth=None,
                       vectorize_method='bow', cluster_overlap='count'):
        """
        :param doc_ids: id of all documents
        :param multi_docs: text content of all documents
        :param ground_truth: group that each element belongs to
        :return: {accuracy:val, precision:val, recall: val, f1:val}
        :param vectorize_method: document vectorizing method, see predict_doc method for more details
        :param cluster_overlap: method to measure overlap between cluster:
              - 'count': count the document that they share in common
              - 'percentage': percentage of share over SUM all nodes of two cluster
        """
        assert ground_truth, 'Invalid ground-truth for evaluating model'
        assert len(ground_truth) == len(multi_docs), 'Ground-truth and document content have different size'
        # compute the prediction of clusters
        predicted_result = self.predict_multi_docs(doc_ids=doc_ids,
                                                   multi_docs=multi_docs,
                                                   vectorize_method=vectorize_method)
        print "Done predicting on test set!"
        # print "Predicted predicted_result: ", predicted_result
        # Now evaluate the result
        # First reassign generated to the ground truth value

        # First transform ground-truth to cluster - like
        truth_clusters = dict()
        # do similar for the ground truth
        for i in xrange(0, len(ground_truth)):
            label = ground_truth[i]
            if label not in truth_clusters:
                truth_clusters[label] = [doc_ids[i]]
            else:
                truth_clusters[label].append(doc_ids[i])

        # convert predicted result to a dictionary form {topic_label: [list of all docs assigned this label]}
        predicted_clusters = dict()
        # the cluster
        for doc_id in predicted_result:
            label = predicted_result[doc_id][0][0]
            # print "Predicted label: ", label
            if label not in predicted_clusters:
                predicted_clusters[label] = [doc_id]
            else:
                predicted_clusters[label].append(doc_id)
        # This is TRICKY, if the size of predicted and truth are different, the create dummy cluster names
        diff = len(predicted_clusters)-len(truth_clusters)
        for i in xrange(0,abs(diff)):
            if diff < 0:
                predicted_clusters['dummy'+str(i)] = []
            elif diff > 0:
                truth_clusters['dummy' + str(i)] = []

        # make dictionaries
        truth_topic_names = truth_clusters.keys()
        truth_topic_dictionary = {k: v for k, v in enumerate(truth_topic_names)}    # id-> name
        inv_truth_topic_dictionary = {truth_topic_dictionary[k]:k for k in truth_topic_dictionary}
        predicted_topic_names = predicted_clusters.keys()
        predicted_topic_dictionary = {k: v for k, v in enumerate(predicted_topic_names)} # id -> name
        inv_predicted_topic_dictionary = {predicted_topic_dictionary[k]:k for k in predicted_topic_dictionary}
        print 'Done making dictionaries'

        # Compute the overlap matrix. Its columns are the ground truth topics, the rows is the generated topics, cell
        # contain the overlap statistic between the two cluster, depend on the parameter cluster_overlap
        overlap_matrix = np.zeros([len(predicted_clusters),len(truth_clusters)],dtype=np.float)
        for i in xrange(0,len(predicted_topic_names)):
            gen_topic = predicted_topic_names[i]
            for j in xrange(0,len(truth_topic_names)):
                truth_topic = truth_topic_names[j]
                if cluster_overlap == 'count':
                    # print '\n',predicted_clusters
                    overlap_matrix[i][j] = len(set(predicted_clusters[gen_topic])
                                               .intersection(set(truth_clusters[truth_topic])))
                elif cluster_overlap == 'percentage':
                    overlap_matrix[i][j] = len(set(predicted_clusters[gen_topic])
                                               .intersection(set(truth_clusters[truth_topic])))\
                                           / (len(set(predicted_clusters[gen_topic]))
                                              + len(set(truth_clusters[truth_topic]))
                                              )
        print 'Done making overlap matrix'
        # Greedy approach find the best match from the generated to ground truth topic
        # print overlap_matrix
        best_permutation = []
        best_sum_score = 0.0
        '''
        # find all posible permutation of generated label
        # print 'predicted_topic_names: ', predicted_topic_names
        #  for permu in permutations(predicted_topic_names):
        permu_count = 0
        for permu in permutations(predicted_topic_names):
            permu_count +=1
            print '[{0}]---> permu: {1}'.format(permu_count,permu)
            sum_score = 0.0
            for i in xrange(0, len(predicted_topic_names)):
                # print '---> permu[i]: ', permu[i]
                # print '---> inv_predicted_topic_dictionary[permu[i]]: ', inv_predicted_topic_dictionary[permu[i]]
                # print '---> overlap_matrix[inv_predicted_topic_dictionary[permu[i]]][i]: ',
                # overlap_matrix[inv_predicted_topic_dictionary[permu[i]]][i]

                sum_score += overlap_matrix[inv_predicted_topic_dictionary[permu[i]]][i]
            if sum_score > best_sum_score:
                best_sum_score = sum_score
                best_permutation = permu
        
        # First make a matrix of factorial(len(predicted_topic_names)) x len(truth_topic_names). Each row corresponding
        # to one possible order of the predict_label, values are overlap picked from overlap_matrix (given this order)
        # then we sum row and get the best permutation as the one that give highest overlap
        # all possible combination of predicted label indices
        n_pre_topics = len(predicted_topic_names)
        possible_order = np.array(list(permutations(list(xrange(0,n_pre_topics))))) # convert to array, each row is posible indecess
        rows = np.reshape(possible_order,(1,possible_order.shape[0]*n_pre_topics))[0]  # convert to 1-D array
        cols = np.matlib.repmat(np.array(list(xrange(0,n_pre_topics))), 1, possible_order.shape[0])[0]  # 1-D array too
        score_matrix = overlap_matrix[rows,cols].reshape(possible_order.shape[0],n_pre_topics)
        max_permutation_index = np.argmax(np.sum(a=score_matrix, axis=1))
        for index in possible_order[max_permutation_index]:
            best_permutation.append(predicted_topic_names[index])
        '''
        # find all posible permutation of generated label
        # print 'predicted_topic_names: ', predicted_topic_names
        #  for permu in permutations(predicted_topic_names):
        n_pre_topics = len(predicted_topic_names)
        #print '-------> n_pre_topics', n_pre_topics
        # exit(0)
        cols = np.array(xrange(0,n_pre_topics))
        # convert to array, each row is possibleindicess
        permu_count = 0
        # possible_order = np.array(list(permutations(list(xrange(0, n_pre_topics)))))
        # print possible_order
        # for i in xrange(0, possible_order.shape[0]): # go through every possible permutation
        MAX_ITERATION = 5000000000
        # all_permutations = list(permutations(xrange(0, n_pre_topics)))
        #shuffle(all_permutations)
        # what it does? It seek for permutation that has the higest score in MAX_ITERATION consecutive random loop
        dominant_count = 0
        max_dominant = 200000000
        #for per in all_permutations: # go through every possible permutation
        for per in permutations(xrange(0, n_pre_topics)): # go through every possible permutation
            permu_count += 1
            if permu_count > MAX_ITERATION: # break the loop if max iteration reached
                break
            if np.random.rand() < 0.6:  # random choose to keep or disregard the combination
                continue
            if permu_count % 100000 == 0:
                print "+++++++ ", permu_count
            if dominant_count == max_dominant and best_sum_score!=0:
                break
            # Compute sum overlap for this permutation
            rows = np.array(per)
            sum_score = np.sum(overlap_matrix[rows,cols])
            # print best_sum_score
            if sum_score > best_sum_score:
                dominant_count = 0
                best_sum_score = sum_score
                print best_sum_score
                best_permutation = [predicted_topic_names[i] for i in rows]
            else:
                dominant_count += 1

        print 'Done search for best permutation'
        assert best_permutation, "best permutation is still None!"
        assert best_sum_score > 0, "best sum score 0.0 means no overlap was found!"
        # Now we had the best permutation, from the first to the last of the list, it map to the first to the last of
        #   ground-truth values
        # Now we update the predicted topic dictionary such that its id mention the same cluster
        #   as in truth topic dictionary
        # print 'truth_topic_dictionary:', truth_topic_dictionary
        # print 'best_permutation', best_permutation
        predicted_topic_dictionary = dict()
        for key in truth_topic_dictionary:
            # print "key: ",key
            # print "truth_topic_dictionary[key]", truth_topic_dictionary[key]
            predicted_topic_dictionary[key] = best_permutation[truth_topic_names.index(truth_topic_dictionary[key])]
        # {k: best_permutation[truth_topic_names.index(truth_topic_dictionary[k])]
        #                              for k in truth_topic_dictionary}
        # print 'CCC', predicted_topic_dictionary
        inv_predicted_topic_dictionary = {predicted_topic_dictionary[k]: k for k in predicted_topic_dictionary}
        # OK, we got it. Now make material for measurement:
        y_predicted = [inv_predicted_topic_dictionary[predicted_result[doc_id][0][0]] for doc_id in doc_ids]
        print 'Predicted', y_predicted
        y_truth = [inv_truth_topic_dictionary[label] for label in ground_truth]
        print 'Truth', y_truth
        # return
        rs = {
            'accuracy': metrics.accuracy_score(y_truth,y_predicted),
            'precision_macro': metrics.precision_score(y_truth,y_predicted, average='macro'),
            'precision_micro': metrics.precision_score(y_truth, y_predicted, average='micro'),
            'precision_weighted': metrics.precision_score(y_truth, y_predicted, average='weighted'),
            # 'average_precision': metrics.average_precision_score(y_truth, y_predicted),
            'recall_macro': metrics.recall_score(y_truth,y_predicted, average='macro'),
            'recall_micro': metrics.recall_score(y_truth, y_predicted, average='micro'),
            'recall_weighted': metrics.recall_score(y_truth, y_predicted, average='weighted'),
            'f1_macro': metrics.f1_score(y_truth, y_predicted, average='macro'),
            'f1_micro': metrics.f1_score(y_truth, y_predicted, average='micro'),
            'f1_weighted': metrics.f1_score(y_truth, y_predicted, average='weighted'),
        }
        # maybe_print("Model evaluation result: {0}".format(rs))
        return rs
