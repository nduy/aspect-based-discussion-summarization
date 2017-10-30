import rdflib
from rdflib.namespace import RDFS, SKOS, DCTERMS
import Queue
import logging
import networkx as nx
import operator
from nltk.stem import WordNetLemmatizer

'''
This is an implementation of WSDM-2013 paper (title: Unsupervised Graph based Topic Labelling Using DBPedia

Given a list of topic words (eg: ['Atom','Energy','Electron','Quantum','Orbit','Particle']) it would output:Quantum_mechanics,
Particle_physics', 'Concepts_in_physics', 'Orbits', 'Chemistry

Forked from https://gitlab.lif.univ-mrs.fr/balamurali.ar/labelling_rest.git

'''

logger = logging.getLogger("Dbpedia Labeller")
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
DBPediaURI= 'http://dbpedia.org/resource/'
# Lemmatizer
lemmatizer = WordNetLemmatizer()


def createGraph(topicURI,relation,toExpandURI):
    graph=rdflib.Graph()
    logger.info('Parsing the uri:%s',topicURI)
    graph.parse(topicURI)
    #get the tuples and add to a graph
    for s,p,o in graph.triples( (None, relation, None) ):
        if 'dbpedia' in o:
            toExpandURI.append(o)
        
    return (graph.triples((None, relation, None)),toExpandURI)


def expandConcepts(topicWord,expansionList,noHops):
    uri = DBPediaURI +  topicWord
    q1 = Queue.Queue()
    q2=[]
    q1.put(uri)
    graph = rdflib.Graph()
    
    for hop in range(0,noHops):
        while not q1.empty():
            uri= q1.get()
            for relation in expansionList:
                try:
                    (tempGraph,q2)= createGraph(uri, relation,q2)
                    graph += tempGraph
                except Exception:
                    pass
        map(q1.put, q2)
        q2=[]
    logger.info('The length of the graph is %d', len(graph))

    if len(graph) ==0:
        logger.info('Trying to split the word if applicable')
        subwords = [lemmatizer.lemmatize(w.lower()) for w in topicWord.split(u"_")]
        if len(subwords)>1:
            for subword in subwords:
                graph += expandConcepts(subword,expansionList,noHops)
    return graph
    

def findCentralNode(graph):
    '''
    Using network X, to create merged graph and find the central node
    NOTE: uncomment different centrality features based 
    '''
    network=nx.Graph()
    network.add_edges_from([(i,k) for i,j,k in graph])
    nodes=nx.degree_centrality(network)
    #nodes=nx.betweenness_centrality(network)
    #nodes=nx.load_centrality(network)
    #nodes=nx.closeness_centrality(network)
    return nodes


def cleanLabels(sortedNodes):
    '''
    Cleans the labels if they contain
    '''
    sortedNodes =map(lambda x: x.split(':')[1] if ':' in x else x, sortedNodes)
    return sortedNodes


def DBPprocess(topicWordList):
    '''
    creates the graph and expands based on the set of relations and based on the centrality returns possible labels (5 nos)
    '''
    expansionList=[SKOS.broader, SKOS.broaderOf,RDFS.subClassOf,DCTERMS.subject]
    noHops=2
    graph=rdflib.Graph()
    for topicWord in topicWordList:
        graph+=expandConcepts(topicWord.title(), expansionList, noHops)       
    nodes=findCentralNode(graph)
    
    #start the cleaning process
    sortedNodes= dict(sorted(nodes.iteritems(), key=operator.itemgetter(1), reverse=True)[:2])
    sortedNodes=map(lambda x: x.split('/')[-1],sortedNodes)
    sortedNodes =cleanLabels(sortedNodes)
    return sortedNodes
    

if __name__ =='__main__':
    topicWordList=['waste', 'collection', 'rubbish']
    nodes=DBPprocess(topicWordList)
    print nodes
    