# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":

    #collect the 100 last sentences from dependency treebank
    train_sents, test_sents = load_data(n_sentences=100)

    #Initialize the parser with spacy and stanza
    spacy_parser = DependencyParser(version='spacy')
    stanza_parser = DependencyParser(version='stanza')

    #get the dependency graph
    spacy_dependency_graph = spacy_parser.parse(train_sents=train_sents)
    stanza_dependency_graph = stanza_parser.parse(train_sents=train_sents)

    #evaluate LAS and UAS for each parser
    evaluate(d_graph_spacy=spacy_dependency_graph, d_graph_stanza=stanza_dependency_graph, test_sents=test_sents)
