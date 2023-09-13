# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.corpus import treebank
import spacy
import en_core_web_sm
from spacy.tokenizer import Tokenizer
from nltk.tag import NgramTagger
from nltk.tag import RegexpTagger
import numpy as np
import math

#Accuracy metric from nltk.metrics.scores
def accuracy(reference, test):
    """
    Given a list of reference values and a corresponding list of test
    values, return the fraction of corresponding values that are
    equal.  In particular, return the fraction of indices
    ``0<i<=len(test)`` such that ``test[i] == reference[i]``.

    :type reference: list
    :param reference: An ordered list of reference values.
    :type test: list
    :param test: A list of values to compare against the corresponding
        reference values.
    :raise ValueError: If ``reference`` and ``length`` do not have the
        same length.
    """
    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    return sum(x == y for x, y in zip(reference, test)) / len(test)

def load_data():

    nlp = en_core_web_sm.load()

    # We overwrite the spacy tokenizer with a custom one, that split by whitespace only
    nlp.tokenizer = Tokenizer(nlp.vocab) # Tokenize by whitespace

    total_size = len(treebank.tagged_sents())
    train_indx = math.ceil(total_size * 0.8)

    trn_data_tagged = treebank.tagged_sents(tagset='universal')[:train_indx]
    tst_data_tagged = treebank.tagged_sents(tagset='universal')[train_indx:]

    trn_data = treebank.sents()[:train_indx]
    tst_data = treebank.sents()[train_indx:]

    print('Dataset:')
    print("Total: {}; Tag Train: {}; Tag Test: {}".format(total_size, len(trn_data_tagged), len(tst_data_tagged)))
    print("Total: {}; Train: {}; Test: {}".format(total_size, len(trn_data), len(tst_data)))

    return trn_data_tagged, tst_data_tagged, trn_data, tst_data

def NgramTagger_NLTK(train_data_tagged, test_data_tagged, test_data):

    #training the NgramTagger with default parameters
    ngramtagger = NgramTagger(n=1, train=train_data_tagged)
    tagged_sents = ngramtagger.tag_sents(test_data)
    acc = []
    for x,y in zip(test_data_tagged, tagged_sents):
        acc.append(accuracy(x,y))
    nltk_accuracy = np.mean(acc)
    print('accuracy nltk [No backoff tagger]: ', nltk_accuracy)

    # Adding Backoff Tagger (Rule-based POS-Tagging)
    rules = [
        (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'),   # cardinal numbers
        (r'(The|the|A|a|An|an)$', 'DET'),   # articles
        (r'.*able$', 'ADJ'),                # adjectives
        (r'.*ness$', 'NOUN'),               # nouns formed from adjectives
        (r'.*ly$', 'ADV'),                  # adverbs
        (r'.*s$', 'NOUN'),                  # plural nouns
        (r'.*ing$', 'VERB'),                # gerunds
        (r'.*ed$', 'VERB'),                 # past tense verbs
        (r'[\.,!\?:;\'"]', '.'),            # punctuation (extension) 
        (r'.*', 'NOUN')                     # nouns (default)
    ]
    re_tagger = RegexpTagger(rules)

    ngramtagger = NgramTagger(n=1, train=train_data_tagged, backoff=re_tagger)
    tagged_sents = ngramtagger.tag_sents(test_data)     
    acc = []
    for x,y in zip(test_data_tagged, tagged_sents):
        acc.append(accuracy(x,y))
    nltk_accuracy = np.mean(acc)
    print('accuracy nltk [backoff Rule-based POS-Tagging]: ', nltk_accuracy)

    #Trying different cutoff parameters
    print('\nTesting different cutoff parameters...')
    for cutoff in [0,1,2,3,10]:
        ngramtagger = NgramTagger(n=1, train=train_data_tagged, cutoff=cutoff, backoff=re_tagger)
        tagged_sents = ngramtagger.tag_sents(test_data)
        acc = []
        for x,y in zip(test_data_tagged, tagged_sents):
            acc.append(accuracy(x,y))
        
        print("Cutoff: {}; Accuracy: {}".format(cutoff, np.mean(acc)))

    return nltk_accuracy

#defining a function to use the nlp spacy pipeline
def list_to_string(sentence):
    string = sentence[0]
    for i in range(1,len(sentence)):
        string = string + ' ' + sentence[i]
    return string

def Spacy_POS_Tags(test_data_tagged, test_data):

    #mapping pos tags from nltk to spacy
    mapping_spacy_to_NLTK = {
        "ADJ": "ADJ",
        "ADP": "ADP",
        "ADV": "ADV",
        "AUX": "VERB",
        "CCONJ": "CONJ",
        "DET": "DET",
        "INTJ": "X",
        "NOUN": "NOUN",
        "NUM": "NUM",
        "PART": "PRT",
        "PRON": "PRON",
        "PROPN": "NOUN",
        "PUNCT": ".",
        "SCONJ": "CONJ",
        "SYM": "X",
        "VERB": "VERB",
        "X": "X"
    }

    nlp = en_core_web_sm.load()

    # We overwrite the spacy tokenizer with a custom one, that split by whitespace only
    nlp.tokenizer = Tokenizer(nlp.vocab) # Tokenize by whitespace

    #getting the nltk mapped POS-tags for the test_set
    spacy_tagged_sents = []
    for sent in test_data:
        doc = nlp(list_to_string(sent))
        spacy_tagged_sents.append([(x,y) for x,y in zip([t.text for t in doc],[mapping_spacy_to_NLTK.get(t.pos_) for t in doc])])

    #evaluate spaCy accuracy
    acc = []
    for x,y in zip(test_data_tagged, spacy_tagged_sents):
        acc.append(accuracy(x,y))

    spacy_accuracy = np.mean(acc)
    print('accuracy spacy: ', spacy_accuracy)

    return spacy_accuracy
