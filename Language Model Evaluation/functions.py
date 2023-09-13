# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.corpus import gutenberg
from nltk.lm.preprocessing import flatten
from sklearn.model_selection import train_test_split
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import StupidBackoff
from collections import defaultdict
import numpy as np
import math

def load_data():
    # Dataset
    macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-macbeth.txt')]

    #splitting the dataset into a training set and a small test set
    macbeth_sents_train, test_sentences = train_test_split(macbeth_sents, test_size=0.01, random_state=42)
    print('Dataset:', len(macbeth_sents),'Train Set:', len(macbeth_sents_train),'Test Set:', len(test_sentences))

    #flatten the words to build a vocabulary
    macbeth_words = flatten(macbeth_sents_train)

    #building a vocabulary
    lex = Vocabulary(macbeth_words, unk_cutoff=2)

    #removing the rare words with the tag 'UNK' 
    macbeth_oov = [list(lex.lookup(sent)) for sent in macbeth_sents_train]

    return macbeth_oov, lex, macbeth_sents_train, test_sentences

def StupidBackoff_nltk(data_oov, lex, train, test):

    #trying some parameters alpha
    for a in [0.2, 0.4, 0.8, 1.0]:

        #building everytime the n-grams since it returns a lazy-generator
        train, vocab = padded_everygram_pipeline(2, data_oov)
        
        #building the StupidBackoff model from nltk
        sb = StupidBackoff(order=2, alpha=a)
        sb.fit(train, vocab)
        
        #building the n-grams with rare words for the validation set
        ngrams, _ = padded_everygram_pipeline(sb.order, [lex.lookup(sent.split()) for sent in flatten(test)])

        max_sequence = []
        for gen in ngrams:
            for g in gen:
                if len(g) == sb.order:
                    max_sequence.append(g)
        
        #calculate the perplexity of the model
        print('with alpha=',a,' Perplexity score:',sb.perplexity(max_sequence))

class MyStupidBackoff:
    def __init__(self, n, alpha):
        self.n = n
        self.alpha = alpha
        self.prob = defaultdict(lambda: defaultdict(lambda: 0.0))
    
    #function to compute the probability table given the corpus
    def compute_probabilities(self, data):
        word_count = 0
        counts = defaultdict(lambda: defaultdict(lambda: 0.0))
    
        for sent in data:
            sent = tuple(sent)
        
            for word in range(len(sent)):
                word_count += 1
                for i in range(self.n):
                    if word-i < 0:
                        break
                    counts[sent[word-i:word]][sent[word]] += 1
                
        self.prob = defaultdict(lambda: defaultdict(lambda: 0.0))
        for context in counts.keys():
            q = 0
            for w in counts[context].keys():
                q += counts[context][w]
            for w in counts[context].keys():
                self.prob[context][w] = counts[context][w] / q
            
        return self.prob
    
    #compute the probability of the word w given the model with the stupid backoff technique
    def get_prob(self, context, w):
        if context in self.prob and w in self.prob[context]:
            return self.prob[context][w]
        else:
            return self.alpha * self.get_prob(context[1:], w)
    
    #compute the perplexity given a list of sentences
    def perplexity(self, data):
        perp = 0.0
        T = 0
    
        for sentence in data:
            sentence = tuple(sentence)
            for i in range(1, len(sentence)):
                nc = min(self.n-1 , i)
                context = sentence[i-nc: i]
                perp += -math.log(self.get_prob(context, sentence[i]))
                T += 1
        perp = math.exp(perp/T)
        return perp
