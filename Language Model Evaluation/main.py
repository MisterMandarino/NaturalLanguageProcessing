# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    
    #load data
    data, vocab, train_sents, test_sents = load_data()

    # NLTK StupidBackoff
    print('\n#### StupidBackoff NLTK ####')
    StupidBackoff_nltk(data, vocab, train_sents, test_sents)

    # StupidBackoff Implementation
    print('\n#### StupidBackoff Implementation ####')
    mysb = MyStupidBackoff(n=2, alpha=0.4) #building the model
    probabilities = mysb.compute_probabilities(data) #compute the probabily table from the train set

    train_oov = [list(vocab.lookup(sent)) for sent in train_sents] #building the list of sentences with oov words to train the model 
    test_oov = [list(vocab.lookup(sent)) for sent in test_sents] #building the list of sentences with oov words to test the model 

    #compute the perplexity of the model
    print('perplexity (alpha: 0.4): ', mysb.perplexity(test_oov))

