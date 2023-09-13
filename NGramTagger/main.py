# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import nltk

if __name__ == "__main__":
    
    #nltk.download('treebank')
    #nltk.download('universal_tagset')

    #load data
    train_tagged, test_tagged, train, test = load_data()

    #Evaluate NLTK NgramTagger
    print('\nEvaluate NLTK NgramTagger...')
    nltk_accuracy = NgramTagger_NLTK(train_data_tagged=train_tagged, test_data_tagged=test_tagged, test_data=test)

    #Evaluate SpaCy POS-tags
    print('\nEvaluate SpaCy POS tagger...')
    spacy_accuracy = Spacy_POS_Tags(test_data_tagged=test_tagged, test_data=test)

    #Results
    print('\n#### Evaluation Results ####')
    print('NLTK accuracy: ', nltk_accuracy)
    print('SpaCy accuracy: ', spacy_accuracy)
    
