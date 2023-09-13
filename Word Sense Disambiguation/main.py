# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":

    #collecting data (also data with collocational features)
    data, data_collocational, labels = load_data()

    #evaluate the model with BOW vector only
    model = WSD_Model(labels=labels, collocational_vector=False, concatenate=False)
    model.evaluate(data=data, data_col=data_collocational)

    #evaluate the model with Collocational vector only
    model = WSD_Model(labels=labels, collocational_vector=True, concatenate=False)
    model.evaluate(data=data, data_col=data_collocational)

    #evaluate the model with the concatenation (Bow + collocational) vector
    model = WSD_Model(labels=labels, collocational_vector=True, concatenate=True)
    model.evaluate(data=data, data_col=data_collocational)

    #Evaluate Lesk original and Graph-based (lesk similarity) metrics 
    print('\nEvaluate Lesk algorithm..')
    evaluate_wsd(metric='lesk_original')
    print('\nEvaluate Pedersen (Graph-based) algorithm..')
    evaluate_wsd(metric='lesk_similarity')

