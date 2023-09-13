# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    
    #load the data
    data = fetch_20newsgroups(subset='all')

    #print number of samples and classes
    print("Classes: {}".format(len(list(data.target_names))))
    print("Samples: {}".format(len(data.data)))

    #Linear SVM with CountVectorization
    print('\nLinear SVM with CountVectorization')
    LinearSVM(data, transformation='CountVectorizer', cross_validation=False)

    #Linear SVM with TF-IDF Transformation
    print('\nLinear SVM with TF-IDF')
    LinearSVM(data, transformation='TFIDF', cross_validation=False)

    #Testing different parameters with Linear SVM (TF-IDF)
    print('\nTesting parameters with TF-IDF...')
    min_cut_off = [1, 3, 5]
    max_cut_off = [1000, 200, 50]
    stop_words = [False, True, True]
    lowercase = [False, True, True]
    LinearSVM(data, transformation='TFIDF', cross_validation=True,
               min_cutoff=min_cut_off, max_cutoff=max_cut_off, 
               stop_words=stop_words, lowercase=lowercase)

