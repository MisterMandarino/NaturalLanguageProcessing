# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS
import numpy as np

def LinearSVM(data, transformation='CountVectorizer', cross_validation=False, min_cutoff=None, max_cutoff=None, stop_words=None, lowercase=None):
    
    #Initialize the vectorizer for feature extraction
    if transformation == 'CountVectorizer':
        vectorizer = CountVectorizer(binary=True)
    elif transformation == 'TFIDF':
        vectorizer = TfidfVectorizer()

    if cross_validation==False:
        #transform the data
        data_vector = vectorizer.fit_transform(data.data)

    #defining the number of splits
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)

    if transformation=='CountVectorizer':
        #initialize the Linear SVM Classifier (set the C hyperparameter low to help convergence)
        svc = LinearSVC(C=0.0008)
    elif transformation=='TFIDF':
        #initialize the Linear SVM Classifier (set the C hyperparameter High to help convergence)
        svc = LinearSVC(C=1)
    if cross_validation:

        #Testing various parameters for the TF-IDF Transformation using CrossValidation
        for max_df, min_df, word, lower in zip(max_cutoff, min_cutoff, stop_words, lowercase):

            if word:
                tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=list(SKLEARN_STOP_WORDS), lowercase=lower)
            else:
                tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=None, lowercase=lower)

            #transform the data
            data_tfidf = tfidf_vectorizer.fit_transform(data.data)

            #validate the classifier with CrossValidation
            scores = cross_validate(svc, data_tfidf, data.target, cv=stratified_split, scoring=['accuracy'])

            #print the scores
            print(f'parameters:\t min cutoff [{min_df}]\t max cutoff [{max_df}]\t use stop words [{word}]\t use lowercase [{lower}]')
            print('Accuracy: {:.3}\n'.format(sum(scores['test_accuracy'])/len(scores['test_accuracy'])))
    else:

        mean_accuracy = []
        for i, (train_index, test_index) in enumerate(stratified_split.split(data_vector, data.target)):
            
            #Train the classifier for each stratified split
            svc.fit(data_vector[train_index], data.target[train_index])

            #Predict the class
            svc.predict(data_vector[test_index])

            #validate the classifier at each fold using accuracy metric
            accuracy = svc.score(data_vector[test_index], data.target[test_index])
            print("Fold {} Accuracy: {:.3}".format(i+1,accuracy))
            mean_accuracy.append(accuracy)

        #printing the mean accuracy
        print('mean accuracy: {:.3}'.format(np.mean(mean_accuracy)))
