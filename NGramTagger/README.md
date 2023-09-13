**This file is not mandatory**
But if you want, here your can add your comments or anything that you want to share with us
regarding the exercise.

Lab Exercise: Comparative Evaluation of NLTK Tagger and Spacy Tagger

Train and evaluate NgramTagger
    -experiment with different tagger parameters
    -some of them have cut-off

Evaluate spacy POS-tags on the same test set
    -create mapping from spacy to NLTK POS-tags
        -SPACY list https://universaldependencies.org/u/pos/index.html
        -NLTK list https://github.com/slavpetrov/universal-pos-tags
    -convert output to the required format (see format above)
        -flatten into a list
    -evaluate using accuracy from nltk.metrics
    
Dataset: treebank
Expected output: NLTK: Accuracy SPACY: Accuracy